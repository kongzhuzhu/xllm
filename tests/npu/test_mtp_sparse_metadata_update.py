# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real NPU coverage of direct writes to fixed Unified metadata arenas."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from xllm.python.model_executor.runners.mtp_sparse_metadata import (
    MtpSparseMetadataBinding,
    MtpSparseMetadataStorage,
    MtpSparsePositionStorage,
)


@pytest.fixture(scope="module")
def npu_device() -> torch.device:
    index = os.environ.get("XLLM_TEST_NPU_DEVICE")
    library = os.environ.get("XLLM_TEST_NATIVE_LIBRARY")
    if index is None or library is None:
        pytest.skip("set XLLM_TEST_NPU_DEVICE and XLLM_TEST_NATIVE_LIBRARY for the real fused updater")
    pytest.importorskip("torch_npu")
    torch.npu.set_device(int(index))
    torch.ops.load_library(library)
    return torch.device(f"npu:{index}")


@pytest.mark.parametrize(
    "steps,batch,position_dtype,step_major",
    [
        (1, 1, torch.int32, False),
        (3, 3, torch.int64, False),
        (3, 3, torch.int32, True),
        (5, 8, torch.int64, True),
    ],
)
def test_fused_metadata_replay_and_shrinking_tail(
    npu_device: torch.device, steps: int, batch: int, position_dtype: torch.dtype, step_major: bool
) -> None:
    capacity = 64
    block_size = 256 if steps == 5 else 128

    def metadata(table: torch.Tensor, lengths: torch.Tensor, slots: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(slot_mapping=slots, kv_seq_lens=lengths, block_table=table)

    draft_storage = MtpSparseMetadataStorage(
        metadata,
        [batch * (2 if step == 0 else 1) for step in range(steps)],
        capacity,
        npu_device,
        share_block_tables=True,
    )
    target_storage = MtpSparseMetadataStorage(metadata, [batch * (steps + 1)], capacity, npu_device)
    draft_storage.arena.fill_(-777)
    target_storage.arena.fill_(-777)
    draft, target = draft_storage.metadata, target_storage.metadata[0]
    position_storage = MtpSparsePositionStorage(batch, steps, npu_device)
    position_storage.arena.fill_(-777)
    binding = MtpSparseMetadataBinding(
        draft,
        (draft_storage.arena,),
        (target,),
        (target_storage.arena,),
        position_storage=position_storage,
        batch_size=batch,
        speculative_tokens=steps,
        table_capacity=capacity,
        block_size=block_size,
        target_step_major=step_major,
    )
    tensors = [
        getattr(item, field) for item in (*draft, target) for field in ("slot_mapping", "kv_seq_lens", "block_table")
    ]
    tensors.append(position_storage.arena)
    addresses = [value.data_ptr() for value in tensors]
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream(device=npu_device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(graph, stream=stream):
        output = torch.cat([value.flatten() for value in tensors])
    torch.npu.current_stream().wait_stream(stream)
    retained = []
    for generation, columns in enumerate((33, 41, 64, 35, 33)):
        # A strided source and reordered physical pages catch row/layout errors.
        table_host = (
            torch.arange(batch * (columns + 7), dtype=torch.int32).reshape(batch, columns + 7) + 17 + generation
        ).flip(0)
        source = table_host.to(npu_device)[:, :columns]
        positions = [block_size - 2 + row % 3 + generation for row in range(batch)]
        lengths = [position + 1 for position in positions]
        first_lengths, first_slots = [], []
        for row, position in enumerate(positions):
            first_lengths.extend((max(1, lengths[row] - 1), lengths[row]))
            first_slots.extend(
                (
                    -1
                    if row % 2
                    else int(table_host[row, (position - 1) // block_size]) * block_size + (position - 1) % block_size,
                    int(table_host[row, position // block_size]) * block_size + position % block_size,
                )
            )
        binding.update(
            source,
            torch.tensor(positions, dtype=position_dtype, device=npu_device),
            torch.tensor(lengths, dtype=torch.int32, device=npu_device),
            torch.tensor(first_lengths, dtype=torch.int32, device=npu_device),
            torch.tensor(first_slots, dtype=torch.int32, device=npu_device),
        )
        expected = []
        for role in range(steps + 1):
            if role == 0:
                rows = [(row, repair) for row in range(batch) for repair in range(2)]
                slots, kv = first_slots, first_lengths
            else:
                if role < steps:
                    rows = [(row, role) for row in range(batch)]
                elif step_major:
                    rows = [(row, step) for step in range(steps + 1) for row in range(batch)]
                else:
                    rows = [(row, step) for row in range(batch) for step in range(steps + 1)]
                slots = [
                    int(table_host[row, (positions[row] + step) // block_size]) * block_size
                    + (positions[row] + step) % block_size
                    for row, step in rows
                ]
                kv = [lengths[row] + step for row, step in rows]
            tables = [
                int(table_host[row, column]) if column < columns else 0 for row, _ in rows for column in range(capacity)
            ]
            expected.extend(slots)
            expected.extend(kv)
            expected.extend(tables)
        expected.extend(position + step for step in range(steps) for position in positions)
        expected.extend(position + step for position in positions for step in range(steps + 1))
        expected.extend(position + offset for position in positions for offset in (-1, 0))
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            graph.replay()
        torch.npu.current_stream().wait_stream(stream)
        snapshot = output.clone()
        reference = torch.tensor(expected, dtype=torch.int64)
        torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)
        assert [value.data_ptr() for value in tensors] == addresses
        retained.append((snapshot, reference))
    for snapshot, reference in retained:
        torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)
    stream.synchronize()
    graph.reset()
