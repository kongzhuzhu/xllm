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

"""Opt-in MTP runner capture/replay contract; not a full GLM model test."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class _Draft(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding.from_pretrained(torch.arange(128, dtype=torch.float32).reshape(32, 4))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        hidden = input_embedding + self.embed_tokens(input_ids) + positions[:, None]
        if mtp_topk_indices is None:
            topk = (input_ids + positions)[:, None, None].expand(-1, 1, 4).contiguous()
        else:
            # Deliberately alias the persistent input: runner output ownership
            # must protect the returned state from the next input-buffer fill.
            topk = mtp_topk_indices
            hidden = hidden + topk[:, 0, :]
        return hidden, None, topk


class _Backend:
    page_size = 4

    def __init__(self, is_mla: bool) -> None:
        self.is_mla = is_mla

    def prepare(self, metadata: SimpleNamespace, *, graph_mode: bool = False) -> None:
        assert graph_mode


def _metadata(rows: int, device: torch.device) -> SimpleNamespace:
    indices = torch.arange(rows, dtype=torch.int32, device=device)
    ones = torch.ones(rows, dtype=torch.int32, device=device)
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
        expanded_decode_metadata=None,
        q_cu_seq_lens=None,
        slot_mapping=indices * 4,
        block_table=indices[:, None],
        paged_kv_indptr=torch.arange(rows + 1, dtype=torch.int32, device=device),
        paged_kv_indices=indices,
        paged_kv_last_page_len=ones,
        kv_seq_lens=ones,
        kv_seq_lens_host_values=[1] * rows,
        kv_cu_seq_lens=torch.arange(rows + 1, dtype=torch.int32, device=device),
    )


@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("max_batch", [6, 8])
@torch.inference_mode()
def test_mtp_capture_replay_with_changing_topk_and_batch(is_mla: bool, max_batch: int) -> None:
    device_id = os.environ.get("XLLM_TEST_MTP_GRAPH_DEVICE")
    native_library = os.environ.get("XLLM_TEST_NATIVE_LIBRARY")
    if device_id is None or native_library is None:
        pytest.skip("set XLLM_TEST_MTP_GRAPH_DEVICE and XLLM_TEST_NATIVE_LIBRARY")
    assert device_id.isdecimal()
    assert Path(native_library).is_file()
    import torch_npu  # noqa: F401

    torch.ops.load_library(native_library)
    import xllm.python as runtime

    runtime.initialize_runtime()
    from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner

    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    model = _Draft().to(device)
    runner = DecodeAclGraphRunner(
        model,
        _Backend(is_mla),
        device,
        max_batch=max_batch,
        max_model_len=16,
        decode_batch_size_limit=3,
    )
    saved_topk = []
    # Revisit existing buckets and capture both first/reused-topk paths after
    # shrinking to one row, matching MTP graph warmup's 2-to-1 transition.
    for step, rows in enumerate((3, 3, 4, 5, 6, 2, 1, 1, 3)):
        ids = torch.arange(rows, dtype=torch.int32, device=device) + step
        positions = ids + 2
        embedding = torch.full((rows, 4), float(step), device=device)
        topk = None
        if step not in (0, 6):
            topk = (torch.arange(rows * 4, dtype=torch.int32, device=device) + 10 * step).reshape(rows, 1, 4)
            topk = topk.flip(0)
        metadata = _metadata(rows, device)
        # A request contributes one or two repair rows. The explicit request
        # limit must still admit three requests with five/six execution rows.
        metadata.dp_global_sequence_nums = ((rows + 1) // 2,)
        if is_mla and rows == 3:
            # Two requests can produce three draft rows after acceptance
            # repair. MLA does not consume their recurrent-state indices.
            metadata.linear_state_indices = torch.tensor([7, 9], dtype=torch.int32, device=device)
        expected = model(ids, positions, embedding, topk)
        assert runner.can_execute(ids, metadata, embedding, topk)
        actual = runner.execute(ids, positions, metadata, embedding, topk)
        torch.testing.assert_close(actual[0], expected[0])
        assert actual[1] is None
        torch.testing.assert_close(actual[2], expected[2])
        saved_topk.append((actual[2], expected[2].cpu()))
        for retained, reference in saved_topk:
            torch.testing.assert_close(retained.cpu(), reference)
    assert len(runner._graphs) == 6
    rejected_ids = torch.arange(4, dtype=torch.int32, device=device)
    rejected_metadata = _metadata(4, device)
    rejected_metadata.dp_global_sequence_nums = (4,)
    assert not runner.can_execute(rejected_ids, rejected_metadata)
