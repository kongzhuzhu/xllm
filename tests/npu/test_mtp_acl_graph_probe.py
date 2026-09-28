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

"""Real-NPU probes of the composite runner for several independent K variants.

Run separately from tests/python, whose conftest installs runtime stubs.
Deterministic bodies exercise graph mechanics and acceptance, not real GLM
attention.
"""

from __future__ import annotations

import os
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from tests.python.mtp_graph_test_utils import make_recipe, output_tensors, scalar_reference
from xllm.python.model_executor.runners.mtp_acl_graph import MtpAclGraphRunner, MtpGraphRecipe, MtpSamplingPlan


@pytest.mark.parametrize("steps", [3, 5])
def test_owned_sparse_metadata_updates_reach_captured_consumers(npu_device: torch.device, steps: int) -> None:
    library = os.environ.get("XLLM_TEST_NATIVE_LIBRARY")
    if not library:
        pytest.skip("set XLLM_TEST_NATIVE_LIBRARY for the real attention backend")
    import xllm.python as runtime

    torch.ops.load_library(library)
    runtime.initialize_runtime()
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.model_executor.runners.mtp_acl_graph import _pack_metadata_tensors

    variants = []
    cache = torch.empty(4, 128, 1, 16, device=npu_device)
    for batch in (1, 3):
        metadata, backends = [], []
        for rows in (batch * 2, *([batch] * (steps - 1)), batch * (steps + 1)):
            item = SimpleNamespace(
                slot_mapping=torch.full((rows,), 127, dtype=torch.int32, device=npu_device),
                block_table=torch.zeros(rows, 32, dtype=torch.int32, device=npu_device),
                kv_seq_lens=torch.full((rows,), 128, dtype=torch.int32, device=npu_device),
                q_cu_seq_lens=None,
                q_seq_lens=None,
                expanded_decode_metadata=None,
                is_prefill=False,
                is_chunked_prefill=False,
            )
            metadata.append(item)
        arenas = _pack_metadata_tensors(metadata)
        for item in metadata:
            backend = NpuPagedAttentionBackend(
                num_heads=1,
                num_kv_heads=1,
                head_dim=16,
                scale=0.25,
                sliding_window=0,
                is_mla=True,
                device=npu_device,
                dtype=torch.bfloat16,
            )
            backend.bind_kv_caches([LayerCache(key=cache, value=cache, index=cache)])
            backend.prepare_owned_graph_metadata(item)
            backends.append(backend)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = torch.cat(
                [
                    backend._mla_actual_seq_kv + backend._block_table_i32[:, 0] + item.slot_mapping
                    for backend, item in zip(backends, metadata)
                ]
            )
        torch.npu.current_stream().wait_stream(stream)
        variants.append((metadata, backends, arenas, stream, graph, output))

    retained = []
    for generation in range(4):
        for metadata, backends, arenas, stream, graph, output in variants:
            expected = []
            for step, (item, backend) in enumerate(zip(metadata, backends)):
                rows = item.kv_seq_lens.numel()
                # Cross a page boundary, reorder rows and alternate graph
                # variants without changing their independently owned storage.
                lengths = torch.arange(rows, dtype=torch.int32).flip(0) + 129 + generation + step
                table = torch.full((rows, 32), generation + step, dtype=torch.int32)
                slots = lengths + 127
                item.kv_seq_lens.copy_(lengths.to(npu_device))
                item.block_table.copy_(table.to(npu_device))
                item.slot_mapping.copy_(slots.to(npu_device))
                backend.update_graph_metadata(item)
                expected.append(lengths + table[:, 0] + slots)
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                graph.replay()
            torch.npu.current_stream().wait_stream(stream)
            snapshot = output.clone()
            reference = torch.cat(expected)
            torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)
            retained.append((snapshot, reference))
    for snapshot, reference in retained:
        torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)


@pytest.fixture(scope="module")
def npu_device() -> torch.device:
    device_index = os.environ.get("XLLM_TEST_NPU_DEVICE")
    if device_index is None:
        pytest.skip("set XLLM_TEST_NPU_DEVICE for the real MTP ACL graph probe")
    pytest.importorskip("torch_npu")
    torch.npu.set_device(int(device_index))
    library = os.environ.get("XLLM_TEST_NATIVE_LIBRARY")
    if library:
        torch.ops.load_library(library)
    return torch.device(f"npu:{device_index}")


@pytest.mark.parametrize("steps", [1, 3, 5])
def test_mtp_recipe_replay_every_rejection_position(npu_device: torch.device, steps: int) -> None:
    rejects_device = torch.arange(steps + 1, device=npu_device)
    runner = MtpAclGraphRunner(make_recipe(rejects_device, steps))
    eager = MtpAclGraphRunner(make_recipe(rejects_device, steps), backend="eager")
    seeds = [2 + row for row in range(steps + 1)]
    positions = [10 + row * 3 for row in range(steps + 1)]
    kv_lengths = [position + 1 for position in positions]
    runner.capture(
        torch.tensor(seeds, device=npu_device),
        torch.tensor(positions, device=npu_device),
        torch.tensor(kv_lengths, dtype=torch.int32, device=npu_device),
    )
    rejects = list(range(steps + 1))
    expected = scalar_reference(seeds, positions, kv_lengths, rejects, steps)
    inputs = (
        torch.tensor(seeds, device=npu_device),
        torch.tensor(positions, device=npu_device),
        torch.tensor(kv_lengths, dtype=torch.int32, device=npu_device),
    )
    eager_output = output_tensors(eager.execute(*inputs))
    actual = runner.execute(*inputs)
    for name, tensor in output_tensors(actual).items():
        torch.testing.assert_close(tensor.cpu(), expected[name], rtol=0, atol=0)
        torch.testing.assert_close(tensor, eager_output[name], rtol=0, atol=0)
    # C++ Unified now passes the current column of its fused int32 [B,2]
    # output directly; the graph-owned destination performs the only cast.
    token_rows = torch.tensor([[seed - 1, seed] for seed in seeds], dtype=torch.int32, device=npu_device)
    replayed = runner.execute(token_rows[:, 1], inputs[1], inputs[2])
    for name, tensor in output_tensors(replayed).items():
        torch.testing.assert_close(tensor.cpu(), expected[name], rtol=0, atol=0)
    runner.close()


@pytest.mark.parametrize("steps", [3, 5])
@pytest.mark.parametrize("logprobs", [False, True])
def test_mtp_greedy_bfloat16_scores_match_probability_reference(
    npu_device: torch.device, steps: int, logprobs: bool
) -> None:
    rejects = torch.arange(steps + 1, device=npu_device)
    fast = make_recipe(rejects, steps, logits_dtype=torch.bfloat16)
    fast.runtime_outputs_only = True
    reference = make_recipe(rejects, steps, logits_dtype=torch.bfloat16)
    for recipe, probabilities in ((fast, False), (reference, True)):
        recipe.target_sampling = MtpSamplingPlan(
            batch_size=steps + 1,
            return_probs=probabilities,
            logprobs=logprobs,
            max_top_logprobs=3 if logprobs else 0,
        )
    inputs = (
        torch.arange(steps + 1, device=npu_device),
        torch.arange(steps + 1, device=npu_device) + 127,
        torch.arange(steps + 1, dtype=torch.int32, device=npu_device) + 128,
    )
    runner = MtpAclGraphRunner(fast)
    eager = MtpAclGraphRunner(reference, backend="eager")
    runner.capture(*inputs)
    for offset in (0, 7):
        updated_inputs = (inputs[0] + offset, inputs[1] + offset, inputs[2] + offset)
        expected = eager.execute(*updated_inputs)
        actual = runner.execute(*updated_inputs)
        expected_compact = torch.stack(
            [
                expected.target_embeddings[row, index]
                for row, count in enumerate(expected.accepted_count.cpu().tolist())
                for index in (max(count - 1, 0), count)
            ]
        )
        torch.testing.assert_close(actual.target_embeddings, expected_compact, rtol=0, atol=0)
        for name in ("committed_tokens", "accepted_count"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
        assert actual.token_state is not None
        assert actual.token_state.dtype == torch.uint8
        assert actual.committed_tokens.untyped_storage().data_ptr() == actual.token_state.data_ptr()
        assert actual.accepted_count.untyped_storage().data_ptr() == actual.token_state.data_ptr()
        for name in ("logprobs", "top_logprobs", "top_tokens"):
            value, expected_value = getattr(actual, name), getattr(expected, name)
            if value is None or expected_value is None:
                assert value is expected_value
            else:
                torch.testing.assert_close(value.cpu(), expected_value.cpu(), rtol=1e-5, atol=1e-5)
        assert actual.target_probs is None
    runner.close()


@pytest.mark.parametrize("steps", [3, 5])
@pytest.mark.parametrize("runtime_outputs", [False, True])
def test_mtp_replay_output_survives_the_next_replay(
    npu_device: torch.device, steps: int, runtime_outputs: bool
) -> None:
    """A returned graph result must not alias the next replay's buffers."""
    recipe = make_recipe(torch.tensor([1], device=npu_device), steps)
    recipe.runtime_outputs_only = runtime_outputs
    runner = MtpAclGraphRunner(recipe)

    def tensors(output: object) -> dict[str, torch.Tensor]:
        if runtime_outputs:
            return {
                field.name: getattr(output, field.name)
                for field in fields(output)
                if isinstance(getattr(output, field.name), torch.Tensor)
            }
        return output_tensors(output)

    first_inputs = (
        torch.tensor([2], device=npu_device),
        torch.tensor([10], dtype=torch.int32, device=npu_device),
        torch.tensor([11], dtype=torch.int32, device=npu_device),
    )
    second_inputs = (
        torch.tensor([9], device=npu_device),
        torch.tensor([20], dtype=torch.int32, device=npu_device),
        torch.tensor([21], dtype=torch.int32, device=npu_device),
    )
    runner.capture(*first_inputs)
    first = runner.execute(*first_inputs)
    first_snapshot = {name: tensor.clone() for name, tensor in tensors(first).items()}
    second = runner.execute(*second_inputs)
    assert not torch.equal(first_snapshot["committed_tokens"], tensors(second)["committed_tokens"])

    runner.close()
    for name, expected in first_snapshot.items():
        torch.testing.assert_close(tensors(first)[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden_size", [16, 8193])
def test_compact_commit_hidden_tiles_and_old_output_lifetime(
    npu_device: torch.device, dtype: torch.dtype, hidden_size: int
) -> None:
    # More than 32 requests exercises the count tile boundary; H=8193
    # exercises the hidden tile and unaligned tail. Every rejection position
    # and all-accepted are present, with independently recognizable rows.
    batch, steps = 33, 5
    target_cpu = torch.arange(batch * (steps + 1)).view(batch, steps + 1)
    draft_cpu = target_cpu[:, :steps].clone()
    counts = [row % (steps + 1) for row in range(batch)]
    for row, count in enumerate(counts):
        if count < steps:
            draft_cpu[row, count] = -7
    hidden_cpu = (torch.arange(batch * (steps + 1) * hidden_size) % 997).view(-1, hidden_size).to(dtype)
    draft, target, hidden = [value.to(npu_device) for value in (draft_cpu, target_cpu, hidden_cpu)]
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(graph, stream=stream):
        state, compact = torch.ops.xllm_ops.mtp_greedy_commit(draft, target, hidden)
    torch.npu.current_stream().wait_stream(stream)
    expected = torch.stack(
        [
            hidden_cpu[row * (steps + 1) + index]
            for row, count in enumerate(counts)
            for index in (max(count - 1, 0), count)
        ]
    )
    graph.replay()
    retained = compact.clone()
    torch.testing.assert_close(retained.cpu(), expected, rtol=0, atol=0)
    hidden.add_(1)
    graph.replay()
    torch.testing.assert_close(compact.cpu(), expected + 1, rtol=0, atol=0)
    torch.testing.assert_close(retained.cpu(), expected, rtol=0, atol=0)
    torch.npu.synchronize()
    graph.reset()


def test_next_prepare_writes_graph_embedding_destination(npu_device: torch.device) -> None:
    def body(
        ids: torch.Tensor, positions: torch.Tensor, step: int, embedding: torch.Tensor | None, topk: torch.Tensor | None
    ) -> torch.Tensor:
        del positions, step, topk
        hidden = ids.float().unsqueeze(1)
        return hidden if embedding is None else hidden + embedding

    def head(hidden: torch.Tensor) -> torch.Tensor:
        words = torch.arange(32, device=npu_device).float()
        return -(words.unsqueeze(0) - hidden.remainder(32)).abs()

    def recipe() -> MtpGraphRecipe:
        return MtpGraphRecipe(
            body,
            head,
            body,
            head,
            batch_size=1,
            speculative_tokens=3,
            vocab_size=32,
            device=npu_device,
            kv_seq_lens=torch.ones(1, dtype=torch.int32, device=npu_device),
        )

    runner, reference = MtpAclGraphRunner(recipe()), MtpAclGraphRunner(recipe(), backend="eager")
    inputs = (
        torch.tensor([2], device=npu_device),
        torch.tensor([7], device=npu_device),
        torch.tensor([8], dtype=torch.int32, device=npu_device),
    )
    embedding = torch.ones(1, 1, device=npu_device)
    runner.capture(*inputs, embedding)
    destination = runner.draft_embedding_destination
    assert destination is not None and destination.data_ptr() != embedding.data_ptr()
    retained = runner.execute(*inputs, embedding)
    retained_tokens = retained.committed_tokens.cpu().clone()
    for value in (3, 5, 9):
        # Models C++ MtpPrepareNextDraft writing the final destination on the
        # producer stream after the previous replay's reads have completed.
        destination.fill_(value)
        actual = runner.execute(*inputs, destination)
        expected = reference.execute(*inputs, torch.full_like(embedding, value))
        for name, tensor in output_tensors(actual).items():
            torch.testing.assert_close(tensor, output_tensors(expected)[name], rtol=0, atol=0)
    torch.testing.assert_close(retained.committed_tokens.cpu(), retained_tokens, rtol=0, atol=0)
    runner.close()
