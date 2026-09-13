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

"""Opt-in native CP planner probe for a single-node prefill contract."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def _load_native_cp_library() -> None:
    library = os.environ.get("XLLM_TEST_NATIVE_LIBRARY")
    if not library:
        pytest.skip("set XLLM_TEST_NATIVE_LIBRARY for the native CP probe")
    assert Path(library).is_file(), f"native operator library does not exist: {library}"
    torch.ops.load_library(library)


def test_native_cp_plan_and_shard_contract() -> None:
    """Exercise the production C++ planner and Python shard path on CPU."""
    from xllm.python.model_executor.cp_utils import build_cp_context, cp_shard_rows

    source = torch.arange(11, dtype=torch.float32).view(-1, 1)
    source[2] = float("nan")
    contexts = [
        build_cp_context([5, 6], [7, 8], cp_size=2, cp_rank=rank, device=torch.device("cpu")) for rank in (0, 1)
    ]
    assert contexts[0].total_local == contexts[1].total_local

    for context in contexts:
        local = cp_shard_rows(source, context)
        valid = context.shard_valid_mask
        torch.testing.assert_close(local[~valid], torch.zeros_like(local[~valid]))
        expected = source.index_select(0, context.shard_gather_index)
        assert torch.equal(torch.isnan(local[valid]), torch.isnan(expected[valid]))
        assert context.q_cu_seqlens
        assert context.kv_cu_seqlens
        assert context.segment_kv_seq_lens


def _causal_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Dense right-aligned causal oracle, including the cached prefix."""
    prefix_rows = key.shape[0] - query.shape[0]
    query_positions = prefix_rows + torch.arange(query.shape[0])
    key_positions = torch.arange(key.shape[0])
    scores = query @ key.T / query.shape[-1] ** 0.5
    scores.masked_fill_(key_positions[None, :] > query_positions[:, None], -torch.inf)
    return scores.softmax(dim=-1) @ value


@pytest.mark.parametrize("cp_size", [2, 3, 4])
@pytest.mark.parametrize(
    ("query_lengths", "prefix_lengths"),
    [([1], [0]), ([0, 1, 5], [7, 3, 128]), ([5, 6], [0, 0]), ([3, 5], [128, 256]), ([305, 17], [500, 0])],
)
def test_native_cp_all_rank_restore_and_causal_prefix_oracle(
    cp_size: int, query_lengths: list[int], prefix_lengths: list[int]
) -> None:
    """Validate every rank's host plan without requiring a model or NPU.

    Concatenating rank outputs models the all-gather layout; this does not
    claim to test the transport or the production attention kernel.
    """
    from xllm.python.model_executor.cp_utils import build_cp_context, cp_shard_positions, cp_shard_rows

    generator = torch.Generator().manual_seed(1716)
    queries = [torch.randn(length, 4, dtype=torch.float64, generator=generator) for length in query_lengths]
    kv_lengths = [query + prefix for query, prefix in zip(query_lengths, prefix_lengths, strict=True)]
    keys = [torch.randn(length, 4, dtype=torch.float64, generator=generator) for length in kv_lengths]
    values = [torch.randn(length, 4, dtype=torch.float64, generator=generator) for length in kv_lengths]
    source = torch.cat(queries)
    positions = torch.cat(
        [torch.arange(prefix, prefix + query) for query, prefix in zip(query_lengths, prefix_lengths, strict=True)]
    )
    reference = torch.cat([_causal_attention(q, k, v) for q, k, v in zip(queries, keys, values, strict=True)])
    contexts = [
        build_cp_context(query_lengths, kv_lengths, cp_size, rank, torch.device("cpu")) for rank in range(cp_size)
    ]
    shards = [cp_shard_rows(source, context) for context in contexts]
    assert len({context.total_local for context in contexts}) == 1
    owned_rows = torch.cat([context.shard_index[context.shard_valid_mask] for context in contexts])
    torch.testing.assert_close(owned_rows.sort().values, torch.arange(source.shape[0]))

    local_outputs = []
    for context, shard in zip(contexts, shards, strict=True):
        local_positions = cp_shard_positions(positions, context)
        torch.testing.assert_close(
            local_positions[~context.shard_valid_mask], torch.zeros_like(local_positions[~context.shard_valid_mask])
        )
        output = torch.zeros_like(shard)
        query_begin = 0
        for sequence_id, query_end, kv_end in zip(
            context.segment_seq_indices.tolist(), context.q_cu_seqlens, context.segment_kv_seq_lens, strict=True
        ):
            rows = context.query_index[query_begin:query_end]
            # Position ids provide an independent check of the segment's
            # absolute prefix boundary, including zero-query requests.
            torch.testing.assert_close(local_positions[rows], torch.arange(kv_end - rows.numel(), kv_end))
            output[rows] = _causal_attention(shard[rows], keys[sequence_id][:kv_end], values[sequence_id][:kv_end])
            query_begin = query_end
        assert query_begin == int(context.shard_valid_mask.sum())
        local_outputs.append(output)

    rank_major_source = torch.cat(shards)
    rank_major_output = torch.cat(local_outputs)
    for context in contexts:
        torch.testing.assert_close(rank_major_source[context.restore_index], source)
        torch.testing.assert_close(rank_major_output[context.restore_index], reference, rtol=1e-12, atol=1e-12)
