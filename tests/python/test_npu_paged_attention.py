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

"""Tests for the NPU paged-attention backend."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torch_npu", reason="NPU paged-attention tests require torch_npu")

from xllm.python.attention.backend import LayerCache  # noqa: E402
from xllm.python.attention.npu_paged_attention import (  # noqa: E402
    NpuPagedAttentionBackend,
)
from xllm.python.model_executor.forward_context import (  # noqa: E402
    AclGraphCaptureContext,
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)


def test_uses_first_nonempty_key_cache() -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    linear_cache = LayerCache(
        key=None,
        value=None,
        conv=torch.empty(8, 3, 64),
        ssm=torch.empty(8, 2, 4, 4),
    )
    key_cache = torch.empty(17, 128, 2, 64)
    value_cache = torch.empty_like(key_cache)

    backend.bind_kv_caches(
        [
            linear_cache,
            LayerCache(key=key_cache, value=value_cache),
        ]
    )

    assert backend.num_kv_blocks == 17
    assert backend.page_size == 128


def test_quant_indexer_capture_does_not_reuse_warmup_metadata() -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=True,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    backend._mla_actual_seq_q = torch.tensor([1, 2], dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.tensor([7, 31], dtype=torch.int32)
    backend._mla_max_seqlen_q = 1
    backend._mla_max_seqlen_k = 128
    warmup_metadata = torch.tensor([7, 31], dtype=torch.int32)
    captured_metadata = torch.tensor([8, 32], dtype=torch.int32)
    context_args = (backend, torch.device("cpu"), SimpleNamespace(), [])
    with patch(
        "xllm.python.attention.npu_paged_attention.kernels.quant_lightning_indexer_metadata",
        side_effect=[warmup_metadata, captured_metadata],
        create=True,
    ) as generate:
        with forward_context(ForwardContext(*context_args)):
            for _ in range(2):
                assert backend._get_quant_indexer_metadata(64, 1, 128, 2048, 1) is warmup_metadata
        assert generate.call_count == 1

        capture = AclGraphCaptureContext(stream=None, tasks=[])
        with forward_context(ForwardContext(*context_args, acl_graph=capture)):
            for _ in range(2):
                assert backend._get_quant_indexer_metadata(64, 1, 128, 2048, 1) is captured_metadata
        # All layers share one metadata producer inside capture, independently
        # of the eager warmup cache that prepare() will subsequently clear.
        assert generate.call_count == 2


def test_quant_indexer_graph_entries_use_distinct_metadata_buffers() -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=True,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    backend._mla_max_seqlen_q = 1
    backend._mla_max_seqlen_k = 128
    context_args = (backend, torch.device("cpu"), SimpleNamespace(), [])
    state = AclGraphExecutionState({})
    first_metadata = torch.tensor([7, 31], dtype=torch.int32)
    second_metadata = torch.tensor([11, 47], dtype=torch.int32)
    with patch(
        "xllm.python.attention.npu_paged_attention.kernels.quant_lightning_indexer_metadata",
        side_effect=[first_metadata, second_metadata],
        create=True,
    ) as generate:
        backend._mla_actual_seq_q = torch.tensor([1, 2], dtype=torch.int32)
        backend._mla_actual_seq_kv = torch.tensor([7, 31], dtype=torch.int32)
        with forward_context(ForwardContext(*context_args, execution_state=state)):
            assert backend._get_quant_indexer_metadata(64, 1, 128, 2048, 1) is first_metadata

        backend._mla_actual_seq_q = torch.tensor([1, 4], dtype=torch.int32)
        backend._mla_actual_seq_kv = torch.tensor([11, 47], dtype=torch.int32)
        with forward_context(ForwardContext(*context_args, execution_state=state)):
            assert backend._get_quant_indexer_metadata(64, 1, 128, 2048, 1) is second_metadata

    assert generate.call_count == 2
