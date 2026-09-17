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

"""Unit tests for xllm.python.model_executor.executor.

Tests the device-conditional backend dispatch, ModelExecutor construction
validation, and execution routing — using CPU mocks so no GPU/NPU required.
"""

from __future__ import annotations

import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import pytest
import torch
import torch.nn as nn

# conftest.py stands in for xllm.python, whose import would bind the active
# platform's kernel package and reach for operators from the C++ binary.
from xllm.python.attention.backend import (  # noqa: E402
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    normalize_layer_caches,
)
from xllm.python.layers.attention import Attention  # noqa: E402
from xllm.python.model_executor.executor import (  # noqa: E402
    ModelExecutor,
    _create_attention_backend,
    _resolve_graph_backend,
)
from xllm.python.model_executor.forward_context import (  # noqa: E402
    ForwardContext,
    forward_context,
    get_forward_context,
    record_layer_event,
)
from xllm.python.model_executor.runners.decode_acl_graph import (  # noqa: E402
    DecodeAclGraphRunner,
)
from xllm.python.model_executor.runners.decode_cuda_graph import (  # noqa: E402
    DecodeCudaGraphRunner,
    _decode_graph_buckets,
)
from xllm.python.model_executor.runners.eager import EagerRunner  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubAttentionBackend(AttentionBackend):
    """Minimal backend that records calls for assertion."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self._kv_caches: list[LayerCache] = []
        self._prepared = False

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        self._kv_caches = kv_caches

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        self._prepared = True

    def execute(self, q, k, v, layer) -> torch.Tensor:
        return q

    @property
    def num_kv_blocks(self) -> int:
        return 0

    @property
    def page_size(self) -> int:
        return 1


class _PagedStubAttentionBackend(StubAttentionBackend):
    @property
    def page_size(self) -> int:
        return 4


class _MlaStubAttentionBackend(StubAttentionBackend):
    @property
    def is_mla(self) -> bool:
        return True


def _make_attention_layer(
    num_heads=8,
    num_kv_heads=2,
    head_dim=64,
    scale=0.125,
    sliding_window=0,
    layer_id=0,
) -> Attention:
    return Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        sliding_window=sliding_window,
        layer_id=layer_id,
    )


class _FakeModel(nn.Module):
    """Model with configurable number of uniform Attention layers."""

    def __init__(self, num_layers: int = 2, device: str = "cpu", **attn_kwargs):
        super().__init__()
        self.model = nn.Linear(1, 1)  # execution_model placeholder
        self.layers = nn.ModuleList([_make_attention_layer(layer_id=i, **attn_kwargs) for i in range(num_layers)])
        self._param = nn.Parameter(torch.zeros(1, device=device))

    def forward(self, input_ids, positions):
        return input_ids


class _FakeModelHeterogeneous(nn.Module):
    """Model with non-uniform Attention layers (should fail validation)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self.attn1 = _make_attention_layer(num_heads=8, layer_id=0)
        self.attn2 = _make_attention_layer(num_heads=4, layer_id=1)
        self._param = nn.Parameter(torch.zeros(1))


class _FakeModelNoAttention(nn.Module):
    """Model without any Attention layers."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self._param = nn.Parameter(torch.zeros(1))


class _FailingLayerSynchronizer:
    def record_event(self, layer_id: int) -> bool:
        return False


def test_attention_backend_defaults_to_non_mla() -> None:
    assert StubAttentionBackend().is_mla is False


def test_record_layer_event_propagates_record_failure() -> None:
    context = ForwardContext(
        attention_backend=StubAttentionBackend(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        layer_synchronizer=_FailingLayerSynchronizer(),
    )

    with (
        forward_context(context),
        pytest.raises(RuntimeError, match="failed to record layer completion event for layer 3"),
    ):
        record_layer_event(3)


# ---------------------------------------------------------------------------
# Tests: graph backend resolution
# ---------------------------------------------------------------------------


class TestNpuGraphBackendResolution:
    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_enable_graph_selects_aclgraph_on_npu(self, _mock_is_npu):
        config = {"enable_graph": True, "python_graph_backend": "off"}
        assert _resolve_graph_backend(config) == "aclgraph"

    def test_glm_mtp_allows_aclgraph_for_cross_draft_topk_state(self) -> None:
        config = {
            "model_type": "glm_moe_dsa_mtp",
            "enable_graph": True,
            "python_graph_backend": "aclgraph",
        }

        assert _resolve_graph_backend(config) == "aclgraph"


# ---------------------------------------------------------------------------
# Tests: _create_attention_backend dispatch
# ---------------------------------------------------------------------------


class TestCreateAttentionBackend:
    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_deepseek_v4_creates_dsa_backend(self, _mock_is_npu):
        attn = _make_attention_layer(head_dim=512)
        module = types.ModuleType("xllm.python.attention.dsa_attention")
        module.DsaAttentionBackend = StubAttentionBackend
        config = {
            "model_type": "deepseek_v4",
            "compress_ratios": [1, 4, 128],
            "num_hidden_layers": 3,
            "window_size": 128,
            "index_topk": 512,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "qk_rope_head_dim": 64,
        }
        with patch.dict(sys.modules, {module.__name__: module}):
            backend = _create_attention_backend(attn, torch.device("npu"), torch.bfloat16, config)

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["attn_head_dim"] == 512
        assert backend.init_kwargs["n_layers"] == 3
        assert backend.init_kwargs["compress_ratios"] == [1, 4, 128]

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    @patch(
        "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
        StubAttentionBackend,
    )
    def test_npu_device_creates_npu_backend(self, _mock_is_npu):
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        backend = _create_attention_backend(
            attn,
            torch.device("npu"),
            torch.float16,
            {"enable_mla": False},
        )
        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["num_heads"] == 8
        assert backend.init_kwargs["num_kv_heads"] == 1
        assert backend.init_kwargs["head_dim"] == 256
        assert backend.init_kwargs["is_mla"] is False

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    @patch(
        "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
        StubAttentionBackend,
    )
    def test_prefill_cp_uses_npu_backend_with_dcp_group(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = MagicMock()
        sfa_module.dcp_layer_options = MagicMock(return_value=512)

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
        ):
            backend = _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.float16,
                {"cp_size": 4, "enable_mla": True},
            )

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["is_mla"] is True
        sfa_module.SfaDcpAttentionBackend.assert_not_called()

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_decode_cp1_uses_sfa_dcp_backend(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = StubAttentionBackend
        sfa_module.dcp_layer_options = MagicMock(return_value=512)

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
        ):
            backend = _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.float16,
                {"cp_size": 1, "enable_mla": True},
                max_num_reqs=3,
            )

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["dcp_group"] is dcp_group
        assert backend.init_kwargs["index_topk"] == 512
        assert backend.init_kwargs["max_num_reqs"] == 3

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=False,
    )
    @patch(
        "xllm.python.model_executor.executor.current_platform.is_cuda",
        return_value=True,
    )
    def test_cuda_device_creates_flashinfer_backend(self, _mock_is_cuda, _mock_is_npu):
        attn = _make_attention_layer()
        module = types.ModuleType("xllm.python.attention.flashinfer")
        module.FlashInferBackend = StubAttentionBackend
        with patch.dict(sys.modules, {module.__name__: module}):
            backend = _create_attention_backend(attn, torch.device("cuda"), torch.float16)
        assert isinstance(backend, StubAttentionBackend)


# ---------------------------------------------------------------------------
# Tests: ModelExecutor construction
# ---------------------------------------------------------------------------


class TestModelExecutorConstruction:
    @pytest.mark.parametrize("graph_backend", ["cudagraphs", "inductor", "trace"])
    def test_npu_glm_mtp_rejects_non_acl_graph_before_backend_creation(self, graph_backend: str) -> None:
        config = {
            "model_type": "glm_moe_dsa_mtp",
            "python_graph_backend": graph_backend,
            "max_position_embeddings": 128,
        }
        with (
            patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
            patch("xllm.python.model_executor.executor._create_attention_backend") as create_backend,
            pytest.raises(NotImplementedError, match="GLM MTP.*aclgraph"),
        ):
            # No attention layers: admission must reject the backend before
            # model inspection or any graph/attention allocation.
            ModelExecutor(_FakeModelNoAttention(), config, max_seqs_per_batch=3)
        create_backend.assert_not_called()

    @pytest.mark.parametrize("graph_backend,enable_graph", [("off", True), ("ACLGRAPH", True), ("off", False)])
    def test_npu_glm_mtp_graph_selection(self, graph_backend: str, enable_graph: bool) -> None:
        config = {
            "model_type": "glm_moe_dsa_mtp",
            "python_graph_backend": graph_backend,
            "enable_graph": enable_graph,
            "max_position_embeddings": 128,
        }
        with (
            patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
            patch("xllm.python.model_executor.executor._create_attention_backend", return_value=StubAttentionBackend()),
        ):
            executor = ModelExecutor(_FakeModel(num_layers=1), config, max_seqs_per_batch=3)
        if enable_graph:
            assert isinstance(executor.decode_graph_runner, DecodeAclGraphRunner)
            assert executor.decode_graph_runner.max_batch == 6
        else:
            assert executor.decode_graph_runner is None
        assert executor.inductor_runner is None

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_valid_model_creates_executor(self, _mock_backend):
        model = _FakeModel(num_layers=3)
        config = {"python_graph_backend": "off"}
        executor = ModelExecutor(model, config, max_seqs_per_batch=4)

        assert executor._num_attention_layers == 3
        assert executor.decode_graph_runner is None
        assert executor.inductor_runner is None

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_no_attention_layers_raises(self, _mock_backend):
        model = _FakeModelNoAttention()
        with pytest.raises(ValueError, match="does not contain an Attention layer"):
            ModelExecutor(model, {}, max_seqs_per_batch=4)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_heterogeneous_attention_raises(self, _mock_backend):
        model = _FakeModelHeterogeneous()
        with pytest.raises(ValueError, match="identical attention configuration"):
            ModelExecutor(model, {}, max_seqs_per_batch=4)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_graph_backend_off_variants(self, _mock_backend):
        for off_value in ("off", "", "none", "0"):
            model = _FakeModel(num_layers=1)
            executor = ModelExecutor(model, {"python_graph_backend": off_value}, max_seqs_per_batch=4)
            assert executor.decode_graph_runner is None
            assert executor.inductor_runner is None

    @patch("xllm.python.model_executor.runners.decode_cuda_graph.DecodeCudaGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_data_parallel_cuda_graph_is_supported(self, mock_create, mock_graph_runner):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "dp_size": 2,
                "dp_rank": 1,
                "max_position_embeddings": 128,
                "python_graph_backend": "cudagraphs",
            },
            max_seqs_per_batch=4,
        )

        mock_graph_runner.assert_called_once_with(
            model.model,
            mock_create.return_value,
            torch.device("cpu"),
            4,
            128,
            2,
            1,
        )

    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_data_parallel_rejects_unsupported_graph_backend(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        with pytest.raises(NotImplementedError, match="supports cudagraphs and aclgraph only"):
            ModelExecutor(
                model,
                {
                    "dp_size": 2,
                    "max_position_embeddings": 128,
                    "python_graph_backend": "inductor",
                },
                max_seqs_per_batch=4,
            )

    @patch("xllm.python.model_executor.executor._create_attention_backend", return_value=StubAttentionBackend())
    def test_context_parallel_rejects_data_parallel_combination(self, _mock_create):
        with pytest.raises(NotImplementedError, match="Python CP requires dp_size == 1"):
            ModelExecutor(
                _FakeModel(num_layers=1),
                {"cp_size": 2, "dp_size": 2, "python_graph_backend": "off"},
                max_seqs_per_batch=4,
            )

    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_glm_mtp_capacity_includes_repair_rows(
        self,
        mock_create: MagicMock,
        mock_graph_runner: MagicMock,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        ModelExecutor(
            model,
            {
                "model_type": "glm_moe_dsa_mtp",
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
            },
            max_seqs_per_batch=2,
            num_decoding_tokens=1,
        )

        # Two requests can carry three or four rows when one or both need the
        # previous token's KV repaired after accepting every draft token.
        assert mock_create.call_args.args[-1] == 4
        assert mock_graph_runner.call_args.args[3] == 4
        # Later draft steps still decode one token per request. Capacity must
        # not change the interpretation of the scheduler's token counts.
        assert mock_graph_runner.call_args.args[-1] == 1

    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_capacity_respects_decode_batch_limit(
        self,
        mock_create,
        mock_graph_runner,
    ):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
            },
            max_seqs_per_batch=256,
            num_decoding_tokens=4,
            acl_graph_decode_batch_size_limit=16,
        )

        # The SFA-DCP metadata builder is token-row based. MTP expands each
        # sequence into num_decoding_tokens rows, so its capacity must cover
        # the expanded graph input as well.
        assert mock_create.call_args.args[-1] == 256 * 4
        mock_graph_runner.assert_called_once_with(
            model.model,
            mock_create.return_value,
            torch.device("cpu"),
            64,
            128,
            1,
            0,
            16,
            4,
        )


@pytest.mark.parametrize("dp_rank", [0, 1])
@pytest.mark.parametrize(
    "max_sequences,limit,model_type,decode_tokens,logical_counts,execution_counts,capacity",
    [
        (256, 16, "glm_moe_dsa", 4, (16, 16), (64, 64), 64),
        (256, 3, "glm_moe_dsa_mtp", 1, (3, 2), (6, 3), 6),
        (5, None, "glm_moe_dsa", 4, (3, 2), (12, 8), 12),
        (5, 3, "glm_moe_dsa_mtp", 1, (3, 2), (6, 3), 6),
        (256, 3, "glm_moe_dsa_mtp", 1, (0, 3), (1, 5), 6),
    ],
)
def test_acl_executor_preserves_per_rank_sequence_capacity(
    dp_rank: int,
    max_sequences: int,
    limit: int | None,
    model_type: str,
    decode_tokens: int,
    logical_counts: tuple[int, ...],
    execution_counts: tuple[int, ...],
    capacity: int,
) -> None:
    # Construct the real runner through the executor: overriding max_batch on
    # a runner would hide double DP division and token-row rounding bugs.
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=_PagedStubAttentionBackend(),
    ):
        executor = ModelExecutor(
            _FakeModel(num_layers=1),
            {
                "model_type": model_type,
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
                "dp_size": 2,
                "dp_rank": dp_rank,
            },
            max_seqs_per_batch=max_sequences,
            num_decoding_tokens=decode_tokens,
            acl_graph_decode_batch_size_limit=limit,
        )
    runner = executor.decode_graph_runner
    assert isinstance(runner, DecodeAclGraphRunner)
    rows = execution_counts[dp_rank]
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        expanded_decode_metadata=None,
        dp_global_sequence_nums=logical_counts,
        dp_execution_token_counts=execution_counts,
        dp_is_decode=(1, 1),
        slot_mapping=torch.arange(rows, dtype=torch.int32),
        block_table=torch.zeros((rows, 1), dtype=torch.int32),
        kv_seq_lens=torch.ones(rows, dtype=torch.int32),
        kv_seq_lens_host_values=[1] * rows,
        paged_kv_indptr=torch.arange(rows + 1, dtype=torch.int32),
        paged_kv_indices=torch.zeros(rows, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(rows, dtype=torch.int32),
        kv_cu_seq_lens=None,
        q_cu_seq_lens=None,
    )
    assert runner.max_batch == capacity
    assert runner.can_execute(torch.zeros(rows, dtype=torch.int32), metadata)
    assert runner._padded_batch_size(rows, metadata) == capacity
    if limit is not None:
        metadata.dp_global_sequence_nums = (limit + 1, 0)
        assert not runner.can_execute(torch.zeros(rows, dtype=torch.int32), metadata)


class TestDecodeCudaGraphDataParallelKeys:
    @staticmethod
    def _runner(dp_rank: int = 0) -> DecodeCudaGraphRunner:
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.max_batch = 16
        runner.dp_size = 2
        runner.dp_rank = dp_rank
        runner._graphs = {}
        return runner

    @staticmethod
    def _metadata(
        token_counts: list[int] | tuple[int, ...] = (),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=False,
            dp_execution_token_counts=tuple(1 if count == 0 else count for count in token_counts),
        )

    def test_graph_key_uses_global_max_data_parallel_bucket(self):
        runner = self._runner()
        input_ids = torch.zeros(3, dtype=torch.int32)

        first = runner._graph_key(input_ids, self._metadata([3, 1]))
        second = runner._graph_key(input_ids, self._metadata([3, 2]))

        assert first == (4, (4, 4))
        assert second == first

    def test_data_parallel_warmup_uses_local_batch_capacity(self):
        assert _decode_graph_buckets(16, 2) == [1, 2, 4, 8]
        assert _decode_graph_buckets(20, 2) == [1, 2, 4, 8, 16]

    def test_single_rank_graph_key_reuses_padded_bucket(self):
        runner = self._runner()
        runner.dp_size = 1
        runner.dp_rank = 0

        first = runner._graph_key(torch.zeros(3, dtype=torch.int32), self._metadata([3]))
        second = runner._graph_key(torch.zeros(4, dtype=torch.int32), self._metadata([4]))

        assert first == (4, (4,))
        assert second == first

    def test_graph_key_accepts_dummy_execution_row_for_empty_data_parallel_rank(self):
        runner = self._runner(dp_rank=1)
        input_ids = torch.zeros(1, dtype=torch.int32)

        assert runner._graph_key(input_ids, self._metadata([5, 0])) == (
            8,
            (8, 8),
        )

    def test_graph_key_rejects_unbalanced_unwarmed_bucket(self):
        runner = self._runner()
        input_ids = torch.zeros(9, dtype=torch.int32)

        assert runner._graph_key(input_ids, self._metadata([9, 7])) is None

    def test_can_execute_requires_warmed_graph(self):
        runner = self._runner()
        input_ids = torch.zeros(3, dtype=torch.int32)
        metadata = self._metadata([3, 1])
        graph_key = runner._graph_key(input_ids, metadata)

        assert not runner.can_execute(input_ids, metadata)
        runner._graphs[graph_key] = object()
        assert runner.can_execute(input_ids, metadata)

    @pytest.mark.parametrize("token_counts", ([3], [3, -1], [3, 2]))
    def test_graph_key_rejects_invalid_data_parallel_metadata(self, token_counts):
        runner = self._runner(dp_rank=1)
        input_ids = torch.zeros(1, dtype=torch.int32)

        with pytest.raises(RuntimeError):
            runner._graph_key(input_ids, self._metadata(token_counts))


# ---------------------------------------------------------------------------
# Tests: DecodeAclGraphRunner speculative metadata
# ---------------------------------------------------------------------------


class TestDecodeAclGraphSpeculativeMetadata:
    @staticmethod
    def _runner() -> DecodeAclGraphRunner:
        return DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=4,
            max_model_len=8,
        )

    @staticmethod
    def _metadata() -> SimpleNamespace:
        return SimpleNamespace(
            slot_mapping=torch.arange(4, dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0, 1, 2, 4, 6], dtype=torch.int32),
            paged_kv_indices=torch.tensor([10, 10, 20, 21, 20, 21], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([3, 4, 3, 4], dtype=torch.int32),
            q_cu_seq_lens=torch.tensor([0, 2, 4], dtype=torch.int32),
            kv_cu_seq_lens=torch.tensor([0, 4, 12], dtype=torch.int32),
            kv_seq_lens_host=torch.tensor([4, 8], dtype=torch.int32),
            kv_seq_lens_host_values=[4, 8],
            block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            kv_seq_lens=torch.tensor([4, 8], dtype=torch.int32),
            q_seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            expanded_decode_metadata=SimpleNamespace(
                enabled=True,
                kv_seq_lens=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
                block_table=torch.tensor(
                    [[10, 11], [10, 11], [20, 21], [20, 21]],
                    dtype=torch.int32,
                ),
                paged_kv_indptr=torch.tensor([0, 1, 2, 4, 6], dtype=torch.int32),
                paged_kv_indices=torch.tensor([10, 10, 20, 21, 20, 21], dtype=torch.int32),
                paged_kv_last_page_len=torch.tensor([3, 4, 3, 4], dtype=torch.int32),
                paged_attention_tiling_data=None,
                kv_seq_lens_host=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
                kv_seq_lens_host_values=[3, 4, 7, 8],
            ),
            is_prefill=False,
            is_chunked_prefill=True,
        )

    def test_expanded_metadata_selects_matching_paged_kv_rows(self) -> None:
        runner = self._runner()
        (
            block_table,
            kv_seq_lens,
            _,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = runner._decode_metadata(self._metadata())

        assert block_table.tolist() == [
            [10, 11],
            [10, 11],
            [20, 21],
            [20, 21],
        ]
        assert kv_seq_lens.tolist() == [3, 4, 7, 8]
        assert paged_kv_indptr.tolist() == [0, 1, 2, 4, 6]
        assert paged_kv_indices.tolist() == [10, 10, 20, 21, 20, 21]
        assert paged_kv_last_page_len.tolist() == [3, 4, 3, 4]

    def test_decode_metadata_rebuilds_token_row_paging_metadata(self) -> None:
        metadata = self._metadata()
        metadata.expanded_decode_metadata = None
        metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
        metadata.block_table = torch.tensor(
            [[10, 11], [10, 11], [20, 21], [20, 21]],
            dtype=torch.int32,
        )
        metadata.kv_seq_lens = torch.tensor([3, 4, 7, 8], dtype=torch.int32)
        metadata.kv_seq_lens_host_values = [3, 4, 7, 8]
        metadata.paged_kv_indptr = torch.tensor([0, 1, 3], dtype=torch.int32)
        metadata.paged_kv_indices = torch.tensor([10, 20, 21], dtype=torch.int32)
        metadata.paged_kv_last_page_len = torch.tensor([4, 4], dtype=torch.int32)

        (
            _,
            _,
            _,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._runner()._decode_metadata(metadata)

        assert paged_kv_indptr.tolist() == [0, 1, 2, 4, 6]
        assert paged_kv_indices.tolist() == [10, 10, 20, 21, 20, 21]
        assert paged_kv_last_page_len.tolist() == [3, 4, 3, 4]

    def test_expanded_chunked_verify_can_use_decode_graph(self) -> None:
        runner = self._runner()
        input_ids = torch.arange(4, dtype=torch.int32)

        assert runner.can_execute(input_ids, self._metadata())

    def test_decode_batch_limit_uses_speculative_tokens_and_dp_global_max(
        self,
    ) -> None:
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=64,
            max_model_len=8,
            decode_batch_size_limit=16,
            num_decoding_tokens=4,
        )
        metadata = SimpleNamespace(
            dp_execution_token_counts=(32, 64),
        )

        assert runner._decode_batch_sizes(
            torch.zeros(32, dtype=torch.int32),
            metadata,
        ) == (8, 16)

    def test_mtp3_batch_eight_uses_32_row_graph_bucket(self) -> None:
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=64,
            max_model_len=8,
            decode_batch_size_limit=16,
            num_decoding_tokens=4,
        )
        metadata = SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=False,
            dp_execution_token_counts=(),
        )

        with patch.object(
            runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ):
            assert runner.can_execute(
                torch.zeros(32, dtype=torch.int32),
                metadata,
            )

    def test_warmup_captures_with_scheduler_metadata_once(self) -> None:
        runner = self._runner()
        input_ids = torch.arange(4, dtype=torch.int32)
        positions = torch.arange(4, dtype=torch.int32)
        metadata = self._metadata()
        graph_key = runner._graph_key(
            padded_batch_size=4,
            is_expanded=True,
            input_embedding=None,
        )

        with patch.object(runner, "_prepare_graph_entry") as prepare:
            runner.warmup(input_ids, positions, metadata)
            prepare.assert_called_once_with(
                input_ids,
                positions,
                metadata,
                None,
            )

            runner._graphs[graph_key] = object()
            runner.warmup(input_ids, positions, metadata)
            prepare.assert_called_once()

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            (
                "block_table",
                torch.arange(8, dtype=torch.int32),
                "block_table must be two-dimensional",
            ),
            (
                "kv_seq_lens",
                torch.tensor([3, 4, 7], dtype=torch.int32),
                "kv_seq_lens must contain one value per sequence",
            ),
            (
                "kv_seq_lens_host",
                torch.tensor([3, 4, 7], dtype=torch.int32),
                "kv_seq_lens_host must contain one value per sequence",
            ),
            (
                "paged_kv_indptr",
                torch.tensor([0, 1, 2, 4], dtype=torch.int32),
                "paged_kv_indptr must contain one offset per sequence",
            ),
            (
                "paged_kv_indices",
                torch.tensor([[10, 10], [20, 21]], dtype=torch.int32),
                "paged_kv_indices must be a non-empty flat page list",
            ),
            (
                "paged_kv_last_page_len",
                torch.tensor([3, 4, 3], dtype=torch.int32),
                "paged_kv_last_page_len must contain one value per sequence",
            ),
        ],
    )
    def test_expanded_metadata_shape_mismatch_fails(
        self,
        field: str,
        value: torch.Tensor,
        message: str,
    ) -> None:
        metadata = self._metadata()
        setattr(metadata.expanded_decode_metadata, field, value)

        with pytest.raises(RuntimeError, match=message):
            self._runner()._decode_metadata(metadata)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            (
                "paged_kv_indptr",
                torch.tensor([1, 1, 2, 4, 6], dtype=torch.int32),
                "must start at zero",
            ),
            (
                "paged_kv_indptr",
                torch.tensor([0, 2, 1, 4, 6], dtype=torch.int32),
                "must be monotonic",
            ),
            (
                "paged_kv_indptr",
                torch.tensor([0, 1, 2, 4, 5], dtype=torch.int32),
                "terminal page offset must match page count",
            ),
            (
                "paged_kv_last_page_len",
                torch.tensor([3, 4, 0, 4], dtype=torch.int32),
                "last-page lengths must be positive",
            ),
            (
                "paged_kv_last_page_len",
                torch.tensor([3, 4, 5, 4], dtype=torch.int32),
                "must not exceed block size",
            ),
        ],
    )
    def test_expanded_paged_metadata_invariant_fails(
        self,
        field: str,
        value: torch.Tensor,
        message: str,
    ) -> None:
        metadata = self._metadata()
        setattr(metadata.expanded_decode_metadata, field, value)

        with pytest.raises(RuntimeError, match=message):
            self._runner()._decode_metadata(metadata)

    def test_expanded_page_count_exceeding_capacity_fails(self) -> None:
        metadata = self._metadata()
        metadata.expanded_decode_metadata.kv_seq_lens_host_values = [3, 4, 7, 9]

        with pytest.raises(RuntimeError, match="exceeds block-table capacity"):
            self._runner()._decode_metadata(metadata)

    @pytest.mark.parametrize(
        ("input_ids", "slot_mapping", "message"),
        [
            (
                torch.arange(3, dtype=torch.int32),
                torch.arange(4, dtype=torch.int32),
                "input_ids must contain one token per metadata row",
            ),
            (
                torch.arange(4, dtype=torch.int32),
                torch.arange(3, dtype=torch.int32),
                "slot_mapping must contain one slot per token",
            ),
        ],
    )
    def test_token_layout_mismatch_fails(
        self,
        input_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        message: str,
    ) -> None:
        with pytest.raises(RuntimeError, match=message):
            self._runner()._validate_decode_token_layout(
                input_ids,
                torch.arange(4, dtype=torch.int32),
                slot_mapping,
                metadata_row_count=4,
            )

    def test_replay_returns_static_output_view(self) -> None:
        runner = self._runner()
        batch_size = 3
        padded_batch_size = 4
        static_output = torch.arange(12).reshape(padded_batch_size, 3)
        graph = MagicMock()
        entry = SimpleNamespace(
            graph=graph,
            batch_size=padded_batch_size,
            static_output=static_output,
            static_metadata=SimpleNamespace(),
            graph_tasks=[],
            execution_state=SimpleNamespace(persistent_buffers={}),
        )
        graph_key = runner._graph_key(
            padded_batch_size,
            is_expanded=False,
            input_embedding=None,
        )
        runner._graphs[graph_key] = entry

        replay_stream = MagicMock()
        update_stream = MagicMock()
        replay_done_event = MagicMock()
        update_done_event = MagicMock()
        current_stream = MagicMock()
        runner._stream = replay_stream
        runner._update_stream = update_stream
        runner._replay_done_event = replay_done_event
        runner._update_done_event = update_done_event
        fake_npu = SimpleNamespace(
            current_stream=MagicMock(return_value=current_stream),
            stream=MagicMock(return_value=nullcontext()),
        )
        metadata = SimpleNamespace(expanded_decode_metadata=None)

        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(
                runner,
                "_prepare_graph_entry",
                return_value=entry,
            ),
        ):
            output = runner.execute(
                torch.arange(batch_size, dtype=torch.int32),
                torch.arange(batch_size, dtype=torch.int32),
                metadata,
            )

        assert output.shape == (batch_size, 3)
        assert output.data_ptr() == static_output.data_ptr()
        output[0, 0] = -1
        assert static_output[0, 0].item() == -1
        replay_stream.wait_stream.assert_called_once_with(current_stream)
        current_stream.wait_stream.assert_called_once_with(replay_stream)
        graph.replay.assert_called_once_with()

    @pytest.mark.parametrize("new_bucket", [False, True])
    def test_graph_buffer_update_waits_for_previous_replay(self, new_bucket: bool) -> None:
        runner = self._runner()
        metadata = self._metadata()
        entry = SimpleNamespace(
            graph=None if new_bucket else MagicMock(),
            static_metadata=SimpleNamespace(),
            execution_state=SimpleNamespace(persistent_buffers={}),
        )
        if not new_bucket:
            runner._graphs[runner._graph_key(4, True, None)] = entry
        replay_stream = MagicMock()
        update_stream = MagicMock()
        replay_done_event = MagicMock()
        current_stream = MagicMock()
        runner._stream = replay_stream
        runner._update_stream = update_stream
        runner._replay_done_event = replay_done_event
        runner.attention_backend.prepare = MagicMock()
        fake_npu = SimpleNamespace(current_stream=MagicMock(return_value=current_stream))

        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(runner, "_allocate_entry", return_value=entry),
            patch.object(runner, "_capture"),
            patch.object(runner, "_fill_entry") as fill_entry,
        ):
            fill_entry.side_effect = lambda *args: current_stream.wait_event.assert_called_once_with(replay_done_event)
            result = runner._prepare_graph_entry(
                torch.arange(4, dtype=torch.int32),
                torch.arange(4, dtype=torch.int32),
                metadata,
                None,
            )

        assert result is entry
        current_stream.wait_event.assert_called_once_with(replay_done_event)
        fill_entry.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.bind_kv_caches
# ---------------------------------------------------------------------------


class TestNormalizeLayerCaches:
    def test_legacy_five_slot_cache_keeps_generic_layout(self):
        tensors = tuple(torch.full((1,), value) for value in range(1, 6))

        cache = normalize_layer_caches([tensors])[0]

        assert cache.key is tensors[0]
        assert cache.value is tensors[1]
        assert cache.index is tensors[2]
        assert cache.conv is tensors[3]
        assert cache.ssm is tensors[4]
        assert cache.swa is None
        assert cache.compress_kv_state is None
        assert cache.compress_score_state is None
        assert cache.compress_index_kv_state is None
        assert cache.compress_index_score_state is None
        assert cache.indexer_scale is None

    def test_deepseek_v4_eleven_slot_cache_maps_all_slots(self):
        tensors = tuple(torch.full((1,), value) for value in range(1, 12))

        cache = normalize_layer_caches([tensors])[0]

        assert (
            cache.key,
            cache.value,
            cache.index,
            cache.conv,
            cache.ssm,
            cache.swa,
            cache.compress_kv_state,
            cache.compress_score_state,
            cache.compress_index_kv_state,
            cache.compress_index_score_state,
            cache.indexer_scale,
        ) == tensors

    def test_empty_deepseek_v4_slots_are_normalized_to_none(self):
        cache = normalize_layer_caches([(torch.ones(1), torch.ones(1), *(torch.empty(0),) * 9)])[0]

        assert cache.key is not None
        assert cache.value is not None
        assert cache.index is None
        assert cache.indexer_scale is None


class TestBindKvCaches:
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_correct_count(self, mock_create):
        backend = StubAttentionBackend()
        mock_create.return_value = backend
        model = _FakeModel(num_layers=2)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv, kv])
        assert len(backend._kv_caches) == 2

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_wrong_count_raises(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=2)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        with pytest.raises(ValueError, match="layer count does not match"):
            executor.bind_kv_caches([kv])

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_idempotent(self, mock_create):
        backend = StubAttentionBackend()
        mock_create.return_value = backend
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])
        executor.bind_kv_caches([kv])  # should not raise or re-bind


@pytest.mark.parametrize(
    ("config", "exception", "message"),
    [
        ({"cp_size": 0}, ValueError, "cp_size"),
        ({"dp_size": 0}, ValueError, "dp_size"),
        ({"cp_size": 2, "cp_rank": -1}, ValueError, "cp_rank"),
        ({"cp_size": 2, "cp_rank": 2}, ValueError, "cp_rank"),
        ({"kv_split_size": -1}, ValueError, "kv_split_size"),
        ({"cp_size": 4, "kv_split_size": 3}, ValueError, "divisor"),
        ({"cp_size": 1, "dp_size": 2, "kv_split_size": 2}, NotImplementedError, "DCP requires dp_size == 1"),
    ],
)
def test_executor_rejects_invalid_cp_dcp_topology_before_backend_creation(
    config: dict, exception: type[Exception], message: str
) -> None:
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
        patch("xllm.python.model_executor.executor._create_attention_backend") as create_backend,
        pytest.raises(exception, match=message),
    ):
        ModelExecutor(_FakeModel(num_layers=1), config, max_seqs_per_batch=4)
    create_backend.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        {"cp_size": 4, "cp_rank": 3, "kv_split_size": 0},
        {"cp_size": 4, "cp_rank": 3, "kv_split_size": 2},
        {"cp_size": 1, "dp_size": 2, "kv_split_size": 1},
        {"cp_size": 1, "dp_size": 1, "kv_split_size": 2},
    ],
)
def test_executor_preserves_supported_cp_dcp_topologies(config: dict) -> None:
    config = {
        "model_type": "glm_moe_dsa",
        "enable_disagg_pd": True,
        "instance_role": "PREFILL",
        **config,
    }
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
        patch("xllm.python.model_executor.executor._create_attention_backend", return_value=StubAttentionBackend()),
    ):
        executor = ModelExecutor(_FakeModel(num_layers=1), config, max_seqs_per_batch=4)
    assert executor.eager_runner.cp_size == config["cp_size"]
    assert executor.eager_runner.cp_rank == config.get("cp_rank", 0)


@pytest.mark.parametrize("graph_backend", ["cudagraphs", "inductor", "reduce-overhead"])
def test_cp_rejects_unsupported_graph_backend_before_backend_creation(graph_backend: str) -> None:
    with (
        patch("xllm.python.model_executor.executor._create_attention_backend") as create_backend,
        patch("xllm.python.model_executor.runners.decode_cuda_graph.DecodeCudaGraphRunner"),
        pytest.raises(NotImplementedError, match="Context-Parallel"),
    ):
        ModelExecutor(
            _FakeModel(num_layers=1),
            {"cp_size": 2, "max_position_embeddings": 128, "python_graph_backend": graph_backend},
            max_seqs_per_batch=4,
        )
    create_backend.assert_not_called()


@pytest.mark.parametrize("graph_backend", ["off", "ACLGraph"])
def test_cp_preserves_eager_prefill_with_resolved_acl_graph(graph_backend: str) -> None:
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
        patch("xllm.python.model_executor.executor._create_attention_backend", return_value=StubAttentionBackend()),
    ):
        executor = ModelExecutor(
            _FakeModel(num_layers=1),
            {
                "model_type": "glm_moe_dsa",
                "cp_size": 2,
                "kv_split_size": 1,
                "max_position_embeddings": 128,
                "enable_graph": True,
                "python_graph_backend": graph_backend,
            },
            max_seqs_per_batch=4,
        )
    assert executor.eager_runner.cp_size == 2
    assert executor.inductor_runner is None
    assert isinstance(executor.decode_graph_runner, DecodeAclGraphRunner)
    assert not executor.decode_graph_runner.can_execute(
        torch.ones(3, dtype=torch.int32),
        SimpleNamespace(is_prefill=True, is_chunked_prefill=False, expanded_decode_metadata=None),
    )


@pytest.mark.parametrize(
    ("overrides", "decoding_tokens", "message"),
    [
        ({"model_type": "qwen3_5"}, 1, "model_type"),
        ({"model_type": "deepseek_v32"}, 1, "model_type"),
        ({"model_type": "glm_moe_dsa_mtp", "is_draft_engine": True}, 1, "model_type"),
        ({"model_type": ""}, 1, "model_type"),
        ({"instance_role": "DECODE"}, 1, "DEFAULT or PREFILL"),
        ({"instance_role": "MIX"}, 1, "DEFAULT or PREFILL"),
        ({"task_type": "embed"}, 1, "generate"),
        ({"num_speculative_tokens": 3, "speculative_algorithm": "mTp"}, 1, "MTP"),
        ({}, 4, "MTP"),
        ({"num_speculative_tokens": 3, "speculative_algorithm": "DFlash2"}, 1, "aux-hidden"),
        ({"num_speculative_tokens": 3, "speculative_algorithm": "DSpark"}, 1, "aux-hidden"),
        ({"num_speculative_tokens": 3, "speculative_algorithm": "Eagle3"}, 1, "aux-hidden"),
        ({"kv_split_size": 0}, 1, "disaggregated PD"),
        ({"kv_split_size": 2, "instance_role": "PREFILL"}, 1, "disaggregated PD"),
        ({"kv_split_size": 2, "enable_disagg_pd": True}, 1, "PREFILL"),
    ],
)
def test_cp_model_admission_rejects_before_backend_creation(
    overrides: dict, decoding_tokens: int, message: str
) -> None:
    config = {"model_type": "glm_moe_dsa", "cp_size": 2, "kv_split_size": 1, **overrides}
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
        patch("xllm.python.model_executor.executor._create_attention_backend") as create_backend,
        pytest.raises(NotImplementedError, match=message),
    ):
        ModelExecutor(_FakeModel(num_layers=1), config, max_seqs_per_batch=3, num_decoding_tokens=decoding_tokens)
    create_backend.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "glm_moe_dsa", "cp_size": 2, "kv_split_size": 1},
        {
            "model_type": "glm_moe_dsa",
            "cp_size": 2,
            "kv_split_size": 0,
            "enable_disagg_pd": True,
            "instance_role": "PREFILL",
        },
        {"model_type": "qwen3", "cp_size": 2, "kv_split_size": 1},
        {"model_type": "glm_moe_dsa", "cp_size": 1, "kv_split_size": 2, "num_speculative_tokens": 3},
        {"model_type": "glm_moe_dsa_mtp", "cp_size": 1, "kv_split_size": 2, "is_draft_engine": True},
    ],
)
def test_cp_model_admission_preserves_prefill_and_separate_dcp_mtp(config: dict) -> None:
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
        patch("xllm.python.model_executor.executor._create_attention_backend", return_value=StubAttentionBackend()),
    ):
        executor = ModelExecutor(_FakeModel(num_layers=1), config, max_seqs_per_batch=3)
    assert executor.eager_runner.cp_size == config["cp_size"]


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.execute routing
# ---------------------------------------------------------------------------


def _make_eager_runner(*, is_mla: bool = True) -> EagerRunner:
    runner = object.__new__(EagerRunner)
    backend_type = _MlaStubAttentionBackend if is_mla else StubAttentionBackend
    runner.attention_backend = backend_type()
    runner.cp_size = 4
    runner.cp_rank = 2
    runner.device = torch.device("cpu")
    runner.layer_caches = []
    runner.model = MagicMock(return_value=torch.ones(2))
    return runner


def test_eager_runner_preserves_qwen_pure_prefill_cp_context_contract() -> None:
    runner = _make_eager_runner(is_mla=False)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=torch.tensor([3, 5], dtype=torch.int32),
        kv_seq_lens_host=None,
    )

    with patch(
        "xllm.python.model_executor.runners.eager.build_cp_context",
        return_value=object(),
    ) as build_context:
        runner.execute(torch.zeros(8), torch.arange(8), metadata)

    build_context.assert_called_once_with(
        [3, 5],
        [3, 5],
        4,
        2,
        torch.device("cpu"),
    )


def test_eager_runner_preserves_qwen_missing_length_fallback() -> None:
    runner = _make_eager_runner(is_mla=False)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=None,
        kv_seq_lens_host=None,
    )

    with patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context:
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert runner.attention_backend._prepared
    runner.model.assert_called_once()


def test_eager_runner_builds_cp_context_for_chunked_prefill() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=torch.tensor([3, 5], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([11, 13], dtype=torch.int32),
    )

    with patch(
        "xllm.python.model_executor.runners.eager.build_cp_context",
        return_value=object(),
    ) as build_context:
        runner.execute(torch.zeros(8), torch.arange(8), metadata)

    build_context.assert_called_once_with(
        [3, 5],
        [11, 13],
        4,
        2,
        torch.device("cpu"),
    )


@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize(
    ("input_shape", "position_shape", "embedding_rows"),
    [
        ((7,), (7,), None),
        ((9,), (9,), None),
        ((2, 4), (8,), None),
        ((8,), (7,), None),
        ((8,), (2, 4), None),
        ((8,), (8,), 7),
    ],
)
def test_cp_rejects_misaligned_packed_inputs_before_prepare(
    is_mla: bool,
    input_shape: tuple[int, ...],
    position_shape: tuple[int, ...],
    embedding_rows: int | None,
) -> None:
    runner = _make_eager_runner(is_mla=is_mla)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=torch.tensor([3, 5], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([11, 13], dtype=torch.int32),
    )
    embedding = None if embedding_rows is None else torch.zeros(embedding_rows, 4)
    with (
        patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context,
        pytest.raises(ValueError, match="CP packed"),
    ):
        runner.execute(torch.zeros(input_shape), torch.zeros(position_shape), metadata, embedding)
    build_context.assert_not_called()
    assert not runner.attention_backend._prepared
    runner.model.assert_not_called()


def test_eager_runner_rejects_mixed_cp_before_collective() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=True,
        is_spec_verify=False,
    )

    with (
        patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context,
        pytest.raises(NotImplementedError, match="mixed batches"),
    ):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert not runner.attention_backend._prepared


def test_eager_runner_rejects_mla_spec_verify_cp_before_collective() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=False,
        is_spec_verify=True,
    )

    with (
        patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context,
        pytest.raises(NotImplementedError, match="MTP speculative verification"),
    ):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert not runner.attention_backend._prepared


@pytest.mark.parametrize(
    ("is_mla", "is_chunked_prefill", "is_mixed", "is_spec_verify"),
    [
        (False, True, True, False),
        (False, True, False, True),
        (True, False, False, True),
    ],
)
def test_eager_runner_preserves_non_cp_fallback(
    is_mla: bool,
    is_chunked_prefill: bool,
    is_mixed: bool,
    is_spec_verify: bool,
) -> None:
    runner = _make_eager_runner(is_mla=is_mla)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=is_chunked_prefill,
        is_mixed=is_mixed,
        is_spec_verify=is_spec_verify,
    )

    def execute_model(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert get_forward_context().cp_context is None
        return input_ids + positions

    runner.model.side_effect = execute_model
    with patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context:
        output = runner.execute(torch.ones(1), torch.ones(1), metadata)

    build_context.assert_not_called()
    assert runner.attention_backend._prepared
    runner.model.assert_called_once()
    torch.testing.assert_close(output, torch.full((1,), 2.0))


@pytest.mark.parametrize(
    ("q_seq_lens_host", "kv_seq_lens_host"),
    [
        (None, None),
        (None, torch.tensor([1], dtype=torch.int32)),
        (torch.tensor([1], dtype=torch.int32), None),
    ],
)
def test_eager_runner_rejects_missing_cp_lengths(
    q_seq_lens_host: torch.Tensor | None,
    kv_seq_lens_host: torch.Tensor | None,
) -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=q_seq_lens_host,
        kv_seq_lens_host=kv_seq_lens_host,
    )

    with pytest.raises(RuntimeError, match="requires host query and KV"):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    assert not runner.attention_backend._prepared


@pytest.mark.parametrize(
    "bad_lengths",
    [
        torch.empty(1, dtype=torch.int32, device="meta"),
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([1.5], dtype=torch.float32),
    ],
)
@pytest.mark.parametrize("field", ["q_seq_lens_host", "kv_seq_lens_host"])
def test_cp_rejects_invalid_host_lengths_before_prepare(bad_lengths: torch.Tensor, field: str) -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
    )
    setattr(metadata, field, bad_lengths)
    with (
        patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context,
        pytest.raises(ValueError, match="CPU int32/int64 vector"),
    ):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)
    build_context.assert_not_called()
    assert not runner.attention_backend._prepared
    runner.model.assert_not_called()


class TestExecuteRouting:
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_execute_without_bind_raises(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        metadata = MagicMock(spec=AttentionMetadata)
        with pytest.raises(RuntimeError, match="KV caches are not bound"):
            executor.execute(torch.zeros(1), torch.zeros(1), metadata)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_execute_routes_to_eager_runner(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        metadata = MagicMock(spec=AttentionMetadata)
        executor.eager_runner = MagicMock()
        grad_enabled = None

        def execute(*_args):
            nonlocal grad_enabled
            grad_enabled = torch.is_grad_enabled()
            return torch.ones(5)

        executor.eager_runner.execute.side_effect = execute

        result = executor.execute(torch.zeros(1), torch.zeros(1), metadata)
        executor.eager_runner.execute.assert_called_once()
        assert grad_enabled is False
        assert torch.equal(result, torch.ones(5))

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_inductor_runner_takes_priority_over_eager(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        executor.inductor_runner = MagicMock()
        executor.inductor_runner.execute.return_value = torch.ones(3)

        metadata = MagicMock(spec=AttentionMetadata)
        result = executor.execute(torch.zeros(1), torch.zeros(1), metadata)
        executor.inductor_runner.execute.assert_called_once()
        assert torch.equal(result, torch.ones(3))

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_mtp_topk_state_routes_to_eager(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)
        executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
        executor.inductor_runner = MagicMock()
        executor.eager_runner = MagicMock()
        executor.eager_runner.execute.return_value = torch.ones(2)
        metadata = MagicMock(spec=AttentionMetadata)
        mtp_topk = torch.zeros((1, 1, 2), dtype=torch.int32)
        input_ids = torch.zeros(1)
        positions = torch.zeros(1)

        result = executor.execute(
            input_ids,
            positions,
            metadata,
            mtp_topk_indices=mtp_topk,
        )

        executor.inductor_runner.execute.assert_not_called()
        args = executor.eager_runner.execute.call_args.args
        assert args == (input_ids, positions, metadata, None, None, mtp_topk)
        assert torch.equal(result, torch.ones(2))

    @pytest.mark.parametrize("runner_cls", [DecodeAclGraphRunner, DecodeCudaGraphRunner])
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_mtp_topk_only_routes_to_acl_graph(self, mock_create: MagicMock, runner_cls: type) -> None:
        mock_create.return_value = StubAttentionBackend()
        executor = ModelExecutor(_FakeModel(num_layers=1), {}, max_seqs_per_batch=4)
        executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
        graph_runner = create_autospec(runner_cls, instance=True)
        graph_runner.can_execute.return_value = True
        executor.decode_graph_runner = graph_runner
        executor.eager_runner = MagicMock()
        ids = torch.arange(2, dtype=torch.int32)
        metadata = MagicMock(spec=AttentionMetadata)
        topk = torch.ones((2, 1, 8), dtype=torch.int32)

        result = executor.execute(ids, ids, metadata, mtp_topk_indices=topk)

        if runner_cls is DecodeAclGraphRunner:
            graph_runner.can_execute.assert_called_once_with(ids, metadata, None, mtp_topk_indices=topk)
            graph_runner.execute.assert_called_once_with(ids, ids, metadata, None, mtp_topk_indices=topk)
            assert result is graph_runner.execute.return_value
            executor.eager_runner.execute.assert_not_called()
        else:
            graph_runner.can_execute.assert_not_called()
            executor.eager_runner.execute.assert_called_once_with(ids, ids, metadata, None, None, topk)

    @pytest.mark.parametrize("runner_cls", [DecodeAclGraphRunner, DecodeCudaGraphRunner])
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_graph_warmup_uses_scheduler_inputs(self, mock_create: MagicMock, runner_cls: type) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        input_ids = torch.zeros(4, dtype=torch.int32)
        positions = torch.arange(4, dtype=torch.int32)
        metadata = MagicMock(spec=AttentionMetadata)
        graph_runner = create_autospec(runner_cls, instance=True)
        graph_runner.can_execute.return_value = True
        graph_runner.execute.return_value = torch.ones(4)
        executor.decode_graph_runner = graph_runner

        result = executor.execute(input_ids, positions, metadata)

        if runner_cls is DecodeAclGraphRunner:
            graph_runner.warmup.assert_called_once_with(input_ids, positions, metadata, None)
        else:
            graph_runner.warmup.assert_not_called()
        graph_runner.execute.assert_called_once_with(
            input_ids,
            positions,
            metadata,
            None,
        )
        assert torch.equal(result, torch.ones(4))
