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

"""Tests for the NPU ACL decode-graph runner."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
)


def _runner() -> DecodeAclGraphRunner:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    return DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )


def _metadata(linear_state_indices: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        slot_mapping=torch.arange(4, dtype=torch.int32),
        paged_kv_indptr=torch.arange(5, dtype=torch.int32),
        paged_kv_indices=torch.tensor([10, 20, 30, 40], dtype=torch.int32),
        paged_kv_last_page_len=torch.arange(1, 5, dtype=torch.int32),
        block_table=torch.tensor(
            [[10, 0], [20, 0], [30, 0], [40, 0]],
            dtype=torch.int32,
        ),
        kv_seq_lens=torch.arange(1, 5, dtype=torch.int32),
        kv_seq_lens_host_values=[1, 2, 3, 4],
        kv_cu_seq_lens=torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32),
        linear_state_indices=linear_state_indices,
        expanded_decode_metadata=None,
    )


@pytest.mark.parametrize("capacity,rows", [(3, 3), (6, 5), (6, 6), (10, 9), (18, 17)])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_final_partial_bucket_preserves_declared_graph_capacity(capacity: int, rows: int, dp_size: int) -> None:
    runner = _runner()
    runner.max_batch = capacity
    runner.dp_size = dp_size
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        dp_execution_token_counts=(rows, rows) if dp_size > 1 else (),
        dp_is_decode=(1, 1) if dp_size > 1 else (),
    )
    ids = torch.arange(rows, dtype=torch.int32)
    with patch.object(runner, "_has_compatible_decode_metadata", return_value=True):
        assert runner.can_execute(ids, metadata)
    # The last bucket must fit backend buffers allocated to exactly capacity.
    # Rounding it up would overflow those buffers; dropping it uses eager for
    # valid requests (e.g. three MTP requests with five/six repair rows).
    with (
        patch.object(runner, "_allocate_entry", side_effect=RuntimeError("capture boundary")) as allocate,
        pytest.raises(RuntimeError, match="capture boundary"),
    ):
        runner._prepare_graph_entry(ids, ids, metadata, None)
    assert allocate.call_args.args[0] == capacity
    runner._graphs[runner._graph_key(capacity, False, None)] = SimpleNamespace()
    with patch.object(runner, "_prepare_graph_entry") as prepare:
        runner.warmup(ids, ids, metadata)
    prepare.assert_not_called()


@pytest.mark.parametrize("dp_size", [1, 2])
def test_partial_bucket_still_rejects_rows_above_capacity(dp_size: int) -> None:
    runner = _runner()
    runner.max_batch = 6
    runner.dp_size = dp_size
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        dp_execution_token_counts=(7, 7) if dp_size > 1 else (),
        dp_is_decode=(1, 1) if dp_size > 1 else (),
    )
    ids = torch.arange(7, dtype=torch.int32)
    with patch.object(runner, "_has_compatible_decode_metadata", return_value=True):
        assert not runner.can_execute(ids, metadata)
    with pytest.raises(ValueError, match="exceeds ACL graph capacity"):
        runner.warmup(ids, ids, metadata)


def test_dp_warmup_checks_global_bucket_already_captured() -> None:
    runner = _runner()
    runner.dp_size = 2
    ids = torch.arange(2, dtype=torch.int32)
    metadata = SimpleNamespace(dp_execution_token_counts=(2, 5))
    runner._graphs[runner._graph_key(8, False, None)] = SimpleNamespace()
    with patch.object(runner, "_prepare_graph_entry") as prepare:
        runner.warmup(ids, ids, metadata)
    prepare.assert_not_called()


@pytest.mark.parametrize(
    ("logical_sequence_counts", "rows", "expected"),
    [
        ((3,), 5, True),
        ((3,), 6, True),
        ((4,), 4, False),
    ],
)
@pytest.mark.parametrize("num_decoding_tokens", [1, 4])
def test_batch_limit_uses_logical_sequences_for_mtp_repair_rows(
    logical_sequence_counts: tuple[int, ...],
    rows: int,
    expected: bool,
    num_decoding_tokens: int,
) -> None:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        decode_batch_size_limit=3,
        num_decoding_tokens=num_decoding_tokens,
    )
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        dp_global_sequence_nums=logical_sequence_counts,
        dp_execution_token_counts=(rows,),
    )
    input_ids = torch.arange(rows, dtype=torch.int32)

    with patch.object(runner, "_has_compatible_decode_metadata", return_value=True):
        assert runner.can_execute(input_ids, metadata) is expected


@pytest.mark.parametrize(
    ("logical_sequence_counts", "message"),
    [((1, 2), "one value per DP rank"), ((-1,), "nonnegative")],
)
def test_batch_limit_rejects_malformed_logical_sequence_counts(
    logical_sequence_counts: tuple[int, ...], message: str
) -> None:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        decode_batch_size_limit=3,
    )
    metadata = SimpleNamespace(dp_global_sequence_nums=logical_sequence_counts)

    with pytest.raises(RuntimeError, match=message):
        runner._decode_batch_sizes(torch.zeros(1, dtype=torch.int32), metadata)


@pytest.mark.parametrize("dp_rank", [0, 1])
@pytest.mark.parametrize(
    ("logical_sequence_counts", "execution_counts", "expected"),
    [
        ((0, 3), (1, 5), True),
        ((3, 2), (6, 3), True),
        ((0, 4), (1, 4), False),
        ((3, 2), (7, 3), False),
    ],
)
def test_dp_batch_limit_and_row_capacity_agree_across_ranks(
    dp_rank: int,
    logical_sequence_counts: tuple[int, ...],
    execution_counts: tuple[int, ...],
    expected: bool,
) -> None:
    runner = _runner()
    runner.dp_size = 2
    runner.dp_rank = dp_rank
    runner.max_batch = 6
    runner.decode_batch_size_limit = 3
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        dp_global_sequence_nums=logical_sequence_counts,
        dp_execution_token_counts=execution_counts,
        dp_is_decode=(1, 1),
    )
    input_ids = torch.arange(execution_counts[dp_rank], dtype=torch.int32)
    with patch.object(runner, "_has_compatible_decode_metadata", return_value=True):
        assert runner.can_execute(input_ids, metadata) is expected
    if expected:
        assert runner._padded_batch_size(input_ids.numel(), metadata) == 6


def test_new_capture_waits_for_input_and_metadata_writes() -> None:
    runner = _runner()
    runner._stream = MagicMock()
    current_stream = MagicMock()
    entry = SimpleNamespace(
        batch_size=1,
        static_metadata=SimpleNamespace(),
        execution_state=SimpleNamespace(persistent_buffers={}),
        static_mtp_topk_indices=None,
    )
    fake_npu = SimpleNamespace(
        current_stream=MagicMock(return_value=current_stream),
        stream=MagicMock(side_effect=lambda stream: nullcontext()),
        synchronize=MagicMock(),
        NPUGraph=MagicMock(),
        graph=MagicMock(side_effect=lambda graph, stream: nullcontext()),
    )

    def forward_static(capture_entry: SimpleNamespace) -> torch.Tensor:
        # A fresh bucket can contain pending input copies, page-table kernels,
        # and backend preparation on the scheduler's stream. Even its first
        # eager warmup forward must wait for all of those writes.
        assert capture_entry is entry
        runner._stream.wait_stream.assert_called_once_with(current_stream)
        return torch.ones(1)

    with (
        patch.object(torch, "npu", fake_npu, create=True),
        patch.object(runner, "_forward_static", side_effect=forward_static) as forward,
    ):
        runner._capture(entry)

    assert forward.call_count >= 2
    assert entry.graph is fake_npu.NPUGraph.return_value
    torch.testing.assert_close(entry.static_output, torch.ones(1))


def test_mtp_graph_output_slices_and_detaches_topk_buffer() -> None:
    hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    topk = torch.arange(24, dtype=torch.int64).reshape(4, 2, 3)

    sliced = DecodeAclGraphRunner._slice_output((hidden, None, topk), 2)

    assert isinstance(sliced, tuple)
    sliced_hidden, sliced_aux, sliced_topk = sliced
    assert sliced_aux is None
    assert torch.equal(sliced_hidden, hidden[:2])
    assert torch.equal(sliced_topk, topk[:2])
    assert sliced_topk.data_ptr() != topk.data_ptr()
    topk.zero_()
    assert torch.count_nonzero(sliced_topk) > 0


def test_mtp_graph_key_separates_first_step_and_topk_shapes() -> None:
    topk = torch.ones((4, 1, 8), dtype=torch.int32)
    key = DecodeAclGraphRunner._graph_key(8, False, None, topk)
    assert key != DecodeAclGraphRunner._graph_key(8, False, None)
    assert key == DecodeAclGraphRunner._graph_key(8, False, None, topk + 1)
    assert key != DecodeAclGraphRunner._graph_key(8, False, None, topk[:, :, :4])


def test_graph_aux_hidden_output_preserves_two_tensor_contract() -> None:
    hidden = torch.arange(8).reshape(4, 2)
    aux_hidden = hidden + 8
    result = DecodeAclGraphRunner._slice_output((hidden, aux_hidden), 2)
    assert len(result) == 2
    torch.testing.assert_close(result[0], hidden[:2])
    torch.testing.assert_close(result[1], aux_hidden[:2])


def test_mtp_topk_input_changes_without_reallocating_capture_buffer() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = input_ids.clone()
    metadata = _metadata(input_ids)
    topk = torch.arange(24, dtype=torch.int32).reshape(4, 2, 3)
    entry = runner._allocate_entry(8, input_ids, positions, metadata, topk)
    address = entry.static_mtp_topk_indices.data_ptr()

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        for source in (topk, topk.flip(0) + 7):
            runner._fill_entry(entry, input_ids, positions, metadata, 4, None, source)
            assert entry.static_mtp_topk_indices.data_ptr() == address
            torch.testing.assert_close(entry.static_mtp_topk_indices[:4], source)
            assert torch.count_nonzero(entry.static_mtp_topk_indices[4:]) == 0


def test_linear_state_indices_use_stable_graph_buffer() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 7, 11, 15], dtype=torch.int32))
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )
    static_indices = entry.static_metadata.linear_state_indices
    data_ptr = static_indices.data_ptr()

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )
        assert static_indices.tolist() == [3, 7, 11, 15, 0, 0, 0, 0]

        metadata.linear_state_indices = torch.tensor(
            [4, 8, 12, 16],
            dtype=torch.int32,
        )
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert static_indices.data_ptr() == data_ptr
    assert static_indices.tolist() == [4, 8, 12, 16, 0, 0, 0, 0]


def test_mla_repair_rows_do_not_expand_unrelated_linear_state_indices() -> None:
    runner = _runner()
    runner.attention_backend.is_mla = True
    metadata = _metadata(torch.tensor([3, 7], dtype=torch.int32))
    metadata.slot_mapping = metadata.slot_mapping[:3]
    metadata.block_table = metadata.block_table[:3]
    metadata.kv_seq_lens = metadata.kv_seq_lens[:3]
    metadata.kv_seq_lens_host_values = [1, 2, 3]
    metadata.kv_cu_seq_lens = metadata.kv_cu_seq_lens[:4]
    metadata.paged_kv_indptr = metadata.paged_kv_indptr[:4]
    metadata.paged_kv_indices = metadata.paged_kv_indices[:3]
    metadata.paged_kv_last_page_len = metadata.paged_kv_last_page_len[:3]
    ids = torch.arange(3, dtype=torch.int32)
    entry = runner._allocate_entry(4, ids, ids, metadata)

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        runner._fill_entry(entry, ids, ids, metadata, 3, None)

    # The scheduler's two recurrent-state IDs are irrelevant to MLA. The
    # three attention rows (repair + current, current) need no state remap.
    assert torch.count_nonzero(entry.static_metadata.linear_state_indices) == 0


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("decoding_tokens", [1, 2])
def test_dcp_graph_metadata_uses_logical_pages(expanded: bool, decoding_tokens: int) -> None:
    runner = _runner()
    runner.attention_backend.logical_page_size = 8
    runner.max_model_len = 16
    runner.num_decoding_tokens = decoding_tokens
    metadata = _metadata(torch.tensor([3, 7], dtype=torch.int32))
    metadata.slot_mapping = torch.tensor([87, 88], dtype=torch.int32)
    metadata.block_table = torch.tensor([[10, 11], [10, 11]], dtype=torch.int32)
    metadata.kv_seq_lens = torch.tensor([8, 9], dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [8, 9]
    # The unexpanded input still has sequence-scoped paged metadata. The
    # runner must rebuild it for the two token rows using logical pages.
    metadata.paged_kv_indptr = torch.tensor([0, 2], dtype=torch.int32)
    metadata.paged_kv_indices = torch.tensor([10, 11], dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
    if expanded:
        metadata.expanded_decode_metadata = SimpleNamespace(
            enabled=True,
            block_table=metadata.block_table,
            kv_seq_lens=metadata.kv_seq_lens,
            kv_seq_lens_host_values=[8, 9],
            kv_seq_lens_host=None,
            paged_kv_indptr=torch.tensor([0, 1, 3], dtype=torch.int32),
            paged_kv_indices=torch.tensor([10, 10, 11], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([8, 1], dtype=torch.int32),
            paged_attention_tiling_data=None,
        )

    _, _, _, indptr, indices, last_page_lens = runner._decode_metadata(metadata)
    assert indptr.tolist() == [0, 1, 3]
    assert indices.tolist() == [10, 10, 11]
    assert last_page_lens.tolist() == [8, 1]

    entry = runner._allocate_entry(
        padded_batch_size=2,
        input_ids=torch.tensor([42, 43], dtype=torch.int32),
        positions=torch.tensor([7, 8], dtype=torch.int32),
        metadata=metadata,
    )
    assert entry.static_metadata.block_table.shape == (2, 3)


def test_mtp_linear_state_indices_repeat_for_expanded_rows() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 7], dtype=torch.int32))
    metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
    metadata.block_table = torch.tensor([[10, 0], [20, 0], [30, 0], [40, 0]], dtype=torch.int32)
    metadata.kv_seq_lens = torch.arange(1, 5, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [1, 2, 3, 4]
    metadata.paged_kv_indptr = torch.arange(5, dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.arange(1, 5, dtype=torch.int32)
    metadata.expanded_decode_metadata = SimpleNamespace(
        enabled=True,
        kv_seq_lens=metadata.kv_seq_lens,
        block_table=metadata.block_table,
        paged_kv_indptr=metadata.paged_kv_indptr,
        paged_kv_indices=metadata.paged_kv_indices,
        paged_kv_last_page_len=metadata.paged_kv_last_page_len,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=metadata.kv_seq_lens_host_values,
    )
    entry = runner._allocate_entry(
        padded_batch_size=4,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert entry.static_metadata.linear_state_indices.tolist() == [3, 3, 7, 7]


def _dp_metadata(
    token_counts: tuple[int, int],
    dp_is_decode: tuple[int, int] = (1, 1),
) -> SimpleNamespace:
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        dp_execution_token_counts=tuple(1 if count == 0 else count for count in token_counts),
        dp_is_decode=dp_is_decode,
    )


def test_dp_empty_rank_uses_group_wide_acl_graph_bucket() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
    )

    with patch.object(
        runner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runner.can_execute(
            torch.zeros(1, dtype=torch.int32),
            _dp_metadata((5, 0)),
        )


def test_dp_mixed_step_does_not_enter_acl_decode_graph() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )

    with patch.object(
        runner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert not runner.can_execute(
            torch.zeros(3, dtype=torch.int32),
            _dp_metadata((3, 2), dp_is_decode=(0, 1)),
        )


def test_dp_acl_graph_requires_group_wide_token_counts() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )
    metadata = _dp_metadata((3, 2))
    metadata.dp_execution_token_counts = (3,)

    with (
        patch.object(
            runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="valid dp_execution_token_counts"),
    ):
        runner.can_execute(torch.zeros(3, dtype=torch.int32), metadata)
