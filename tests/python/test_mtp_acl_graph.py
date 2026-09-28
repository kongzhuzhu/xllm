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

from __future__ import annotations

import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.python.mtp_graph_test_utils import make_recipe, output_tensors, scalar_reference
from xllm.python import distributed
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
    get_execution_buffer,
)
from xllm.python.model_executor.runners.mtp_acl_graph import (
    MtpAclGraphRunner,
    MtpGraphOutput,
    MtpGraphRecipe,
    MtpRoleAdapter,
    _pack_metadata_tensors,
)
from xllm.python.model_executor.runners.mtp_sampling import (
    MtpSamplingPlan,
    _gumbel_argmax,
    probabilistic_acceptance,
    sample_logits,
)
from xllm.python.model_executor.runners.mtp_sparse_metadata import MtpSparsePositionStorage


@pytest.mark.parametrize("steps", [1, 3, 5])
@pytest.mark.parametrize("prepared_positions", [False, True])
def test_eager_recipe_matches_scalar_oracle_for_each_fixed_k(steps: int, prepared_positions: bool) -> None:
    rejection_steps = torch.arange(steps + 1)
    storage = MtpSparsePositionStorage(steps + 1, steps, torch.device("cpu")) if prepared_positions else None
    recipe = make_recipe(rejection_steps, steps, position_storage=storage)
    activation_calls: list[str] = []
    recipe.draft_activate = lambda: activation_calls.append("draft")
    recipe.target_activate = lambda: activation_calls.append("target")
    prepare_calls: list[torch.Tensor] = []
    runner = MtpAclGraphRunner(recipe, backend="eager", prepare=lambda seed, positions, kv: prepare_calls.append(seed))
    seeds = [2 + row for row in range(steps + 1)]
    positions = [10 + row * 3 for row in range(steps + 1)]
    if storage is not None:
        storage.positions.copy_(
            torch.tensor(
                [position + step for step in range(steps) for position in positions]
                + [position + step for position in positions for step in range(steps + 1)]
            )
        )
    kv_lengths = [position + 1 for position in positions]
    expected = scalar_reference(seeds, positions, kv_lengths, rejection_steps.tolist(), steps)
    output = runner.execute(torch.tensor(seeds), torch.tensor(positions), torch.tensor(kv_lengths, dtype=torch.int32))
    for name, actual in output_tensors(output).items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
    assert len(prepare_calls) == 1
    assert activation_calls == ["draft"] * steps + ["target"]
    if storage is not None:
        # Replay must consume the prepared arena; the per-invocation base is
        # still used for the next-state contract, never to overwrite the arena.
        changed_base = torch.tensor(positions) + 100
        replay = runner.execute(torch.tensor(seeds), changed_base, torch.tensor(kv_lengths, dtype=torch.int32))
        torch.testing.assert_close(replay.committed_tokens, expected["committed_tokens"], rtol=0, atol=0)
        torch.testing.assert_close(replay.next_state.positions, expected["next_positions"] + 100, rtol=0, atol=0)


def test_mtp_graph_output_clone_detaches_every_replay_tensor() -> None:
    values = {
        "accepted_ids": torch.tensor([[1, -1]]),
        "accepted_mask": torch.tensor([[True, False]]),
        "accepted_count": torch.tensor([1], dtype=torch.int32),
        "next_tokens": torch.tensor([2]),
        "committed_tokens": torch.tensor([[1, 2]]),
        "draft_tokens": torch.tensor([[1]]),
        "target_tokens": torch.tensor([[1, 2]]),
        "next_positions": torch.tensor([3]),
        "next_kv_seq_lens": torch.tensor([4], dtype=torch.int32),
        "next_embeddings": torch.tensor([[5.0]]),
        "next_topk_indices": torch.tensor([[6]]),
        "target_embeddings": torch.tensor([[[7.0], [8.0]]]),
        "target_probs": torch.tensor([[[0.2, 0.8], [0.3, 0.7]]]),
        "committed_log_probs": torch.tensor([[-0.2, -0.3]]),
        "target_top_log_probs": torch.tensor([[[-0.2], [-0.3]]]),
        "target_top_tokens": torch.tensor([[[1], [2]]]),
    }
    output = MtpGraphOutput(**values)
    cloned = MtpAclGraphRunner._clone_graph_output(output)

    for name, value in values.items():
        cloned_value = getattr(cloned, name)
        assert cloned_value is not value
        assert torch.equal(cloned_value, value)

    values["committed_tokens"].fill_(-99)
    values["target_embeddings"].fill_(-99)
    assert cloned.committed_tokens.tolist() == [[1, 2]]
    assert cloned.target_embeddings.tolist() == [[[7.0], [8.0]]]


@pytest.mark.parametrize("with_lengths", [False, True])
def test_penalty_padding_neither_penalizes_nor_overwrites_token_zero(with_lengths: bool) -> None:
    logits = torch.tensor([[10.0, 6.0, 1.0], [10.0, 6.0, 1.0]])
    plan = MtpSamplingPlan(
        batch_size=2,
        repetition_penalties=torch.tensor([2.0, 2.0]),
        unique_token_ids=torch.tensor([[2, 0, 0], [0, 0, 0]]),
        unique_token_counts=torch.tensor([[1, 0, 0], [1, 0, 0]]),
        unique_token_ids_lens=torch.tensor([1, 1]) if with_lengths else None,
        return_probs=False,
        logprobs=True,
    )
    owned = plan.clone_for_graph()
    expected = torch.tensor([[10.0, 6.0, 0.5], [5.0, 6.0, 1.0]])
    sampled = sample_logits(logits, owned)
    assert sampled.tokens.tolist() == [0, 1]
    torch.testing.assert_close(sampled.log_probs, expected.log_softmax(-1))
    infinite_logits = logits.clone()
    infinite_logits[0, 2] = float("-inf")
    assert sample_logits(infinite_logits, owned).tokens.tolist() == [0, 1]
    expanded = owned.expand_for_rows(4)
    assert sample_logits(logits.repeat_interleave(2, 0), expanded).tokens.tolist() == [0, 0, 1, 1]
    if with_lengths:
        plan.unique_token_ids_lens.zero_()
        owned.update_from(plan)
        assert sample_logits(logits, owned).tokens.tolist() == [0, 0]


def test_sampling_plan_applies_temperature_and_top_k() -> None:
    plan = MtpSamplingPlan(
        batch_size=2,
        do_sample=torch.zeros(2, dtype=torch.bool),
        temperatures=torch.ones(2),
        top_k=torch.ones(2, dtype=torch.long),
        all_greedy_sample=True,
        return_probs=True,
    )
    logits = torch.tensor([[1.0, 4.0, 3.0], [5.0, 2.0, 4.0]])
    sampled = sample_logits(logits, plan)
    assert sampled.tokens.tolist() == [1, 0]
    assert torch.equal(sampled.probs.argmax(dim=-1), sampled.tokens)


def test_sampling_plan_matches_unlimited_top_k_and_bitmask_contracts() -> None:
    logits = torch.tensor([[1.0, 4.0, 3.0], [1.0, 4.0, 3.0]])
    plan = MtpSamplingPlan(
        batch_size=2,
        do_sample=torch.zeros(2, dtype=torch.bool),
        top_k=torch.tensor([0, 1], dtype=torch.long),
        filter_bitmask=torch.tensor([[0b010], [0b111]], dtype=torch.int64),
        all_greedy_sample=True,
        return_probs=True,
    )
    sampled = sample_logits(logits, plan)
    # top_k=0 is unlimited; the bitmask leaves token 1 as the only option.
    assert sampled.tokens.tolist() == [1, 1]
    assert sampled.probs.shape == logits.shape


def test_sampling_plan_mixed_mode_uses_request_do_sample() -> None:
    logits = torch.tensor([[1.0, 4.0, 3.0], [1.0, 4.0, 3.0]])
    plan = MtpSamplingPlan(
        batch_size=2,
        do_sample=torch.tensor([False, True]),
        top_k=torch.ones(2, dtype=torch.long),
        all_greedy_sample=False,
        all_random_sample=False,
    )
    sampled = sample_logits(logits, plan)
    assert sampled.tokens[0].item() == 1
    assert sampled.tokens[1].item() == 1


def test_random_sampling_runs_tp_consensus_before_target_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = MtpSamplingPlan(
        batch_size=1,
        do_sample=torch.ones(1, dtype=torch.bool),
        all_random_sample=True,
        all_greedy_sample=False,
    )
    calls: list[tuple[torch.Tensor, int, str]] = []

    def broadcast(value: torch.Tensor, src: int, group_name: str) -> None:
        calls.append((value.clone(), src, group_name))
        value.fill_(2)

    monkeypatch.setattr(distributed, "tp_world_size", lambda _device: 2)
    monkeypatch.setattr(distributed, "broadcast_", broadcast)
    sampled = sample_logits(
        torch.tensor([[0.0, 1.0, 2.0]]),
        plan,
        uniform=torch.full((1, 3), 0.5),
    )

    assert sampled.tokens.tolist() == [2]
    assert len(calls) == 1
    assert calls[0][1:] == (0, "tp")


def test_gumbel_max_matches_categorical_distribution() -> None:
    """The ACL-safe sampler must preserve multinomial probabilities."""
    sample_count = 32768
    logits = torch.log(torch.tensor([[1.0, 3.0, 6.0]])).expand(sample_count, -1)
    generator = torch.Generator().manual_seed(20260922)
    uniform = torch.rand((sample_count, logits.shape[-1]), generator=generator)

    sampled = _gumbel_argmax(torch.log_softmax(logits, dim=-1), uniform)
    frequencies = torch.bincount(sampled, minlength=logits.shape[-1]).to(torch.float32) / sample_count
    expected = torch.softmax(logits[0], dim=-1)
    torch.testing.assert_close(frequencies, expected, rtol=0, atol=0.015)


def test_probabilistic_acceptance_explicit_uniforms_force_residual_recovery() -> None:
    draft_tokens = torch.tensor([[1, 2]], dtype=torch.long)
    target_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    draft_probs = torch.tensor([[[0.0, 0.8, 0.2], [0.2, 0.1, 0.7]]])
    target_probs = torch.tensor([[[0.6, 0.2, 0.2], [0.2, 0.1, 0.7], [0.2, 0.3, 0.5]]])
    accepted_ids, accepted_mask, accepted_count, next_tokens = probabilistic_acceptance(
        draft_tokens,
        draft_probs,
        target_tokens,
        target_probs,
        torch.ones(1, dtype=torch.bool),
        acceptance_uniform=torch.tensor([[0.9, 0.0]]),
        recovery_uniform=torch.full((1, 2, 3), 0.5),
    )
    assert accepted_ids.tolist() == [[-1, -1]]
    assert accepted_mask.tolist() == [[False, False]]
    assert accepted_count.tolist() == [0]
    assert next_tokens.tolist() == [0]


def test_probabilistic_acceptance_accepts_equal_proposals() -> None:
    draft_tokens = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
    target_tokens = torch.tensor([[1, 2, 0], [2, 1, 3]], dtype=torch.long)
    draft_probs = torch.tensor([[[0.1, 0.7, 0.2], [0.2, 0.1, 0.7]], [[0.2, 0.1, 0.7], [0.1, 0.7, 0.2]]])
    target_probs = torch.cat((draft_probs, torch.tensor([[[0.2, 0.3, 0.5]], [[0.1, 0.2, 0.7]]])), dim=1)
    accepted_ids, accepted_mask, accepted_count, next_tokens = probabilistic_acceptance(
        draft_tokens,
        draft_probs,
        target_tokens,
        target_probs,
        torch.ones(2, dtype=torch.bool),
    )
    assert accepted_ids.tolist() == [[1, 2], [2, 1]]
    assert accepted_mask.tolist() == [[True, True], [True, True]]
    assert accepted_count.tolist() == [2, 2]
    assert next_tokens.tolist() == [0, 3]


def test_greedy_target_keeps_equality_acceptance_with_random_draft() -> None:
    batch_size = 1
    speculative_tokens = 2
    vocab_size = 8

    def body(
        ids: torch.Tensor,
        positions: torch.Tensor,
        step: int,
        input_embedding: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        del positions, step, input_embedding, topk_indices
        return ids.to(torch.float32).unsqueeze(-1)

    def head(hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((hidden.shape[0], vocab_size))
        token_ids = hidden.squeeze(-1).to(torch.long).remainder(vocab_size)
        return logits.scatter(1, token_ids.unsqueeze(-1), 3.0)

    draft_plan = MtpSamplingPlan(
        batch_size=batch_size,
        top_k=torch.ones(batch_size, dtype=torch.long),
        do_sample=torch.ones(batch_size, dtype=torch.bool),
        all_random_sample=True,
        all_greedy_sample=False,
        return_probs=True,
    )
    target_plan = MtpSamplingPlan(
        batch_size=batch_size,
        do_sample=torch.zeros(batch_size, dtype=torch.bool),
        all_random_sample=False,
        all_greedy_sample=True,
        return_probs=True,
    )
    recipe = MtpGraphRecipe(
        body,
        head,
        body,
        head,
        batch_size=batch_size,
        speculative_tokens=speculative_tokens,
        vocab_size=vocab_size,
        device=torch.device("cpu"),
        draft_sampling=draft_plan,
        target_sampling=target_plan,
    )
    output = MtpAclGraphRunner(recipe, backend="eager").execute(torch.tensor([1]), torch.tensor([10]))
    assert output.accepted_count.tolist() == [speculative_tokens]
    assert output.next_state.token_ids.numel() == batch_size


def test_eager_recipe_carries_mtp_embedding_and_topk_state() -> None:
    batch_size = 2
    speculative_tokens = 3
    seen: list[tuple[int, torch.Tensor | None, torch.Tensor | None]] = []

    def draft_body(
        ids: torch.Tensor,
        positions: torch.Tensor,
        step: int,
        input_embedding: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        seen.append(
            (
                step,
                None if input_embedding is None else input_embedding.clone(),
                None if topk_indices is None else topk_indices.clone(),
            )
        )
        hidden = (ids + positions + step + 1).unsqueeze(-1)
        next_topk = (ids + step + 11).unsqueeze(-1)
        return hidden, None, next_topk

    def target_body(
        ids: torch.Tensor,
        positions: torch.Tensor,
        step: int,
        input_embedding: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        del positions, step, input_embedding, topk_indices
        hidden = ids.unsqueeze(-1)
        topk = (ids + 100).unsqueeze(-1)
        return hidden, None, topk

    def head(hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.full((hidden.shape[0], 64), -100.0)
        return logits.scatter_(1, hidden.remainder(64).to(torch.long), 1.0)

    runner = MtpAclGraphRunner(
        MtpGraphRecipe(
            draft_body,
            head,
            target_body,
            head,
            batch_size=batch_size,
            speculative_tokens=speculative_tokens,
            vocab_size=64,
            device=torch.device("cpu"),
        ),
        backend="eager",
    )
    initial_embedding = torch.tensor([[500.0], [600.0]])
    initial_topk = torch.tensor([[7], [8]], dtype=torch.long)
    output = runner.execute(
        torch.tensor([1, 2]),
        torch.tensor([10, 20]),
        draft_input_embedding=initial_embedding,
        draft_topk_indices=initial_topk,
    )

    assert [step for step, _, _ in seen] == [0, 1, 2]
    assert torch.equal(seen[0][1], initial_embedding)
    assert torch.equal(seen[0][2], initial_topk)
    assert torch.equal(seen[1][1], torch.tensor([[12], [23]]))
    assert torch.equal(seen[1][2], torch.tensor([[12], [13]]))
    assert torch.equal(seen[2][1], torch.tensor([[25], [46]]))
    assert torch.equal(seen[2][2], torch.tensor([[24], [35]]))
    assert output.next_state.embeddings is not None
    assert output.next_state.topk_indices is not None


def test_role_adapter_keeps_draft_and_target_metadata_scoped() -> None:
    calls: list[tuple[str, int, object, torch.Tensor | None, torch.Tensor | None]] = []

    class FakeRunner:
        def __init__(self, role: str) -> None:
            self.role = role

        def execute(
            self,
            ids: torch.Tensor,
            positions: torch.Tensor,
            metadata: object,
            embedding: torch.Tensor | None,
            synchronizer: object,
            topk: torch.Tensor | None,
        ) -> torch.Tensor:
            del synchronizer
            calls.append((self.role, int(ids.numel()), metadata, embedding, topk))
            return ids.unsqueeze(-1)

    class FakeExecutor:
        def __init__(self, role: str) -> None:
            self.eager_runner = FakeRunner(role)

    draft = MtpRoleAdapter(FakeExecutor("draft"), ("d0", "d1", "d2"), speculative_tokens=3)
    target = MtpRoleAdapter(FakeExecutor("target"), ("target-k+1",), speculative_tokens=3, target=True)
    ids = torch.tensor([1, 2])
    positions = torch.tensor([10, 20])
    embedding = torch.ones(2, 1)
    topk = torch.ones(2, 1, dtype=torch.long)

    draft(ids, positions, 1, embedding, topk)
    target(torch.arange(8), torch.arange(8), -1, None, None)

    assert calls[0][0:3] == ("draft", 2, "d1")
    assert calls[1][0:3] == ("target", 8, "target-k+1")


def test_serial_mtp_attention_workspace_uses_role_shared_pool() -> None:
    shared: dict[tuple[object, ...], object] = {}
    first_state = AclGraphExecutionState({}, shared_persistent_buffers=shared)
    second_state = AclGraphExecutionState({}, shared_persistent_buffers=shared)
    device = torch.device("cpu")

    class Backend:
        pass

    metadata = object()
    context = ForwardContext(Backend(), device, metadata, [], execution_state=first_state)
    with forward_context(context):
        first = get_execution_buffer(("FIA_WORKSPACE", 2), lambda: torch.zeros(4), shared=True)
    context = ForwardContext(Backend(), device, metadata, [], execution_state=second_state)
    with forward_context(context):
        second = get_execution_buffer(("FIA_WORKSPACE", 2), lambda: torch.ones(4), shared=True)

    assert first.data_ptr() == second.data_ptr()
    assert first_state.persistent_buffers == {}
    assert second_state.persistent_buffers == {}


@pytest.mark.parametrize("source_dtype", [torch.int32, torch.int64])
def test_role_adapter_selects_current_rows_from_first_draft_repair_layout(source_dtype: torch.dtype) -> None:
    seen: list[tuple[torch.Tensor, torch.Tensor, object, torch.Tensor]] = []

    class FakeRunner:
        def execute(
            self,
            ids: torch.Tensor,
            positions: torch.Tensor,
            metadata: object,
            embedding: torch.Tensor | None,
            synchronizer: object,
            topk: torch.Tensor | None,
        ) -> tuple[torch.Tensor, None, torch.Tensor]:
            del synchronizer, topk
            assert embedding is not None
            seen.append((ids, positions, metadata, embedding))
            return ids.unsqueeze(-1), None, (ids + 100).unsqueeze(-1)

    class FakeExecutor:
        eager_runner = FakeRunner()

    prepared_positions = torch.zeros(4, dtype=torch.long) if source_dtype == torch.int32 else None
    adapter = MtpRoleAdapter(
        FakeExecutor(),
        ("d0", "d1"),
        speculative_tokens=2,
        repair_token_ids=torch.tensor([[4, -1], [5, -1]], dtype=source_dtype)[:, 0],
        repair_positions=prepared_positions,
    )
    if prepared_positions is not None:
        prepared_positions.copy_(torch.tensor([19, 20, 29, 30]))
    output = adapter(
        torch.tensor([9, 10]),
        torch.tensor([20, 30]),
        0,
        torch.arange(4, dtype=torch.float32).reshape(4, 1),
        None,
    )

    assert seen[0][0].tolist() == [4, 9, 5, 10]
    assert seen[0][0].dtype == torch.long
    assert seen[0][1].tolist() == [19, 20, 29, 30]
    assert seen[0][2] == "d0"
    assert seen[0][3].shape == (4, 1)
    assert isinstance(output, tuple)
    assert output[0].flatten().tolist() == [9, 10]


@pytest.mark.parametrize("steps", [3, 5])
def test_metadata_arena_retains_distinct_step_values_and_stable_addresses(steps: int) -> None:
    metadata = [
        SimpleNamespace(
            kv_seq_lens=torch.tensor([7 + step, 15 + step], dtype=torch.int32),
            block_table=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            expanded_decode_metadata=SimpleNamespace(
                kv_seq_lens=torch.tensor([8 + step, 16 + step], dtype=torch.int32)
            ),
        )
        for step in range(steps)
    ]
    originals = [item.kv_seq_lens.clone() for item in metadata]
    retained = [item.kv_seq_lens for item in metadata]
    arenas = _pack_metadata_tensors(metadata)
    assert len(arenas) == 1
    pointers = [item.kv_seq_lens.data_ptr() for item in metadata]
    assert len(set(pointers)) == steps
    for step, item in enumerate(metadata):
        assert item.kv_seq_lens is retained[step]
        assert item.kv_seq_lens.untyped_storage().data_ptr() == arenas[0].data_ptr()
        torch.testing.assert_close(item.kv_seq_lens, originals[step])
        item.kv_seq_lens.copy_(torch.tensor([128 + step, 256 + step], dtype=torch.int32))
    for step, item in enumerate(metadata):
        assert item.kv_seq_lens.data_ptr() == pointers[step]
        assert item.kv_seq_lens.tolist() == [128 + step, 256 + step]
        assert item.expanded_decode_metadata.kv_seq_lens.tolist() == [8 + step, 16 + step]
        assert item.block_table.tolist() == [[1, 2], [3, 4]]


def test_metadata_arena_preserves_aliases_and_separates_dtypes() -> None:
    lengths = torch.tensor([3, 7], dtype=torch.int32)
    metadata = SimpleNamespace(kv_seq_lens=lengths, slot_mapping=lengths, block_table=torch.tensor([[1, 2]]))
    arenas = _pack_metadata_tensors((metadata,))
    assert len(arenas) == 2
    assert metadata.kv_seq_lens.data_ptr() == metadata.slot_mapping.data_ptr()
    torch.testing.assert_close(lengths, torch.tensor([3, 7], dtype=torch.int32))


def test_greedy_without_probability_outputs_skips_vocab_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0], [9.0, 5.0, 7.0]])
    controls = dict(temperatures=torch.tensor([0.0, 0.7]), top_k=torch.tensor([1, 2]), top_p=torch.tensor([0.9, 1.0]))
    reference = sample_logits(logits, MtpSamplingPlan(batch_size=2, return_probs=True, **controls))

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("greedy token-only sampling performed vocabulary normalization/sort")

    monkeypatch.setattr(torch, "log_softmax", unexpected)
    monkeypatch.setattr(torch, "sort", unexpected)
    sampled = sample_logits(logits, MtpSamplingPlan(batch_size=2, return_probs=False, **controls))
    torch.testing.assert_close(sampled.tokens, reference.tokens)
    assert sampled.probs is None and sampled.log_probs is None
    half_logits = logits.to(torch.bfloat16)
    monkeypatch.setattr(torch.Tensor, "to", unexpected)
    assert sample_logits(half_logits, MtpSamplingPlan(batch_size=2, return_probs=False)).tokens.tolist() == [1, 0]


@pytest.mark.parametrize("logprobs,top_width", [(True, 0), (False, 2)])
def test_greedy_probability_outputs_remain_available(logprobs: bool, top_width: int) -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    plan = MtpSamplingPlan(batch_size=1, return_probs=False, logprobs=logprobs, max_top_logprobs=top_width)
    sampled = sample_logits(logits, plan)
    assert sampled.probs is None
    torch.testing.assert_close(sampled.log_probs, torch.log_softmax(logits, -1))
    assert sampled.tokens.item() == 1


@pytest.mark.parametrize("steps", [3, 5])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("logprobs,top_width", [(False, 0), (True, 0), (True, 2), (False, 2)])
def test_greedy_recipe_fast_path_matches_full_probability_recipe(
    steps: int, logits_dtype: torch.dtype, logprobs: bool, top_width: int
) -> None:
    rejects = torch.arange(steps + 1)
    fast = make_recipe(rejects, steps, logits_dtype=logits_dtype)
    reference = make_recipe(rejects, steps, logits_dtype=logits_dtype)
    for recipe, probabilities in ((fast, False), (reference, True)):
        recipe.draft_sampling = MtpSamplingPlan(batch_size=steps + 1, return_probs=probabilities)
        recipe.target_sampling = MtpSamplingPlan(
            batch_size=steps + 1,
            return_probs=probabilities,
            logprobs=logprobs,
            max_top_logprobs=top_width,
        )
    inputs = (torch.arange(steps + 1), torch.arange(steps + 1) + 127, torch.arange(steps + 1, dtype=torch.int32) + 128)
    expected = MtpAclGraphRunner(reference, backend="eager").execute(*inputs)
    actual = MtpAclGraphRunner(fast, backend="eager").execute(*inputs)
    for name, tensor in output_tensors(actual).items():
        torch.testing.assert_close(tensor, output_tensors(expected)[name])
    for name in ("logprobs", "top_logprobs", "top_tokens"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    assert actual.target_probs is None
    arena_address = fast._position_arena.data_ptr()
    MtpAclGraphRunner(fast, backend="eager").execute(inputs[0], inputs[1] + 4, inputs[2] + 4)
    assert fast._position_arena.data_ptr() == arena_address


def test_draft_temporaries_live_only_as_long_as_their_consumers(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_recipe(torch.tensor([1]), 3)
    recipe.draft_sampling = MtpSamplingPlan(batch_size=1, return_probs=True, logprobs=True, max_top_logprobs=3)
    recipe.target_sampling = MtpSamplingPlan(batch_size=1, return_probs=False)
    draft_forward, draft_head, target_forward = recipe.draft_forward, recipe.draft_logits, recipe.target_forward
    logits_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    hidden_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def draft(*args: object) -> torch.Tensor:
        hidden = draft_forward(*args)
        hidden_refs.append(weakref.ref(hidden))
        return hidden

    def head(hidden: torch.Tensor) -> torch.Tensor:
        logits = draft_head(hidden)
        logits_refs.append(weakref.ref(logits))
        return logits

    def target(*args: object) -> torch.Tensor:
        assert len(logits_refs) == len(hidden_refs) == 3
        assert all(value() is None for value in logits_refs + hidden_refs)
        return target_forward(*args)

    recipe.draft_forward, recipe.draft_logits, recipe.target_forward = draft, head, target
    normalize = Mock(side_effect=AssertionError("greedy draft has no probability/logprob consumer"))
    monkeypatch.setattr(torch, "log_softmax", normalize)
    actual = MtpAclGraphRunner(recipe, backend="eager").execute(
        torch.tensor([2]), torch.tensor([10]), torch.tensor([11], dtype=torch.int32)
    )
    expected = scalar_reference([2], [10], [11], [1], 3)
    for name, value in output_tensors(actual).items():
        torch.testing.assert_close(value, expected[name])


def test_runner_close_waits_for_replay_and_output_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    recipe = make_recipe(torch.tensor([1]), 3)
    runner = MtpAclGraphRunner(recipe)
    runner._capture_stream = SimpleNamespace(synchronize=lambda: order.append("replay-complete"))
    runner._graph = SimpleNamespace(reset=lambda: order.append("graph-freed"))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda device: SimpleNamespace(synchronize=lambda: order.append("snapshots-complete"))
        ),
        raising=False,
    )
    runner.close()
    runner.close()
    assert order == ["replay-complete", "snapshots-complete", "graph-freed"]
    inputs = (torch.tensor([2]), torch.tensor([10]), torch.tensor([11], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="closed"):
        runner.execute(*inputs)
    with pytest.raises(RuntimeError, match="closed"):
        runner.capture(*inputs)


@pytest.mark.parametrize("steps", [3, 5])
@pytest.mark.parametrize("logprobs", [False, True])
def test_runtime_output_uses_only_consumed_snapshots_and_greedy_heads(steps: int, logprobs: bool) -> None:
    from dataclasses import fields

    from xllm.python.model_executor.runners.base import SpeculativeRuntimeOutput

    rejects = torch.arange(steps + 1)
    fast, reference = make_recipe(rejects, steps), make_recipe(rejects, steps)
    for recipe in (fast, reference):
        recipe.draft_sampling = MtpSamplingPlan(batch_size=steps + 1, return_probs=False)
        recipe.target_sampling = MtpSamplingPlan(batch_size=steps + 1, return_probs=False, logprobs=logprobs)
    fast.runtime_outputs_only = True
    draft_head, target_head = fast.draft_logits, fast.target_logits
    fast.draft_greedy = lambda hidden: draft_head(hidden).argmax(-1)
    fast.target_greedy = lambda hidden: target_head(hidden).argmax(-1)
    fast.draft_logits = Mock(side_effect=AssertionError("greedy draft must not gather full logits"))
    if not logprobs:
        fast.target_logits = Mock(side_effect=AssertionError("plain greedy target must not gather full logits"))
    inputs = (
        torch.arange(steps + 1),
        torch.arange(steps + 1, dtype=torch.int32) + 7,
        torch.arange(steps + 1, dtype=torch.int32) + 8,
    )
    actual = MtpAclGraphRunner(fast, backend="eager").execute(*inputs)
    expected = MtpAclGraphRunner(reference, backend="eager").execute(*inputs)
    assert isinstance(actual, SpeculativeRuntimeOutput)
    assert not hasattr(actual, "next_state")
    expected_compact = torch.stack(
        [
            expected.target_embeddings[row, index]
            for row, count in enumerate(expected.accepted_count.tolist())
            for index in (max(count - 1, 0), count)
        ]
    )
    torch.testing.assert_close(actual.target_embeddings, expected_compact)
    for field in fields(actual):
        if field.name not in ("token_state", "target_embeddings"):
            torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name))
    assert actual.token_state is not None
    assert actual.committed_tokens.untyped_storage().data_ptr() == actual.token_state.data_ptr()
    assert actual.accepted_count.untyped_storage().data_ptr() == actual.token_state.data_ptr()
    assert actual.token_state.numel() == (steps + 1) * ((steps + 1) * 8 + 4)
    snapshot = MtpAclGraphRunner._clone_graph_output(actual)
    with torch.inference_mode():
        actual.committed_tokens.fill_(-99)
        actual.target_embeddings.fill_(-99)
    torch.testing.assert_close(snapshot.committed_tokens, expected.committed_tokens)
    torch.testing.assert_close(snapshot.accepted_count, expected.accepted_count)
    assert snapshot.token_state.data_ptr() != actual.token_state.data_ptr()
    assert snapshot.committed_tokens.untyped_storage().data_ptr() == snapshot.token_state.data_ptr()
    assert snapshot.accepted_count.untyped_storage().data_ptr() == snapshot.token_state.data_ptr()
    torch.testing.assert_close(snapshot.target_embeddings, expected_compact)
