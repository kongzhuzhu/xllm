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

import pytest
import torch

from tests.python.mtp_graph_test_utils import make_recipe, output_tensors, scalar_reference
from xllm.python.attention.backend import LayerCache
from xllm.python.model_executor.runners.mtp_acl_graph import (
    MtpAclGraphRunner,
    MtpGraphRecipe,
    MtpRoleAdapter,
    _committed_tokens,
    _normalize_request_major_rows,
    greedy_acceptance,
)
from xllm.python.model_executor.runners.mtp_kv_oracle import (
    PagedKvSnapshot,
    build_attention_read_slots,
)
from xllm.python.model_executor.runners.mtp_sampling import (
    MtpSamplingPlan,
    MtpSamplingRandomInputs,
    probabilistic_acceptance,
    sample_logits,
)


def test_paged_kv_snapshot_restores_selected_physical_payloads() -> None:
    key = torch.arange(2 * 4 * 1 * 3, dtype=torch.float32).reshape(2, 4, 1, 3)
    value = key + 100
    index = torch.arange(2 * 4 * 1 * 2, dtype=torch.float32).reshape(2, 4, 1, 2)
    scale = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4, 1)
    expected_key = key.clone()
    expected_value = value.clone()
    expected_index = index.clone()
    expected_scale = scale.clone()
    cache = LayerCache(key, value, index=index, indexer_scale=scale)
    slots = torch.tensor([-1, 0, 3, 5], dtype=torch.long)
    snapshot = PagedKvSnapshot.capture([("target", [cache], slots)])

    key.view(-1, 1, 3).index_fill_(0, torch.tensor([0, 3, 5]), -1)
    value.view(-1, 1, 3).index_fill_(0, torch.tensor([0, 3, 5]), -2)
    index.view(-1, 1, 2).index_fill_(0, torch.tensor([0, 3, 5]), -3)
    scale.view(-1, 1).index_fill_(0, torch.tensor([0, 3, 5]), -4)
    snapshot.restore()

    assert torch.equal(key, expected_key)
    assert torch.equal(value, expected_value)
    assert torch.equal(index, expected_index)
    assert torch.equal(scale, expected_scale)
    assert snapshot.tensor_count == 4


def test_attention_read_slots_follow_kv_lengths_and_page_table() -> None:
    block_table = torch.tensor([[4, 2, -1], [7, 1, -1]], dtype=torch.int32)
    read_slots = build_attention_read_slots(block_table, [5, 3], page_size=4)

    assert read_slots[0].tolist() == [16, 17, 18, 19, 8]
    assert read_slots[1].tolist() == [28, 29, 30]

    with pytest.raises(ValueError, match="invalid page"):
        build_attention_read_slots(torch.tensor([[4, -1]], dtype=torch.int32), [5], page_size=4)


def test_mtp_metadata_rows_normalize_sequence_and_step_major_layouts() -> None:
    sequence_major = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    step_major = torch.tensor([0, 3, 1, 4, 2, 5], dtype=torch.long)

    expected = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    assert torch.equal(
        _normalize_request_major_rows(sequence_major, 2, 3, step_major_layout=False),
        expected,
    )
    assert torch.equal(
        _normalize_request_major_rows(step_major, 2, 3, step_major_layout=True),
        expected,
    )


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5])
def test_greedy_acceptance_every_rejection_position(steps: int) -> None:
    draft = torch.arange(steps, dtype=torch.long).expand(steps + 1, -1).clone()
    target = torch.arange(steps + 1, dtype=torch.long).expand(steps + 1, -1).clone()
    for reject in range(steps):
        target[reject, reject] = 1000 + reject

    accepted_ids, accepted_mask, accepted_count, next_tokens = greedy_acceptance(draft, target)

    for reject in range(steps + 1):
        assert accepted_count[reject].item() == reject
        assert accepted_ids[reject].tolist() == list(range(reject)) + [-1] * (steps - reject)
        assert accepted_mask[reject].tolist() == [True] * reject + [False] * (steps - reject)
        assert next_tokens[reject].item() == (1000 + reject if reject < steps else steps)


def test_committed_tokens_put_replacement_at_first_rejection() -> None:
    # The scalar acceptance helper is the direct contract check; recipe tests
    # above cover model-produced rows.
    accepted_ids, _, accepted_count, next_token = greedy_acceptance(
        torch.tensor([[10, 11, 12]]), torch.tensor([[10, 99, 88, 77]])
    )
    committed = _committed_tokens(accepted_ids, accepted_count, next_token, 3)
    assert committed.tolist() == [[10, 99, -1, -1]]


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5])
def test_eager_recipe_matches_scalar_oracle_for_each_fixed_k(steps: int) -> None:
    rejection_steps = torch.arange(steps + 1)
    recipe = make_recipe(rejection_steps, steps)
    activation_calls: list[str] = []
    recipe.draft_activate = lambda: activation_calls.append("draft")
    recipe.target_activate = lambda: activation_calls.append("target")
    prepare_calls: list[torch.Tensor] = []
    runner = MtpAclGraphRunner(recipe, backend="eager", prepare=lambda seed, positions, kv: prepare_calls.append(seed))
    seeds = [2 + row for row in range(steps + 1)]
    positions = [10 + row * 3 for row in range(steps + 1)]
    kv_lengths = [position + 1 for position in positions]
    expected = scalar_reference(seeds, positions, kv_lengths, rejection_steps.tolist(), steps)
    output = runner.execute(torch.tensor(seeds), torch.tensor(positions), torch.tensor(kv_lengths, dtype=torch.int32))
    for name, actual in output_tensors(output).items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
    assert len(prepare_calls) == 1
    assert activation_calls == ["draft"] * steps + ["target"]


@pytest.mark.parametrize("steps", [0, -1])
def test_recipe_rejects_nonpositive_k(steps: int) -> None:
    with pytest.raises(ValueError, match="speculative_tokens"):
        make_recipe(torch.tensor([0]), steps)


def test_acceptance_rejects_zero_draft_width() -> None:
    with pytest.raises(ValueError, match="at least one draft"):
        greedy_acceptance(torch.empty(2, 0, dtype=torch.long), torch.zeros(2, 1, dtype=torch.long))


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


def test_sampling_explicit_uniforms_are_replay_deterministic() -> None:
    plan = MtpSamplingPlan(
        batch_size=1,
        do_sample=torch.ones(1, dtype=torch.bool),
        all_random_sample=True,
        all_greedy_sample=False,
    )
    logits = torch.zeros((1, 4))
    uniform = torch.tensor([[0.2, 0.4, 0.6, 0.8]], dtype=torch.float32)
    first = sample_logits(logits, plan, uniform=uniform)
    second = sample_logits(logits, plan, uniform=uniform)
    assert first.tokens.item() == second.tokens.item()
    torch.testing.assert_close(first.probs, second.probs, rtol=0, atol=0)


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


def test_sampling_random_inputs_validate_fixed_graph_layout() -> None:
    random_inputs = MtpSamplingRandomInputs(
        draft_uniform=torch.zeros((2, 3, 5)),
        target_uniform=torch.zeros((2, 4, 5)),
        acceptance_uniform=torch.zeros((2, 3)),
        recovery_uniform=torch.zeros((2, 3, 5)),
    )
    assert random_inputs.layout_signature()[0] == ((2, 3, 5), "torch.float32", "cpu")


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


def test_eager_recipe_runs_random_sampling_and_probability_acceptance() -> None:
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

    plan = MtpSamplingPlan(
        batch_size=batch_size,
        do_sample=torch.ones(batch_size, dtype=torch.bool),
        all_random_sample=True,
        all_greedy_sample=False,
        return_probs=True,
    )
    runner = MtpAclGraphRunner(
        MtpGraphRecipe(
            body,
            head,
            body,
            head,
            batch_size=batch_size,
            speculative_tokens=speculative_tokens,
            vocab_size=vocab_size,
            device=torch.device("cpu"),
            draft_sampling=plan,
            target_sampling=plan,
        ),
        backend="eager",
    )
    output = runner.execute(torch.tensor([1]), torch.tensor([10]))
    assert output.accepted_count.shape == (batch_size,)
    assert output.next_state.token_ids.shape == (batch_size,)
    assert output.next_state.token_ids.device.type == "cpu"


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


def test_role_adapter_selects_current_rows_from_first_draft_repair_layout() -> None:
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

    adapter = MtpRoleAdapter(
        FakeExecutor(),
        ("d0", "d1"),
        speculative_tokens=2,
        repair_token_ids=torch.tensor([4, 5]),
    )
    output = adapter(
        torch.tensor([9, 10]),
        torch.tensor([20, 30]),
        0,
        torch.arange(4, dtype=torch.float32).reshape(4, 1),
        None,
    )

    assert seen[0][0].tolist() == [4, 9, 5, 10]
    assert seen[0][1].tolist() == [19, 20, 29, 30]
    assert seen[0][2] == "d0"
    assert seen[0][3].shape == (4, 1)
    assert isinstance(output, tuple)
    assert output[0].flatten().tolist() == [9, 10]
