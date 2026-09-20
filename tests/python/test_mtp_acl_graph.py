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
from xllm.python.model_executor.runners.mtp_acl_graph import (
    MtpAclGraphRunner,
    MtpGraphRecipe,
    MtpRoleAdapter,
    _committed_tokens,
    greedy_acceptance,
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
