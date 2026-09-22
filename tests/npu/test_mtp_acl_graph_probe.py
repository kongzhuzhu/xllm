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

import pytest
import torch

from tests.python.mtp_graph_test_utils import make_recipe, output_tensors, scalar_reference
from xllm.python.model_executor.runners.mtp_acl_graph import MtpAclGraphRunner, MtpGraphRecipe
from xllm.python.model_executor.runners.mtp_sampling import (
    MtpSamplingPlan,
    MtpSamplingRandomInputs,
)


@pytest.fixture(scope="module")
def npu_device() -> torch.device:
    device_index = os.environ.get("XLLM_TEST_NPU_DEVICE")
    if device_index is None:
        pytest.skip("set XLLM_TEST_NPU_DEVICE for the real MTP ACL graph probe")
    pytest.importorskip("torch_npu")
    torch.npu.set_device(int(device_index))
    return torch.device(f"npu:{device_index}")


@pytest.mark.parametrize("steps", [1, 2, 3, 4])
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
    assert runner.entry.capability.speculative_tokens == steps
    assert runner.entry.captured and runner.entry.generation == 1
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
    assert runner.entry.generation == 1


@pytest.mark.parametrize("mixed", [False, True])
def test_mtp_sampling_random_replay_stays_on_device(npu_device: torch.device, mixed: bool) -> None:
    """Random sampling and probability acceptance must be captured in one graph."""
    batch_size = 1
    speculative_tokens = 3
    vocab_size = 16
    plan = MtpSamplingPlan(
        batch_size=batch_size,
        do_sample=torch.ones(batch_size, dtype=torch.bool, device=npu_device),
        all_random_sample=not mixed,
        all_greedy_sample=False,
        return_probs=True,
    )

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
        token = hidden.squeeze(-1).to(torch.long).remainder(vocab_size)
        offsets = torch.arange(vocab_size, device=hidden.device, dtype=torch.float32)
        logits = -(offsets.unsqueeze(0) - token.unsqueeze(1)).abs()
        return logits

    recipe = MtpGraphRecipe(
        body,
        head,
        body,
        head,
        batch_size=batch_size,
        speculative_tokens=speculative_tokens,
        vocab_size=vocab_size,
        device=npu_device,
        kv_seq_lens=torch.zeros(batch_size, dtype=torch.int32, device=npu_device),
        draft_sampling=plan,
        target_sampling=plan,
    )
    runner = MtpAclGraphRunner(recipe)
    runner.capture(
        torch.tensor([1], device=npu_device),
        torch.tensor([10], device=npu_device),
        torch.tensor([10], dtype=torch.int32, device=npu_device),
    )
    first = runner.execute(
        torch.tensor([1], device=npu_device),
        torch.tensor([10], device=npu_device),
        torch.tensor([10], dtype=torch.int32, device=npu_device),
    )
    second = runner.execute(
        torch.tensor([2], device=npu_device),
        torch.tensor([11], device=npu_device),
        torch.tensor([11], dtype=torch.int32, device=npu_device),
    )
    for output in (first, second):
        assert output.accepted_count.device.type == "npu"
        assert output.next_state.token_ids.device.type == "npu"
        assert bool(output.accepted_count.ge(0).all().cpu())
        assert bool(output.accepted_count.le(speculative_tokens).all().cpu())
    assert runner._static_output is not None
    assert runner._static_output.target_probs is not None
    assert runner._static_output.target_probs.shape == (batch_size, speculative_tokens + 1, vocab_size)


def test_mtp_sampling_fixed_random_inputs_match_eager_oracle(npu_device: torch.device) -> None:
    """The ACL graph and eager recipe consume the same explicit Device draws."""
    batch_size = 1
    speculative_tokens = 3
    vocab_size = 16
    plan = MtpSamplingPlan(
        batch_size=batch_size,
        do_sample=torch.ones(batch_size, dtype=torch.bool, device=npu_device),
        all_random_sample=True,
        all_greedy_sample=False,
        return_probs=True,
        logprobs=True,
        max_top_logprobs=3,
    )
    random_inputs = MtpSamplingRandomInputs(
        draft_uniform=torch.linspace(
            0.05,
            0.95,
            batch_size * speculative_tokens * vocab_size,
            device=npu_device,
            dtype=torch.float32,
        ).reshape(batch_size, speculative_tokens, vocab_size),
        target_uniform=torch.linspace(
            0.95,
            0.05,
            batch_size * (speculative_tokens + 1) * vocab_size,
            device=npu_device,
            dtype=torch.float32,
        ).reshape(batch_size, speculative_tokens + 1, vocab_size),
        acceptance_uniform=torch.full((batch_size, speculative_tokens), 0.25, device=npu_device),
        recovery_uniform=torch.full((batch_size, speculative_tokens, vocab_size), 0.5, device=npu_device),
    )

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
        token = hidden.squeeze(-1).to(torch.long).remainder(vocab_size)
        offsets = torch.arange(vocab_size, device=hidden.device, dtype=torch.float32)
        return -(offsets.unsqueeze(0) - token.unsqueeze(1)).abs()

    def recipe() -> MtpGraphRecipe:
        return MtpGraphRecipe(
            body,
            head,
            body,
            head,
            batch_size=batch_size,
            speculative_tokens=speculative_tokens,
            vocab_size=vocab_size,
            device=npu_device,
            kv_seq_lens=torch.zeros(batch_size, dtype=torch.int32, device=npu_device),
            draft_sampling=plan,
            target_sampling=plan,
            sampling_random_inputs=random_inputs,
        )

    runner = MtpAclGraphRunner(recipe())
    eager = MtpAclGraphRunner(recipe(), backend="eager")
    inputs = (
        torch.tensor([1], device=npu_device),
        torch.tensor([10], device=npu_device),
        torch.tensor([10], dtype=torch.int32, device=npu_device),
    )
    runner.capture(*inputs)
    actual = runner.execute(*inputs)
    expected = eager.execute(*inputs)
    for actual_tensor, expected_tensor in (
        (actual.accepted_ids, expected.accepted_ids),
        (actual.accepted_count, expected.accepted_count),
        (actual.committed_tokens, expected.committed_tokens),
        (actual.next_state.token_ids, expected.next_state.token_ids),
        (actual.logprobs, expected.logprobs),
        (actual.top_tokens, expected.top_tokens),
        (actual.top_logprobs, expected.top_logprobs),
        (actual.target_probs, expected.target_probs),
    ):
        assert actual_tensor is not None and expected_tensor is not None
        torch.testing.assert_close(actual_tensor.cpu(), expected_tensor.cpu(), rtol=2e-4, atol=2e-4)


def test_mtp_mixed_greedy_random_batch_matches_eager_oracle(
    npu_device: torch.device,
) -> None:
    """A single captured graph must support greedy and random rows together."""
    batch_size = 2
    speculative_tokens = 3
    vocab_size = 16
    plan = MtpSamplingPlan(
        batch_size=batch_size,
        do_sample=torch.tensor([False, True], dtype=torch.bool, device=npu_device),
        all_random_sample=False,
        all_greedy_sample=False,
        return_probs=True,
        logprobs=True,
        max_top_logprobs=3,
    )
    random_inputs = MtpSamplingRandomInputs(
        draft_uniform=torch.linspace(
            0.05,
            0.95,
            batch_size * speculative_tokens * vocab_size,
            device=npu_device,
            dtype=torch.float32,
        ).reshape(batch_size, speculative_tokens, vocab_size),
        target_uniform=torch.linspace(
            0.95,
            0.05,
            batch_size * (speculative_tokens + 1) * vocab_size,
            device=npu_device,
            dtype=torch.float32,
        ).reshape(batch_size, speculative_tokens + 1, vocab_size),
        acceptance_uniform=torch.full((batch_size, speculative_tokens), 0.25, device=npu_device),
        recovery_uniform=torch.full((batch_size, speculative_tokens, vocab_size), 0.5, device=npu_device),
    )

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
        token = hidden.squeeze(-1).to(torch.long).remainder(vocab_size)
        offsets = torch.arange(vocab_size, device=hidden.device, dtype=torch.float32)
        return -(offsets.unsqueeze(0) - token.unsqueeze(1)).abs()

    def recipe() -> MtpGraphRecipe:
        return MtpGraphRecipe(
            body,
            head,
            body,
            head,
            batch_size=batch_size,
            speculative_tokens=speculative_tokens,
            vocab_size=vocab_size,
            device=npu_device,
            kv_seq_lens=torch.zeros(batch_size, dtype=torch.int32, device=npu_device),
            draft_sampling=plan,
            target_sampling=plan,
            sampling_random_inputs=random_inputs,
        )

    runner = MtpAclGraphRunner(recipe())
    eager = MtpAclGraphRunner(recipe(), backend="eager")
    inputs = (
        torch.tensor([1, 2], device=npu_device),
        torch.tensor([10, 20], device=npu_device),
        torch.tensor([10, 20], dtype=torch.int32, device=npu_device),
    )
    runner.capture(*inputs)
    actual = runner.execute(*inputs)
    expected = eager.execute(*inputs)
    for actual_tensor, expected_tensor in (
        (actual.accepted_ids, expected.accepted_ids),
        (actual.accepted_count, expected.accepted_count),
        (actual.committed_tokens, expected.committed_tokens),
        (actual.next_state.token_ids, expected.next_state.token_ids),
        (actual.logprobs, expected.logprobs),
        (actual.top_tokens, expected.top_tokens),
        (actual.top_logprobs, expected.top_logprobs),
        (actual.target_probs, expected.target_probs),
    ):
        assert actual_tensor is not None and expected_tensor is not None
        torch.testing.assert_close(actual_tensor.cpu(), expected_tensor.cpu(), rtol=2e-4, atol=2e-4)
