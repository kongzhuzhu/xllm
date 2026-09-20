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

"""Small deterministic adapters and an independent scalar oracle for MTP tests."""

from __future__ import annotations

import torch

from xllm.python.model_executor.runners.base import SpeculativeExecutionOutput
from xllm.python.model_executor.runners.mtp_acl_graph import MtpGraphRecipe

VOCAB_SIZE = 37


def make_recipe(rejection_steps: torch.Tensor, speculative_tokens: int) -> MtpGraphRecipe:
    """Inject one mismatch per row; later draft/target tokens match again."""
    batch_size = rejection_steps.numel()

    def draft_body(
        ids: torch.Tensor,
        positions: torch.Tensor,
        step: int,
        input_embedding: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        del step, input_embedding, topk_indices
        return (ids + positions + 1).remainder(VOCAB_SIZE).unsqueeze(-1)

    def target_body(
        ids: torch.Tensor,
        positions: torch.Tensor,
        step: int,
        input_embedding: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        del step, input_embedding, topk_indices
        next_ids = (ids + positions + 1).reshape(batch_size, speculative_tokens + 1)
        offsets = torch.arange(speculative_tokens + 1, device=ids.device)
        mismatch = offsets.unsqueeze(0).eq(rejection_steps.unsqueeze(1)) & offsets.lt(speculative_tokens)
        return (next_ids + mismatch.to(torch.long)).remainder(VOCAB_SIZE).reshape(-1, 1)

    def head(hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.full((hidden.shape[0], VOCAB_SIZE), -100.0, device=hidden.device)
        return logits.scatter_(1, hidden, 1.0)

    return MtpGraphRecipe(
        draft_body,
        head,
        target_body,
        head,
        batch_size=batch_size,
        speculative_tokens=speculative_tokens,
        vocab_size=VOCAB_SIZE,
        device=rejection_steps.device,
        kv_seq_lens=torch.zeros(batch_size, dtype=torch.int32, device=rejection_steps.device),
    )


def output_tensors(output: SpeculativeExecutionOutput) -> dict[str, torch.Tensor]:
    assert output.draft_tokens is not None
    assert output.target_tokens is not None
    assert output.next_state.kv_seq_lens is not None
    return {
        "draft_tokens": output.draft_tokens,
        "target_tokens": output.target_tokens,
        "accepted_ids": output.accepted_ids,
        "accepted_mask": output.accepted_mask,
        "accepted_count": output.accepted_count,
        "committed_tokens": output.committed_tokens,
        "next_tokens": output.next_state.token_ids,
        "next_positions": output.next_state.positions,
        "next_kv_seq_lens": output.next_state.kv_seq_lens,
    }


def scalar_reference(
    seeds: list[int],
    positions: list[int],
    kv_seq_lens: list[int],
    rejection_steps: list[int],
    speculative_tokens: int,
) -> dict[str, torch.Tensor]:
    """Compute expected results with Python lists, without the graph recipe."""
    rows: dict[str, list] = {
        key: []
        for key in (
            "draft_tokens",
            "target_tokens",
            "accepted_ids",
            "accepted_mask",
            "accepted_count",
            "committed_tokens",
            "next_tokens",
            "next_positions",
            "next_kv_seq_lens",
        )
    }
    for seed, position, kv_length, reject in zip(seeds, positions, kv_seq_lens, rejection_steps, strict=True):
        proposals: list[int] = []
        current = seed
        for step in range(speculative_tokens):
            current = (current + position + step + 1) % VOCAB_SIZE
            proposals.append(current)
        targets = [
            (token + position + step + 1 + int(step == reject and step < speculative_tokens)) % VOCAB_SIZE
            for step, token in enumerate([seed, *proposals])
        ]
        accepted = 0
        for proposal, target in zip(proposals, targets):
            if proposal != target:
                break
            accepted += 1
        rows["draft_tokens"].append(proposals)
        rows["target_tokens"].append(targets)
        rows["accepted_ids"].append(proposals[:accepted] + [-1] * (speculative_tokens - accepted))
        rows["accepted_mask"].append([True] * accepted + [False] * (speculative_tokens - accepted))
        rows["accepted_count"].append(accepted)
        rows["next_tokens"].append(targets[accepted])
        committed = proposals[:accepted] + [targets[accepted]]
        committed.extend([-1] * (speculative_tokens - accepted))
        rows["committed_tokens"].append(committed)
        # The replacement/bonus is the next input seed. Advance by the full
        # emitted length, matching C++ build_accepted_token_metadata().
        rows["next_positions"].append(position + accepted + 1)
        rows["next_kv_seq_lens"].append(kv_length + accepted + 1)
    dtypes = {"accepted_mask": torch.bool, "accepted_count": torch.int32, "next_kv_seq_lens": torch.int32}
    return {key: torch.tensor(value, dtype=dtypes.get(key, torch.long)) for key, value in rows.items()}
