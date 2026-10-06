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

from xllm.python.model_executor.runners.base import SpeculativeRuntimeOutput
from xllm.python.model_executor.runners.mtp_acl_graph import MtpGraphRecipe
from xllm.python.model_executor.runners.mtp_sparse_metadata import MtpSparsePositionStorage

VOCAB_SIZE = 37


def make_recipe(
    rejection_steps: torch.Tensor,
    speculative_tokens: int,
    *,
    logits_dtype: torch.dtype = torch.float32,
    position_storage: MtpSparsePositionStorage | None = None,
) -> MtpGraphRecipe:
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
        return (ids + positions + 1).remainder(VOCAB_SIZE).unsqueeze(-1).float()

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
        return (next_ids + mismatch.to(torch.long)).remainder(VOCAB_SIZE).reshape(-1, 1).float()

    def head(hidden: torch.Tensor) -> torch.Tensor:
        # Distinct scores keep top-k deterministic and logprobs nontrivial;
        # the maximum still selects exactly the scalar oracle's hidden ID.
        offsets = torch.arange(VOCAB_SIZE, device=hidden.device)
        distances = (offsets.unsqueeze(0) - hidden).remainder(VOCAB_SIZE)
        return (-distances.to(torch.float32) / 16).to(logits_dtype)

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
        position_storage=position_storage,
    )


def output_tensors(output: SpeculativeRuntimeOutput) -> dict[str, torch.Tensor]:
    return {
        "accepted_count": output.accepted_count,
        "committed_tokens": output.committed_tokens,
        "target_embeddings": output.target_embeddings,
    }


def commit_reference(
    draft: torch.Tensor, target: torch.Tensor, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU-only scalar oracle for the native commit contract."""
    assert draft.device.type == target.device.type == hidden.device.type == "cpu"
    batch, steps = draft.shape
    tokens = torch.full_like(target, -1)
    counts = torch.zeros(batch, dtype=torch.int32)
    selected = []
    for row in range(batch):
        count = 0
        while count < steps and draft[row, count] == target[row, count]:
            count += 1
        counts[row] = count
        tokens[row, : count + 1] = target[row, : count + 1]
        selected.extend((row * (steps + 1) + max(count - 1, 0), row * (steps + 1) + count))
    state = torch.cat((tokens.flatten().view(torch.uint8), counts.view(torch.uint8)))
    return state, hidden[selected]


def scalar_reference(
    seeds: list[int],
    positions: list[int],
    rejection_steps: list[int],
    speculative_tokens: int,
) -> dict[str, torch.Tensor]:
    """Compute expected results with Python lists, without the graph recipe."""
    rows: dict[str, list] = {
        key: []
        for key in (
            "draft_tokens",
            "target_tokens",
            "accepted_count",
            "committed_tokens",
        )
    }
    for seed, position, reject in zip(seeds, positions, rejection_steps, strict=True):
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
        rows["accepted_count"].append(accepted)
        committed = proposals[:accepted] + [targets[accepted]]
        committed.extend([-1] * (speculative_tokens - accepted))
        rows["committed_tokens"].append(committed)
    result = {
        key: torch.tensor(value, dtype=torch.int32 if key == "accepted_count" else torch.long)
        for key, value in rows.items()
    }
    result["target_embeddings"] = torch.tensor(
        [
            [targets[index]]
            for targets, count in zip(rows["target_tokens"], rows["accepted_count"], strict=True)
            for index in (max(count - 1, 0), count)
        ],
        dtype=torch.float32,
    )
    return result
