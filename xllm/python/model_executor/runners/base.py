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

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
)
from xllm.python.model_executor.forward_context import LayerSynchronizer

ModelExecutionOutput = (
    torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
)


@dataclass(frozen=True)
class SpeculativeDeviceState:
    """Fixed-address Device state consumed by the next MTP iteration."""

    token_ids: torch.Tensor
    positions: torch.Tensor
    kv_seq_lens: torch.Tensor | None
    embeddings: torch.Tensor | None
    cache_slots: torch.Tensor | None
    topk_indices: torch.Tensor | None
    committed_mask: torch.Tensor


@dataclass(frozen=True)
class SpeculativeExecutionOutput:
    """Device outputs of one complete speculative iteration.

    The tensors are graph-entry-owned views. A caller must keep the entry
    generation alive until every asynchronous consumer has finished; the
    next graph replay may overwrite these views.
    """

    accepted_ids: torch.Tensor
    accepted_mask: torch.Tensor
    accepted_count: torch.Tensor
    next_state: SpeculativeDeviceState
    # Fixed-width target-context rows in the same layout consumed by the
    # runtime embedding cache: accepted draft prefix, replacement/bonus, then
    # -1 padding after the first rejection.
    committed_tokens: torch.Tensor
    # Target hidden rows [batch, K + 1, hidden] used to update the embedding
    # cache without a host-side gather.
    target_embeddings: torch.Tensor | None = None
    draft_tokens: torch.Tensor | None = None
    target_tokens: torch.Tensor | None = None
    logprobs: torch.Tensor | None = None
    top_logprobs: torch.Tensor | None = None


class BaseRunner(ABC):
    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
    ) -> None:
        self.model = model
        self.attention_backend = attention_backend
        self.device = device
        self.layer_caches: list[LayerCache] = []

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        self.layer_caches = layer_caches

    @abstractmethod
    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
    ) -> ModelExecutionOutput:
        pass
