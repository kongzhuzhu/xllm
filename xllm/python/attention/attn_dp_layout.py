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

"""Reference layouts for the three-collective GLM attention DP/TP path.

The target path keeps ``q_a_proj`` and ``kv_a_proj_with_mqa`` replicated while
sharding the head-dependent weights. This module describes the logical tensor
layouts at C2 (Q heads), C3 (attention latent), and C4 (O reduction). It is a
CPU-testable reference and deliberately contains no process-group calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttnDpLayout:
    """Shape contract for one attention DP/TP group."""

    world_size: int
    num_heads: int
    q_head_dim: int
    kv_lora_rank: int
    hidden_size: int
    owner_heads: int | None = None

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if self.num_heads <= 0 or self.num_heads % self.world_size:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by world_size ({self.world_size})")
        if self.q_head_dim <= 0 or self.kv_lora_rank <= 0 or self.hidden_size <= 0:
            raise ValueError("attention dimensions must be positive")
        if self.owner_heads is not None and self.owner_heads <= 0:
            raise ValueError("owner_heads must be positive")

    @property
    def local_heads(self) -> int:
        return self.num_heads // self.world_size

    @property
    def owner_local_heads(self) -> int:
        """Heads executed by a TP token-owner rank.

        The original reference path uses one rank for both ownership and
        weight sharding, so this equals ``local_heads``.  In DP2/TP8 the
        candidate has 16 weight ranks but only 8 token-owner ranks per DP
        replica, hence each owner receives two weight shards.
        """
        return self.owner_heads if self.owner_heads is not None else self.local_heads

    def _validate_rank_tensors(
        self,
        tensors: Sequence[torch.Tensor],
        expected_rank: int,
        expected_shape: tuple[int, ...] | None = None,
        name: str = "tensor",
    ) -> None:
        if len(tensors) != self.world_size:
            raise ValueError(f"{name} needs {self.world_size} rank tensors, got {len(tensors)}")
        for rank, tensor in enumerate(tensors):
            if tensor.ndim != expected_rank:
                raise ValueError(f"{name}[{rank}] must have rank {expected_rank}, got {tensor.ndim}")
            if expected_shape is not None and tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{name}[{rank}] must have shape {expected_shape}, got {tuple(tensor.shape)}")

    def c2_q_heads_to_owners(
        self,
        q_heads_by_rank: Sequence[torch.Tensor],
        local_tokens: int,
    ) -> tuple[torch.Tensor, ...]:
        """Assemble each owner's complete Q heads from head-sharded ranks.

        Each source rank owns ``[global_tokens, local_heads, q_head_dim]`` and
        has already produced the Q heads for every global token. The result for
        owner ``r`` is ``[local_tokens, num_heads, q_head_dim]`` containing only
        that owner's token rows.
        """

        if local_tokens <= 0:
            raise ValueError(f"local_tokens must be positive, got {local_tokens}")
        global_tokens = local_tokens * self.world_size
        expected_shape = (global_tokens, self.local_heads, self.q_head_dim)
        self._validate_rank_tensors(q_heads_by_rank, 3, expected_shape, "q_heads_by_rank")
        return tuple(
            torch.cat(
                [q_heads[owner * local_tokens : (owner + 1) * local_tokens] for q_heads in q_heads_by_rank],
                dim=1,
            )
            for owner in range(self.world_size)
        )

    def c3_attention_to_weight_ranks(
        self,
        attention_by_owner: Sequence[torch.Tensor],
        local_tokens: int,
    ) -> tuple[torch.Tensor, ...]:
        """Pack complete owner attention outputs for head-sharded weights.

        Each owner supplies ``[local_tokens, num_heads, kv_lora_rank]``. The
        result for weight rank ``r`` is ``[global_tokens, local_heads,
        kv_lora_rank]`` in owner-rank order.
        """

        if local_tokens <= 0:
            raise ValueError(f"local_tokens must be positive, got {local_tokens}")
        expected_shape = (local_tokens, self.num_heads, self.kv_lora_rank)
        self._validate_rank_tensors(attention_by_owner, 3, expected_shape, "attention_by_owner")
        return tuple(
            torch.cat(
                [
                    attention[:, rank * self.local_heads : (rank + 1) * self.local_heads]
                    for attention in attention_by_owner
                ],
                dim=0,
            )
            for rank in range(self.world_size)
        )

    def c4_o_partials_to_owners(
        self,
        o_partials_by_rank: Sequence[torch.Tensor],
        local_tokens: int,
    ) -> tuple[torch.Tensor, ...]:
        """Reduce full-token O partials and return each owner's token rows."""

        if local_tokens <= 0:
            raise ValueError(f"local_tokens must be positive, got {local_tokens}")
        global_tokens = local_tokens * self.world_size
        expected_shape = (global_tokens, self.hidden_size)
        self._validate_rank_tensors(o_partials_by_rank, 2, expected_shape, "o_partials_by_rank")
        reduced = torch.stack(tuple(o_partials_by_rank), dim=0).sum(dim=0)
        return tuple(reduced[owner * local_tokens : (owner + 1) * local_tokens] for owner in range(self.world_size))

    def c4_o_partials_to_global(
        self,
        o_partials_by_rank: Sequence[torch.Tensor],
        local_tokens: int,
    ) -> torch.Tensor:
        """Reference C4 AllReduce result with global token rows retained."""

        if local_tokens <= 0:
            raise ValueError(f"local_tokens must be positive, got {local_tokens}")
        global_tokens = local_tokens * self.world_size
        expected_shape = (global_tokens, self.hidden_size)
        self._validate_rank_tensors(o_partials_by_rank, 2, expected_shape, "o_partials_by_rank")
        return torch.stack(tuple(o_partials_by_rank), dim=0).sum(dim=0)


__all__ = ["AttnDpLayout"]
