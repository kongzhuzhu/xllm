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

"""Pack/unpack helpers for the three-collective MLA attention path.

The helpers keep communication layout separate from the model kernels.  C2
and C3 use equal-size all-to-all chunks within each DP group; C4 reduces
the partial outputs over the full attention weight world.
The caller must provide the global-token local-head input required by C2.  A
local-token q_b projection cannot satisfy that contract when q_b is sharded.
"""

from __future__ import annotations

import torch

from xllm.python import distributed
from xllm.python.attention.attn_dp_layout import AttnDpLayout


def q_head_all_to_all_dp_tp(
    q_heads_by_weight_rank: torch.Tensor,
    layout: AttnDpLayout,
    local_tokens: int,
    padded_tokens: int,
    tp_size: int,
    dp_size: int,
    tp_rank: int,
    dp_rank: int,
    group_name: str = "dp",
) -> torch.Tensor:
    """Route TP * DP weight shards to the ordinary TP token owners.

    ``q_heads_by_weight_rank`` contains every DP owner's padded rows on each
    weight rank: ``[dp_size*padded_tokens, weight_heads, q_dim]``.  Ranks with
    the same TP slot form one DP process group, so the equal chunks already
    have the destination-owner layout and no full-world zero padding is
    needed.  Each owner concatenates the ``dp_size`` received head shards and
    therefore runs the same number of heads as the baseline TP attention.
    """
    if tp_size <= 0 or dp_size <= 0 or tp_size * dp_size != layout.world_size:
        raise ValueError("tp_size*dp_size must equal the attention weight world")
    if not 0 <= tp_rank < tp_size or not 0 <= dp_rank < dp_size:
        raise ValueError("invalid TP/DP rank")
    if local_tokens <= 0 or padded_tokens < local_tokens:
        raise ValueError("invalid local/padded token count")
    expected = (dp_size * padded_tokens, layout.local_heads, layout.q_head_dim)
    if tuple(q_heads_by_weight_rank.shape) != expected:
        raise ValueError(f"C2 expects {expected}, got {tuple(q_heads_by_weight_rank.shape)}")

    received = torch.empty_like(q_heads_by_weight_rank)
    distributed.all_to_all_single(received, q_heads_by_weight_rank.contiguous(), group_name)
    received = received.view(dp_size, padded_tokens, layout.local_heads, layout.q_head_dim)
    owner_chunks = [received[source_dp] for source_dp in range(dp_size)]
    return torch.cat(owner_chunks, dim=1).narrow(0, 0, local_tokens).contiguous()


def attention_latent_all_to_all_dp_tp(
    attention_by_owner: torch.Tensor,
    layout: AttnDpLayout,
    local_tokens: int,
    padded_tokens: int,
    tp_size: int,
    dp_size: int,
    tp_rank: int,
    dp_rank: int,
    group_name: str = "dp",
) -> torch.Tensor:
    """Return owner attention rows to the matching DP weight ranks."""
    if tp_size <= 0 or dp_size <= 0 or tp_size * dp_size != layout.world_size:
        raise ValueError("tp_size*dp_size must equal the attention weight world")
    if not 0 <= tp_rank < tp_size or not 0 <= dp_rank < dp_size:
        raise ValueError("invalid TP/DP rank")
    if local_tokens <= 0 or padded_tokens < local_tokens:
        raise ValueError("invalid local/padded token count")
    expected = (local_tokens, layout.owner_local_heads, layout.kv_lora_rank)
    if tuple(attention_by_owner.shape) != expected:
        raise ValueError(f"C3 expects {expected}, got {tuple(attention_by_owner.shape)}")
    padded = attention_by_owner.new_zeros((padded_tokens, layout.owner_local_heads, layout.kv_lora_rank))
    padded[:local_tokens].copy_(attention_by_owner)
    packed = (
        padded.reshape(padded_tokens, dp_size, layout.local_heads, layout.kv_lora_rank)
        .permute(1, 0, 2, 3)
        .reshape(dp_size * padded_tokens, layout.local_heads, layout.kv_lora_rank)
        .contiguous()
    )
    received = torch.empty_like(packed)
    distributed.all_to_all_single(received, packed, group_name)
    # Equal received chunks are already in source-owner row order.
    return received


def quantized_value_all_to_all_dp_tp(
    quantized_value_by_owner: torch.Tensor,
    pertoken_scale: torch.Tensor,
    layout: AttnDpLayout,
    v_head_dim: int,
    local_tokens: int,
    padded_tokens: int,
    tp_size: int,
    dp_size: int,
    tp_rank: int,
    dp_rank: int,
    group_name: str = "dp",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route full-owner INT8 V shards and their shared dynamic scales (C3).

    ``quantized_value_by_owner`` is ``[local_tokens, owner_heads * v_head_dim]``
    after quantizing the complete owner row. Each shard's INT8 features and
    the exact FP32 scale bytes share one homogeneous payload and one all-to-all.
    """
    if tp_size <= 0 or dp_size <= 0 or tp_size * dp_size != layout.world_size:
        raise ValueError("tp_size*dp_size must equal the attention weight world")
    if not 0 <= tp_rank < tp_size or not 0 <= dp_rank < dp_size:
        raise ValueError("invalid TP/DP rank")
    if local_tokens <= 0 or padded_tokens < local_tokens:
        raise ValueError("invalid local/padded token count")
    if v_head_dim <= 0:
        raise ValueError("v_head_dim must be positive")
    expected_features = layout.owner_local_heads * v_head_dim
    if tuple(quantized_value_by_owner.shape) != (local_tokens, expected_features):
        raise ValueError(
            f"C3 quantized V expects {(local_tokens, expected_features)}, got {tuple(quantized_value_by_owner.shape)}"
        )
    if quantized_value_by_owner.dtype != torch.int8:
        raise ValueError("C3 quantized V must be int8")
    if tuple(pertoken_scale.shape) not in ((local_tokens,), (local_tokens, 1)):
        raise ValueError(
            f"C3 per-token scale expects {(local_tokens,)} or {(local_tokens, 1)}, got {tuple(pertoken_scale.shape)}"
        )
    pertoken_scale = pertoken_scale.reshape(local_tokens)
    if pertoken_scale.dtype != torch.float32:
        raise ValueError("C3 per-token scales must be float32")
    if pertoken_scale.device != quantized_value_by_owner.device:
        raise ValueError("C3 per-token scales and quantized V must use the same device")
    shard_features = layout.local_heads * v_head_dim
    if expected_features != shard_features * dp_size:
        raise ValueError("owner V features must divide evenly across the DP weight shards")

    padded_value = quantized_value_by_owner.new_zeros((padded_tokens, expected_features))
    padded_value[:local_tokens].copy_(quantized_value_by_owner)
    padded_scale = pertoken_scale.new_ones((padded_tokens,))
    padded_scale[:local_tokens].copy_(pertoken_scale)
    scale_bytes = padded_scale.contiguous().view(torch.uint8).reshape(padded_tokens, 4)
    payload_width = shard_features + scale_bytes.shape[-1]
    packed = torch.empty(
        (dp_size * padded_tokens, payload_width),
        dtype=torch.uint8,
        device=quantized_value_by_owner.device,
    )
    for shard in range(dp_size):
        target = packed.narrow(0, shard * padded_tokens, padded_tokens)
        feature_slice = padded_value.narrow(1, shard * shard_features, shard_features)
        target[:, :shard_features].copy_(feature_slice.view(torch.uint8))
        target[:, shard_features:].copy_(scale_bytes)

    received = torch.empty_like(packed)
    distributed.all_to_all_single(received, packed, group_name)
    quantized_value = (
        received[:, :shard_features]
        .contiguous()
        .view(torch.int8)
        .reshape(dp_size * padded_tokens, layout.local_heads, v_head_dim)
    )
    received_scale = received[:, shard_features:].contiguous().view(torch.float32).reshape(-1)
    return quantized_value, received_scale


def o_all_reduce_dp_tp(
    o_partials_by_weight_rank: torch.Tensor,
    layout: AttnDpLayout,
    padded_tokens: int,
    dp_size: int,
    group_name: str = "attn_dp",
) -> torch.Tensor:
    """Sum all attention weight shards while retaining padded DP rows (C4)."""
    expected = (dp_size * padded_tokens, layout.hidden_size)
    if dp_size <= 0 or padded_tokens <= 0 or tuple(o_partials_by_weight_rank.shape) != expected:
        raise ValueError(f"C4 expects {expected}, got {tuple(o_partials_by_weight_rank.shape)}")
    output = o_partials_by_weight_rank.contiguous()
    distributed.all_reduce_(output, group_name)
    return output


__all__ = [
    "q_head_all_to_all_dp_tp",
    "attention_latent_all_to_all_dp_tp",
    "quantized_value_all_to_all_dp_tp",
    "o_all_reduce_dp_tp",
]
