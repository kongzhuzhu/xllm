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

"""Graph-safe sampling primitives for the fused MTP recipe.

The C++ sampler remains the eager semantic oracle.  This module mirrors its
tensor semantics without reading a token or a probability on the host.  All
row counts are fixed before capture; ``expand_for_rows`` only repeats fixed
request controls for the statically expanded target rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from xllm.python import distributed


@dataclass(frozen=True)
class MtpSamplingPlan:
    """Fixed-shape sampling controls owned by one captured graph variant."""

    batch_size: int
    do_sample: torch.Tensor | None = None
    temperatures: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    top_k: torch.Tensor | None = None
    frequency_penalties: torch.Tensor | None = None
    presence_penalties: torch.Tensor | None = None
    repetition_penalties: torch.Tensor | None = None
    unique_token_ids: torch.Tensor | None = None
    unique_token_counts: torch.Tensor | None = None
    unique_token_ids_lens: torch.Tensor | None = None
    filter_mask: torch.Tensor | None = None
    filter_bitmask: torch.Tensor | None = None
    all_random_sample: bool = False
    all_greedy_sample: bool = True
    return_probs: bool = True
    logprobs: bool = False
    max_top_logprobs: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], *, batch_size: int) -> MtpSamplingPlan:
        """Build a plan from the C++ SamplingParameters bridge."""
        fields = {
            name: value.get(name)
            for name in (
                "do_sample",
                "temperatures",
                "top_p",
                "top_k",
                "frequency_penalties",
                "presence_penalties",
                "repetition_penalties",
                "unique_token_ids",
                "unique_token_counts",
                "unique_token_ids_lens",
                "filter_mask",
                "filter_bitmask",
            )
        }
        return cls(
            batch_size=int(value.get("batch_size", batch_size)),
            **fields,
            all_random_sample=bool(value.get("all_random_sample", False)),
            all_greedy_sample=bool(value.get("all_greedy_sample", True)),
            return_probs=bool(value.get("return_probs", True)),
            logprobs=bool(value.get("logprobs", False)),
            max_top_logprobs=int(value.get("max_top_logprobs", 0)),
        )

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("MTP sampling batch_size must be positive")
        if self.all_random_sample and self.all_greedy_sample:
            raise ValueError("MTP sampling plan cannot be both random and greedy")
        if not self.all_random_sample and not self.all_greedy_sample and self.do_sample is None:
            raise ValueError("mixed MTP sampling requires per-request do_sample")
        for name in (
            "do_sample",
            "temperatures",
            "top_p",
            "top_k",
            "frequency_penalties",
            "presence_penalties",
            "repetition_penalties",
            "filter_mask",
            "filter_bitmask",
        ):
            value = getattr(self, name)
            if value is not None and value.shape[0] != self.batch_size:
                raise ValueError(f"{name} must have one row per MTP request")
            if value is not None and name not in ("filter_mask", "filter_bitmask") and value.ndim != 1:
                raise ValueError(f"{name} must be a vector")
        if self.unique_token_ids_lens is not None and self.unique_token_ids_lens.shape != (self.batch_size,):
            raise ValueError("unique token lengths must have one value per request")
        if (self.unique_token_ids is None) != (self.unique_token_counts is None):
            raise ValueError("unique token ids and counts must be provided together")
        if self.unique_token_ids is not None:
            if self.unique_token_ids.ndim != 2 or self.unique_token_counts is None:
                raise ValueError("unique token controls must be [batch, unique_tokens]")
            if self.unique_token_counts.shape != self.unique_token_ids.shape:
                raise ValueError("unique token ids and counts must have the same shape")
            if self.unique_token_ids.shape[0] != self.batch_size:
                raise ValueError("unique token controls must have one row per request")
        for name in ("filter_mask", "filter_bitmask"):
            value = getattr(self, name)
            if value is not None and value.ndim != 2:
                raise ValueError(f"{name} must be a matrix")

    @property
    def plain_greedy(self) -> bool:
        """Whether logits need no request-specific distribution transforms."""
        return self.all_greedy_sample and all(
            value is None
            for value in (
                self.temperatures,
                self.top_k,
                self.top_p,
                self.frequency_penalties,
                self.presence_penalties,
                self.repetition_penalties,
                self.filter_mask,
                self.filter_bitmask,
            )
        )

    @property
    def mode(self) -> str:
        if self.all_random_sample:
            return "random"
        if self.all_greedy_sample:
            return "greedy"
        return "mixed"

    def request_do_sample(self, *, device: torch.device) -> torch.Tensor:
        if self.all_greedy_sample:
            return torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        if self.all_random_sample:
            return torch.ones(self.batch_size, dtype=torch.bool, device=device)
        if self.do_sample is not None:
            return self.do_sample.to(device=device, dtype=torch.bool)
        raise RuntimeError("mixed MTP sampling requires per-request do_sample")

    def expand_for_rows(self, rows: int) -> MtpSamplingPlan:
        """Repeat request controls for a fixed request-major row expansion."""
        if rows <= 0 or rows % self.batch_size != 0:
            raise ValueError("expanded sampling rows must be a positive batch multiple")
        repeats = rows // self.batch_size

        def repeat(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.repeat_interleave(repeats, dim=0)

        return MtpSamplingPlan(
            batch_size=rows,
            do_sample=repeat(self.do_sample),
            temperatures=repeat(self.temperatures),
            top_p=repeat(self.top_p),
            top_k=repeat(self.top_k),
            frequency_penalties=repeat(self.frequency_penalties),
            presence_penalties=repeat(self.presence_penalties),
            repetition_penalties=repeat(self.repetition_penalties),
            unique_token_ids=repeat(self.unique_token_ids),
            unique_token_counts=repeat(self.unique_token_counts),
            unique_token_ids_lens=repeat(self.unique_token_ids_lens),
            filter_mask=repeat(self.filter_mask),
            filter_bitmask=repeat(self.filter_bitmask),
            all_random_sample=self.all_random_sample,
            all_greedy_sample=self.all_greedy_sample,
            return_probs=self.return_probs,
            logprobs=self.logprobs,
            max_top_logprobs=self.max_top_logprobs,
        )

    def clone_for_graph(self) -> MtpSamplingPlan:
        """Clone controls so request-owned storage cannot invalidate a graph."""

        def clone(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.detach().clone().contiguous()

        return MtpSamplingPlan(
            batch_size=self.batch_size,
            do_sample=clone(self.do_sample),
            temperatures=clone(self.temperatures),
            top_p=clone(self.top_p),
            top_k=clone(self.top_k),
            frequency_penalties=clone(self.frequency_penalties),
            presence_penalties=clone(self.presence_penalties),
            repetition_penalties=clone(self.repetition_penalties),
            unique_token_ids=clone(self.unique_token_ids),
            unique_token_counts=clone(self.unique_token_counts),
            unique_token_ids_lens=clone(self.unique_token_ids_lens),
            filter_mask=clone(self.filter_mask),
            filter_bitmask=clone(self.filter_bitmask),
            all_random_sample=self.all_random_sample,
            all_greedy_sample=self.all_greedy_sample,
            return_probs=self.return_probs,
            logprobs=self.logprobs,
            max_top_logprobs=self.max_top_logprobs,
        )

    def layout_signature(self) -> tuple[object, ...]:
        def signature(value: torch.Tensor | None) -> tuple[object, ...] | None:
            if value is None:
                return None
            return (tuple(value.shape), str(value.dtype), str(value.device))

        return (
            self.batch_size,
            self.mode,
            self.return_probs,
            self.logprobs,
            self.max_top_logprobs,
            tuple(
                signature(getattr(self, name))
                for name in (
                    "do_sample",
                    "temperatures",
                    "top_p",
                    "top_k",
                    "frequency_penalties",
                    "presence_penalties",
                    "repetition_penalties",
                    "unique_token_ids",
                    "unique_token_counts",
                    "unique_token_ids_lens",
                    "filter_mask",
                    "filter_bitmask",
                )
            ),
        )

    def update_from(self, other: MtpSamplingPlan) -> None:
        """Copy replay controls into stable graph-owned tensors."""
        if self.batch_size != other.batch_size or self.layout_signature() != other.layout_signature():
            raise RuntimeError("MTP sampling plan shape, mode, or output contract changed")

        for name in (
            "do_sample",
            "temperatures",
            "top_p",
            "top_k",
            "frequency_penalties",
            "presence_penalties",
            "repetition_penalties",
            "unique_token_ids",
            "unique_token_counts",
            "unique_token_ids_lens",
            "filter_mask",
            "filter_bitmask",
        ):
            current = getattr(self, name)
            incoming = getattr(other, name)
            if current is not None:
                assert incoming is not None
                current.copy_(incoming)


def coerce_sampling_plan(
    value: MtpSamplingPlan | Mapping[str, object] | None,
    *,
    batch_size: int,
) -> MtpSamplingPlan | None:
    if value is None or isinstance(value, MtpSamplingPlan):
        return value
    return MtpSamplingPlan.from_mapping(value, batch_size=batch_size)


@dataclass(frozen=True)
class MtpSampledLogits:
    tokens: torch.Tensor
    probs: torch.Tensor | None
    log_probs: torch.Tensor | None


def _apply_penalties(logits: torch.Tensor, plan: MtpSamplingPlan) -> torch.Tensor:
    if plan.unique_token_ids is None or plan.unique_token_counts is None:
        return logits
    ids = plan.unique_token_ids.to(device=logits.device, dtype=torch.long)
    counts = plan.unique_token_counts.to(device=logits.device, dtype=logits.dtype)
    valid = counts > 0
    if plan.unique_token_ids_lens is not None:
        lengths = plan.unique_token_ids_lens.to(device=logits.device)
        valid = valid & (torch.arange(ids.shape[1], device=logits.device) < lengths.unsqueeze(-1))
    # Padding may repeat a real token (usually zero). A masked scatter of
    # scores would still overwrite that real token with the padding value.
    # Scatter additive deltas instead: invalid entries contribute exactly zero.
    ids = torch.where(valid, ids, 0)
    original = logits.gather(dim=-1, index=ids)
    scores = original
    if plan.frequency_penalties is not None:
        penalties = plan.frequency_penalties.to(device=logits.device, dtype=logits.dtype)
        scores = scores - counts * penalties.unsqueeze(-1)
    if plan.presence_penalties is not None:
        penalties = plan.presence_penalties.to(device=logits.device, dtype=logits.dtype)
        scores = scores - valid.to(logits.dtype) * penalties.unsqueeze(-1)
    if plan.repetition_penalties is not None:
        penalties = plan.repetition_penalties.to(device=logits.device, dtype=logits.dtype)
        penalties = penalties.unsqueeze(-1).clamp_min(torch.finfo(logits.dtype).tiny)
        scores = torch.where(scores < 0, scores * penalties, scores / penalties)
    # Infinite logits remain infinite under finite penalties; subtracting
    # them would otherwise introduce NaNs into the additive update.
    delta = torch.where(valid & torch.isfinite(original), scores - original, 0)
    return logits.scatter_add(dim=-1, index=ids, src=delta)


def _apply_filter(logits: torch.Tensor, plan: MtpSamplingPlan) -> torch.Tensor:
    if plan.filter_bitmask is not None:
        bitmask = plan.filter_bitmask.to(device=logits.device)
        vocab = logits.shape[-1]
        token_ids = torch.arange(vocab, device=logits.device, dtype=torch.long)
        word_ids = token_ids // 32
        bit_ids = token_ids.remainder(32)
        words = bitmask.gather(dim=-1, index=word_ids.view(1, -1).expand(logits.shape[0], -1))
        allowed = ((words.to(torch.long) >> bit_ids.view(1, -1)) & 1).to(torch.bool)
        logits = logits.masked_fill(~allowed, float("-inf"))
    elif plan.filter_mask is not None:
        filter_mask = plan.filter_mask.to(device=logits.device, dtype=logits.dtype)
        logits = logits + filter_mask
    return logits


def _apply_temperature(logits: torch.Tensor, plan: MtpSamplingPlan) -> torch.Tensor:
    if plan.temperatures is not None:
        temperatures = plan.temperatures.to(device=logits.device, dtype=logits.dtype)
        temperatures = torch.where(temperatures == 0, torch.ones_like(temperatures), temperatures)
        logits = logits / temperatures.unsqueeze(-1)
    return logits


def _apply_top_k_top_p(logits: torch.Tensor, plan: MtpSamplingPlan) -> torch.Tensor:
    logits = _apply_temperature(logits, plan)
    if plan.top_k is None and plan.top_p is None:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    vocab = logits.shape[-1]
    if plan.top_k is not None:
        raw_top_k = plan.top_k.to(device=logits.device, dtype=torch.long)
        # C++ apply_top_k_top_p treats non-positive top_k as unlimited.
        top_k = torch.where(raw_top_k <= 0, torch.full_like(raw_top_k, vocab), raw_top_k.clamp_max(vocab))
        positions = torch.arange(vocab, device=logits.device, dtype=torch.long).view(1, -1)
        sorted_logits = sorted_logits.masked_fill(positions >= top_k.unsqueeze(-1), float("-inf"))
    if plan.top_p is not None:
        top_p = plan.top_p.to(device=logits.device, dtype=logits.dtype).unsqueeze(-1)
        probs = torch.softmax(sorted_logits, dim=-1, dtype=torch.float32)
        cumulative = probs.cumsum(dim=-1)
        remove = (cumulative - probs) > top_p
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.zeros_like(logits).scatter(dim=-1, index=sorted_indices, src=sorted_logits)


def _gumbel_argmax(log_probs: torch.Tensor, uniform: torch.Tensor | None = None) -> torch.Tensor:
    if uniform is None:
        uniform = torch.rand(log_probs.shape, dtype=torch.float32, device=log_probs.device)
    elif uniform.shape != log_probs.shape or uniform.device != log_probs.device:
        raise ValueError("sampling uniforms must match the log-probability shape and device")
    uniform = uniform.clamp_min(torch.finfo(torch.float32).tiny)
    noise = -torch.log(-torch.log(uniform))
    return (log_probs.to(torch.float32) + noise).argmax(dim=-1)


def _tp_consensus(value: torch.Tensor) -> torch.Tensor:
    """Keep sampled control tokens identical across one attention TP group."""
    if distributed.tp_world_size(value.device) <= 1:
        return value
    value = value.contiguous()
    distributed.broadcast_(value, src=0, group_name="tp")
    return value


def sample_logits(
    logits: torch.Tensor,
    plan: MtpSamplingPlan,
    *,
    uniform: torch.Tensor | None = None,
    require_probs: bool = False,
) -> MtpSampledLogits:
    """Apply all fixed-shape sampling controls and sample on Device."""
    if logits.ndim != 2 or logits.shape[0] != plan.batch_size:
        raise ValueError("sampling logits must have shape [plan.batch_size, vocab]")
    needs_probs = require_probs or plan.return_probs or not plan.all_greedy_sample
    needs_log_probs = needs_probs or plan.logprobs or plan.max_top_logprobs > 0
    if (
        not needs_log_probs
        and plan.unique_token_ids is None
        and plan.filter_mask is None
        and plan.filter_bitmask is None
        and plan.temperatures is None
    ):
        # Plain greedy preserves the logits' ordering and needs no FP32
        # vocabulary-sized temporary. This is the production v1 contract.
        return MtpSampledLogits(tokens=logits.argmax(dim=-1), probs=None, log_probs=None)
    processed = logits.to(torch.float32)
    processed = _apply_penalties(processed, plan)
    processed = _apply_filter(processed, plan)
    if not needs_log_probs:
        # Top-k/top-p retain the maximum. Without a requested distribution,
        # greedy sampling needs neither a vocabulary sort nor normalization.
        tokens = _apply_temperature(processed, plan).argmax(dim=-1)
        return MtpSampledLogits(tokens=tokens, probs=None, log_probs=None)
    processed = _apply_top_k_top_p(processed, plan)
    log_probs = torch.log_softmax(processed, dim=-1)
    probs = log_probs.exp() if needs_probs else None
    greedy_tokens = processed.argmax(dim=-1)
    if plan.all_greedy_sample:
        tokens = greedy_tokens
    else:
        random_tokens = _gumbel_argmax(log_probs, uniform)
        if plan.all_random_sample:
            tokens = random_tokens
        else:
            tokens = torch.where(plan.request_do_sample(device=logits.device), random_tokens, greedy_tokens)
        tokens = _tp_consensus(tokens)
    return MtpSampledLogits(tokens=tokens.to(torch.long), probs=probs, log_probs=log_probs)


def probabilistic_acceptance(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_tokens: torch.Tensor,
    target_probs: torch.Tensor,
    do_sample: torch.Tensor,
    *,
    acceptance_uniform: torch.Tensor | None = None,
    recovery_uniform: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run mixed greedy/probability MTP acceptance and residual recovery on Device."""
    if draft_tokens.ndim != 2 or target_tokens.ndim != 2:
        raise ValueError("MTP token matrices must be two-dimensional")
    batch_size, speculative_tokens = draft_tokens.shape
    if target_tokens.shape != (batch_size, speculative_tokens + 1):
        raise ValueError("target tokens must contain K+1 rows")
    if speculative_tokens < 1:
        raise ValueError("MTP acceptance requires at least one draft token")
    if draft_probs.ndim != 3 or draft_probs.shape[:2] != (batch_size, speculative_tokens):
        raise ValueError("draft probabilities must be [batch, K, vocab]")
    if target_probs.shape != (batch_size, speculative_tokens + 1, draft_probs.shape[-1]):
        raise ValueError("target probabilities must be [batch, K+1, vocab]")

    target_draft_probs = target_probs[:, :speculative_tokens]
    selected_target = target_draft_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
    selected_draft = draft_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
    selected_draft = selected_draft.clamp_min(torch.finfo(selected_draft.dtype).tiny)
    acceptance_probability = (selected_target / selected_draft).clamp(max=1.0)
    if acceptance_uniform is None:
        acceptance_uniform = torch.rand(acceptance_probability.shape, device=draft_tokens.device)
    elif acceptance_uniform.shape != draft_tokens.shape or acceptance_uniform.device != draft_tokens.device:
        raise ValueError("acceptance uniforms must match the draft token shape and device")
    random_accept = acceptance_uniform < acceptance_probability
    greedy_accept = draft_tokens.eq(target_tokens[:, :speculative_tokens])
    do_sample = do_sample.to(device=draft_tokens.device, dtype=torch.bool).view(batch_size, 1)
    accepted = torch.where(do_sample, random_accept, greedy_accept)
    accepted = _tp_consensus(accepted.to(torch.int32)).to(torch.bool)
    accepted_mask = torch.cumprod(accepted.to(torch.int32), dim=1).to(torch.bool)
    accepted_count = accepted_mask.sum(dim=1, dtype=torch.int32)

    residual = (target_draft_probs - draft_probs).clamp_min(0)
    residual_sum = residual.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(residual, 1.0 / residual.shape[-1])
    residual_probs = torch.where(
        residual_sum > 0,
        residual / residual_sum.clamp_min(torch.finfo(residual.dtype).tiny),
        uniform,
    )
    # log(0) must remain -inf: masked tokens cannot be recovered, even with
    # extreme noise or a very small positive residual mass.
    residual_tokens = _gumbel_argmax(torch.log(residual_probs), recovery_uniform)
    residual_tokens = _tp_consensus(residual_tokens)
    residual_tokens = residual_tokens.reshape(batch_size, speculative_tokens)
    random_replacement = residual_tokens[:, 0]
    replacement_indices = accepted_count.to(torch.long).clamp_max(speculative_tokens - 1)
    random_replacement = residual_tokens.gather(1, replacement_indices.unsqueeze(-1)).squeeze(-1)

    first_rejection = (~accepted_mask).to(torch.int32).argmax(dim=1)
    greedy_replacement = target_tokens[:, :speculative_tokens].gather(1, first_rejection.unsqueeze(-1)).squeeze(-1)
    replacement = torch.where(do_sample.squeeze(-1), random_replacement, greedy_replacement)
    all_accepted = accepted_count.eq(speculative_tokens)
    next_tokens = torch.where(all_accepted, target_tokens[:, speculative_tokens], replacement)
    accepted_ids = torch.where(accepted_mask, draft_tokens, torch.full_like(draft_tokens, -1))
    return accepted_ids, accepted_mask, accepted_count, next_tokens
