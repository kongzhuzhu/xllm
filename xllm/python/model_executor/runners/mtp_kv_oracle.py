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

"""Opt-in, synchronous KV payload oracle for unsharded paged MTP caches.

The oracle snapshots all planned writes, runs an independent eager recipe,
restores the original payload in place, then checks the real graph result.
Capture warmup writes are restored as well. No cache allocation is rebound,
and a failed comparison propagates to the worker. This diagnostic doubles
model execution and synchronizes the device; it is not a performance mode.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch

from scripts.logger import logger
from xllm.python.attention.backend import LayerCache
from xllm.python.model_executor.runners.base import SpeculativeExecutionOutput
from xllm.python.model_executor.runners.mtp_acl_graph import (
    MtpAclGraphRunner,
    MtpGraphRecipe,
    MtpRoleAdapter,
)

_PAGED_FIELDS = frozenset(("key", "value", "index", "indexer_scale"))


def build_attention_read_slots(
    block_table: torch.Tensor,
    kv_seq_lens: Sequence[int] | torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, ...]:
    """Materialize the physical slots an attention invocation may read."""
    if block_table.ndim != 2:
        raise ValueError("attention read block table must be two-dimensional")
    if page_size <= 0:
        raise ValueError("attention read page size must be positive")
    if isinstance(kv_seq_lens, torch.Tensor):
        lengths = kv_seq_lens.detach().cpu().reshape(-1).tolist()
    else:
        lengths = list(kv_seq_lens)
    if len(lengths) != block_table.shape[0]:
        raise ValueError("attention read KV lengths must match block-table rows")
    table = block_table.detach().cpu().to(torch.long)
    result: list[torch.Tensor] = []
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        if length < 0:
            raise ValueError("attention read KV lengths must be nonnegative")
        offsets = torch.arange(length, dtype=torch.long)
        page_indices = offsets // page_size
        if page_indices.numel() and int(page_indices[-1]) >= table.shape[1]:
            raise ValueError("attention read KV length exceeds block-table capacity")
        block_ids = table[row].index_select(0, page_indices) if page_indices.numel() else table[row][:0]
        if block_ids.numel() and bool(block_ids.lt(0).any()):
            raise ValueError("attention read block table contains an invalid page")
        result.append(block_ids * page_size + offsets.remainder(page_size))
    return tuple(result)


@dataclass(frozen=True)
class _SlotPayload:
    name: str
    cache: torch.Tensor
    slots: torch.Tensor
    values: torch.Tensor


class PagedKvSnapshot:
    """Save/restore selected ND cache slots without changing bound addresses."""

    def __init__(self, payloads: Sequence[_SlotPayload]) -> None:
        self._payloads = tuple(payloads)

    @classmethod
    def capture(
        cls,
        roles: Sequence[tuple[str, Sequence[LayerCache], torch.Tensor]],
    ) -> PagedKvSnapshot:
        payloads: list[_SlotPayload] = []
        for role, caches, write_slots in roles:
            if not caches:
                raise ValueError(f"{role} KV oracle requires bound layer caches")
            # Dynamic unique and CPU validation deliberately stay outside capture.
            slots_host = torch.unique(write_slots.detach().cpu().long())
            slots_host = slots_host[slots_host.ge(0)]
            if slots_host.numel() == 0 or int(slots_host.min()) < 0:
                raise ValueError(f"{role} KV oracle requires nonnegative, nonempty write slots")
            for layer, cache in enumerate(caches):
                if cache.key is None or cache.value is None or cache.key.ndim != 4:
                    raise ValueError(f"{role}/{layer} KV oracle requires ND paged key/value caches")
                page_shape = cache.key.shape[:2]
                for field in fields(cache):
                    tensor = getattr(cache, field.name)
                    if tensor is None:
                        continue
                    name = f"{role}/{layer}/{field.name}"
                    if field.name not in _PAGED_FIELDS:
                        raise ValueError(f"{name}: unsupported mutable cache in KV oracle")
                    if tensor.ndim < 3 or tensor.shape[:2] != page_shape or not tensor.is_contiguous():
                        raise ValueError(f"{name}: KV oracle requires contiguous [block, page, ...] layout")
                    flat = tensor.view(page_shape[0] * page_shape[1], *tensor.shape[2:])
                    if int(slots_host.max()) >= flat.shape[0]:
                        raise ValueError(f"{name}: write slot exceeds cache capacity")
                    slots = slots_host.to(device=tensor.device)
                    payloads.append(_SlotPayload(name, flat, slots, flat.index_select(0, slots)))
        return cls(payloads)

    @property
    def tensor_count(self) -> int:
        return len(self._payloads)

    @torch.inference_mode()
    def restore(self) -> None:
        for payload in self._payloads:
            payload.cache.index_copy_(0, payload.slots, payload.values)

    def assert_matches(self, reference: PagedKvSnapshot) -> None:
        if len(self._payloads) != len(reference._payloads):
            raise AssertionError("MTP KV oracle cache binding count changed")
        for actual, expected in zip(self._payloads, reference._payloads, strict=True):
            if actual.name != expected.name:
                raise AssertionError("MTP KV oracle cache binding order changed")
            torch.testing.assert_close(actual.slots, expected.slots, rtol=0, atol=0)
            torch.testing.assert_close(
                actual.values.cpu(),
                expected.values.cpu(),
                rtol=0,
                atol=0,
                msg=f"MTP KV payload mismatch at {actual.name}",
            )


def _output_snapshot(output: SpeculativeExecutionOutput) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for field in fields(output):
        value = getattr(output, field.name)
        if isinstance(value, torch.Tensor):
            tensors[field.name] = value.detach().cpu().clone()
    for field in fields(output.next_state):
        value = getattr(output.next_state, field.name)
        if isinstance(value, torch.Tensor):
            tensors[f"next_state.{field.name}"] = value.detach().cpu().clone()
    return tensors


@dataclass(frozen=True)
class _ReferenceResult:
    payload: PagedKvSnapshot
    output: dict[str, torch.Tensor]


class MtpKvPayloadOracle:
    """Compare every target/draft write and output from identical initial KV."""

    def __init__(
        self, recipe: MtpGraphRecipe, draft_caches: Sequence[LayerCache], target_caches: Sequence[LayerCache]
    ) -> None:
        if not isinstance(recipe.draft_forward, MtpRoleAdapter) or not isinstance(
            recipe.target_forward, MtpRoleAdapter
        ):
            raise TypeError("MTP KV oracle requires real role adapters")
        self._recipe = recipe
        self._draft_caches = tuple(draft_caches)
        self._target_caches = tuple(target_caches)
        self._checks = 0
        self._rejected_target_slots: tuple[torch.Tensor, ...] | None = None
        self._expected_next_base_positions: torch.Tensor | None = None

    def snapshot(self) -> PagedKvSnapshot:
        return PagedKvSnapshot.capture(
            (
                ("draft", self._draft_caches, self._recipe.draft_forward.cache_write_slots()),
                ("target", self._target_caches, self._recipe.target_forward.cache_write_slots()),
            )
        )

    def observe_next_attention_read_set(self, base_positions: torch.Tensor) -> None:
        """Reject stale reads of rejected slots on the next target request.

        Chunked target verification may read a slot that it rewrites in the
        same invocation.  That is a causal intra-chunk dependency, not a read
        of the rejected payload left by the preceding invocation.  Remove the
        current target write set before checking the rejected-slot invariant.
        """
        if self._rejected_target_slots is None:
            return
        expected = self._expected_next_base_positions
        if expected is None or not torch.equal(base_positions.detach().cpu().reshape(-1), expected):
            # Capture warmup and a later user request can share one graph
            # variant while belonging to different sequences.  Do not carry
            # rejected physical slots across that request boundary.
            self._rejected_target_slots = None
            self._expected_next_base_positions = None
            logger.info("MTP rejected-slot attention probe reset at request boundary")
            return
        target = self._recipe.target_forward
        block_table, kv_seq_lens, page_size = target.attention_read_plan(self._recipe.batch_size)
        read_slots = build_attention_read_slots(block_table, kv_seq_lens, page_size)
        verify_slots = target.verify_slots(self._recipe.batch_size)
        for request, (rejected, readable, writes) in enumerate(
            zip(self._rejected_target_slots, read_slots, verify_slots, strict=True)
        ):
            rewritten = set(writes[writes.ge(0)].tolist())
            historical = [slot for slot in readable.tolist() if slot not in rewritten]
            overlap = sorted(set(rejected.tolist()).intersection(historical))
            if overlap:
                raise AssertionError(
                    "MTP rejected target slots are readable on next attention "
                    f"request={request}: overlap={overlap}, rejected={rejected.tolist()}, "
                    f"readable={readable.tolist()}, rewritten={writes[writes.ge(0)].tolist()}"
                )
        logger.info(
            "MTP rejected-slot attention read probe passed: requests=%s page_size=%s",
            len(read_slots),
            page_size,
        )

    def record_rejected_target_slots(self, output: SpeculativeExecutionOutput) -> None:
        """Remember target suffix slots that must stay outside the next read set."""
        accepted_count = output.accepted_count.detach().cpu().reshape(-1)
        verify_slots = self._recipe.target_forward.verify_slots(self._recipe.batch_size).detach().cpu()
        if accepted_count.numel() != verify_slots.shape[0]:
            raise AssertionError("MTP accepted count does not match target verify batch")
        rejected: list[torch.Tensor] = []
        for request, count_value in enumerate(accepted_count.tolist()):
            committed_width = int(count_value) + 1
            if committed_width < 0 or committed_width > verify_slots.shape[1]:
                raise AssertionError(f"MTP accepted count is outside target verify width for request={request}")
            suffix = verify_slots[request, committed_width:]
            rejected.append(suffix[suffix.ge(0)].contiguous())
        self._rejected_target_slots = tuple(rejected)
        self._expected_next_base_positions = output.next_state.positions.detach().cpu().reshape(-1).clone()

    @torch.inference_mode()
    def run_reference(
        self,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor | None,
        draft_input_embedding: torch.Tensor | None,
        draft_topk_indices: torch.Tensor | None,
    ) -> _ReferenceResult:
        initial = self.snapshot()
        recipe = self._recipe
        eager_recipe = MtpGraphRecipe(
            recipe.draft_forward.create_eager_reference(),
            recipe.draft_logits,
            recipe.target_forward.create_eager_reference(),
            recipe.target_logits,
            batch_size=recipe.batch_size,
            speculative_tokens=recipe.speculative_tokens,
            vocab_size=recipe.vocab_size,
            device=recipe.device,
            kv_seq_lens=kv_seq_lens,
            draft_activate=recipe.draft_activate,
            target_activate=recipe.target_activate,
            draft_sampling=recipe.draft_sampling,
            target_sampling=recipe.target_sampling,
            sampling_random_inputs=recipe.sampling_random_inputs,
            trace_intermediates=recipe.trace_intermediates,
        )
        # Preserve caller-owned input views, including previous graph outputs.
        inputs = tuple(
            value.clone() if value is not None else None
            for value in (seed_token_ids, base_positions, kv_seq_lens, draft_input_embedding, draft_topk_indices)
        )
        try:
            output = MtpAclGraphRunner(eager_recipe, backend="eager").execute(*inputs)
            result = _ReferenceResult(self.snapshot(), _output_snapshot(output))
        finally:
            initial.restore()
        return result

    def compare(self, output: SpeculativeExecutionOutput, reference: _ReferenceResult) -> None:
        actual_payload = self.snapshot()
        actual_payload.assert_matches(reference.payload)
        actual_output = _output_snapshot(output)
        if actual_output.keys() != reference.output.keys():
            raise AssertionError("MTP KV oracle output fields changed")
        for name, actual in actual_output.items():
            torch.testing.assert_close(
                actual,
                reference.output[name],
                rtol=0,
                atol=0,
                msg=f"MTP eager/graph output mismatch at {name}",
            )
        self._checks += 1
        trace_fields = (
            "draft_hidden",
            "draft_logits",
            "draft_topk_indices",
            "target_logits",
            "target_topk_indices",
        )
        trace_enabled = all(name in actual_output for name in trace_fields)
        logger.info(
            "MTP KV payload oracle passed: pid=%s iteration=%s tensors=%s accepted_count=%s "
            "intermediates=%s exact=true",
            os.getpid(),
            self._checks,
            actual_payload.tensor_count,
            reference.output["accepted_count"].tolist(),
            trace_enabled,
        )
