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

"""Commit the greedy prefix and count directly into one output allocation."""

import tilelang.language as T
from tilelang import tvm

MTP_GREEDY_COMMIT_PASS_CONFIGS = {
    # Every GM access is bounded by explicit row/column loop limits. Avoid
    # redundant predicated loads, which this Ascend codegen emits inside
    # LocalTensor::SetValue as invalid nested C++ declarations.
    "tl.disable_safe_memory_legalize": True,
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def build_mtp_greedy_commit_kernel(hidden_bits: int = 16) -> tvm.tir.PrimFunc:
    if hidden_bits not in (16, 32):
        raise ValueError("Unified MTP hidden rows must contain 16-bit or 32-bit elements")
    hidden_dtype = f"uint{hidden_bits}"

    @T.prim_func
    def mtp_greedy_commit(
        draft_handle: T.handle,
        target_handle: T.handle,
        committed_handle: T.handle,
        count_handle: T.handle,
        hidden_handle: T.handle,
        compact_handle: T.handle,
        batch_size: T.int32,
        speculative_tokens: T.int32,
        hidden_size: T.int32,
    ):
        draft = T.match_buffer(draft_handle, (batch_size * speculative_tokens,), "int64")
        target = T.match_buffer(target_handle, (batch_size * (speculative_tokens + 1),), "int64")
        committed = T.match_buffer(committed_handle, (batch_size * (speculative_tokens + 1),), "int64")
        counts = T.match_buffer(count_handle, (T.max(batch_size, 1),), "int32")
        hidden = T.match_buffer(hidden_handle, (batch_size * (speculative_tokens + 1) * hidden_size,), hidden_dtype)
        compact = T.match_buffer(compact_handle, (batch_size * 2 * hidden_size,), hidden_dtype)
        with T.Kernel(1, is_npu=True) as (_, vid), T.Scope("V"):
            # One writer avoids inter-core stores to the same cache line for
            # the small B=1/3/8 decode batches. Only two selected hidden rows
            # are copied.
            if vid == 0:
                token_tile = T.alloc_ub((32,), "int64")
                count_tile = T.alloc_ub((32,), "int32")
                hidden_tile = T.alloc_ub((8192,), hidden_dtype)
                accepted = T.alloc_var("int32")
                for group in T.serial(T.ceildiv(batch_size, 32)):
                    row_count = T.min(32, batch_size - group * 32)
                    for lane in T.serial(row_count):
                        row = group * 32 + lane
                        accepted = speculative_tokens
                        for step in T.serial(speculative_tokens):
                            if draft[row * speculative_tokens + step] != target[row * (speculative_tokens + 1) + step]:
                                accepted = T.min(accepted, step)
                        count_tile[lane] = accepted
                        for chunk in T.serial(T.ceildiv(speculative_tokens + 1, 32)):
                            width = T.min(32, speculative_tokens + 1 - chunk * 32)
                            for offset in T.serial(width):
                                column = chunk * 32 + offset
                                token_tile[offset] = T.Cast("int64", -1)
                                if column <= accepted:
                                    token_tile[offset] = target[row * (speculative_tokens + 1) + column]
                            start = row * (speculative_tokens + 1) + chunk * 32
                            T.copy(token_tile, committed[start : start + width])
                        for pair_row in T.serial(2):
                            source_row = row * (speculative_tokens + 1) + T.max(accepted + pair_row - 1, 0)
                            for chunk in T.serial(T.ceildiv(hidden_size, 8192)):
                                width = T.min(8192, hidden_size - chunk * 8192)
                                source = source_row * hidden_size + chunk * 8192
                                destination = (row * 2 + pair_row) * hidden_size + chunk * 8192
                                T.copy(hidden[source : source + width], hidden_tile)
                                T.copy(hidden_tile, compact[destination : destination + width])
                    T.copy(count_tile, counts[group * 32 : group * 32 + row_count])

    return mtp_greedy_commit
