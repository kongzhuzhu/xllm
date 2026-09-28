/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <limits>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_MTP_SPARSE_METADATA_UPDATE_REGISTRY_INC
#error "XLLM_TL_MTP_SPARSE_METADATA_UPDATE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {
#include XLLM_TL_MTP_SPARSE_METADATA_UPDATE_REGISTRY_INC

constexpr int64_t kMaxOffset = std::numeric_limits<int32_t>::max();
}  // namespace

void mtp_sparse_metadata_update(const torch::Tensor& block_table,
                                const torch::Tensor& base_positions,
                                const torch::Tensor& base_kv_lengths,
                                const torch::Tensor& first_kv_lengths,
                                const torch::Tensor& first_slots,
                                const torch::Tensor& layout,
                                const torch::Tensor& draft_arena,
                                const torch::Tensor& target_arena,
                                const torch::Tensor& position_arena,
                                int64_t speculative_tokens,
                                int64_t table_capacity,
                                int64_t block_size,
                                bool target_step_major) {
  CHECK(block_table.device().is_privateuseone());
  CHECK_EQ(block_table.dim(), 2);
  CHECK_EQ(block_table.scalar_type(), torch::kInt32);
  CHECK_EQ(block_table.stride(1), 1);
  const int64_t batch = block_table.size(0);
  const int64_t columns = block_table.size(1);
  CHECK_GT(batch, 0);
  CHECK_GT(columns, 0);
  CHECK_GT(speculative_tokens, 0);
  CHECK_LE(speculative_tokens, (kMaxOffset - 3) / 3);
  CHECK_GE(table_capacity, columns);
  CHECK_EQ(table_capacity % 32, 0);
  CHECK_GT(block_size, 0);
  CHECK_LE(block_size, kMaxOffset);
  CHECK_LE(table_capacity, kMaxOffset);
  CHECK_LE(batch, kMaxOffset / (2 * speculative_tokens + 3));
  CHECK_GE(block_table.stride(0), columns);
  CHECK_LE(block_table.stride(0), kMaxOffset / batch);
  CHECK_EQ(base_positions.numel(), batch);
  CHECK(base_positions.scalar_type() == torch::kInt32 ||
        base_positions.scalar_type() == torch::kInt64);
  CHECK(base_positions.is_contiguous());
  CHECK_EQ(base_positions.device(), block_table.device());
  CHECK_EQ(base_kv_lengths.numel(), batch);
  CHECK_EQ(first_kv_lengths.numel(), batch * 2);
  CHECK_EQ(first_slots.numel(), batch * 2);
  CHECK_EQ(layout.numel(), (speculative_tokens + 1) * 3);
  CHECK_EQ(position_arena.numel(), (2 * speculative_tokens + 3) * batch);
  CHECK_EQ(position_arena.scalar_type(), torch::kInt64);
  CHECK_EQ(position_arena.device(), block_table.device());
  CHECK(position_arena.is_contiguous());
  for (const torch::Tensor* tensor : {&base_kv_lengths,
                                      &first_kv_lengths,
                                      &first_slots,
                                      &layout,
                                      &draft_arena,
                                      &target_arena}) {
    CHECK_EQ(tensor->device(), block_table.device());
    CHECK_EQ(tensor->scalar_type(), torch::kInt32);
    CHECK(tensor->is_contiguous());
    CHECK_GT(tensor->numel(), 0);
    CHECK_LE(tensor->numel(), kMaxOffset);
  }
  // Layout offsets and the final destinations are checked once by the
  // graph binding owner. They are immutable for the lifetime of this variant.
  const auto spec = make_mtp_sparse_metadata_update_specialization(
      MtpSparseMetadataUpdatePositionBits{
          static_cast<int32_t>(base_positions.element_size() * 8)});
  const auto* entry = find_mtp_sparse_metadata_update_kernel_entry(spec);
  CHECK(entry != nullptr)
      << available_mtp_sparse_metadata_update_variant_keys();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(block_table.device().index()).stream();
  entry->fn(reinterpret_cast<uint8_t*>(block_table.data_ptr()),
            reinterpret_cast<uint8_t*>(base_positions.data_ptr()),
            reinterpret_cast<uint8_t*>(base_kv_lengths.data_ptr()),
            reinterpret_cast<uint8_t*>(first_kv_lengths.data_ptr()),
            reinterpret_cast<uint8_t*>(first_slots.data_ptr()),
            reinterpret_cast<uint8_t*>(layout.data_ptr()),
            reinterpret_cast<uint8_t*>(draft_arena.data_ptr()),
            reinterpret_cast<uint8_t*>(target_arena.data_ptr()),
            reinterpret_cast<uint8_t*>(position_arena.data_ptr()),
            static_cast<int32_t>(batch),
            static_cast<int32_t>(speculative_tokens),
            static_cast<int32_t>(columns),
            static_cast<int32_t>(block_table.stride(0)),
            static_cast<int32_t>(table_capacity),
            static_cast<int32_t>(block_size),
            static_cast<int32_t>(target_step_major),
            static_cast<int32_t>(draft_arena.numel()),
            static_cast<int32_t>(target_arena.numel()),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
