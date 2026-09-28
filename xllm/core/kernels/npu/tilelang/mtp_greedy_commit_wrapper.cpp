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

#include <cstdint>
#include <limits>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_MTP_GREEDY_COMMIT_REGISTRY_INC
#error "XLLM_TL_MTP_GREEDY_COMMIT_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {
#include XLLM_TL_MTP_GREEDY_COMMIT_REGISTRY_INC
constexpr int64_t kMaxOffset = std::numeric_limits<int32_t>::max();
}  // namespace

std::tuple<torch::Tensor, torch::Tensor> mtp_greedy_commit(
    const torch::Tensor& draft,
    const torch::Tensor& target,
    const torch::Tensor& hidden) {
  CHECK(draft.device().is_privateuseone());
  CHECK_EQ(target.device(), draft.device());
  CHECK_EQ(draft.scalar_type(), torch::kInt64);
  CHECK_EQ(target.scalar_type(), torch::kInt64);
  CHECK_EQ(draft.dim(), 2);
  CHECK_EQ(target.dim(), 2);
  CHECK(draft.is_contiguous());
  CHECK(target.is_contiguous());
  const int64_t batch = draft.size(0);
  const int64_t steps = draft.size(1);
  CHECK_GT(batch, 0);
  CHECK_GT(steps, 0);
  CHECK_LT(steps, kMaxOffset);
  CHECK_LE(batch, kMaxOffset / (steps + 1));
  CHECK_EQ(target.size(0), batch);
  CHECK_EQ(target.size(1), steps + 1);
  CHECK_EQ(hidden.device(), draft.device());
  CHECK(hidden.is_contiguous());
  CHECK_EQ(hidden.dim(), 2);
  CHECK_EQ(hidden.size(0), batch * (steps + 1));
  CHECK(hidden.scalar_type() == torch::kFloat16 ||
        hidden.scalar_type() == torch::kBFloat16 ||
        hidden.scalar_type() == torch::kFloat32);
  const int64_t hidden_size = hidden.size(1);
  CHECK_GT(hidden_size, 0);
  CHECK_LE(hidden_size, kMaxOffset / (batch * (steps + 1)));
  torch::Tensor compact =
      torch::empty({batch * 2, hidden_size}, hidden.options());
  const int64_t token_bytes = batch * (steps + 1) * sizeof(int64_t);
  torch::Tensor state = torch::empty(
      {token_bytes + batch * static_cast<int64_t>(sizeof(int32_t))},
      draft.options().dtype(torch::kUInt8));
  const auto spec =
      make_mtp_greedy_commit_specialization(MtpGreedyCommitHiddenBits{
          static_cast<int32_t>(hidden.element_size() * 8)});
  const auto* entry = find_mtp_greedy_commit_kernel_entry(spec);
  CHECK(entry != nullptr) << available_mtp_greedy_commit_variant_keys();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(draft.device().index()).stream();
  auto* state_bytes = state.data_ptr<uint8_t>();
  entry->fn(reinterpret_cast<uint8_t*>(draft.data_ptr()),
            reinterpret_cast<uint8_t*>(target.data_ptr()),
            state_bytes,
            state_bytes + token_bytes,
            reinterpret_cast<uint8_t*>(hidden.data_ptr()),
            reinterpret_cast<uint8_t*>(compact.data_ptr()),
            static_cast<int32_t>(batch),
            static_cast<int32_t>(steps),
            static_cast<int32_t>(hidden_size),
            stream);
  return {state, compact};
}
}  // namespace xllm::kernel::npu::tilelang
