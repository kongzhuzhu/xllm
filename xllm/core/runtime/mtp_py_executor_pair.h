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

#pragma once

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xllm {

class PyExecutorImpl;

namespace detail {

struct MtpPyGraphOutput {
  torch::Tensor accepted_ids;
  torch::Tensor accepted_mask;
  torch::Tensor accepted_count;
  torch::Tensor committed_tokens;
  torch::Tensor next_token_ids;
  torch::Tensor next_positions;
  torch::Tensor next_kv_seq_lens;
  torch::Tensor next_embeddings;
  torch::Tensor next_topk_indices;
  torch::Tensor target_embeddings;
  torch::Tensor target_probs;
  torch::Tensor committed_log_probs;
  torch::Tensor target_top_log_probs;
  torch::Tensor target_top_tokens;
};

class __attribute__((visibility("hidden"))) MtpPyExecutorPair final {
 public:
  static std::unique_ptr<MtpPyExecutorPair> create(
      PyExecutorImpl& target_executor,
      PyExecutorImpl& draft_executor,
      const std::vector<pybind11::object>& draft_metadata,
      const pybind11::object& target_metadata,
      const torch::Tensor& repair_token_ids,
      const torch::Tensor& kv_seq_lens,
      int32_t batch_size,
      int32_t speculative_tokens,
      int64_t vocab_size,
      const pybind11::object& draft_sampling_plan,
      const pybind11::object& target_sampling_plan,
      bool target_step_major_layout);

  ~MtpPyExecutorPair();

  MtpPyGraphOutput capture_and_execute(
      const torch::Tensor& seed_token_ids,
      const torch::Tensor& base_positions,
      const torch::Tensor& kv_seq_lens,
      const torch::Tensor& draft_input_embedding,
      const torch::Tensor& draft_topk_indices = torch::Tensor());

  bool can_update_metadata(const std::vector<pybind11::object>& draft_metadata,
                           const pybind11::object& target_metadata) const;

  bool can_update_sampling_plans(
      const pybind11::object& draft_sampling_plan,
      const pybind11::object& target_sampling_plan) const;

  std::string metadata_key(const std::vector<pybind11::object>& draft_metadata,
                           const pybind11::object& target_metadata,
                           const pybind11::object& draft_sampling_plan,
                           const pybind11::object& target_sampling_plan) const;

  MtpPyGraphOutput update_and_execute(
      const std::vector<pybind11::object>& draft_metadata,
      const pybind11::object& target_metadata,
      const torch::Tensor& repair_token_ids,
      const torch::Tensor& seed_token_ids,
      const torch::Tensor& base_positions,
      const torch::Tensor& kv_seq_lens,
      const torch::Tensor& draft_input_embedding,
      const torch::Tensor& draft_topk_indices,
      const pybind11::object& draft_sampling_plan,
      const pybind11::object& target_sampling_plan);

 private:
  explicit MtpPyExecutorPair(pybind11::object runner);

  pybind11::object runner_;
};

}  // namespace detail
}  // namespace xllm
