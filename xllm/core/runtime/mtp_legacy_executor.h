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
#include <memory>

#include "core/framework/speculative/mtp_json_object_state.h"
#include "core/runtime/mtp_runtime.h"

namespace xllm {
namespace detail {
class NpuJsonDraftTokenHandoff;
}
// On-demand compatibility execution. Borrows the runtime's one target/draft
// pair and KV/cache; never constructs a Worker or a second model.
class MtpLegacyExecutor final {
 public:
  explicit MtpLegacyExecutor(MtpRuntime& runtime);
  ~MtpLegacyExecutor();
  void retire_prelaunch();
  void prepare_validate_inputs(const ForwardInput& inputs,
                               ForwardInput& validate_inputs,
                               bool static_graph_tasks_prepared = false,
                               bool record_ready_event = true);

  void prepare_validate_inputs(const ForwardInput& inputs,
                               ForwardInput& validate_inputs,
                               const std::vector<int32_t>& per_seq_val_tokens);

  std::optional<ForwardOutput> step_decode(const ForwardInput& inputs);
  std::optional<ForwardOutput> step_empty(const ForwardInput& inputs);

 private:
  void prepare_draft_inputs(const ForwardInput& inputs,
                            ForwardInput& draft_inputs,
                            int32_t position_offset);

  bool pending_draft_context_matches(const ForwardInput& input) const;

  void record_validate_metrics(
      SampleOutput& validate_output,
      int32_t num_speculative_tokens,
      const std::vector<int32_t>* pruned_prefix_lengths = nullptr) const;

  void write_target_context_to_cache(const ForwardInput& input,
                                     const SampleOutput& validate_output,
                                     int32_t num_speculative_tokens);

  bool can_prelaunch_next_first_draft(const ForwardInput& input) const;

  void prepare_next_first_draft_template(const ForwardInput& input,
                                         ForwardInput& combined_input);

  void submit_pending_first_draft(const ForwardInput& batch_identity_input,
                                  ForwardInput draft_input);

  bool prepare_static_mtp_graph_tasks_before_final_draft(
      const ForwardInput& input);

  void enqueue_next_first_draft(const ForwardInput& input,
                                const SampleOutput& validate_output,
                                const torch::Tensor& base_positions,
                                const torch::Tensor& base_kv_seq_lens,
                                ForwardInput combined_input);

  std::optional<ForwardOutput> run_adaptive_validate(
      const ForwardInput& input,
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input,
      int32_t num_speculative_tokens);

  bool supports_combined_first_draft_execution() const;

  SampleOutput validate(const SamplingParameters& sampling_params,
                        const std::vector<ForwardOutput>& draft_outputs,
                        const ForwardOutput& target_output,
                        int32_t num_speculative_tokens,
                        const std::vector<int32_t>* pruned_prefix_lengths,
                        const torch::Tensor& target_filter_mask,
                        const torch::Tensor& target_filter_bitmask,
                        const std::vector<uint8_t>& invalid_draft);

  void fill_validate_input_from_draft_outputs(
      const ForwardInput& input,
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input,
      const std::vector<int32_t>& per_seq_val_tokens,
      const detail::JsonDraftValidationScratch* json_scratch,
      Stream& compute_stream);

  std::optional<ForwardOutput> run_validate(
      const ForwardInput& input,
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input,
      int32_t num_speculative_tokens,
      const std::vector<int32_t>* pruned_prefix_lengths,
      const detail::JsonDraftValidationScratch* json_scratch);

  std::optional<ForwardOutput> run_validate(
      const ForwardInput& input,
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input,
      int32_t num_speculative_tokens,
      const std::vector<int32_t>& per_seq_val_tokens,
      const std::vector<int32_t>* pruned_prefix_lengths,
      const detail::JsonDraftValidationScratch* json_scratch);

  bool can_use_combined_first_draft() const;
  SampleOutput validate(const SamplingParameters& sampling_params,
                        const DraftProposal& draft_proposal,
                        const ForwardOutput& target_output,
                        int32_t num_speculative_tokens,
                        const std::vector<int32_t>* pruned_prefix_lengths,
                        const torch::Tensor& target_filter_mask,
                        const torch::Tensor& target_filter_bitmask,
                        const std::vector<uint8_t>& invalid_draft);
  MtpRuntime& runtime_;
  struct PendingDraftContext {
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    std::vector<int32_t> dp_global_token_nums;
    std::vector<int32_t> dp_global_sequence_nums;
    std::vector<int32_t> raw_dp_global_token_nums;
    std::vector<uint64_t> dp_global_batch_generations;
    std::optional<ForwardOutput> output;
    ForwardInput prepared_input;
  };
  PendingDraftContext pending_draft_context_;
#if defined(USE_NPU)
  // Stable-address sources consumed by the target ACL graph's leading input
  // update. The existing H2D preparation overlaps with the final draft, so no
  // extra graph-external D2D launch is introduced.
  torch::Tensor spec_verify_attention_host_buffer_;
  torch::Tensor spec_verify_attention_device_buffer_;
  uint64_t spec_verify_attention_buffer_capacity_ = 0;
  std::shared_ptr<int> spec_verify_attention_buffer_owner_ =
      std::make_shared<int>(0);

  // Stable validate-sampling controls for the common single-sequence greedy
  // path.  Their values depend on speculative width, not tensor-parallel
  // topology, and are rebuilt only when that width changes.
  torch::Tensor mtp_validate_greedy_indices_;
  torch::Tensor mtp_validate_greedy_do_sample_;
  std::unique_ptr<detail::NpuJsonDraftTokenHandoff> json_draft_token_handoff_;
#endif
};
}  // namespace xllm
