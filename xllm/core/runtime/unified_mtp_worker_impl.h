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
#include "core/runtime/mtp_worker_impl.h"

namespace xllm {
class UnifiedMtpExecutor;
bool supports_unified_mtp_request(const LlmForwardInput& input, bool adaptive);
// Unified MTP worker. Shared model/cache lifetime is supplied by
// MTPWorkerImpl<LlmForwardInput>; unsupported requests use its legacy decode
// path.
class UnifiedMtpWorkerImpl : public MTPWorkerImpl<LlmForwardInput> {
 public:
  UnifiedMtpWorkerImpl(const ParallelArgs& parallel_args,
                       const torch::Device& device,
                       const runtime::Options& options,
                       WorkerType worker_type);
  ~UnifiedMtpWorkerImpl() override;
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;
  bool uses_worker_task_pipeline() const override { return true; }
  ::xllm::Status create_task_pipeline(
      std::unique_ptr<TaskExecutionPipeline>& output) override;
  void prepare_task_pipeline_input(const LlmForwardInput& input,
                                   LlmForwardInput& prepared) override;
  std::optional<ForwardOutput> execute_task_pipeline(
      const LlmForwardInput& prepared) override;
  std::optional<ForwardOutput> step(const LlmForwardInput& input) override;

 protected:
  std::optional<ForwardOutput> step_decode(
      const LlmForwardInput& input) override;
  bool supports_unified_python_mtp_graph(const LlmForwardInput& input) const;
  void clear_unified_device_state();
  UnifiedMtpExecutor& unified_executor();

 private:
  bool supports_unified_configuration() const;
  bool unified_graph_capable_ = false;
  // Launch-thread-only JSON overlap context. Task results remain owned by
  // the shared pipeline; only a completed token copy survives slot reuse.
  ForwardOutput task_json_output_;
  std::vector<std::string> task_json_sample_sequence_ids_;
  std::optional<ForwardOutput> step_unified(const LlmForwardInput& input);
  std::optional<ForwardOutput> run_unified_python_mtp_graph(
      const LlmForwardInput& input,
      const LlmForwardInput& current_draft_input,
      int32_t num_speculative_tokens);
  std::unique_ptr<UnifiedMtpExecutor> unified_executor_;
};
}  // namespace xllm
