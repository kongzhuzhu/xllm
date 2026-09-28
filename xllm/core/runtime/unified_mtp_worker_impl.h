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
#include "core/runtime/mtp_runtime.h"

namespace xllm {
class UnifiedMtpExecutor;
bool supports_unified_mtp_request(const ForwardInput& input, bool adaptive);
// Independent plugin entry. Shared model/cache lifetime is supplied by
// MtpRuntime; only unsupported requests enter its lazy compatibility executor.
// Virtual dispatch points also allow model-specific plugin extensions.
class UnifiedMtpWorkerImpl : public MtpRuntime {
 public:
  UnifiedMtpWorkerImpl(const ParallelArgs& parallel_args,
                       const torch::Device& device,
                       const runtime::Options& options,
                       WorkerType worker_type);
  ~UnifiedMtpWorkerImpl() override;
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;
  bool task_models_loaded() const override;
  bool uses_worker_task_pipeline() const override { return true; }
  std::optional<ForwardOutput> step(const ForwardInput& input) override;

 protected:
  std::optional<ForwardOutput> step_decode(const ForwardInput& input) override;
  bool supports_unified_python_mtp_graph(
      const ForwardInput& input) const override;
  void clear_unified_device_state() override;
  UnifiedMtpExecutor& unified_executor();

 private:
  bool supports_unified_configuration() const;
  bool unified_graph_capable_ = false;
  std::optional<ForwardOutput> step_unified(const ForwardInput& input);
  std::optional<ForwardOutput> run_unified_python_mtp_graph(
      const ForwardInput& input,
      const ForwardInput& current_draft_input,
      int32_t num_speculative_tokens);
  std::unique_ptr<UnifiedMtpExecutor> unified_executor_;
};
}  // namespace xllm
