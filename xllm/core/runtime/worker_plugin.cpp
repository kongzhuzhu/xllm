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

#include "core/runtime/worker_plugin.h"

#include <glog/logging.h>

#include <mutex>
#include <unordered_map>
#include <utility>

#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/runtime/mtp_worker_impl.h"
#include "core/runtime/unified_mtp_worker_impl.h"

namespace xllm {
namespace {
template <typename WorkerClass>
std::unique_ptr<WorkerImpl> make_mtp_worker(const ParallelArgs& args,
                                            const torch::Device& device,
                                            const runtime::Options& options,
                                            WorkerType type) {
  CHECK(options.enable_speculative_decode() &&
        SpeculativeConfig::is_mtp_algorithm(options.speculative_algorithm()))
      << "MTP worker plugins require the MTP speculative algorithm";
  CHECK(!options.enable_task_pipeline())
      << "MTP worker plugins do not use the task pipeline";
  return std::make_unique<WorkerClass>(args, device, options, type);
}
std::unique_ptr<WorkerImpl> make_unified_worker(const ParallelArgs& args,
                                                const torch::Device& device,
                                                const runtime::Options& options,
                                                WorkerType type) {
#if defined(USE_NPU)
  CHECK(device.is_privateuseone() && type == WorkerType::LLM &&
        ModelConfig::is_python_model_impl(
            ModelConfig::get_instance().model_impl()))
      << "unified_mtp worker plugin requires Python NPU LLM";
  return make_mtp_worker<UnifiedMtpWorkerImpl>(args, device, options, type);
#else
  LOG(FATAL) << "unified_mtp worker plugin requires an NPU build";
  return nullptr;
#endif
}
struct WorkerPlugins {
  std::mutex mutex;
  std::unordered_map<std::string, WorkerPluginCreator> creators = {
      {"unified_mtp", make_unified_worker},
      {"legacy_mtp", make_mtp_worker<MTPWorkerImpl>}};
};
WorkerPlugins& plugins() {
  static WorkerPlugins registry;
  return registry;
}
}  // namespace

void register_worker_plugin(std::string name, WorkerPluginCreator creator) {
  CHECK(!name.empty());
  CHECK(creator != nullptr);
  auto& registry = plugins();
  std::lock_guard<std::mutex> lock(registry.mutex);
  CHECK(registry.creators.emplace(std::move(name), creator).second)
      << "Duplicate worker plugin registration";
}

std::unique_ptr<WorkerImpl> create_worker_plugin(
    const std::string& name,
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    WorkerType type) {
  WorkerPluginCreator creator = nullptr;
  {
    auto& registry = plugins();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto it = registry.creators.find(name);
    CHECK(it != registry.creators.end()) << "Unknown worker plugin: " << name;
    creator = it->second;
  }
  // Construct after releasing the startup registry lock.
  auto worker = creator(parallel_args, device, options, type);
  CHECK(worker != nullptr) << "Worker plugin returned no worker: " << name;
  return worker;
}

std::string select_worker_plugin(const std::string& requested,
                                 bool mtp_enabled,
                                 bool python_npu,
                                 bool unified_enabled) {
  if (!requested.empty()) {
    return requested;
  }
  return mtp_enabled && python_npu && unified_enabled ? "unified_mtp" : "";
}
}  // namespace xllm
