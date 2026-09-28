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
#include <string>

#include "core/runtime/worker_impl.h"

namespace xllm {
using WorkerPluginCreator =
    std::unique_ptr<WorkerImpl> (*)(const ParallelArgs&,
                                    const torch::Device&,
                                    const runtime::Options&,
                                    WorkerType);

// Register during application startup. Selection/construction happen once per
// Worker; no registry access or lock exists in the decode path.
void register_worker_plugin(std::string name, WorkerPluginCreator creator);
std::unique_ptr<WorkerImpl> create_worker_plugin(
    const std::string& name,
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    WorkerType worker_type);

// Pure startup policy, also usable without loading a model/device.
std::string select_worker_plugin(const std::string& requested,
                                 bool mtp_enabled,
                                 bool python_npu,
                                 bool unified_enabled);
}  // namespace xllm
