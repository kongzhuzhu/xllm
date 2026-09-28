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

#include "core/runtime/mtp_worker_impl.h"

#include "core/runtime/mtp_legacy_executor.h"

namespace xllm {
MTPWorkerImpl::~MTPWorkerImpl() { drain_pending_execution(); }
std::optional<ForwardOutput> MTPWorkerImpl::step_decode(
    const ForwardInput& input) {
  return legacy_executor().step_decode(input);
}
}  // namespace xllm
