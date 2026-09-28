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
// Legacy entry retained for other model backends and Eagle3. Python Unified
// has its own sibling worker and does not instantiate this adapter.
class MTPWorkerImpl : public MtpRuntime {
 public:
  using MtpRuntime::MtpRuntime;
  ~MTPWorkerImpl() override;

 protected:
  std::optional<ForwardOutput> step_decode(const ForwardInput& input) override;
};
}  // namespace xllm
