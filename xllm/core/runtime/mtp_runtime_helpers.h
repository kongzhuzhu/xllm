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

#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/speculative/embedding_cache.h"
#include "core/platform/stream.h"
#include "core/runtime/forward_params.h"

namespace xllm::mtp_detail {
void check_mtp_decode_states(
    const std::vector<EmbeddingCache::DecodeState>& states,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& token_ids_host,
    bool allow_overlap_fake_token);
}  // namespace xllm::mtp_detail
