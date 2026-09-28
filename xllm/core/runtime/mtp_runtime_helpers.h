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

namespace xllm {
class WorkerImpl;
}

namespace xllm::mtp_detail {
void broadcast_tokens_in_group(torch::Tensor& tokens,
                               ProcessGroup* process_group,
                               int32_t root_rank = 0);

bool should_broadcast_spec_tokens(const ParallelArgs& parallel_args,
                                  bool enable_spec_token_broadcast,
                                  bool all_greedy_sample);

void broadcast_spec_tokens(torch::Tensor& tokens,
                           const ParallelArgs& parallel_args);

void record_metadata_ready_event(Stream& stream, ForwardInput& input);

void finish_metadata_prepare(Stream& stream, ForwardInput& input);

void record_current_metadata_ready_event(ForwardInput& input, Stream& stream);

void record_output_ready_event(ForwardOutput& output, Stream& stream);

void finalize_output_on_stream(ForwardOutput& output,
                               Stream& stream,
                               bool allow_async);

void clear_ready_events(ForwardInput& input);

std::optional<ForwardOutput> run_worker_no_sync_impl(
    WorkerImpl& worker,
    const ForwardInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    ForwardInput& processed_input);

void check_mtp_decode_states(
    const std::vector<EmbeddingCache::DecodeState>& states,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& token_ids_host,
    bool allow_overlap_fake_token);
}  // namespace xllm::mtp_detail
