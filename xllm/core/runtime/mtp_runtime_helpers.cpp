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

#include "core/runtime/mtp_runtime_helpers.h"

#include <glog/logging.h>

#include "core/runtime/llm_worker_impl.h"
#include "core/util/slice.h"

namespace xllm::mtp_detail {
void broadcast_tokens_in_group(torch::Tensor& tokens,
                               ProcessGroup* process_group,
                               int32_t root_rank) {
  if (process_group == nullptr || process_group->world_size() <= 1 ||
      !tokens.defined()) {
    return;
  }
  tokens = tokens.contiguous();
  process_group->broadcast(tokens, root_rank);
}

bool should_broadcast_spec_tokens(const ParallelArgs& parallel_args,
                                  bool enable_spec_token_broadcast,
                                  bool all_greedy_sample) {
  const bool use_orthogonal_cp_consensus =
      parallel_args.cp_size() > 1 && parallel_args.tp_group_ != nullptr &&
      parallel_args.cp_group_ != nullptr &&
      parallel_args.cp_group_ != parallel_args.tp_group_;
  return use_orthogonal_cp_consensus ||
         (enable_spec_token_broadcast && !all_greedy_sample);
}

void broadcast_spec_tokens(torch::Tensor& tokens,
                           const ParallelArgs& parallel_args) {
  // DeepSeek-V4 TORCH publishes orthogonal TP and CP groups. Other backends
  // retain their existing single-group speculative broadcast behavior.
  ProcessGroup* tp_group = parallel_args.tp_group_ != nullptr
                               ? parallel_args.tp_group_
                               : parallel_args.process_group_;
  const bool use_orthogonal_cp_consensus =
      parallel_args.cp_size() > 1 && parallel_args.tp_group_ != nullptr &&
      parallel_args.cp_group_ != nullptr &&
      parallel_args.cp_group_ != parallel_args.tp_group_;
  if (!use_orthogonal_cp_consensus) {
    broadcast_tokens_in_group(tokens, tp_group);
    return;
  }

  broadcast_tokens_in_group(tokens, parallel_args.tp_group_);
  if (parallel_args.cp_group_ != parallel_args.tp_group_) {
    broadcast_tokens_in_group(tokens, parallel_args.cp_group_);
  }
}

void record_metadata_ready_event(Stream& stream, ForwardInput& input) {
  input.metadata_ready_event = stream.record_event_or_sync();
}

void finish_metadata_prepare(Stream& stream, ForwardInput& input) {
  record_metadata_ready_event(stream, input);
}

void record_current_metadata_ready_event(ForwardInput& input, Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait speculative metadata ready event";
  record_metadata_ready_event(stream, input);
}

void record_output_ready_event(ForwardOutput& output, Stream& stream) {
  StreamEventPtr event = stream.record_event();
  if (event == nullptr) {
    const int32_t ret = stream.synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize MTP compute stream, ret=" << ret;
  }
  output.ready_event = event;
}

void finalize_output_on_stream(ForwardOutput& output,
                               Stream& stream,
                               bool allow_async) {
  if (allow_async) {
    record_output_ready_event(output, stream);
    return;
  }
  const int32_t ret = stream.synchronize();
  CHECK_EQ(ret, 0) << "failed to synchronize MTP compute stream, ret=" << ret;
  output.retained_inputs.clear();
}

void clear_ready_events(ForwardInput& input) {
  input.metadata_ready_event.reset();
}

std::optional<ForwardOutput> run_worker_no_sync_impl(
    WorkerImpl& worker,
    const ForwardInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    ForwardInput& processed_input) {
  worker.prepare_work_before_execute_on_stream(
      input,
      processed_input,
      prepare_stream,
      /*record_ready_event=*/&prepare_stream != &compute_stream);
  if (auto* llm_worker = dynamic_cast<LLMWorkerImpl*>(&worker);
      llm_worker != nullptr) {
    return llm_worker->execute_no_sync_on_stream(
        processed_input, compute_stream, /*record_ready_event=*/false);
  }
  return worker.execute_no_sync_on_stream(processed_input, compute_stream);
}

void check_mtp_decode_states(
    const std::vector<EmbeddingCache::DecodeState>& states,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& token_ids_host,
    bool allow_overlap_fake_token) {
  CHECK(!request_ids.empty())
      << "MTP decode requires request ids for bootstrap state validation";
  CHECK_EQ(states.size(), request_ids.size())
      << "MTP decode request/state count mismatch";
  CHECK_GE(token_ids_host.numel(), static_cast<int64_t>(states.size()))
      << "MTP decode token/state count mismatch";

  Slice<int32_t> token_ids = {token_ids_host.data_ptr<int32_t>(),
                              static_cast<size_t>(token_ids_host.numel())};
  for (int32_t i = 0; i < static_cast<int32_t>(states.size()); ++i) {
    const EmbeddingCache::DecodeState& state = states[i];
    const int32_t token_id = token_ids[i];
    CHECK(state.valid) << "MTP decode missing target state, request_id="
                       << request_ids[i];
    CHECK_EQ(state.request_id, request_ids[i])
        << "MTP decode target state request mismatch";
    CHECK(state.embedding.defined())
        << "MTP decode target state embedding is undefined, request_id="
        << request_ids[i];
    if (token_id < 0) {
      CHECK(allow_overlap_fake_token)
          << "MTP decode fake token is only allowed with schedule overlap, "
          << "request_id=" << request_ids[i];
      CHECK_GE(state.token_id, 0)
          << "MTP decode fake token requires a valid cached target token, "
          << "request_id=" << request_ids[i];
      continue;
    }
    CHECK_EQ(state.token_id, token_id)
        << "MTP decode target state token mismatch, request_id="
        << request_ids[i];
  }
}
}  // namespace xllm::mtp_detail
