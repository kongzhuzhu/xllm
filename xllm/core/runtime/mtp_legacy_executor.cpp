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

#include "core/runtime/mtp_legacy_executor.h"

#include <glog/logging.h>

#include "core/runtime/mtp_runtime.h"
#include "core/runtime/mtp_runtime_helpers.h"
#if defined(USE_NPU)
#include <acl/acl.h>
#include <c10/core/DeviceType.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>

#include "common/metrics.h"
#include "core/framework/block/block_utils.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/model/mtp_utils.h"
#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#endif

#include "core/framework/speculative/adaptive_pruning_helpers.h"
#include "core/framework/speculative/draft_extend_input.h"
#include "core/framework/speculative/mtp_async_input_builder.h"
#include "core/framework/speculative/mtp_async_state.h"
#include "core/framework/speculative/spec_input_builder.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/framework/speculative/speculative_profile_registry.h"
#include "core/layers/common/dsa_topk_share_plan.h"
#include "runtime/llm_worker_impl.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
using mtp_detail::broadcast_spec_tokens;
using mtp_detail::check_mtp_decode_states;
using mtp_detail::clear_ready_events;
using mtp_detail::finalize_output_on_stream;
using mtp_detail::finish_metadata_prepare;
using mtp_detail::record_current_metadata_ready_event;
using mtp_detail::record_metadata_ready_event;
using mtp_detail::run_worker_no_sync_impl;
using mtp_detail::should_broadcast_spec_tokens;
namespace {

bool has_active_dp_tokens(const ForwardInput& input) {
  const ParallelInput& parallel = input.input_params.parallel;
  const std::vector<int32_t>& token_nums =
      parallel.raw_dp_global_token_nums.empty()
          ? parallel.dp_global_token_nums
          : parallel.raw_dp_global_token_nums;
  return std::any_of(token_nums.begin(), token_nums.end(), [](int32_t count) {
    return count > 0;
  });
}

void wait_metadata_ready_event(const ForwardInput& input, Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait speculative metadata ready event";
}

#if defined(USE_NPU)
void clear_expanded_spec_verify_graph_input(ModelInputParams& input_params) {
  input_params.graph.use_expanded_decode_for_spec_verify_attention = false;
  input_params.graph.expanded_kv_seq_lens = torch::Tensor();
  input_params.graph.expanded_block_tables = torch::Tensor();
  input_params.graph.expanded_paged_kv_indptr = torch::Tensor();
  input_params.graph.expanded_paged_kv_indices = torch::Tensor();
  input_params.graph.expanded_paged_kv_last_page_len = torch::Tensor();
  input_params.graph.expanded_tiling_data = torch::Tensor();
  input_params.graph.expanded_kv_seq_lens_vec.clear();
}
#endif

#if defined(USE_NPU)
bool build_expanded_spec_verify_graph_host_input(
    ModelInputParams& input_params) {
  clear_expanded_spec_verify_graph_input(input_params);
  if (!input_params.is_spec_verify ||
      !input_params.meta.batch_forward_type.is_chunked_prefill()) {
    return false;
  }

  const std::vector<int32_t>& q_seq_lens =
      input_params.attention.host.q_seq_lens;
  const std::vector<int32_t>& kv_seq_lens =
      input_params.attention.host.kv_seq_lens;
  if (q_seq_lens.empty() || kv_seq_lens.empty()) {
    return false;
  }
  std::vector<int32_t> expanded_kv_seq_lens =
      layer::ExpandedDecodeMetadataBuilder::build_tokenwise_kv_seq_lens(
          q_seq_lens, kv_seq_lens);
  if (expanded_kv_seq_lens.empty()) {
    return false;
  }

  input_params.graph.use_expanded_decode_for_spec_verify_attention = true;
  input_params.graph.expanded_kv_seq_lens_vec = std::move(expanded_kv_seq_lens);
  return true;
}
#endif

#if defined(USE_NPU)
void bind_expanded_spec_verify_graph_input(ModelInputParams& input_params,
                                           const torch::Device& device,
                                           bool kv_lens_already_bound,
                                           int32_t block_size) {
  if (!input_params.graph.use_expanded_decode_for_spec_verify_attention) {
    return;
  }
  CHECK(input_params.attention.device.block_tables.defined())
      << "spec verify block tables must be rebuilt before graph input";
  const auto& q_seq_lens = input_params.attention.host.q_seq_lens;
  CHECK_GE(input_params.attention.device.block_tables.size(0),
           static_cast<int64_t>(q_seq_lens.size()))
      << "spec verify block table rows are fewer than sequences";
  std::vector<torch::Tensor> expanded_block_rows;
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(q_seq_lens.size());
       ++seq_idx) {
    for (int32_t token_idx = 0;
         token_idx < q_seq_lens[static_cast<size_t>(seq_idx)];
         ++token_idx) {
      expanded_block_rows.emplace_back(
          input_params.attention.device.block_tables.select(/*dim=*/0,
                                                            seq_idx));
    }
  }

  if (!kv_lens_already_bound) {
    torch::Tensor expanded_kv_seq_lens_host =
        torch::tensor(input_params.graph.expanded_kv_seq_lens_vec,
                      torch::TensorOptions()
                          .dtype(torch::kInt)
                          .device(torch::kCPU)
                          .pinned_memory(true));
    input_params.graph.expanded_kv_seq_lens =
        expanded_kv_seq_lens_host.to(device, /*non_blocking=*/true);
  }

  // ATB consumes this tensor as dense row-major storage. Keep the generic
  // fallback contiguous; a zero-stride expand view is rejected at runtime.
  torch::Tensor expanded_block_tables = torch::stack(expanded_block_rows, 0);
  layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      input_params,
      input_params.graph.expanded_kv_seq_lens,
      expanded_block_tables,
      input_params.graph.expanded_kv_seq_lens_vec,
      block_size);
}
#endif

#if defined(USE_NPU)
void build_expanded_spec_verify_graph_input(ModelInputParams& input_params,
                                            const torch::Device& device,
                                            int32_t block_size) {
  build_expanded_spec_verify_graph_host_input(input_params);
  bind_expanded_spec_verify_graph_input(
      input_params, device, false, block_size);
}
#endif

torch::Tensor clone_host_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK(tensor.device().is_cpu()) << "expected a CPU host tensor";
  return tensor.contiguous().clone();
}

void stabilize_decode_host_tensors(ForwardInput& input) {
  input.token_ids_host = clone_host_tensor(input.token_ids_host);
  input.positions_host = clone_host_tensor(input.positions_host);
  input.input_params.attention.host.block_tables =
      clone_host_tensor(input.input_params.attention.host.block_tables);
  for (torch::Tensor& block_table : input.input_params.multi_block_tables) {
    block_table = clone_host_tensor(block_table);
  }
}

void set_token_ids_device_tensor(ForwardInput& input,
                                 const torch::Tensor& token_ids,
                                 const torch::TensorOptions& token_options,
                                 Stream& compute_stream) {
  CHECK(token_ids.defined()) << "draft token_ids must be defined";
  torch::Tensor flat_token_ids = token_ids.flatten();
  CHECK_EQ(flat_token_ids.numel(), input.input_params.meta.num_sequences)
      << "draft token count must match num_sequences";

  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  input.device_tensors_ready = false;
  input.token_ids_host = torch::Tensor();
  input.token_ids =
      safe_to(flat_token_ids, token_options, /*non_blocking=*/true);
  input.device_tensors_ready = true;
}

void set_positions_tensor(ForwardInput& input,
                          const torch::Tensor& positions_host,
                          const torch::TensorOptions& device_options) {
  input.device_tensors_ready = false;
  input.positions_host = positions_host;
  input.positions =
      safe_to(input.positions_host, device_options, /*non_blocking=*/true);
  input.device_tensors_ready = true;
}
}  // namespace

#if defined(USE_NPU)
namespace detail {

struct DraftTokenHandoffMetrics final {
  bvar::LatencyRecorder* copy_submission = nullptr;
  bvar::LatencyRecorder* ready_wait = nullptr;
  bvar::LatencyRecorder* bulk_read = nullptr;
  bvar::LatencyRecorder* total_handoff = nullptr;
};

class NpuJsonDraftTokenHandoff final {
 public:
  NpuJsonDraftTokenHandoff(const int64_t max_sequences_per_batch,
                           const int32_t num_speculative_tokens)
      : max_sequences_per_batch_(max_sequences_per_batch) {
    const int32_t metric_count = std::max(num_speculative_tokens, 0);
    metrics_.reserve(metric_count);
    for (int32_t draft_index = 0; draft_index < metric_count; ++draft_index) {
      const std::string metric_key = std::to_string(draft_index);
      DraftTokenHandoffMetrics metrics;
      metrics.copy_submission =
          MULTI_HISTOGRAM_speculative_draft_token_copy_submission_latency_microseconds
              .get_stats({metric_key});
      metrics.ready_wait =
          MULTI_HISTOGRAM_speculative_draft_token_ready_wait_latency_microseconds
              .get_stats({metric_key});
      metrics.bulk_read =
          MULTI_HISTOGRAM_speculative_draft_token_bulk_read_latency_microseconds
              .get_stats({metric_key});
      metrics.total_handoff =
          MULTI_HISTOGRAM_speculative_draft_token_handoff_latency_microseconds
              .get_stats({metric_key});
      metrics_.emplace_back(metrics);
    }
  }

  ~NpuJsonDraftTokenHandoff() {
    if (wait_stream_ != nullptr) {
      const aclError ret = aclrtDestroyStream(wait_stream_);
      if (ret != ACL_SUCCESS) {
        LOG(WARNING)
            << "Failed to destroy JSON draft token handoff wait stream: "
            << ret;
      }
    }
    if (ready_event_ == nullptr) {
      return;
    }
    const aclError ret = aclrtDestroyEvent(ready_event_);
    if (ret != ACL_SUCCESS) {
      LOG(WARNING) << "Failed to destroy JSON draft token handoff event: "
                   << ret;
    }
  }

  std::vector<int32_t> read_tokens(const torch::Tensor& next_tokens,
                                   Stream& compute_stream,
                                   const int32_t draft_index) {
    Timer total_timer;
    const DraftTokenHandoffMetrics* metrics = get_metrics(draft_index);
    const int64_t token_count = next_tokens.numel();
    bool copy_submitted = false;
    bool handoff_ready = false;
    std::vector<int32_t> token_ids;

    if (can_use_async_handoff(next_tokens, token_count)) {
      c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
      if (ensure_resources()) {
        Timer copy_submission_timer;
        try {
          torch::Tensor host_tokens =
              pinned_tokens_.narrow(/*dim=*/0, /*start=*/0, token_count);
          copy_submitted = true;
          host_tokens.copy_(next_tokens.flatten(), /*non_blocking=*/true);
        } catch (const c10::Error& error) {
          disable_event_path("submit host copy", error.what());
        } catch (const std::exception& error) {
          disable_event_path("submit host copy", error.what());
        }
        observe(metrics == nullptr ? nullptr : metrics->copy_submission,
                copy_submission_timer.elapsed_microseconds());

        if (copy_submitted && event_path_enabled_) {
          const aclError record_ret = aclrtRecordEvent(
              ready_event_, compute_stream.get_stream()->stream());
          if (record_ret != ACL_SUCCESS) {
            disable_event_path("record host-copy event", record_ret);
          } else {
            Timer ready_wait_timer;
            const aclError wait_event_ret =
                aclrtStreamWaitEvent(wait_stream_, ready_event_);
            const aclError reset_ret =
                wait_event_ret == ACL_SUCCESS
                    ? aclrtResetEvent(ready_event_, wait_stream_)
                    : wait_event_ret;
            const aclError wait_ret =
                reset_ret == ACL_SUCCESS
                    ? aclrtSynchronizeStreamWithTimeout(wait_stream_,
                                                        /*timeout=*/-1)
                    : reset_ret;
            observe(metrics == nullptr ? nullptr : metrics->ready_wait,
                    ready_wait_timer.elapsed_microseconds());
            if (wait_ret != ACL_SUCCESS) {
              disable_event_path("synchronize host-copy event", wait_ret);
            } else {
              Timer bulk_read_timer;
              token_ids =
                  copy_json_draft_token_ids(pinned_tokens_.data_ptr<int64_t>(),
                                            static_cast<size_t>(token_count));
              observe(metrics == nullptr ? nullptr : metrics->bulk_read,
                      bulk_read_timer.elapsed_microseconds());
              handoff_ready = true;
            }
          }
        }
      }
    }

    if (!handoff_ready) {
      if (copy_submitted) {
        synchronize_after_failed_handoff(compute_stream);
      }
      COUNTER_INC(speculative_draft_token_handoff_fallback_total);
      token_ids = read_tokens_synchronously(next_tokens);
    }

    const int64_t total_microseconds =
        static_cast<int64_t>(total_timer.elapsed_microseconds());
    observe(metrics == nullptr ? nullptr : metrics->total_handoff,
            total_microseconds);
    HISTOGRAM_OBSERVE(speculative_draft_token_d2h_latency_microseconds,
                      total_microseconds);
    return token_ids;
  }

 private:
  bool can_use_async_handoff(const torch::Tensor& next_tokens,
                             const int64_t token_count) const {
    return event_path_enabled_ && max_sequences_per_batch_ > 0 &&
           token_count >= 0 && token_count <= max_sequences_per_batch_ &&
           !next_tokens.device().is_cpu() && next_tokens.is_contiguous() &&
           next_tokens.scalar_type() == torch::kLong;
  }

  bool ensure_resources() {
    if (!pinned_tokens_.defined()) {
      try {
        pinned_tokens_ = torch::empty({max_sequences_per_batch_},
                                      torch::TensorOptions()
                                          .dtype(torch::kLong)
                                          .device(torch::kCPU)
                                          .pinned_memory(true));
      } catch (const c10::Error& error) {
        disable_event_path("allocate pinned host token buffer", error.what());
        return false;
      } catch (const std::exception& error) {
        disable_event_path("allocate pinned host token buffer", error.what());
        return false;
      }
    }

    if (ready_event_ == nullptr) {
      aclError create_ret =
          aclrtCreateEventWithFlag(&ready_event_, ACL_EVENT_SYNC);
      if (create_ret != ACL_SUCCESS) {
        create_ret = aclrtCreateEvent(&ready_event_);
      }
      if (create_ret != ACL_SUCCESS) {
        ready_event_ = nullptr;
        disable_event_path("create host-copy event", create_ret);
        return false;
      }
    }

    if (wait_stream_ == nullptr) {
      const aclError create_ret = aclrtCreateStream(&wait_stream_);
      if (create_ret != ACL_SUCCESS) {
        wait_stream_ = nullptr;
        disable_event_path("create host-copy wait stream", create_ret);
        return false;
      }
    }
    return true;
  }

  std::vector<int32_t> read_tokens_synchronously(
      const torch::Tensor& next_tokens) const {
    torch::Tensor host_tokens =
        safe_to(next_tokens.flatten(), torch::kCPU).contiguous();
    if (host_tokens.scalar_type() == torch::kLong) {
      return copy_json_draft_token_ids(
          host_tokens.data_ptr<int64_t>(),
          static_cast<size_t>(host_tokens.numel()));
    }

    std::vector<int32_t> token_ids;
    token_ids.reserve(host_tokens.numel());
    for (int64_t token_index = 0; token_index < host_tokens.numel();
         ++token_index) {
      token_ids.emplace_back(
          static_cast<int32_t>(host_tokens[token_index].item<int64_t>()));
    }
    return token_ids;
  }

  void synchronize_after_failed_handoff(Stream& compute_stream) const {
    if (wait_stream_ != nullptr) {
      const aclError wait_ret =
          aclrtSynchronizeStreamWithTimeout(wait_stream_, /*timeout=*/-1);
      if (wait_ret != ACL_SUCCESS) {
        LOG(ERROR) << "Failed to synchronize JSON draft token handoff wait "
                      "stream after fallback: "
                   << wait_ret;
      }
    }
    const int32_t compute_ret = compute_stream.synchronize();
    if (compute_ret != 0) {
      LOG(ERROR) << "Failed to synchronize MTP compute stream after JSON draft "
                    "token handoff fallback: "
                 << compute_ret;
    }
  }

  const DraftTokenHandoffMetrics* get_metrics(const int32_t draft_index) const {
    if (draft_index < 0 ||
        draft_index >= static_cast<int32_t>(metrics_.size())) {
      return nullptr;
    }
    return &metrics_[draft_index];
  }

  void observe(bvar::LatencyRecorder* recorder,
               const double elapsed_microseconds) const {
    if (recorder != nullptr) {
      *recorder << static_cast<int64_t>(elapsed_microseconds);
    }
  }

  void disable_event_path(const std::string& operation, const aclError ret) {
    if (event_path_enabled_) {
      LOG(WARNING)
          << "Disabling NPU JSON draft token asynchronous handoff after "
          << operation << " failed: " << ret;
    }
    event_path_enabled_ = false;
  }

  void disable_event_path(const std::string& operation,
                          const std::string& error) {
    if (event_path_enabled_) {
      LOG(WARNING)
          << "Disabling NPU JSON draft token asynchronous handoff after "
          << operation << " failed: " << error;
    }
    event_path_enabled_ = false;
  }

  int64_t max_sequences_per_batch_ = 0;
  torch::Tensor pinned_tokens_;
  aclrtEvent ready_event_ = nullptr;
  aclrtStream wait_stream_ = nullptr;
  bool event_path_enabled_ = true;
  std::vector<DraftTokenHandoffMetrics> metrics_;
};

}  // namespace detail
#endif

MtpLegacyExecutor::MtpLegacyExecutor(MtpRuntime& runtime) : runtime_(runtime) {}
MtpLegacyExecutor::~MtpLegacyExecutor() = default;

void MtpLegacyExecutor::retire_prelaunch() {
  if (!pending_draft_context_.output.has_value()) {
    return;
  }
  CHECK_EQ(runtime_.compute_stream_->synchronize(), 0);
  pending_draft_context_ = PendingDraftContext();
}

using adaptive_pruning::apply_pruned_prefix_lengths;
using adaptive_pruning::clamp_prefix_lengths;
using adaptive_pruning::has_selected_probs_by_step;
using adaptive_pruning::max_pruned_prefix_length;
using adaptive_pruning::selected_probs_by_step;
using adaptive_pruning::sync_pruned_boundary_outputs;
using adaptive_pruning::truncate_draft_outputs;
std::optional<ForwardOutput> MtpLegacyExecutor::step_empty(
    const ForwardInput& input) {
  const bool use_prelaunched_first_draft =
      input.input_params.meta.batch_forward_type.is_decode() &&
      can_use_combined_first_draft() && pending_draft_context_matches(input);
  if (pending_draft_context_.output.has_value() &&
      !use_prelaunched_first_draft) {
    // The preceding validation may have speculatively submitted draft-0 before
    // the scheduler learned that the batch had finished.  Keep its graph/input
    // buffers alive until the queued work completes, then discard the result.
    // This is a batch-exit slow path and is never taken in steady decode.
    const int32_t ret = runtime_.compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to drain final MTP draft prelaunch, ret="
                     << ret;
    pending_draft_context_ = PendingDraftContext();
  }
  runtime_.flush_pending_target_context();

  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    ForwardInput target_prepared;
    ForwardInput draft_prepared;
    auto output =
        runtime_.run_worker_no_sync(*runtime_.impl_, input, target_prepared);
    auto draft_output = runtime_.run_worker_no_sync(
        *runtime_.draft_impl_, input, draft_prepared);
    if (draft_output.has_value()) {
      transfer_retained_inputs(*output, draft_output.value());
    }
    clear_all_output_embeddings(*output);
    finalize_output_on_stream(
        *output, *runtime_.compute_stream_, runtime_.enable_schedule_overlap());
    return output;
  } else {
    ForwardInput draft_extend_prepared;
    std::vector<ForwardInput> draft_step_prepared(
        runtime_.options_.num_speculative_tokens());
    ForwardInput target_prepared;
    std::vector<ForwardOutput> draft_outputs;
    draft_outputs.reserve(runtime_.options_.num_speculative_tokens());

    ForwardInput new_input = input;
    scale_speculative_parallel_token_counts(new_input.input_params,
                                            /*multiplier=*/2);
    if (use_prelaunched_first_draft) {
      draft_outputs.emplace_back(
          std::move(pending_draft_context_.output.value()));
      draft_extend_prepared = std::move(pending_draft_context_.prepared_input);
      pending_draft_context_ = PendingDraftContext();
    } else {
      draft_outputs.emplace_back(
          run_worker_no_sync_impl(*runtime_.draft_impl_,
                                  new_input,
                                  *runtime_.prepare_stream_,
                                  *runtime_.compute_stream_,
                                  draft_extend_prepared)
              .value());
    }

    for (int32_t i = 1; i < runtime_.options_.num_speculative_tokens(); ++i) {
      draft_outputs.emplace_back(runtime_
                                     .run_worker_no_sync(*runtime_.draft_impl_,
                                                         input,
                                                         draft_step_prepared[i])
                                     .value());
    }

    new_input = input;
    scale_speculative_parallel_token_counts(
        new_input.input_params, runtime_.options_.num_speculative_tokens() + 1);
    // Deadlock-safety under DP: this rank's shard is empty but all peers
    // decode, so busy peers allgather their pruned validate counts before the
    // target forward. Join that allgather in lockstep with this rank's uniform
    // count.
    runtime_.sync_dp_global_token_nums_for_idle_rank(new_input.input_params);
    ForwardOutput output =
        runtime_.run_worker_no_sync(*runtime_.impl_, new_input, target_prepared)
            .value();
    for (ForwardOutput& draft_output : draft_outputs) {
      transfer_retained_inputs(output, draft_output);
    }
    clear_all_output_embeddings(output);
    finalize_output_on_stream(
        output, *runtime_.compute_stream_, runtime_.enable_schedule_overlap());
    if (can_prelaunch_next_first_draft(input)) {
      ForwardInput next_first_draft_input = input;
      scale_parallel_token_counts(next_first_draft_input.input_params.parallel,
                                  /*multiplier=*/2);
      submit_pending_first_draft(input, std::move(next_first_draft_input));
    }
    return output;
  }
}

std::optional<ForwardOutput> MtpLegacyExecutor::step_decode(
    const ForwardInput& raw_input) {
  const bool has_bootstrap =
      raw_input.input_params.embedding.mtp_bootstrap_embeddings.defined();
  if (has_bootstrap) {
    runtime_.clear_unified_device_state();
    runtime_.flush_pending_target_context();
  }
  std::optional<ForwardInput> stabilized_input;
  if (runtime_.use_chunked_prefill_spec_verify_path()) {
    stabilized_input.emplace(raw_input);
    stabilize_decode_host_tensors(*stabilized_input);
  }
  const ForwardInput& decode_input =
      stabilized_input.has_value() ? *stabilized_input : raw_input;
  const int32_t num_speculative_tokens =
      runtime_.options_.num_speculative_tokens();
  const bool matching_device_target_context =
      runtime_.pending_target_context_matches(decode_input);
  const bool has_json_object_states = !decode_input.json_object_states.empty();
  const bool use_adaptive_speculative_decode =
      runtime_.adaptive_enabled() &&
      SpeculativeProfileRegistry::get_instance()
          .has_validate_time_predictor() &&
      !has_json_object_states;
  const mtp_async::DecodeRoute route = mtp_async::select_decode_route({
      .unified_graph_capable = false,
      .schedule_overlap = runtime_.enable_schedule_overlap(),
      .combined_draft_supported = supports_combined_first_draft_execution(),
      .pending_target_matches = matching_device_target_context,
      .device_context_ready =
          runtime_.device_target_context_ready_for_batch(decode_input),
      .prelaunched_draft_matches =
          !has_bootstrap && pending_draft_context_matches(decode_input),
      .has_json_states = has_json_object_states,
      .adaptive = use_adaptive_speculative_decode,
  });
  const bool use_prelaunched_first_draft = route.prelaunched_draft;
  const bool use_device_target_context = route.device_target_context;
  COUNTER_INC(speculative_legacy_decode_steps_total);
  runtime_.clear_unified_device_state();
  const torch::Tensor accepted_tokens =
      runtime_.pending_target_context_.accepted_tokens;
  const torch::Tensor accepted_embeddings =
      runtime_.pending_target_context_.accepted_embeddings;
  const torch::Tensor target_base_positions =
      runtime_.pending_target_context_.base_positions;
  const torch::Tensor target_base_kv_seq_lens =
      runtime_.pending_target_context_.base_kv_seq_lens;
  const StreamEventPtr target_context_ready_event =
      runtime_.pending_target_context_.ready_event;
  if (pending_draft_context_.output.has_value() &&
      !use_prelaunched_first_draft) {
    // A batch transition invalidates the speculative prelaunch.  Drain it
    // before releasing its graph/input buffers; this slow path is outside
    // steady decode and preserves cache/buffer lifetime correctness.
    const int32_t ret = runtime_.compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to drain stale MTP draft prelaunch, ret="
                     << ret;
    pending_draft_context_ = PendingDraftContext();
  }
  if (!use_device_target_context) {
    // Batch transitions are uncommon in steady decode.  Materialize the most
    // recent target state before preparing the next graph invocation. The
    // unified path retains Device state and needs no Host metadata correction.
    runtime_.flush_pending_target_context();
    if (matching_device_target_context) {
      // The first target-context publication for a new batch establishes the
      // scheduler's corrected position/KV base. Subsequent publications can
      // derive that base fully on device without waiting for the scheduler.
      runtime_.device_context_ready_embedding_ids_ =
          decode_input.input_params.embedding.embedding_ids;
      runtime_.device_context_ready_request_ids_ =
          decode_input.input_params.embedding.request_ids;
    } else if (!runtime_.device_target_context_ready_for_batch(decode_input)) {
      runtime_.device_context_ready_embedding_ids_.clear();
      runtime_.device_context_ready_request_ids_.clear();
    }
  }
  ForwardInput current_draft_input;
  Timer timer;
  CHECK(runtime_.embedding_cache_ != nullptr)
      << "MTP embedding cache is not allocated";

  runtime_.prepare_mtp_bootstrap(decode_input);

  ForwardInput input =
      stabilized_input.has_value() ? std::move(*stabilized_input) : raw_input;
  ForwardInput metadata_template = input;
  if (use_prelaunched_first_draft) {
    // The first draft was fully prepared and submitted by the preceding
    // run_validate(). Host metadata is corrected below after the accepted-token
    // update; continuous DSA drafts use device correction, while target
    // verification still consumes the exact Host metadata.
  } else if (use_device_target_context) {
    c10::StreamGuard stream_guard =
        runtime_.compute_stream_->set_stream_guard();

    // Clone host tensors before mutating the shallow-copied template.
    metadata_template.token_ids_host =
        clone_host_tensor(metadata_template.token_ids_host);
    metadata_template.positions_host =
        clone_host_tensor(metadata_template.positions_host);

    // Build fixed-shape host metadata immediately while target verification is
    // still running. Use the maximum accepted draft offset for conservative
    // graph planning; actual values replace every device tensor below.
    int32_t* template_positions =
        metadata_template.positions_host.data_ptr<int32_t>();
    int32_t* template_tokens =
        metadata_template.token_ids_host.data_ptr<int32_t>();
    auto& template_kv_lens =
        metadata_template.input_params.attention.host.kv_seq_lens;
    for (int32_t seq_id = 0;
         seq_id < metadata_template.input_params.meta.num_sequences;
         ++seq_id) {
      template_positions[seq_id] += num_speculative_tokens;
      template_kv_lens[seq_id] += num_speculative_tokens;
      if (template_tokens[seq_id] < 0) {
        template_tokens[seq_id] = 0;
      }
    }

    std::vector<EmbeddingCache::DecodeState> template_states(
        metadata_template.input_params.meta.num_sequences);
    const torch::Tensor& placeholder =
        runtime_.embedding_cache_->embedding_placeholder();
    for (int32_t seq_id = 0;
         seq_id < metadata_template.input_params.meta.num_sequences;
         ++seq_id) {
      template_states[seq_id].valid = true;
      template_states[seq_id].request_id =
          metadata_template.input_params.embedding.request_ids[seq_id];
      template_states[seq_id].token_id = template_tokens[seq_id];
      template_states[seq_id].embedding = placeholder;
    }
    runtime_.prepare_draft_extend_inputs(metadata_template,
                                         template_states,
                                         current_draft_input,
                                         /*force_two_rows=*/true);
    wait_metadata_ready_event(current_draft_input, *runtime_.compute_stream_);
    clear_ready_events(current_draft_input);

    mtp_async::prepare_next_draft_from_accepted_state(
        current_draft_input,
        input,
        accepted_tokens,
        accepted_embeddings,
        runtime_.embedding_cache_->embedding_placeholder(),
        target_base_positions,
        target_base_kv_seq_lens,
        /*use_chunked_prefill=*/false,
        /*rebuild_expanded_decode_metadata=*/true,
        runtime_.logical_block_size());
  } else {
    // First decode after prefill and batch transitions use the host cache.
    std::vector<EmbeddingCache::DecodeState> last_states =
        runtime_.embedding_cache_->read_decode_states(
            input.input_params.embedding.embedding_ids,
            input.input_params.embedding.request_ids);
    CHECK_EQ(last_states.size(),
             input.input_params.embedding.embedding_ids.size())
        << "decode target state count mismatch";
    // Synthetic graph warmup requests do not have a target decode history.
    // Draft preparation already falls back to the initialized placeholder for
    // invalid states, so reserve strict cache validation for real requests.
    if (!input.input_params.meta.is_graph_warmup) {
      check_mtp_decode_states(last_states,
                              input.input_params.embedding.request_ids,
                              input.token_ids_host,
                              runtime_.enable_schedule_overlap());
    }
    runtime_.update_decode_step_input(input, last_states);
    metadata_template = input;
    runtime_.prepare_draft_extend_inputs(input,
                                         last_states,
                                         current_draft_input,
                                         /*force_two_rows=*/false);
  }
  std::vector<ForwardOutput> draft_outputs;
  ForwardInput validate_input, next_step_input;
  std::vector<ForwardInput> draft_prepared(num_speculative_tokens);
  detail::JsonDraftValidationScratch json_scratch;
  std::vector<uint8_t> json_invalid_suffix;
  const bool use_continuous_dsa_drafts =
      (use_device_target_context || use_prelaunched_first_draft) &&
      runtime_.combined_draft_execution_path_ ==
          mtp_async::CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION;
  std::vector<ForwardInput> later_draft_inputs;
  torch::Tensor accepted_base_positions;
  torch::Tensor accepted_base_kv_seq_lens;
  if (use_continuous_dsa_drafts) {
    later_draft_inputs.resize(num_speculative_tokens);
    const ForwardInput& combined_draft_input =
        use_prelaunched_first_draft ? pending_draft_context_.prepared_input
                                    : current_draft_input;
    const int64_t batch_size = input.input_params.meta.num_sequences;
    CHECK_EQ(combined_draft_input.positions.numel(), batch_size * 2)
        << "combined draft positions must contain [repair,current] rows";
    CHECK_EQ(
        combined_draft_input.input_params.attention.device.kv_seq_lens.numel(),
        batch_size * 2)
        << "combined draft KV lengths must contain [repair,current] rows";
    accepted_base_positions =
        combined_draft_input.positions.view({batch_size, 2}).select(1, 1);
    accepted_base_kv_seq_lens =
        combined_draft_input.input_params.attention.device.kv_seq_lens
            .view({batch_size, 2})
            .select(1, 1);

    for (int32_t draft_idx = 1; draft_idx < num_speculative_tokens;
         ++draft_idx) {
      // Only the fixed B layout is needed on prepare_stream. Reusing offset 0
      // avoids extending an already conservative Host template past the
      // scheduler-allocated block range; device metadata is replaced below.
      prepare_draft_inputs(metadata_template,
                           later_draft_inputs[draft_idx],
                           /*position_offset=*/0);
    }
  }

  const auto materialize_pending_target_host_state = [&]() {
    runtime_.flush_pending_target_context();
    std::vector<EmbeddingCache::DecodeState> resolved_states =
        runtime_.embedding_cache_->read_decode_states(
            input.input_params.embedding.embedding_ids,
            input.input_params.embedding.request_ids);
    // The scheduler input contains the conservative overlap placeholder.
    // Force cache correction before comparing it with the accepted target.
    input.token_ids_host = torch::full_like(input.token_ids_host, -1);
    runtime_.update_decode_step_input(input, resolved_states);
    check_mtp_decode_states(resolved_states,
                            input.input_params.embedding.request_ids,
                            input.token_ids_host,
                            /*allow_overlap_fake_token=*/false);
    metadata_template = input;
  };

  if (has_json_object_states) {
    json_invalid_suffix.assign(current_draft_input.json_object_states.size(),
                               static_cast<uint8_t>(0));
    json_scratch.states_after.reserve(num_speculative_tokens);
    json_scratch.invalid_draft_step_major.reserve(
        current_draft_input.json_object_states.size() *
        static_cast<size_t>(num_speculative_tokens));
  }
  draft_outputs.reserve(num_speculative_tokens);
  const bool reuse_mtp_topk_state = layer::is_mtp_dsa_topk_reuse_enabled(
      runtime_.draft_impl_->context_.get_model_args());
  MtpTopkStatePtr mtp_topk_state;
  for (int32_t draft_idx = 0; draft_idx < num_speculative_tokens; ++draft_idx) {
    const bool is_final_draft = draft_idx == num_speculative_tokens - 1;
    const bool static_graph_tasks_prepared =
        is_final_draft && !use_continuous_dsa_drafts &&
        prepare_static_mtp_graph_tasks_before_final_draft(input);
    if (reuse_mtp_topk_state) {
      current_draft_input.input_params.mtp_topk_state = mtp_topk_state;
    }
    std::optional<ForwardOutput> draft_output_opt;
    if (use_prelaunched_first_draft && draft_idx == 0) {
      draft_output_opt = std::move(pending_draft_context_.output);
      draft_prepared[draft_idx] =
          std::move(pending_draft_context_.prepared_input);
      pending_draft_context_ = PendingDraftContext();
    } else {
      if (runtime_.uses_embedded_eagle3_draft()) {
        draft_output_opt =
            runtime_.run_worker_no_sync(*runtime_.draft_impl_,
                                        current_draft_input,
                                        draft_prepared[draft_idx]);
      } else {
        draft_output_opt = run_worker_no_sync_impl(*runtime_.draft_impl_,
                                                   current_draft_input,
                                                   *runtime_.compute_stream_,
                                                   *runtime_.compute_stream_,
                                                   draft_prepared[draft_idx]);
      }
    }

    if ((use_device_target_context || use_prelaunched_first_draft) &&
        !use_continuous_dsa_drafts && draft_idx == 0) {
      // The next draft forward is already queued behind target validation.
      // It can start immediately when rejection sampling finishes while the
      // worker materializes the accepted state for later draft/target metadata
      // on CPU.  This synchronization is therefore outside the NPU critical
      // path rather than sitting between target and draft launches.
      materialize_pending_target_host_state();
    }

    if (use_continuous_dsa_drafts && is_final_draft) {
      // Queue target metadata before the Host target-context wait. The fixed
      // template can be copied immediately; its real device values are
      // corrected after a prepare-stream wait on the previous target event.
      prepare_validate_inputs(metadata_template,
                              validate_input,
                              /*static_graph_tasks_prepared=*/false,
                              /*record_ready_event=*/false);
      {
        c10::StreamGuard stream_guard =
            runtime_.prepare_stream_->set_stream_guard();
        CHECK(runtime_.prepare_stream_->wait_event(target_context_ready_event))
            << "failed to wait pending target state on prepare stream";
        mtp_async::prepare_target_verify_from_accepted_state(
            validate_input,
            accepted_tokens,
            target_base_positions,
            target_base_kv_seq_lens,
            runtime_.logical_block_size(),
            runtime_.use_chunked_prefill_spec_verify_path(),
            runtime_.uses_step_major_validate_layout());
        validate_input.retained_device_tensors = {
            accepted_tokens, target_base_positions, target_base_kv_seq_lens};
        finish_metadata_prepare(*runtime_.prepare_stream_, validate_input);
      }

      // Host cache materialization is still required before staging the next
      // target context, but it no longer blocks target metadata submission.
      materialize_pending_target_host_state();
    }

    // Overlap next-step input preparation with async draft forward.
    if (is_final_draft) {
      if (!use_continuous_dsa_drafts) {
        prepare_validate_inputs(
            metadata_template, validate_input, static_graph_tasks_prepared);
      }
    } else if (use_continuous_dsa_drafts) {
      next_step_input = std::move(later_draft_inputs[draft_idx + 1]);
      c10::StreamGuard stream_guard =
          runtime_.compute_stream_->set_stream_guard();
      wait_metadata_ready_event(next_step_input, *runtime_.compute_stream_);
      clear_ready_events(next_step_input);
      mtp_async::prepare_later_draft_from_device_base(
          next_step_input,
          input,
          accepted_base_positions,
          accepted_base_kv_seq_lens,
          draft_idx + 1,
          runtime_.logical_block_size());
    } else {
      prepare_draft_inputs(metadata_template, next_step_input, draft_idx + 1);
    }

    CHECK(draft_output_opt.has_value())
        << "draft output is empty in speculative step";

    draft_outputs.emplace_back(std::move(draft_output_opt.value()));
    const SamplingParameters& draft_sampling_params =
        draft_prepared[draft_idx].sampling_params;
    {
      c10::StreamGuard stream_guard =
          runtime_.compute_stream_->set_stream_guard();
      if (reuse_mtp_topk_state) {
        mtp_topk_state = specBuilder::select_mtp_topk_state_for_next_step(
            draft_outputs.back().mtp_topk_state, draft_sampling_params);
      }
      // Keep draft tokens consistent across the consensus group.
      if (should_broadcast_spec_tokens(
              runtime_.parallel_args_,
              runtime_.get_optimization_config().enable_spec_token_broadcast,
              draft_sampling_params.all_greedy_sample)) {
        SampleOutput& draft_sample = draft_outputs.back().sample_output;
        broadcast_spec_tokens(draft_sample.next_tokens,
                              runtime_.parallel_args_);
      }
      runtime_.process_draft_sample_output(draft_outputs.back().sample_output);
    }
    bool halt_json_draft = false;
    if (has_json_object_states) {
      const torch::Tensor& next_tokens =
          draft_outputs.back().sample_output.next_tokens;
      CHECK(next_tokens.defined())
          << "draft next_tokens must be defined for JSON grammar handling";
      std::vector<int32_t> draft_token_ids;
#if defined(USE_NPU)
      if (json_draft_token_handoff_ == nullptr) {
        json_draft_token_handoff_ =
            std::make_unique<detail::NpuJsonDraftTokenHandoff>(
                static_cast<int64_t>(runtime_.options_.max_seqs_per_batch()),
                num_speculative_tokens);
      }
      draft_token_ids = json_draft_token_handoff_->read_tokens(
          next_tokens, *runtime_.compute_stream_, draft_idx);
#else
      Timer draft_token_d2h_timer;
      torch::Tensor draft_tokens =
          safe_to(next_tokens.flatten(), torch::kCPU).contiguous();
      HISTOGRAM_OBSERVE(
          speculative_draft_token_d2h_latency_microseconds,
          static_cast<int64_t>(draft_token_d2h_timer.elapsed_microseconds()));
      draft_token_ids.reserve(draft_tokens.numel());
      for (int64_t token_idx = 0; token_idx < draft_tokens.numel();
           ++token_idx) {
        draft_token_ids.emplace_back(
            static_cast<int32_t>(draft_tokens[token_idx].item<int64_t>()));
      }
#endif
      halt_json_draft =
          detail::append_json_draft_step(current_draft_input.json_object_states,
                                         json_invalid_suffix,
                                         draft_token_ids,
                                         json_scratch);
    }
    if (draft_idx == num_speculative_tokens - 1) {
      continue;
    }

    const SampleOutput& last_output = draft_outputs.back().sample_output;
    runtime_.check_draft_input_embedding(last_output.embeddings, "decode");
    std::vector<JsonObjectGrammarState> previous_draft_states =
        std::move(current_draft_input.json_object_states);
    current_draft_input = next_step_input;
    current_draft_input.json_object_states = std::move(previous_draft_states);
    set_token_ids_device_tensor(current_draft_input,
                                last_output.next_tokens,
                                current_draft_input.token_ids.options(),
                                *runtime_.compute_stream_);
    if (last_output.embeddings.defined()) {
      current_draft_input.input_params.embedding.input_embedding =
          last_output.embeddings;
      // input_embedding is produced and consumed on compute_stream_; FIFO
      // ordering replaces the same-stream EventRecord/EventWait pair.
    }
    if (has_json_object_states) {
      if (halt_json_draft) {
        // Illegal draft under the current mask: stop further draft forwards
        // and pad remaining speculative slots with -1 so validate rejects them.
        prepare_validate_inputs(input, validate_input);
        while (draft_outputs.size() <
               static_cast<size_t>(num_speculative_tokens)) {
          ForwardOutput rejected_output = draft_outputs.back();
          CHECK(rejected_output.sample_output.next_tokens.defined());
          rejected_output.sample_output.next_tokens = torch::full_like(
              rejected_output.sample_output.next_tokens, /*fill_value=*/-1);
          draft_outputs.push_back(std::move(rejected_output));
          detail::append_json_draft_step(
              current_draft_input.json_object_states,
              json_invalid_suffix,
              std::vector<int32_t>(
                  current_draft_input.json_object_states.size(), -1),
              json_scratch);
        }
        break;
      }
      current_draft_input.sampling_params.filter_bitmask =
          build_json_object_filter_bitmask(
              current_draft_input.json_object_states,
              runtime_.device_,
              JsonObjectMaskBuildPhase::DRAFT);
      current_draft_input.sampling_params.filter_mask = torch::Tensor();
    }
    record_current_metadata_ready_event(current_draft_input,
                                        *runtime_.compute_stream_);
  }
  const double draft_latency_ms = timer.elapsed_milliseconds();
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_latency_ms / 1000.0);

  if (use_adaptive_speculative_decode) {
    return run_adaptive_validate(
        input, draft_outputs, validate_input, num_speculative_tokens);
  }

  return run_validate(input,
                      draft_outputs,
                      validate_input,
                      num_speculative_tokens,
                      /*pruned_prefix_lengths=*/nullptr,
                      has_json_object_states ? &json_scratch : nullptr);
}

void MtpLegacyExecutor::fill_validate_input_from_draft_outputs(
    const ForwardInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input,
    const std::vector<int32_t>& per_seq_val_tokens,
    const detail::JsonDraftValidationScratch* json_scratch,
    Stream& compute_stream) {
  CHECK(!per_seq_val_tokens.empty()) << "per_seq_val_tokens must not be empty";
  const int32_t num_sequences = static_cast<int32_t>(per_seq_val_tokens.size());
  const int32_t max_val_tokens =
      *std::max_element(per_seq_val_tokens.begin(), per_seq_val_tokens.end());
  CHECK(validate_input.token_ids.defined())
      << "validate token_ids must be prepared before draft token fill";
  CHECK_EQ(validate_input.token_ids.dim(), 1)
      << "validate token_ids must be flat";

  const torch::TensorOptions token_options = validate_input.token_ids.options();
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  wait_metadata_ready_event(validate_input, compute_stream);

  if (json_scratch != nullptr) {
    CHECK_EQ(input.json_object_states.size(),
             static_cast<size_t>(num_sequences))
        << "JSON grammar state rows must match validation sequences";
    CHECK_EQ(validate_input.token_ids.numel(),
             static_cast<int64_t>(num_sequences) * max_val_tokens)
        << "adaptive MTP validation is not supported for JSON grammar rows";
    const int32_t num_speculative_tokens = max_val_tokens - 1;
    CHECK_EQ(json_scratch->states_after.size(),
             static_cast<size_t>(num_speculative_tokens));
    validate_input.json_object_states = detail::build_json_validation_states(
        input.json_object_states,
        *json_scratch,
        validate_input.json_object_invalid_draft);
    validate_input.sampling_params.filter_bitmask =
        build_json_object_filter_bitmask(validate_input.json_object_states,
                                         runtime_.device_,
                                         JsonObjectMaskBuildPhase::TARGET);
    validate_input.sampling_params.filter_mask = torch::Tensor();
  }

  validate_input.device_tensors_ready = false;
  auto& fused_draft_tokens =
      validate_input.input_params.graph.spec_verify_draft_token_sources;
  fused_draft_tokens.clear();
#if defined(USE_NPU)
  const bool use_fused_verify_token_update =
      validate_input.input_params.graph.spec_verify_source_addresses_stable &&
      validate_input.input_params.graph.input_tokens_override.defined() &&
      runtime_.supports_explicit_spec_verify_replay_update() &&
      kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
          max_val_tokens, runtime_.options_.block_size());
  if (use_fused_verify_token_update) {
    fused_draft_tokens.reserve(draft_outputs.size());
    for (const ForwardOutput& draft_output : draft_outputs) {
      const torch::Tensor& next_tokens = draft_output.sample_output.next_tokens;
      CHECK(next_tokens.defined() && next_tokens.numel() == num_sequences &&
            next_tokens.scalar_type() == torch::kInt64 &&
            next_tokens.device() == validate_input.token_ids.device() &&
            next_tokens.is_contiguous())
          << "fused speculative verify input update requires one contiguous "
             "int64 token per sequence and draft step";
      fused_draft_tokens.emplace_back(next_tokens.flatten());
    }
    validate_input.device_tensors_ready = true;
    return;
  }
#endif
  const int32_t total_val_tokens =
      static_cast<int32_t>(validate_input.token_ids.numel());
  const bool is_uniform = (total_val_tokens == num_sequences * max_val_tokens);

  if (is_uniform) {
    // Fast path: all seqs share the same val_tokens. Stack the draft-token
    // columns once into a contiguous [batch, num_draft_tokens] tensor and
    // issue a single strided slice-copy, instead of one device kernel per
    // draft step. This is the static (non-adaptive) MTP baseline and runs
    // every decode step; launch-bound NPU decode is measurably faster with
    // one op than num_speculative_tokens ops writing the same bytes.
    const int32_t num_draft_tokens = max_val_tokens - 1;
    const bool step_major_validate_layout =
        runtime_.uses_step_major_validate_layout();
    torch::Tensor validate_token_rows =
        step_major_validate_layout
            ? validate_input.token_ids.view({max_val_tokens, num_sequences})
            : validate_input.token_ids.view({num_sequences, max_val_tokens});
    if (num_draft_tokens > 0) {
      std::vector<torch::Tensor> draft_token_columns;
      draft_token_columns.reserve(static_cast<size_t>(num_draft_tokens));
      for (int32_t i = 0; i < num_draft_tokens; ++i) {
        CHECK(static_cast<size_t>(i) < draft_outputs.size())
            << "draft_outputs index out of range for step " << i;
        const torch::Tensor& next_tokens =
            draft_outputs[static_cast<size_t>(i)].sample_output.next_tokens;
        CHECK(next_tokens.defined())
            << "draft next_tokens must be defined for validate token fill";
        draft_token_columns.push_back(safe_to(
            next_tokens.flatten(), token_options, /*non_blocking=*/true));
      }
      torch::Tensor packed_drafts =
          torch::stack(draft_token_columns, /*dim=*/1);
      if (step_major_validate_layout) {
        validate_token_rows
            .slice(/*dim=*/0, /*start=*/1, /*end=*/max_val_tokens)
            .copy_(packed_drafts.transpose(/*dim0=*/0, /*dim1=*/1),
                   /*non_blocking=*/true);
      } else {
        validate_token_rows
            .slice(/*dim=*/1, /*start=*/1, /*end=*/max_val_tokens)
            .copy_(packed_drafts, /*non_blocking=*/true);
      }
    }
  } else {
    // Slow path: per-seq variable-length, group by draft step.
    std::vector<int64_t> dst_idx_vec;
    std::vector<int64_t> src_idx_vec;
    std::vector<int64_t> step_vec;
    int32_t offset = 0;
    int32_t max_draft_step = -1;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      const int32_t seq_val_tokens =
          per_seq_val_tokens[static_cast<size_t>(seq_id)];
      const int32_t seq_draft_tokens = seq_val_tokens - 1;
      for (int32_t draft_idx = 0; draft_idx < seq_draft_tokens; ++draft_idx) {
        dst_idx_vec.push_back(offset + draft_idx + 1);
        src_idx_vec.push_back(seq_id);
        step_vec.push_back(draft_idx);
        max_draft_step = std::max(max_draft_step, draft_idx);
      }
      offset += seq_val_tokens;
    }

    if (!dst_idx_vec.empty()) {
      // Move all indices to device once (instead of a per-step H2D copy) and
      // select each step's entries on-device via a boolean mask.
      const torch::TensorOptions long_dev_opts =
          torch::TensorOptions()
              .dtype(torch::kLong)
              .device(validate_input.token_ids.device());
      torch::Tensor dst_idx_all =
          safe_to(torch::tensor(dst_idx_vec,
                                torch::TensorOptions().dtype(torch::kLong)),
                  long_dev_opts,
                  /*non_blocking=*/true);
      torch::Tensor src_idx_all =
          safe_to(torch::tensor(src_idx_vec,
                                torch::TensorOptions().dtype(torch::kLong)),
                  long_dev_opts,
                  /*non_blocking=*/true);
      torch::Tensor step_all = safe_to(
          torch::tensor(step_vec, torch::TensorOptions().dtype(torch::kLong)),
          long_dev_opts,
          /*non_blocking=*/true);
      for (int32_t step = 0; step <= max_draft_step; ++step) {
        CHECK(static_cast<size_t>(step) < draft_outputs.size())
            << "draft_outputs index out of range for step " << step;
        const torch::Tensor& next_tokens =
            draft_outputs[static_cast<size_t>(step)].sample_output.next_tokens;
        CHECK(next_tokens.defined())
            << "draft next_tokens must be defined for validate token fill";
        torch::Tensor flat_tokens = safe_to(
            next_tokens.flatten(), token_options, /*non_blocking=*/true);

        torch::Tensor step_mask = step_all.eq(step);
        torch::Tensor step_dst = dst_idx_all.masked_select(step_mask);
        if (step_dst.numel() == 0) {
          continue;
        }
        torch::Tensor step_src = src_idx_all.masked_select(step_mask);
        torch::Tensor gathered = flat_tokens.index_select(/*dim=*/0, step_src);
        validate_input.token_ids.index_copy_(/*dim=*/0, step_dst, gathered);
      }
    }
  }
  validate_input.device_tensors_ready = true;
  record_metadata_ready_event(compute_stream, validate_input);
}

std::optional<ForwardOutput> MtpLegacyExecutor::run_adaptive_validate(
    const ForwardInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input,
    int32_t num_speculative_tokens) {
  const int32_t batch_size = input.input_params.meta.num_sequences;
  std::vector<double> per_seq_kv_lens(static_cast<size_t>(batch_size), 0.0);
  const Slice<int32_t> kv_seq_lens =
      input.input_params.attention.host.kv_seq_lens;
  for (int32_t i = 0; i < batch_size; ++i) {
    if (static_cast<size_t>(i) < kv_seq_lens.size()) {
      per_seq_kv_lens[static_cast<size_t>(i)] = static_cast<double>(
          specBuilder::calc_kv_len(kv_seq_lens, i, /*offset=*/0));
    }
  }

  // All ranks compute pruning independently. Inputs must be deterministic
  // across ranks so every rank derives the same effective validate width.
  // The per-rank measured draft latency is NOT deterministic, so pruning is
  // driven purely by validate-time marginal cost (full_draft_time_ms = 0).
  std::vector<int32_t> prefix_lengths;
  const bool has_probs = has_selected_probs_by_step(draft_outputs);
  if (has_probs) {
    prefix_lengths =
        runtime_.adaptive_spec_controller_->select_pruned_prefix_lengths(
            selected_probs_by_step(draft_outputs),
            /*full_draft_time_ms=*/0.0,
            per_seq_kv_lens);
  } else {
    LOG_FIRST_N(WARNING, 1)
        << "Adaptive speculative pruning disabled: draft exposed neither "
           "probs nor logits for selected-prob computation. Falling back to "
           "full speculative width.";
    prefix_lengths.assign(static_cast<size_t>(batch_size),
                          num_speculative_tokens);
  }
  clamp_prefix_lengths(prefix_lengths, batch_size, num_speculative_tokens);
  int32_t effective_speculative_tokens =
      max_pruned_prefix_length(prefix_lengths, num_speculative_tokens);
  if (effective_speculative_tokens <= 0) {
    effective_speculative_tokens = 1;
  }

  // Qwen3.5 GDN spec-verify commits the recurrent/conv checkpoint selected by
  // the previous step's num_accepted_tokens (nat): the GDN kernel indexes
  // ssm_state as nat - 1 and requires nat <= this step's validate width. Floor
  // effective_speculative_tokens by the batch's max nat so uniform_val_tokens
  // >= max(nat) + 1. nat itself must stay the true accepted count (never
  // clamped) so the committed checkpoint matches the last accepted token;
  // clamping it would commit a stale checkpoint (see issue #2247). Read from
  // embedding_cache directly since input.num_accepted_tokens_host is populated
  // by prepare_validate_inputs which hasn't run yet here.
  if (runtime_.supports_explicit_spec_verify_replay_update() &&
      runtime_.embedding_cache_ != nullptr &&
      !input.input_params.embedding.embedding_ids.empty()) {
    std::vector<int32_t> nat =
        runtime_.embedding_cache_->read_accepted_prefix_lengths(
            input.input_params.embedding.embedding_ids,
            input.input_params.embedding.request_ids);
    int32_t max_nat = 0;
    for (int32_t v : nat) {
      max_nat = std::max(max_nat, v);
    }
    effective_speculative_tokens =
        std::max(effective_speculative_tokens, max_nat);
    effective_speculative_tokens =
        std::min(effective_speculative_tokens, num_speculative_tokens);
  }

  std::vector<int32_t> per_seq_val_tokens(static_cast<size_t>(batch_size));
  // Qwen3.5 GatedDeltaNet spec-verify path requires dense same-length validate
  // tokens across sequences (see qwen3_gated_delta_net_base.cpp:405-408). On
  // Qwen3.5 we still take the batch-max pruning benefit (effective_sl < max_sl
  // when the controller decides to shrink), but every seq gets the same
  // validate width. On non-Qwen3.5 models we keep per-seq variable-length
  // tokens for maximum pruning benefit.
  const bool require_uniform_val_tokens =
      runtime_.requires_uniform_validate_width();
  const int32_t uniform_val_tokens = effective_speculative_tokens + 1;
  for (int32_t i = 0; i < batch_size; ++i) {
    per_seq_val_tokens[static_cast<size_t>(i)] =
        require_uniform_val_tokens
            ? uniform_val_tokens
            : std::max(prefix_lengths[static_cast<size_t>(i)], 1) + 1;
  }
  std::vector<ForwardOutput> pruned_draft_outputs =
      truncate_draft_outputs(draft_outputs, effective_speculative_tokens);
  // If the controller did not actually prune any sequence, treat this as the
  // static path: pass nullptr so run_validate takes the async handoff tail
  // and skips the no-op pruned post-processing.
  const bool has_actual_prune =
      std::any_of(prefix_lengths.begin(),
                  prefix_lengths.end(),
                  [num_speculative_tokens](int32_t p) {
                    return p < num_speculative_tokens;
                  });
  prepare_validate_inputs(input, validate_input, per_seq_val_tokens);
  return run_validate(input,
                      pruned_draft_outputs,
                      validate_input,
                      effective_speculative_tokens,
                      per_seq_val_tokens,
                      has_actual_prune ? &prefix_lengths : nullptr,
                      /*json_scratch=*/nullptr);
}

std::optional<ForwardOutput> MtpLegacyExecutor::run_validate(
    const ForwardInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths,
    const detail::JsonDraftValidationScratch* json_scratch) {
  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t val_tokens = num_speculative_tokens + 1;
  std::vector<int32_t> per_seq_val_tokens(static_cast<size_t>(batch_size),
                                          val_tokens);
  return run_validate(input,
                      draft_outputs,
                      validate_input,
                      num_speculative_tokens,
                      per_seq_val_tokens,
                      pruned_prefix_lengths,
                      json_scratch);
}

std::optional<ForwardOutput> MtpLegacyExecutor::run_validate(
    const ForwardInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>& per_seq_val_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths,
    const detail::JsonDraftValidationScratch* json_scratch) {
  Timer timer;
  ForwardInput target_prepared;
  fill_validate_input_from_draft_outputs(input,
                                         draft_outputs,
                                         validate_input,
                                         per_seq_val_tokens,
                                         json_scratch,
                                         *runtime_.compute_stream_);
  // Under DP, publish this rank's true validate token count to all DP peers so
  // DpEpPadding computes matching MoE all-to-all pads. per_seq_val_tokens is
  // uniform (N+1) on the static path and varlen on the adaptive path; both
  // funnel through here, so every DP rank runs the collective in lockstep.
  // No-op when the DP group spans a single rank.
  int32_t local_total_val_tokens = 0;
  for (int32_t v : per_seq_val_tokens) {
    local_total_val_tokens += v;
  }
  runtime_.sync_dp_global_token_nums_after_prune(validate_input.input_params,
                                                 local_total_val_tokens);
  ForwardOutput target_output;
  if (runtime_.uses_embedded_eagle3_draft()) {
    target_output = runtime_
                        .run_worker_no_sync(
                            *runtime_.impl_, validate_input, target_prepared)
                        .value();
  } else {
    target_output = run_worker_no_sync_impl(*runtime_.impl_,
                                            validate_input,
                                            *runtime_.compute_stream_,
                                            *runtime_.compute_stream_,
                                            target_prepared)
                        .value();
  }
  const double target_latency_ms = timer.elapsed_milliseconds();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              target_latency_ms / 1000.0);

  const int32_t batch_size = static_cast<int32_t>(per_seq_val_tokens.size());
  const int32_t max_val_tokens = num_speculative_tokens + 1;
  const int32_t total_tokens =
      static_cast<int32_t>(target_output.logits.size(0));
  const int32_t vocab_size =
      static_cast<int32_t>(target_output.logits.size(-1));
  const int64_t padded_total =
      static_cast<int64_t>(batch_size) * max_val_tokens;

  // For the uniform fast path we only need to reinterpret target_output.logits
  // as `[padded_total, vocab]` — no ForwardOutput copy required. Only the
  // variable-length slow path materializes a separate padded output.
  std::optional<ForwardOutput> padded_target_output_slow;
  const bool needs_padding =
      (total_tokens != static_cast<int32_t>(padded_total));
  if (needs_padding) {
    // Slow path: per-seq variable-length, scatter into padded layout. Pad
    // next_tokens with 0 (MTP's established padding); trailing pads are masked
    // to -1 by apply_pruned_prefix_lengths downstream regardless.
    padded_target_output_slow.emplace(target_output);
    adaptive_pruning::scatter_varlen_target_output_to_dense(
        *padded_target_output_slow,
        per_seq_val_tokens,
        batch_size,
        max_val_tokens,
        /*next_token_pad_value=*/0);
  }
  // Uniform fast path uses a scoped local ForwardOutput whose only diff is
  // logits viewed to [padded_total, vocab]; slow path uses the materialized
  // padded copy above. Both are const-ref'd into validate() below.
  ForwardOutput uniform_target_view;
  if (!needs_padding) {
    uniform_target_view = target_output;
    uniform_target_view.logits =
        target_output.logits.view({padded_total, vocab_size});
  }
  const ForwardOutput& target_output_for_validate =
      needs_padding ? *padded_target_output_slow : uniform_target_view;

  const bool prelaunch_next_first_draft =
      pruned_prefix_lengths == nullptr && can_prelaunch_next_first_draft(input);
  ForwardInput next_first_draft_input;
  if (prelaunch_next_first_draft) {
    // This input is independent of the accepted token.  Prepare it on the
    // auxiliary stream while target verification is still executing; the
    // compute stream consumes it through a device-side event after rejection
    // sampling, with no host synchronization.
    prepare_next_first_draft_template(input, next_first_draft_input);
  }

  // verify the proposals with target and update the batch
  timer.reset();
  SampleOutput val_output;
  {
    c10::StreamGuard stream_guard =
        runtime_.compute_stream_->set_stream_guard();
    val_output = validate(input.sampling_params,
                          draft_outputs,
                          target_output_for_validate,
                          num_speculative_tokens,
                          pruned_prefix_lengths,
                          validate_input.sampling_params.filter_mask,
                          validate_input.sampling_params.filter_bitmask,
                          validate_input.json_object_invalid_draft);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  if (pruned_prefix_lengths != nullptr ||
      runtime_.uses_embedded_eagle3_draft()) {
    // Adaptive pruning path: per-seq validate width is variable, which is
    // incompatible with the async handoff's fixed-width base-state derivation.
    // Use the synchronous tail: unify tokens, then write target context inline.
    if (should_broadcast_spec_tokens(
            runtime_.parallel_args_,
            runtime_.get_optimization_config().enable_spec_token_broadcast,
            input.sampling_params.all_greedy_sample)) {
      c10::StreamGuard stream_guard =
          runtime_.compute_stream_->set_stream_guard();
      broadcast_spec_tokens(val_output.next_tokens, runtime_.parallel_args_);
    }

    const int32_t ret = runtime_.compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize MTP compute stream, ret=" << ret;
    target_output.retained_inputs.clear();
    val_output.next_tokens = val_output.next_tokens.to(torch::kCPU);
    // Record adaptive-prune-aware draft/accept counts on the already-CPU
    // next_tokens. Static path lets worker_service count on the async-handoff
    // CPU tensor with no extra device sync.
    record_validate_metrics(
        val_output, num_speculative_tokens, pruned_prefix_lengths);
    write_target_context_to_cache(input, val_output, num_speculative_tokens);

    if (!runtime_.enable_schedule_overlap() && !runtime_.driver_ &&
        !runtime_.dp_driver_) {
      return std::nullopt;
    }
    clear_all_output_embeddings(target_output);
    val_output.embeddings = torch::Tensor();
    target_output.sample_output = val_output;
    return target_output;
  }

  const int64_t num_val_tokens = runtime_.options_.num_speculative_tokens() + 1;
  torch::Tensor validate_positions = validate_input.positions;
  CHECK_EQ(validate_positions.numel(), batch_size * num_val_tokens)
      << "validate positions must contain one row per speculative token";
  const torch::Tensor& validate_kv_seq_lens =
      validate_input.input_params.attention.device.kv_seq_lens;
  CHECK_GE(validate_kv_seq_lens.numel(), batch_size)
      << "validate KV lengths must be sequence-scoped";
  torch::Tensor accepted_tokens_host =
      runtime_.acquire_accepted_tokens_host_buffer(val_output.next_tokens);
  torch::Tensor accepted_tokens_cpu_result = accepted_tokens_host;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
  StreamEventPtr target_context_ready_event;
  {
    c10::StreamGuard stream_guard =
        runtime_.compute_stream_->set_stream_guard();

    // Catch-all for cross-rank RNG divergence: unify accepted tokens before
    // deriving any device-resident state used by the next draft iteration.
    if (should_broadcast_spec_tokens(
            runtime_.parallel_args_,
            runtime_.get_optimization_config().enable_spec_token_broadcast,
            input.sampling_params.all_greedy_sample)) {
      broadcast_spec_tokens(val_output.next_tokens, runtime_.parallel_args_);
    }

    base_positions = validate_positions.view({batch_size, num_val_tokens})
                         .select(/*dim=*/1, /*index=*/0)
                         .contiguous();
    base_kv_seq_lens = mtp_async::extract_target_base_kv_seq_lens(
        validate_kv_seq_lens,
        batch_size,
        num_val_tokens,
        runtime_.use_chunked_prefill_spec_verify_path(),
        runtime_.uses_step_major_validate_layout());

    accepted_tokens_host.copy_(val_output.next_tokens,
                               /*non_blocking=*/true);
    // The event covers consensus, base-state derivation, and the D2H copy.
    target_context_ready_event = runtime_.compute_stream_->record_event();
  }
  if (target_context_ready_event == nullptr) {
    const int32_t ret = runtime_.compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize MTP target context, ret=" << ret;
  }
  std::vector<size_t> failed_sequence_rows;
  if (!input.json_object_states.empty()) {
    CHECK_EQ(accepted_tokens_host.dim(), 2)
        << "MTP JSON accepted output must be [sequence, token]";
    CHECK_EQ(input.sample_sequence_ids.size(), input.json_object_states.size())
        << "MTP JSON sequence ids must match grammar state rows";
    CHECK_EQ(input.input_params.embedding.embedding_ids.size(),
             input.json_object_states.size())
        << "MTP JSON embedding ids must match grammar state rows";
    CHECK(target_context_ready_event == nullptr ||
          target_context_ready_event->synchronize())
        << "failed to wait for accepted MTP tokens before JSON replay";

    const torch::Tensor accepted_tokens =
        accepted_tokens_host.to(torch::kInt64).contiguous();
    const std::vector<detail::JsonAcceptedTokenMismatch> mismatches =
        detail::find_json_accepted_token_mismatches(
            input.json_object_states,
            accepted_tokens.const_data_ptr<int64_t>(),
            static_cast<size_t>(accepted_tokens.size(0)),
            static_cast<size_t>(accepted_tokens.size(1)));
    std::unordered_set<std::string> failed_sequence_ids;
    failed_sequence_ids.reserve(input.json_object_errors.size() +
                                target_output.json_object_errors.size() +
                                mismatches.size());
    for (const JsonObjectOutputError& error : input.json_object_errors) {
      CHECK(!error.sample_sequence_id.empty())
          << "MTP JSON input error requires a sampled sequence id";
      failed_sequence_ids.emplace(error.sample_sequence_id);
    }
    for (const JsonObjectOutputError& error :
         target_output.json_object_errors) {
      CHECK(!error.sample_sequence_id.empty())
          << "MTP JSON output error requires a sampled sequence id";
      failed_sequence_ids.emplace(error.sample_sequence_id);
    }
    for (const detail::JsonAcceptedTokenMismatch& mismatch : mismatches) {
      const std::string& sample_sequence_id =
          input.sample_sequence_ids[mismatch.sequence_index];
      CHECK(!sample_sequence_id.empty())
          << "MTP JSON mismatch requires a sampled sequence id";
      LOG(ERROR) << "MTP JSON accepted output replay mismatch: sequence_id="
                 << sample_sequence_id
                 << ", token_offset=" << mismatch.token_offset
                 << ", token_id=" << mismatch.token_id
                 << ", committed_tokens=" << mismatch.committed_tokens
                 << ", state_fingerprint=" << mismatch.state_fingerprint;
      if (failed_sequence_ids.emplace(sample_sequence_id).second) {
        target_output.json_object_errors.push_back(
            {sample_sequence_id,
             "accepted MTP output violates json_object grammar, token_id=" +
                 std::to_string(mismatch.token_id)});
      }
    }
    failed_sequence_rows.reserve(failed_sequence_ids.size());
    for (size_t sequence_index = 0;
         sequence_index < input.sample_sequence_ids.size();
         ++sequence_index) {
      if (failed_sequence_ids.contains(
              input.sample_sequence_ids[sequence_index])) {
        failed_sequence_rows.emplace_back(sequence_index);
      }
    }
    CHECK_EQ(failed_sequence_rows.size(), failed_sequence_ids.size())
        << "MTP JSON errors must reference sampled rows in the current batch";
  }
  const bool has_failed_sequence_rows = !failed_sequence_rows.empty();
  runtime_.stage_target_context_write(input,
                                      val_output,
                                      base_positions,
                                      base_kv_seq_lens,
                                      target_context_ready_event,
                                      std::move(accepted_tokens_host),
                                      torch::Tensor(),
                                      std::move(failed_sequence_rows));
  if (prelaunch_next_first_draft && !has_failed_sequence_rows) {
    // Submit the next iteration's first draft before returning to the
    // scheduler.  This is the actual asynchronous boundary: scheduler/host
    // accepted-state work can no longer sit between target validation and the
    // next draft launch.
    enqueue_next_first_draft(input,
                             val_output,
                             base_positions,
                             base_kv_seq_lens,
                             std::move(next_first_draft_input));
  }
  target_output.ready_event = target_context_ready_event;

  // Target validation consumes all draft outputs on the same compute stream;
  // keep their prepared inputs alive on target_output until the target-context
  // event completes. Copy (not move): draft_outputs live beyond this loop.
  for (const ForwardOutput& draft_output : draft_outputs) {
    copy_retained_inputs(target_output, draft_output);
  }

  if (!runtime_.enable_schedule_overlap()) {
    runtime_.flush_pending_target_context();
    target_output.retained_inputs.clear();
    target_output.ready_event.reset();
    val_output.next_tokens = std::move(accepted_tokens_cpu_result);
  }

  if (!runtime_.enable_schedule_overlap() && !runtime_.driver_ &&
      !runtime_.dp_driver_) {
    return std::nullopt;
  }
  clear_all_output_embeddings(target_output);
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

void MtpLegacyExecutor::write_target_context_to_cache(
    const ForwardInput& input,
    const SampleOutput& validate_output,
    int32_t num_speculative_tokens) {
  CHECK(runtime_.embedding_cache_ != nullptr)
      << "embedding_cache_ must be initialized before target cache write";
  CHECK(!input.input_params.embedding.embedding_ids.empty())
      << "target context cache write requires embedding ids";
  runtime_.embedding_cache_->write_target_context(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      validate_output.next_tokens,
      validate_output.embeddings,
      num_speculative_tokens);
}

bool MtpLegacyExecutor::supports_combined_first_draft_execution() const {
#if defined(USE_NPU)
  if (runtime_.draft_impl_ == nullptr ||
      runtime_.draft_impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    return false;
  }

  // The ATB speculative path expects CHUNKED_PREFILL metadata instead of the
  // eager two-row DECODE input used by the prelaunch.
  if (::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel()) {
    return false;
  }

  const std::string& npu_backend =
      ::xllm::KernelConfig::get_instance().npu_kernel_backend();
  return runtime_.device_.unwrap().is_privateuseone() &&
         mtp_async::supports_combined_draft_configuration(
             runtime_.combined_draft_execution_path_,
             npu_backend,
             runtime_.parallel_args_.dp_size());
#else
  return false;
#endif
}

bool MtpLegacyExecutor::can_use_combined_first_draft() const {
  return runtime_.enable_schedule_overlap() &&
         supports_combined_first_draft_execution();
}

bool MtpLegacyExecutor::can_prelaunch_next_first_draft(
    const ForwardInput& input) const {
  if (!can_use_combined_first_draft()) {
    return false;
  }
  if (runtime_.supports_unified_python_mtp_graph(input) &&
      input.json_object_states.empty() &&
      !(runtime_.adaptive_enabled() &&
        SpeculativeProfileRegistry::get_instance()
            .has_validate_time_predictor())) {
    // The next unified replay includes draft-0. Do not write its KV early
    // through the legacy runner or create a second producer of that state.
    return false;
  }
  const bool requires_dp_symmetric_prelaunch =
      runtime_.parallel_args_.dp_size() > 1 &&
      runtime_.combined_draft_execution_path_ ==
          mtp_async::CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION;
  if (requires_dp_symmetric_prelaunch) {
    return has_active_dp_tokens(input);
  }
  return runtime_.device_target_context_ready_for_batch(input);
}

void MtpLegacyExecutor::prepare_next_first_draft_template(
    const ForwardInput& input,
    ForwardInput& combined_input) {
  CHECK(runtime_.embedding_cache_ != nullptr);

  ForwardInput metadata_template = input;
  // Clone host tensors before mutating the shallow-copied template.
  metadata_template.token_ids_host =
      clone_host_tensor(metadata_template.token_ids_host);
  metadata_template.positions_host =
      clone_host_tensor(metadata_template.positions_host);
  const int32_t num_speculative_tokens =
      runtime_.options_.num_speculative_tokens();
  int32_t* template_positions =
      metadata_template.positions_host.data_ptr<int32_t>();
  int32_t* template_tokens =
      metadata_template.token_ids_host.data_ptr<int32_t>();
  auto& template_kv_lens =
      metadata_template.input_params.attention.host.kv_seq_lens;
  for (int32_t seq_id = 0;
       seq_id < metadata_template.input_params.meta.num_sequences;
       ++seq_id) {
    template_positions[seq_id] += num_speculative_tokens;
    template_kv_lens[seq_id] += num_speculative_tokens;
    if (template_tokens[seq_id] < 0) {
      template_tokens[seq_id] = 0;
    }
  }

  std::vector<EmbeddingCache::DecodeState> template_states(
      metadata_template.input_params.meta.num_sequences);
  const torch::Tensor& placeholder =
      runtime_.embedding_cache_->embedding_placeholder();
  for (int32_t seq_id = 0;
       seq_id < metadata_template.input_params.meta.num_sequences;
       ++seq_id) {
    template_states[seq_id].valid = true;
    template_states[seq_id].request_id =
        metadata_template.input_params.embedding.request_ids[seq_id];
    template_states[seq_id].token_id = template_tokens[seq_id];
    template_states[seq_id].embedding = placeholder;
  }

  runtime_.prepare_draft_extend_inputs(metadata_template,
                                       template_states,
                                       combined_input,
                                       /*force_two_rows=*/true,
                                       /*wait_for_compute_stream=*/false);
  combined_input.skip_sampling_for_logits_only = false;
}

void MtpLegacyExecutor::enqueue_next_first_draft(
    const ForwardInput& input,
    const SampleOutput& validate_output,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    ForwardInput combined_input) {
  CHECK(validate_output.next_tokens.defined());
  CHECK(validate_output.embeddings.defined());
  CHECK(runtime_.embedding_cache_ != nullptr);

  c10::StreamGuard stream_guard = runtime_.compute_stream_->set_stream_guard();
  wait_metadata_ready_event(combined_input, *runtime_.compute_stream_);
  clear_ready_events(combined_input);

  // Interleave [repair, current] rows in one decode batch. Every transformer
  // layer projects both rows, writes both KV rows, and only then launches
  // PagedAttention. Same-stream ordering therefore makes repair KV visible to
  // the current row without a host wait or a separate repair forward.
  mtp_async::prepare_next_draft_from_accepted_state(
      combined_input,
      input,
      validate_output.next_tokens,
      validate_output.embeddings,
      runtime_.embedding_cache_->embedding_placeholder(),
      base_positions,
      base_kv_seq_lens,
      /*use_chunked_prefill=*/false,
      /*rebuild_expanded_decode_metadata=*/false,
      runtime_.logical_block_size());

  submit_pending_first_draft(input, std::move(combined_input));
}

void MtpLegacyExecutor::submit_pending_first_draft(
    const ForwardInput& batch_identity_input,
    ForwardInput draft_input) {
  CHECK(!pending_draft_context_.output.has_value())
      << "MTP first-draft prelaunch was not consumed";
  pending_draft_context_.embedding_ids =
      batch_identity_input.input_params.embedding.embedding_ids;
  pending_draft_context_.request_ids =
      batch_identity_input.input_params.embedding.request_ids;
  pending_draft_context_.dp_global_token_nums =
      batch_identity_input.input_params.parallel.dp_global_token_nums;
  pending_draft_context_.dp_global_sequence_nums =
      batch_identity_input.input_params.parallel.dp_global_sequence_nums;
  pending_draft_context_.raw_dp_global_token_nums =
      batch_identity_input.input_params.parallel.raw_dp_global_token_nums;
  pending_draft_context_.dp_global_batch_generations =
      batch_identity_input.input_params.parallel.dp_global_batch_generations;
  pending_draft_context_.output =
      run_worker_no_sync_impl(*runtime_.draft_impl_,
                              draft_input,
                              *runtime_.compute_stream_,
                              *runtime_.compute_stream_,
                              pending_draft_context_.prepared_input);
  CHECK(pending_draft_context_.output.has_value())
      << "failed to prelaunch next MTP first draft";
}

bool MtpLegacyExecutor::pending_draft_context_matches(
    const ForwardInput& input) const {
  return pending_draft_context_.output.has_value() &&
         pending_draft_context_.embedding_ids ==
             input.input_params.embedding.embedding_ids &&
         pending_draft_context_.request_ids ==
             input.input_params.embedding.request_ids &&
         pending_draft_context_.dp_global_token_nums ==
             input.input_params.parallel.dp_global_token_nums &&
         pending_draft_context_.dp_global_sequence_nums ==
             input.input_params.parallel.dp_global_sequence_nums &&
         pending_draft_context_.raw_dp_global_token_nums ==
             input.input_params.parallel.raw_dp_global_token_nums &&
         pending_draft_context_.dp_global_batch_generations ==
             input.input_params.parallel.dp_global_batch_generations;
}

void MtpLegacyExecutor::record_validate_metrics(
    SampleOutput& validate_output,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths) const {
  CHECK(validate_output.next_tokens.defined())
      << "validate output tokens are undefined";
  CHECK_EQ(validate_output.next_tokens.dim(), 2)
      << "validate output tokens should be [batch, width]";
  const int32_t batch_size =
      static_cast<int32_t>(validate_output.next_tokens.size(0));
  CHECK_EQ(validate_output.next_tokens.size(1), num_speculative_tokens + 1)
      << "validate output width mismatch";

  CHECK(validate_output.next_tokens.device().is_cpu())
      << "record_validate_metrics expects next_tokens already on CPU to avoid "
         "a blocking device sync on the hot path";
  std::vector<int32_t> proposed_tokens(static_cast<size_t>(batch_size),
                                       num_speculative_tokens);
  for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    if (pruned_prefix_lengths != nullptr) {
      CHECK_EQ(pruned_prefix_lengths->size(), static_cast<size_t>(batch_size))
          << "adaptive pruning prefix length batch mismatch";
      proposed_tokens[static_cast<size_t>(seq_id)] =
          std::clamp((*pruned_prefix_lengths)[static_cast<size_t>(seq_id)],
                     0,
                     num_speculative_tokens);
    }
  }
  validate_output.speculative_token_stats =
      calculate_mtp_speculative_token_stats(validate_output.next_tokens,
                                            proposed_tokens);
  int64_t num_draft_tokens = 0;
  int64_t accepted_count = 0;
  for (const SpeculativeTokenStats& stats :
       validate_output.speculative_token_stats) {
    num_draft_tokens += stats.proposed_tokens;
    accepted_count += stats.accepted_tokens;
  }
  COUNTER_ADD(speculative_num_draft_tokens_total, num_draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_total, accepted_count);
}

void MtpLegacyExecutor::prepare_validate_inputs(
    const ForwardInput& input,
    ForwardInput& validate_input,
    bool static_graph_tasks_prepared,
    bool record_ready_event) {
  c10::StreamGuard stream_guard = runtime_.prepare_stream_->set_stream_guard();
  validate_input = input;
  clear_ready_events(validate_input);
  validate_input.device_tensors_ready = false;
  auto& input_params = validate_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  torch::TensorOptions token_options = validate_input.token_ids.options();
  torch::TensorOptions position_options = validate_input.positions.options();

  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t num_val_tokens = runtime_.options_.num_speculative_tokens() + 1;
  const int32_t total_num_val_tokens = num_sequences * num_val_tokens;
  const int32_t logical_block_size = runtime_.logical_block_size();
  const bool positions_decoupled =
      runtime_.positions_are_decoupled_from_kv_length();
#if defined(USE_NPU)
  const bool use_explicit_spec_verify_replay_update =
      runtime_.should_use_explicit_spec_verify_replay_update(input);
  const bool expand_python_mtp_linear_state_ids =
      num_val_tokens > 1 &&
      ModelConfig::is_python_model_impl(runtime_.context_.get_model_impl()) &&
      !input_params.embedding.linear_state_ids.empty();
  std::vector<int32_t> expanded_linear_state_ids;
  if (expand_python_mtp_linear_state_ids) {
    CHECK_EQ(input_params.embedding.linear_state_ids.size(),
             static_cast<size_t>(num_sequences));
    expanded_linear_state_ids.reserve(
        static_cast<size_t>(total_num_val_tokens));
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      expanded_linear_state_ids.insert(
          expanded_linear_state_ids.end(),
          num_val_tokens,
          input_params.embedding.linear_state_ids[static_cast<size_t>(seq_id)]);
    }
  }
#else
  const bool use_explicit_spec_verify_replay_update = false;
#endif
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  Slice<int32_t> token_ids = {
      input.token_ids_host.data_ptr<int32_t>(),
      static_cast<size_t>(input.token_ids_host.numel())};
  Slice<int32_t> positions = {
      input.positions_host.data_ptr<int32_t>(),
      static_cast<size_t>(input.positions_host.numel())};
  Slice<int32_t> kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  const bool use_atb_spec_kernel =
      ::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel() ||
      runtime_.use_chunked_prefill_spec_verify_path();
  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(total_num_val_tokens);
  buf.out_positions.reserve(total_num_val_tokens);
  buf.out_new_cache_slots.reserve(total_num_val_tokens);
  if (!use_atb_spec_kernel) {
    buf.out_kv_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_cu_seq_lens.reserve(total_num_val_tokens);
    buf.out_block_tables.reserve(static_cast<size_t>(total_num_val_tokens) *
                                 row_ctx.block_table_stride);
  }

  std::vector<int32_t> atb_kv_seq_lens_vec;
  std::vector<int32_t> atb_q_seq_lens_vec;
  std::vector<int32_t> atb_q_cu_seq_lens_vec;
  int32_t atb_kv_max_seq_len = 0;
  const bool step_major_validate_layout =
      runtime_.uses_step_major_validate_layout();
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t start_position = positions[seq_id];
    const int32_t kv_len =
        specBuilder::calc_kv_len(kv_seq_lens, seq_id, /*offset=*/0);
    if (!positions_decoupled) {
      CHECK_EQ(start_position + 1, kv_len)
          << "validate position/kv_len mismatch, seq_id=" << seq_id
          << ", start_position=" << start_position << ", kv_len=" << kv_len;
    }

    if (use_atb_spec_kernel) {
      const int32_t kv_len_after_validation =
          kv_len + runtime_.options_.num_speculative_tokens();
      specBuilder::update_kv_seq_lens_and_max(
          atb_kv_seq_lens_vec, kv_len_after_validation, atb_kv_max_seq_len);
      specBuilder::append_q_seq_len(
          atb_q_seq_lens_vec, atb_q_cu_seq_lens_vec, num_val_tokens);
    }
  }
  if (step_major_validate_layout) {
    for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
      for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
        specBuilder::RowSpec row;
        row.seq_id = seq_id;
        row.token_id = val_idx == 0 ? token_ids[seq_id] : -val_idx;
        row.position_offset = val_idx;
        row.append_kv_len = true;
        row.append_q_len_one = true;
        row.append_block_table = true;
        specBuilder::append_decode_row(row_ctx, row, logical_block_size, buf);
      }
    }
  } else {
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
        specBuilder::RowSpec row;
        row.seq_id = seq_id;
        row.token_id = val_idx == 0 ? token_ids[seq_id] : -val_idx;
        row.position_offset = val_idx;
        row.append_kv_len = !use_atb_spec_kernel;
        row.append_q_len_one = !use_atb_spec_kernel;
        row.append_block_table = !use_atb_spec_kernel;
        specBuilder::append_decode_row(row_ctx, row, logical_block_size, buf);
      }
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "validate positions/tokens mismatch";

  if (!use_explicit_spec_verify_replay_update) {
    specBuilder::set_token_position_tensors(validate_input,
                                            buf.out_token_ids,
                                            buf.out_positions,
                                            token_options,
                                            position_options);
  }
  if (!use_atb_spec_kernel) {
    input_params.meta.num_sequences = total_num_val_tokens;
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  } else {
    input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  }
  if (use_atb_spec_kernel) {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     num_val_tokens,
                                     std::move(atb_q_seq_lens_vec),
                                     std::move(atb_q_cu_seq_lens_vec),
                                     atb_kv_max_seq_len,
                                     std::move(atb_kv_seq_lens_vec));
  } else {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     1,
                                     std::move(buf.out_q_seq_lens),
                                     std::move(buf.out_q_cu_seq_lens),
                                     buf.meta.kv_max_seq_len,
                                     std::move(buf.out_kv_seq_lens),
                                     /*update_block_tables=*/true);
  }

  if (!input.json_object_states.empty()) {
    // Validation installs exact per-position JSON masks after the draft
    // tokens are known. Do not repeat the inherited one-row mask here.
    validate_input.sampling_params.filter_mask = torch::Tensor();
    validate_input.sampling_params.filter_bitmask = torch::Tensor();
  }
  auto& validate_sampling_params = validate_input.sampling_params;
#if defined(USE_NPU)
  // update_sampling_params() uses repeat_interleave on device tensors.  For a
  // single greedy sequence that work only expands [0] and [false] into fixed
  // validation controls, yet it lands on the final-draft -> target dependency
  // chain.  Reuse stable controls after warmup and retain the generic builder
  // for sampling, penalties, multi-sequence batches and other backends.
  const bool use_stable_greedy_validate_sampling =
      use_explicit_spec_verify_replay_update &&
      validate_sampling_params.all_greedy_sample &&
      validate_sampling_params.selected_token_idxes.defined() &&
      validate_sampling_params.selected_token_idxes.numel() == 1 &&
      validate_sampling_params.sample_idxes.defined() &&
      validate_sampling_params.sample_idxes.numel() == 1 &&
      validate_sampling_params.do_sample.defined() &&
      validate_sampling_params.do_sample.numel() == 1 &&
      !validate_sampling_params.frequency_penalties.defined() &&
      !validate_sampling_params.presence_penalties.defined() &&
      !validate_sampling_params.repetition_penalties.defined() &&
      !validate_sampling_params.temperatures.defined() &&
      !validate_sampling_params.top_p.defined() &&
      !validate_sampling_params.top_k.defined() &&
      !validate_sampling_params.unique_token_ids.defined() &&
      !validate_sampling_params.unique_token_counts.defined() &&
      !validate_sampling_params.unique_token_ids_lens.defined();
  if (use_stable_greedy_validate_sampling) {
    if (!mtp_validate_greedy_indices_.defined() ||
        mtp_validate_greedy_indices_.numel() != total_num_val_tokens) {
      mtp_validate_greedy_indices_ = torch::arange(
          total_num_val_tokens,
          torch::TensorOptions().dtype(torch::kInt).device(runtime_.device_));
      mtp_validate_greedy_do_sample_ = torch::zeros(
          {total_num_val_tokens},
          torch::TensorOptions().dtype(torch::kBool).device(runtime_.device_));
    }
    validate_sampling_params.selected_token_idxes =
        mtp_validate_greedy_indices_;
    validate_sampling_params.sample_idxes = mtp_validate_greedy_indices_;
    validate_sampling_params.do_sample = mtp_validate_greedy_do_sample_;
    validate_sampling_params.all_random_sample = false;
    validate_sampling_params.all_greedy_sample = true;
  } else {
    runtime_.update_sampling_params(
        validate_sampling_params, num_val_tokens, total_num_val_tokens);
  }
#else
  runtime_.update_sampling_params(
      validate_sampling_params, num_val_tokens, total_num_val_tokens);
#endif

  scale_speculative_parallel_token_counts(input_params, num_val_tokens);

  std::vector<int32_t> accepted_prefix_lengths;
  if (runtime_.use_chunked_prefill_spec_verify_path()) {
    input_params.embedding.input_embedding = torch::Tensor();
    input_params.is_spec_verify = true;
    if (!input_params.attention.host.q_seq_lens.empty()) {
      std::vector<int32_t> q_cu_seq_lens_vec;
      q_cu_seq_lens_vec.reserve(input_params.meta.num_sequences + 1);
      q_cu_seq_lens_vec.emplace_back(0);
      for (int32_t q_len : input_params.attention.host.q_seq_lens) {
        q_cu_seq_lens_vec.emplace_back(q_cu_seq_lens_vec.back() + q_len);
      }
      input_params.attention.host.q_cu_seq_lens = std::move(q_cu_seq_lens_vec);
    }
    accepted_prefix_lengths.assign(num_sequences, 1);
    if (runtime_.embedding_cache_ != nullptr &&
        !input.input_params.embedding.embedding_ids.empty()) {
      accepted_prefix_lengths =
          runtime_.embedding_cache_->read_accepted_prefix_lengths(
              input.input_params.embedding.embedding_ids,
              input.input_params.embedding.request_ids);
    }
    // num_accepted_tokens must stay the true accepted count. The Qwen3.5 GDN
    // spec-verify kernel uses it to select which recurrent/conv checkpoint to
    // commit (checkpoint index = nat - 1); it is a logical checkpoint index,
    // not the conv_state physical history capacity. Clamping it commits a
    // stale checkpoint whenever 4+ tokens were accepted (see issue #2247).
    input_params.num_accepted_tokens_host.assign(
        accepted_prefix_lengths.begin(), accepted_prefix_lengths.end());
    if (!use_explicit_spec_verify_replay_update) {
      input_params.num_accepted_tokens =
          torch::tensor(accepted_prefix_lengths, token_options);
    }
  }

#if defined(USE_NPU)
  if (use_explicit_spec_verify_replay_update) {
    build_expanded_spec_verify_graph_host_input(input_params);

    auto& attention = input_params.attention;
    CHECK(attention.host.block_tables.defined());
    CHECK_EQ(attention.host.block_tables.dim(), 2);
    CHECK_EQ(attention.host.block_tables.size(0), num_sequences);
    CHECK_EQ(attention.host.block_tables.scalar_type(), torch::kInt32);
    const int64_t active_block_table_width =
        attention.host.block_tables.size(1);
    const int64_t verify_block_table_width =
        runtime_.spec_verify_block_table_width(attention.host.block_tables);
    if (active_block_table_width != verify_block_table_width) {
      torch::Tensor padded_block_tables = torch::zeros(
          {num_sequences, verify_block_table_width},
          attention.host.block_tables.options().device(torch::kCPU));
      padded_block_tables.narrow(1, 0, active_block_table_width)
          .copy_(attention.host.block_tables);
      attention.host.block_tables = std::move(padded_block_tables);
    }

    const bool initialize_stable_buffer =
        !spec_verify_attention_host_buffer_.defined();
    attention.attention_host_buffer = spec_verify_attention_host_buffer_;
    attention.attention_device_buffer = spec_verify_attention_device_buffer_;
    attention.attention_buffer_bytes = 0;
    attention.attention_buffer_capacity =
        spec_verify_attention_buffer_capacity_;
    attention.attention_buffer_owner = spec_verify_attention_buffer_owner_;

    const int64_t expanded_block_rows = static_cast<int64_t>(
        input_params.graph.expanded_kv_seq_lens_vec.size());
    CHECK_EQ(expanded_block_rows, total_num_val_tokens);
    const torch::Tensor dense_block_source =
        attention.host.block_tables.contiguous();
    std::vector<int32_t> expanded_block_tables_dense(
        static_cast<size_t>(expanded_block_rows * verify_block_table_width));
    const int32_t* block_row = dense_block_source.data_ptr<int32_t>();
    for (int64_t row = 0; row < expanded_block_rows; ++row) {
      const int64_t sequence_index = row / num_val_tokens;
      std::memcpy(
          expanded_block_tables_dense.data() + row * verify_block_table_width,
          block_row + sequence_index * verify_block_table_width,
          static_cast<size_t>(verify_block_table_width) * sizeof(int32_t));
    }
    torch::Tensor expanded_block_tables_flat;
    std::vector<AttentionInput::PackedIntInput> extra_int_inputs;
    extra_int_inputs.push_back({&buf.out_token_ids,
                                &validate_input.token_ids_host,
                                &validate_input.token_ids});
    extra_int_inputs.push_back({&buf.out_positions,
                                &validate_input.positions_host,
                                &validate_input.positions});
    if (!expanded_linear_state_ids.empty()) {
      extra_int_inputs.push_back(
          {&expanded_linear_state_ids,
           nullptr,
           &input_params.embedding.linear_state_indices});
    }
    if (!input_params.num_accepted_tokens_host.empty()) {
      extra_int_inputs.push_back({&accepted_prefix_lengths,
                                  nullptr,
                                  &input_params.num_accepted_tokens});
    }
    if (!input_params.graph.expanded_kv_seq_lens_vec.empty()) {
      extra_int_inputs.push_back({&input_params.graph.expanded_kv_seq_lens_vec,
                                  nullptr,
                                  &input_params.graph.expanded_kv_seq_lens});
    }
    extra_int_inputs.push_back(
        {&expanded_block_tables_dense, nullptr, &expanded_block_tables_flat});
    attention.rebuild_device_buffer(
        runtime_.device_,
        extra_int_inputs,
        initialize_stable_buffer
            ? AttentionInput::BufferReusePolicy::GROWABLE
            : AttentionInput::BufferReusePolicy::FIXED_CAPACITY);
    if (initialize_stable_buffer && num_sequences > 0) {
      // Source views are part of the explicit replay contract. Reserve enough
      // packed storage for any configured decode batch before the first graph
      // captures their addresses, so a later larger batch cannot relocate the
      // buffer and invalidate an already captured variant.
      const uint64_t max_sequences =
          static_cast<uint64_t>(runtime_.options_.max_seqs_per_batch());
      const uint64_t batch_scale =
          (max_sequences + static_cast<uint64_t>(num_sequences) - 1) /
          static_cast<uint64_t>(num_sequences);
      const uint64_t reserve_capacity =
          attention.attention_buffer_bytes * batch_scale;
      if (reserve_capacity > attention.attention_buffer_capacity) {
        attention.reserve_device_buffer_capacity(reserve_capacity,
                                                 runtime_.device_);
        attention.rebuild_device_buffer(
            runtime_.device_,
            extra_int_inputs,
            AttentionInput::BufferReusePolicy::FIXED_CAPACITY);
      }
    }
    spec_verify_attention_host_buffer_ = attention.attention_host_buffer;
    spec_verify_attention_device_buffer_ = attention.attention_device_buffer;
    spec_verify_attention_buffer_capacity_ =
        attention.attention_buffer_capacity;
    input_params.graph.expanded_block_tables = expanded_block_tables_flat.view(
        {expanded_block_rows, verify_block_table_width});
    layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
        input_params,
        input_params.graph.expanded_kv_seq_lens,
        input_params.graph.expanded_block_tables,
        input_params.graph.expanded_kv_seq_lens_vec,
        logical_block_size);
    input_params.graph.input_tokens_override = validate_input.token_ids;
    input_params.graph.spec_verify_source_addresses_stable = true;
  } else {
    std::vector<AttentionInput::PackedIntInput> extra_int_inputs;
    if (!expanded_linear_state_ids.empty()) {
      extra_int_inputs.reserve(1);
      extra_int_inputs.push_back(
          {&expanded_linear_state_ids,
           nullptr,
           &input_params.embedding.linear_state_indices});
    }
    input_params.attention.rebuild_device_buffer(runtime_.device_,
                                                 extra_int_inputs);
    if (runtime_.supports_explicit_spec_verify_replay_update()) {
      build_expanded_spec_verify_graph_input(
          input_params, runtime_.device_, logical_block_size);
    }
  }
#else
  input_params.attention.rebuild_device_buffer(runtime_.device_);
#endif
  validate_input.device_tensors_ready = true;
  // This metadata is independent of the in-flight final draft. Keep it on the
  // auxiliary stream and hand it to the compute stream with a device event.
#if defined(USE_NPU)
  input_params.graph.spec_verify_static_graph_tasks_prepared =
      use_explicit_spec_verify_replay_update && static_graph_tasks_prepared;
#endif
  if (record_ready_event) {
    finish_metadata_prepare(*runtime_.prepare_stream_, validate_input);
  }
}

bool MtpLegacyExecutor::prepare_static_mtp_graph_tasks_before_final_draft(
    const ForwardInput& input) {
#if defined(USE_NPU)
  if (!runtime_.should_use_explicit_spec_verify_replay_update(input) ||
      input.input_params.embedding.linear_state_ids.size() != 1 ||
      runtime_.embedding_cache_ == nullptr ||
      input.input_params.embedding.embedding_ids.empty()) {
    return false;
  }
  const auto& block_tables = input.input_params.attention.host.block_tables;
  if (!block_tables.defined() || block_tables.dim() != 2 ||
      block_tables.size(0) != 1) {
    return false;
  }
  const std::vector<int32_t> accepted_prefix_lengths =
      runtime_.embedding_cache_->read_accepted_prefix_lengths(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids);
  if (accepted_prefix_lengths.size() != 1) {
    return false;
  }
  const int64_t verify_block_table_width =
      runtime_.spec_verify_block_table_width(block_tables);
  const auto& kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  if (kv_seq_lens.size() != 1) {
    return false;
  }
  const int64_t spec_width = runtime_.options_.num_speculative_tokens() + 1;
  const int64_t base_kv_seq_len = kv_seq_lens.front();
  const int64_t spec_verify_max_kv_seq_len = base_kv_seq_len + spec_width - 1;
  const SpecVerifyGraphTaskSignal signal{
      .linear_state_id = input.input_params.embedding.linear_state_ids.front(),
      .num_accepted_tokens = accepted_prefix_lengths.front(),
      .spec_width = spec_width,
      .block_table_width = verify_block_table_width,
      .base_kv_seq_len = base_kv_seq_len,
      .max_kv_seq_len = spec_verify_max_kv_seq_len,
  };
  auto* llm_target = dynamic_cast<LLMWorkerImpl*>(runtime_.impl_.get());
  if (llm_target == nullptr) {
    return false;
  }
  return llm_target->prepare_static_mtp_graph_tasks(signal,
                                                    *runtime_.compute_stream_);
#else
  (void)input;
  return false;
#endif
}

void MtpLegacyExecutor::prepare_validate_inputs(
    const ForwardInput& input,
    ForwardInput& validate_input,
    const std::vector<int32_t>& per_seq_val_tokens) {
  c10::StreamGuard stream_guard = runtime_.prepare_stream_->set_stream_guard();
  validate_input = input;
  clear_ready_events(validate_input);
  validate_input.device_tensors_ready = false;
  auto& input_params = validate_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  torch::TensorOptions token_options = validate_input.token_ids.options();
  torch::TensorOptions position_options = validate_input.positions.options();

  const int32_t num_sequences = input_params.meta.num_sequences;
  CHECK_EQ(per_seq_val_tokens.size(), static_cast<size_t>(num_sequences))
      << "per_seq_val_tokens size mismatch with num_sequences";
  int32_t total_num_val_tokens = 0;
  int32_t max_val_tokens = 0;
  for (int32_t i = 0; i < num_sequences; ++i) {
    total_num_val_tokens += per_seq_val_tokens[static_cast<size_t>(i)];
    max_val_tokens =
        std::max(max_val_tokens, per_seq_val_tokens[static_cast<size_t>(i)]);
  }
  const int32_t logical_block_size = runtime_.logical_block_size();
  const bool positions_decoupled =
      runtime_.positions_are_decoupled_from_kv_length();
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  Slice<int32_t> token_ids = {
      input.token_ids_host.data_ptr<int32_t>(),
      static_cast<size_t>(input.token_ids_host.numel())};
  Slice<int32_t> positions = {
      input.positions_host.data_ptr<int32_t>(),
      static_cast<size_t>(input.positions_host.numel())};
  Slice<int32_t> kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  const bool use_atb_spec_kernel =
      ::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel() ||
      runtime_.use_chunked_prefill_spec_verify_path();
  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(total_num_val_tokens);
  buf.out_positions.reserve(total_num_val_tokens);
  buf.out_new_cache_slots.reserve(total_num_val_tokens);
  if (!use_atb_spec_kernel) {
    buf.out_kv_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_cu_seq_lens.reserve(total_num_val_tokens);
    buf.out_block_tables.reserve(static_cast<size_t>(total_num_val_tokens) *
                                 row_ctx.block_table_stride);
  }

  std::vector<int32_t> atb_kv_seq_lens_vec;
  std::vector<int32_t> atb_q_seq_lens_vec;
  std::vector<int32_t> atb_q_cu_seq_lens_vec;
  int32_t atb_kv_max_seq_len = 0;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t seq_val_tokens =
        per_seq_val_tokens[static_cast<size_t>(seq_id)];
    const int32_t start_position = positions[seq_id];
    const int32_t kv_len =
        specBuilder::calc_kv_len(kv_seq_lens, seq_id, /*offset=*/0);
    if (!positions_decoupled) {
      CHECK_EQ(start_position + 1, kv_len)
          << "validate position/kv_len mismatch, seq_id=" << seq_id
          << ", start_position=" << start_position << ", kv_len=" << kv_len;
    }

    for (int32_t val_idx = 0; val_idx < seq_val_tokens; ++val_idx) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.token_id = val_idx == 0 ? token_ids[seq_id] : -val_idx;
      row.position_offset = val_idx;
      row.append_kv_len = !use_atb_spec_kernel;
      row.append_q_len_one = !use_atb_spec_kernel;
      row.append_block_table = !use_atb_spec_kernel;
      specBuilder::append_decode_row(row_ctx, row, logical_block_size, buf);
    }

    if (use_atb_spec_kernel) {
      const int32_t kv_len_after_validation = kv_len + seq_val_tokens - 1;
      specBuilder::update_kv_seq_lens_and_max(
          atb_kv_seq_lens_vec, kv_len_after_validation, atb_kv_max_seq_len);
      specBuilder::append_q_seq_len(
          atb_q_seq_lens_vec, atb_q_cu_seq_lens_vec, seq_val_tokens);
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "validate positions/tokens mismatch";

  specBuilder::set_token_position_tensors(validate_input,
                                          buf.out_token_ids,
                                          buf.out_positions,
                                          token_options,
                                          position_options);
  if (!use_atb_spec_kernel) {
    input_params.meta.num_sequences = total_num_val_tokens;
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  } else {
    input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  }
  if (use_atb_spec_kernel) {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     max_val_tokens,
                                     std::move(atb_q_seq_lens_vec),
                                     std::move(atb_q_cu_seq_lens_vec),
                                     atb_kv_max_seq_len,
                                     std::move(atb_kv_seq_lens_vec));
  } else {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     1,
                                     std::move(buf.out_q_seq_lens),
                                     std::move(buf.out_q_cu_seq_lens),
                                     buf.meta.kv_max_seq_len,
                                     std::move(buf.out_kv_seq_lens),
                                     /*update_block_tables=*/true);
  }

  runtime_.update_sampling_params(
      validate_input.sampling_params, per_seq_val_tokens, total_num_val_tokens);

  for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
    token_num = total_num_val_tokens;
  }

#if defined(USE_NPU)
  const bool expand_python_mtp_linear_state_ids =
      ModelConfig::is_python_model_impl(runtime_.context_.get_model_impl()) &&
      !input_params.embedding.linear_state_ids.empty();
  std::vector<int32_t> expanded_linear_state_ids;
  if (expand_python_mtp_linear_state_ids) {
    CHECK_EQ(input_params.embedding.linear_state_ids.size(),
             static_cast<size_t>(num_sequences));
    expanded_linear_state_ids.reserve(
        static_cast<size_t>(total_num_val_tokens));
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      expanded_linear_state_ids.insert(
          expanded_linear_state_ids.end(),
          per_seq_val_tokens[static_cast<size_t>(seq_id)],
          input_params.embedding.linear_state_ids[static_cast<size_t>(seq_id)]);
    }
  }
#endif

  if (runtime_.use_chunked_prefill_spec_verify_path()) {
    input_params.embedding.input_embedding = torch::Tensor();
    input_params.is_spec_verify = true;
    if (!input_params.attention.host.q_seq_lens.empty()) {
      std::vector<int32_t> q_cu_seq_lens_vec;
      q_cu_seq_lens_vec.reserve(num_sequences + 1);
      q_cu_seq_lens_vec.emplace_back(0);
      for (int32_t q_len : input_params.attention.host.q_seq_lens) {
        q_cu_seq_lens_vec.emplace_back(q_cu_seq_lens_vec.back() + q_len);
      }
      input_params.attention.host.q_cu_seq_lens = std::move(q_cu_seq_lens_vec);
    }
    std::vector<int32_t> accepted_prefix_lengths(num_sequences, 1);
    if (runtime_.embedding_cache_ != nullptr &&
        !input.input_params.embedding.embedding_ids.empty()) {
      accepted_prefix_lengths =
          runtime_.embedding_cache_->read_accepted_prefix_lengths(
              input.input_params.embedding.embedding_ids,
              input.input_params.embedding.request_ids);
    }
    // num_accepted_tokens must stay the true accepted count: the Qwen3.5 GDN
    // spec-verify kernel commits the recurrent/conv checkpoint at index
    // nat - 1, so clamping it would commit a stale state whenever 4+ tokens
    // were accepted (see issue #2247). The conv1d kernel already clamps its
    // own physical conv_state read offset internally, and the tiling check
    // no longer rejects nat > segment length, so no host-side clamp is needed.
    input_params.num_accepted_tokens =
        torch::tensor(accepted_prefix_lengths, token_options);
    input_params.num_accepted_tokens_host.assign(
        accepted_prefix_lengths.begin(), accepted_prefix_lengths.end());
  }

#if defined(USE_NPU)
  std::vector<AttentionInput::PackedIntInput> extra_int_inputs;
  if (!expanded_linear_state_ids.empty()) {
    extra_int_inputs.reserve(1);
    extra_int_inputs.push_back({&expanded_linear_state_ids,
                                nullptr,
                                &input_params.embedding.linear_state_indices});
  }
  input_params.attention.rebuild_device_buffer(runtime_.device_,
                                               extra_int_inputs);
  if (runtime_.supports_explicit_spec_verify_replay_update()) {
    build_expanded_spec_verify_graph_input(
        input_params, runtime_.device_, logical_block_size);
  }
#else
  input_params.attention.rebuild_device_buffer(runtime_.device_);
#endif
  validate_input.device_tensors_ready = true;
  finish_metadata_prepare(*runtime_.prepare_stream_, validate_input);
}

void MtpLegacyExecutor::prepare_draft_inputs(const ForwardInput& input,
                                             ForwardInput& draft_input,
                                             int32_t position_offset) {
  c10::StreamGuard stream_guard = runtime_.prepare_stream_->set_stream_guard();
  draft_input = input;
  runtime_.prepare_draft_sampling(draft_input.sampling_params);
  clear_ready_events(draft_input);
  draft_input.device_tensors_ready = false;

  auto& input_params = draft_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t logical_block_size = runtime_.logical_block_size();
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  specBuilder::DecodeBuildBuffers buf;
  buf.out_positions.reserve(num_sequences);
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_new_cache_slots.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    specBuilder::RowSpec row;
    row.seq_id = seq_id;
    row.position_offset = position_offset;
    row.append_token = false;
    specBuilder::append_decode_row(row_ctx, row, logical_block_size, buf);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "draft kv slots/positions mismatch";

  torch::TensorOptions position_options = input.positions.options();
  set_positions_tensor(draft_input,
                       specBuilder::make_cpu_int_tensor(buf.out_positions),
                       position_options);
  specBuilder::update_input_params(
      input_params,
      buf,
      input_params.meta.q_max_seq_len,
      std::move(input_params.attention.host.q_seq_lens),
      std::move(input_params.attention.host.q_cu_seq_lens),
      buf.meta.kv_max_seq_len,
      std::move(buf.out_kv_seq_lens));
  if (runtime_.supports_explicit_spec_verify_replay_update()) {
    input_params.attention.host.q_cu_seq_lens.clear();
    input_params.attention.host.q_cu_seq_lens.reserve(
        input_params.meta.num_sequences + 1);
    input_params.attention.host.q_cu_seq_lens.emplace_back(0);
    for (int32_t i = 0; i < input_params.meta.num_sequences; ++i) {
      input_params.attention.host.q_cu_seq_lens.emplace_back(
          input_params.attention.host.q_cu_seq_lens.back() +
          input_params.get_q_seq_len(i));
    }
  }
  input_params.attention.rebuild_device_buffer(runtime_.device_);
#if defined(USE_NPU)
  // Later draft steps share the B cache variant.
  runtime_.draft_impl_->prepare_dp_ep_padding_on_stream(
      input_params, *runtime_.prepare_stream_);
#endif
  // token_ids is intentionally filled later from the previous draft output.
  draft_input.device_tensors_ready = false;

  // Positions/KV metadata do not depend on the in-flight draft result. Prepare
  // them concurrently; token ids and embeddings are filled on compute_stream.
  finish_metadata_prepare(*runtime_.prepare_stream_, draft_input);
}

SampleOutput MtpLegacyExecutor::validate(
    const SamplingParameters& sampling_params,
    const std::vector<ForwardOutput>& draft_outputs,
    const ForwardOutput& target_output,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths,
    const torch::Tensor& target_filter_mask,
    const torch::Tensor& target_filter_bitmask,
    const std::vector<uint8_t>& invalid_draft) {
  const int32_t num_target_tokens =
      target_output.sample_output.next_tokens.numel();
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  CHECK_EQ(num_target_tokens % num_val_tokens, 0);

  std::vector<torch::Tensor> draft_token_ids_steps;
  std::vector<torch::Tensor> draft_probs_steps;
  draft_token_ids_steps.reserve(draft_outputs.size());
  draft_probs_steps.reserve(draft_outputs.size());
  for (const auto& draft_output : draft_outputs) {
    draft_token_ids_steps.emplace_back(draft_output.sample_output.next_tokens);
    draft_probs_steps.emplace_back(draft_output.sample_output.probs);
  }

  DraftProposal draft_proposal = specBuilder::build_validate_proposal(
      draft_token_ids_steps,
      draft_probs_steps,
      /*draft_probs_required=*/
      draft_probs_required(runtime_.draft_sampling_mode_,
                           sampling_params.all_greedy_sample));
  return validate(sampling_params,
                  draft_proposal,
                  target_output,
                  num_speculative_tokens,
                  pruned_prefix_lengths,
                  target_filter_mask,
                  target_filter_bitmask,
                  invalid_draft);
}

SampleOutput MtpLegacyExecutor::validate(
    const SamplingParameters& sampling_params,
    const DraftProposal& draft_proposal,
    const ForwardOutput& target_output,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths,
    const torch::Tensor& target_filter_mask,
    const torch::Tensor& target_filter_bitmask,
    const std::vector<uint8_t>& invalid_draft) {
  const int32_t num_target_tokens =
      target_output.sample_output.next_tokens.numel();
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  CHECK_EQ(num_target_tokens % num_val_tokens, 0);
  const int32_t batch_size = num_target_tokens / num_val_tokens;
  const int32_t vocab_size = target_output.logits.size(/*dim=*/-1);
  const bool use_greedy_token_ids =
      !runtime_.requires_probability_based_validation() &&
      sampling_params.all_greedy_sample && !target_output.logprobs;

  using torch::indexing::None;
  using ISlice = torch::indexing::Slice;
  const bool step_major_validate_layout =
      runtime_.uses_step_major_validate_layout();
  torch::Tensor target_next_tokens = target_output.sample_output.next_tokens;
  torch::Tensor target_logits;
  torch::Tensor target_embeddings = target_output.sample_output.embeddings;
  if (step_major_validate_layout) {
    target_next_tokens = target_next_tokens.view({num_val_tokens, batch_size})
                             .transpose(/*dim0=*/0, /*dim1=*/1)
                             .contiguous()
                             .view({num_target_tokens});
    target_logits =
        target_output.logits.view({num_val_tokens, batch_size, vocab_size})
            .permute({1, 0, 2})
            .contiguous();
    if (target_embeddings.defined()) {
      target_embeddings =
          target_embeddings
              .view({num_val_tokens, batch_size, target_embeddings.size(-1)})
              .permute({1, 0, 2})
              .contiguous();
    }
  } else {
    target_logits =
        target_output.logits.view({batch_size, num_val_tokens, vocab_size});
    if (target_embeddings.defined()) {
      target_embeddings = target_embeddings.view(
          {batch_size, num_val_tokens, target_embeddings.size(-1)});
    }
  }
  torch::Tensor bonus_token_ids =
      target_next_tokens
          .index({"...", ISlice(num_val_tokens - 1, None, num_val_tokens)})
          .view({-1, 1});

  SampleOutput sample_output;
  if (use_greedy_token_ids) {
    torch::Tensor target_token_ids =
        target_next_tokens.view({batch_size, num_val_tokens});
    torch::Tensor target_draft_token_ids = target_token_ids.slice(
        /*dim=*/1, /*start=*/0, /*end=*/num_val_tokens - 1);
    torch::Tensor draft_token_ids = draft_proposal.token_ids();
    // Keep the original target ID block and let the verifier convert IDs in UB.
    // Target and bonus remain views of the same production storage.
#if defined(USE_NPU)
    if (target_draft_token_ids.device().type() !=
        c10::DeviceType::PrivateUse1) {
      draft_token_ids = draft_token_ids.to(target_draft_token_ids);
    }
#else
    draft_token_ids = draft_token_ids.to(target_draft_token_ids);
#endif
    auto [accepted_token_ids, masked_accepted_token_ids] =
        RejectionSampler::greedy_sample_from_token_ids(
            draft_token_ids,
            target_draft_token_ids,
            bonus_token_ids,
            /*mask_out_rejected_tokens=*/true);
    (void)accepted_token_ids;

    sample_output.next_tokens = masked_accepted_token_ids;
    sample_output.embeddings = target_embeddings;
  } else {
    if (target_filter_bitmask.defined()) {
      CHECK(target_output.filter_bitmask_applied_to_logits)
          << "packed target logits must be filtered by the target sampler";
      CHECK_EQ(target_filter_bitmask.dim(), 2)
          << "MTP JSON filter bitmask must be 2-D";
      CHECK_EQ(target_filter_bitmask.size(0), num_target_tokens)
          << "MTP JSON filter bitmask row count mismatch";
      CHECK_EQ(target_filter_bitmask.size(1), (vocab_size + 31) / 32)
          << "MTP JSON filter bitmask vocabulary width mismatch";
    } else if (target_filter_mask.defined()) {
      CHECK_EQ(target_filter_mask.dim(), 2)
          << "MTP JSON filter mask must be 2-D";
      CHECK_EQ(target_filter_mask.size(0), num_target_tokens)
          << "MTP JSON filter mask row count mismatch";
      CHECK_EQ(target_filter_mask.size(1), vocab_size)
          << "MTP JSON filter mask vocabulary mismatch";
      target_logits =
          target_logits +
          target_filter_mask.view({batch_size, num_val_tokens, vocab_size});
    }

    const torch::Tensor& draft_token_ids = draft_proposal.token_ids();
    torch::Tensor validation_draft_token_ids = draft_token_ids;
    if (!invalid_draft.empty()) {
      CHECK_EQ(invalid_draft.size(),
               static_cast<size_t>(batch_size * (num_val_tokens - 1)))
          << "MTP invalid draft mask shape mismatch";
      torch::Tensor invalid_mask =
          torch::tensor(
              invalid_draft,
              torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
              .view({batch_size, num_val_tokens - 1})
              .to(torch::kBool)
              .to(draft_token_ids.device());
      torch::Tensor target_sampled_tokens =
          target_output.sample_output.next_tokens
              .view({batch_size, num_val_tokens})
              .slice(/*dim=*/1, /*start=*/0, /*end=*/num_val_tokens - 1);
      validation_draft_token_ids =
          torch::where(invalid_mask,
                       target_sampled_tokens.to(draft_token_ids.device()),
                       draft_token_ids);
    }
    DraftProposal validation_proposal(std::move(validation_draft_token_ids),
                                      draft_proposal.draft_probs());
    sample_output = spec_verify::run_rejection_sampling(
        {.do_sample = sampling_params.do_sample,
         .all_random_sample = sampling_params.all_random_sample,
         .all_greedy_sample = sampling_params.all_greedy_sample},
        validation_proposal,
        target_logits,
        target_output,
        bonus_token_ids,
        runtime_.enable_fused_kernel_);

    if (!invalid_draft.empty()) {
      torch::Tensor target_sampled_tokens =
          target_output.sample_output.next_tokens.view(
              {batch_size, num_val_tokens});
      for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
        int32_t first_invalid = -1;
        for (int32_t draft_idx = 0; draft_idx < num_val_tokens - 1;
             ++draft_idx) {
          if (invalid_draft[static_cast<size_t>(seq_id * (num_val_tokens - 1) +
                                                draft_idx)] != 0) {
            first_invalid = draft_idx;
            break;
          }
        }
        if (first_invalid < 0) {
          continue;
        }

        torch::Tensor output_row = sample_output.next_tokens.select(0, seq_id);
        output_row.select(0, first_invalid)
            .copy_(target_sampled_tokens.index({seq_id, first_invalid}));
        if (first_invalid + 1 < num_val_tokens) {
          output_row
              .narrow(/*dim=*/0,
                      /*start=*/first_invalid + 1,
                      /*length=*/num_val_tokens - first_invalid - 1)
              .fill_(-1);
        }
      }
    }
    // process embedding
    sample_output.embeddings = target_embeddings;
  }

  if (pruned_prefix_lengths != nullptr) {
    // Build cut/keep masks once from pruned_prefix_lengths and reuse across
    // both helpers below, so we avoid re-uploading prefix_lengths and
    // rebuilding identical arange+eq+logical_and masks per call.
    const adaptive_pruning::PrunedPrefixMasks pruning_masks =
        adaptive_pruning::build_pruned_prefix_masks(
            *pruned_prefix_lengths,
            num_speculative_tokens,
            sample_output.next_tokens.device());
    sync_pruned_boundary_outputs(sample_output,
                                 target_output,
                                 batch_size,
                                 num_val_tokens,
                                 pruning_masks);
    apply_pruned_prefix_lengths(sample_output,
                                target_output.sample_output.next_tokens,
                                num_speculative_tokens,
                                pruning_masks);
  }

  return sample_output;
}

}  // namespace xllm
