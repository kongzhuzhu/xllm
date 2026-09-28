/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/kv_cache_transfer/kv_cache_transfer.h"
#include "core/framework/speculative/adaptive_speculative_controller.h"
#include "core/framework/speculative/embedding_cache.h"
#include "core/framework/speculative/mtp_async_state.h"
#include "core/runtime/speculative_worker_impl.h"

namespace xllm {

class MtpLegacyExecutor;
// Shared model, KV and accepted-state ownership. Decode policy belongs to a
// concrete worker; the optional compatibility executor never owns models/KV.
class MtpRuntime : public SpeculativeWorkerImpl {
  friend class MtpLegacyExecutor;

 public:
  MtpRuntime(const ParallelArgs& parallel_args,
             const torch::Device& device,
             const runtime::Options& options,
             WorkerType worker_type);

  ~MtpRuntime() override;

 protected:
  // For derived classes (e.g. Eagle3WorkerImpl) that need custom options for
  // target and draft models. `options` is passed to WorkerImpl (preserves
  // enable_schedule_overlap etc.), `target_options` / `draft_options` are used
  // to create the respective workers.
  MtpRuntime(const ParallelArgs& parallel_args,
             const torch::Device& device,
             const runtime::Options& options,
             const runtime::Options& target_options,
             const runtime::Options& draft_options,
             WorkerType worker_type,
             bool enable_adaptive_speculative_decode = false);

 public:
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override;

  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

#if defined(USE_NPU) || defined(USE_MLU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

  ForwardInput update_input_by_last_step_output(ForwardInput& inputs) override;
  ForwardInput update_input_by_last_step_output_for_schedule_overlap(
      ForwardInput& inputs) override;
  void prepare_work_before_execute(const ForwardInput& inputs,
                                   ForwardInput& processed_inputs) override;

 protected:
  void drain_pending_execution();
  MtpLegacyExecutor& legacy_executor();
  void retire_legacy_prelaunch();
  virtual void clear_unified_device_state() {}
  virtual bool supports_unified_python_mtp_graph(const ForwardInput&) const {
    return false;
  }
  std::optional<ForwardOutput> step_decode(const ForwardInput& input) override =
      0;
  std::optional<ForwardOutput> step_empty(const ForwardInput& input) override;

  // MTP composite: leaves own model-specific NPU input preparation.
  bool owns_npu_parallel_input_prepare() const override;

  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;

  // Hook for algorithm-specific draft output post-processing during decode.
  virtual void process_draft_sample_output(SampleOutput& sample_output);

  virtual void check_draft_input_embedding(const torch::Tensor& /*embedding*/,
                                           const std::string& /*phase*/) const {
  }
  virtual bool share_target_lm_head_with_draft() const { return true; }

  // PD separation: placeholder size for empty embedding slot. Default: 1x
  // hidden_size. Eagle3 overrides to 3 * target_hidden_size.
  virtual int64_t get_embedding_placeholder_size();
  bool should_use_separate_draft_kv_cache_shape() const;
  KVCacheShape draft_kv_cache_shape(
      const KVCacheShape& target_kv_cache_shape) const override;

  // prepare inputs for draft model at Prefill phase.
  void prepare_prefill_inputs(const ForwardInput& inputs,
                              ForwardInput& prefill_inputs);
  void prepare_draft_sampling(SamplingParameters& sampling_params) const;
  bool supports_explicit_spec_verify_replay_update() const;
  bool should_use_explicit_spec_verify_replay_update(
      const ForwardInput& input) const;
  // Returns true when the target model's spec-verify kernel requires the
  // validate width (val_tokens) to be identical across every sequence in the
  // batch. Currently Qwen3.5 GDN's FusedRecurrentGatedDeltaRule spec-verify
  // path has this constraint; other paths accept per-seq variable widths.
  // Kept separate from supports_explicit_spec_verify_replay_update() so the
  // two capabilities can diverge for future targets.
  bool requires_uniform_validate_width() const;
  // A DCP block-table entry covers one logical KV page per KV split. All
  // speculative cache-slot and expanded-attention metadata must use this
  // logical size rather than the per-rank physical allocator block size.
  int32_t logical_block_size() const;
  int64_t spec_verify_block_table_width(
      const torch::Tensor& block_tables) const;
  // Returns true when validation must use chunked-prefill to avoid the
  // FlashInfer batch-decode read-before-write race on the bonus token.
  bool use_chunked_prefill_spec_verify_path() const;
  bool uses_embedded_eagle3_draft() const;
  // Multiaxis RoPE positions can include a prompt-dependent offset and do not
  // identify the corresponding KV cache length.
  bool positions_are_decoupled_from_kv_length() const;
  bool requires_probability_based_validation() const;
  bool uses_step_major_validate_layout() const;
  void synchronize_embedded_eagle3_forward();
  std::optional<ForwardOutput> run_worker_no_sync(
      WorkerImpl& worker,
      const ForwardInput& input,
      ForwardInput& processed_input);

  void update_decode_step_input(
      ForwardInput& input,
      const std::vector<EmbeddingCache::DecodeState>& last_states) const;

  // Build draft-side input from cached target context at decode step start.
  void prepare_draft_extend_inputs(
      const ForwardInput& base_input,
      const std::vector<EmbeddingCache::DecodeState>& last_states,
      ForwardInput& extend_input,
      bool force_two_rows = false,
      bool wait_for_compute_stream = true);

  struct PendingTargetContext {
    uint64_t generation = 0;
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    // Both tensors stay on device.  A steady-state overlap step consumes them
    // by queueing gather/update ops behind rejection sampling on the same
    // stream.  They are materialized on CPU only when the batch shape/order
    // changes and the host cache fallback is required.
    torch::Tensor accepted_tokens;
    torch::Tensor accepted_tokens_host;
    torch::Tensor accepted_count_host;
    torch::Tensor accepted_embeddings;
    torch::Tensor base_positions;
    torch::Tensor base_kv_seq_lens;
    std::vector<uint8_t> json_constrained_rows;
    std::vector<size_t> failed_rows;
    StreamEventPtr ready_event;
  };

  void stage_target_context_write(const ForwardInput& input,
                                  const SampleOutput& validate_output,
                                  torch::Tensor base_positions,
                                  torch::Tensor base_kv_seq_lens,
                                  StreamEventPtr ready_event,
                                  torch::Tensor accepted_tokens_host,
                                  torch::Tensor accepted_count_host,
                                  std::vector<size_t> failed_rows);
  torch::Tensor acquire_accepted_tokens_host_buffer(
      const torch::Tensor& accepted_tokens);

  bool pending_target_context_matches(const ForwardInput& input) const;

  void prepare_mtp_bootstrap(const ForwardInput& input);
  bool device_target_context_ready_for_batch(const ForwardInput& input) const;
  void flush_pending_target_context(size_t keep_latest = 0);
  torch::Tensor snapshot_pending_target_tokens();

  bool adaptive_enabled() const;

 protected:
  // Rejection sampling produces accepted state on the compute stream.  Keep
  // that state device-resident so the next overlap task can be fully enqueued
  // without waiting for target verification to finish.
  // Multiple overlap turns may complete before the scheduler returns to the
  // host. Each context owns its device/host tensors and completion event until
  // its generation is flushed in submission order.
  PendingTargetContext pending_target_context_;
  std::deque<PendingTargetContext> pending_target_context_queue_;
  uint64_t next_pending_target_generation_ = 1;
  std::vector<int32_t> device_context_ready_embedding_ids_;
  std::vector<std::string> device_context_ready_request_ids_;
  // Non-overlap buffers reserve the configured batch capacity. Overlap leases
  // use bounded pools and retire only after their event and consumers finish.
  torch::Tensor accepted_tokens_host_buffer_;
  std::vector<torch::Tensor> accepted_tokens_host_pool_;

  // Classified once when the corresponding models are loaded. Decode-path
  // decisions only read these closed policies.
  mtp_async::TargetSpecVerifyMode target_spec_verify_mode_ =
      mtp_async::TargetSpecVerifyMode::GENERIC;
  mtp_async::CombinedDraftExecutionPath combined_draft_execution_path_ =
      mtp_async::CombinedDraftExecutionPath::UNSUPPORTED;

  // Allocated only when compatibility execution/empty-rank participation is
  // needed. This object borrows the model and cache owners above.
  std::unique_ptr<MtpLegacyExecutor> legacy_executor_;

 private:
  bool execution_drained_ = false;
};
}  // namespace xllm
