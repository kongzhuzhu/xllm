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

#include "core/runtime/mtp_runtime.h"

#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>

#include "common/metrics.h"
#include "core/runtime/mtp_legacy_executor.h"
#include "core/runtime/mtp_runtime_helpers.h"
#if defined(USE_NPU) || defined(USE_MLU)
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif
#include "core/framework/block/block_utils.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/framework/model/mtp_utils.h"
#include "core/framework/multimodal/mm_data.h"
#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#endif

#include "core/framework/speculative/draft_extend_input.h"
#include "core/framework/speculative/mtp_async_state.h"
#include "core/framework/speculative/spec_input_builder.h"
#include "core/framework/speculative/spec_verify.h"
#include "runtime/llm_worker_impl.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
using mtp_detail::broadcast_spec_tokens;
using mtp_detail::clear_ready_events;
using mtp_detail::finalize_output_on_stream;
using mtp_detail::finish_metadata_prepare;
using mtp_detail::record_current_metadata_ready_event;
using mtp_detail::run_worker_no_sync_impl;
using mtp_detail::should_broadcast_spec_tokens;
namespace {
constexpr int64_t kMaxSpecVerifyGraphUpdateBlockTableWidth = (1 << 15) - 1;
constexpr size_t kMaxPendingTargetContexts = 2;
torch::Tensor acquire_pinned_output_buffer(const torch::Tensor& source,
                                           std::vector<torch::Tensor>& pool) {
  // Every lease is a distinct TensorImpl/view. The pending context retains
  // it through event synchronization; serializers may retain it longer.
  // Unique storage therefore proves both producer and consumers retired.
  auto available =
      std::find_if(pool.begin(), pool.end(), [&source](const auto& buffer) {
        return buffer.storage().use_count() == 1 &&
               buffer.scalar_type() == source.scalar_type() &&
               buffer.numel() >= source.numel();
      });
  if (available == pool.end()) {
    available = std::find_if(pool.begin(), pool.end(), [](const auto& buffer) {
      return buffer.storage().use_count() == 1;
    });
  }
  torch::Tensor buffer;
  if (available != pool.end()) {
    buffer = *available;
  }
  if (!buffer.defined() || buffer.scalar_type() != source.scalar_type() ||
      buffer.numel() < source.numel()) {
    buffer =
        torch::empty({source.numel()},
                     source.options().device(torch::kCPU).pinned_memory(true));
    if (available != pool.end()) {
      *available = buffer;
    } else if (pool.size() < kMaxPendingTargetContexts + 2) {
      pool.emplace_back(buffer);
    }
  }
  // If all bounded pool slots are leased, this allocation belongs solely to
  // the current consumers. Never overwrite an older output to avoid a copy.
  return buffer.narrow(0, 0, source.numel()).view(source.sizes());
}

void clear_mla_prefixcache_workspace(ForwardInput& input) {
  auto& attention = input.input_params.attention.device;
  attention.history_compressed_kv = torch::Tensor();
  attention.history_k_rope = torch::Tensor();
}

torch::Tensor to_cpu_int_tensor_for_read(const torch::Tensor& values) {
  return safe_to(values.flatten(),
                 torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU),
                 false)
      .contiguous();
}

void replace_host_token_placeholders(ForwardInput& input,
                                     int32_t placeholder,
                                     const torch::Tensor& replacements,
                                     const torch::TensorOptions& token_options,
                                     bool refresh_device = true) {
  CHECK(replacements.defined())
      << "speculative replacement tokens must be defined";
  CHECK(input.token_ids_host.defined())
      << "token_ids_host must be defined before speculative token update";
  CHECK(input.token_ids_host.device().is_cpu())
      << "token_ids_host must stay on CPU";
  CHECK_EQ(input.token_ids_host.scalar_type(), torch::kInt)
      << "token_ids_host must be int32";

  input.device_tensors_ready = false;
  torch::Tensor replacement_cpu = to_cpu_int_tensor_for_read(replacements);
  int32_t* token_ids = input.token_ids_host.data_ptr<int32_t>();
  const size_t num_token_ids =
      static_cast<size_t>(input.token_ids_host.numel());
  Slice<int32_t> replacement_ids = {
      replacement_cpu.data_ptr<int32_t>(),
      static_cast<size_t>(replacement_cpu.numel())};

  size_t replacement_idx = 0;
  for (size_t i = 0; i < num_token_ids; ++i) {
    if (token_ids[i] != placeholder) {
      continue;
    }
    CHECK_LT(replacement_idx, replacement_ids.size())
        << "not enough speculative replacement tokens";
    token_ids[i] = replacement_ids[replacement_idx++];
  }
  CHECK_EQ(replacement_idx, replacement_ids.size())
      << "unused speculative replacement tokens";

  if (refresh_device) {
    input.token_ids =
        safe_to(input.token_ids_host, token_options, /*non_blocking=*/true);
    input.device_tensors_ready = true;
  }
}

runtime::Options mtp_target_options(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(false)
      .enable_graph_aux_hidden_states(true);
  return opts;
}

runtime::Options mtp_draft_options(const runtime::Options& options) {
  runtime::Options draft_options = options;
  draft_options.enable_schedule_overlap(false)
      .is_draft_engine(true)
      .num_decoding_tokens(1)
      .num_speculative_tokens(0)
      .enable_graph_aux_hidden_states(true);
  return draft_options;
}

ParallelArgs mtp_draft_parallel_args(const ParallelArgs& parallel_args,
                                     const runtime::Options& options) {
  if (!options.enable_mtp_draft_body_tp1()) {
    return parallel_args;
  }
  CHECK(parallel_args.single_rank_group_ != nullptr)
      << "MTP draft body TP1 requires a single-rank process group";
  ParallelArgs draft_args = parallel_args;
  draft_args.rank(0)
      .world_size(1)
      .dp_size(1)
      .ep_size(1)
      .cp_size(1)
      .tp_size(1)
      .sp_size(1);
  draft_args.mapping_data(nlohmann::json{});
  draft_args.process_group_ = parallel_args.single_rank_group_;
  draft_args.dp_local_process_group_ = parallel_args.single_rank_group_;
  draft_args.lm_head_group_ = parallel_args.tp_group_;
  draft_args.tp_group_ = parallel_args.single_rank_group_;
  draft_args.cp_group_ = parallel_args.single_rank_group_;
  draft_args.moe_ep_group_ = parallel_args.single_rank_group_;
  draft_args.moe_tp_group_ = parallel_args.single_rank_group_;
  return draft_args;
}
}  // namespace

MtpRuntime::MtpRuntime(const ParallelArgs& parallel_args,
                       const torch::Device& device,
                       const runtime::Options& options,
                       WorkerType worker_type)
    : MtpRuntime(parallel_args,
                 device,
                 options,
                 mtp_target_options(options),
                 mtp_draft_options(options),
                 worker_type,
                 /*enable_adaptive_speculative_decode=*/true) {}

MtpRuntime::MtpRuntime(const ParallelArgs& parallel_args,
                       const torch::Device& device,
                       const runtime::Options& options,
                       const runtime::Options& target_options,
                       const runtime::Options& draft_options,
                       WorkerType worker_type,
                       bool enable_adaptive_speculative_decode)
    : SpeculativeWorkerImpl(parallel_args,
                            device,
                            options,
                            target_options,
                            worker_type) {
  draft_impl_ = std::make_unique<LLMWorkerImpl>(
      mtp_draft_parallel_args(parallel_args, options),
      device,
      mtp_draft_options(draft_options));
  if (enable_adaptive_speculative_decode &&
      options.enable_adaptive_speculative_decode()) {
    adaptive_spec_controller_ =
        std::make_unique<AdaptiveSpeculativeController>(options);
  }
}

MtpRuntime::~MtpRuntime() {
  // Drain the outer worker before derived-owned input/context storage dies.
  drain_pending_execution();
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  legacy_executor_.reset();
}

void MtpRuntime::drain_pending_execution() {
  if (execution_drained_) {
    return;
  }
  folly::Promise<folly::Unit> barrier;
  auto future = barrier.getFuture();
  threadpool_.schedule([barrier = std::move(barrier)]() mutable {
    barrier.setValue(folly::unit);
  });
  std::move(future).get();
  // Destruction may run on an RPC thread with a different current device.
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  CHECK_EQ(compute_stream_->synchronize(), 0);
  CHECK_EQ(prepare_stream_->synchronize(), 0);
  execution_drained_ = true;
}

MtpLegacyExecutor& MtpRuntime::legacy_executor() {
  if (legacy_executor_ == nullptr) {
    legacy_executor_ = std::make_unique<MtpLegacyExecutor>(*this);
  }
  return *legacy_executor_;
}

void MtpRuntime::retire_legacy_prelaunch() {
  if (legacy_executor_ != nullptr) {
    legacy_executor_->retire_prelaunch();
  }
}

std::optional<ForwardOutput> MtpRuntime::step_empty(const ForwardInput& input) {
  return legacy_executor().step_empty(input);
}

bool MtpRuntime::init_model(const std::string& model_weights_path,
                            int32_t random_seed,
                            MasterStatus master_status) {
  // Load target model via base class
  bool result = true;
  const bool loading_target =
      impl_->get_status() == WorkerImpl::Status::UNINITIALIZED;
  if (loading_target) {
    result = SpeculativeWorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    result = draft_impl_->WorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  }

  if (impl_ != nullptr && impl_->get_status() == WorkerImpl::Status::LOADED) {
    context_ = impl_->context_;
    target_spec_verify_mode_ = mtp_async::classify_target_spec_verify_mode(
        context_.get_model_args().model_type());
  }

  if (draft_impl_ != nullptr &&
      draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    combined_draft_execution_path_ =
        mtp_async::classify_combined_draft_execution_path(
            draft_impl_->context_.get_model_args().model_type());
    const bool draft_owns_shared_weights =
        options_.enable_mtp_draft_body_tp1() &&
        combined_draft_execution_path_ ==
            mtp_async::CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION;
    // Qwen3.5 draft checkpoints contain complete embedding and LMHead weights.
    // Other MTP drafts retain their existing target-weight sharing contract;
    // only their transformer body is replicated with TP1 parallel arguments.
    if (!draft_owns_shared_weights) {
      const bool python_weights_shared =
          draft_impl_->share_weights_from(*impl_);
      if (!python_weights_shared) {
        const bool share_lm_head = share_target_lm_head_with_draft();
#if defined(USE_NPU)
        if (::xllm::KernelConfig::get_instance().npu_kernel_backend() !=
            "TORCH") {
          if (share_lm_head) {
            auto head = impl_->get_npu_lm_head();
            draft_impl_->set_npu_lm_head(head);
          }
          auto word_embedding = impl_->get_npu_word_embedding();
          if (impl_->has_restored_npu_word_embedding()) {
            draft_impl_->set_restored_npu_word_embedding(word_embedding);
          } else {
            draft_impl_->set_npu_word_embedding(word_embedding);
          }
        } else {
          if (share_lm_head) {
            auto head = impl_->get_lm_head();
            draft_impl_->set_lm_head(head);
          }
          auto word_embedding = impl_->get_word_embedding();
          draft_impl_->set_word_embedding(word_embedding);
        }
#else
        if (share_lm_head) {
          auto head = impl_->get_lm_head();
          draft_impl_->set_lm_head(head);
        }
        auto word_embedding = impl_->get_word_embedding();
        draft_impl_->set_word_embedding(word_embedding);
#endif
      }
    }
  }
#if defined(USE_NPU)
  if (result && supports_explicit_spec_verify_replay_update()) {
    CHECK_EQ(::xllm::KernelConfig::get_instance().npu_kernel_backend(), "TORCH")
        << "Qwen3.5 MTP only supports NPU Torch backend";
  }
#endif
  return result;
}

std::tuple<int64_t, int64_t> MtpRuntime::estimate_kv_cache_capacity() {
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);
  return estimate_kv_cache_capacity_with_draft(
      *draft_impl_, mtp_target_options(options_), mtp_draft_options(options_));
}

int64_t MtpRuntime::get_embedding_placeholder_size() {
  // DeepSeek-V4 MTP stashes the pre-hc_head 3D hidden flattened to
  // [num_tokens, hc_mult*hidden], so the cache placeholder must cover
  // hc_mult*hidden per row.
  if (impl_ != nullptr) {
    const ModelArgs& args = impl_->context_.get_model_args();
    return mtp_hidden_state_width(args);
  }
  return static_cast<int64_t>(embedding_size_);
}

bool MtpRuntime::should_use_separate_draft_kv_cache_shape() const {
  return uses_embedded_eagle3_draft();
}

KVCacheShape MtpRuntime::draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape) const {
  if (should_use_separate_draft_kv_cache_shape()) {
    return build_draft_kv_cache_shape(target_kv_cache_shape);
  }
  if (options_.enable_mtp_draft_body_tp1()) {
    return build_draft_kv_cache_shape(target_kv_cache_shape,
                                      /*draft_world_size=*/1);
  }
  return target_kv_cache_shape;
}

bool MtpRuntime::uses_embedded_eagle3_draft() const {
  if (impl_ == nullptr || draft_impl_ == nullptr ||
      impl_->get_status() == WorkerImpl::Status::UNINITIALIZED ||
      draft_impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    return false;
  }
  return ::xllm::uses_embedded_eagle3_draft(options_.speculative_algorithm(),
                                            impl_->context_.get_model_args());
}

bool MtpRuntime::positions_are_decoupled_from_kv_length() const {
  if (impl_ == nullptr) {
    return false;
  }
  const ModelArgs& target_args = impl_->context_.get_model_args();
  return !target_args.rope_scaling_mrope_section().empty();
}

bool MtpRuntime::requires_probability_based_validation() const {
  // Keep validation independent of graph/eager sampled-token layouts.
  return uses_embedded_eagle3_draft();
}

bool MtpRuntime::uses_step_major_validate_layout() const {
  return uses_embedded_eagle3_draft();
}

void MtpRuntime::synchronize_embedded_eagle3_forward() {
#if defined(USE_NPU)
  if (uses_embedded_eagle3_draft()) {
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize embedded Eagle3 forward, ret="
                     << ret;
  }
#endif
}

std::optional<ForwardOutput> MtpRuntime::run_worker_no_sync(
    WorkerImpl& worker,
    const ForwardInput& input,
    ForwardInput& processed_input) {
  std::optional<ForwardOutput> output = run_worker_no_sync_impl(
      worker, input, *prepare_stream_, *compute_stream_, processed_input);
  synchronize_embedded_eagle3_forward();
  if (uses_embedded_eagle3_draft()) {
    clear_mla_prefixcache_workspace(processed_input);
    if (output.has_value()) {
      for (const std::shared_ptr<ForwardInput>& retained_input :
           output->retained_inputs) {
        if (retained_input != nullptr) {
          clear_mla_prefixcache_workspace(*retained_input);
        }
      }
    }
  }
  return output;
}

bool MtpRuntime::supports_explicit_spec_verify_replay_update() const {
  if (target_spec_verify_mode_ ==
      mtp_async::TargetSpecVerifyMode::QWEN3_5_EXPANDED_VERIFY) {
    return true;
  }
  // The Python NPU paged-attention runner consumes expanded metadata and can
  // replay its ACL graph for DeepSeek MLA. The native target executor has no
  // corresponding MLA spec-verify graph path yet, so it remains generic.
  return target_spec_verify_mode_ ==
             mtp_async::TargetSpecVerifyMode::DEEPSEEK_V32_EXPANDED_VERIFY &&
         ModelConfig::is_python_model_impl(context_.get_model_impl());
}

bool MtpRuntime::requires_uniform_validate_width() const {
  // Currently only Qwen3.5's GDN spec-verify kernel requires uniform width;
  // this happens to coincide with the QWEN3_5_EXPANDED_VERIFY mode used by
  // supports_explicit_spec_verify_replay_update, but the two are semantically
  // distinct capabilities (graph-update capability vs. per-seq varlen kernel
  // support).
  return target_spec_verify_mode_ ==
         mtp_async::TargetSpecVerifyMode::QWEN3_5_EXPANDED_VERIFY;
}

bool MtpRuntime::should_use_explicit_spec_verify_replay_update(
    const ForwardInput& input) const {
#if defined(USE_NPU)
  const torch::Tensor& block_tables =
      input.input_params.attention.host.block_tables;
  if (!::xllm::ExecutionConfig::get_instance().enable_graph() ||
      !::xllm::ExecutionConfig::get_instance()
           .enable_graph_mode_decode_no_padding() ||
      !supports_explicit_spec_verify_replay_update() ||
      options_.num_speculative_tokens() <= 0 ||
      input.input_params.meta.num_sequences <= 0 ||
      options_.block_size() <= 0 || !block_tables.defined() ||
      block_tables.dim() != 2 ||
      block_tables.size(0) != input.input_params.meta.num_sequences) {
    return false;
  }
  const int64_t block_table_width = spec_verify_block_table_width(block_tables);
  if (block_table_width <= 0 ||
      block_table_width > kMaxSpecVerifyGraphUpdateBlockTableWidth) {
    return false;
  }
  const int64_t spec_width =
      static_cast<int64_t>(options_.num_speculative_tokens()) + 1;
  return kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
      spec_width, options_.block_size());
#else
  (void)input;
  return false;
#endif
}

int64_t MtpRuntime::spec_verify_block_table_width(
    const torch::Tensor& block_tables) const {
  CHECK(block_tables.defined() && block_tables.dim() == 2);
  CHECK_GT(options_.block_size(), 0);
  int64_t required_width = block_tables.size(1);
  if (impl_ != nullptr) {
    const int64_t declared_capacity =
        mtp_async::speculative_verify_block_table_capacity(
            impl_->context_.get_model_args().max_position_embeddings(),
            logical_block_size());
    CHECK_LE(required_width, declared_capacity)
        << "block table width exceeds the model position capacity";
    required_width = declared_capacity;
  }
  return required_width;
}

bool MtpRuntime::use_chunked_prefill_spec_verify_path() const {
  return target_spec_verify_mode_ ==
             mtp_async::TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL ||
         supports_explicit_spec_verify_replay_update();
}

int32_t MtpRuntime::logical_block_size() const {
  CHECK_GT(options_.block_size(), 0);
  const int32_t kv_split_size = parallel_args_.kv_split_size_effective();
  CHECK_GT(kv_split_size, 0);
  return options_.block_size() * kv_split_size;
}

bool MtpRuntime::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];
  // init_model() must run first so dtype_/embedding_size_ are initialized.
  embedding_cache_ = std::make_shared<EmbeddingCache>(num_blocks);
  if (embedding_cache_) {
    int64_t size = get_embedding_placeholder_size();
    if (size > 0) {
      embedding_cache_->set_placeholder(
          torch::zeros({size}, torch::dtype(dtype_).device(device_)));
    }
  }
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);
  prepare_hierarchy_kv_cache_transfers();

  bool target_allocated = true;
  const auto target_status = impl_->get_status();
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = impl_->allocate_kv_cache(kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }

  bool draft_allocated = true;
  const auto draft_status = draft_impl_->get_status();
  if (draft_status == WorkerImpl::Status::LOADED) {
    draft_allocated =
        draft_impl_->allocate_kv_cache(draft_kv_cache_shape(kv_cache_shape));
  } else {
    CHECK_EQ(draft_status, WorkerImpl::Status::READY);
  }

  const bool allocated = target_allocated && draft_allocated;
  if (allocated) {
    finalize_hierarchy_kv_cache_transfers();
  }
  return allocated;
}

#if defined(USE_NPU) || defined(USE_MLU)
bool MtpRuntime::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);
  prepare_hierarchy_kv_cache_transfers();

  if (kv_cache_transfer_ == nullptr) {
    kv_cache_transfer_ = std::make_shared<MooncakeKVCacheTransferDefault>(
        device_.index(),
        options_.transfer_listen_port(),
        device_,
        context_.get_model_args().model_type());

    int32_t device_id = device_.index();
    kv_cache_transfer_->initialize(device_id);
  }

  bool target_allocated = true;
  const auto target_status = impl_->get_status();
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = impl_->allocate_kv_cache_with_transfer(
        kv_cache_transfer_, kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }

  bool draft_allocated = true;
  const auto draft_status = draft_impl_->get_status();
  if (draft_status == WorkerImpl::Status::LOADED) {
    draft_allocated = draft_impl_->allocate_kv_cache_with_transfer(
        kv_cache_transfer_, draft_kv_cache_shape(kv_cache_shape));
  } else {
    CHECK_EQ(draft_status, WorkerImpl::Status::READY);
  }

  embedding_cache_ = std::make_shared<EmbeddingCache>(num_blocks);
  if (embedding_cache_) {
    int64_t size = get_embedding_placeholder_size();
    if (size > 0) {
      embedding_cache_->set_placeholder(
          torch::zeros({size}, torch::dtype(dtype_).device(device_)));
    }
  }
  const bool allocated = target_allocated && draft_allocated;
  if (allocated) {
    finalize_hierarchy_kv_cache_transfers();
  }
  return allocated;
}
#endif

ForwardInput MtpRuntime::update_input_by_last_step_output(
    ForwardInput& inputs) {
  return inputs;
}

ForwardInput MtpRuntime::update_input_by_last_step_output_for_schedule_overlap(
    ForwardInput& inputs) {
  update_json_object_states_by_last_step_output(inputs);
  sanitize_json_object_error_inputs(inputs);
  return update_input_by_last_step_output(inputs);
}

void MtpRuntime::prepare_work_before_execute(const ForwardInput& input,
                                             ForwardInput& processed_input) {
  // Composite skips CP prepare; leaves run it on their execution streams.
  SpeculativeWorkerImpl::prepare_work_before_execute(input, processed_input);
}

bool MtpRuntime::owns_npu_parallel_input_prepare() const { return false; }

std::optional<ForwardOutput> MtpRuntime::step_prefill(
    const ForwardInput& input) {
  retire_legacy_prelaunch();
  flush_pending_target_context();

  Timer timer;
  ForwardInput target_prepared;
  ForwardInput draft_prepared;

  // run the target model to get first token and hidden states
  ForwardOutput output =
      run_worker_no_sync(*impl_, input, target_prepared).value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // MTP path that depends on hidden states.
  ForwardInput prefill_input;
  prepare_prefill_inputs(input, prefill_input);

  // prepare input for draft model
  auto& embeddings = output.sample_output.embeddings;
  check_draft_input_embedding(embeddings, "prefill");

  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // Target prefill seeds the MTP decode cache. Under orthogonal CP x TP each
    // CP shard samples independently unless this token is synchronized across
    // both axes; caching divergent tokens makes the first decode input disagree
    // with the non-driver CP shard's DecodeState.
    if (should_broadcast_spec_tokens(
            parallel_args_,
            get_optimization_config().enable_spec_token_broadcast,
            input.sampling_params.all_greedy_sample)) {
      broadcast_spec_tokens(output.sample_output.next_tokens, parallel_args_);
    }
    if (embeddings.defined()) {
      prefill_input.input_params.embedding.input_embedding = embeddings.clone();
    }
    if (output.sample_output.next_tokens.defined()) {
      replace_host_token_placeholders(prefill_input,
                                      -1,
                                      output.sample_output.next_tokens,
                                      prefill_input.token_ids.options());
    }
    if (embeddings.defined() || output.sample_output.next_tokens.defined()) {
      record_current_metadata_ready_event(prefill_input, *compute_stream_);
    }
  }
  // generate kv cache for draft model
  timer.reset();
  ForwardOutput draft_output =
      run_worker_no_sync(*draft_impl_, prefill_input, draft_prepared).value();
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    process_draft_sample_output(draft_output.sample_output);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  if (input.sampling_params.selected_token_idxes.defined()) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    prepare_first_draft_inputs(*embedding_cache_, input, output);
  } else {
    clear_all_output_embeddings(output);
  }

  transfer_retained_inputs(output, draft_output);
  finalize_output_on_stream(
      output, *compute_stream_, enable_schedule_overlap());

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

void MtpRuntime::prepare_prefill_inputs(const ForwardInput& input,
                                        ForwardInput& prefill_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  prefill_input = input.to(device_, dtype_);
  prepare_draft_sampling(prefill_input.sampling_params);
  clear_ready_events(prefill_input);
  auto& input_params = prefill_input.input_params;
  // The Qwen draft is a pure full-attention model; without this cleanup the
  // target's recurrent slot metadata makes MTP prefill enter a stateful path
  // it has neither a validity mask nor a recurrent cache for.
  input_params.clear_linear_attention_state();
  auto& extra_token_ids = input_params.embedding.extra_token_ids;

  const torch::Tensor& token_ids = input.token_ids_host;
  Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                     static_cast<size_t>(token_ids.numel())};

  int32_t start_idx = 0;
  std::vector<int32_t> new_token_ids;
  new_token_ids.reserve(token_ids.numel());
  for (int32_t i = 0; i < input_params.meta.num_sequences; ++i) {
    int32_t q_len = input_params.get_q_seq_len(i);
    Slice<int32_t> tokens_ids_slice_i =
        tokens_ids_slice.slice(start_idx + 1, start_idx + q_len);
    start_idx += q_len;
    new_token_ids.insert(new_token_ids.end(),
                         tokens_ids_slice_i.begin(),
                         tokens_ids_slice_i.end());
    new_token_ids.emplace_back(extra_token_ids[i]);
  }
  prefill_input.device_tensors_ready = false;
  prefill_input.token_ids_host =
      specBuilder::make_cpu_int_tensor(new_token_ids);
  prefill_input.token_ids = safe_to(prefill_input.token_ids_host,
                                    prefill_input.positions.options(),
                                    /*non_blocking=*/true);
  prefill_input.device_tensors_ready = true;
  finish_metadata_prepare(*prepare_stream_, prefill_input);
}

void MtpRuntime::prepare_draft_sampling(
    SamplingParameters& sampling_params) const {
  if (draft_sampling_mode_ == DraftSamplingMode::GREEDY) {
    force_greedy_draft_sampling(sampling_params);
  } else {
    sampling_params.logprobs = false;
    sampling_params.max_top_logprobs = 0;
    // Qwen3.5 derives adaptive probabilities from logits instead.
    sampling_params.return_probs =
        !sampling_params.all_greedy_sample ||
        (adaptive_enabled() && !supports_explicit_spec_verify_replay_update());
  }
}

void MtpRuntime::prepare_mtp_bootstrap(const ForwardInput& decode_input) {
  const auto& embedding = decode_input.input_params.embedding;
  if (embedding.mtp_bootstrap_embeddings.defined()) {
    CHECK(decode_input.token_ids_host.defined())
        << "MTP bootstrap requires host token ids";
    CHECK(decode_input.token_ids_host.device().is_cpu())
        << "MTP bootstrap host token ids must be on CPU";
    CHECK_EQ(decode_input.token_ids_host.scalar_type(), torch::kInt)
        << "MTP bootstrap host token ids must be int32";

    torch::Tensor bootstrap_embeddings =
        safe_to(embedding.mtp_bootstrap_embeddings,
                torch::dtype(dtype_).device(device_));
    CHECK_EQ(bootstrap_embeddings.size(0),
             static_cast<int64_t>(embedding.mtp_bootstrap_row_idxes.size()))
        << "MTP bootstrap row count mismatch";

    Slice<int32_t> token_ids = {
        decode_input.token_ids_host.data_ptr<int32_t>(),
        static_cast<size_t>(decode_input.token_ids_host.numel())};
    for (int32_t i = 0;
         i < static_cast<int32_t>(embedding.mtp_bootstrap_row_idxes.size());
         ++i) {
      const int32_t row_idx = embedding.mtp_bootstrap_row_idxes[i];
      CHECK_GE(row_idx, 0) << "MTP bootstrap row index should be valid";
      CHECK_LT(row_idx, static_cast<int32_t>(embedding.embedding_ids.size()))
          << "MTP bootstrap row index exceeds embedding ids";
      CHECK_LT(row_idx, static_cast<int32_t>(embedding.request_ids.size()))
          << "MTP bootstrap row index exceeds request ids";
      CHECK_LT(static_cast<int64_t>(row_idx),
               decode_input.token_ids_host.numel())
          << "MTP bootstrap row index exceeds token ids";
      embedding_cache_->write_mtp_bootstrap_context(
          embedding.embedding_ids[row_idx],
          embedding.request_ids[row_idx],
          token_ids[row_idx],
          bootstrap_embeddings[i]);
    }
  }
}

void MtpRuntime::stage_target_context_write(const ForwardInput& input,
                                            const SampleOutput& validate_output,
                                            torch::Tensor base_positions,
                                            torch::Tensor base_kv_seq_lens,
                                            StreamEventPtr ready_event,
                                            torch::Tensor accepted_tokens_host,
                                            torch::Tensor accepted_count_host,
                                            std::vector<size_t> failed_rows) {
  PendingTargetContext context;
  context.generation = next_pending_target_generation_++;
  context.embedding_ids = input.input_params.embedding.embedding_ids;
  context.request_ids = input.input_params.embedding.request_ids;
  context.accepted_tokens = validate_output.next_tokens;
  context.accepted_tokens_host = std::move(accepted_tokens_host);
  context.accepted_count_host = std::move(accepted_count_host);
  context.accepted_embeddings = validate_output.embeddings;
  context.base_positions = std::move(base_positions);
  context.base_kv_seq_lens = std::move(base_kv_seq_lens);
  context.json_constrained_rows.reserve(input.json_object_states.size());
  for (const JsonObjectGrammarState& state : input.json_object_states) {
    context.json_constrained_rows.emplace_back(state.initialized() ? 1U : 0U);
  }
  context.failed_rows = std::move(failed_rows);
  context.ready_event = std::move(ready_event);
  if (pending_target_context_queue_.size() >= kMaxPendingTargetContexts) {
    flush_pending_target_context(/*keep_latest=*/kMaxPendingTargetContexts - 1);
  }
  pending_target_context_queue_.emplace_back(std::move(context));
  // Keep the newest context as the device-side fast-path view. The queue owns
  // the actual lifetimes and is drained in generation order by flush.
  pending_target_context_ = pending_target_context_queue_.back();
}

torch::Tensor MtpRuntime::acquire_accepted_tokens_host_buffer(
    const torch::Tensor& accepted_tokens) {
  CHECK(accepted_tokens.defined()) << "accepted tokens must be defined";
  CHECK_GT(accepted_tokens.numel(), 0) << "accepted tokens must not be empty";
  const int64_t required_capacity = accepted_tokens.numel();
  const int64_t configured_capacity =
      static_cast<int64_t>(options_.max_seqs_per_batch()) *
      (static_cast<int64_t>(options_.num_speculative_tokens()) + 1);
  if (enable_schedule_overlap()) {
    return acquire_pinned_output_buffer(accepted_tokens,
                                        accepted_tokens_host_pool_);
  }
  const bool needs_allocation =
      !accepted_tokens_host_buffer_.defined() ||
      accepted_tokens_host_buffer_.scalar_type() !=
          accepted_tokens.scalar_type() ||
      accepted_tokens_host_buffer_.numel() < required_capacity;
  if (needs_allocation) {
    const int64_t capacity = std::max(required_capacity, configured_capacity);
    accepted_tokens_host_buffer_ = torch::empty(
        {capacity},
        accepted_tokens.options().device(torch::kCPU).pinned_memory(true));
  }

  return accepted_tokens_host_buffer_
      .narrow(/*dim=*/0, /*start=*/0, required_capacity)
      .view(accepted_tokens.sizes());
}

bool MtpRuntime::pending_target_context_matches(
    const ForwardInput& input) const {
  if (pending_target_context_queue_.empty()) {
    return false;
  }
  const PendingTargetContext& context = pending_target_context_queue_.back();
  return context.failed_rows.empty() && context.accepted_tokens.defined() &&
         context.embedding_ids == input.input_params.embedding.embedding_ids &&
         context.request_ids == input.input_params.embedding.request_ids;
}

bool MtpRuntime::device_target_context_ready_for_batch(
    const ForwardInput& input) const {
  return device_context_ready_embedding_ids_ ==
             input.input_params.embedding.embedding_ids &&
         device_context_ready_request_ids_ ==
             input.input_params.embedding.request_ids;
}

torch::Tensor MtpRuntime::snapshot_pending_target_tokens() {
  // Retain storage, not a copy of its contents, until the D2H event completes.
  torch::Tensor host_tokens = pending_target_context_.accepted_tokens_host;
  CHECK(host_tokens.defined());
  flush_pending_target_context();
  return host_tokens.clone();
}

void MtpRuntime::flush_pending_target_context(size_t keep_latest) {
  while (pending_target_context_queue_.size() > keep_latest) {
    pending_target_context_ = std::move(pending_target_context_queue_.front());
    pending_target_context_queue_.pop_front();
    if (!pending_target_context_.accepted_tokens.defined()) {
      pending_target_context_ = PendingTargetContext();
      continue;
    }
    CHECK(pending_target_context_.ready_event == nullptr ||
          pending_target_context_.ready_event->synchronize())
        << "failed to wait for pending MTP target context generation "
        << pending_target_context_.generation;
    CHECK(embedding_cache_ != nullptr)
        << "embedding_cache_ must be initialized before target cache write";
    const int32_t num_speculative_tokens = options_.num_speculative_tokens();
    const int32_t num_validation_tokens = num_speculative_tokens + 1;
    const torch::Tensor accepted_tokens =
        pending_target_context_.accepted_tokens_host.contiguous();
    CHECK_EQ(accepted_tokens.numel() % num_validation_tokens, 0)
        << "MTP validation output width mismatch";
    const torch::Tensor output_tokens =
        accepted_tokens.view({-1, num_validation_tokens});
    if (!pending_target_context_.json_constrained_rows.empty()) {
      CHECK_EQ(pending_target_context_.json_constrained_rows.size(),
               static_cast<size_t>(output_tokens.size(0)))
          << "MTP JSON row metadata mismatch";
    }
    int64_t constrained_accepted = 0;
    int64_t plain_accepted = 0;
    int64_t constrained_draft = 0;
    int64_t plain_draft = 0;
    const torch::Tensor& accepted_count_host =
        pending_target_context_.accepted_count_host;
    if (accepted_count_host.defined()) {
      CHECK(accepted_count_host.device().is_cpu())
          << "accepted count host state must be on CPU";
      CHECK_EQ(accepted_count_host.dim(), 1)
          << "accepted count host state must be a vector";
      CHECK_EQ(accepted_count_host.size(0), output_tokens.size(0))
          << "accepted count host state batch mismatch";
      CHECK(accepted_count_host.scalar_type() == torch::kInt ||
            accepted_count_host.scalar_type() == torch::kLong)
          << "accepted count host state must be int32 or int64";
    }
    const int32_t* counts_i32 =
        accepted_count_host.defined() &&
                accepted_count_host.scalar_type() == torch::kInt
            ? accepted_count_host.const_data_ptr<int32_t>()
            : nullptr;
    const int64_t* counts_i64 =
        accepted_count_host.defined() &&
                accepted_count_host.scalar_type() == torch::kLong
            ? accepted_count_host.const_data_ptr<int64_t>()
            : nullptr;
    const int64_t count_stride =
        accepted_count_host.defined() ? accepted_count_host.stride(0) : 0;
    CHECK(output_tokens.device().is_cpu());
    CHECK(output_tokens.scalar_type() == torch::kLong ||
          output_tokens.scalar_type() == torch::kInt);
    const int64_t* tokens_i64 = output_tokens.scalar_type() == torch::kLong
                                    ? output_tokens.const_data_ptr<int64_t>()
                                    : nullptr;
    const int32_t* tokens_i32 = output_tokens.scalar_type() == torch::kInt
                                    ? output_tokens.const_data_ptr<int32_t>()
                                    : nullptr;
    for (int64_t sequence_idx = 0; sequence_idx < output_tokens.size(0);
         ++sequence_idx) {
      const bool constrained =
          !pending_target_context_.json_constrained_rows.empty() &&
          pending_target_context_
                  .json_constrained_rows[static_cast<size_t>(sequence_idx)] !=
              0U;
      int64_t accepted = 0;
      if (accepted_count_host.defined()) {
        const int64_t offset = sequence_idx * count_stride;
        accepted =
            counts_i32 != nullptr ? counts_i32[offset] : counts_i64[offset];
        CHECK_GE(accepted, 0);
        CHECK_LE(accepted, num_speculative_tokens);
      } else {
        int64_t rejected = 0;
        for (int32_t token_idx = 0; token_idx < num_validation_tokens;
             ++token_idx) {
          const int64_t offset =
              sequence_idx * num_validation_tokens + token_idx;
          const int64_t token =
              tokens_i64 != nullptr ? tokens_i64[offset] : tokens_i32[offset];
          if (token < 0) {
            ++rejected;
          }
        }
        accepted = num_speculative_tokens -
                   std::min<int64_t>(rejected, num_speculative_tokens);
      }
      if (constrained) {
        constrained_accepted += accepted;
        constrained_draft += num_speculative_tokens;
      } else {
        plain_accepted += accepted;
        plain_draft += num_speculative_tokens;
      }
    }
    COUNTER_ADD(speculative_num_accepted_tokens_constrained_total,
                constrained_accepted);
    COUNTER_ADD(speculative_num_accepted_tokens_plain_total, plain_accepted);
    COUNTER_ADD(speculative_num_draft_tokens_constrained_total,
                constrained_draft);
    COUNTER_ADD(speculative_num_draft_tokens_plain_total, plain_draft);
    if (pending_target_context_.failed_rows.empty()) {
      embedding_cache_->write_target_context(
          pending_target_context_.embedding_ids,
          pending_target_context_.request_ids,
          pending_target_context_.accepted_tokens_host,
          pending_target_context_.accepted_embeddings,
          pending_target_context_.accepted_count_host,
          options_.num_speculative_tokens());
    } else {
      CHECK(pending_target_context_.request_ids.empty() ||
            pending_target_context_.request_ids.size() ==
                pending_target_context_.embedding_ids.size())
          << "target context request ids must match embedding ids";
      std::unordered_set<size_t> failed_row_set(
          pending_target_context_.failed_rows.begin(),
          pending_target_context_.failed_rows.end());
      CHECK_EQ(failed_row_set.size(),
               pending_target_context_.failed_rows.size())
          << "target context failed rows must be unique";

      std::vector<int32_t> failed_embedding_ids;
      failed_embedding_ids.reserve(pending_target_context_.failed_rows.size());
      for (const size_t failed_row : pending_target_context_.failed_rows) {
        CHECK_LT(failed_row, pending_target_context_.embedding_ids.size())
            << "target context failed row exceeds embedding ids";
        failed_embedding_ids.emplace_back(
            pending_target_context_.embedding_ids[failed_row]);
      }
      embedding_cache_->clear(failed_embedding_ids);

      for (size_t sequence_index = 0;
           sequence_index < pending_target_context_.embedding_ids.size();
           ++sequence_index) {
        if (failed_row_set.contains(sequence_index)) {
          continue;
        }
        const std::vector<int32_t> row_embedding_ids = {
            pending_target_context_.embedding_ids[sequence_index]};
        const std::vector<std::string> row_request_ids =
            pending_target_context_.request_ids.empty()
                ? std::vector<std::string>()
                : std::vector<std::string>{
                      pending_target_context_.request_ids[sequence_index]};
        embedding_cache_->write_target_context(
            row_embedding_ids,
            row_request_ids,
            pending_target_context_.accepted_tokens_host.narrow(
                /*dim=*/0, /*start=*/sequence_index, /*length=*/1),
            pending_target_context_.accepted_embeddings.dim() == 2
                ? pending_target_context_.accepted_embeddings.narrow(
                      /*dim=*/0, /*start=*/sequence_index * 2, /*length=*/2)
                : pending_target_context_.accepted_embeddings.narrow(
                      /*dim=*/0, /*start=*/sequence_index, /*length=*/1),
            pending_target_context_.accepted_count_host.defined()
                ? pending_target_context_.accepted_count_host.narrow(
                      /*dim=*/0, /*start=*/sequence_index, /*length=*/1)
                : torch::Tensor(),
            options_.num_speculative_tokens());
      }
    }
    pending_target_context_ = PendingTargetContext();
  }
  if (!pending_target_context_queue_.empty()) {
    pending_target_context_ = pending_target_context_queue_.back();
  }
}

bool MtpRuntime::adaptive_enabled() const {
  return adaptive_spec_controller_ != nullptr &&
         adaptive_spec_controller_->enabled();
}

void MtpRuntime::process_draft_sample_output(SampleOutput& sample_output) {
  if (draft_sampling_mode_ == DraftSamplingMode::GREEDY) {
    sample_output.probs = torch::Tensor();
  }
}

void MtpRuntime::update_decode_step_input(
    ForwardInput& input,
    const std::vector<EmbeddingCache::DecodeState>& last_states) const {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "decode context state count mismatch";
  const bool enable_cache_correction = enable_schedule_overlap();
  const bool positions_decoupled = positions_are_decoupled_from_kv_length();

  std::vector<int32_t> token_ids_vec;
  std::vector<int32_t> kv_seq_lens_vec;
  token_ids_vec.reserve(num_sequences);
#if defined(USE_NPU)
  kv_seq_lens_vec.reserve(num_sequences);
#else
  kv_seq_lens_vec.reserve(num_sequences + 1);
#endif

  const torch::Tensor& token_ids_cpu = input.token_ids_host;
  const torch::Tensor& positions_cpu = input.positions_host;
  Slice<int32_t> input_token_ids = {token_ids_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(token_ids_cpu.numel())};
  Slice<int32_t> input_positions = {positions_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(positions_cpu.numel())};
  std::vector<int32_t> positions_vec;
  positions_vec.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size())
        << "decode context token seq_id out of range, seq_id=" << seq_id;
    CHECK_LT(static_cast<size_t>(seq_id), input_positions.size())
        << "decode context position seq_id out of range, seq_id=" << seq_id;
    const EmbeddingCache::DecodeState& state = last_states[seq_id];
    const int32_t input_token_id = input_token_ids[seq_id];
    const bool input_is_fake_token = input_token_id < 0;
    const bool use_cache_correction =
        enable_cache_correction && input_is_fake_token && state.valid;
    const bool use_fake_context =
        enable_cache_correction && input_is_fake_token && !state.valid;
    const int32_t position_offset =
        use_cache_correction ? state.position_offset : 0;
    int32_t current_position = input_positions[seq_id] + position_offset;
    int32_t current_kv_len = specBuilder::calc_kv_len(
        input.input_params.attention.host.kv_seq_lens, seq_id, position_offset);
    int32_t expected_kv_len =
        positions_decoupled ? current_kv_len : current_position + 1;
    if (use_chunked_prefill_spec_verify_path()) {
      const torch::Tensor& block_tables =
          input.input_params.attention.host.block_tables;
      if (block_tables.defined() && block_tables.dim() == 2 &&
          seq_id < block_tables.size(0)) {
        const int32_t allocated_kv_len =
            static_cast<int32_t>(block_tables.size(1)) * logical_block_size();
        const int32_t validate_width = options_.num_speculative_tokens() + 1;
        const int32_t max_valid_position = allocated_kv_len - validate_width;
        if (current_position > max_valid_position) {
          CHECK_GT(allocated_kv_len, 0)
              << "decode context has empty block table, seq_id=" << seq_id;
          CHECK_GE(max_valid_position, 0)
              << "decode context block table is too small for validation, "
              << "seq_id=" << seq_id
              << ", allocated_kv_len=" << allocated_kv_len
              << ", validate_width=" << validate_width;
          CHECK_LE(current_position - max_valid_position,
                   options_.num_speculative_tokens() + 1)
              << "decode context position exceeds allocated blocks, seq_id="
              << seq_id << ", current_position=" << current_position
              << ", current_kv_len=" << current_kv_len
              << ", allocated_kv_len=" << allocated_kv_len;
          current_position = max_valid_position;
          expected_kv_len = current_position + 1;
          current_kv_len = std::min(current_kv_len, expected_kv_len);
        }
      }
    }
    if (use_chunked_prefill_spec_verify_path() &&
        current_kv_len < expected_kv_len) {
      // Qwen3.5/MiMo MTP can receive a scheduler KV length that has not yet
      // caught up with the speculative placeholder resolved into
      // current_position. Normalize only the lag explainable by the current
      // speculative step.
      CHECK_LE(expected_kv_len - current_kv_len,
               options_.num_speculative_tokens() + 1)
          << "decode context kv_len lag is too large, seq_id=" << seq_id
          << ", current_position=" << current_position
          << ", current_kv_len=" << current_kv_len;
      current_kv_len = expected_kv_len;
    }
    if (use_chunked_prefill_spec_verify_path() &&
        current_kv_len > expected_kv_len) {
      // The first decode step can carry the prompt KV length while the decode
      // position is still initialized to zero. Align the position to the KV
      // context before building the MTP draft input.
      current_position = current_kv_len - 1;
      expected_kv_len = current_kv_len;
    }

    CHECK_EQ(expected_kv_len, current_kv_len)
        << "decode context position/kv_len mismatch, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;

    token_ids_vec.emplace_back((use_cache_correction || use_fake_context)
                                   ? state.token_id
                                   : input_token_id);
    positions_vec.emplace_back(current_position);
    specBuilder::append_seq_len_by_layout(kv_seq_lens_vec, current_kv_len);
  }

  input.token_ids_host = specBuilder::make_cpu_int_tensor(token_ids_vec);
  input.positions_host = specBuilder::make_cpu_int_tensor(positions_vec);
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  input.device_tensors_ready = false;
}

void MtpRuntime::prepare_draft_extend_inputs(
    const ForwardInput& base_input,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    ForwardInput& extend_input,
    bool force_two_rows,
    bool wait_for_compute_stream) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  // Regular draft preparation may consume tensors produced by the previous
  // compute. The placeholder-only first-draft prelaunch has no such dependency.
  if (wait_for_compute_stream) {
    prepare_stream_->wait_stream(*compute_stream_);
  }
  extend_input = base_input;
  prepare_draft_sampling(extend_input.sampling_params);
  clear_ready_events(extend_input);
  extend_input.device_tensors_ready = false;
  auto& input_params = extend_input.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;

  const bool dp_enabled = parallel_args_.dp_size() > 1;
  const bool use_chunked_prefill =
      ::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel();
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "draft extend state count mismatch";

  const int32_t logical_block_size = this->logical_block_size();
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(base_input);
  torch::TensorOptions token_options = extend_input.token_ids.options();
  torch::TensorOptions position_options = extend_input.positions.options();
  Slice<int32_t> token_ids = {
      base_input.token_ids_host.data_ptr<int32_t>(),
      static_cast<size_t>(base_input.token_ids_host.numel())};

  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(num_sequences * 2);
  buf.out_positions.reserve(num_sequences * 2);
  buf.out_new_cache_slots.reserve(num_sequences * 2);
  buf.out_kv_seq_lens.reserve(num_sequences * (use_chunked_prefill ? 1 : 2));
  buf.out_q_seq_lens.reserve(num_sequences * (use_chunked_prefill ? 1 : 2));
  buf.out_q_cu_seq_lens.reserve(num_sequences * 2);
  if (!use_chunked_prefill) {
    buf.out_block_tables.reserve(static_cast<size_t>(num_sequences) * 2 *
                                 row_ctx.block_table_stride);
  }
  std::vector<torch::Tensor> expanded_embeddings;
  std::vector<int32_t> selected_row_idx;
  expanded_embeddings.reserve(num_sequences * 2);
  selected_row_idx.reserve(num_sequences);

  auto to_worker_device = [this](const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.device() == device_) {
      return tensor;
    }
    return tensor.to(device_);
  };

  torch::Tensor placeholder = embedding_cache_->embedding_placeholder();
  CHECK(placeholder.defined())
      << "embedding placeholder must be initialized for fake draft context";
  placeholder = to_worker_device(placeholder);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    auto add_row = [&](int32_t token_id,
                       int32_t position_offset,
                       const torch::Tensor& embedding) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.token_id = token_id >= 0 ? token_id : 0;
      row.position_offset = position_offset;
      row.append_kv_len = !use_chunked_prefill;
      row.append_q_len_one = !use_chunked_prefill;
      row.append_block_table = !use_chunked_prefill;
      specBuilder::append_decode_row(row_ctx, row, logical_block_size, buf);
      if (embedding.defined()) {
        expanded_embeddings.emplace_back(to_worker_device(embedding));
      } else {
        expanded_embeddings.emplace_back(placeholder);
      }
    };

    EmbeddingCache::DecodeState state = last_states[seq_id];
    const int32_t current_token_id = token_ids[seq_id];
    if (!state.valid || state.token_id != current_token_id) {
      state = EmbeddingCache::DecodeState();
      state.token_id = current_token_id >= 0 ? current_token_id : 0;
    }
    if (use_chunked_prefill) {
      int32_t prev_token_id = state.prev_token_id;
      torch::Tensor prev_embedding = state.prev_embedding;
      const bool prev_is_placeholder = prev_token_id < 0;
      if (prev_is_placeholder) {
        prev_token_id = current_token_id >= 0 ? current_token_id : 0;
        prev_embedding = torch::Tensor();
      }
      add_row(prev_token_id, /*position_offset=*/-1, prev_embedding);
      if (prev_is_placeholder) {
        // Redirect to padding block 0 to avoid overwriting correct KV cache.
        buf.out_new_cache_slots.back() = 0;
      }
      add_row(state.token_id, /*position_offset=*/0, state.embedding);
      specBuilder::append_seq_len_by_layout(buf.out_q_seq_lens, 2);
      const int32_t kv_len = specBuilder::calc_kv_len(
          base_input.input_params.attention.host.kv_seq_lens,
          seq_id,
          /*offset=*/0);
      specBuilder::update_kv_seq_lens_and_max(
          buf.out_kv_seq_lens, kv_len, buf.meta.kv_max_seq_len);
      selected_row_idx.emplace_back(2 * seq_id + 1);
      continue;
    }
    // Keep DP draft-extend rows uniform. Empty DP ranks skip draft preparation,
    // so this path must not depend on a row-count collective reached by only
    // active ranks.
    const bool use_two_rows =
        force_two_rows || dp_enabled || state.all_draft_accepted;
    if (use_two_rows) {
      int32_t prev_token_id = state.prev_token_id;
      int32_t prev_position_offset = -1;
      torch::Tensor prev_embedding = state.prev_embedding;
      const bool prev_is_placeholder = prev_token_id < 0;
      if (prev_is_placeholder) {
        // Embedded Eagle3 on DP needs uniform draft-extend rows. On the first
        // decode step there is no real previous target token yet, so use the
        // current verifier hidden state instead of falling back to the draft
        // model's token embedding.
        prev_token_id = state.token_id;
        prev_embedding = state.embedding;
      }
      CHECK_GE(prev_token_id, 0)
          << "Eagle/MTP draft extend previous row requires a real token";
      add_row(prev_token_id, prev_position_offset, prev_embedding);
      if (prev_is_placeholder) {
        // Redirect to padding block 0 to avoid overwriting correct KV cache.
        buf.out_new_cache_slots.back() = 0;
      }
    }

    selected_row_idx.emplace_back(
        static_cast<int32_t>(expanded_embeddings.size()));
    add_row(state.token_id, /*position_offset=*/0, state.embedding);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "draft extend slots/positions mismatch";
  CHECK_EQ(expanded_embeddings.size(), buf.out_positions.size())
      << "draft extend embeddings/positions mismatch";

  specBuilder::set_token_position_tensors(extend_input,
                                          buf.out_token_ids,
                                          buf.out_positions,
                                          token_options,
                                          position_options);
  if (use_chunked_prefill) {
    input_params.meta.num_sequences = num_sequences;
    input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
    std::vector<int32_t> q_cu_seq_lens_vec;
    q_cu_seq_lens_vec.reserve(buf.out_q_seq_lens.size());
    int32_t cumulative_q_len = 0;
    for (int32_t q_len : buf.out_q_seq_lens) {
      cumulative_q_len += q_len;
      q_cu_seq_lens_vec.emplace_back(cumulative_q_len);
    }
    specBuilder::update_input_params(input_params,
                                     buf,
                                     /*q_max_seq_len=*/2,
                                     std::move(buf.out_q_seq_lens),
                                     std::move(q_cu_seq_lens_vec),
                                     buf.meta.kv_max_seq_len,
                                     std::move(buf.out_kv_seq_lens),
                                     /*update_block_tables=*/false);
  } else {
    input_params.meta.num_sequences =
        static_cast<int32_t>(buf.out_positions.size());
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    specBuilder::update_input_params(input_params,
                                     buf,
                                     1,
                                     std::move(buf.out_q_seq_lens),
                                     std::move(buf.out_q_cu_seq_lens),
                                     buf.meta.kv_max_seq_len,
                                     std::move(buf.out_kv_seq_lens),
                                     /*update_block_tables=*/true);
  }
  if (supports_explicit_spec_verify_replay_update()) {
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
  input_params.attention.rebuild_device_buffer(device_);

  input_params.embedding.input_embedding = torch::stack(expanded_embeddings);
  check_draft_input_embedding(input_params.embedding.input_embedding,
                              "decode extend");

  if (!input_params.parallel.dp_global_token_nums.empty()) {
    if (use_chunked_prefill) {
      scale_parallel_token_counts(input_params.parallel, /*multiplier=*/2);
    } else if (dp_enabled) {
      constexpr int32_t num_extend_tokens = 2;
      scale_parallel_token_counts(input_params.parallel, num_extend_tokens);
    } else if (input_params.parallel.dp_global_token_nums.size() == 1) {
      input_params.parallel.dp_global_token_nums[0] =
          static_cast<int32_t>(buf.out_positions.size());
    }
  }
  input_params.expert.eplb_decode_token_mask = eplb::expand_decode_token_mask(
      input_params.expert.eplb_decode_token_mask,
      static_cast<int32_t>(buf.out_positions.size()) / num_sequences);

#if defined(USE_NPU)
  // The extend layout is the 2B cache variant during steady overlap decode.
  draft_impl_->prepare_dp_ep_padding_on_stream(input_params, *prepare_stream_);
#endif

  auto& params = extend_input.sampling_params;
  torch::TensorOptions idx_options =
      params.selected_token_idxes.defined()
          ? params.selected_token_idxes.options()
          : torch::dtype(torch::kInt).device(device_);
  if (use_chunked_prefill || dp_enabled || force_two_rows) {
    // These layouts always append two rows per sequence and select the second
    // row.  Build the tiny control tensor directly on device; copying a
    // temporary pinned CPU tensor forces its allocator to synchronize before
    // the asynchronous H2D has completed.
    params.selected_token_idxes = torch::arange(
        /*start=*/1,
        /*end=*/2 * num_sequences,
        /*step=*/2,
        idx_options);
  } else {
    params.selected_token_idxes =
        safe_to(specBuilder::make_cpu_int_tensor(selected_row_idx),
                idx_options,
                /*non_blocking=*/true);
  }
  if (!params.sample_idxes.defined()) {
    // This control tensor is always the identity mapping. Generate it directly
    // on device instead of allocating a short-lived pinned H2D source.
    params.sample_idxes = torch::arange(
        /*start=*/0, /*end=*/num_sequences, idx_options);
  }
  extend_input.device_tensors_ready = true;
  finish_metadata_prepare(*prepare_stream_, extend_input);
}

}  // namespace xllm
