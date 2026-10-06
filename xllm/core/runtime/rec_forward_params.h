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

#include "core/framework/model/rec_model_params.h"
#include "core/runtime/forward_params.h"

namespace xllm {

class RecForwardInput;

struct StepDecodeMeta {
  int32_t batch_size = 0;
  int32_t beam_width = 1;
  int32_t current_round = 0;
  int32_t total_round = 0;
  // [batch_size * beam_width, n_kv_heads, step_rounds, head_dim]
  std::vector<int64_t> full_kv_shape;
  std::vector<int32_t> decode_positions_vec;
};

namespace detail {

bool unpack_from_input_host_buffer(const RecForwardInput& input,
                                   const torch::Device& device,
                                   torch::ScalarType dtype,
                                   RecForwardInput& output,
                                   bool materialize_device_buffer);

}  // namespace detail

class RecForwardInput final {
 public:
  RecForwardInput() = default;
  RecForwardInput(const RecForwardInput&) = delete;
  RecForwardInput& operator=(const RecForwardInput&) = delete;
  RecForwardInput(RecForwardInput&&) = default;
  RecForwardInput& operator=(RecForwardInput&&) = default;

  RecForwardInput clone() const {
    RecForwardInput inputs;
    inputs.token_ids = token_ids;
    inputs.positions = positions;
    inputs.token_ids_host = token_ids_host;
    inputs.positions_host = positions_host;
    inputs.input_params = input_params.clone();
    inputs.sampling_params = sampling_params;
    inputs.decoder_sampling_params = decoder_sampling_params;
    inputs.step_decode = step_decode;
    inputs.transfer_kv_infos = transfer_kv_infos;
    inputs.input_host_sample_count = input_host_sample_count;
    inputs.sample_sequence_ids = sample_sequence_ids;
    inputs.sample_prior_output_rows = sample_prior_output_rows;
    inputs.json_object_states = json_object_states;
    inputs.json_object_state_snapshots = json_object_state_snapshots;
    inputs.runtime = runtime;
    return inputs;
  }

  RecForwardInput to(const torch::Device& device,
                     torch::ScalarType dtype) const {
    if (runtime.device_tensors_ready) {
      return clone();
    }
    if (runtime.input_host_buffer_has_layout) {
      RecForwardInput buffer_inputs;
      const bool materialize_device_buffer =
          ExecutionConfig::get_instance().use_contiguous_input_buffer() &&
          detail::supports_contiguous_forward_input_buffer(device);
      if (detail::unpack_from_input_host_buffer(
              *this, device, dtype, buffer_inputs, materialize_device_buffer)) {
        if (buffer_inputs.runtime.device_tensors_ready) {
          return buffer_inputs;
        }
        return buffer_inputs.to(device, dtype);
      }
    }
    RecForwardInput inputs;
    inputs.token_ids_host = host_token_ids();
    inputs.positions_host = host_positions();
    inputs.token_ids = safe_to(inputs.token_ids_host, device, true);
    inputs.positions = detail::normalize_positions_for_device(
        safe_to(inputs.positions_host, device, true));
    inputs.input_params = input_params.to(device);
    inputs.sampling_params = sampling_params.to(device, dtype);
    inputs.decoder_sampling_params = decoder_sampling_params.to(device, dtype);
    inputs.step_decode = step_decode;
    inputs.transfer_kv_infos = transfer_kv_infos;
    inputs.input_host_sample_count = input_host_sample_count;
    inputs.sample_sequence_ids = sample_sequence_ids;
    inputs.sample_prior_output_rows = sample_prior_output_rows;
    inputs.json_object_states = json_object_states;
    inputs.json_object_state_snapshots = json_object_state_snapshots;
    inputs.runtime = runtime;
    inputs.runtime.device_tensors_ready = true;
    return inputs;
  }

  const torch::Tensor& host_token_ids() const {
    return token_ids_host.defined() ? token_ids_host : token_ids;
  }

  const torch::Tensor& host_positions() const {
    return positions_host.defined() ? positions_host : positions;
  }

  const StepDecodeMeta* step_meta() const {
    return step_decode ? &*step_decode : nullptr;
  }

  bool has_step_meta() const { return step_decode.has_value(); }

  torch::Tensor token_ids;
  torch::Tensor positions;
  torch::Tensor token_ids_host;
  torch::Tensor positions_host;
  mutable RecModelParams input_params;
  SamplingParameters sampling_params;
  SamplingParameters decoder_sampling_params;
  std::optional<StepDecodeMeta> step_decode;
  std::vector<TransferKVInfo> transfer_kv_infos;
  std::vector<std::string> sample_sequence_ids;
  std::vector<int32_t> sample_prior_output_rows;
  std::vector<JsonObjectGrammarState> json_object_states;
  std::vector<JsonObjectGrammarSnapshot> json_object_state_snapshots;
  int32_t input_host_sample_count = kUnknownPackedSampleCount;
  ForwardRuntimeState runtime;
};

}  // namespace xllm
