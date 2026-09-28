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

#include "core/runtime/py_attention_metadata.h"

#include <pybind11/stl.h>
#include <torch/python.h>

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

#include "core/framework/model/model_input_params.h"
#include "core/layers/common/attention_metadata.h"
#include "core/util/pybind_helper.h"

namespace py = pybind11;

namespace xllm {
namespace {

void copy_stable_tensor(torch::Tensor& destination,
                        const torch::Tensor& source,
                        const char* name) {
  TORCH_CHECK(destination.defined() == source.defined(),
              "MTP graph metadata field ",
              name,
              " changed definedness during replay");
  if (!source.defined()) {
    return;
  }
  TORCH_CHECK(destination.sizes() == source.sizes(),
              "MTP graph metadata field ",
              name,
              " changed shape from ",
              destination.sizes(),
              " to ",
              source.sizes());
  TORCH_CHECK(destination.scalar_type() == source.scalar_type(),
              "MTP graph metadata field ",
              name,
              " changed dtype");
  TORCH_CHECK(destination.device() == source.device(),
              "MTP graph metadata field ",
              name,
              " changed device");
  destination.copy_(source);
}

void copy_stable_optional_tensor(std::optional<torch::Tensor>& destination,
                                 const std::optional<torch::Tensor>& source,
                                 const char* name) {
  TORCH_CHECK(destination.has_value() == source.has_value(),
              "MTP graph metadata field ",
              name,
              " changed optional presence during replay");
  if (source.has_value()) {
    copy_stable_tensor(*destination, *source, name);
  }
}

struct PythonObjectHolder final {
  explicit PythonObjectHolder(py::object value) : value(std::move(value)) {}

  ~PythonObjectHolder() { clear_python_object(value); }

  py::object value;
};

}  // namespace

PythonAttentionMetadata::PythonAttentionMetadata(py::object value)
    : object_holder_(std::make_shared<PythonObjectHolder>(std::move(value))) {}

py::object PythonAttentionMetadata::value() const {
  return std::static_pointer_cast<PythonObjectHolder>(object_holder_)->value;
}

void register_attention_metadata_views(py::module_& module) {
  py::class_<PyExpandedDecodeMetadataView>(module, "ExpandedDecodeMetadataView")
      .def_property_readonly("enabled", &PyExpandedDecodeMetadataView::enabled)
      .def_property_readonly("kv_seq_lens",
                             &PyExpandedDecodeMetadataView::kv_seq_lens)
      .def_property_readonly("block_table",
                             &PyExpandedDecodeMetadataView::block_table)
      .def_property_readonly("paged_kv_indptr",
                             &PyExpandedDecodeMetadataView::paged_kv_indptr)
      .def_property_readonly("paged_kv_indices",
                             &PyExpandedDecodeMetadataView::paged_kv_indices)
      .def_property_readonly(
          "paged_kv_last_page_len",
          &PyExpandedDecodeMetadataView::paged_kv_last_page_len)
      .def_property_readonly(
          "paged_attention_tiling_data",
          &PyExpandedDecodeMetadataView::paged_attention_tiling_data)
      .def_property_readonly("kv_seq_lens_host",
                             &PyExpandedDecodeMetadataView::kv_seq_lens_host)
      .def_property_readonly(
          "kv_seq_lens_host_values",
          &PyExpandedDecodeMetadataView::kv_seq_lens_host_values);

  py::class_<PyAttentionMetadataView>(module, "AttentionMetadataView")
      .def_property("prepared_attention_state",
                    &PyAttentionMetadataView::prepared_attention_state,
                    &PyAttentionMetadataView::set_prepared_attention_state)
      .def_property_readonly("slot_mapping",
                             &PyAttentionMetadataView::slot_mapping)
      .def_property_readonly("local_slot_mapping",
                             &PyAttentionMetadataView::local_slot_mapping)
      .def_property_readonly("kv_split_size",
                             &PyAttentionMetadataView::kv_split_size)
      .def_property_readonly("kv_split_rank",
                             &PyAttentionMetadataView::kv_split_rank)
      .def_property_readonly("has_kv_shard",
                             &PyAttentionMetadataView::has_kv_shard)
      .def_property_readonly("paged_kv_indptr",
                             &PyAttentionMetadataView::paged_kv_indptr)
      .def_property_readonly("paged_kv_indices",
                             &PyAttentionMetadataView::paged_kv_indices)
      .def_property_readonly("paged_kv_last_page_len",
                             &PyAttentionMetadataView::paged_kv_last_page_len)
      .def_property_readonly("qo_indptr", &PyAttentionMetadataView::qo_indptr)
      .def_property_readonly("q_cu_seq_lens",
                             &PyAttentionMetadataView::q_cu_seq_lens)
      .def_property_readonly(
          "q_cu_seq_lens_host_values",
          &PyAttentionMetadataView::q_cu_seq_lens_host_values)
      .def_property_readonly("kv_cu_seq_lens",
                             &PyAttentionMetadataView::kv_cu_seq_lens)
      .def_property_readonly("kv_seq_lens_host",
                             &PyAttentionMetadataView::kv_seq_lens_host)
      .def_property_readonly("kv_seq_lens_host_values",
                             &PyAttentionMetadataView::kv_seq_lens_host_values)
      .def_property_readonly("q_seq_lens_host",
                             &PyAttentionMetadataView::q_seq_lens_host)
      .def_property_readonly("multi_block_tables",
                             &PyAttentionMetadataView::multi_block_tables)
      .def_property_readonly("block_table",
                             &PyAttentionMetadataView::block_table)
      .def_property_readonly("kv_seq_lens",
                             &PyAttentionMetadataView::kv_seq_lens)
      .def_property_readonly("linear_state_indices",
                             &PyAttentionMetadataView::linear_state_indices)
      .def_property_readonly("has_initial_state",
                             &PyAttentionMetadataView::has_initial_state)
      .def_property_readonly(
          "dp_execution_token_counts",
          &PyAttentionMetadataView::dp_execution_token_counts)
      .def_property_readonly("dp_global_sequence_nums",
                             &PyAttentionMetadataView::dp_global_sequence_nums)
      .def_property_readonly("dp_is_decode",
                             &PyAttentionMetadataView::dp_is_decode)
      .def_property_readonly("q_seq_lens", &PyAttentionMetadataView::q_seq_lens)
      .def_property_readonly("expanded_decode_metadata",
                             &PyAttentionMetadataView::expanded_decode_metadata)
      .def_property_readonly("max_query_len",
                             &PyAttentionMetadataView::max_query_len)
      .def_property_readonly("max_seq_len",
                             &PyAttentionMetadataView::max_seq_len)
      .def_property("dsa_metadata",
                    &PyAttentionMetadataView::dsa_metadata,
                    &PyAttentionMetadataView::set_dsa_metadata)
      .def_property("dsa_positions",
                    &PyAttentionMetadataView::dsa_positions,
                    &PyAttentionMetadataView::set_dsa_positions)
      .def_property("dsa_cos_sin",
                    &PyAttentionMetadataView::dsa_cos_sin,
                    &PyAttentionMetadataView::set_dsa_cos_sin)
      .def_property("dsa_c4_cos_sin",
                    &PyAttentionMetadataView::dsa_c4_cos_sin,
                    &PyAttentionMetadataView::set_dsa_c4_cos_sin)
      .def_property("dsa_c128_cos_sin",
                    &PyAttentionMetadataView::dsa_c128_cos_sin,
                    &PyAttentionMetadataView::set_dsa_c128_cos_sin)
      .def_property("dsa_graph_block_table_cols",
                    &PyAttentionMetadataView::dsa_graph_block_table_cols,
                    &PyAttentionMetadataView::set_dsa_graph_block_table_cols)
      .def_property("dsa_graph_mode",
                    &PyAttentionMetadataView::dsa_graph_mode,
                    &PyAttentionMetadataView::set_dsa_graph_mode)
      .def_property_readonly("is_prefill", &PyAttentionMetadataView::is_prefill)
      .def_property_readonly("is_chunked_prefill",
                             &PyAttentionMetadataView::is_chunked_prefill)
      .def_property_readonly("is_mixed", &PyAttentionMetadataView::is_mixed)
      .def_property_readonly("is_spec_verify",
                             &PyAttentionMetadataView::is_spec_verify)
      .def("update_from", &PyAttentionMetadataView::update_from)
      .def("clone_for_graph", &PyAttentionMetadataView::clone_for_graph);
}

PyExpandedDecodeMetadataView::PyExpandedDecodeMetadataView(
    std::shared_ptr<layer::AttentionMetadata> metadata)
    : metadata_(std::move(metadata)) {}

bool PyExpandedDecodeMetadataView::enabled() const {
  return metadata().enabled;
}

py::object PyExpandedDecodeMetadataView::kv_seq_lens() const {
  return optional_tensor(metadata().kv_seq_lens);
}

py::object PyExpandedDecodeMetadataView::block_table() const {
  return optional_tensor(metadata().block_table);
}

py::object PyExpandedDecodeMetadataView::paged_kv_indptr() const {
  return optional_tensor(metadata().paged_kv_indptr);
}

py::object PyExpandedDecodeMetadataView::paged_kv_indices() const {
  return optional_tensor(metadata().paged_kv_indices);
}

py::object PyExpandedDecodeMetadataView::paged_kv_last_page_len() const {
  return optional_tensor(metadata().paged_kv_last_page_len);
}

py::object PyExpandedDecodeMetadataView::paged_attention_tiling_data() const {
  return optional_tensor(metadata().paged_attention_tiling_data);
}

py::object PyExpandedDecodeMetadataView::kv_seq_lens_host() const {
  return optional_tensor(metadata().kv_seq_lens_host);
}

const std::vector<int32_t>&
PyExpandedDecodeMetadataView::kv_seq_lens_host_values() const {
  return metadata().kv_seq_lens_host_vec;
}

const layer::ExpandedDecodeMetadata& PyExpandedDecodeMetadataView::metadata()
    const {
  return metadata_->expanded_decode;
}

PyAttentionMetadataView::PyAttentionMetadataView(
    std::shared_ptr<layer::AttentionMetadata> metadata)
    : metadata_(std::move(metadata)),
      kv_seq_lens_host_(
          make_host_int32_view(metadata_, metadata_->kv_seq_lens_vec)),
      q_seq_lens_host_(
          make_host_int32_view(metadata_, metadata_->q_seq_lens_vec)) {}

PyAttentionMetadataView::PyAttentionMetadataView(
    std::shared_ptr<layer::AttentionMetadata> metadata,
    const ModelInputParams& params)
    : PyAttentionMetadataView(std::move(metadata)) {
  multi_block_tables_ = params.multi_block_tables;
  linear_state_indices_ = params.embedding.linear_state_indices;
  // Python model kernels consume materialized execution rows. Empty DP ranks
  // therefore contribute the worker-created dummy row instead of zero rows.
  dp_execution_token_counts_ = params.parallel.dp_global_token_nums;
  dp_global_sequence_nums_ = params.parallel.dp_global_sequence_nums;
  for (int32_t& count : dp_execution_token_counts_) {
    if (count == 0) {
      count = 1;
    }
  }
  dp_is_decode_ = params.parallel.dp_is_decode;
}

const torch::Tensor& PyAttentionMetadataView::slot_mapping() const {
  return metadata_->slot_mapping;
}

py::object PyAttentionMetadataView::local_slot_mapping() const {
  if (metadata_->kv_shard_batch_metadata == nullptr) {
    return py::none();
  }
  return optional_tensor(
      metadata_->kv_shard_batch_metadata->local_slot_mapping);
}

int32_t PyAttentionMetadataView::kv_split_size() const {
  if (metadata_->kv_shard_batch_metadata == nullptr) {
    return 1;
  }
  return metadata_->kv_shard_batch_metadata->kv_split_size;
}

int32_t PyAttentionMetadataView::kv_split_rank() const {
  if (metadata_->kv_shard_batch_metadata == nullptr) {
    return 0;
  }
  return metadata_->kv_shard_batch_metadata->kv_split_rank;
}

bool PyAttentionMetadataView::has_kv_shard() const {
  return metadata_->kv_shard_batch_metadata != nullptr;
}

const torch::Tensor& PyAttentionMetadataView::paged_kv_indptr() const {
  return metadata_->paged_kv_indptr;
}

const torch::Tensor& PyAttentionMetadataView::paged_kv_indices() const {
  return metadata_->paged_kv_indices;
}

const torch::Tensor& PyAttentionMetadataView::paged_kv_last_page_len() const {
  return metadata_->paged_kv_last_page_len;
}

py::object PyAttentionMetadataView::qo_indptr() const {
  return optional_tensor(metadata_->qo_indptr);
}

py::object PyAttentionMetadataView::q_cu_seq_lens() const {
  return optional_tensor(metadata_->q_cu_seq_lens);
}

const std::vector<int64_t>& PyAttentionMetadataView::q_cu_seq_lens_host_values()
    const {
  return metadata_->q_cu_seq_lens_host_vec;
}

py::object PyAttentionMetadataView::kv_cu_seq_lens() const {
  return optional_tensor(metadata_->kv_cu_seq_lens);
}

py::object PyAttentionMetadataView::kv_seq_lens_host() const {
  return optional_tensor(kv_seq_lens_host_);
}

const std::vector<int32_t>& PyAttentionMetadataView::kv_seq_lens_host_values()
    const {
  return metadata_->kv_seq_lens_vec;
}

py::object PyAttentionMetadataView::block_table() const {
  return optional_tensor(metadata_->block_table);
}

py::object PyAttentionMetadataView::kv_seq_lens() const {
  return optional_tensor(metadata_->kv_seq_lens);
}

py::object PyAttentionMetadataView::linear_state_indices() const {
  return optional_tensor(linear_state_indices_);
}

py::object PyAttentionMetadataView::has_initial_state() const {
  return optional_tensor(metadata_->has_initial_states);
}

const std::vector<int32_t>& PyAttentionMetadataView::dp_execution_token_counts()
    const {
  return dp_execution_token_counts_;
}

const std::vector<int32_t>& PyAttentionMetadataView::dp_global_sequence_nums()
    const {
  return dp_global_sequence_nums_;
}

const std::vector<int32_t>& PyAttentionMetadataView::dp_is_decode() const {
  return dp_is_decode_;
}

py::object PyAttentionMetadataView::q_seq_lens() const {
  return optional_tensor(metadata_->q_seq_lens);
}

py::object PyAttentionMetadataView::q_seq_lens_host() const {
  return optional_tensor(q_seq_lens_host_);
}

py::list PyAttentionMetadataView::multi_block_tables() const {
  py::list tables;
  for (const torch::Tensor& table : multi_block_tables_) {
    tables.append(optional_tensor(table));
  }
  return tables;
}

PyExpandedDecodeMetadataView PyAttentionMetadataView::expanded_decode_metadata()
    const {
  return PyExpandedDecodeMetadataView(metadata_);
}

int64_t PyAttentionMetadataView::max_query_len() const {
  return metadata_->max_query_len;
}

int64_t PyAttentionMetadataView::max_seq_len() const {
  return metadata_->max_seq_len;
}

py::object PyAttentionMetadataView::prepared_attention_state() const {
  if (!prepared_attention_holder_) {
    return py::none();
  }
  return std::static_pointer_cast<PythonObjectHolder>(
             prepared_attention_holder_)
      ->value;
}

void PyAttentionMetadataView::set_prepared_attention_state(py::object value) {
  if (value.is_none()) {
    prepared_attention_holder_.reset();
    return;
  }
  prepared_attention_holder_ =
      std::make_shared<PythonObjectHolder>(std::move(value));
}

py::object PyAttentionMetadataView::dsa_metadata() const {
  if (!dsa_metadata_holder_) {
    return py::none();
  }
  return std::static_pointer_cast<PythonObjectHolder>(dsa_metadata_holder_)
      ->value;
}

void PyAttentionMetadataView::set_dsa_metadata(py::object value) {
  if (value.is_none()) {
    dsa_metadata_holder_.reset();
    return;
  }
  dsa_metadata_holder_ = std::make_shared<PythonObjectHolder>(std::move(value));
}

py::object PyAttentionMetadataView::dsa_positions() const {
  return optional_tensor(dsa_positions_);
}

void PyAttentionMetadataView::set_dsa_positions(py::object value) {
  dsa_positions_ = tensor_from_python(value);
}

py::object PyAttentionMetadataView::dsa_cos_sin() const {
  return optional_tensor(dsa_cos_sin_);
}

void PyAttentionMetadataView::set_dsa_cos_sin(py::object value) {
  dsa_cos_sin_ = tensor_from_python(value);
}

py::object PyAttentionMetadataView::dsa_c4_cos_sin() const {
  return optional_tensor(dsa_c4_cos_sin_);
}

void PyAttentionMetadataView::set_dsa_c4_cos_sin(py::object value) {
  dsa_c4_cos_sin_ = tensor_from_python(value);
}

py::object PyAttentionMetadataView::dsa_c128_cos_sin() const {
  return optional_tensor(dsa_c128_cos_sin_);
}

void PyAttentionMetadataView::set_dsa_c128_cos_sin(py::object value) {
  dsa_c128_cos_sin_ = tensor_from_python(value);
}

int64_t PyAttentionMetadataView::dsa_graph_block_table_cols() const {
  return dsa_graph_block_table_cols_;
}

void PyAttentionMetadataView::set_dsa_graph_block_table_cols(int64_t value) {
  dsa_graph_block_table_cols_ = value;
}

bool PyAttentionMetadataView::dsa_graph_mode() const { return dsa_graph_mode_; }

void PyAttentionMetadataView::set_dsa_graph_mode(bool value) {
  dsa_graph_mode_ = value;
}

bool PyAttentionMetadataView::is_prefill() const {
  return metadata_->is_prefill;
}

bool PyAttentionMetadataView::is_chunked_prefill() const {
  return metadata_->is_chunked_prefill;
}

bool PyAttentionMetadataView::is_mixed() const { return metadata_->is_mixed; }

bool PyAttentionMetadataView::is_spec_verify() const {
  return metadata_->is_spec_verify;
}

void PyAttentionMetadataView::update_from(
    const PyAttentionMetadataView& source) {
  TORCH_CHECK(metadata_ != nullptr && source.metadata_ != nullptr,
              "MTP graph metadata view is not initialized");
  TORCH_CHECK(metadata_->is_prefill == source.metadata_->is_prefill,
              "MTP graph metadata changed prefill/decode mode");
  TORCH_CHECK(
      metadata_->is_chunked_prefill == source.metadata_->is_chunked_prefill,
      "MTP graph metadata changed chunked-prefill mode");
  TORCH_CHECK(metadata_->is_mixed == source.metadata_->is_mixed,
              "MTP graph metadata changed mixed mode");
  TORCH_CHECK(metadata_->is_spec_verify == source.metadata_->is_spec_verify,
              "MTP graph metadata changed spec-verify mode");

  copy_stable_tensor(
      metadata_->slot_mapping, source.metadata_->slot_mapping, "slot_mapping");
  copy_stable_tensor(metadata_->paged_kv_indptr,
                     source.metadata_->paged_kv_indptr,
                     "paged_kv_indptr");
  copy_stable_tensor(metadata_->paged_kv_indices,
                     source.metadata_->paged_kv_indices,
                     "paged_kv_indices");
  copy_stable_tensor(metadata_->paged_kv_last_page_len,
                     source.metadata_->paged_kv_last_page_len,
                     "paged_kv_last_page_len");
  copy_stable_tensor(metadata_->q_cu_seq_lens,
                     source.metadata_->q_cu_seq_lens,
                     "q_cu_seq_lens");
  copy_stable_tensor(metadata_->kv_cu_seq_lens,
                     source.metadata_->kv_cu_seq_lens,
                     "kv_cu_seq_lens");
  copy_stable_tensor(
      metadata_->kv_seq_lens, source.metadata_->kv_seq_lens, "kv_seq_lens");
  copy_stable_tensor(
      metadata_->q_seq_lens, source.metadata_->q_seq_lens, "q_seq_lens");
  copy_stable_tensor(
      metadata_->block_table, source.metadata_->block_table, "block_table");
  copy_stable_tensor(metadata_->has_initial_states,
                     source.metadata_->has_initial_states,
                     "has_initial_states");
#if defined(USE_NPU)
  copy_stable_tensor(metadata_->q_seq_lens_host,
                     source.metadata_->q_seq_lens_host,
                     "q_seq_lens_host");
  copy_stable_tensor(metadata_->kv_seq_lens_host,
                     source.metadata_->kv_seq_lens_host,
                     "kv_seq_lens_host");
  copy_stable_tensor(metadata_->paged_attention_tiling_data,
                     source.metadata_->paged_attention_tiling_data,
                     "paged_attention_tiling_data");
  copy_stable_tensor(metadata_->fia_attn_mask,
                     source.metadata_->fia_attn_mask,
                     "fia_attn_mask");
#endif

  copy_stable_optional_tensor(
      metadata_->qo_indptr, source.metadata_->qo_indptr, "qo_indptr");
  copy_stable_tensor(
      metadata_->mrope_cos, source.metadata_->mrope_cos, "mrope_cos");
  copy_stable_tensor(
      metadata_->mrope_sin, source.metadata_->mrope_sin, "mrope_sin");
  copy_stable_tensor(
      metadata_->attn_mask, source.metadata_->attn_mask, "attn_mask");

  TORCH_CHECK(metadata_->expanded_decode.enabled ==
                  source.metadata_->expanded_decode.enabled,
              "MTP graph expanded-decode mode changed");
  if (metadata_->expanded_decode.enabled) {
    copy_stable_tensor(metadata_->expanded_decode.kv_seq_lens,
                       source.metadata_->expanded_decode.kv_seq_lens,
                       "expanded.kv_seq_lens");
    copy_stable_tensor(metadata_->expanded_decode.block_table,
                       source.metadata_->expanded_decode.block_table,
                       "expanded.block_table");
    copy_stable_tensor(metadata_->expanded_decode.paged_kv_indptr,
                       source.metadata_->expanded_decode.paged_kv_indptr,
                       "expanded.paged_kv_indptr");
    copy_stable_tensor(metadata_->expanded_decode.paged_kv_indices,
                       source.metadata_->expanded_decode.paged_kv_indices,
                       "expanded.paged_kv_indices");
    copy_stable_tensor(metadata_->expanded_decode.paged_kv_last_page_len,
                       source.metadata_->expanded_decode.paged_kv_last_page_len,
                       "expanded.paged_kv_last_page_len");
    copy_stable_tensor(
        metadata_->expanded_decode.paged_attention_tiling_data,
        source.metadata_->expanded_decode.paged_attention_tiling_data,
        "expanded.paged_attention_tiling_data");
    copy_stable_tensor(metadata_->expanded_decode.kv_seq_lens_host,
                       source.metadata_->expanded_decode.kv_seq_lens_host,
                       "expanded.kv_seq_lens_host");
    TORCH_CHECK(
        metadata_->expanded_decode.kv_seq_lens_host_vec.size() ==
            source.metadata_->expanded_decode.kv_seq_lens_host_vec.size(),
        "MTP graph expanded host KV length count changed");
    metadata_->expanded_decode.kv_seq_lens_host_vec =
        source.metadata_->expanded_decode.kv_seq_lens_host_vec;
  }

  TORCH_CHECK(multi_block_tables_.size() == source.multi_block_tables_.size(),
              "MTP graph metadata multi-block table count changed");
  for (size_t i = 0; i < multi_block_tables_.size(); ++i) {
    copy_stable_tensor(multi_block_tables_[i],
                       source.multi_block_tables_[i],
                       "multi_block_tables");
  }
  copy_stable_tensor(linear_state_indices_,
                     source.linear_state_indices_,
                     "linear_state_indices");
  copy_stable_tensor(dsa_positions_, source.dsa_positions_, "dsa_positions");
  copy_stable_tensor(dsa_cos_sin_, source.dsa_cos_sin_, "dsa_cos_sin");
  copy_stable_tensor(dsa_c4_cos_sin_, source.dsa_c4_cos_sin_, "dsa_c4_cos_sin");
  copy_stable_tensor(
      dsa_c128_cos_sin_, source.dsa_c128_cos_sin_, "dsa_c128_cos_sin");

  TORCH_CHECK(kv_seq_lens_host_values().size() ==
                  source.kv_seq_lens_host_values().size(),
              "MTP graph metadata host KV length count changed");
  TORCH_CHECK(metadata_->q_seq_lens_vec.size() ==
                  source.metadata_->q_seq_lens_vec.size(),
              "MTP graph metadata host Q length count changed");
  std::copy(source.metadata_->kv_seq_lens_vec.begin(),
            source.metadata_->kv_seq_lens_vec.end(),
            metadata_->kv_seq_lens_vec.begin());
  std::copy(source.metadata_->q_seq_lens_vec.begin(),
            source.metadata_->q_seq_lens_vec.end(),
            metadata_->q_seq_lens_vec.begin());
  if (kv_seq_lens_host_.defined()) {
    kv_seq_lens_host_.copy_(source.kv_seq_lens_host_);
  }
  if (q_seq_lens_host_.defined()) {
    q_seq_lens_host_.copy_(source.q_seq_lens_host_);
  }
  dp_execution_token_counts_ = source.dp_execution_token_counts_;
  dp_global_sequence_nums_ = source.dp_global_sequence_nums_;
  dp_is_decode_ = source.dp_is_decode_;

  // Decode max lengths are request data and grow after every accepted token.
  // They are updated in the stable view; graph-shape changes are guarded by
  // tensor capacities above, not by the current scalar length value.
  metadata_->max_query_len = source.metadata_->max_query_len;
  metadata_->max_seq_len = source.metadata_->max_seq_len;
  dsa_metadata_holder_ = source.dsa_metadata_holder_;
  dsa_graph_block_table_cols_ = source.dsa_graph_block_table_cols_;
  dsa_graph_mode_ = source.dsa_graph_mode_;
}

PyAttentionMetadataView PyAttentionMetadataView::clone_for_graph() const {
  TORCH_CHECK(metadata_ != nullptr,
              "MTP graph metadata view is not initialized");
  auto cloned = std::make_shared<layer::AttentionMetadata>(*metadata_);
  const auto clone_tensor = [](const torch::Tensor& tensor) {
    return tensor.defined() ? tensor.clone() : torch::Tensor();
  };
  cloned->q_cu_seq_lens = clone_tensor(metadata_->q_cu_seq_lens);
  cloned->kv_cu_seq_lens = clone_tensor(metadata_->kv_cu_seq_lens);
  cloned->kv_seq_lens = clone_tensor(metadata_->kv_seq_lens);
  cloned->q_seq_lens = clone_tensor(metadata_->q_seq_lens);
  cloned->block_table = clone_tensor(metadata_->block_table);
  cloned->slot_mapping = clone_tensor(metadata_->slot_mapping);
  cloned->mrope_cos = clone_tensor(metadata_->mrope_cos);
  cloned->mrope_sin = clone_tensor(metadata_->mrope_sin);
  cloned->paged_kv_indptr = clone_tensor(metadata_->paged_kv_indptr);
  cloned->paged_kv_indices = clone_tensor(metadata_->paged_kv_indices);
  cloned->paged_kv_last_page_len =
      clone_tensor(metadata_->paged_kv_last_page_len);
  cloned->qo_indptr =
      metadata_->qo_indptr.has_value()
          ? std::optional<torch::Tensor>(clone_tensor(*metadata_->qo_indptr))
          : std::nullopt;
  cloned->full_k_cache = clone_tensor(metadata_->full_k_cache);
  cloned->full_v_cache = clone_tensor(metadata_->full_v_cache);
  cloned->unshared_k_cache = clone_tensor(metadata_->unshared_k_cache);
  cloned->unshared_v_cache = clone_tensor(metadata_->unshared_v_cache);
  cloned->step_tensor = clone_tensor(metadata_->step_tensor);
  cloned->chunk_indices = clone_tensor(metadata_->chunk_indices);
  cloned->batch = clone_tensor(metadata_->batch);
  cloned->token_block_offset = clone_tensor(metadata_->token_block_offset);
  cloned->has_initial_states = clone_tensor(metadata_->has_initial_states);
  cloned->attn_mask = clone_tensor(metadata_->attn_mask);
#if defined(USE_NPU)
  cloned->q_seq_lens_host = clone_tensor(metadata_->q_seq_lens_host);
  cloned->kv_seq_lens_host = clone_tensor(metadata_->kv_seq_lens_host);
  cloned->paged_attention_tiling_data =
      clone_tensor(metadata_->paged_attention_tiling_data);
  cloned->fia_attn_mask = clone_tensor(metadata_->fia_attn_mask);
#endif
  cloned->expanded_decode.kv_seq_lens =
      clone_tensor(metadata_->expanded_decode.kv_seq_lens);
  cloned->expanded_decode.block_table =
      clone_tensor(metadata_->expanded_decode.block_table);
  cloned->expanded_decode.paged_kv_indptr =
      clone_tensor(metadata_->expanded_decode.paged_kv_indptr);
  cloned->expanded_decode.paged_kv_indices =
      clone_tensor(metadata_->expanded_decode.paged_kv_indices);
  cloned->expanded_decode.paged_kv_last_page_len =
      clone_tensor(metadata_->expanded_decode.paged_kv_last_page_len);
  cloned->expanded_decode.paged_attention_tiling_data =
      clone_tensor(metadata_->expanded_decode.paged_attention_tiling_data);
  cloned->expanded_decode.kv_seq_lens_host =
      clone_tensor(metadata_->expanded_decode.kv_seq_lens_host);

  PyAttentionMetadataView result(std::move(cloned));
  result.multi_block_tables_.reserve(multi_block_tables_.size());
  for (const torch::Tensor& table : multi_block_tables_) {
    result.multi_block_tables_.push_back(clone_tensor(table));
  }
  result.linear_state_indices_ = clone_tensor(linear_state_indices_);
  result.dp_execution_token_counts_ = dp_execution_token_counts_;
  result.dp_global_sequence_nums_ = dp_global_sequence_nums_;
  result.dp_is_decode_ = dp_is_decode_;
  result.dsa_metadata_holder_ = dsa_metadata_holder_;
  result.dsa_positions_ = clone_tensor(dsa_positions_);
  result.dsa_cos_sin_ = clone_tensor(dsa_cos_sin_);
  result.dsa_c4_cos_sin_ = clone_tensor(dsa_c4_cos_sin_);
  result.dsa_c128_cos_sin_ = clone_tensor(dsa_c128_cos_sin_);
  result.dsa_graph_block_table_cols_ = dsa_graph_block_table_cols_;
  result.dsa_graph_mode_ = dsa_graph_mode_;
  return result;
}

torch::Tensor PyAttentionMetadataView::make_host_int32_view(
    const std::shared_ptr<layer::AttentionMetadata>& metadata,
    std::vector<int32_t>& host_vec) {
  if (host_vec.empty()) {
    return torch::Tensor();
  }

  std::shared_ptr<layer::AttentionMetadata> owner = metadata;
  return torch::from_blob(
      host_vec.data(),
      {static_cast<int64_t>(host_vec.size())},
      [owner = std::move(owner)](void*) mutable { owner.reset(); },
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
}

}  // namespace xllm
