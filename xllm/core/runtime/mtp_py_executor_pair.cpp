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

#include "core/runtime/mtp_py_executor_pair.h"

#include <glog/logging.h>
#include <torch/python.h>

#include <utility>

#include "core/runtime/py_executor_impl.h"
#include "core/util/pybind_helper.h"

namespace py = pybind11;

namespace xllm::detail {
namespace {

MtpPyGraphOutput parse_graph_output(const py::object& output) {
  py::object next_state = output.attr("next_state");
  MtpPyGraphOutput result;
  result.accepted_ids = output.attr("accepted_ids").cast<torch::Tensor>();
  result.accepted_mask = output.attr("accepted_mask").cast<torch::Tensor>();
  result.accepted_count = output.attr("accepted_count").cast<torch::Tensor>();
  result.committed_tokens =
      output.attr("committed_tokens").cast<torch::Tensor>();
  result.next_token_ids = next_state.attr("token_ids").cast<torch::Tensor>();
  result.next_positions = next_state.attr("positions").cast<torch::Tensor>();
  result.next_kv_seq_lens = tensor_from_python(next_state.attr("kv_seq_lens"));
  result.next_embeddings = tensor_from_python(next_state.attr("embeddings"));
  result.next_topk_indices =
      tensor_from_python(next_state.attr("topk_indices"));
  result.target_embeddings =
      tensor_from_python(output.attr("target_embeddings"));
  result.target_probs = tensor_from_python(output.attr("target_probs"));
  result.committed_log_probs = tensor_from_python(output.attr("logprobs"));
  result.target_top_log_probs = tensor_from_python(output.attr("top_logprobs"));
  result.target_top_tokens = tensor_from_python(output.attr("top_tokens"));
  return result;
}

py::list metadata_list(const std::vector<py::object>& metadata) {
  py::list result;
  for (const py::object& item : metadata) {
    result.append(item);
  }
  return result;
}

}  // namespace

std::unique_ptr<MtpPyExecutorPair> MtpPyExecutorPair::create(
    PyExecutorImpl& target_executor,
    PyExecutorImpl& draft_executor,
    const std::vector<py::object>& draft_metadata,
    const py::object& target_metadata,
    const torch::Tensor& repair_token_ids,
    const torch::Tensor& kv_seq_lens,
    int32_t batch_size,
    int32_t speculative_tokens,
    int64_t vocab_size,
    const py::object& draft_sampling_plan,
    const py::object& target_sampling_plan,
    bool target_step_major_layout) {
  py::gil_scoped_acquire gil;
  py::object runner =
      target_executor.create_mtp_graph_runner(draft_executor,
                                              draft_metadata,
                                              target_metadata,
                                              repair_token_ids,
                                              kv_seq_lens,
                                              batch_size,
                                              speculative_tokens,
                                              vocab_size,
                                              draft_sampling_plan,
                                              target_sampling_plan,
                                              target_step_major_layout);
  return std::unique_ptr<MtpPyExecutorPair>(
      new MtpPyExecutorPair(std::move(runner)));
}

MtpPyExecutorPair::MtpPyExecutorPair(py::object runner)
    : runner_(std::move(runner)) {}

MtpPyExecutorPair::~MtpPyExecutorPair() { clear_python_object(runner_); }

MtpPyGraphOutput MtpPyExecutorPair::capture_and_execute(
    const torch::Tensor& seed_token_ids,
    const torch::Tensor& base_positions,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& draft_input_embedding,
    const torch::Tensor& draft_topk_indices) {
  CHECK(runner_);
  py::gil_scoped_acquire gil;
  py::object optional_topk =
      draft_topk_indices.defined() ? py::cast(draft_topk_indices) : py::none();
  LOG(INFO) << "MTP unified pair capture begin";
  try {
    runner_.attr("capture")(seed_token_ids,
                            base_positions,
                            kv_seq_lens,
                            draft_input_embedding,
                            optional_topk);
  } catch (const py::error_already_set& error) {
    LOG(ERROR) << "MTP unified pair capture failed: " << error.what();
    throw;
  }
  LOG(INFO) << "MTP unified pair capture done; replay begin";
  py::object output;
  try {
    output = runner_.attr("execute")(seed_token_ids,
                                     base_positions,
                                     kv_seq_lens,
                                     draft_input_embedding,
                                     optional_topk);
  } catch (const py::error_already_set& error) {
    LOG(ERROR) << "MTP unified pair replay failed: " << error.what();
    throw;
  }
  LOG(INFO) << "MTP unified pair replay done";
  return parse_graph_output(output);
}

bool MtpPyExecutorPair::can_update_metadata(
    const std::vector<py::object>& draft_metadata,
    const py::object& target_metadata) const {
  CHECK(runner_);
  py::gil_scoped_acquire gil;
  return runner_
      .attr("can_update_metadata")(metadata_list(draft_metadata),
                                   target_metadata)
      .cast<bool>();
}

bool MtpPyExecutorPair::can_update_sampling_plans(
    const py::object& draft_sampling_plan,
    const py::object& target_sampling_plan) const {
  CHECK(runner_);
  py::gil_scoped_acquire gil;
  return runner_
      .attr("can_update_sampling_plans")(draft_sampling_plan,
                                         target_sampling_plan)
      .cast<bool>();
}

std::string MtpPyExecutorPair::metadata_key(
    const std::vector<py::object>& draft_metadata,
    const py::object& target_metadata,
    const py::object& draft_sampling_plan,
    const py::object& target_sampling_plan) const {
  CHECK(runner_);
  py::gil_scoped_acquire gil;
  py::object key = runner_.attr("metadata_key")(metadata_list(draft_metadata),
                                                target_metadata,
                                                draft_sampling_plan,
                                                target_sampling_plan);
  return py::repr(key).cast<std::string>();
}

MtpPyGraphOutput MtpPyExecutorPair::update_and_execute(
    const std::vector<py::object>& draft_metadata,
    const py::object& target_metadata,
    const torch::Tensor& repair_token_ids,
    const torch::Tensor& seed_token_ids,
    const torch::Tensor& base_positions,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& draft_input_embedding,
    const torch::Tensor& draft_topk_indices,
    const py::object& draft_sampling_plan,
    const py::object& target_sampling_plan) {
  CHECK(runner_);
  py::gil_scoped_acquire gil;
  py::object optional_topk =
      draft_topk_indices.defined() ? py::cast(draft_topk_indices) : py::none();
  runner_.attr("update_metadata")(
      metadata_list(draft_metadata), target_metadata, repair_token_ids);
  runner_.attr("update_sampling_plans")(draft_sampling_plan,
                                        target_sampling_plan);
  LOG(INFO) << "MTP unified pair replay begin (reused graph)";
  py::object output;
  try {
    output = runner_.attr("execute")(seed_token_ids,
                                     base_positions,
                                     kv_seq_lens,
                                     draft_input_embedding,
                                     optional_topk);
  } catch (const py::error_already_set& error) {
    LOG(ERROR) << "MTP unified pair replay failed: " << error.what();
    throw;
  }
  LOG(INFO) << "MTP unified pair replay done (reused graph)";
  return parse_graph_output(output);
}

}  // namespace xllm::detail
