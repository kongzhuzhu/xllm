if(NOT BUILD_TESTING)
  return()
endif()

if(USE_NPU)
  add_test(
    NAME python_executor_test
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
      "${Python3_EXECUTABLE}" -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  set_tests_properties(
    python_executor_test
    PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )

  # Run separately so tests/python runtime stubs cannot affect the probe.
  add_test(
    NAME python_hccl_decode_graph_test
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
      "${Python3_EXECUTABLE}" -m pytest -q
      "${CMAKE_CURRENT_SOURCE_DIR}/npu/test_hccl_decode_graph.py"
  )
  set_tests_properties(
    python_hccl_decode_graph_test
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      RUN_SERIAL TRUE
  )

  # SFA-DCP capture/replay uses two NPUs and an HCCL process group. Keep it
  # separate from the Python contract suite so the subprocess workers do not
  # inherit its runtime stubs or share devices with another NPU test.
  add_test(
    NAME python_sfa_dcp_hccl_graph_test
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
      "${Python3_EXECUTABLE}" -m pytest -q
      "${CMAKE_CURRENT_SOURCE_DIR}/npu/test_sfa_dcp_hccl_graph.py"
  )
  set_tests_properties(
    python_sfa_dcp_hccl_graph_test
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      RUN_SERIAL TRUE
  )

  add_test(
    NAME python_cp_context_native_test
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
      "${Python3_EXECUTABLE}" -m pytest -q
      "${CMAKE_CURRENT_SOURCE_DIR}/npu/test_cp_context_native.py"
  )
  set_tests_properties(
    python_cp_context_native_test
    PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
endif()

add_subdirectory(api_service)
add_subdirectory(core)
add_subdirectory(function_call)
add_subdirectory(models)
