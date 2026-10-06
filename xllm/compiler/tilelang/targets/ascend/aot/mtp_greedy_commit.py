# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tilelang

from xllm.python.kernels_npu.tilelang import mtp_greedy_commit as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang.mtp_greedy_commit import (
    MTP_GREEDY_COMMIT_PASS_CONFIGS,
    build_mtp_greedy_commit_kernel,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class MtpGreedyCommitKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("hidden_bits", "int32")]
    SPECIALIZATIONS = [{"variant_key": f"hidden_{bits}", "hidden_bits": bits} for bits in (16, 32)]

    @staticmethod
    def generate_source(hidden_bits: int) -> str:
        tilelang.disable_cache()
        kernel = build_mtp_greedy_commit_kernel(hidden_bits)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=MTP_GREEDY_COMMIT_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
