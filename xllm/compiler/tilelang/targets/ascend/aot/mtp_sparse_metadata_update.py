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

from xllm.python.kernels_npu.tilelang import mtp_sparse_metadata_update as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang.mtp_sparse_metadata_update import (
    MTP_SPARSE_METADATA_PASS_CONFIGS,
    build_mtp_sparse_metadata_update_kernel,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class MtpSparseMetadataUpdateKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("position_bits", "int32")]
    SPECIALIZATIONS = [{"variant_key": f"positions_{bits}", "position_bits": bits} for bits in (32, 64)]

    @staticmethod
    def generate_source(position_bits: int) -> str:
        tilelang.disable_cache()
        kernel = build_mtp_sparse_metadata_update_kernel(position_bits)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=MTP_SPARSE_METADATA_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
