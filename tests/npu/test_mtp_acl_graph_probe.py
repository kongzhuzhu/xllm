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

"""Real-NPU probes of the composite runner for several independent K variants.

Run separately from tests/python, whose conftest installs runtime stubs.
Deterministic bodies exercise graph mechanics and acceptance, not real GLM
attention.
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.python.mtp_graph_test_utils import make_recipe, output_tensors, scalar_reference
from xllm.python.model_executor.runners.mtp_acl_graph import MtpAclGraphRunner


@pytest.fixture(scope="module")
def npu_device() -> torch.device:
    device_index = os.environ.get("XLLM_TEST_NPU_DEVICE")
    if device_index is None:
        pytest.skip("set XLLM_TEST_NPU_DEVICE for the real MTP ACL graph probe")
    pytest.importorskip("torch_npu")
    torch.npu.set_device(int(device_index))
    return torch.device(f"npu:{device_index}")


@pytest.mark.parametrize("steps", [1, 2, 3, 4])
def test_mtp_recipe_replay_every_rejection_position(npu_device: torch.device, steps: int) -> None:
    rejects_device = torch.arange(steps + 1, device=npu_device)
    runner = MtpAclGraphRunner(make_recipe(rejects_device, steps))
    eager = MtpAclGraphRunner(make_recipe(rejects_device, steps), backend="eager")
    seeds = [2 + row for row in range(steps + 1)]
    positions = [10 + row * 3 for row in range(steps + 1)]
    kv_lengths = [position + 1 for position in positions]
    runner.capture(
        torch.tensor(seeds, device=npu_device),
        torch.tensor(positions, device=npu_device),
        torch.tensor(kv_lengths, dtype=torch.int32, device=npu_device),
    )
    assert runner.entry.capability.speculative_tokens == steps
    assert runner.entry.captured and runner.entry.generation == 1
    rejects = list(range(steps + 1))
    expected = scalar_reference(seeds, positions, kv_lengths, rejects, steps)
    inputs = (
        torch.tensor(seeds, device=npu_device),
        torch.tensor(positions, device=npu_device),
        torch.tensor(kv_lengths, dtype=torch.int32, device=npu_device),
    )
    eager_output = output_tensors(eager.execute(*inputs))
    actual = runner.execute(*inputs)
    for name, tensor in output_tensors(actual).items():
        torch.testing.assert_close(tensor.cpu(), expected[name], rtol=0, atol=0)
        torch.testing.assert_close(tensor, eager_output[name], rtol=0, atol=0)
    assert runner.entry.generation == 1
