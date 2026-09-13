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

"""Opt-in two-rank SFA-DCP HCCL graph regression."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_two_rank_sfa_dcp_hccl_graph_replay(tmp_path: Path) -> None:
    device_list = os.environ.get("XLLM_TEST_DCP_HCCL_DEVICES")
    native_library = os.environ.get("XLLM_TEST_NATIVE_LIBRARY")
    if device_list is None or native_library is None:
        pytest.skip("set XLLM_TEST_DCP_HCCL_DEVICES and XLLM_TEST_NATIVE_LIBRARY")
    parts = device_list.split(",")
    if len(parts) != 2 or not all(part.strip().isdecimal() for part in parts):
        raise AssertionError("XLLM_TEST_DCP_HCCL_DEVICES must contain two nonnegative NPU ids")
    devices = (int(parts[0]), int(parts[1]))
    if devices[0] == devices[1]:
        raise AssertionError("SFA-DCP HCCL ranks must use distinct devices")
    library = Path(native_library)
    if not library.is_file():
        raise AssertionError(f"native operator library does not exist: {library}")

    rendezvous = tmp_path / "rendezvous"
    rendezvous.touch()
    worker = Path(__file__).with_name("sfa_dcp_hccl_worker.py")
    repo = Path(__file__).parents[2]
    python_path = os.pathsep.join(filter(None, (str(repo), os.environ.get("PYTHONPATH", ""))))
    logs = []
    processes = []
    for rank, device in enumerate(devices):
        log_path = tmp_path / f"rank-{rank}.log"
        logs.append(log_path)
        log = log_path.open("w", encoding="utf-8")
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(worker),
                    "--rank",
                    str(rank),
                    "--device",
                    str(device),
                    "--rendezvous",
                    str(rendezvous),
                    "--native-library",
                    str(library),
                ],
                cwd=repo,
                env={
                    **os.environ,
                    "PYTHONPATH": python_path,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
        log.close()

    try:
        returncodes = [process.wait(timeout=180) for process in processes]
    except subprocess.TimeoutExpired:
        for process in processes:
            process.kill()
        raise AssertionError(f"SFA-DCP HCCL graph probe timed out; logs: {logs}") from None
    if returncodes != [0, 0]:
        details = "\n".join(f"{path}:\n{path.read_text(encoding='utf-8')}" for path in logs)
        raise AssertionError(f"SFA-DCP HCCL graph ranks failed: {returncodes}\n{details}")
