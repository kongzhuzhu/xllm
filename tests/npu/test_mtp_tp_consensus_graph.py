# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-rank HCCL ACL graph probe for MTP sampling decisions.

Set ``XLLM_TEST_HCCL_DEVICES`` to two verified-free logical NPU ids. The
probe fails when a rank skips the graph collective or when rank-local random
decisions leak into the accepted token/state.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import pytest


def test_two_rank_mtp_tp_consensus_graph(tmp_path: Path) -> None:
    device_list = os.environ.get("XLLM_TEST_HCCL_DEVICES")
    if device_list is None:
        pytest.skip("set XLLM_TEST_HCCL_DEVICES to two verified-free logical NPU ids")
    parts = [part.strip() for part in device_list.split(",")]
    assert len(parts) == 2 and all(part.isdecimal() for part in parts)
    devices = tuple(int(part) for part in parts)
    assert devices[0] != devices[1]

    artifact = Path(tempfile.mkdtemp(prefix="mtp-tp-consensus-", dir=tmp_path))
    worker = Path(__file__).with_name("mtp_tp_consensus_worker.py")
    repo = Path(__file__).resolve().parents[2]
    commands: list[list[str]] = []
    processes: list[subprocess.Popen[bytes]] = []
    port = 29731
    with ExitStack() as stack:
        try:
            for rank in range(2):
                command = [
                    sys.executable,
                    str(worker),
                    "--rank",
                    str(rank),
                    "--device",
                    str(devices[rank]),
                    "--port",
                    str(port),
                    "--artifact",
                    str(artifact),
                ]
                commands.append(command)
                log = stack.enter_context((artifact / f"rank-{rank}.log").open("xb"))
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=repo,
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            deadline = time.monotonic() + 180
            while True:
                codes = [process.poll() for process in processes]
                if all(code is not None for code in codes) or any(code not in (None, 0) for code in codes):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"MTP TP consensus probe timed out: {commands}")
                time.sleep(0.1)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    codes = [process.returncode for process in processes]
    if codes != [0, 0]:
        logs = "\n".join((artifact / f"rank-{rank}.log").read_text() for rank in range(2))
        raise AssertionError(f"MTP TP consensus rank failures {codes}\n{logs}")
    results = [json.loads((artifact / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert [result["tp_rank"] for result in results] == [0, 1]
    assert all(result["tp_world_size"] == 2 for result in results)
    assert all(result["sampled_values"] == [0, 1, 0, 1] for result in results)
    assert all(result["accepted_count"] == 2 for result in results)
    assert all(result["next_token"] == 2 for result in results)
    assert all(result["graph_replays"] == 4 for result in results)
    assert all(result["acceptance_graph_replays"] == 3 for result in results)

    # Keep the source and result artifacts when requested for remote diagnosis;
    # normal pytest cleanup remains the default.
    if os.environ.get("XLLM_TEST_HCCL_ARTIFACT_DIR"):
        destination = Path(os.environ["XLLM_TEST_HCCL_ARTIFACT_DIR"]) / artifact.name
        shutil.copytree(artifact, destination, dirs_exist_ok=True)
