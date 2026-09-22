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

"""One rank of the two-rank MTP TP-consensus graph probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xllm.python import distributed
from xllm.python.model_executor.runners.mtp_sampling import (
    MtpSamplingPlan,
    probabilistic_acceptance,
    sample_logits,
)


def _sample_inputs(rank: int, value: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Give each rank a different local winner while rank zero owns consensus."""
    if (value + rank) % 2 == 0:
        winner = 0
    else:
        winner = 1
    logits = torch.full((1, 3), -100.0, device=device)
    logits[0, winner] = 100.0
    uniform = torch.full((1, 3), 0.5, device=device)
    return logits, uniform


def _acceptance_inputs(rank: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    draft_tokens = torch.tensor([[0, 1]], dtype=torch.long, device=device)
    target_tokens = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    draft_probs = torch.tensor(
        [[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]],
        dtype=torch.float32,
        device=device,
    )
    first_row = [0.9, 0.1, 0.0] if rank == 0 else [0.1, 0.9, 0.0]
    target_probs = torch.tensor(
        [[first_row, [0.1, 0.9, 0.0], [0.2, 0.3, 0.5]]],
        dtype=torch.float32,
        device=device,
    )
    do_sample = torch.ones(1, dtype=torch.bool, device=device)
    acceptance_uniform = torch.full((1, 2), 0.5, device=device)
    recovery_uniform = torch.full((1, 2, 3), 0.5, device=device)
    return (
        draft_tokens,
        draft_probs,
        target_tokens,
        target_probs,
        do_sample,
        acceptance_uniform,
        recovery_uniform,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    distributed.init_tp_group(
        host="127.0.0.1",
        port=args.port,
        rank=args.rank,
        world_size=2,
        device=str(device),
        global_rank=args.rank,
        global_world_size=2,
        group_index=0,
    )
    assert distributed.tp_world_size(device) == 2

    plan = MtpSamplingPlan(
        batch_size=1,
        do_sample=torch.ones(1, dtype=torch.bool, device=device),
        all_random_sample=True,
        all_greedy_sample=False,
    )
    logits, uniform = _sample_inputs(args.rank, 0, device)
    # Warm up the collective outside capture. This also verifies the local
    # rank-specific winner is not accidentally used as the consensus result.
    sampled = sample_logits(logits, plan, uniform=uniform)
    assert int(sampled.tokens.cpu().item()) == 0

    static_token = torch.empty((1,), dtype=torch.long, device=device)
    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=capture_stream):
        static_token.copy_(sample_logits(logits, plan, uniform=uniform).tokens)
    torch.npu.synchronize()

    sampled_values: list[int] = []
    for value in (0, 1, 0, 1):
        next_logits, next_uniform = _sample_inputs(args.rank, value, device)
        logits.copy_(next_logits)
        uniform.copy_(next_uniform)
        static_token.fill_(-1)
        graph.replay()
        torch.npu.synchronize()
        actual = int(static_token.cpu().item())
        expected = 0 if value % 2 == 0 else 1
        assert actual == expected, (args.rank, value, actual, expected)
        sampled_values.append(actual)

    acceptance_inputs = _acceptance_inputs(args.rank, device)
    static_count = torch.empty((1,), dtype=torch.int32, device=device)
    static_next = torch.empty((1,), dtype=torch.long, device=device)
    acceptance_graph = torch.npu.NPUGraph()
    with torch.npu.graph(acceptance_graph, stream=capture_stream):
        result = probabilistic_acceptance(
            *acceptance_inputs[:5],
            acceptance_uniform=acceptance_inputs[5],
            recovery_uniform=acceptance_inputs[6],
        )
        static_count.copy_(result[2])
        static_next.copy_(result[3])
    torch.npu.synchronize()
    for _ in range(3):
        static_count.fill_(-1)
        static_next.fill_(-1)
        acceptance_graph.replay()
        torch.npu.synchronize()
        assert int(static_count.cpu().item()) == 2
        assert int(static_next.cpu().item()) == 2

    result = {
        "rank": args.rank,
        "tp_rank": distributed.tp_rank(device),
        "tp_world_size": distributed.tp_world_size(device),
        "sampled_values": sampled_values,
        "accepted_count": int(static_count.cpu().item()),
        "next_token": int(static_next.cpu().item()),
        "graph_replays": 4,
        "acceptance_graph_replays": 3,
        "device": str(device),
        "pid": os.getpid(),
    }
    (args.artifact / f"rank-{args.rank}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
