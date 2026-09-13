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

"""One rank of the opt-in SFA-DCP HCCL graph regression."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu


def _reference(outputs: list[torch.Tensor], lses: list[torch.Tensor], rank: int) -> torch.Tensor:
    start, end = rank * 32, (rank + 1) * 32
    output = torch.stack([value[:, start:end] for value in outputs], dim=0).float()
    lse = torch.stack([value[:, start:end] for value in lses], dim=0)
    weights = torch.softmax(lse, dim=0).unsqueeze(-1)
    return (output * weights).sum(0).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--rendezvous", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    args = parser.parse_args()

    torch.ops.load_library(str(args.native_library))
    import xllm.python as runtime

    runtime.initialize_runtime()
    torch.npu.set_device(args.device)
    dist.init_process_group(
        "hccl",
        init_method=args.rendezvous.resolve().as_uri(),
        rank=args.rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )

    from xllm.python.attention.kv_shard_layout import KVShardLayout
    from xllm.python.layers.sfa_dcp import AscendSFADCPImpl
    from xllm.python.model_executor.forward_context import (
        AclGraphExecutionState,
        ForwardContext,
        forward_context,
    )

    class Group:
        world_size = 2
        device_group = dist.group.WORLD

        def __init__(self, rank: int) -> None:
            self.rank_in_group = rank

    impl = AscendSFADCPImpl(
        Group(args.rank),
        scale=1.0,
        index_topk=2048,
        layout=KVShardLayout(16, 2, args.rank),
    )
    output = torch.full((1, 64, 128), float(args.rank + 1), dtype=torch.bfloat16, device="npu")
    lse = torch.full((1, 64), float(args.rank), dtype=torch.float32, device="npu")
    context = ForwardContext(
        None,
        torch.device(f"npu:{args.device}"),
        None,
        [],
        execution_state=AclGraphExecutionState({}),
    )

    with forward_context(context):
        impl._merge_dcp_outputs(output, lse)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream()
    with forward_context(context), torch.npu.graph(graph, stream=capture_stream):
        captured = impl._merge_dcp_outputs(output, lse)
    torch.npu.synchronize()

    for step in range(3):
        output.fill_(float((args.rank + 1) * (step + 2)))
        lse.fill_(float(args.rank + step))
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        gathered_outputs = [torch.empty_like(output) for _ in range(2)]
        gathered_lses = [torch.empty_like(lse) for _ in range(2)]
        dist.all_gather(gathered_outputs, output)
        dist.all_gather(gathered_lses, lse)
        torch.npu.synchronize()
        torch.testing.assert_close(
            captured.cpu(), _reference(gathered_outputs, gathered_lses, args.rank).cpu(), atol=3e-2, rtol=3e-2
        )

    dist.destroy_process_group()
    if args.rank == 0:
        print("SFA DCP HCCL graph probe passed")


if __name__ == "__main__":
    main()
