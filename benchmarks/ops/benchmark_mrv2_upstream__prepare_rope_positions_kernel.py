# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties
from vllm.v1.worker.gpu.mm.rope import RopeState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    init_triton_ascend_device_properties()

    for num_reqs, query_len in [(16, 1), (64, 8), (128, 1)]:
        num_dims = 3
        max_model_len = 4096
        total_tokens = num_reqs * query_len
        query_start_loc = torch.arange(num_reqs + 1, device=args.device,
                                       dtype=torch.int32) * query_len
        prefill_positions = torch.arange(num_reqs * num_dims * max_model_len,
                                         device=args.device, dtype=torch.int32).reshape(
                                             num_reqs * num_dims, max_model_len)
        rope_state = SimpleNamespace(
            positions=torch.empty((num_dims, total_tokens + 1), device=args.device,
                                  dtype=torch.int64),
            prefill_positions=SimpleNamespace(gpu=prefill_positions),
            prefill_delta=SimpleNamespace(
                gpu=torch.randint(0, 32, (num_reqs,), device=args.device, dtype=torch.int32)),
            num_dims=num_dims,
            max_model_len=max_model_len,
        )
        idx_mapping = torch.arange(num_reqs, device=args.device, dtype=torch.int32)
        prefill_lens = torch.full((num_reqs,), max_model_len, device=args.device,
                                  dtype=torch.int32)
        computed = torch.randint(0, max_model_len - query_len, (num_reqs,),
                                 device=args.device, dtype=torch.int32)
        fn = lambda: RopeState.prepare_positions(rope_state, idx_mapping, query_start_loc,
                                                 prefill_lens, computed)
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_prepare_rope_positions_kernel num_reqs={num_reqs} query_len={query_len} "
              f"latency_us={latency_us:.2f} checksum={int(rope_state.positions.sum().item())}")


if __name__ == "__main__":
    main()

