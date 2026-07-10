# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties, set_npu_device
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.sample.logprob import _fill_logprob_token_ids_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for batch_size, num_topk, num_cols in [(32, 5, 5), (512, 5, 8)]:
        sampled = torch.randint(0, 32_000, (batch_size,), device=args.device)
        topk = torch.randint(0, 32_000, (batch_size, num_topk), device=args.device,
                             dtype=torch.int32)
        expanded = torch.arange(batch_size, device=args.device, dtype=torch.int32)
        num_custom = torch.zeros(batch_size, device=args.device, dtype=torch.int32)
        per_req = torch.empty((batch_size, 16), device=args.device, dtype=torch.int32)
        out_ids = torch.empty((batch_size, 1 + num_cols), device=args.device,
                              dtype=torch.int64)
        mask = torch.empty_like(out_ids, dtype=torch.bool)
        fn = lambda: _fill_logprob_token_ids_kernel[(batch_size,)](
            out_ids, out_ids.stride(0), mask, mask.stride(0), sampled, topk,
            topk.stride(0), expanded, num_custom, per_req, per_req.stride(0),
            NUM_TOPK=num_topk, PADDED_COLS=triton.next_power_of_2(num_cols))
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_fill_logprob_token_ids_kernel batch_size={batch_size} "
              f"num_cols={num_cols} latency_us={latency_us:.2f} "
              f"checksum={int(out_ids.sum().item())}")


if __name__ == "__main__":
    main()

