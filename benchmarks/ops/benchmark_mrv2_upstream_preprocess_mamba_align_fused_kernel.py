# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.triton_utils import triton
from vllm.v1.worker.mamba_utils import preprocess_mamba_align_fused_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    max_num_reqs = 16
    mamba_block_size = 128
    for num_reqs in [1, 16]:
        idx_mapping = torch.arange(num_reqs, device=args.device, dtype=torch.int32)
        # Pre-advance state_idx (>=0 so the advance + reset path runs).
        state_idx = torch.randint(
            0, 8, (max_num_reqs,), device=args.device, dtype=torch.int32
        )
        num_computed_tokens = torch.randint(
            1, 512, (max_num_reqs,), device=args.device, dtype=torch.int32
        )
        query_start_loc = torch.arange(
            0, (max_num_reqs + 1) * 4, 4, device=args.device, dtype=torch.int32
        )
        # num_accepted >= 1 so token bias is non-trivial.
        num_accepted_tokens = torch.randint(
            1, 4, (max_num_reqs,), device=args.device, dtype=torch.int32
        )
        src_col = torch.zeros(max_num_reqs, device=args.device, dtype=torch.int32)
        src_off = torch.zeros(max_num_reqs, device=args.device, dtype=torch.int32)
        fn = lambda: preprocess_mamba_align_fused_kernel[
            (triton.cdiv(num_reqs, 256),)
        ](
            idx_mapping,
            state_idx,
            num_computed_tokens,
            query_start_loc,
            num_accepted_tokens,
            src_col,
            src_off,
            num_reqs,
            BLOCK_SIZE=256,
            MAMBA_BLOCK_SIZE=mamba_block_size,
        )
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(
            f"op=preprocess_mamba_align_fused_kernel num_reqs={num_reqs} "
            f"mamba_block_size={mamba_block_size} latency_us={latency_us:.2f} "
            f"checksum={int(src_col.sum().item())}"
        )


if __name__ == "__main__":
    main()
