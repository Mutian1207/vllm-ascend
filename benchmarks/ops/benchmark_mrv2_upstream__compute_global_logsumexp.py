# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties, set_npu_device
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_logsumexp,
)


@triton.jit
def _bench_kernel(out, local_max, local_max_stride, local_sumexp, local_sumexp_stride,
                  vocab_num_blocks, PADDED_VOCAB_NUM_BLOCKS: tl.constexpr):
    row = tl.program_id(0)
    lse = _compute_global_logsumexp(local_max, local_max_stride, local_sumexp,
                                    local_sumexp_stride, row, vocab_num_blocks,
                                    PADDED_VOCAB_NUM_BLOCKS)
    tl.store(out + row, lse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for num_logits, num_blocks in [(64, 4), (64, 19)]:
        padded = triton.next_power_of_2(num_blocks)
        local_max = torch.randn((num_logits, num_blocks), device=args.device)
        local_sumexp = torch.rand((num_logits, num_blocks), device=args.device) + 1
        out = torch.empty(num_logits, device=args.device)
        fn = lambda: _bench_kernel[(num_logits,)](
            out, local_max, local_max.stride(0), local_sumexp, local_sumexp.stride(0),
            num_blocks, PADDED_VOCAB_NUM_BLOCKS=padded)
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_compute_global_logsumexp num_logits={num_logits} num_blocks={num_blocks} "
              f"latency_us={latency_us:.2f} checksum={float(out.sum().item()):.3f}")


if __name__ == "__main__":
    main()
