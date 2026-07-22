# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties, set_npu_device
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax


@triton.jit
def _bench_gumbel_block_argmax_kernel(out_idx, out_val, logits_ptr, logits_stride,
                                      idx_mapping, temp, seed, pos, vocab_size,
                                      BLOCK_SIZE: tl.constexpr):
    token_idx = tl.program_id(0)
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + token_idx * logits_stride + block,
                     mask=mask, other=float("-inf")).to(tl.float32)
    value, idx = gumbel_block_argmax(
        logits, block, mask, token_idx, idx_mapping, temp, seed, pos, None, 0,
        None, vocab_size, APPLY_TEMPERATURE=True, USE_FP64=False)
    tl.store(out_idx + token_idx, idx)
    tl.store(out_val + token_idx, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for num_tokens, vocab_size in [(128, 1024), (8192, 1024)]:
        logits = torch.randn((num_tokens, vocab_size), device=args.device)
        out_idx = torch.empty(num_tokens, device=args.device, dtype=torch.int64)
        out_val = torch.empty(num_tokens, device=args.device)
        idx_mapping = torch.arange(num_tokens, device=args.device, dtype=torch.int32)
        temp = torch.ones(num_tokens, device=args.device)
        seed = torch.arange(num_tokens, device=args.device, dtype=torch.int64)
        pos = torch.arange(num_tokens, device=args.device, dtype=torch.int64)
        fn = lambda: _bench_gumbel_block_argmax_kernel[(num_tokens,)](
            out_idx, out_val, logits, logits.stride(0), idx_mapping, temp, seed,
            pos, vocab_size, BLOCK_SIZE=triton.next_power_of_2(vocab_size))
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=gumbel_block_argmax num_tokens={num_tokens} vocab_size={vocab_size} "
              f"latency_us={latency_us:.2f} checksum={int(out_idx.sum().item())}")


if __name__ == "__main__":
    main()
