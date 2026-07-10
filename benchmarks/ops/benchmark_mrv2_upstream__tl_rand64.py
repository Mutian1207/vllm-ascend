# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand64


@triton.jit
def _rand64_bench_kernel(out, seed, numel, INCLUDES_ZERO: tl.constexpr,
                         BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < numel
    vals = tl_rand64(seed, block, includes_zero=INCLUDES_ZERO)
    tl.store(out + block, vals, mask=mask)


def case_tl_rand64(device):
    for numel in (1024, 65536):
        out = torch.empty(numel, device=device, dtype=torch.float64)
        grid = (triton.cdiv(numel, 256),)
        yield f"numel={numel}", lambda: _rand64_bench_kernel[grid](out, 123, numel, INCLUDES_ZERO=False, BLOCK_SIZE=256), lambda: float(out.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_tl_rand64(args.device):
        try:
            latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        except Exception as e:
            message = str(e)
            if "hfusion.isnan" in message and "f64" in message:
                print(f"op=tl_rand64 {spec} skipped=NPU_UNSUPPORTED_FP64_ISNAN")
                continue
            raise
        print(f"op=tl_rand64 {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
