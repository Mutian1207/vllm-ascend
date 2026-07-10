# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties, set_npu_device
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import _load_ptr


@triton.jit
def _bench_load_ptr_kernel(ptrs, value):
    out = _load_ptr(ptrs, tl.int32)
    tl.store(out, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    out = torch.empty(1, device=args.device, dtype=torch.int32)
    ptrs = torch.tensor([out.data_ptr()], device=args.device, dtype=torch.uint64)
    fn = lambda: _bench_load_ptr_kernel[(1,)](ptrs, 123)
    latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
    print(f"op=_load_ptr latency_us={latency_us:.2f} checksum={int(out.item())}")


if __name__ == "__main__":
    main()

