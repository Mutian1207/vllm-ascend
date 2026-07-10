# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.gpu.block_table import _gather_block_tables_kernel


def _ptrs(tensors, device):
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.uint64, device=device)


def case_gather_block_tables(device):
    for num_reqs, groups, blocks in ((16, 1, 64), (64, 2, 128)):
        src = [torch.arange(num_reqs * blocks, device=device, dtype=torch.int32).reshape(num_reqs, blocks) + g for g in range(groups)]
        dst = [torch.empty_like(s) for s in src]
        src_ptrs = _ptrs(src, device)
        dst_ptrs = _ptrs(dst, device)
        strides = torch.tensor([blocks] * groups, device=device, dtype=torch.int64)
        nb = torch.full((groups, num_reqs), blocks, device=device, dtype=torch.int32)
        batch_idx = torch.arange(num_reqs, device=device, dtype=torch.int32)
        yield f"num_reqs={num_reqs} groups={groups} blocks={blocks}", lambda: _gather_block_tables_kernel[(groups, num_reqs)](batch_idx, src_ptrs, dst_ptrs, strides, nb, nb.stride(0), num_reqs, BLOCK_SIZE=64), lambda: int(sum(d.sum().item() for d in dst))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_gather_block_tables(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_gather_block_tables_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
