# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import _compute_slot_mappings_kernel


def _ptrs(tensors, device):
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.uint64, device=device)


def case_compute_slot_mappings(device):
    for num_reqs, qlen in ((16, 8), (64, 4)):
        max_tokens = num_reqs * qlen
        block_size = 16
        max_blocks = 256
        bt = torch.arange(num_reqs * max_blocks, device=device, dtype=torch.int32).reshape(num_reqs, max_blocks)
        ptrs = _ptrs([bt], device)
        strides = torch.tensor([max_blocks], device=device, dtype=torch.int64)
        block_sizes = torch.tensor([block_size], device=device, dtype=torch.int32)
        idx = torch.arange(num_reqs, device=device, dtype=torch.int32)
        qsl = torch.arange(num_reqs + 1, device=device, dtype=torch.int32) * qlen
        pos = torch.arange(max_tokens, device=device, dtype=torch.int64)
        out = torch.empty((1, max_tokens), device=device, dtype=torch.int64)
        yield f"num_reqs={num_reqs} query_len={qlen}", lambda: _compute_slot_mappings_kernel[(1, num_reqs + 1)](max_tokens, idx, qsl, pos, ptrs, strides, block_sizes, out, out.stride(0), 0, CP_SIZE=1, CP_INTERLEAVE=1, PAD_ID=PAD_SLOT_ID, TRITON_BLOCK_SIZE=64), lambda: int(out.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_compute_slot_mappings(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_compute_slot_mappings_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
