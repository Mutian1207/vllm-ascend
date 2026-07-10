# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens


def case_prepare_pos_seq_lens(device):
    for num_reqs, qlen in ((16, 8), (128, 4)):
        max_num_reqs = max(256, num_reqs)
        total = num_reqs * qlen
        idx = torch.arange(num_reqs, device=device, dtype=torch.int32)
        qsl = torch.arange(num_reqs + 1, device=device, dtype=torch.int32) * qlen
        num_comp = torch.arange(num_reqs, device=device, dtype=torch.int32)
        pos = torch.empty(total, device=device, dtype=torch.int64)
        seq = torch.empty(max_num_reqs, device=device, dtype=torch.int32)
        yield f"num_reqs={num_reqs} query_len={qlen}", lambda: prepare_pos_seq_lens(idx, qsl, num_comp, pos, seq), lambda: int(seq.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_prepare_pos_seq_lens(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_prepare_pos_seq_lens_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
