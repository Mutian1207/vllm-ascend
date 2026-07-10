# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens


def case_dcp(device):
    for num_reqs in (16, 128):
        max_num_reqs = max(256, num_reqs)
        seq_lens = torch.randint(1, 4096, (max_num_reqs,), device=device, dtype=torch.int32)
        out = torch.empty_like(seq_lens)
        yield f"num_reqs={num_reqs} max_num_reqs={max_num_reqs}", lambda out=out, seq_lens=seq_lens, num_reqs=num_reqs: prepare_dcp_local_seq_lens(out, seq_lens, num_reqs, 2, 1, 64), lambda: int(out.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_dcp(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_dcp_local_seq_lens_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
