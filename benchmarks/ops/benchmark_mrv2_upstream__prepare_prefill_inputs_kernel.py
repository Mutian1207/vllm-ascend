# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)


def case_input_prepare_prefill(device):
    from vllm.v1.worker.gpu.input_batch import prepare_prefill_inputs
    for num_reqs, qlen in ((16, 8), (128, 4)):
        max_len = 256
        total = num_reqs * qlen
        input_ids = torch.empty(total, device=device, dtype=torch.int32)
        next_prefill = torch.empty(num_reqs, device=device, dtype=torch.int32)
        idx = torch.arange(num_reqs, device=device, dtype=torch.int32)
        qsl = torch.arange(num_reqs + 1, device=device, dtype=torch.int32) * qlen
        all_ids = torch.randint(0, 32000, (num_reqs, max_len), device=device, dtype=torch.int32)
        prefill_len = torch.full((num_reqs,), max_len, device=device, dtype=torch.int32)
        num_comp = torch.zeros(num_reqs, device=device, dtype=torch.int32)
        yield f"num_reqs={num_reqs} query_len={qlen}", lambda: prepare_prefill_inputs(input_ids, next_prefill, idx, qsl, all_ids, prefill_len, num_comp), lambda: int(input_ids.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_input_prepare_prefill(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_prepare_prefill_inputs_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
