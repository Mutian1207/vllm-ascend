# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties
from vllm.v1.worker.gpu.model_states.mamba_hybrid import _scatter_num_accepted_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    init_triton_ascend_device_properties()

    for num_reqs in [16, 128, 512]:
        idx_mapping = torch.arange(num_reqs, device=args.device, dtype=torch.int32)
        num_sampled = torch.randint(0, 4, (num_reqs,), device=args.device, dtype=torch.int32)
        num_accepted = torch.empty(num_reqs, device=args.device, dtype=torch.int32)
        fn = lambda: _scatter_num_accepted_kernel[(num_reqs,)](
            idx_mapping, num_sampled, num_accepted)
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_scatter_num_accepted_kernel num_reqs={num_reqs} "
              f"latency_us={latency_us:.2f} checksum={int(num_accepted.sum().item())}")


if __name__ == "__main__":
    main()

