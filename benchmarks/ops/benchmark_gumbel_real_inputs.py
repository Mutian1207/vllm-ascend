#!/usr/bin/env python3
"""Benchmark MRv2 Gumbel sampler with real dumped inputs.

The input files are produced by setting VLLM_ASCEND_DUMP_GUMBEL_INPUTS when
running a real vLLM service. Run this script in the NPU and GPU environments
with the same case directory, then compare the generated JSON summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch


def _sync(backend: str) -> None:
    if backend == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()


def _event_pair(backend: str):
    if backend == "gpu":
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)


def _load_sampler(backend: str):
    if backend == "gpu":
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        return gumbel_sample

    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

    return gumbel_sample


def _normalize_case(case_path: Path, case: dict[str, Any]) -> dict[str, Any]:
    metadata = case.get("metadata", {})
    name = case_path.stem
    if isinstance(metadata, dict):
        name = str(metadata.get("name") or name)

    idx_mapping = case.get("idx_mapping")
    if idx_mapping is None:
        idx_mapping = case["expanded_idx_mapping"]

    seeds = case.get("seeds")
    if seeds is None:
        seeds = case["seed"]

    return {
        "name": name,
        "metadata": metadata if isinstance(metadata, dict) else {},
        "logits": case["logits"],
        "temperature": case["temperature"],
        "seeds": seeds,
        "pos": case["pos"],
        "idx_mapping": idx_mapping,
        "apply_temperature": bool(case["apply_temperature"]),
        "output_processed_logits": case.get("output_processed_logits"),
        "output_processed_logits_col": case.get("output_processed_logits_col"),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _make_device_case(case: dict[str, Any], device: torch.device) -> dict[str, Any]:
    processed = None
    if case["output_processed_logits"] is not None:
        processed = torch.empty_like(case["output_processed_logits"].to(device))

    processed_col = None
    if case["output_processed_logits_col"] is not None:
        processed_col = case["output_processed_logits_col"].to(device)

    return {
        "logits": case["logits"].to(device),
        "idx_mapping": case["idx_mapping"].to(device),
        "temperature": case["temperature"].to(device),
        "seeds": case["seeds"].to(device),
        "pos": case["pos"].to(device),
        "apply_temperature": case["apply_temperature"],
        "output_processed_logits": processed,
        "output_processed_logits_col": processed_col,
    }


def _run_one_case(
    sampler,
    backend: str,
    device_case: dict[str, Any],
    warmup: int,
    repeat: int,
) -> tuple[list[float], torch.Tensor]:
    kwargs = {
        "apply_temperature": device_case["apply_temperature"],
        "output_processed_logits": device_case["output_processed_logits"],
        "output_processed_logits_col": device_case["output_processed_logits_col"],
        "use_fp64": False,
    }

    with torch.inference_mode():
        for _ in range(warmup):
            sampled = sampler(
                device_case["logits"],
                device_case["idx_mapping"],
                device_case["temperature"],
                device_case["seeds"],
                device_case["pos"],
                **kwargs,
            )
        _sync(backend)

        latencies_ms: list[float] = []
        last_sampled = sampled
        for _ in range(repeat):
            start, end = _event_pair(backend)
            start.record()
            last_sampled = sampler(
                device_case["logits"],
                device_case["idx_mapping"],
                device_case["temperature"],
                device_case["seeds"],
                device_case["pos"],
                **kwargs,
            )
            end.record()
            _sync(backend)
            latencies_ms.append(start.elapsed_time(end))

    return latencies_ms, last_sampled


def _summarize_case(
    backend: str,
    case_path: Path,
    case: dict[str, Any],
    latencies_ms: list[float],
) -> dict[str, Any]:
    logits = case["logits"]
    temperature = case["temperature"]
    return {
        "backend": backend,
        "case_file": case_path.name,
        "case": case["name"],
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "temperature_shape": list(temperature.shape),
        "temperature_unique": sorted(
            {float(x) for x in temperature.flatten().tolist()}
        )[:16],
        "idx_mapping_shape": list(case["idx_mapping"].shape),
        "pos_dtype": str(case["pos"].dtype),
        "apply_temperature": case["apply_temperature"],
        "has_output_processed_logits": case["output_processed_logits"] is not None,
        "repeat": len(latencies_ms),
        "latency_ms_mean": statistics.fmean(latencies_ms),
        "latency_ms_median": statistics.median(latencies_ms),
        "latency_ms_min": min(latencies_ms),
        "latency_ms_max": max(latencies_ms),
        "latency_ms_p90": _percentile(latencies_ms, 90),
        "latency_ms_p99": _percentile(latencies_ms, 99),
    }


def _run(args: argparse.Namespace) -> None:
    os.environ.pop("VLLM_ASCEND_DUMP_GUMBEL_INPUTS", None)

    if args.backend == "gpu":
        device = torch.device(args.device or "cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device or "npu:0")
        torch.npu.set_device(device)

    sampler = _load_sampler(args.backend)
    case_paths = sorted(args.case_dir.glob("*.pt"))
    if args.max_cases is not None:
        case_paths = case_paths[: args.max_cases]

    results = []
    for case_path in case_paths:
        raw_case = torch.load(case_path, map_location="cpu")
        case = _normalize_case(case_path, raw_case)
        device_case = _make_device_case(case, device)
        latencies_ms, sampled = _run_one_case(
            sampler, args.backend, device_case, args.warmup, args.repeat
        )
        _sync(args.backend)
        item = _summarize_case(args.backend, case_path, case, latencies_ms)
        item["sampled_checksum"] = int(sampled.long().sum().item())
        results.append(item)
        print(json.dumps(item, indent=2))

    output = {
        "backend": args.backend,
        "device": str(device),
        "case_dir": str(args.case_dir),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "created_at": time.time(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}")


def _compare(args: argparse.Namespace) -> None:
    gpu = json.loads(args.gpu_result.read_text())
    npu = json.loads(args.npu_result.read_text())
    gpu_items = {item["case_file"]: item for item in gpu["results"]}
    npu_items = {item["case_file"]: item for item in npu["results"]}
    summary = []
    for case_file in sorted(gpu_items.keys() & npu_items.keys()):
        gpu_item = gpu_items[case_file]
        npu_item = npu_items[case_file]
        gpu_mean = gpu_item["latency_ms_mean"]
        npu_mean = npu_item["latency_ms_mean"]
        summary.append(
            {
                "case_file": case_file,
                "logits_shape": gpu_item["logits_shape"],
                "gpu_latency_ms_mean": gpu_mean,
                "npu_latency_ms_mean": npu_mean,
                "npu_over_gpu_mean_ratio": npu_mean / gpu_mean
                if gpu_mean != 0
                else None,
                "gpu_latency_ms_p99": gpu_item["latency_ms_p99"],
                "npu_latency_ms_p99": npu_item["latency_ms_p99"],
            }
        )
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--backend", choices=("gpu", "npu"), required=True)
    run.add_argument("--case-dir", type=Path, required=True)
    run.add_argument("--device", default=None)
    run.add_argument("--warmup", type=int, default=20)
    run.add_argument("--repeat", type=int, default=100)
    run.add_argument("--max-cases", type=int, default=None)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(func=_run)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--gpu-result", type=Path, required=True)
    compare.add_argument("--npu-result", type=Path, required=True)
    compare.set_defaults(func=_compare)

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = _parse_args()
    parsed_args.func(parsed_args)
