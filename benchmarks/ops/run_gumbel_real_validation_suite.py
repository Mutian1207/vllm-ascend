#!/usr/bin/env python3
"""Run accuracy and latency validation for dumped real Gumbel inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import benchmark_gumbel_real_inputs as latency_bench
import compare_gumbel_precision as precision


def _json_ready(item: Any) -> Any:
    if isinstance(item, torch.Tensor):
        return item.tolist()
    if isinstance(item, dict):
        return {key: _json_ready(value) for key, value in item.items()}
    if isinstance(item, list):
        return [_json_ready(value) for value in item]
    if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
        return None
    return item


def _load_cases(case_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    cases = []
    for case_path in sorted(case_dir.glob("*.pt")):
        raw_case = torch.load(case_path, map_location="cpu")
        cases.append((case_path, precision._normalize_case(case_path, raw_case)))
    return cases


def _setup_device(backend: str, device_arg: str | None) -> torch.device:
    if backend == "gpu":
        device = torch.device(device_arg or "cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device(device_arg or "npu:0")
        torch.npu.set_device(device)
    return device


def _scan_temp06_rows(
    cases: list[tuple[Path, dict[str, Any]]],
    top1_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    candidates = []
    for case_path, case in cases:
        logits = case["logits"].float()
        idx_mapping = case["idx_mapping"].long()
        temperature = case["temperature"].float()
        apply_temperature = bool(case["apply_temperature"])
        for row_idx in range(logits.shape[0]):
            req_idx = int(idx_mapping[row_idx].item())
            if req_idx < 0:
                continue
            temp = float(temperature[req_idx].item())
            item = {
                "case_file": case_path.name,
                "row_idx": row_idx,
                "req_idx": req_idx,
                "temperature": temp,
                "candidate": False,
            }
            if temp != 0.0:
                effective_logits = logits[row_idx]
                if apply_temperature:
                    effective_logits = effective_logits / temp
                probs = torch.softmax(effective_logits.double(), dim=-1)
                top_probs, top_indices = torch.topk(probs, k=min(10, probs.numel()))
                item.update(
                    {
                        "top1_prob": float(top_probs[0].item()),
                        "top10_prob": float(top_probs.sum().item()),
                        "top5_token_ids": top_indices[:5].tolist(),
                        "top5_probs": [float(x) for x in top_probs[:5].tolist()],
                    }
                )
                item["candidate"] = item["top1_prob"] <= top1_threshold
                if item["candidate"]:
                    candidates.append(item)
            rows.append(item)
    candidates.sort(key=lambda item: item["top1_prob"])
    return rows, candidates


def _run_temp0_verify(
    backend: str,
    device: torch.device,
    cases: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    sampler = precision._load_sampler(backend)
    results = []
    for case_path, case in cases:
        logits = case["logits"].float().to(device)
        idx_mapping = case["idx_mapping"].to(device)
        temperature = case["temperature"].to(device)
        seeds = case["seeds"].to(device)
        pos = case["pos"].to(device)
        with torch.inference_mode():
            sampled = sampler(
                logits,
                idx_mapping,
                temperature,
                seeds,
                pos,
                apply_temperature=case["apply_temperature"],
                use_fp64=False,
            )
            precision._sync(backend)

        sampled_cpu = sampled.cpu().long()
        mapped_temperature = case["temperature"][case["idx_mapping"].long()]
        item = {
            "backend": backend,
            "case": case["name"],
            "case_file": case_path.name,
            "logits_shape": list(case["logits"].shape),
            "apply_temperature": case["apply_temperature"],
            "all_mapped_temperature_zero": bool(torch.all(mapped_temperature == 0)),
        }
        if case["sampled_ascend"] is not None and backend == "npu":
            item["npu_replay_exact_to_dump"] = bool(
                torch.equal(sampled_cpu, case["sampled_ascend"].long())
            )
        if item["all_mapped_temperature_zero"]:
            argmax = case["logits"].float().argmax(dim=-1).long()
            item["exact_to_argmax"] = bool(torch.equal(sampled_cpu, argmax))
        results.append(item)
    return results


def _summarize_temp0(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(results),
        "all_temperature_zero": sum(
            bool(item.get("all_mapped_temperature_zero")) for item in results
        ),
        "exact_to_argmax_true": sum(
            bool(item.get("exact_to_argmax")) for item in results
        ),
        "npu_replay_exact_to_dump_true": sum(
            bool(item.get("npu_replay_exact_to_dump")) for item in results
        ),
    }


def _run_accuracy_candidates(
    backend: str,
    device: torch.device,
    cases: list[tuple[Path, dict[str, Any]]],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    sampler = precision._load_sampler(backend)
    case_by_file = {case_path.name: (case_path, case) for case_path, case in cases}
    accuracy_args = SimpleNamespace(
        samples=args.samples,
        chunk_size=args.chunk_size,
        topk=args.topk,
        seed=args.seed,
    )
    results = []
    for candidate in candidates:
        case_path, case = case_by_file[candidate["case_file"]]
        item = precision._run_accuracy_row(
            sampler,
            backend,
            device,
            case,
            case_path,
            int(candidate["row_idx"]),
            accuracy_args,
        )
        if item is not None:
            results.append(item)
            printable = {
                key: value
                for key, value in item.items()
                if key not in ("topk_indices", "observed_bins", "expected_bins")
            }
            print(json.dumps(_json_ready(printable), indent=2))
    return results


def _summarize_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    checked = [item for item in results if item.get("chi2_pass_10sigma") is not None]
    passed = [item for item in checked if item.get("chi2_pass_10sigma") is True]
    failed = [item for item in checked if item.get("chi2_pass_10sigma") is False]
    nulls = [item for item in results if item.get("chi2_pass_10sigma") is None]
    tv_values = [
        float(item["tv_vs_softmax"])
        for item in results
        if item.get("tv_vs_softmax") is not None
    ]
    chi2_values = [
        float(item["chi2"]) for item in results if item.get("chi2") is not None
    ]
    return {
        "rows": len(results),
        "chi2_checked": len(checked),
        "chi2_passed": len(passed),
        "chi2_failed": len(failed),
        "chi2_null": len(nulls),
        "tv_vs_softmax_mean": statistics.fmean(tv_values) if tv_values else None,
        "tv_vs_softmax_max": max(tv_values) if tv_values else None,
        "chi2_mean": statistics.fmean(chi2_values) if chi2_values else None,
        "chi2_max": max(chi2_values) if chi2_values else None,
        "failed": [
            {
                "case_file": item["case_file"],
                "row_idx": item["row_idx"],
                "chi2": item.get("chi2"),
            }
            for item in failed
        ],
    }


def _run_latency(
    backend: str,
    device: torch.device,
    case_dir: Path,
    output: Path,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    sampler = latency_bench._load_sampler(backend)
    results = []
    for case_path in sorted(case_dir.glob("*.pt")):
        raw_case = torch.load(case_path, map_location="cpu")
        case = latency_bench._normalize_case(case_path, raw_case)
        device_case = latency_bench._make_device_case(case, device)
        latencies_ms, sampled = latency_bench._run_one_case(
            sampler, backend, device_case, warmup, repeat
        )
        latency_bench._sync(backend)
        item = latency_bench._summarize_case(backend, case_path, case, latencies_ms)
        item["sampled_checksum"] = int(sampled.long().sum().item())
        results.append(item)
        print(json.dumps(item, indent=2))
    output_data = {
        "backend": backend,
        "device": str(device),
        "case_dir": str(case_dir),
        "warmup": warmup,
        "repeat": repeat,
        "created_at": time.time(),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(output_data), indent=2) + "\n")
    return output_data


def _summarize_latency(result: dict[str, Any]) -> dict[str, Any]:
    means = [float(item["latency_ms_mean"]) for item in result["results"]]
    p99s = [float(item["latency_ms_p99"]) for item in result["results"]]
    return {
        "cases": len(result["results"]),
        "latency_ms_mean_avg": statistics.fmean(means) if means else None,
        "latency_ms_mean_min": min(means) if means else None,
        "latency_ms_mean_max": max(means) if means else None,
        "latency_ms_p99_avg": statistics.fmean(p99s) if p99s else None,
        "latency_ms_p99_max": max(p99s) if p99s else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gpu", "npu"), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--temp0-dir", type=Path, required=True)
    parser.add_argument("--temp06-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0xABCD)
    parser.add_argument("--top1-threshold", type=float, default=0.9)
    parser.add_argument("--max-accuracy-rows", type=int, default=20)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeat", type=int, default=100)
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.pop("VLLM_ASCEND_DUMP_GUMBEL_INPUTS", None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = _setup_device(args.backend, args.device)
    temp0_cases = _load_cases(args.temp0_dir)
    temp06_cases = _load_cases(args.temp06_dir)
    temp06_rows, candidates = _scan_temp06_rows(temp06_cases, args.top1_threshold)
    selected_candidates = candidates
    if args.max_accuracy_rows > 0:
        selected_candidates = candidates[: args.max_accuracy_rows]

    (args.output_dir / "temp06_scan.json").write_text(
        json.dumps(_json_ready(temp06_rows), indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "temp06_accuracy_candidates.json").write_text(
        json.dumps(_json_ready(selected_candidates), indent=2) + "\n",
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "backend": args.backend,
        "device": str(device),
        "temp0_dir": str(args.temp0_dir),
        "temp06_dir": str(args.temp06_dir),
        "output_dir": str(args.output_dir),
        "temp0_cases": len(temp0_cases),
        "temp06_cases": len(temp06_cases),
        "temp06_rows": len(temp06_rows),
        "temp06_nonzero_temp_rows": sum(
            float(item["temperature"]) != 0.0 for item in temp06_rows
        ),
        "top1_threshold": args.top1_threshold,
        "accuracy_candidates": len(candidates),
        "accuracy_rows_run": len(selected_candidates),
    }

    temp0_verify = _run_temp0_verify(args.backend, device, temp0_cases)
    (args.output_dir / f"{args.backend}_temp0_verify.json").write_text(
        json.dumps(_json_ready(temp0_verify), indent=2) + "\n", encoding="utf-8"
    )
    summary["temp0_verify"] = _summarize_temp0(temp0_verify)

    if not args.skip_accuracy:
        accuracy_results = _run_accuracy_candidates(
            args.backend, device, temp06_cases, selected_candidates, args
        )
        result_dir = args.output_dir / "temp06" / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backend": args.backend,
                "samples": args.samples,
                "topk": args.topk,
                "chunk_size": args.chunk_size,
                "results": accuracy_results,
            },
            result_dir / f"{args.backend}_accuracy.pt",
        )
        (args.output_dir / f"{args.backend}_temp06_accuracy.json").write_text(
            json.dumps(_json_ready(accuracy_results), indent=2) + "\n",
            encoding="utf-8",
        )
        summary["accuracy"] = _summarize_accuracy(accuracy_results)

    if not args.skip_latency:
        temp0_latency = _run_latency(
            args.backend,
            device,
            args.temp0_dir,
            args.output_dir / f"{args.backend}_temp0_benchmark.json",
            args.latency_warmup,
            args.latency_repeat,
        )
        temp06_latency = _run_latency(
            args.backend,
            device,
            args.temp06_dir,
            args.output_dir / f"{args.backend}_temp06_benchmark.json",
            args.latency_warmup,
            args.latency_repeat,
        )
        summary["temp0_latency"] = _summarize_latency(temp0_latency)
        summary["temp06_latency"] = _summarize_latency(temp06_latency)

    summary_path = args.output_dir / f"{args.backend}_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_ready(summary), indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
