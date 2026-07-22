#!/usr/bin/env python3
"""Distribution accuracy checks for MRv2 Gumbel sampler real inputs.

This script consumes real inputs dumped from
``vllm_ascend.worker.v2.sample.gumbel``. Run the same dumped cases on NPU and
GPU, then compare both empirical distributions against the theoretical
softmax distribution. This follows the same statistical idea as upstream
``test_gpu_gumbel_sample.py``: correctness is distribution fidelity, not
single-draw token equality.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def _sync(backend: str) -> None:
    if backend == "gpu":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()


def _load_sampler(backend: str):
    if backend == "gpu":
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        return gumbel_sample

    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

    return gumbel_sample


def _normalize_case(case_path: Path, case: dict[str, object]) -> dict[str, object]:
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
        "logits": case["logits"],
        "temperature": case["temperature"],
        "seeds": seeds,
        "pos": case["pos"],
        "idx_mapping": idx_mapping,
        "apply_temperature": bool(case["apply_temperature"]),
        "sampled_ascend": case.get("sampled_ascend"),
    }


def _parse_row_indices(row_indices: str | None) -> list[int] | None:
    if row_indices is None:
        return None
    return [int(item) for item in row_indices.split(",") if item]


def _cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = lhs.reshape(-1).double()
    rhs = rhs.reshape(-1).double()
    return torch.nn.functional.cosine_similarity(lhs, rhs, dim=0).item()


def _topk_other_bins(
    sampled: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    sampled = sampled.cpu().long()
    topk_indices = topk_indices.cpu().long()
    counts = torch.empty(topk_indices.numel() + 1, dtype=torch.float64)
    top_total = 0
    for i, token_id in enumerate(topk_indices.tolist()):
        count = (sampled == token_id).sum().item()
        counts[i] = count
        top_total += count
    counts[-1] = sampled.numel() - top_total
    return counts


def _binned_distribution_metrics(
    observed_bins: torch.Tensor,
    expected_bins: torch.Tensor,
) -> dict[str, float | bool | None]:
    observed_bins = observed_bins.double()
    expected_bins = expected_bins.double()
    observed_probs = observed_bins / observed_bins.sum()
    expected_probs = expected_bins / expected_bins.sum()
    expected_count = expected_probs * observed_bins.sum()

    tv = (observed_probs - expected_probs).abs().sum().item() / 2
    valid = expected_count >= 5
    chi2 = None
    chi2_pass = None
    if valid.sum().item() > 1:
        chi2_value = (
            ((observed_bins[valid] - expected_count[valid]) ** 2)
            / expected_count[valid]
        ).sum().item()
        df = valid.sum().item() - 1
        threshold = df + 10 * (2 * df) ** 0.5
        chi2 = chi2_value
        chi2_pass = chi2_value < threshold

    return {
        "tv_vs_softmax": tv,
        "cosine_vs_softmax": _cosine(observed_probs, expected_probs),
        "chi2": chi2,
        "chi2_pass_10sigma": chi2_pass,
    }


def _run_accuracy_row(
    sampler,
    backend: str,
    device: torch.device,
    case: dict[str, object],
    case_path: Path,
    row_idx: int,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    logits_cpu = case["logits"].float()
    idx_mapping_cpu = case["idx_mapping"].long()
    temperature_cpu = case["temperature"].float()

    req_idx = int(idx_mapping_cpu[row_idx].item())
    if req_idx < 0:
        return None

    temp = float(temperature_cpu[req_idx].item())
    logits_1d_cpu = logits_cpu[row_idx].contiguous()
    vocab_size = logits_1d_cpu.numel()
    if temp == 0.0:
        return {
            "case": case["name"],
            "case_file": case_path.name,
            "row_idx": row_idx,
            "req_idx": req_idx,
            "temperature": temp,
            "apply_temperature": case["apply_temperature"],
            "vocab_size": vocab_size,
            "skipped": "temperature_zero",
        }

    effective_logits = logits_1d_cpu
    if case["apply_temperature"]:
        effective_logits = effective_logits / temp
    reference_probs = torch.softmax(effective_logits.double(), dim=-1)
    topk = min(args.topk, vocab_size)
    top_probs, topk_indices = torch.topk(reference_probs, k=topk)
    expected_bins = torch.cat(
        (top_probs, (1 - top_probs.sum()).reshape(1))
    ) * args.samples

    logits_1d = logits_1d_cpu.to(device)
    temperature = torch.tensor([temp], dtype=torch.float32, device=device)
    seed = torch.tensor([args.seed + row_idx], dtype=torch.int64, device=device)
    observed_bins = torch.zeros(topk + 1, dtype=torch.float64)
    base_pos = int(case["pos"][row_idx].item())

    with torch.inference_mode():
        for start in range(0, args.samples, args.chunk_size):
            size = min(args.chunk_size, args.samples - start)
            # AscendC custom op expects contiguous rows; avoid a 0-stride expand.
            logits = logits_1d.unsqueeze(0).repeat(size, 1).contiguous()
            idx_mapping = torch.zeros(size, dtype=torch.int32, device=device)
            pos = torch.arange(
                base_pos + start,
                base_pos + start + size,
                dtype=torch.int64,
                device=device,
            )
            sampled = sampler(
                logits,
                idx_mapping,
                temperature,
                seed,
                pos,
                apply_temperature=case["apply_temperature"],
                use_fp64=False,
            )
            _sync(backend)
            observed_bins += _topk_other_bins(sampled, topk_indices)

    metrics = _binned_distribution_metrics(observed_bins, expected_bins)
    return {
        "case": case["name"],
        "case_file": case_path.name,
        "row_idx": row_idx,
        "req_idx": req_idx,
        "temperature": temp,
        "apply_temperature": case["apply_temperature"],
        "vocab_size": vocab_size,
        "samples": args.samples,
        "topk": topk,
        "topk_indices": topk_indices.cpu(),
        "observed_bins": observed_bins.cpu(),
        "expected_bins": expected_bins.cpu(),
        **metrics,
    }


def _json_ready(item: object) -> object:
    if isinstance(item, torch.Tensor):
        return item.tolist()
    if isinstance(item, dict):
        return {key: _json_ready(value) for key, value in item.items()}
    if isinstance(item, list):
        return [_json_ready(value) for value in item]
    return item


def _accuracy(args: argparse.Namespace) -> None:
    os.environ.pop("VLLM_ASCEND_DUMP_GUMBEL_INPUTS", None)

    if args.backend == "gpu":
        device = torch.device(args.device or "cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device or "npu:0")
        torch.npu.set_device(device)

    sampler = _load_sampler(args.backend)
    row_indices = _parse_row_indices(args.row_indices)
    case_paths = sorted(args.case_dir.glob("*.pt"))
    if args.max_cases is not None:
        case_paths = case_paths[: args.max_cases]

    results = []
    for case_path in case_paths:
        raw_case = torch.load(case_path, map_location="cpu")
        case = _normalize_case(case_path, raw_case)
        num_rows = case["logits"].shape[0]
        if row_indices is None:
            rows = list(range(min(args.max_rows_per_case, num_rows)))
        else:
            rows = [row for row in row_indices if row < num_rows]

        for row_idx in rows:
            item = _run_accuracy_row(
                sampler, args.backend, device, case, case_path, row_idx, args
            )
            if item is None:
                continue
            results.append(item)
            printable = {
                key: value
                for key, value in item.items()
                if key not in ("topk_indices", "observed_bins", "expected_bins")
            }
            print(json.dumps(_json_ready(printable), indent=2))

    result_dir = args.case_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backend": args.backend,
            "samples": args.samples,
            "topk": args.topk,
            "chunk_size": args.chunk_size,
            "results": results,
        },
        result_dir / f"{args.backend}_accuracy.pt",
    )


def _accuracy_compare(args: argparse.Namespace) -> None:
    result_dir = args.case_dir / "results"
    gpu = torch.load(result_dir / "gpu_accuracy.pt", map_location="cpu")
    npu = torch.load(result_dir / "npu_accuracy.pt", map_location="cpu")
    gpu_items = {
        (item["case_file"], item["row_idx"]): item for item in gpu["results"]
    }
    npu_items = {
        (item["case_file"], item["row_idx"]): item for item in npu["results"]
    }

    summary = []
    for key in sorted(gpu_items.keys() & npu_items.keys()):
        gpu_item = gpu_items[key]
        npu_item = npu_items[key]
        item = {
            "case": gpu_item["case"],
            "case_file": key[0],
            "row_idx": key[1],
            "gpu_tv_vs_softmax": gpu_item.get("tv_vs_softmax"),
            "npu_tv_vs_softmax": npu_item.get("tv_vs_softmax"),
            "gpu_cosine_vs_softmax": gpu_item.get("cosine_vs_softmax"),
            "npu_cosine_vs_softmax": npu_item.get("cosine_vs_softmax"),
            "gpu_chi2_pass_10sigma": gpu_item.get("chi2_pass_10sigma"),
            "npu_chi2_pass_10sigma": npu_item.get("chi2_pass_10sigma"),
        }

        if "observed_bins" in gpu_item and "observed_bins" in npu_item:
            gpu_probs = gpu_item["observed_bins"].double()
            gpu_probs = gpu_probs / gpu_probs.sum()
            npu_probs = npu_item["observed_bins"].double()
            npu_probs = npu_probs / npu_probs.sum()
            item["npu_gpu_tv"] = (gpu_probs - npu_probs).abs().sum().item() / 2
            item["npu_gpu_cosine"] = _cosine(gpu_probs, npu_probs)
        summary.append(item)

    print(json.dumps(_json_ready(summary), indent=2))


def _verify(args: argparse.Namespace) -> None:
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

    summary = []
    for case_path in case_paths:
        raw_case = torch.load(case_path, map_location="cpu")
        case = _normalize_case(case_path, raw_case)
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
            _sync(args.backend)

        sampled_cpu = sampled.cpu().long()
        mapped_temperature = case["temperature"][case["idx_mapping"].long()]
        item = {
            "backend": args.backend,
            "case": case["name"],
            "case_file": case_path.name,
            "logits_shape": list(case["logits"].shape),
            "apply_temperature": case["apply_temperature"],
            "all_mapped_temperature_zero": bool(torch.all(mapped_temperature == 0)),
        }

        if case["sampled_ascend"] is not None and args.backend == "npu":
            sampled_ascend = case["sampled_ascend"].long()
            item["npu_replay_exact_to_dump"] = bool(
                torch.equal(sampled_cpu, sampled_ascend)
            )

        if item["all_mapped_temperature_zero"]:
            argmax = case["logits"].float().argmax(dim=-1).long()
            item["exact_to_argmax"] = bool(torch.equal(sampled_cpu, argmax))

        summary.append(item)

    print(json.dumps(_json_ready(summary), indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    accuracy = subparsers.add_parser("accuracy")
    accuracy.add_argument("--backend", choices=("gpu", "npu"), required=True)
    accuracy.add_argument("--case-dir", type=Path, required=True)
    accuracy.add_argument("--device", default=None)
    accuracy.add_argument("--samples", type=int, default=20_000)
    accuracy.add_argument("--chunk-size", type=int, default=64)
    accuracy.add_argument("--topk", type=int, default=256)
    accuracy.add_argument("--max-cases", type=int, default=5)
    accuracy.add_argument("--max-rows-per-case", type=int, default=1)
    accuracy.add_argument("--row-indices", default=None)
    accuracy.add_argument("--seed", type=int, default=0xABCD)
    accuracy.set_defaults(func=_accuracy)

    accuracy_compare = subparsers.add_parser("accuracy-compare")
    accuracy_compare.add_argument("--case-dir", type=Path, required=True)
    accuracy_compare.set_defaults(func=_accuracy_compare)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--backend", choices=("gpu", "npu"), required=True)
    verify.add_argument("--case-dir", type=Path, required=True)
    verify.add_argument("--device", default=None)
    verify.add_argument("--max-cases", type=int, default=None)
    verify.set_defaults(func=_verify)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = _parse_args()
    parsed_args.func(parsed_args)
