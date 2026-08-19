# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from vllm.triton_utils import triton


BLOCK_SIZE = 8192


def resolve_device(device: str) -> torch.device:
    """Resolve auto/cuda/npu without requiring torch_npu on GPU hosts."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch_npu is not None and torch.npu.is_available():
            device = "npu:0"
        else:
            raise RuntimeError(
                "neither a CUDA GPU nor an Ascend NPU is available"
            )

    resolved = torch.device(device)

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available"
            )
    elif resolved.type == "npu":
        if torch_npu is None:
            raise RuntimeError(
                "NPU was requested but torch_npu is not installed"
            )
        if not torch.npu.is_available():
            raise RuntimeError(
                "NPU was requested but is not available"
            )
    else:
        raise ValueError(
            f"unsupported device {device!r}; "
            "use auto, cuda:N, or npu:N"
        )

    return resolved


def accelerator_module(device: torch.device):
    if device.type == "cuda":
        return torch.cuda

    if device.type == "npu":
        return torch.npu

    raise ValueError(f"unsupported accelerator: {device}")


def synchronize(device: torch.device) -> None:
    accelerator_module(device).synchronize(device)


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    category: str
    rows: int
    vocab: int
    seed: int = 0
    pattern: str = "all_allowed"


@dataclasses.dataclass(frozen=True)
class KernelSpec:
    label: str
    path: str
    entry: str
    sha256: str


BUILTIN_CASES: dict[str, Case] = {
    "biz_rows16_vocab32000": Case(
        name="biz_rows16_vocab32000",
        category="business",
        rows=16,
        vocab=32000,
    ),
    "biz_rows64_vocab32000": Case(
        name="biz_rows64_vocab32000",
        category="business",
        rows=64,
        vocab=32000,
    ),
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def parse_kernel_spec(
    text: str,
) -> tuple[str, str, str]:
    if "=" not in text or ":" not in text:
        raise ValueError(
            "--kernel format must be LABEL=FILE:ENTRY, "
            f"got: {text!r}"
        )

    label, target = text.split("=", 1)
    path, entry = target.rsplit(":", 1)

    path = str(
        Path(path).expanduser().resolve()
    )

    if not Path(path).is_file():
        raise FileNotFoundError(path)

    return label, path, entry


def load_kernel(
    text: str,
) -> tuple[KernelSpec, Any]:
    label, path, entry = parse_kernel_spec(text)
    digest = sha256_file(path)

    module_name = (
        f"_triton_case_{label}_"
        f"{digest[:12]}_{time.time_ns()}"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot load module from {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, entry):
        raise AttributeError(
            f"{path} has no entry {entry!r}"
        )

    return (
        KernelSpec(
            label=label,
            path=path,
            entry=entry,
            sha256=digest,
        ),
        getattr(module, entry),
    )


def case_from_dict(
    data: dict[str, Any],
) -> Case:
    return Case(
        name=str(data["name"]),
        category=str(
            data.get("category", "business")
        ),
        rows=int(data["rows"]),
        vocab=int(data["vocab"]),
        seed=int(data.get("seed", 0)),
        pattern=str(
            data.get("pattern", "all_allowed")
        ),
    )


def load_cases_file(path: str) -> list[Case]:
    data = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if isinstance(data, dict):
        if "cases" in data:
            data = data["cases"]
        else:
            data = [
                {
                    "name": name,
                    **value,
                }
                for name, value in data.items()
            ]

    if not isinstance(data, list):
        raise ValueError(
            "cases file must be a JSON list, "
            "a {name: case} dict, or "
            "{'cases': [...]}"
        )

    return [
        case_from_dict(x)
        for x in data
    ]


def resolve_cases(
    args: argparse.Namespace,
) -> list[Case]:
    cases: list[Case] = []

    if args.cases_file:
        cases.extend(
            load_cases_file(args.cases_file)
        )

    if args.case_json:
        data = json.loads(args.case_json)

        if "name" not in data:
            data["name"] = "cli_case"

        cases.append(
            case_from_dict(data)
        )

    selected_names: list[str] = []

    if args.case:
        selected_names.append(args.case)

    if args.cases:
        selected_names.extend(
            x.strip()
            for x in args.cases.split(",")
            if x.strip()
        )

    for name in selected_names:
        if name not in BUILTIN_CASES:
            raise KeyError(
                f"unknown built-in case: {name}; "
                f"available={sorted(BUILTIN_CASES)}"
            )

        cases.append(
            BUILTIN_CASES[name]
        )

    if not cases:
        cases = list(
            BUILTIN_CASES.values()
        )

    if (
        args.rows is not None
        or args.vocab is not None
    ):
        if len(cases) != 1:
            raise ValueError(
                "--rows/--vocab overrides require "
                "exactly one resolved case"
            )

        c = cases[0]

        cases = [
            dataclasses.replace(
                c,
                name=f"{c.name}_override",
                rows=(
                    args.rows
                    if args.rows is not None
                    else c.rows
                ),
                vocab=(
                    args.vocab
                    if args.vocab is not None
                    else c.vocab
                ),
            )
        ]

    seen: set[str] = set()
    out: list[Case] = []

    for case in cases:
        if case.name in seen:
            raise ValueError(
                f"duplicate case name: {case.name}"
            )

        seen.add(case.name)
        out.append(case)

    return out


def build_inputs(
    case: Case,
    device: str,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(case.seed)

    logits = torch.randn(
        (case.rows, case.vocab),
        device=device,
        dtype=torch.float32,
    )

    logits_indices = torch.arange(
        case.rows,
        device=device,
        dtype=torch.int32,
    )

    words = triton.cdiv(
        case.vocab,
        32,
    )

    if case.pattern == "all_allowed":
        bitmask = torch.full(
            (case.rows, words),
            -1,
            device=device,
            dtype=torch.int32,
        )

    elif case.pattern == "all_blocked":
        bitmask = torch.zeros(
            (case.rows, words),
            device=device,
            dtype=torch.int32,
        )

    elif case.pattern == "random":
        generator = torch.Generator(
            device="cpu"
        )
        generator.manual_seed(case.seed)

        bitmask_cpu = torch.randint(
            -(2**31),
            2**31 - 1,
            (case.rows, words),
            dtype=torch.int32,
            generator=generator,
        )

        bitmask = bitmask_cpu.to(device)

    else:
        raise ValueError(
            "pattern must be "
            "all_allowed|all_blocked|random, "
            f"got {case.pattern!r}"
        )

    return {
        "logits": logits,
        "logits_indices": logits_indices,
        "bitmask": bitmask,
    }


def launch_kernel(
    kernel: Any,
    case: Case,
    tensors: dict[str, torch.Tensor],
) -> None:
    logits = tensors["logits"]
    logits_indices = tensors["logits_indices"]
    bitmask = tensors["bitmask"]

    grid = (
        case.rows,
        triton.cdiv(
            case.vocab,
            BLOCK_SIZE,
        ),
    )

    kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        bitmask,
        bitmask.stride(0),
        case.vocab,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def build_reference(
    case: Case,
    base: dict[str, torch.Tensor],
) -> torch.Tensor:
    expected = base["logits"].clone()
    packed = base["bitmask"].to(
        torch.int64
    )

    bit_ids = torch.arange(
        32,
        device=packed.device,
        dtype=torch.int64,
    )

    bit_values = (
        torch.ones(
            (32,),
            device=packed.device,
            dtype=torch.int64,
        )
        << bit_ids
    )

    blocked = (
        packed[:, :, None]
        & bit_values[None, None, :]
    ) == 0

    blocked = blocked.reshape(
        case.rows,
        -1,
    )

    blocked = blocked[
        :,
        : case.vocab,
    ]

    expected.masked_fill_(
        blocked,
        -float("inf"),
    )

    return expected


def validate_kernel(
    kernel: Any,
    case: Case,
    base: dict[str, torch.Tensor],
) -> dict[str, Any]:
    tensors = {
        "logits": base["logits"].clone(),
        "logits_indices": base[
            "logits_indices"
        ],
        "bitmask": base["bitmask"],
    }

    expected = build_reference(
        case,
        base,
    )

    launch_kernel(
        kernel,
        case,
        tensors,
    )

    synchronize(
        base["logits"].device
    )

    actual = tensors["logits"]

    equal = torch.equal(
        actual,
        expected,
    )

    mismatch_count = int(
        torch.count_nonzero(
            actual != expected
        ).item()
    )

    return {
        "pass": bool(equal),
        "mismatch_count": mismatch_count,
    }


def percentile(
    values: list[float],
    q: float,
) -> float:
    if len(values) == 1:
        return values[0]

    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(
        lo + 1,
        len(xs) - 1,
    )
    frac = pos - lo

    return (
        xs[lo] * (1.0 - frac)
        + xs[hi] * frac
    )


def measure_wall_us(
    fn,
    *,
    iters: int,
    device: torch.device,
) -> float:
    synchronize(device)
    start = time.perf_counter()

    for _ in range(iters):
        fn()

    synchronize(device)

    return (
        (time.perf_counter() - start)
        * 1e6
        / iters
    )


def measure_device_us(
    fn,
    *,
    iters: int,
    device: torch.device,
) -> float:
    accelerator = accelerator_module(
        device
    )

    start = accelerator.Event(
        enable_timing=True
    )
    end = accelerator.Event(
        enable_timing=True
    )

    start.record()

    for _ in range(iters):
        fn()

    end.record()
    end.synchronize()

    return (
        float(start.elapsed_time(end))
        * 1000.0
        / iters
    )


def benchmark_group(
    loaded_kernels: list[
        tuple[KernelSpec, Any]
    ],
    case: Case,
    base: dict[str, torch.Tensor],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    timing: str,
    order: str,
    seed: int,
) -> dict[str, list[float]]:
    device = base["logits"].device

    run_inputs: dict[
        str,
        dict[str, torch.Tensor],
    ] = {}

    for spec, _ in loaded_kernels:
        run_inputs[spec.label] = {
            "logits": base[
                "logits"
            ].clone(),
            "logits_indices": base[
                "logits_indices"
            ],
            "bitmask": base["bitmask"],
        }

    # Compile each kernel once, then warm it independently.
    for spec, kernel in loaded_kernels:
        tensors = run_inputs[
            spec.label
        ]

        launch_kernel(
            kernel,
            case,
            tensors,
        )
        synchronize(device)

        for _ in range(warmup):
            launch_kernel(
                kernel,
                case,
                tensors,
            )

        synchronize(device)

    measure = (
        measure_device_us
        if timing == "device"
        else measure_wall_us
    )

    timings: dict[
        str,
        list[float],
    ] = {
        spec.label: []
        for spec, _ in loaded_kernels
    }

    rng = random.Random(seed)

    for repeat_idx in range(repeats):
        current = list(
            loaded_kernels
        )

        if order == "balanced-random":
            rng.shuffle(current)

        elif (
            order == "reverse-alternate"
            and repeat_idx % 2
        ):
            current.reverse()

        elif order != "fixed":
            raise ValueError(
                f"unsupported order: {order}"
            )

        for spec, kernel in current:
            tensors = run_inputs[
                spec.label
            ]

            def fn(
                kernel=kernel,
                tensors=tensors,
            ):
                launch_kernel(
                    kernel,
                    case,
                    tensors,
                )

            latency_us = measure(
                fn,
                iters=iters,
                device=device,
            )

            timings[
                spec.label
            ].append(latency_us)

    return timings


def summarize(
    values: list[float],
) -> dict[str, float]:
    return {
        "p10_us": percentile(
            values,
            0.10,
        ),
        "p50_us": percentile(
            values,
            0.50,
        ),
        "p90_us": percentile(
            values,
            0.90,
        ),
        "mean_us": statistics.mean(
            values
        ),
        "min_us": min(values),
        "max_us": max(values),
        "std_us": (
            statistics.pstdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


def run_performance_case(
    loaded_kernels: list[
        tuple[KernelSpec, Any]
    ],
    case: Case,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base = build_inputs(
        case,
        args.device,
    )

    validation: dict[
        str,
        Any,
    ] = {}

    if args.preflight:
        for spec, kernel in loaded_kernels:
            validation[
                spec.label
            ] = validate_kernel(
                kernel,
                case,
                base,
            )

        failed = [
            label
            for label, result
            in validation.items()
            if not result["pass"]
        ]

        if failed:
            raise AssertionError(
                "correctness preflight "
                f"failed: {failed}"
            )

    timings = benchmark_group(
        loaded_kernels,
        case,
        base,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        timing=args.timing,
        order=args.order,
        seed=args.seed,
    )

    kernel_results: dict[
        str,
        Any,
    ] = {}

    for spec, _ in loaded_kernels:
        kernel_results[
            spec.label
        ] = {
            "kernel": dataclasses.asdict(
                spec
            ),
            "validation": validation.get(
                spec.label
            ),
            "timing": summarize(
                timings[spec.label]
            ),
        }

    return {
        "case": dataclasses.asdict(case),
        "launch": {
            "BLOCK_SIZE": BLOCK_SIZE,
            "grid": [
                case.rows,
                triton.cdiv(
                    case.vocab,
                    BLOCK_SIZE,
                ),
            ],
        },
        "task": "performance",
        "timing_mode": args.timing,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "order": args.order,
        "kernels": kernel_results,
    }


def run_correctness_case(
    loaded_kernels: list[
        tuple[KernelSpec, Any]
    ],
    case: Case,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base = build_inputs(
        case,
        args.device,
    )

    results = {}

    for spec, kernel in loaded_kernels:
        results[
            spec.label
        ] = {
            "kernel": dataclasses.asdict(
                spec
            ),
            "validation": validate_kernel(
                kernel,
                case,
                base,
            ),
        }

    return {
        "case": dataclasses.asdict(case),
        "task": "correctness",
        "kernels": results,
    }


def run_simulator_case(
    loaded_kernels: list[
        tuple[KernelSpec, Any]
    ],
    case: Case,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if len(loaded_kernels) != 1:
        raise ValueError(
            "simulator task requires "
            "exactly one --kernel"
        )

    spec, kernel = loaded_kernels[0]

    base = build_inputs(
        case,
        args.device,
    )

    launch_kernel(
        kernel,
        case,
        base,
    )

    synchronize(
        base["logits"].device
    )

    contract = {
        "case": dataclasses.asdict(case),
        "kernel": dataclasses.asdict(spec),
        "launch": {
            "BLOCK_SIZE": BLOCK_SIZE,
            "grid": [
                case.rows,
                triton.cdiv(
                    case.vocab,
                    BLOCK_SIZE,
                ),
            ],
        },
        "inputs": {
            name: {
                "shape": list(
                    tensor.shape
                ),
                "dtype": str(
                    tensor.dtype
                ),
                "stride": list(
                    tensor.stride()
                ),
                "device": str(
                    tensor.device
                ),
            }
            for name, tensor
            in base.items()
        },
    }

    if args.dump_inputs:
        torch.save(
            {
                key: value.cpu()
                for key, value
                in base.items()
            },
            args.dump_inputs,
        )

        contract[
            "dump_inputs"
        ] = str(
            Path(
                args.dump_inputs
            ).resolve()
        )

    if args.dump_contract:
        Path(
            args.dump_contract
        ).write_text(
            json.dumps(
                contract,
                indent=2,
            ),
            encoding="utf-8",
        )

    return {
        "task": "simulator",
        **contract,
    }


def print_performance(
    result: dict[str, Any],
) -> None:
    case = result["case"]
    launch = result["launch"]

    print(
        f"\ncase={case['name']} "
        f"rows={case['rows']} "
        f"vocab={case['vocab']} "
        f"pattern={case['pattern']} "
        f"grid={tuple(launch['grid'])} "
        f"BLOCK_SIZE="
        f"{launch['BLOCK_SIZE']}"
    )

    first_label = next(
        iter(result["kernels"])
    )

    baseline_p50 = result[
        "kernels"
    ][first_label]["timing"]["p50_us"]

    for label, item in result[
        "kernels"
    ].items():
        timing = item["timing"]

        speedup = (
            baseline_p50
            / timing["p50_us"]
        )

        print(
            f"kernel={label:<24} "
            f"p50={timing['p50_us']:.3f} us "
            f"p10={timing['p10_us']:.3f} "
            f"p90={timing['p90_us']:.3f} "
            f"mean={timing['mean_us']:.3f} "
            f"min={timing['min_us']:.3f} "
            f"max={timing['max_us']:.3f} "
            f"speedup_vs_{first_label}="
            f"{speedup:.3f}x"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kernel",
        action="append",
        required=True,
        help=(
            "LABEL=FILE:ENTRY; repeat for "
            "N-kernel comparison"
        ),
    )

    parser.add_argument(
        "--task",
        choices=(
            "performance",
            "correctness",
            "simulator",
        ),
        default="performance",
    )

    parser.add_argument("--case")
    parser.add_argument("--cases")
    parser.add_argument("--cases-file")
    parser.add_argument("--case-json")
    parser.add_argument(
        "--rows",
        type=int,
    )
    parser.add_argument(
        "--vocab",
        type=int,
    )

    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "accelerator device: auto, cuda:N, "
            "or npu:N (default: auto)"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--timing",
        choices=(
            "wall",
            "device",
        ),
        default="wall",
    )

    parser.add_argument(
        "--order",
        choices=(
            "balanced-random",
            "reverse-alternate",
            "fixed",
        ),
        default="balanced-random",
    )

    parser.add_argument(
        "--preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--dump-inputs"
    )
    parser.add_argument(
        "--dump-contract"
    )
    parser.add_argument(
        "--json-out"
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    device = resolve_device(
        args.device
    )

    accelerator_module(
        device
    ).set_device(device)

    args.device = str(device)

    loaded_kernels = [
        load_kernel(text)
        for text in args.kernel
    ]

    cases = resolve_cases(args)

    all_results: list[
        dict[str, Any]
    ] = []

    for case in cases:
        if args.task == "performance":
            result = run_performance_case(
                loaded_kernels,
                case,
                args,
            )

            print_performance(result)

        elif args.task == "correctness":
            result = run_correctness_case(
                loaded_kernels,
                case,
                args,
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                )
            )

        else:
            result = run_simulator_case(
                loaded_kernels,
                case,
                args,
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                )
            )

        all_results.append(result)

    if args.json_out:
        output = {
            "runtime": {
                "python": (
                    sys.version.split()[0]
                ),
                "torch": torch.__version__,
                "device": args.device,
            },
            "results": all_results,
        }

        Path(
            args.json_out
        ).write_text(
            json.dumps(
                output,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()