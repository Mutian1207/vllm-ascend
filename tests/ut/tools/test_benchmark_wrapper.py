import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[3] / "benchmarks" / "ops" / "benchmark_wrapper.py"
SPEC = importlib.util.spec_from_file_location("benchmark_wrapper", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark_wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_wrapper)


def test_load_json_object(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps({"wrapper": "module:function"}))

    assert benchmark_wrapper.load_cases(path) == [{"wrapper": "module:function"}]


def test_load_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"name": "first"}\n{"name": "second"}\n')

    assert [case["name"] for case in benchmark_wrapper.load_cases(path)] == ["first", "second"]


def test_select_cases_from_mixed_capture():
    cases = [
        {"kernel": "first", "arguments": {}},
        {"kernel": "second", "arguments": {}},
        {"kernel": "first", "arguments": {"size": 2}},
    ]

    assert benchmark_wrapper.select_cases(cases, "first") == [cases[0], cases[2]]
    assert benchmark_wrapper.select_cases(cases, "first", max_cases=1) == [cases[0]]


def test_select_cases_rejects_missing_kernel():
    with pytest.raises(ValueError, match="No input records"):
        benchmark_wrapper.select_cases([{"kernel": "first"}], "missing")


def test_select_cases_rejects_invalid_max_cases():
    with pytest.raises(ValueError, match="max_cases must be a positive integer"):
        benchmark_wrapper.select_cases([{"kernel": "first"}], "first", max_cases=0)


def test_resolve_wrapper():
    assert benchmark_wrapper.resolve_wrapper("json:loads") is json.loads


def test_resolve_wrapper_requires_colon():
    with pytest.raises(ValueError, match="file.py:function"):
        benchmark_wrapper.resolve_wrapper("json.loads")


def test_resolve_wrapper_from_file(tmp_path):
    wrapper_file = tmp_path / "wrapper.py"
    wrapper_file.write_text("def add_one(value):\n    return value + 1\n")

    wrapper = benchmark_wrapper.resolve_wrapper(f"{wrapper_file}:add_one")

    assert wrapper(2) == 3


def test_percentile_and_summary():
    values = [1.0, 2.0, 3.0, 4.0]

    assert benchmark_wrapper.percentile(values, 50) == 2.5
    assert benchmark_wrapper.summarize(values) == {
        "min_ms": 1.0,
        "max_ms": 4.0,
        "mean_ms": 2.5,
        "p50_ms": 2.5,
        "p90_ms": pytest.approx(3.7),
        "p99_ms": pytest.approx(3.97),
    }


def test_build_wrapper_invoker():
    invoke = benchmark_wrapper.build_invoker(lambda value: value + 1, "wrapper", None)

    assert invoke([2], {}) == 3


def test_build_triton_invoker_uses_captured_grid():
    launches = []

    class FakeKernel:
        arg_names = ["output_ptr", "BLOCK_SIZE"]

        def __getitem__(self, grid):
            launches.append(("grid", grid))
            return lambda **kwargs: launches.append(("arguments", kwargs))

    invoke = benchmark_wrapper.build_invoker(FakeKernel(), "triton", [8, 1])
    invoke([], {"output": "tensor", "BLOCK_SIZE": 1024})

    assert launches == [
        ("grid", (8, 1)),
        ("arguments", {"output_ptr": "tensor", "BLOCK_SIZE": 1024}),
    ]


@pytest.mark.parametrize("grid", [None, [], [0, 1], ["8", 1]])
def test_build_triton_invoker_rejects_invalid_grid(grid):
    with pytest.raises(ValueError, match="Triton mode requires"):
        benchmark_wrapper.build_invoker(object(), "triton", grid)


@pytest.mark.parametrize("shape", [None, [1, -1], [1, "2"]])
def test_invalid_tensor_shape(shape):
    with pytest.raises(ValueError, match="Tensor shape"):
        benchmark_wrapper._make_tensor({"shape": shape}, "cpu")


def test_materialize_cpu_tensor():
    value = benchmark_wrapper.materialize(
        {"shape": [2, 3], "dtype": "torch.int32", "initializer": "ones"},
        "cpu",
    )

    assert value.shape == (2, 3)
    assert value.dtype == benchmark_wrapper.torch.int32
    assert value.tolist() == [[1, 1, 1], [1, 1, 1]]


def test_arguments_field_can_supply_keyword_arguments(monkeypatch):
    calls = []

    class FakeEvent:
        def record(self):
            pass

        def elapsed_time(self, _end):
            return 1.25

    class FakeNPU:
        Event = lambda self, enable_timing: FakeEvent()  # noqa: E731

        def set_device(self, _device):
            pass

        def synchronize(self):
            pass

    monkeypatch.setattr(benchmark_wrapper, "resolve_wrapper", lambda _path: lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(benchmark_wrapper.torch, "npu", FakeNPU())

    result = benchmark_wrapper.benchmark_case(
        {
            "wrapper": "module:function",
            "device": "npu:0",
            "warmup": 1,
            "profiling_rounds": 2,
            "arguments": {"block_size": 128},
        }
    )

    assert calls == [{"block_size": 128}] * 3
    assert result["latencies_ms"] == [1.25, 1.25]


def test_device_override_rewrites_captured_tensor_devices():
    args = type(
        "Args",
        (),
        {
            "wrapper": None,
            "mode": None,
            "device": "npu:1",
            "warmup": None,
            "profiling_rounds": None,
        },
    )()
    case = {
        "arguments": {
            "tensor": {"shape": [2], "dtype": "float32", "device": "npu:0"},
            "scalar": 4,
        }
    }

    overridden = benchmark_wrapper._override(case, args)

    assert overridden["arguments"]["tensor"]["device"] == "npu:1"
    assert overridden["arguments"]["scalar"] == 4
