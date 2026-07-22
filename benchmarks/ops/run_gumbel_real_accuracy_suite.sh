#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

BACKEND="${BACKEND:-npu}"
if [[ "${BACKEND}" == "gpu" ]]; then
  DEVICE="${DEVICE:-cuda:0}"
else
  DEVICE="${DEVICE:-npu:0}"
fi

TEMP0_SRC="${TEMP0_SRC:-/home/lingmutian/tmp/gumbel_real_cases_temp0}"
TEMP06_SRC="${TEMP06_SRC:-/home/lingmutian/tmp/gumbel_real_cases_temp06_v2}"
WORK_DIR="${WORK_DIR:-/home/lingmutian/tmp/gumbel_accuracy_suite}"

SAMPLES="${SAMPLES:-100000}"
TOPK="${TOPK:-1024}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
MAX_ROWS_PER_CASE="${MAX_ROWS_PER_CASE:-1000000}"

TEMP0_DIR="${WORK_DIR}/temp0"
TEMP06_DIR="${WORK_DIR}/temp06"
LOG_DIR="${WORK_DIR}/logs"

mkdir -p "${TEMP0_DIR}" "${TEMP06_DIR}" "${LOG_DIR}"

if compgen -G "${TEMP0_SRC}/*.pt" >/dev/null; then
  cp -a "${TEMP0_SRC}"/*.pt "${TEMP0_DIR}/"
else
  echo "No temp=0 dump files found in ${TEMP0_SRC}" >&2
fi

if compgen -G "${TEMP06_SRC}/*.pt" >/dev/null; then
  cp -a "${TEMP06_SRC}"/*.pt "${TEMP06_DIR}/"
else
  echo "No temp=0.6 dump files found in ${TEMP06_SRC}" >&2
fi

echo "backend=${BACKEND}"
echo "device=${DEVICE}"
echo "work_dir=${WORK_DIR}"
echo "temp0_files=$(find "${TEMP0_DIR}" -maxdepth 1 -name '*.pt' | wc -l)"
echo "temp06_files=$(find "${TEMP06_DIR}" -maxdepth 1 -name '*.pt' | wc -l)"

if compgen -G "${TEMP0_DIR}/*.pt" >/dev/null; then
  echo "Running temp=0 argmax/replay verification..."
  python benchmarks/ops/compare_gumbel_precision.py verify \
    --backend "${BACKEND}" \
    --case-dir "${TEMP0_DIR}" \
    --device "${DEVICE}" \
    | tee "${LOG_DIR}/${BACKEND}_temp0_verify.json"
fi

if compgen -G "${TEMP06_DIR}/*.pt" >/dev/null; then
  echo "Running temp=0.6 upstream-style distribution accuracy..."
  python benchmarks/ops/compare_gumbel_precision.py accuracy \
    --backend "${BACKEND}" \
    --case-dir "${TEMP06_DIR}" \
    --device "${DEVICE}" \
    --samples "${SAMPLES}" \
    --topk "${TOPK}" \
    --chunk-size "${CHUNK_SIZE}" \
    --max-rows-per-case "${MAX_ROWS_PER_CASE}" \
    | tee "${LOG_DIR}/${BACKEND}_temp06_accuracy.jsonl"
fi

python - "${WORK_DIR}" "${BACKEND}" <<'PY'
import json
import sys
from pathlib import Path

import torch

work_dir = Path(sys.argv[1])
backend = sys.argv[2]
result_path = work_dir / "temp06" / "results" / f"{backend}_accuracy.pt"
if not result_path.exists():
    print(f"No accuracy result found: {result_path}")
    raise SystemExit(0)

data = torch.load(result_path, map_location="cpu")
summary = {
    "backend": backend,
    "samples": data["samples"],
    "topk": data["topk"],
    "total_items": len(data["results"]),
    "skipped_temperature_zero": 0,
    "chi2_checked": 0,
    "chi2_passed": 0,
    "chi2_failed": 0,
    "chi2_null": 0,
}

failed = []
for item in data["results"]:
    if item.get("skipped") == "temperature_zero":
        summary["skipped_temperature_zero"] += 1
        continue
    if item.get("chi2_pass_10sigma") is True:
        summary["chi2_checked"] += 1
        summary["chi2_passed"] += 1
    elif item.get("chi2_pass_10sigma") is False:
        summary["chi2_checked"] += 1
        summary["chi2_failed"] += 1
        failed.append({
            "case_file": item["case_file"],
            "row_idx": item["row_idx"],
            "chi2": item.get("chi2"),
        })
    else:
        summary["chi2_null"] += 1

summary["failed"] = failed
summary_path = work_dir / "temp06" / "results" / f"{backend}_accuracy_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"Wrote {summary_path}")
PY
