#!/usr/bin/env bash
set -euo pipefail

scene_id="${1:?usage: bash run_msprof.sh T0}"
profile_root="${PROFILE_ROOT:-/home/lingmutian/tmp/profile_dir}"

msprof \
  --application="python run_offline.py --scene ${scene_id} --profile none" \
  --output="${profile_root}/msprof/${scene_id}" \
  --model-execution=on \
  --runtime-api=on \
  --aicpu=on \
  --ai-core=on \
  --task-time=on
