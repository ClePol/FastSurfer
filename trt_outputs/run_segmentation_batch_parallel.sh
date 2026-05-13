#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <run-label> <threads-per-subject>" >&2
  exit 2
fi

run_label="$1"
threads="$2"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="${root_dir}/trt_outputs/run_segmentation_one.sh"

subjects=(
  "114823_MR1"
  "114823_MR2"
  "115320_MR1"
)

pids=()
for sid in "${subjects[@]}"; do
  "${runner}" "${run_label}" "${threads}" "${sid}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

exit "${failed}"
