#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <run-label> <threads> <subject-id> <seg-source-label>" >&2
  exit 2
fi

run_label="$1"
threads="$2"
sid="$3"
seg_source_label="$4"

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_subject="${root_dir}/trt_outputs/${seg_source_label}/${sid}"
dest_root="${root_dir}/trt_outputs/${run_label}"
dest_subject="${dest_root}/${sid}"
log_dir="${root_dir}/trt_outputs/logs/${run_label}"

if [[ ! -d "${source_subject}" ]]; then
  echo "missing source subject: ${source_subject}" >&2
  exit 1
fi

if [[ -e "${dest_subject}" ]]; then
  echo "destination already exists: ${dest_subject}" >&2
  exit 1
fi

mkdir -p "${dest_root}" "${log_dir}"
cp -a "${source_subject}" "${dest_subject}"

image="deepmi/fastsurfer:validation-fastsurferdev0d9a962-fastsurfer-dev-0d9a962"
license="${root_dir}/trt_outputs/fs_license.txt"
data_root="/groups/ag-reuter/datasets/hcp_tr_t1/fs72_hires"
uid_gid="$(id -u):$(id -g)"
user_name="$(id -un)"
log_file="${log_dir}/${sid}.log"

echo "[$(date --iso-8601=seconds)] starting surface ${sid} (${threads} threads)" | tee "${log_file}"
/usr/bin/time -v docker run --rm \
  --user "${uid_gid}" \
  --env FASTSURFER_HOME=/fastsurfer-dev \
  --env PYTHONHASHSEED=0 \
  --env SOURCE_DATE_EPOCH=0 \
  --env TQDM_DISABLE=1 \
  --env USER="${user_name}" \
  --env LOGNAME="${user_name}" \
  -v "${root_dir}:/fastsurfer-dev:ro" \
  -v "${root_dir}/trt_outputs/checkpoints:/fastsurfer-dev/checkpoints:ro" \
  -v "${data_root}:/data/hcp:ro" \
  -v "${root_dir}/trt_outputs:/out" \
  -v "${license}:/fs_license/license.txt:ro" \
  --entrypoint /fastsurfer-dev/run_fastsurfer.sh \
  "${image}" \
  --sd "/out/${run_label}" \
  --sid "${sid}" \
  --t1 "/data/hcp/${sid}/mri/orig.mgz" \
  --surf_only \
  --threads "${threads}" \
  --fs_license /fs_license/license.txt \
  >> "${log_file}" 2>&1
echo "[$(date --iso-8601=seconds)] finished surface ${sid}" | tee -a "${log_file}"
