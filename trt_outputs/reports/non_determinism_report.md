# FastSurfer Determinism Investigation

Date: 2026-05-12  
Checkout: `dev` at `528fc86` (`remove mac requirements`)  
Runtime: cached Docker image `deepmi/fastsurfer:validation-fastsurferdev0d9a962-fastsurfer-dev-0d9a962`, with the current checkout bind-mounted at `/fastsurfer-dev`.

## Inputs

- `/groups/ag-reuter/datasets/hcp_tr_t1/fs72_hires/114823_MR1/mri/orig.mgz`
- `/groups/ag-reuter/datasets/hcp_tr_t1/fs72_hires/114823_MR2/mri/orig.mgz`
- `/groups/ag-reuter/datasets/hcp_tr_t1/fs72_hires/115320_MR1/mri/orig.mgz`

## Environment Notes

- The checkout has no `.env`/`.venv`, no host `torch`, and no host FreeSurfer CLI.
- Docker was required for a usable FastSurfer/FreeSurfer/PyTorch runtime.
- `/groups` could not be mounted wholesale because container UID/GID traversal hit permissions. A narrow bind mount of `/groups/ag-reuter/datasets/hcp_tr_t1/fs72_hires` worked.
- The checkout has no checkpoint directory, so checkpoint files were copied from the cached runtime image into `trt_outputs/checkpoints` and mounted read-only over `/fastsurfer-dev/checkpoints`.

## Completed Runs

### `114823_MR1`, `--threads 8`, two full segmentation repeats

Outputs:

- `trt_outputs/seg_current_threads8_run1/114823_MR1`
- `trt_outputs/seg_current_threads8_run2/114823_MR1`
- comparison JSON: `trt_outputs/reports/compare_threads8_run1_vs_run2_114823_MR1.json`

Result:

- All image voxel arrays compared equal.
- Stats table numeric rows compared equal after removing comment headers.
- Differences were metadata/path/timestamp only:
  - `stats/*`: `cmdline`, container hostname, output paths, file timestamps.
  - `mri/transforms/*.lta`: output path and creation timestamp.
  - `surf/callosum.surf`: four bytes in the FreeSurfer surface creation timestamp only.

Timing:

- Run1 wall time: 30:11.07.
- Run2 wall time: 26:10.34.
- Same settings still had substantial performance variability, likely host contention.

### `114823_MR1`, `--threads 1`, partial segmentation

Output:

- `trt_outputs/seg_current_threads1_run1/114823_MR1`

Stopped after HypVINN axial because CPU runtime was impractical.

Timing:

- FastSurferVINN coronal: 657.5145 s.
- FastSurferVINN sagittal: 663.1117 s.
- FastSurferVINN axial: 666.1536 s.
- CerebNet: 131.12 s.
- HypVINN axial: 928.0220 s.

Comparison against `--threads 8` run1:

- `aparc.DKTatlas+aseg.deep.mgz` differed in 1 voxel: index `(148, 187, 197)`, label `2026` vs `58`.
- Downstream CC/aseg outputs differed in 10 voxels.
- N4 outputs differed by small intensity changes (`orig_nu.mgz`: 243 voxels, max abs diff 1).

Interpretation:

- Same-thread repeated outputs are stable for tested full run.
- Cross-thread outputs are not numerically identical. This is deterministic thread-count sensitivity, not observed run-to-run nondeterminism.

### Parallel all-subject attempt

Output:

- `trt_outputs/seg_patched_warn_threads8_run1`

Stopped during HypVINN. Three concurrent 8-thread containers caused severe oversubscription:

- FastSurferVINN coronal per subject: ~1128-1166 s.
- FastSurferVINN sagittal per subject: ~1188-1208 s.
- FastSurferVINN axial per subject: ~1106-1121 s.
- CerebNet per subject: ~631-649 s.
- HypVINN axial per subject: ~1111-1151 s.

This was slower than running one subject at a time and was not continued to a second repeat.

## Sources Identified

### 1. PyTorch backend determinism

The pipeline did not previously configure PyTorch deterministic backend behavior beyond setting seeds.

Patch added:

- `FastSurferCNN/utils/determinism.py`
- Calls in:
  - `FastSurferCNN/inference.py`
  - `CerebNet/inference.py`
  - `HypVINN/inference.py`

The helper sets:

- `CUBLAS_WORKSPACE_CONFIG`
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- `torch.backends.cudnn.benchmark = False`
- `torch.backends.cudnn.deterministic = True`
- TF32 disabled for CUDA matmul/cuDNN.

Important limitation:

- Strict mode (`warn_only=False`) fails immediately because PyTorch reports `max_unpooling2d_forward_out` has no deterministic implementation in this runtime.
- Therefore FastSurfer can only warn about that operation unless the model/runtime is changed to avoid that op or PyTorch gains a deterministic implementation.

### 2. Reproducible metadata

FastSurfer-generated metadata introduced byte differences even when arrays/stat tables were numerically identical.

Patched:

- `FastSurferCNN/utils/lta.py`
  - Uses `SOURCE_DATE_EPOCH` for `created by ... on ...` lines.
  - Handles UID-only containers by falling back to `UNKNOWN` username.
- `FastSurferCNN/segstats.py`
  - Uses `SOURCE_DATE_EPOCH` for annotated input/output file timestamps.
- `CorpusCallosum/shape/mesh.py`
  - When `SOURCE_DATE_EPOCH` is set, patches nibabel surface writing to use a reproducible FreeSurfer `create_stamp`.

Verification:

- `python3 -m py_compile` passed for patched modules.
- Docker smoke test confirmed PyTorch deterministic algorithms are enabled in warning mode.
- LTA test with `SOURCE_DATE_EPOCH=0` writes `Thu Jan  1 00:00:00 1970`.
- Mini CC surface write with `SOURCE_DATE_EPOCH=0` produced identical bytes across two writes.

Remaining metadata caveat:

- Stats headers still include command lines and output paths. Runs written to different output directories will still differ in those header path strings by design.

## Performance Observations

- `--threads 8` improved FastSurferVINN per-plane inference over `--threads 1`, but not linearly.
- Single subject `114823_MR1`:
  - `--threads 1`: ~657-666 s per FastSurferVINN plane.
  - `--threads 8`: ~165-238 s per FastSurferVINN plane in single-subject repeats.
- Running three 8-thread subjects concurrently was much worse:
  - ~1106-1208 s per FastSurferVINN plane per subject.
- The host had no usable NVIDIA driver (`nvidia-smi` failed), so all inference was CPU-only.

## Current Status

Not fully complete against the original completion criterion.

Completed:

- Identified and patched deterministic backend configuration.
- Identified and patched reproducible metadata support for FastSurfer-generated LTA/stats/CC surface headers.
- Demonstrated full same-thread repeat stability for one requested image at `--threads 8`.
- Demonstrated cross-thread numerical differences and recorded their size/location.
- Recorded performance regressions for single-thread and oversubscribed parallel runs.

Remaining:

- Full repeated verification for all three requested images after the patch.
- Full surface pipeline verification; only `--seg_only` was run because CPU-only full segmentation already took tens of minutes per subject.
- Resolving PyTorch `max_unpooling2d_forward_out` deterministic warning would require replacing/avoiding that operation or using a PyTorch runtime that implements it deterministically.
