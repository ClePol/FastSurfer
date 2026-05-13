# All Test-Case Determinism Verification

Date: 2026-05-13

Subjects:

- `114823_MR1`
- `114823_MR2`
- `115320_MR1`

Docker was run with mapped UID/GID via `--user "$(id -u):$(id -g)"`. The runner also set `USER` and `LOGNAME` to handle unmapped numeric UIDs inside the container.

## Commits

- `c032e74` Make segmentation inference deterministic
- `fb7eddf` Make surface reconstruction deterministic
- `324783c` Make generated metadata reproducible
- `298e5e6` Add determinism investigation artifacts
- `972d78d` Use mapped-user compatible test license mount
- `362baaa` Handle unmapped UIDs in Docker test runners

## Segmentation Runs

Repeated patched segmentation runs:

- `trt_outputs/seg_patched_all_threads8_run1c`
- `trt_outputs/seg_patched_all_threads8_run2`

Comparison report:

- `trt_outputs/reports/compare_seg_patched_all_threads8_run1c_vs_run2.json`

Comparator result:

- missing files: `0`
- data-image voxel differences reported by comparator: `0`
- remaining differences: `27`
- remaining difference kind: regular/text-like files only

The remaining differences are path/provenance text in LTAs and stats files, including output-root labels such as `seg_patched_all_threads8_run1c` vs `seg_patched_all_threads8_run2`.

Numeric stats rows were identical after stripping comment/header lines for checked files:

- `aseg+DKT.VINN.stats`
- `aseg.VINN.stats`
- `cerebellum.CerebNet.stats`
- `hypothalamus.HypVINN.stats`

Wall times:

- run 1:
  - `114823_MR1`: `25:41.79`
  - `114823_MR2`: `24:05.09`
  - `115320_MR1`: `24:25.26`
- run 2:
  - `114823_MR1`: `40:27.22`
  - `114823_MR2`: `36:17.41`
  - `115320_MR1`: `26:07.29`

Performance note: deterministic PyTorch settings caused a clear CPU segmentation slowdown compared with the earlier non-deterministic/default backend runs. PyTorch also emitted `max_unpool2d_forward_out` deterministic-implementation warnings under `warn_only=True`, but repeated CPU output arrays were still identical for these subjects.

## Surface Runs

Repeated patched surface runs:

- `trt_outputs/surf_patched_all_threads8_run1`
- `trt_outputs/surf_patched_all_threads8_run2`

Both surface runs used the same segmentation source:

- `trt_outputs/seg_patched_all_threads8_run1c`

Comparison report:

- `trt_outputs/reports/compare_surface_patched_all_threads8_run1_vs_run2.json`

Comparator result:

- missing files: `0`
- image voxel differences reported by comparator: `0`
- surface geometry/face differences reported by comparator: `0`
- morphometry differences reported by comparator: `0`
- label differences reported by comparator: `0`
- annotation differences reported by comparator: `0`
- remaining differences: `105`
- remaining difference kinds: `78` text/provenance files, `27` regular metadata files
- comparator errors: `3`

The three comparator errors are `stats/aseg.auto.mgz` files. They are text stats files with an `.mgz` suffix, so the comparator tried to open them as gzip-compressed images and correctly failed with `BadGzipFile`.

Numeric stats rows were identical after stripping comment/header lines for checked files:

- `aseg.stats`
- `lh.aparc.DKTatlas.mapped.stats`
- `rh.aparc.DKTatlas.mapped.stats`
- `wmparc.DKTatlas.mapped.stats`

Wall times:

- run 1:
  - `114823_MR1`: `57:59.36`
  - `114823_MR2`: `55:42.04`
  - `115320_MR1`: `43:43.59`
- run 2:
  - `114823_MR1`: `54:43.81`
  - `114823_MR2`: `52:58.86`
  - `115320_MR1`: `41:13.06`

Performance note: surface runtime remained close to the earlier single-subject patched timing. The serialized projection and topology-fix stages did not produce an obvious surface-level regression in this all-subject run.

## Conclusion

For all three test cases, repeated patched runs were deterministic at the data-array level:

- segmentation image outputs matched voxelwise
- surface image outputs matched voxelwise
- surface meshes matched in coordinates and faces
- morphometry, label, and annotation arrays matched
- numeric stats rows matched after ignoring volatile headers

Remaining differences are provenance/path/runtime text, command files, touch/done metadata, and the known stats-file-with-`.mgz`-suffix comparator issue.
