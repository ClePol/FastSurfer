# Surface speedup: overlap final statistics

## Change

`recon_surf/recon-surf.sh` now schedules two independent final surface-statistics tasks in the background:

- mapped DKT anatomical stats (`lh.aparc.DKTatlas.mapped.stats`, `rh.aparc.DKTatlas.mapped.stats`)
- Brodmann-area label/stat generation (`fs_balabels.py`)

Both tasks start after the cortical ribbon is available. They only depend on completed surface geometry, surface registration, annotations, and ribbon-era inputs, so they can overlap with the later `pctsurfcon`, hypointensity relabeling, mapped volume creation, and segmentation-statistics chain. The script waits for both background command files before final completion and appends their logs into the main `recon-surf.log`.

## Timing

Validated on `114823_MR1`, `--surf_only --threads 8`, using Docker with mapped UID/GID.

- Previous optimized reference (`b640cfa` + direct deterministic sphere morphing): `47:25.36`
- Async final statistics run: `46:32.96`
- Wall-time speedup: `52.40 seconds`

The backgrounded tasks themselves took:

- mapped DKT stats: `21.69s + 9.74s`
- BA labels/stats: `32.85s`

Because they overlap the final volume/statistics chain, the wall-time gain is larger than 30 seconds without changing numerical work.

## Validation

Compared against `trt_outputs/surf_patched_all_threads8_run2/114823_MR1` with `trt_outputs/compare_surface_outputs.py`.

Comparator output:

- JSON: `trt_outputs/reports/compare_surface_reference2_vs_async_stats_114823_MR1.json`
- No image, surface geometry, morphometry, label, or annotation data differences were reported.
- Remaining differences are expected generated text/provenance differences: logs, command files, touch files, path/hostname/time-bearing transform text, and stats headers.
- Representative stats body check: `grep -v '^#'` comparison for `lh.aparc.DKTatlas.mapped.stats` was identical.
- The known comparator parser limitation for `stats/aseg.auto.mgz` is still present (`BadGzipFile` on a text-formatted file).

## Determinism risk

The change does not introduce shared output writes between the background jobs and the main final chain. The only shared log is appended after each background command file exits, preserving complete diagnostics while avoiding concurrent writes to `recon-surf.log`.
