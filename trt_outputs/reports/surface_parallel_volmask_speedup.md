# Surface parallel ribbon speedup

Date: 2026-05-14

## Change

Replaced the `recon-all -cortribbon` wrapper in `recon_surf/recon-surf.sh` with the equivalent direct `mris_volmask` call:

```bash
mris_volmask --aseg_name aseg.presurf \
  --label_left_white 2 --label_left_ribbon 3 \
  --label_right_white 41 --label_right_ribbon 42 \
  --save_ribbon --parallel 114823_MR1
```

The `--parallel` flag is used only when the requested FastSurfer thread count is greater than 1. The pipeline also writes `touch/cortical_ribbon.touch` after the direct call to preserve the expected recon-all touch marker.

## Validation Run

- Subject: `114823_MR1`
- Mode: `--surf_only --threads 8`
- Reference: `trt_outputs/surf_patched_all_threads8_run2/114823_MR1`
- Candidate: `trt_outputs/surf_speed_volmask_parallel_threads8_run1/114823_MR1`
- Comparator: `trt_outputs/reports/compare_surface_reference2_vs_parallel_volmask_114823_MR1.json`

## Timing

- Previous optimized surface wall time: `51:15.02`
- Candidate wall time: `48:22.34`
- Net wall-time speedup: `2:52.68`
- Previous ribbon block: `450.01s` via `recon-all -cortribbon`
- Candidate ribbon block: `263.48s` via direct `mris_volmask --parallel`
- Component speedup: `186.53s`

## Output Comparison

The surface-aware comparator reported no image voxel differences, no surface coordinate or face differences, no morphometry differences, no label differences, and no annotation differences.

Remaining differences are text/provenance/runtime files and touch/command metadata. The only missing file in the candidate is `surf/rh.orig.premesh.thread8a`, which is a scratch artifact left in the reference directory from an isolated topology timing test, not a normal pipeline output. The known comparator issue for `stats/aseg.auto.mgz` is unchanged.
