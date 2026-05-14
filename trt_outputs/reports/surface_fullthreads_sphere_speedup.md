# Surface sphere threading speedup

## Change

The cross-sectional surface pipeline now runs the final `mris_sphere -seed 1234`
step directly instead of wrapping it in `recon-all -sphere`. This keeps the
fixed seed, preserves the deterministic output, and lets the command use the
full requested thread count for each hemisphere.

## Timing

Case: `114823_MR1`

Command shape:

- Reference: `--surf_only --threads 8`, prior optimized surface pipeline
- Candidate: `--surf_only --threads 8`, direct full-thread `mris_sphere`

Wall-clock timing:

- Prior optimized surface run: `48:22.34`
- Candidate run: `47:25.36`
- Speedup: `56.98` seconds

Sphere component timing:

- `lh` sphere: `94.10s` through `recon-all -sphere` to `72.55s` direct
  `mris_sphere`
- `rh` sphere: `305.14s` through `recon-all -sphere` to `256.77s` direct
  `mris_sphere`

The direct command still runs inside the existing parallel hemisphere batch.
This means each hemisphere can temporarily use the full thread count if their
sphere stages overlap with other threaded work, but the measured full-pipeline
wall clock remained faster on the validation case.

## Validation

Compared:

- Reference: `trt_outputs/surf_patched_all_threads8_run2/114823_MR1`
- Candidate: `trt_outputs/surf_speed_sphere_fullthreads_threads8_run1/114823_MR1`

Comparator output:

- No MRI voxel array differences were reported.
- No surface geometry differences were reported.
- No morphometry, label, or annotation data differences were reported.
- Differences were limited to logs, command/touch/provenance files, transform
  text provenance, and stats headers that contain output paths, host ids, or
  timestamps.
- The known scratch file `surf/rh.orig.premesh.thread8a` exists only in the
  reference output.
- The known comparator parser issue for `stats/aseg.auto.mgz` remained:
  `BadGzipFile: Not a gzipped file`.

Representative stats diffs (`aseg.stats`, `lh.aparc.DKTatlas.mapped.stats`,
and `wmparc.DKTatlas.mapped.stats`) showed header/provenance differences only;
the reported measures and rows were unchanged.

Detailed comparator JSON:

- `trt_outputs/reports/compare_surface_reference2_vs_fullthreads_sphere_114823_MR1.json`
