# Surface Threaded Topology-Fix Speedup

Date: 2026-05-14

Subject: `114823_MR1`

Reference output:

- `trt_outputs/surf_patched_all_threads8_run2/114823_MR1`

Optimized output:

- `trt_outputs/surf_speed_topofix_threads8_run1/114823_MR1`

Change:

- Keep seeded deterministic spherical projection.
- Run `recon-all -fix` with the existing per-hemisphere surface thread count instead of forcing it to one thread.
- Keep `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1` around the topology-fix command.

Rationale:

- The earlier nondeterminism was introduced by unseeded spherical projection.
- After qsphere is seeded, isolated `mris_fix_topology` runs with 4 threads matched the serial reference exactly for `rh.orig.premesh`.

Isolated validation:

- `rh.orig.premesh.thread4a` vs serial `rh.orig.premesh`: identical coordinates and faces.
- `rh.orig.premesh.thread4b` vs serial `rh.orig.premesh`: identical coordinates and faces.
- `rh.orig.premesh.thread4a` vs `rh.orig.premesh.thread4b`: identical coordinates and faces.

Full-run validation:

- Comparator: `trt_outputs/reports/compare_surface_reference2_vs_threaded_topofix_114823_MR1.json`
- Missing files: `0`
- Image voxel differences: none reported.
- Surface coordinate/face differences: none reported.
- Morphometry differences: none reported.
- Label differences: none reported.
- Annotation differences: none reported.
- Remaining differences: `35`
- Remaining difference kinds: `26` text/provenance files and `9` metadata files.
- Known comparator error: `stats/aseg.auto.mgz` is a text stats file with an `.mgz` suffix.

Numeric stats rows were identical after stripping comment/header lines for:

- `aseg.stats`
- `aseg.presurf.hypos.stats`
- `lh.aparc.DKTatlas.mapped.stats`
- `rh.aparc.DKTatlas.mapped.stats`
- `wmparc.DKTatlas.mapped.stats`
- `lh.BA_exvivo.stats`
- `rh.BA_exvivo.stats`
- `lh.w-g.pct.stats`
- `rh.w-g.pct.stats`

Timing:

- Reference full surface wall time: `54:43.81`
- Optimized full surface wall time: `51:15.02`
- Full-run speedup: about `3:29`

Topology-fix component timings:

- Left hemisphere reference: `0.0209 h`, 1 thread
- Left hemisphere optimized: `0.0138 h`, 4 threads
- Right hemisphere reference: `0.0910 h`, 1 thread
- Right hemisphere optimized: `0.0777 h`, 4 threads

The component improvement is about `1:13`, while the full-run wall-time improvement is about `3:29` for this case.
