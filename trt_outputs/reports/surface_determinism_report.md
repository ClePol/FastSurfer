# FastSurfer Surface Determinism Report

Date: 2026-05-13

Repository: `/home/pollakc/testarea/FastSurfer_determinism/FastSurfer`

Focus: surface pipeline (`--surf_only` / `recon_surf`) determinism after the segmentation output had already been shown to be stable across repeated runs.

## Scope

The detailed repeated surface investigation was run on:

- `114823_MR1`
- input image: `/groups/ag-reuter/datasets/hcp_tr_t1/fs72_hires/114823_MR1/mri/orig.mgz`
- subject output roots under `trt_outputs`

The other requested images were kept as part of the broader test set, but the full repeated surface verification was limited to `114823_MR1` because surface reconstruction is the expensive part of the pipeline and this subject already exposed the right-hemisphere nondeterminism clearly.

All Docker runs used mapped user/group execution:

```bash
docker run --user "$(id -u):$(id -g)" ...
```

The container also needed `USER=pollakc` and `LOGNAME=pollakc`; otherwise Python code that calls `getpass.getuser()` failed for the mapped UID because that UID is not present in the container passwd database.

## Run Artifacts

Pre-fix repeated surface runs:

- `trt_outputs/surf_current_threads8_run1c/114823_MR1`
- `trt_outputs/surf_current_threads8_run2/114823_MR1`

Post-fix repeated surface runs:

- `trt_outputs/surf_patched_threads8_run1/114823_MR1`
- `trt_outputs/surf_patched_threads8_run2/114823_MR1`

Comparison outputs:

- `trt_outputs/reports/compare_surface_threads8_run1c_vs_run2_114823_MR1.json`
- `trt_outputs/reports/compare_surface_threads8_run1c_vs_run2_114823_MR1_v2.json`
- `trt_outputs/reports/compare_surface_patched_threads8_run1_vs_run2_114823_MR1.json`

Helper scripts added under `trt_outputs`:

- `trt_outputs/run_surface_one.sh`
- `trt_outputs/compare_surface_outputs.py`

## Pre-Fix Behavior

Two repeated `--surf_only --threads 8` runs from the same segmentation input were not deterministic.

The first substantive data-level difference appeared before topology correction in:

- `surf/rh.qsphere.nofix`

The two runs had the same topology for this intermediate file, but different coordinates:

- differing coordinate values: `9,873`
- maximum absolute coordinate difference: `7.62939453125e-06`
- mean absolute coordinate difference: `3.020531648381901e-08`

These small differences were amplified by `mris_fix_topology`.

Right hemisphere topology correction differed between repeated runs:

- run 1:
  - `Total Loglikelihood : -25.1834`
  - `nv=102402, nf=204800, ne=307200`
  - `FSRUNTIME@ mris_fix_topology rh 0.1219 hours 4 threads`
- run 2:
  - `Total Loglikelihood : -25.1833`
  - `nv=102464, nf=204924, ne=307386`
  - `FSRUNTIME@ mris_fix_topology rh 0.1713 hours 4 threads`

The left hemisphere did not show this amplification in these two runs:

- both runs ended with `nv=102710, nf=205416, ne=308124`

After the right-hemisphere topology correction diverged, many downstream files also diverged, including:

- `surf/rh.orig.premesh`
- `surf/rh.white`
- `surf/rh.pial.T1`
- `surf/rh.sphere`
- `surf/rh.sphere.reg`
- `surf/rh.thickness`
- `surf/rh.area`
- `surf/rh.area.pial`
- `surf/rh.curv`
- right-hemisphere labels and annotations
- `mri/rh.ribbon.mgz`
- `mri/ribbon.mgz`
- `mri/aseg.mgz`
- `mri/aparc.DKTatlas+aseg.mapped.mgz`
- `mri/wmparc.DKTatlas.mapped.mgz`
- derived stats files

Examples from the pre-fix comparison:

- `rh.orig.premesh`
  - run 1: coordinates `[102402, 3]`, faces `[204800, 3]`
  - run 2: coordinates `[102464, 3]`, faces `[204924, 3]`
- `rh.white` and `rh.pial.T1`
  - run 1: coordinates `[111716, 3]`, faces `[223428, 3]`
  - run 2: coordinates `[111646, 3]`, faces `[223288, 3]`
- `mri/aseg.mgz`
  - differing voxels: `21,423`
  - maximum absolute difference: `42`
- `mri/aparc.DKTatlas+aseg.mapped.mgz`
  - differing voxels: `28,025`
  - maximum absolute difference: `2035`
- `mri/wmparc.DKTatlas.mapped.mgz`
  - differing voxels: `36,784`
  - maximum absolute difference: `5002`

There were also expected metadata and provenance differences in logs, command files, transforms, and stats headers because output paths, timestamps, and runtime fields differ between repeated runs.

## Root Causes

### 1. Unseeded ARPACK start vector in spherical projection

`recon_surf/spherically_project.py` called:

```python
evals, evecs = fem.eigs(k=4)
```

LaPy's `Solver.eigs` supports a fixed random generator through `rng`. Without this, ARPACK can start from a random vector. That made the spectral spherical projection unstable at tiny floating-point scale.

Isolated test on the same `rh.smoothwm.nofix`, before the fix and with one thread:

- output files: `trt_outputs/scratch/rh.qsphere.serial1`, `trt_outputs/scratch/rh.qsphere.serial2`
- faces equal: yes
- differing coordinate values: `35,785`
- maximum absolute coordinate difference: `4.00543212890625e-05`
- mean absolute coordinate difference: `1.789789606220851e-07`

The logs also showed eigenvector sign/order differences between repeats.

### 2. Threaded numeric libraries around spectral projection

Even with the random start vector controlled, this part of the pipeline is sensitive to BLAS/OpenMP-level scheduling. To keep the projection stable independently of the global FastSurfer `--threads` setting, the projection wrapper now forces common numeric thread environment variables to one before importing NumPy/SciPy/LaPy code.

### 3. Topology correction amplifies tiny upstream differences

`mris_fix_topology` is highly sensitive to tiny changes in the qsphere input. The observed right-hemisphere topology changed by dozens of vertices and faces after the initial qsphere difference.

After feeding `mris_fix_topology` the same seeded qsphere and forcing one thread, repeated isolated topology correction produced identical geometry:

- both outputs ended with `nv=102505, nf=205006, ne=307509`
- faces equal: yes
- differing coordinate values: `0`
- maximum absolute coordinate difference: `0`

This showed that once the qsphere input is deterministic and topology correction is run serially, the topology stage is stable for this subject.

## Fixes Applied

### Seed spectral projection

File: `recon_surf/spherically_project.py`

Changed:

```python
evals, evecs = fem.eigs(k=4)
```

to:

```python
evals, evecs = fem.eigs(k=4, rng=0)
```

Isolated seeded projection repeated exactly:

- output files: `trt_outputs/scratch/rh.qsphere.seeded1`, `trt_outputs/scratch/rh.qsphere.seeded2`
- faces equal: yes
- differing coordinate values: `0`
- maximum absolute coordinate difference: `0`

### Force serial numeric libraries during projection

File: `recon_surf/spherically_project_wrapper.py`

The wrapper now sets the following to `1` before importing the projection implementation:

- `OMP_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `MKL_NUM_THREADS`
- `NUMEXPR_NUM_THREADS`
- `VECLIB_MAXIMUM_THREADS`

This isolates the deterministic projection behavior from global `--threads` values.

### Force serial topology correction

File: `recon_surf/recon-surf.sh`

The topology correction call now runs with serial OpenMP/ITK settings and passes serial FreeSurfer thread flags only for the `-fix` step:

```bash
env OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 recon-all ... -fix ... -threads 1 -itkthreads 1
```

Other later `recon-all` stages still use the regular `$fsthreads` setting.

## Post-Fix Verification

Two repeated patched `--surf_only --threads 8` runs were compared:

- `trt_outputs/surf_patched_threads8_run1/114823_MR1`
- `trt_outputs/surf_patched_threads8_run2/114823_MR1`

Result:

- no surface geometry data differences
- no surface face data differences
- no morphometry array differences
- no label data differences
- no annotation data differences
- no image voxel data differences
- remaining differences were metadata/provenance/log text only

The patched comparison found `35` remaining differences:

- `26` text differences
- `9` generic file differences

Those remaining differences are from paths, command files, timestamps, runtime logs, environment/provenance fields, and other metadata. The data arrays compared by the surface-aware comparator were identical.

Patched topology logs were substantively identical across repeated runs:

- left hemisphere:
  - `Total Loglikelihood : -25.3261`
  - `nv=102710, nf=205416, ne=308124`
- right hemisphere:
  - `Total Loglikelihood : -25.0604`
  - `nv=102505, nf=205006, ne=307509`

Numeric stats rows were identical after stripping volatile headers for:

- `stats/aseg.stats`
- `stats/rh.aparc.DKTatlas.mapped.stats`
- `stats/lh.aparc.DKTatlas.mapped.stats`
- `stats/wmparc.DKTatlas.mapped.stats`

## Thread Behavior

Before the surface fixes:

- repeated `--threads 8` surface runs diverged at `rh.qsphere.nofix`
- the divergence was tiny at first but changed the right-hemisphere topology correction result
- downstream right-hemisphere surfaces, labels, mapped volumes, and stats diverged

After the fixes:

- repeated `--threads 8` surface runs were deterministic at the data level for the tested subject
- projection is internally serialized and seeded
- topology correction is internally serialized
- later pipeline stages still receive the normal FastSurfer thread setting

The comparison against previous segmentation runs remains consistent with the earlier conclusion: segmentation repeated runs were already data-identical for the completed same-thread case, while surface reconstruction required fixes.

## Performance

Pre-fix repeated surface runs:

- run 1:
  - recon-surf internal runtime: `0.855` hours
  - Docker wall time: `51:20.06`
  - `mris_fix_topology lh`: `0.0143` hours, `4 threads`
  - `mris_fix_topology rh`: `0.1219` hours, `4 threads`
- run 2:
  - recon-surf internal runtime: `1.015` hours
  - Docker wall time: `1:00:58`
  - `mris_fix_topology lh`: `0.0318` hours, `4 threads`
  - `mris_fix_topology rh`: `0.1713` hours, `4 threads`

Post-fix repeated surface runs:

- run 1:
  - recon-surf internal runtime: `0.897` hours
  - Docker wall time: `53:53.77`
  - `mris_fix_topology lh`: `0.0201` hours, `1 thread`
  - `mris_fix_topology rh`: `0.0911` hours, `1 thread`
- run 2:
  - recon-surf internal runtime: `0.926` hours
  - Docker wall time: `55:37.65`
  - `mris_fix_topology lh`: `0.0221` hours, `1 thread`
  - `mris_fix_topology rh`: `0.0965` hours, `1 thread`

No clear performance regression was observed in these runs. The patched total runtime is within the variability of the two pre-fix runs. In this subject, serial topology correction was not slower than the observed pre-fix threaded right-hemisphere topology correction. This should still be treated as a small sample because host load and topology difficulty can affect runtime.

## Accuracy / Behavioral Impact

The fix changes the final surface result relative to either pre-fix run because the formerly random spectral projection is now anchored to a fixed start vector. For the right hemisphere, the deterministic topology after the fix is:

- `nv=102505, nf=205006, ne=307509`

This differs from both pre-fix repeated runs. That is expected: the old behavior was choosing between nearby numerical outcomes nondeterministically, and topology correction amplified those differences.

No external ground-truth accuracy evaluation was run. The verification here establishes deterministic repeatability and checks that repeated post-fix stats and image/surface data arrays are identical.

## Operational Notes

- Running Docker with `--user "$(id -u):$(id -g)"` works, but `USER` and `LOGNAME` should be set in the container environment for code paths using `getpass.getuser()`.
- The original license path was not readable inside the mapped-UID container, so a readable copy was mounted from `trt_outputs/fs_license.txt`.
- Some FastSurfer wrapper paths can return top-level success even when a nested recon-surf stage logs a failure. For determinism testing, the logs and expected output files are more reliable than Docker's top-level exit status alone.
- Direct absolute `-out /path/...` usage with `mris_fix_topology` was mishandled by FreeSurfer as a subject-relative surface name. Relative `-out` names inside the subject surface directory worked correctly.

## Conclusion

The surface nondeterminism for the tested subject was introduced in spectral spherical projection through an unseeded ARPACK start vector, then amplified by topology correction. Seeding LaPy's eigensolver, serializing the projection numeric libraries, and serializing the topology correction step made repeated `--surf_only --threads 8` runs deterministic at the data level.

Remaining differences after the fix are metadata/provenance differences only, not image, surface, morphometry, label, or annotation data differences.
