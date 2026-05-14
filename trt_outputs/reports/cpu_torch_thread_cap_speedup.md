# CPU PyTorch Thread-Cap Speedup

Date: 2026-05-14

Subject: `114823_MR1`

Reference output:

- `trt_outputs/seg_patched_all_threads8_run1c/114823_MR1`

Optimized output:

- `trt_outputs/seg_speed_torchcap_threads8_run1/114823_MR1`

Change:

- Keep the pipeline-level `--threads 8` setting for N4, stats, and other non-PyTorch tools.
- Cap CPU PyTorch inference threads to the physical-core estimate (`os.cpu_count() // 2`) when the requested thread count is higher.
- Add `FASTSURFER_CPU_TORCH_THREADS` as an override for this cap.

Rationale:

- Running the whole pipeline at `--threads 6` was faster, but changed N4 outputs slightly because N4 also received 6 threads.
- Capping only PyTorch inference threads keeps the faster model-inference behavior while preserving the original 8-thread N4 behavior.

Validation:

- Comparator: `trt_outputs/reports/compare_seg_reference_vs_torch_thread_cap_114823_MR1.json`
- Image-array differences: none reported.
- Missing files: none.
- Remaining differences: path/provenance text files only.
- Numeric stats rows matched after stripping comment/header lines for:
  - `aseg+DKT.VINN.stats`
  - `aseg+DKT.VINN.withCC.stats`
  - `aseg.VINN.stats`
  - `cerebellum.CerebNet.stats`
  - `hypothalamus.HypVINN.stats`

Timing:

- Initial deterministic reference wall time: `25:41.79`
- Previous optimized CPU deterministic-mode wall time: `23:55.33`
- PyTorch thread-cap optimized wall time: `20:58.58`
- Incremental speedup over the previous optimized run: about `2:57`
- Speedup over the initial deterministic reference: about `4:43`

Component timings:

- FastSurferVINN coronal: `153.7142 s` -> `133.3467 s`
- FastSurferVINN sagittal: `161.6531 s` -> `132.6984 s`
- FastSurferVINN axial: `157.5654 s` -> `135.6204 s`
- HypVINN axial: `225.7978 s` -> `206.0720 s`
- HypVINN coronal: `222.3111 s` -> `207.5642 s`
- HypVINN sagittal: `224.5686 s` -> `205.2026 s`
