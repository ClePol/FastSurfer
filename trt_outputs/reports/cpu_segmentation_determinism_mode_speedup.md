# CPU Segmentation Deterministic-Mode Speedup

Date: 2026-05-14

Subject: `114823_MR1`

Reference output:

- `trt_outputs/seg_patched_all_threads8_run1c/114823_MR1`

Optimized output:

- `trt_outputs/seg_speed_cpu_algorithms_off_threads8_run1/114823_MR1`

Change:

- Keep CUDA deterministic backend settings.
- Do not force `torch.use_deterministic_algorithms(True)` when the selected inference device is CPU.

Rationale:

- CPU segmentation was already empirically stable in the previous determinism runs.
- Forcing PyTorch deterministic algorithms on CPU selected a much slower path for operations including `max_unpool2d`.
- CUDA still keeps strict deterministic algorithm selection.

Validation:

- Comparator: `trt_outputs/reports/compare_seg_reference_vs_cpu_algorithms_off_114823_MR1.json`
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

- Reference run wall time: `25:41.79`
- Optimized run wall time: `23:55`
- Speedup: about `1:47`

Component timings:

- VINN coronal inference dropped from about `188.7 s` to `153.7 s`.
- VINN axial inference was `157.6 s` in the optimized run.
- HypVINN CPU model inference also stopped emitting the deterministic `max_unpool2d` warning and stayed data-identical against the reference output.
