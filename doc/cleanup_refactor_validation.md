# Cleanup refactor validation

This note records the behavior-preserving cleanup work after the determinism and
speedup changes.

## Scope

The cleanup was split into incremental commits:

- `154a5d6 Share cropped volume wrapper helpers`
  - Added `recon_surf/cropped_volume.py`.
  - Shared MGH loading, crop bounds, cropped affine construction, save/paste
    helpers, and FreeSurfer `USER`/`LOGNAME` environment setup across cropped
    surface wrappers.
- `9e05feb Factor async command file headers`
  - Added `write_cmdf_header` in `recon_surf/recon-surf.sh`.
  - Reused it for ribbon, hypointensity relabeling, BA labels, mapped stats,
    pctsurfcon, aseg stats, and wmparc command files.
- `16c5fa4 Share CPU TorchScript inference helpers`
  - Added `FastSurferCNN/utils/torchscript.py`.
  - Shared the CPU TorchScript tracing/freezing helpers between FastSurferCNN
    and HypVINN inference.

The planned validation-helper move was skipped because the validation scripts
used for this work are in the sibling `trt_outputs` directory, not tracked under
the FastSurfer checkout.

## Validation

Static checks:

- `python3 -m py_compile recon_surf/cropped_volume.py recon_surf/cropped_ants_denoise.py recon_surf/cropped_mri_segment.py recon_surf/cropped_mri_edit_wm_with_aseg.py recon_surf/cropped_mris_volmask.py`
- `bash -n recon_surf/recon-surf.sh`
- `python3 -m py_compile FastSurferCNN/utils/torchscript.py FastSurferCNN/inference.py HypVINN/inference.py`

CPU segmentation validation on `114823_MR1` compared archived pre-cleanup commit
`027ab86` against the cleanup refactor:

- pre-cleanup: `cleanup_pre_refactor_seg_cpu_threads8_run1`, elapsed `6:11.11`
- cleanup: `cleanup_refactor_seg_cpu_threads8_run1`, elapsed `6:31.99`
- comparator report:
  `trt_outputs/reports/compare_seg_precleanup_vs_cleanup_refactor_114823_MR1.json`
- result: no image-data differences; stats body rows were identical after
  removing volatile comment headers.

Surface validation on `114823_MR1` compared archived pre-cleanup commit
`027ab86` against the cleanup refactor, starting both surface-only runs from the
same CPU segmentation output:

- pre-cleanup: `cleanup_pre_refactor_surface_threads8_run1`, elapsed `15:38.39`
  (`recon-surf-run-time-hours 0.261`)
- cleanup: `cleanup_refactor_surface_threads8_run1`, elapsed `15:48.96`
  (`recon-surf-run-time-hours 0.264`)
- comparator report:
  `trt_outputs/reports/compare_surface_precleanup_vs_cleanup_refactor_114823_MR1.json`
- result: no image, surface, morph, annotation, or label value differences.
  Remaining differences were limited to command files, logs, transform text,
  stats headers, and other provenance text. Stats body rows were identical after
  removing volatile comment headers.

GPU segmentation validation was not rerun for the cleanup refactor because the
available local validation image uses a PyTorch build that does not support the
current RTX 5070 Ti compute capability (`sm_120`).
