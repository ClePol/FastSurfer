#!/usr/bin/env python3
"""Run mri_edit_wm_with_aseg on a cropped working volume.

The FreeSurfer binary is exact but spends most of its time scanning the full
hires volume.  This wrapper crops to the nonzero support of the inputs plus a
margin, runs the binary unchanged, and pastes the result back into the original
grid.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from cropped_volume import bounds_from_mask, crop_slices, freesurfer_env, load_volume, save_volume


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm", required=True, type=Path)
    parser.add_argument("--brain", required=True, type=Path)
    parser.add_argument("--aseg", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--margin", type=int, default=0)
    args = parser.parse_args()

    wm_img, wm = load_volume(args.wm)
    brain_img, brain = load_volume(args.brain)
    aseg_img, aseg = load_volume(args.aseg)
    if wm.shape != brain.shape or wm.shape != aseg.shape:
        raise ValueError(f"Input shapes do not match: wm={wm.shape}, brain={brain.shape}, aseg={aseg.shape}")

    start, stop = bounds_from_mask((wm != 0) | (brain != 0) | (aseg != 0), args.margin)
    crop = crop_slices(start, stop)

    with tempfile.TemporaryDirectory(prefix="cropped-mri-edit-") as tmpdir:
        tmp = Path(tmpdir)
        crop_wm = tmp / "wm.mgz"
        crop_brain = tmp / "brain.mgz"
        crop_aseg = tmp / "aseg.mgz"
        crop_out = tmp / "wm.asegedit.mgz"

        save_volume(wm[crop], wm_img, crop_wm, start)
        save_volume(brain[crop], brain_img, crop_brain, start)
        save_volume(aseg[crop], aseg_img, crop_aseg, start)

        subprocess.run(
            [
                "mri_edit_wm_with_aseg",
                "-keep-in",
                str(crop_wm),
                str(crop_brain),
                str(crop_aseg),
                str(crop_out),
            ],
            check=True,
            env=freesurfer_env(),
        )

        _, edited_crop = load_volume(crop_out)

    out = wm.copy()
    out[crop] = edited_crop.astype(out.dtype, copy=False)
    save_volume(out, wm_img, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
