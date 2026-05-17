#!/usr/bin/env python3
"""Run mri_edit_wm_with_aseg on a cropped working volume.

The FreeSurfer binary is exact but spends most of its time scanning the full
hires volume.  This wrapper crops to the nonzero support of the inputs plus a
margin, runs the binary unchanged, and pastes the result back into the original
grid.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


def _load_volume(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    return img, np.asarray(data)


def _crop_affine(affine: np.ndarray, start: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = start
    return affine @ transform


def _save_like(data: np.ndarray, source: nib.spatialimages.SpatialImage, path: Path, start: np.ndarray) -> None:
    header = source.header.copy()
    image = nib.MGHImage(data, _crop_affine(source.affine, start), header)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def _bounds(mask: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        start = np.zeros(mask.ndim, dtype=int)
        stop = np.array(mask.shape, dtype=int)
    else:
        start = np.maximum(0, coords.min(axis=1) - margin)
        stop = np.minimum(mask.shape, coords.max(axis=1) + 1 + margin)
    return start.astype(int), stop.astype(int)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm", required=True, type=Path)
    parser.add_argument("--brain", required=True, type=Path)
    parser.add_argument("--aseg", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--margin", type=int, default=0)
    args = parser.parse_args()

    wm_img, wm = _load_volume(args.wm)
    brain_img, brain = _load_volume(args.brain)
    aseg_img, aseg = _load_volume(args.aseg)
    if wm.shape != brain.shape or wm.shape != aseg.shape:
        raise ValueError(f"Input shapes do not match: wm={wm.shape}, brain={brain.shape}, aseg={aseg.shape}")

    start, stop = _bounds((wm != 0) | (brain != 0) | (aseg != 0), args.margin)
    crop = tuple(slice(int(s), int(e)) for s, e in zip(start, stop))

    with tempfile.TemporaryDirectory(prefix="cropped-mri-edit-") as tmpdir:
        tmp = Path(tmpdir)
        crop_wm = tmp / "wm.mgz"
        crop_brain = tmp / "brain.mgz"
        crop_aseg = tmp / "aseg.mgz"
        crop_out = tmp / "wm.asegedit.mgz"

        _save_like(wm[crop], wm_img, crop_wm, start)
        _save_like(brain[crop], brain_img, crop_brain, start)
        _save_like(aseg[crop], aseg_img, crop_aseg, start)

        env = os.environ.copy()
        env.setdefault("USER", "fastsurfer")
        env.setdefault("LOGNAME", env["USER"])
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
            env=env,
        )

        _, edited_crop = _load_volume(crop_out)

    out = wm.copy()
    out[crop] = edited_crop.astype(out.dtype, copy=False)
    out_img = nib.MGHImage(out, wm_img.affine, wm_img.header.copy())
    out_img.set_data_dtype(out.dtype)
    nib.save(out_img, str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
