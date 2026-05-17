#!/usr/bin/env python3
"""Run mri_segment on a cropped brain volume."""

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


def _save(data: np.ndarray, source: nib.spatialimages.SpatialImage, path: Path, start: np.ndarray | None = None) -> None:
    affine = source.affine if start is None else _crop_affine(source.affine, start)
    image = nib.MGHImage(data, affine, source.header.copy())
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
    parser.add_argument("--brain", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--margin", type=int, default=8)
    args = parser.parse_args()

    img, data = _load_volume(args.brain)
    start, stop = _bounds(data != 0, args.margin)
    crop = tuple(slice(int(s), int(e)) for s, e in zip(start, stop))

    with tempfile.TemporaryDirectory(prefix="cropped-mri-segment-") as tmpdir:
        tmp = Path(tmpdir)
        cropped_brain = tmp / "brain.mgz"
        cropped_out = tmp / "wm.seg.mgz"
        _save(data[crop], img, cropped_brain, start)
        env = os.environ.copy()
        env.setdefault("USER", "fastsurfer")
        env.setdefault("LOGNAME", env["USER"])
        subprocess.run(
            ["mri_segment", "-wsizemm", "13", "-mprage", str(cropped_brain), str(cropped_out)],
            check=True,
            env=env,
        )
        _, segmented = _load_volume(cropped_out)

    out = np.zeros_like(data)
    out[crop] = segmented.astype(out.dtype, copy=False)
    _save(out, img, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
