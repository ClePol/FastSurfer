#!/usr/bin/env python3
"""Fast approximate replacement for FreeSurfer mri_fill with -segmentation.

This implements the FastSurfer surface-recon use case of mri_fill from wm.mgz
and aseg.presurf.mgz.  The hemisphere split uses a C-backed nearest-label
transform rather than FreeSurfer's exact iterative Voronoi wavefront.  In
testing this changed a very small number of filled voxels and produced small
surface differences in the midsagittal area.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


LH_FILL = 255
RH_FILL = 127
WM_MIN_VAL = 5
WM_EDITED_OFF_VAL = 1
WM_EDITED_ON_VAL = 255

BRAIN_STEM = 16
LEFT_CEREBELLUM_WHITE_MATTER = 7
LEFT_CEREBELLUM_CORTEX = 8
RIGHT_CEREBELLUM_WHITE_MATTER = 46
RIGHT_CEREBELLUM_CORTEX = 47
LEFT_CEREBRAL_WHITE_MATTER = 2
LEFT_CEREBRAL_CORTEX = 3
RIGHT_CEREBRAL_WHITE_MATTER = 41
RIGHT_CEREBRAL_CORTEX = 42
LEFT_LESION = 25
RIGHT_LESION = 57
WMSA_LABELS = {77, 78, 79, 87, 88, 89}

LEFT_LABELS = {
    2, 3, 4, 5, 10, 11, 12, 13, 17, 18, 19, 20, 25, 30, 26, 28, 31,
    32, 33, 34, 35, 36, 37, 38, 39,
}
RIGHT_LABELS = {
    41, 42, 43, 44, 49, 50, 51, 52, 53, 54, 55, 56, 57, 62, 58, 60, 63,
    64, 65, 66, 67, 68, 69, 70, 71,
}
ERASE_LABELS = {
    BRAIN_STEM,
    LEFT_CEREBELLUM_WHITE_MATTER,
    LEFT_CEREBELLUM_CORTEX,
    RIGHT_CEREBELLUM_WHITE_MATTER,
    RIGHT_CEREBELLUM_CORTEX,
}


def _load_int(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    return img, data.astype(np.int16, copy=False)


def _nearest_hemi_labels(seed_values: np.ndarray, control: np.ndarray) -> np.ndarray:
    _, indices = ndimage.distance_transform_edt(control == 0, return_indices=True)
    return seed_values[tuple(indices)].astype(np.uint8, copy=False)


def _largest_cc18(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    structure[0, 0, 0] = 0
    structure[0, 0, 2] = 0
    structure[0, 2, 0] = 0
    structure[0, 2, 2] = 0
    structure[2, 0, 0] = 0
    structure[2, 0, 2] = 0
    structure[2, 2, 0] = 0
    structure[2, 2, 2] = 0
    labeled, nlabels = ndimage.label(mask, structure=structure)
    if nlabels == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == int(np.argmax(counts))


def _remove_holes(mask: np.ndarray) -> np.ndarray:
    background = ~mask
    seed = np.zeros(mask.shape, dtype=bool)
    seed[:, 0, 0] = background[:, 0, 0]
    outside = ndimage.binary_propagation(
        seed,
        structure=ndimage.generate_binary_structure(3, 1),
        mask=background,
    )
    return mask | ~outside


def fill_with_aseg(wm: np.ndarray, aseg: np.ndarray) -> np.ndarray:
    wm = wm.copy()
    wm[np.isin(aseg, list(ERASE_LABELS))] = 0

    fill = np.zeros(wm.shape, dtype=np.uint8)
    control = np.zeros(wm.shape, dtype=np.uint8)
    left = np.isin(aseg, list(LEFT_LABELS))
    right = np.isin(aseg, list(RIGHT_LABELS))
    control[left | right] = 1
    fill[left] = LH_FILL
    fill[right] = RH_FILL
    fill = _nearest_hemi_labels(fill, control)

    edited_off = wm == WM_EDITED_OFF_VAL
    edited_on = wm == WM_EDITED_ON_VAL
    low = wm < WM_MIN_VAL
    keep_low = (
        (aseg == LEFT_LESION)
        | (aseg == RIGHT_LESION)
        | np.isin(aseg, list(WMSA_LABELS))
    )

    fill[edited_off] = 0
    if np.any(edited_on & ~edited_off):
        raise RuntimeError("WM_EDITED_ON_VAL handling is not implemented")
    fill[low & ~keep_low & ~edited_off] = 0

    lh = (fill != RH_FILL) & (fill != 0)
    rh = fill == RH_FILL
    lh = _remove_holes(_largest_cc18(lh))
    rh = _remove_holes(_largest_cc18(rh))

    out = np.zeros(wm.shape, dtype=np.uint8)
    out[lh] = LH_FILL
    out[rh & ~lh] = RH_FILL
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm", required=True, type=Path)
    parser.add_argument("--aseg", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    wm_img, wm = _load_int(args.wm)
    _, aseg = _load_int(args.aseg)
    out = fill_with_aseg(wm, aseg)
    out_img = nib.MGHImage(out, wm_img.affine, wm_img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
