#!/usr/bin/env python3
"""Run mri_segment on a cropped brain volume."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from cropped_volume import bounds_from_mask, crop_slices, freesurfer_env, load_volume, save_volume


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--margin", type=int, default=8)
    args = parser.parse_args()

    img, data = load_volume(args.brain)
    start, stop = bounds_from_mask(data != 0, args.margin)
    crop = crop_slices(start, stop)

    with tempfile.TemporaryDirectory(prefix="cropped-mri-segment-") as tmpdir:
        tmp = Path(tmpdir)
        cropped_brain = tmp / "brain.mgz"
        cropped_out = tmp / "wm.seg.mgz"
        save_volume(data[crop], img, cropped_brain, start)
        subprocess.run(
            ["mri_segment", "-wsizemm", "13", "-mprage", str(cropped_brain), str(cropped_out)],
            check=True,
            env=freesurfer_env(),
        )
        _, segmented = load_volume(cropped_out)

    out = np.zeros_like(data)
    out[crop] = segmented.astype(out.dtype, copy=False)
    save_volume(out, img, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
