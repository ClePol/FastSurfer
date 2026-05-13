#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


IMAGE_SUFFIXES = (".mgz", ".mgh", ".nii", ".nii.gz")
VOLATILE_PARTS = {"/scripts/"}


def is_image(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in IMAGE_SUFFIXES)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel_files(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def compare_images(left: Path, right: Path) -> dict[str, Any]:
    li = nib.load(str(left))
    ri = nib.load(str(right))
    la = np.asanyarray(li.dataobj)
    ra = np.asanyarray(ri.dataobj)
    same_shape = la.shape == ra.shape
    same_dtype = la.dtype == ra.dtype
    result: dict[str, object] = {
        "kind": "image",
        "same_shape": same_shape,
        "left_shape": la.shape,
        "right_shape": ra.shape,
        "same_dtype": same_dtype,
        "left_dtype": str(la.dtype),
        "right_dtype": str(ra.dtype),
    }
    if same_shape:
        diff = la != ra
        result["equal_voxels"] = bool(not np.any(diff))
        result["different_voxels"] = int(np.count_nonzero(diff))
        if np.issubdtype(la.dtype, np.number) and np.issubdtype(ra.dtype, np.number):
            delta = np.asarray(la, dtype=np.float64) - np.asarray(ra, dtype=np.float64)
            result["max_abs_diff"] = float(np.max(np.abs(delta))) if delta.size else 0.0
    return result


def compare_regular_files(left: Path, right: Path) -> dict[str, object]:
    return {
        "kind": "file",
        "same_sha256": digest(left) == digest(right),
        "left_size": left.stat().st_size,
        "right_size": right.stat().st_size,
    }


def compare_files(left: Path, right: Path) -> dict[str, object]:
    if is_image(left):
        try:
            return compare_images(left, right)
        except Exception as exc:
            result = compare_regular_files(left, right)
            result["image_load_error"] = type(exc).__name__
            return result
    return compare_regular_files(left, right)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    left_root = args.left.resolve()
    right_root = args.right.resolve()
    left_files = rel_files(left_root)
    right_files = rel_files(right_root)
    common = sorted(left_files & right_files)

    results = {
        "left": str(left_root),
        "right": str(right_root),
        "missing_left": sorted(str(p) for p in right_files - left_files),
        "missing_right": sorted(str(p) for p in left_files - right_files),
        "differences": {},
    }

    for rel in common:
        rel_s = "/" + rel.as_posix()
        comparison = compare_files(left_root / rel, right_root / rel)
        if comparison.get("kind") == "image":
            is_same = comparison.get("equal_voxels") is True
        else:
            is_same = comparison.get("same_sha256") is True
        if not is_same and not any(part in rel_s for part in VOLATILE_PARTS):
            results["differences"][rel.as_posix()] = comparison

    text = json.dumps(results, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
