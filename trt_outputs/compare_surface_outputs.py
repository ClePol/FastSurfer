#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.freesurfer import io as fsio


IMAGE_SUFFIXES = (".mgz", ".mgh", ".nii", ".nii.gz")
SURFACE_NAMES = {
    "inflated",
    "inflated.nofix",
    "orig",
    "orig.nofix",
    "orig.nofix.predec",
    "orig.premesh",
    "orig.premesh.noorient",
    "pial",
    "pial.T1",
    "qsphere.nofix",
    "smoothwm",
    "smoothwm.nofix",
    "sphere",
    "sphere.reg",
    "white",
    "white.preaparc",
}
MORPH_SUFFIXES = (
    ".area",
    ".area.mid",
    ".area.pial",
    ".avg_curv",
    ".curv",
    ".curv.pial",
    ".defect_borders",
    ".defect_chull",
    ".defect_labels",
    ".jacobian_white",
    ".sulc",
    ".thickness",
    ".volume",
)
VOLATILE_TEXT_PATTERNS = (
    re.compile(r"^(# )?CreationTime "),
    re.compile(r"^(# )?cmdline "),
    re.compile(r"^(# )?cwd "),
    re.compile(r"^(# )?hostname "),
    re.compile(r"^(# )?machine "),
    re.compile(r"^(# )?sysname "),
    re.compile(r"^(# )?user "),
    re.compile(r"^(# )?Date "),
    re.compile(r"^(# )?Started at "),
    re.compile(r"^(# )?Ended   at "),
    re.compile(r"^#@#"),
    re.compile(r"^@#@FS"),
    re.compile(r"^Log file for "),
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel_files(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file() and not p.is_symlink()}


def is_image(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in IMAGE_SUFFIXES)


def is_surface(path: Path) -> bool:
    return path.parent.name == "surf" and path.name.split(".", 1)[-1] in SURFACE_NAMES


def is_morph(path: Path) -> bool:
    return path.parent.name == "surf" and any(path.name.endswith(suffix) for suffix in MORPH_SUFFIXES)


def compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "same_shape": left.shape == right.shape,
        "left_shape": left.shape,
        "right_shape": right.shape,
        "same_dtype": left.dtype == right.dtype,
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
    }
    if left.shape != right.shape:
        return result
    diff = left != right
    result["equal_values"] = bool(not np.any(diff))
    result["different_values"] = int(np.count_nonzero(diff))
    if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
        result["max_abs_diff"] = float(np.max(np.abs(delta))) if delta.size else 0.0
        result["mean_abs_diff"] = float(np.mean(np.abs(delta))) if delta.size else 0.0
    return result


def compare_image(left: Path, right: Path) -> dict[str, Any]:
    li = nib.load(str(left))
    ri = nib.load(str(right))
    result = compare_arrays(np.asanyarray(li.dataobj), np.asanyarray(ri.dataobj))
    result["kind"] = "image"
    return result


def compare_surface(left: Path, right: Path) -> dict[str, Any]:
    lcoords, lfaces = fsio.read_geometry(str(left), read_metadata=False)
    rcoords, rfaces = fsio.read_geometry(str(right), read_metadata=False)
    return {
        "kind": "surface",
        "coords": compare_arrays(lcoords, rcoords),
        "faces": compare_arrays(lfaces, rfaces),
    }


def compare_morph(left: Path, right: Path) -> dict[str, Any]:
    return {
        "kind": "morph",
        "values": compare_arrays(fsio.read_morph_data(str(left)), fsio.read_morph_data(str(right))),
    }


def compare_annot(left: Path, right: Path) -> dict[str, Any]:
    llabels, lctab, lnames = fsio.read_annot(str(left))
    rlabels, rctab, rnames = fsio.read_annot(str(right))
    return {
        "kind": "annot",
        "labels": compare_arrays(llabels, rlabels),
        "ctab": compare_arrays(lctab, rctab),
        "same_names": lnames == rnames,
    }


def compare_label(left: Path, right: Path) -> dict[str, Any]:
    return {
        "kind": "label",
        "vertices": compare_arrays(fsio.read_label(str(left)), fsio.read_label(str(right))),
    }


def normalized_text(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        if any(pattern.search(raw) for pattern in VOLATILE_TEXT_PATTERNS):
            continue
        lines.append(raw)
    return lines


def compare_text(left: Path, right: Path) -> dict[str, Any]:
    ltext = normalized_text(left)
    rtext = normalized_text(right)
    return {
        "kind": "text",
        "same_normalized_text": ltext == rtext,
        "left_lines": len(ltext),
        "right_lines": len(rtext),
    }


def compare_regular(left: Path, right: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "file",
        "same_sha256": digest(left) == digest(right),
        "left_size": left.stat().st_size,
        "right_size": right.stat().st_size,
    }
    if left.suffix in {".log", ".txt", ".stats", ".dat", ".lta", ".xfm", ".yaml"}:
        result.update(compare_text(left, right))
    return result


def compare_files(left: Path, right: Path) -> dict[str, Any]:
    if is_image(left):
        return compare_image(left, right)
    if is_surface(left):
        return compare_surface(left, right)
    if is_morph(left):
        return compare_morph(left, right)
    if left.suffix == ".annot":
        return compare_annot(left, right)
    if left.suffix == ".label":
        return compare_label(left, right)
    return compare_regular(left, right)


def same_result(result: dict[str, Any]) -> bool:
    kind = result["kind"]
    if kind == "image":
        return result.get("equal_values") is True
    if kind == "surface":
        return result["coords"].get("equal_values") is True and result["faces"].get("equal_values") is True
    if kind == "morph":
        return result["values"].get("equal_values") is True
    if kind == "annot":
        return (
            result["labels"].get("equal_values") is True
            and result["ctab"].get("equal_values") is True
            and result["same_names"] is True
        )
    if kind == "label":
        return result["vertices"].get("equal_values") is True
    if kind == "text":
        return result["same_normalized_text"] is True
    return result.get("same_sha256") is True


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

    results: dict[str, Any] = {
        "left": str(left_root),
        "right": str(right_root),
        "missing_left": sorted(str(p) for p in right_files - left_files),
        "missing_right": sorted(str(p) for p in left_files - right_files),
        "differences": {},
        "errors": {},
    }

    for rel in sorted(left_files & right_files):
        try:
            comparison = compare_files(left_root / rel, right_root / rel)
        except Exception as exc:
            results["errors"][rel.as_posix()] = type(exc).__name__ + ": " + str(exc)
            continue
        if not same_result(comparison):
            results["differences"][rel.as_posix()] = comparison

    text = json.dumps(results, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
