# Copyright 2025 DeepMI Lab, German Center for Neurodegenerative Diseases(DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from CorpusCallosum.utils.types import Points2dType


def compute_curvature(path: Points2dType) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Compute curvature by computing edge angles.

    Parameters
    ----------
    path : np.ndarray
        Array of shape (N, 2) containing path coordinates.

    Returns
    -------
    np.ndarray
        Array of angle differences between consecutive edges.
    """
    # compute curvature by computing edge angles
    edges = np.diff(path, axis=0)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    # compute angle differences between consecutive edges
    angle_diffs = np.diff(angles)
    # wrap angles to [-pi, pi]
    angle_diffs = np.mod(angle_diffs + np.pi, 2 * np.pi) - np.pi
    return angle_diffs


def compute_mean_curvature(path: Points2dType) -> float:
    """Compute the mean absolute turning angle of a path in degrees per sample step.

    Parameters
    ----------
    path : np.ndarray
        Array of shape (N, 2) containing path coordinates.

    Returns
    -------
    float
        Mean absolute turning angle between consecutive edges of the path in degrees (per sample step).

    Notes
    -----
    This is not curvature in deg/mm: the value is averaged per sample step and therefore depends on the sampling
    density of `path`. For the CC midline, which is sampled equidistantly with `num_thickness_points` points, values
    are only comparable between analyses that use the same `num_thickness_points` (default 100), but they are
    invariant to the size (length) of the CC.
    """
    curvature = compute_curvature(path)
    if len(curvature) == 0:
        return 0.0
    return np.mean(np.abs(np.degrees(curvature))).item()


def _find_split_indices(
    midline: Points2dType,
    split_points: np.ndarray,
    subdivision_lines: list[Points2dType] | None = None,
) -> list[int]:
    """Find the indices of the midline points closest to where the midline is split.

    Parameters
    ----------
    midline : Points2dType
        Array of shape (N, 2) of equidistant points along the midline.
    split_points : np.ndarray
        Array of shape (K, 2) of split points (used if `subdivision_lines` is None).
    subdivision_lines : list of np.ndarray, optional
        K subdivision lines, each of shape (2, 2) (the two intersections of the line with the contour).
        If passed, the midline is split where it intersects each subdivision line (split points that already lie
        on the midline are used as is).

    Returns
    -------
    list[int]
        Sorted midline indices, starting with 0 and ending with N-1.
    """
    split_indices_midline = [0]
    targets = list(np.asarray(split_points, dtype=float).reshape(-1, 2))
    if subdivision_lines is not None:
        if len(subdivision_lines) != len(targets):
            raise ValueError(f"Got {len(targets)} split points, but {len(subdivision_lines)} subdivision lines.")
        edge_start, edge_vec = midline[:-1], np.diff(midline, axis=0)
        edge_len_sq = np.maximum(np.sum(edge_vec**2, axis=1), np.finfo(float).tiny)
        for k, line in enumerate(subdivision_lines):
            # split points that already lie on the midline (e.g. 'shape' subdivision) are used as is
            proj = np.clip(np.sum((targets[k] - edge_start) * edge_vec, axis=1) / edge_len_sq, 0, 1)
            if np.min(np.linalg.norm(edge_start + proj[:, None] * edge_vec - targets[k], axis=1)) < 1e-6:
                continue
            p0, p1 = np.asarray(line, dtype=float)
            line_vec = p1 - p0
            line_center = (p0 + p1) / 2
            # solve edge_start + t * edge_vec == p0 + s * line_vec for all midline edges
            denom = edge_vec[:, 0] * line_vec[1] - edge_vec[:, 1] * line_vec[0]
            diff = p0[None] - edge_start
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (diff[:, 0] * line_vec[1] - diff[:, 1] * line_vec[0]) / denom
                s = (diff[:, 0] * edge_vec[:, 1] - diff[:, 1] * edge_vec[:, 0]) / denom
            valid = (denom != 0) & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)
            if np.any(valid):
                intersections = edge_start[valid] + t[valid, None] * edge_vec[valid]
                # if the midline crosses the subdivision line more than once, use the crossing closest to the center
                targets[k] = intersections[np.argmin(np.linalg.norm(intersections - line_center, axis=1))]
            # else: no crossing (should not happen for a valid subdivision), fall back to the split point
    for target in targets:
        split_indices_midline.append(int(np.argmin(np.linalg.norm(midline - target, axis=1))))
    split_indices_midline.append(len(midline) - 1)
    return sorted(split_indices_midline)


def calculate_curvature_metrics(
    midline: Points2dType,
    split_points: np.ndarray,
    subdivision_lines: list[Points2dType] | None = None,
) -> tuple[float, float, np.ndarray]:
    """
    Calculate curvature metrics for the CC midline, including overall mean,
    body (central 65%), and subsegment curvatures.

    Parameters
    ----------
    midline : Points2dType
        Equidistant points along the midline.
    split_points : np.ndarray
        Array of shape (K, 2) of points where the midline is split. For the 'shape' subdivision these lie on the
        midline; for the other subdivision methods they are the centers of the subdivision lines (inside the CC).
        The midline is split at the midline points closest to these points.
    subdivision_lines : list of np.ndarray, optional
        Subdivision lines, each of shape (2, 2) in the same space as `midline`. If passed, the midline is split
        where it intersects the subdivision lines instead of at the closest point to `split_points` (exact for all
        subdivision methods).

    Returns
    -------
    mean_curvature : float
        Overall mean absolute turning angle of the midline in degrees per sample step, see `compute_mean_curvature`.
    curvature_body : float
        Mean absolute turning angle of the central 65% of the midline in degrees per sample step.
    curvature_subsegments : np.ndarray
        Mean absolute turning angle for each subsegment in degrees per sample step.

    Notes
    -----
    All values depend on the number of midline points (`num_thickness_points`), see `compute_mean_curvature`.
    """
    mean_curvature = compute_mean_curvature(midline)

    num_midline_points = len(midline)
    # central 65% means we remove 17.5% from each end
    start_idx_body = int(num_midline_points * 0.175)
    end_idx_body = int(num_midline_points * 0.825)
    curvature_body = compute_mean_curvature(midline[start_idx_body:end_idx_body])

    # Find split indices on the midline for subsegment curvature
    split_indices_midline = _find_split_indices(midline, split_points, subdivision_lines)

    _curvature_subsegments = []
    for i in range(len(split_indices_midline) - 1):
        s_idx = split_indices_midline[i]
        e_idx = split_indices_midline[i + 1]
        if e_idx - s_idx >= 2:  # need at least 3 points for curvature
            curv = compute_mean_curvature(midline[s_idx : e_idx + 1])
        else:
            curv = 0.0
        _curvature_subsegments.append(curv)
    curvature_subsegments = np.asarray(_curvature_subsegments)

    return mean_curvature, curvature_body, curvature_subsegments
