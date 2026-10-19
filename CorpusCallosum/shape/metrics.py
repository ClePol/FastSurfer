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

import FastSurferCNN.utils.logging as logging

logger = logging.get_logger(__name__)


# TODO: we could make this more robust by smoothing the contour (the contour is already rotated into the ACPC frame,
#  which is aligned to the anterior and posterior CC endpoints, by the caller)

def _line_segment_intersection(
    line_point: np.ndarray,
    line_dir: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray | None:
    """Compute intersection between an infinite line and a line segment.

    Uses the parametric form:
    - Line: P = line_point + t * line_dir
    - Segment: Q = seg_start + s * (seg_end - seg_start), where s ∈ [0, 1]

    Parameters
    ----------
    line_point : np.ndarray
        A point on the infinite line, shape (2,).
    line_dir : np.ndarray
        Direction vector of the line, shape (2,).
    seg_start : np.ndarray
        Start point of the segment, shape (2,).
    seg_end : np.ndarray
        End point of the segment, shape (2,).
    tol : float
        Tolerance for numerical comparisons.

    Returns
    -------
    np.ndarray | None
        Intersection point as shape (2,) array, or None if no intersection.
    """
    seg_dir = seg_end - seg_start
    
    # Build the linear system: [line_dir, -seg_dir] @ [t, s].T = seg_start - line_point
    # Matrix A = [[line_dir[0], -seg_dir[0]], [line_dir[1], -seg_dir[1]]]
    A = np.array([[line_dir[0], -seg_dir[0]], 
                  [line_dir[1], -seg_dir[1]]])
    b = seg_start - line_point

    # Check if lines are parallel (determinant ≈ 0)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < tol:
        return None

    # Solve for t and s using Cramer's rule (faster than linalg.solve for 2x2)
    t = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
    s = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det

    # Check if intersection is within the segment [0, 1]
    if -tol <= s <= 1.0 + tol:
        return line_point + t * line_dir
    return None


def get_intersections(
    contour: np.ndarray, start_point: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Find intersection points between an infinite line and a closed contour.

    Parameters
    ----------
    contour : np.ndarray
        Array of shape (2, N) containing contour points in ACPC space.
    start_point : np.ndarray
        A point on the line, shape (2,).
    direction : np.ndarray
        Direction vector of the line, shape (2,).

    Returns
    -------
    np.ndarray
        Array of shape (M, 2) containing intersection points, sorted along the direction.
    """
    start_point = np.asarray(start_point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    
    # Normalize direction
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-10:
        return np.empty((0, 2))
    direction = direction / dir_norm

    n_points = contour.shape[1]
    intersections = []

    # Check intersection with each segment of the closed contour
    for i in range(n_points):
        seg_start = contour[:, i]
        seg_end = contour[:, (i + 1) % n_points]  # Wrap around to close the contour
        
        intersection = _line_segment_intersection(
            start_point, direction, seg_start, seg_end
        )
        if intersection is not None:
            intersections.append(intersection)

    if not intersections:
        return np.empty((0, 2))

    points = np.array(intersections)
    
    # Remove duplicate points (can occur at contour vertices)
    if len(points) > 1:
        # Project onto line direction and find unique points
        projections = np.dot(points - start_point, direction)
        # Sort and remove duplicates within tolerance
        sorted_idx = np.argsort(projections)
        points = points[sorted_idx]
        projections = projections[sorted_idx]
        
        # Keep points that are sufficiently far apart
        mask = np.ones(len(points), dtype=bool)
        for i in range(1, len(points)):
            if abs(projections[i] - projections[i - 1]) < 1e-8:
                mask[i] = False
        points = points[mask]

    return points


def calculate_cc_index(cc_contour: np.ndarray) -> float:
    """Calculate the dimensionless corpus callosum index (CCI).

    The AP line intersects the contour 4 times. The measurements are:
    - Anterior thickness: distance between intersection points 1 and 2
    - Posterior thickness: distance between intersection points 3 and 4
    - Middle thickness: perpendicular line through midpoint of AP line

    The index is ``(anterior + middle + posterior) / AP_length``, following
    Figueira et al. (2007), https://doi.org/10.1590/S0004-282X2007000600001.

    Parameters
    ----------
    cc_contour : np.ndarray
        Array of shape (2, N) containing contour points in ACPC space.

    Returns
    -------
    cc_index : float
        Sum of the anterior, middle, and posterior widths divided by the
        anterior-posterior length.
    """
    # Get anterior and posterior points (extremes along x-axis)
    # In ACPC space, X is Anterior-Posterior direction, where Anterior is positive
    posterior_idx = np.argmin(cc_contour[0])  # Minimum X is Posterior
    anterior_idx = np.argmax(cc_contour[0])   # Maximum X is Anterior

    anterior_pt = cc_contour[:, anterior_idx]
    posterior_pt = cc_contour[:, posterior_idx]

    # AP line vector from anterior to posterior
    ap_vector = posterior_pt - anterior_pt
    ap_length = np.linalg.norm(ap_vector)
    ap_unit = ap_vector / ap_length

    # Perpendicular direction (90 degrees rotated)
    perp_unit = np.array([-ap_unit[1], ap_unit[0]])

    # Find where AP line intersects the contour (should be 4 points)
    ap_intersections = get_intersections(
        contour=cc_contour, start_point=anterior_pt, direction=ap_unit
    )

    if len(ap_intersections) != 4:
        logger.error(
            f"AP line should intersect contour exactly 4 times, "
            f"but found {len(ap_intersections)} intersections"
        )
        return 0.0

    # Measurement 1: anterior thickness (between intersection points 1 and 2)
    anterior_thickness = np.linalg.norm(ap_intersections[0] - ap_intersections[1])

    # Measurement 2: posterior thickness (between intersection points 3 and 4)
    posterior_thickness = np.linalg.norm(ap_intersections[2] - ap_intersections[3])

    # AP distance is between outermost intersection points (1 and 4)
    ap_distance = np.linalg.norm(ap_intersections[0] - ap_intersections[3])

    # Midpoint of AP line (between points 1 and 4, or between anterior and posterior extremes)
    midpoint = (ap_intersections[0] + ap_intersections[3]) / 2

    # Measurement 3: perpendicular line through midpoint
    middle_intersections = get_intersections(
        contour=cc_contour, start_point=midpoint, direction=perp_unit
    )

    middle_thickness = np.linalg.norm(middle_intersections[0] - middle_intersections[-1])

    # Calculate CC index
    cc_index = (anterior_thickness + posterior_thickness + middle_thickness) / ap_distance

    return cc_index

