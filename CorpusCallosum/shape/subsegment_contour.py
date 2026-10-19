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

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.spatial import ConvexHull

from CorpusCallosum.utils.types import ContourList, Points2dType, Polygon2dType, Polygon3dType
from FastSurferCNN.utils import ScalarType, Vector2d


def minimum_bounding_rectangle(points: Points2dType) -> np.ndarray[tuple[Literal[4], Literal[2]], np.dtype[ScalarType]]:
    """Find the smallest bounding rectangle for a set of points.

    Parameters
    ----------
    points : array
        An array of shape (N, 2) containing point coordinates.

    Returns
    -------
    np.ndarray
        Array of shape (4, 2) containing coordinates of the bounding box corners.
    """
    pi2 = np.pi / 2.0
    points = np.asarray(points).T

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    # including the edge that closes the loop from last to first point
    edges = np.vstack([hull_points[1:] - hull_points[:-1], hull_points[0] - hull_points[-1]])

    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2), np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def calc_subsegment_areas(split_contours: ContourList) -> np.ndarray[tuple[int], np.dtype[ScalarType]]:
    """Calculate area of each subsegment using the shoelace formula.

    Parameters
    ----------
    split_contours : list of np.ndarray
        List of contour arrays, each of shape (2, N). The list should contain
        a set of nested contours (cumulative subsegments) and the full contour.

    Returns
    -------
    subsegment_areas : array of floats
        Array containing the area of each incremental subsegment.
    """
    # calculate area of each split contour using the shoelace formula
    # we use the absolute value because the orientation of the contour may vary
    areas_cum = []
    for c in split_contours:
        # c is (2, N)
        x = c[0]
        y = c[1]
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        areas_cum.append(area)
    areas_cum = np.asarray(areas_cum)

    if len(areas_cum) == 1:
        return np.asarray(areas_cum[0])
    
    # Sort areas to ensure they are in increasing order of size
    # This handles both cases where subsegments were provided in increasing or decreasing order
    # The set of areas represents a sequence of nested shapes.
    sorted_areas = np.sort(areas_cum)
    
    # Calculate the incremental pieces by taking differences between consecutive sizes
    return np.diff(sorted_areas, prepend=0)


def subsegment_midline_orthogonal(
        midline: Points2dType,
        area_weights: np.ndarray[tuple[int], np.dtype[np.float64]],
        contour: Polygon2dType,
) -> tuple[np.ndarray[tuple[int], np.dtype[ScalarType]], ContourList, np.ndarray, list[Points2dType]]:
    """Subsegment contour orthogonally to the midline based on area weights.

    Parameters
    ----------
    midline : array of floats
        Array of shape (N, 2) containing midline points.
    area_weights : array of floats
        Array of weights for area-based subdivision.
    contour : array of floats
        Array of shape (2, M) containing contour points in as space.

    Returns
    -------
    subsegment_areas : array of floats
        List of subsegment areas.
    split_contours : list of np.ndarray
        List of contour arrays for each subsegment.
    split_points : np.ndarray
        Array of shape (K, 2) containing points where the midline was split.
    subdivision_lines : list of np.ndarray
        List of contour arrays for each subdivision line.

    Notes
    -----
    Subsegments include all previous segments. This means subsegment contour two is the outline of the union
    of subsegment one and subsegment two.
    """
    # midline[0] is Anterior (max X), midline[-1] is Posterior (min X)
    midline_start_idx = np.argmin(np.linalg.norm(contour.T - midline[0], axis=1))
    # roll contour start to midline anterior end
    contour = np.roll(contour, -midline_start_idx, axis=1)

    # Calculate edge indices and fractions for splitting the midline
    # We use len(midline) - 1 because we are looking for intervals between points
    edge_idx_float = (len(midline) - 1) * np.array(area_weights)
    edge_idx, edge_frac = np.divmod(edge_idx_float, 1)
    edge_idx = edge_idx.astype(int)

    # Handle cases where area_weights might reach 1.0, which would lead to an out-of-bounds access
    at_end = edge_idx >= len(midline) - 1
    edge_idx[at_end] = len(midline) - 2
    edge_frac[at_end] = 1.0

    split_points = midline[edge_idx] + (midline[edge_idx + 1] - midline[edge_idx]) * edge_frac[:, None]

    # get edge for each split point
    edge_directions = midline[edge_idx] - midline[edge_idx + 1]
    # get vector perpendicular to each midline edge
    edge_ortho_vectors = np.column_stack((-edge_directions[:, 1], edge_directions[:, 0]))
    edge_ortho_vectors = edge_ortho_vectors / np.linalg.norm(edge_ortho_vectors, axis=1)[:, None]

    # Calculate intersections between the perpendicular lines and the contour
    # vectors from split points to all contour points
    vectors = contour.T[None, :, :] - split_points[:, None, :]  # (K, M, 2)

    # Calculate cross product with ortho vectors to find side of the line (numerator of t)
    # x*oy - y*ox
    side = vectors[:, :, 0] * edge_ortho_vectors[:, None, 1] - vectors[:, :, 1] * edge_ortho_vectors[:, None, 0]

    # Find where the side changes sign, indicating an intersection
    # Handle wrap-around by appending the first side value to the end
    side_wrapped = np.hstack([side, side[:, 0:1]])
    sign_change = (side_wrapped[:, :-1] * side_wrapped[:, 1:]) <= 0

    split_contours: ContourList = []
    subdivision_lines: list[Points2dType] = []

    for pt_idx, split_point in enumerate(split_points):
        # Indices of contour segments that have sign changes for this split point
        seg_indices = np.where(sign_change[pt_idx])[0]

        intersections = []
        num_points = contour.shape[1]
        for i in seg_indices:
            s0 = side[pt_idx, i]
            s1 = side[pt_idx, (i + 1) % num_points]
            if s0 == s1:
                t = 0.5
            else:
                t = s0 / (s0 - s1)

            # intersection point on the segment
            p0 = contour[:, i]
            p1 = contour[:, (i + 1) % num_points]
            intersection_point = p0 + t * (p1 - p0)
            intersections.append((i, intersection_point))


        # get the two intersections closest to split_point
        intersections.sort(key=lambda x: np.linalg.norm(x[1] - split_point))

        # Create new contours by splitting at intersections
        if len(intersections) >= 2:
            first_index, first_intersection = intersections[1]
            second_index, second_intersection = intersections[0]

            if first_index > second_index:
                first_index, second_index = second_index, first_index
                first_intersection, second_intersection = second_intersection, first_intersection

            subdivision_lines.append(np.stack([first_intersection, second_intersection], axis=0))
            first_index += 1

            # connect first and second half
            start_to_cutoff = np.hstack(
                (
                    contour[:, :first_index],
                    first_intersection[:, None],
                    second_intersection[:, None],
                    contour[:, second_index + 1 :],
                )
            )
            split_contours.append(start_to_cutoff)
        else:
            raise ValueError("No intersections found in subdivision, is the contour valid?")

    # add original contour to split_contours to be the union of all subsegments
    split_contours.append(contour)

    return calc_subsegment_areas(split_contours), split_contours, split_points, subdivision_lines


def subdivide_contour_hampel(contour: Polygon2dType, num_rays: int) \
        -> tuple[np.ndarray[tuple[int], np.dtype[np.float64]], ContourList, list[Vector2d], list[Points2dType]]:
    """Subdivide contour based on area weights using equally spaced rays.

    Parameters
    ----------
    contour : np.ndarray
        Array of shape (2, N) containing contour points.
    num_rays : int
        Number of rays to use for subdivision.

    Returns
    -------
    areas : np.ndarray
        Array of areas for each subsegment.
    split_contours : list[np.ndarray]
        List of contour arrays for each subsegment.
    split_points : list[Vector2d]
        List of split points for each subsegment.
    subdivision_lines : list[Points2dType]
        List of contour arrays for each subdivision line.

    Notes
    -----
    The subdivision process:
    1. Finds extreme points in x-direction.
    2. Creates minimal bounding rectangle around contour.
    3. Creates equally spaced rays from lower edge of rectangle.
    4. Finds intersections of rays with contour.
    5. Creates new contours by splitting at intersections.
    6. Returns areas and split contours.
    """
    
    # Find the extreme points in the x-direction
    # Anterior end (max X)
    max_x_index = np.argmax(contour[0])
    contour = np.roll(contour, -max_x_index, axis=1)

    # get minimal bounding box around contour
    min_bounding_rectangle = minimum_bounding_rectangle(contour)

    # get long edges of rectangle
    rectangle_duplicate_last = np.vstack((min_bounding_rectangle, min_bounding_rectangle[0]))
    long_edges = np.diff(rectangle_duplicate_last, axis=0)
    long_edges = np.linalg.norm(long_edges, axis=1)
    long_edges_idx = np.argpartition(long_edges, -2)[-2:]

    # select lower long edge
    min_val = np.inf
    min_idx = None
    for i in long_edges_idx:
        if rectangle_duplicate_last[i][1] < min_val:
            min_val = rectangle_duplicate_last[i][1]
            min_idx = i

        if rectangle_duplicate_last[i + 1][1] < min_val:
            min_val = rectangle_duplicate_last[i + 1][1]
            min_idx = i

    lowest_points = rectangle_duplicate_last[[min_idx, min_idx + 1]]

    # sort lowest points by x coordinate
    if lowest_points[0, 0] < lowest_points[1, 0]:
        lowest_points = lowest_points[::-1]

    # get midpoint of lower edge of rectangle
    midpoint_lower_edge = np.mean(lowest_points, axis=0)

    # get angle of lower edge of rectangle to x-axis
    angle_lower_edge = np.arctan2(
        lowest_points[1, 1] - lowest_points[0, 1], lowest_points[1, 0] - lowest_points[0, 0]
    ) 

    # get angles for equally spaced rays
    angles = np.linspace(-angle_lower_edge, -angle_lower_edge + np.pi, num_rays + 2, endpoint=True)  # + np.pi *3
    angles = angles[1:-1]

    # create ray vectors
    ray_vectors = np.vstack((np.cos(angles), np.sin(angles)))
    # make ray vectors unit length
    ray_vectors = ray_vectors / np.linalg.norm(ray_vectors, axis=0)

    # invert x of ray vectors
    ray_vectors[0] = -ray_vectors[0]

    # Subdivision logic
    split_contours: ContourList = []
    subdivision_lines: list[Points2dType] = []
    split_points: list[Vector2d] = []
    num_points = contour.shape[1]
    for ray_vector in ray_vectors.T:
        intersections = []
        for i in range(num_points):
            segment_start = contour[:, i]
            segment_end = contour[:, (i + 1) % num_points]
            segment_vector = segment_end - segment_start

            # Check for intersection with the ray
            matrix = np.array([segment_vector, -ray_vector]).T
            if np.linalg.matrix_rank(matrix) < 2:
                continue  # Skip parallel lines

            # Solve for intersection
            # matrix * [t, s]^T = midpoint_lower_edge - segment_start
            try:
                t, s = np.linalg.solve(matrix, midpoint_lower_edge - segment_start)
                if 0 <= t < 1:  # Use half-open interval to avoid double-counting vertices
                    intersection_point = segment_start + t * segment_vector
                    intersections.append((i, intersection_point))
            except np.linalg.LinAlgError:
                continue

        # Create new contours by splitting at intersections
        if len(intersections) >= 2:
            # Sort intersections by their position along the contour (index)
            intersections.sort(key=lambda x: x[0])
            
            # For HAMPEL (radial rays), we usually expect two intersections.
            # If there are more, we pick the first and last along the contour.
            first_index, first_intersection = intersections[0]
            second_index, second_intersection = intersections[-1]

            subdivision_lines.append(np.stack([first_intersection, second_intersection], axis=0))
            split_points.append(np.mean([first_intersection, second_intersection], axis=0))
            start_to_cutoff = np.hstack(
                (
                    contour[:, :first_index],
                    first_intersection[:, None],
                    second_intersection[:, None],
                    contour[:, second_index + 1 :],
                )
            )

            # connect first and second half
            split_contours.append(start_to_cutoff)
        else:
            raise ValueError("No intersections found in subdivision, is the contour valid?")

    split_contours = [contour] + split_contours

    # order all anterior to posterior
    split_contours = split_contours[::-1]
    subdivision_lines = subdivision_lines[::-1]
    split_points = split_points[::-1]

    return calc_subsegment_areas(split_contours), split_contours, split_points, subdivision_lines


def subdivide_contour_vertical(
    contour: Polygon2dType,
    area_weights: list[float],
    oriented: bool = False,
    hline_anchor: np.ndarray | None = None
) -> tuple[np.ndarray[tuple[int], np.dtype[np.float64]], ContourList, list[Vector2d], list[Points2dType]]:
    """Subdivide contour based on area weights using vertical lines.

    Divides the contour into segments by drawing vertical lines at positions
    determined by the area weights. The lines are drawn perpendicular to a
    reference line connecting the extreme points of the contour.

    Parameters
    ----------
    contour : np.ndarray
        Array of shape (2, N) containing contour points.
    area_weights : list[float]
        List of weights for area-based subdivision.
    oriented : bool, optional
        If True, use fixed horizontal reference line, by default False.
    hline_anchor : np.ndarray, optional
        Point to anchor horizontal reference line, by default None.

    Returns
    -------
    areas : np.ndarray
        Array of areas for each subsegment.
    split_contours : list[np.ndarray]
        List of contour arrays for each subsegment.
    split_points : list[Vector2d]
        List of split points, the center of each subdivision line (i.e. inside the CC, close to the midline).
    subdivision_lines : list[Points2dType]
        List of contour arrays for each subdivision line.

    Notes
    -----
    The subdivision process:
    1. Finds extreme points in x-direction.
    2. Creates reference line between extremes.
    3. Calculates split points based on area weights.
    4. Divides contour using perpendicular lines at split points.
    
    """
    # Find the extreme points in the x-direction
    # standard ACPC orientation: Anterior has larger X, Posterior has smaller X
    max_x_index = np.argmax(contour[0])
    contour = np.roll(contour, -max_x_index, axis=1)

    max_x_index = 0
    min_x_index = np.argmin(contour[0])

    if oriented:
        contour_x_sorted = np.sort(contour[0])
        min_x = contour_x_sorted[0]
        max_x = contour_x_sorted[-1]
        # order extremes A to P
        extremes = (np.array([max_x, 0]), np.array([min_x, 0]))

        if hline_anchor is not None:
            extremes = (np.array([max_x, hline_anchor[1]]), np.array([min_x, hline_anchor[1]]))
    else:
        # order extremes A to P
        extremes = (contour[:, max_x_index].copy(), contour[:, min_x_index].copy())
        # Calculate the line between the extreme points
        start_point, end_point = extremes
        line_vector = end_point - start_point
        line_length = np.linalg.norm(line_vector)

        # Normalize the line vector
        line_unit_vector = line_vector / line_length

        # Calculate the perpendicular vector
        perp_vector = np.array([-line_unit_vector[1], line_unit_vector[0]])
        perp_vector = perp_vector / np.linalg.norm(perp_vector)

        if hline_anchor is None:
            most_inferior_point = np.min(contour[1])
            # move extreme 1 down 5 mm below inferior point and extreme 2 the 
            # same distance (so the angle stays the same)
            down_distance = (extremes[1][1] - most_inferior_point) * 1.3
            start_point = extremes[0] + down_distance * perp_vector
            end_point = extremes[1] + down_distance * perp_vector

        else:
            # get closest point on line to hline_anchor
            intersection = start_point + line_unit_vector * np.dot(hline_anchor - start_point, line_unit_vector)
            # get distance closest point on line to hline_anchor
            distance = np.linalg.norm(intersection - hline_anchor)
            # move start and end point the same distance
            start_point = extremes[0] + distance * perp_vector
            end_point = extremes[1] + distance * perp_vector

        extremes = (start_point, end_point)

    # Calculate the line between the extreme points
    start_point, end_point = extremes
    line_vector = end_point - start_point
    line_length = np.linalg.norm(line_vector)

    # Normalize the line vector
    line_unit_vector = line_vector / line_length

    # Calculate the perpendicular vector
    perp_vector = np.array([-line_unit_vector[1], line_unit_vector[0]])

    # Calculate points on the reference line (below the CC) based on area weights
    baseline_points = []
    for weight in area_weights:
        # current_weight = np.sum(area_weights[:i])
        split_distance = weight * line_length
        baseline_point = start_point + split_distance * line_unit_vector
        baseline_points.append(baseline_point)

    # Split the contour with the lines perpendicular to the reference line through the baseline points
    split_contours = []
    subdivision_lines: list[Points2dType] = []
    split_points: list[Vector2d] = []
    for baseline_point in baseline_points:
        intersections = []
        # also test the closing edge (last point -> first point) of the closed contour; with i = N-1 as the index of
        # the second intersection, the cumulative subsegment below closes from the second intersection back to point 0
        num_points = contour.shape[1]
        for i in range(num_points):
            segment_start = contour[:, i]
            segment_end = contour[:, (i + 1) % num_points]
            segment_vector = segment_end - segment_start

            # Check for intersection with the perpendicular line
            matrix = np.array([segment_vector, -perp_vector]).T
            if np.linalg.matrix_rank(matrix) < 2:
                continue  # Skip parallel lines

            # Solve for intersection
            t, s = np.linalg.solve(matrix, baseline_point - segment_start)
            if 0 <= t <= 1:
                intersection_point = segment_start + t * segment_vector
                intersections.append((i, intersection_point))

        # Sort intersections by their position along the contour
        # intersections.sort()

        # get the two intersections that have the highest y coordinate
        intersections.sort(key=lambda x: x[1][1], reverse=True)

        # Create new contours by splitting at intersections
        if len(intersections) >= 2:
            first_index, first_intersection = intersections[1]
            second_index, second_intersection = intersections[0]

            if first_index > second_index:
                first_index, second_index = second_index, first_index
                first_intersection, second_intersection = second_intersection, first_intersection

            subdivision_lines.append(np.stack([first_intersection, second_intersection], axis=0))
            # the split point is the center of the subdivision line (inside the CC), not the baseline point
            split_points.append((first_intersection + second_intersection) / 2)
            first_index += 1

            # connect first and second half to create a closed cumulative loop
            # starting from the Anterior end (start of the rolled contour)
            anterior_to_cutoff = np.hstack(
                (
                    contour[:, :first_index],
                    first_intersection[:, None],
                    second_intersection[:, None],
                    contour[:, second_index + 1 :],
                )
            )

            # add cumulative subsegment (Anterior -> Current Split)
            split_contours.append(anterior_to_cutoff)
        else:
            raise ValueError("No intersections found in subdivision, is the contour valid?")

    # add original contour as the final element (Full CC)
    split_contours.append(contour)

    return calc_subsegment_areas(split_contours), split_contours, split_points, subdivision_lines


def transform_to_acpc_standard(
        contour_ras: Polygon2dType | Polygon3dType,
        anterior_pt: Vector2d,
        posterior_pt: Vector2d,
) -> tuple[Polygon2dType, Vector2d, Vector2d, Callable[[Polygon2dType], Polygon2dType]]:
    """Transform contour coordinates to a standard space defined by an anterior and a posterior reference point.

    Transforms the contour coordinates by:
        1. Translating the anterior point to the origin.
        2. Rotating to align the posterior point with the posterior direction.

    The reference points are typically the CC endpoints (the "AC-PC" space of the CC module is aligned to the
    anterior and posterior CC endpoints, not to the commissures).

    Parameters
    ----------
    contour_ras : array of floats
        Array of shape (2, N) or (3, N) containing contour points in RAS space.
    anterior_pt : np.ndarray
        Anterior reference point coordinates in AS space (moved to the origin).
    posterior_pt : np.ndarray
        Posterior reference point coordinates in AS space (rotated onto the negative A axis).

    Returns
    -------
    contour_acpc : np.ndarray
        Transformed contour points in AC-PC space.
    anterior_pt_acpc : np.ndarray
        Anterior reference point in the standard space (origin).
    posterior_pt_acpc : np.ndarray
        Posterior reference point in the standard space.
    rotate_back : callable
        Function to transform points back to RAS space.
    """
    # translate the anterior point to the origin and the posterior point to (-ac_pc_dist, 0)
    translation_matrix = np.array([[1, 0, -anterior_pt[0]], [0, 1, -anterior_pt[1]], [0, 0, 1]])

    ac_pc_vec: Vector2d = posterior_pt - anterior_pt
    ac_pc_dist = np.linalg.norm(ac_pc_vec)

    posterior_vector: Vector2d = np.array([-ac_pc_dist, 0], dtype=float)

    # get angle between ac_pc_vec and posterior_vector
    dot_product = np.dot(ac_pc_vec, posterior_vector)
    norms_product = np.linalg.norm(ac_pc_vec) * np.linalg.norm(posterior_vector)
    theta = np.arccos(dot_product / norms_product)

    # Determine the sign of the angle using the signed z-component of the 2D cross product.
    cross_product = ac_pc_vec[0] * posterior_vector[1] - ac_pc_vec[1] * posterior_vector[0]
    if cross_product < 0:
        theta = -theta

    # create rotation matrix for theta
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    # apply translation and rotation
    if contour_ras.shape[0] == 2:
        contour_ras_homogeneous = np.vstack([contour_ras, np.ones(contour_ras.shape[1])])
    else:
        contour_ras_homogeneous = contour_ras

    contour_acpc: Polygon2dType = (rotation_matrix @ translation_matrix) @ contour_ras_homogeneous
    contour_acpc = contour_acpc[:2, :]

    def rotate_back(x: Polygon2dType) -> Polygon2dType:
        return (np.linalg.inv(rotation_matrix @ translation_matrix) @ np.vstack([x, np.ones(x.shape[1])]))[:2, :]

    return contour_acpc, np.array([0, 0], dtype=float), np.array([-ac_pc_dist, 0], dtype=float), rotate_back


def get_primary_eigenvector(contour_ras: Polygon2dType) -> tuple[Vector2d, Vector2d]:
    """Calculate primary eigenvector of contour points using PCA.

    Computes the principal direction of the contour by:
    1. Centering the points
    2. Computing covariance matrix
    3. Finding eigenvectors
    4. Selecting primary direction

    Parameters
    ----------
    contour_ras : np.ndarray
        Array of shape (2, N) containing contour points in ACPC space (anterior is positive x).

    Returns
    -------
    pt0 : np.ndarray
        Starting (anterior) point for eigenvector line.
    pt1 : np.ndarray
        End (posterior) point for eigenvector line.
    
    """
    # Center the data by subtracting mean
    contour_mean = np.mean(contour_ras, axis=1, keepdims=True)
    contour_centered = contour_ras - contour_mean

    # Calculate covariance matrix
    cov_matrix = np.cov(contour_centered)

    # Get eigenvalues and eigenvectors using PCA
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # make first eigenvector unit length
    primary_eigenvector = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])
    # the sign of eigenvectors is arbitrary, make the eigenvector point posterior (negative x in ACPC space), so pt0
    # is anterior of pt1 (expected by transform_to_acpc_standard)
    if primary_eigenvector[0] > 0:
        primary_eigenvector = -primary_eigenvector
    pt0 = np.mean(contour_ras, axis=1)
    pt0 -= np.array([0, 5])
    pt1 = pt0 + primary_eigenvector * 100
    
    return pt0, pt1

