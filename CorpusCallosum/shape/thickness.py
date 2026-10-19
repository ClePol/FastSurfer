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
from typing import Literal, overload

import numpy as np
from lapy import Solver, TriaMesh
from lapy.diffgeo import compute_rotated_f
from meshpy import triangle

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.utils.types import ContourThickness, Points2dType
from FastSurferCNN.utils.common import suppress_stdout

logger = logging.get_logger(__name__)

def set_contour_zero_idx(contour, idx, anterior_endpoint_idx, posterior_endpoint_idx):
    """Roll contour points to set a new zero index, while keeping track of CC endpoints.

    Parameters
    ----------
    contour : np.ndarray
        Array of contour points.
    idx : int
        New zero index.
    anterior_endpoint_idx : int
        Index of anterior endpoint.
    posterior_endpoint_idx : int
        Index of posterior endpoint.

    Returns
    -------
    contour : np.ndarray
        Rolled contour points.
    anterior_endpoint_idx : int
        Updated anterior endpoint index.
    posterior_endpoint_idx : int
        Updated posterior endpoint index.
"""
    contour = np.roll(contour, -idx, axis=0)
    anterior_endpoint_idx = (anterior_endpoint_idx - idx) % contour.shape[0]
    posterior_endpoint_idx = (posterior_endpoint_idx - idx) % contour.shape[0]
    return contour, anterior_endpoint_idx, posterior_endpoint_idx


def find_closest_edge(point, contour):
    """Find the index of the edge closest to the given point.

    Parameters
    ----------
    point : np.ndarray
        2D point coordinates.
    contour : np.ndarray
        Array of shape (N, 2) containing contour points.

    Returns
    -------
    int
        Index of the closest edge.
    """
    edges_start = contour[:, :2]  # N x 2
    edges_end = np.roll(contour[:, :2], -1, axis=0)  # N x 2
    edges_vec = edges_end - edges_start  # N x 2

    # Calculate projection coefficient for all edges at once
    # (p-a)·(b-a) / |b-a|²
    edge_lengths_sq = np.sum(edges_vec * edges_vec, axis=1)
    # Avoid division by zero for degenerate edges
    valid_edges = edge_lengths_sq > 1e-10
    t = np.zeros(len(edges_start))
    t[valid_edges] = (
        np.sum((point - edges_start[valid_edges]) * edges_vec[valid_edges], axis=1)
        / edge_lengths_sq[valid_edges]
    )
    t = np.clip(t, 0, 1)  # Clamp to edge endpoints

    # Get closest points on all edges
    closest_points = edges_start + t[:, None] * edges_vec

    # Calculate distances to all edges
    distances = np.linalg.norm(point - closest_points, axis=1)

    # Return index of closest edge
    return np.argmin(distances)


def interpolate_linear_on_mesh(
        vertices: np.ndarray,
        trias: np.ndarray,
        vfunc: np.ndarray,
        query_points: np.ndarray,
) -> np.ndarray:
    """Evaluate a piecewise linear vertex function at query points inside a 2D triangle mesh.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (V, 2) containing mesh vertex coordinates.
    trias : np.ndarray
        Array of shape (T, 3) containing triangle vertex indices.
    vfunc : np.ndarray
        Array of shape (V,) containing the function values at the vertices.
    query_points : np.ndarray
        Array of shape (M, 2) containing the points to evaluate.

    Returns
    -------
    np.ndarray
        Array of shape (M,) containing the interpolated function values.

    Notes
    -----
    Unlike scipy.interpolate.griddata, this interpolates within the triangles of the given mesh, i.e. exactly
    like the piecewise linear function that level paths are extracted from. Each query point is assigned to the
    triangle with the largest minimal barycentric coordinate, so points marginally outside the mesh (numerical
    noise) are assigned to the closest triangle.
    """
    a, b, c = (vertices[trias[:, i]] for i in range(3))  # T x 2 each
    v0, v1 = b - a, c - a
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]  # T
    rel = query_points[:, None, :] - a[None]  # M x T x 2
    bary1 = (rel[..., 0] * v1[:, 1] - rel[..., 1] * v1[:, 0]) / det
    bary2 = (v0[:, 0] * rel[..., 1] - v0[:, 1] * rel[..., 0]) / det
    bary = np.stack([1 - bary1 - bary2, bary1, bary2], axis=-1)  # M x T x 3
    tria_idx = np.argmax(bary.min(axis=-1), axis=1)  # M
    bary_of_tria = bary[np.arange(len(query_points)), tria_idx]  # M x 3
    return np.sum(bary_of_tria * vfunc[trias[tria_idx]], axis=1)


def _distance_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    """Compute the minimal distance of a 2D point to an open polyline of shape (N, 2)."""
    seg_start, seg_vec = polyline[:-1], np.diff(polyline, axis=0)
    seg_len_sq = np.sum(seg_vec * seg_vec, axis=1)
    t = np.zeros(len(seg_start))
    valid = seg_len_sq > 1e-20
    t[valid] = np.sum((point - seg_start[valid]) * seg_vec[valid], axis=1) / seg_len_sq[valid]
    closest = seg_start + np.clip(t, 0, 1)[:, None] * seg_vec
    return float(np.min(np.linalg.norm(point - closest, axis=1)))


def find_levelpath_through_point(
        tria: TriaMesh,
        vfunc: np.ndarray,
        level: float,
        point: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Extract the open level path of vfunc at level that passes through (or closest to) point.

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh (with z=0).
    vfunc : np.ndarray
        Function values at the mesh vertices.
    level : float
        Level of the level set.
    point : np.ndarray
        2D point that the level path should pass through, typically where the level was sampled.

    Returns
    -------
    levelpath : np.ndarray
        Array of shape (N, 3) containing the points of the level path.
    length : float
        Length of the level path.

    Raises
    ------
    ValueError
        If the level set has no open component.

    Notes
    -----
    The level set may consist of multiple components, e.g. small additional arcs or loops where vfunc is not
    perfectly monotonic along the boundary. Only open components (from boundary to boundary) are considered, and
    the one closest to point is returned. If the level set is degenerate (e.g. runs along mesh edges), the level is
    nudged slightly.
    """
    vrange = float(np.ptp(vfunc))
    for nudge in (0., 1e-9, -1e-9):
        curves = [c for c in tria.extract_level_paths(vfunc, level + nudge * vrange) if not c.closed]
        if curves:
            break
    else:
        raise ValueError(f"No open level path found at level {level}.")
    curve = min(curves, key=lambda c: _distance_to_polyline(point, c.points[:, :2]))
    return curve.points, curve.length()


@overload
def insert_point_with_thickness(
    contour_in_as_space: np.ndarray,
    contour_thickness: np.ndarray,
    point: np.ndarray,
    thickness_value: float,
    return_index: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def insert_point_with_thickness(
    contour_in_as_space: np.ndarray,
    contour_thickness: np.ndarray,
    point: np.ndarray,
    thickness_value: float,
    return_index: Literal[True],
) -> tuple[np.ndarray, np.ndarray, int]:
    ...


def insert_point_with_thickness(
    contour_in_as_space: np.ndarray,
    contour_thickness: np.ndarray,
    point: np.ndarray,
    thickness_value: float,
    return_index: bool = False
) -> tuple[np.ndarray, np.ndarray, int] | tuple[np.ndarray, np.ndarray]:
    """Inserts a point and its thickness value into the contour.

    Parameters
    ----------
    contour_in_as_space : np.ndarray
        Array of coordinates of the contour in AS space, shape (N, 2).
    contour_thickness : np.ndarray
        Array of thickness values of the contour, shape (N,).
    point : np.ndarray
        2D point to insert, shape (2,).
    thickness_value : float
        Thickness value corresponding to the point.
    return_index : bool, default=False
        If True, return the index where point was inserted, by default False.

    Returns
    -------
    contour_in_as_space : np.ndarray
        Updated contour of shape (N+1, 2).
    contour_thickness : np.ndarray
        Updated thickness values of shape (N+1,).
    insertion_index : int
        The index, where the point was inserted (only if return_index is True).
    """
    existing_distances = np.linalg.norm(contour_in_as_space[:, :2] - point[:2], axis=1)
    existing_matches = np.where(existing_distances <= 1e-10)[0]
    if len(existing_matches) > 0:
        point_idx = int(existing_matches[0])
        if np.isnan(contour_thickness[point_idx]):
            contour_thickness[point_idx] = thickness_value

        # Keep the contour cardinality stable across slices, but avoid inserting
        # an exact duplicate vertex. Exact duplicates create zero-length edges in
        # visualization/mesh code, while skipping the point breaks the equal
        # point-count invariant expected by CCMesh.from_contours().
        before_idx = (point_idx - 1) % len(contour_in_as_space)
        after_idx = (point_idx + 1) % len(contour_in_as_space)
        before_vector = contour_in_as_space[before_idx, :2] - contour_in_as_space[point_idx, :2]
        after_vector = contour_in_as_space[after_idx, :2] - contour_in_as_space[point_idx, :2]
        before_length = np.linalg.norm(before_vector)
        after_length = np.linalg.norm(after_vector)

        if after_length >= before_length and after_length > 0:
            insert_idx = point_idx + 1
            direction = after_vector / after_length
            edge_length = after_length
        elif before_length > 0:
            insert_idx = point_idx
            direction = before_vector / before_length
            edge_length = before_length
        else:
            insert_idx = point_idx + 1
            direction = np.zeros_like(point[:2])
            edge_length = 0.0

        if edge_length > 0:
            step = min(max(edge_length * 1e-6, 1e-8), edge_length * 0.5)
            point = contour_in_as_space[point_idx, :2] + direction * step

        contour_in_as_space = np.insert(contour_in_as_space, insert_idx, point, axis=0)
        contour_thickness = np.insert(contour_thickness, insert_idx, thickness_value)

        if return_index:
            return contour_in_as_space, contour_thickness, insert_idx
        else:
            return contour_in_as_space, contour_thickness

    # Find closest edge for the point
    edge_idx = find_closest_edge(point, contour_in_as_space)

    # Insert point between edge endpoints
    contour_in_as_space = np.insert(contour_in_as_space, edge_idx + 1, point, axis=0)
    contour_thickness = np.insert(contour_thickness, edge_idx + 1, thickness_value)

    if return_index:
        return contour_in_as_space, contour_thickness, edge_idx + 1
    else:
        return contour_in_as_space, contour_thickness


def make_mesh_from_contour(
    contour_2d: np.ndarray,
    max_volume: float = 0.5,
    min_angle: float = 25,
    verbose: bool = False
) -> tuple[Points2dType[np.float64], np.ndarray[tuple[int, Literal[3]], np.dtype[np.int_]]]:
    """Create a triangular mesh from a 2D contour.

    Parameters
    ----------
    contour_2d : np.ndarray
        Array of shape (N, 2) containing contour points.
    max_volume : float, optional
        Maximum triangle area, by default 0.5.
    min_angle : float, optional
        Minimum angle in triangles (degrees), by default 25.
    verbose : bool, optional
        Whether to print mesh generation info, by default False.

    Returns
    -------
    mesh_points : np.ndarray
        Array of shape (M, 2) containing mesh vertices.
    mesh_trias : np.ndarray
        Array of shape (K, 3) containing triangle indices.

    Notes
    -----
    Uses meshpy.triangle to create a constrained Delaunay triangulation
    of the contour. The contour must not have duplicate points.
    """

    facets = np.vstack((np.arange(len(contour_2d)), ((np.arange(len(contour_2d)) + 1) % len(contour_2d)))).T

    # use meshpy to create mesh
    info = triangle.MeshInfo()
    info.set_points(contour_2d)  # needs to be (N, D)
    info.set_facets(facets)
    # NOTE: crashes if contour has duplicate points !!
    mesh = triangle.build(info, max_volume=max_volume, min_angle=min_angle, verbose=verbose)

    mesh_points: Points2dType[np.float64] = np.array(mesh.points, dtype=float)
    mesh_trias: np.ndarray[tuple[int, Literal[3]], np.dtype[np.int_]] = np.array(mesh.elements, dtype=int)

    return mesh_points, mesh_trias


def cc_thickness(
    contour_2d: Points2dType,
    endpoint_idx: tuple[int, int],
    n_points: int = 100,
) -> tuple[float, float, Points2dType , list[Points2dType], ContourThickness, tuple[int, int]]:
    """Calculate corpus callosum thickness using Laplace equation.

    Parameters
    ----------
    contour_2d : np.ndarray
        Array of shape (N, 2) containing contour points.
    endpoint_idx : pair of ints
        Indices of anterior and posterior endpoints in contour.
    n_points : int, default=100
        Number of points for thickness measurement.

    Returns
    -------
    midline_length : float
        Total length of the midline.
    thickness : float
        Mean thickness across all level paths.
    midline_equidistant : np.ndarray
        Equidistant points along the midline in same space as contour2d of shape (N, 2).
    levelpaths : list[np.ndarray]
        Level paths for thickness measurement in same space as contour2d, each of shape (N, 2).
    contour_with_thickness : np.ndarray
        Contour coordinates with thickness information in same space as contour2d of shape (N + 2 * L, 3), where L is
        the number of level paths (the start and end points of each level path are inserted into the contour).
    endpoint_indices : pair of ints
        Pair of updated indices of anterior and posterior endpoint.

    Notes
    -----
    Uses the Laplace equation to compute thickness by:
    1. Creating a triangular mesh from the contour
    2. Setting boundary conditions (0 at endpoints, ±1 on sides)
    3. Solving Laplace equation to get level sets
    4. Computing thickness along level sets
    """
    anterior_endpoint_idx, posterior_endpoint_idx = endpoint_idx

    # standardize contour indices to start at anterior_endpoint_idx, to get consistent levelpath directions
    contour_2d, anterior_endpoint_idx, posterior_endpoint_idx = set_contour_zero_idx(
        contour_2d, anterior_endpoint_idx, anterior_endpoint_idx, posterior_endpoint_idx,
    )

    mesh_points_contour_space, mesh_trias = make_mesh_from_contour(contour_2d)

    # make points 3D by appending z=0, asz space therefore is the contour space (usually AS space) with a zero z-dim
    mesh_points_asz = np.append(mesh_points_contour_space, np.zeros((mesh_points_contour_space.shape[0], 1)), axis=1)

    # compute poisson
    with suppress_stdout():
        tria_asz = TriaMesh(mesh_points_asz, mesh_trias)
        # extract boundary curve
        bdr = np.array(tria_asz.boundary_loops()[0])

        # find index of endpoints in bdr list
        iidx1 = np.where(bdr == anterior_endpoint_idx)[0][0]
        iidx2 = np.where(bdr == posterior_endpoint_idx)[0][0]

        # create boundary condition (0 at endpoints, -1 on one side, 1 on the other):
        if iidx1 > iidx2:
            iidx1, iidx2 = iidx2, iidx1
        dcond = np.ones(bdr.shape)
        dcond[iidx1] = 0
        dcond[iidx2] = 0
        dcond[iidx1 + 1 : iidx2] = -1

        # Extract path
        fem = Solver(tria_asz)
        vfunc = fem.poisson(0, (bdr, dcond))
        midline_length: float
        try:
            midline_equidistant_asz, midline_length = tria_asz.level_path(vfunc, level=0., n_points=n_points + 2)
        except Exception as e:
            logger.error(f"Failed to compute midline level path: {e}")
            # If midline computation fails, we cannot proceed with thickness calculation
            raise RuntimeError(f"Corpus callosum midline computation failed: {e}") from e

        # LaPy orders open level paths by internal graph traversal, not by
        # anatomical endpoint semantics. Downstream subdivision expects the
        # midline to run from anterior to posterior.
        anterior_endpoint = contour_2d[anterior_endpoint_idx]
        start_dist_to_anterior = np.linalg.norm(midline_equidistant_asz[0, :2] - anterior_endpoint)
        end_dist_to_anterior = np.linalg.norm(midline_equidistant_asz[-1, :2] - anterior_endpoint)
        if start_dist_to_anterior > end_dist_to_anterior:
            midline_equidistant_asz = midline_equidistant_asz[::-1]

        midline_equidistant_contour_space: np.ndarray = midline_equidistant_asz[:, :2]

        gf = compute_rotated_f(tria_asz, vfunc)

        # interpolate midline to get levels to evaluate; linear interpolation within the mesh triangles guarantees
        # that the level set of (piecewise linear) gf passes through the respective midline point
        level_of_rotated_laplace_contour_space = interpolate_linear_on_mesh(
            tria_asz.v[:, 0:2], tria_asz.t, gf, midline_equidistant_contour_space,
        )

    # get levels to evaluate
    levelpaths_contour_space: list[Points2dType] = []
    levelpath_lengths = []

    # now, on the rotated laplace function, sample equally spaced (on midline: level_of_rotated_laplace) levelpaths
    contour_thickness = np.full(contour_2d.shape[0], np.nan)
    for current_level, midline_point in zip(
            level_of_rotated_laplace_contour_space[1:-1], midline_equidistant_contour_space[1:-1], strict=True,
    ):
        # levelpath starts at index zero
        try:
            levelpath_asz, lvlpath_length = find_levelpath_through_point(tria_asz, gf, current_level, midline_point)
        except ValueError as e:
            logger.warning(f"Failed to compute level path at level {current_level}: {e}")
            continue

        levelpaths_contour_space.append(levelpath_asz[:, :2])
        levelpath_lengths.append(lvlpath_length)

        levelpath_start = levelpath_asz[0, :2]
        levelpath_end = levelpath_asz[-1, :2]

        contour_2d, contour_thickness, inserted_idx_start = insert_point_with_thickness(
            contour_2d, contour_thickness, levelpath_start, lvlpath_length, return_index=True,
        )
        # keep track of start index
        if inserted_idx_start <= anterior_endpoint_idx:
            anterior_endpoint_idx += 1
        if inserted_idx_start <= posterior_endpoint_idx:
            posterior_endpoint_idx += 1

        contour_2d, contour_thickness, inserted_idx_end = insert_point_with_thickness(
            contour_2d, contour_thickness, levelpath_end, lvlpath_length, return_index=True,
        )
        # keep track of end index
        if inserted_idx_end <= anterior_endpoint_idx:
            anterior_endpoint_idx += 1
        if inserted_idx_end <= posterior_endpoint_idx:
            posterior_endpoint_idx += 1

    contour_2d_with_thickness = np.concatenate([contour_2d, contour_thickness[:, None]], axis=1)

    mean_thickness: float = np.mean(levelpath_lengths).item()
    endpoints: tuple[int, int] = (anterior_endpoint_idx, posterior_endpoint_idx)

    return (
        midline_length,
        mean_thickness,
        midline_equidistant_contour_space,
        levelpaths_contour_space,
        contour_2d_with_thickness,
        endpoints,
    )
