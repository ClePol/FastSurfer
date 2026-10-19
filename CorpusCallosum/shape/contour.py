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

"""
This module provides the ``CCContour`` class for reading, writing, and
manipulating 2D corpus callosum contours together with per-vertex thickness
values. Typical template outputs (from ``fastsurfer_cc.py --save_template_dir``)
emit one set per slice:

- ``contour_<idx>.txt``: CSV with header ``New contour, anterior_endpoint_idx=<a>, posterior_endpoint_idx=<p>`` followed
  by ``x,y`` rows.
- ``thickness_values_<idx>.txt``: CSV with header ``thickness`` and one value per contour vertex.
- ``thickness_measurement_points_<idx>.txt``: CSV with header ``vertex_idx`` listing the vertices where thickness was
  measured.
"""

import re
from pathlib import Path
from typing import TypeVar

import lapy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.data.constants import CC_ANALYSIS_WIDTH_MM
from CorpusCallosum.shape.endpoint_heuristic import find_cc_endpoints, smooth_contour
from CorpusCallosum.shape.thickness import cc_thickness
from CorpusCallosum.utils.colormaps import get_colormap, resolve_color_range, with_neutral_range
from CorpusCallosum.utils.types import Points2dType
from FastSurferCNN.utils import AffineMatrix4x4, Mask2d, Vector2d

logger = logging.get_logger(__name__)

Self = TypeVar("Self", bound="CCContour")


def _interpolate_and_smooth_values(
    points_x: np.ndarray,
    points_y: np.ndarray,
    values: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate values without introducing artificial zero-valued edges."""
    linear_values = scipy.interpolate.griddata(
        (points_x, points_y), values, (x_grid, y_grid), method="linear", fill_value=np.nan,
    )
    nearest_values = scipy.interpolate.griddata(
        (points_x, points_y), values, (x_grid, y_grid), method="nearest",
    )
    grid_values = np.where(np.isnan(linear_values), nearest_values, linear_values)
    return scipy.ndimage.gaussian_filter(grid_values, sigma=5, radius=5)


# FIXME: Maybe CCContur should inherit from Polygon at a later date?
class CCContour:
    """A class for representing and manipulating corpus callosum (CC) contours.

    This class provides functionality for manipulating and analyzing corpus callosum contours.

    Attributes
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing 2D contour points.
    thickness_values : np.ndarray
        Array of shape (N,) for thickness measurements for each contour point.
    endpoint_idxs : tuple[int, int]
        Tuple containing the indices of the anterior and posterior endpoints in the contour.
    lr_position : float
        The left-right position of the contour's slice relative to the midslice in mm.
    original_thickness_vertices : np.ndarray
        Indices of the contour points that carry measured (not interpolated) thickness values.

    Examples
    --------
    >>> from CorpusCallosum.shape.contour import CCContour
    >>>
    >>> contour = CCContour(contour_points, thickness_values,
    >>>                     endpoint_idxs=(anterior_idx, posterior_idx),
    >>>                     lr_position=0.0)
    >>> contour.fill_thickness_values()   # interpolate missing values
    >>> contour.smooth_contour(window_size=5)
    >>> contour.save_contour("contour_0.txt")
    >>> contour.save_thickness_values("thickness_values_0.txt")
    """

    def __init__(
        self,
        points: Points2dType,
        thickness_values: np.ndarray[tuple[int], np.dtype[np.float64]] | None,
        endpoint_idxs: tuple[int, int] | None = None,
        lr_position: float = 0.0
    ):
        """Initialize a CCContour object.

        Parameters
        ----------
        points : np.ndarray
            Array of shape (N, 2) containing 2D contour points.
        thickness_values : np.ndarray, optional
            Array of thickness measurements for each contour point.
        endpoint_idxs : tuple[int, int], optional
            Tuple containing start and end indices for the contour.
        lr_position : float, default=0.0
            The left-right position of the contour's slice relative to the midslice in mm.
        """
        self.points = points
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"Contour must be a (N, 2) array, but is {self.points.shape}")
        self.thickness_values = thickness_values
        if thickness_values is not None:
            if self.points.shape[0] != len(thickness_values):
                raise ValueError(
                    f"Number of contour points ({self.points.shape[0]}) does not match number of thickness values "
                    f"({len(thickness_values)})",
                )
            # write vertex indices where thickness values are not nan
            self.original_thickness_vertices = np.where(~np.isnan(thickness_values))[0]
        else:
            self.original_thickness_vertices = None
        self.lr_position = lr_position

        if endpoint_idxs is None:
            self.endpoint_idxs = (0, len(points) // 2)
        else:
            self.endpoint_idxs = endpoint_idxs

    def __len__(self) -> int:
        """Return the number of points on the contour."""
        return len(self.points)

    @property
    def area(self) -> float:
        """Calculate the area of the contour using the shoelace formula.

        Returns
        -------
        float
            The area of the contour.
        """
        if len(self.points) < 3:
            return 0.0
        x = self.points[:, 0]
        y = self.points[:, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def smooth_contour(self, window_size: int = 5) -> None:
        """Smooth a contour using a moving average filter.

        Parameters
        ----------
        window_size : int, default=5
            Size of the smoothing window.

        Notes
        -----
        Uses smooth_contour from the endpoint_heuristic module.
        """
        self.points = np.array(smooth_contour(*self.points.T, window_size=window_size)).T

    def copy(self) -> "CCContour":
        """Copy the contour.
        """
        return CCContour(
            self.points.copy(),
            self.thickness_values.copy() if self.thickness_values is not None else None,
            self.endpoint_idxs,
            self.lr_position
        )
    
    def get_contour_edge_lengths(self) -> np.ndarray:
        """Get the lengths of the edges of a contour.

        Returns
        -------
        np.ndarray
            Array of edge lengths for the contour.

        Notes
        -----
        Edge lengths are calculated as Euclidean distances between consecutive points
        in the contour, including the edge closing the loop between the last and
        first point.
        """
        edges = np.roll(self.points, -1, axis=0) - self.points
        return np.sqrt(np.sum(edges**2, axis=1))

    def create_levelpaths(
            self,
            num_points: int,
            inplace: bool = False
    ) -> tuple[list[np.ndarray], float, float, np.ndarray, np.ndarray, tuple[int, int]]:
        """Calculate thickness and level paths for the CC contour using Laplace equation.

        Parameters
        ----------
        num_points : int
            Number of points for thickness estimation.
        inplace : bool, default=True
            Whether to update the contour points and thickness values in place.

        Returns
        -------
        levelpaths : list[np.ndarray]
            List of level paths across the CC.
        thickness : float
            Mean thickness of the CC.
        midline_len : float
            Length of the CC midline.
        midline_equi : np.ndarray
            Equidistant points along the midline.
        contour_with_thickness : np.ndarray
            Contour points with thickness information, shape (N, 3).
        endpoint_idxs : tuple[int, int]
            Indices of the anterior and posterior endpoints on the updated contour.
        """

         # FIXME: cache all these values in CCContour, and invalidate the cache, when either points or endpoint_idxs get
         #        changed; alternatively, make points and endpoint_idxs read_only (by creating getter-only properties)
         #        and have all functions that change points or endpoints return a new CCContour object instead.

        midline_len, thickness, midline_equi, levelpaths, contour_with_thickness, endpoint_idxs = \
            cc_thickness(
                self.points,
                self.endpoint_idxs,
                n_points=num_points,
            )

        if inplace:
            self.points = contour_with_thickness[:, :2]
            self.thickness_values = contour_with_thickness[:, 2]
            self.original_thickness_vertices = np.where(~np.isnan(self.thickness_values))[0]
            self.endpoint_idxs = endpoint_idxs

        return levelpaths, thickness, midline_len, midline_equi, contour_with_thickness, endpoint_idxs
    
    def get_thickness_profile(self) -> np.ndarray:
        """Return one thickness value per level path, ordered anterior to posterior.

        Saved template thickness values are attached to vertices of the closed
        contour.  Every level path therefore contributes the same measurement
        at both contour intersections.  Because the two contour sides are
        traversed in opposite directions, measurement vertices contain an
        anterior-to-posterior sequence followed by its reverse.

        Returns
        -------
        np.ndarray
            One-dimensional anterior-to-posterior thickness profile.

        Raises
        ------
        ValueError
            If measurement vertices are unavailable, their number is odd, or
            the measurements on the two contour sides do not agree.
        """
        if self.thickness_values is None or self.original_thickness_vertices is None:
            raise ValueError("Contour does not contain original thickness measurement vertices")

        measurement_values = np.asarray(self.thickness_values)[self.original_thickness_vertices]
        if len(measurement_values) % 2:
            raise ValueError(
                "A thickness profile requires two contour measurements per level path, "
                f"but found {len(measurement_values)} measurement vertices"
            )

        first_side, second_side = np.split(measurement_values, 2)
        if not np.allclose(first_side, second_side[::-1], equal_nan=False):
            raise ValueError(
                "Thickness measurements on opposite contour sides do not describe the same "
                "anterior-to-posterior profile"
            )
        return first_side.copy()

    def fill_thickness_values(self) -> None:
        """Interpolate missing thickness values using weighted averaging.

        Notes
        -----
        The function:
        1. Processes each contour with missing thickness values.
        2. For each missing value:
        - Finds two closest points with known thickness.
        - Calculates distances along contour.
        - Computes weighted average based on inverse distance.
        3. Updates thickness values in place.

        The weights are calculated as inverse distances to ensure closer
        points have more influence on the interpolated value.

        """
        thickness = self.thickness_values
        edge_lengths = self.get_contour_edge_lengths()

        # Find indices of points with known thickness
        known_idx = np.where(~np.isnan(thickness))[0]

        if len(known_idx) == 0:
            logger.warning("No known thickness values; skipping interpolation")
            return
        if len(known_idx) == 1:
            logger.warning("Only one known thickness value; skipping interpolation")
            thickness[np.isnan(thickness)] = thickness[known_idx[0]]
            self.thickness_values = thickness
            return

        # For each point with unknown thickness
        total_length = np.sum(edge_lengths)
        cumulative_lengths = np.concatenate(([0], np.cumsum(edge_lengths)))
        
        for j in range(len(thickness)):
            if not np.isnan(thickness[j]):
                continue

            # Find two closest points with known thickness
            distances = np.zeros(len(known_idx))
            for k, idx in enumerate(known_idx):
                # Calculate distance along contour by summing edge lengths
                # in both directions and taking the minimum
                if idx > j:
                    dist_forward = cumulative_lengths[idx] - cumulative_lengths[j]
                else:
                    dist_forward = cumulative_lengths[j] - cumulative_lengths[idx]
                
                distances[k] = min(dist_forward, total_length - dist_forward)

            # Get indices of two closest points
            closest_indices = known_idx[np.argsort(distances)[:2]]
            closest_distances = np.sort(distances)[:2]

            zero_distance = closest_distances <= 1e-10
            if np.any(zero_distance):
                thickness[j] = np.mean(thickness[closest_indices[zero_distance]])
                continue

            # Calculate weights based on inverse distance
            weights = 1.0 / closest_distances
            weights = weights / np.sum(weights)

            # Calculate weighted average thickness
            thickness[j] = np.sum(weights * thickness[closest_indices])

        self.thickness_values = thickness

    def plot_contour_colorfill(
        self,
        plot_values: np.ndarray,
        title: str | None = None,
        save_path: str | None = None,
        colorbar: bool = True,
        mode: str = "p-value",
        colormap: str | None = None,
        log_scale: bool = False,
        upper_threshold: float | None = None,
        threshold_color: str = "gray",
        colorbar_label: str | None = None,
        color_range: tuple[float, float] | None = None,
        legend: str | None = None,
        threshold: float | None = None,
    ) -> matplotlib.figure.Figure:
        """Plot a contour with levelset visualization.

        Creates a visualization of a contour with interpolated levelsets, useful for
        analyzing the thickness distribution across the corpus callosum.

        Parameters
        ----------
        plot_values : np.ndarray
            Array of values to plot on CC from anterior to posterior (left to right in the plot).
        title : str, optional
            Title for the plot.
        save_path : str, optional
            Path to save the plot. If None, displays interactively.
        colorbar : bool, default=True
            Whether to show the colorbar.
        mode : {"p-value", "icc", "thickness"}, default="p-value"
            Mode of the plot.
        colormap : str, optional
            Predefined or registered Matplotlib colormap. If omitted, use the
            mode-specific legacy colormap.
        log_scale : bool, default=False
            Use logarithmic color normalization. All finite plot values and the
            upper threshold, when provided, must be positive.
        upper_threshold : float, optional
            Draw interpolated values above this limit with ``threshold_color``.
        threshold_color : str, default="gray"
            Matplotlib color used for values above ``upper_threshold``.
        colorbar_label : str, optional
            Override the mode-specific colorbar label.
        color_range : tuple of float, optional
            Fixed minimum and maximum for the color scale in any mode.
        legend : str, optional
            Alternative colorbar label in thickness mode; colorbar_label takes precedence.
        threshold : float, optional
            Show values with an absolute magnitude below this value in neutral gray.
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object.
        """
        plot_values = np.asarray(plot_values, dtype=float)
        if plot_values.ndim != 1:
            raise ValueError(f"plot_values must be one-dimensional, found shape {plot_values.shape}")
        if len(plot_values) < 2:
            raise ValueError("plot_values must contain at least two values")
        if np.isinf(plot_values).any():
            raise ValueError("plot_values must not contain infinite values")

        if mode not in ("p-value", "icc", "thickness"):
            raise ValueError(f"Invalid mode '{mode}'")
        finite_values = plot_values[np.isfinite(plot_values)]
        if finite_values.size == 0:
            raise ValueError("plot_values must contain at least one finite value")
        if log_scale and np.any(finite_values <= 0):
            raise ValueError("Log-scaled plot values must be positive")
        if log_scale and upper_threshold is not None and upper_threshold <= 0:
            raise ValueError("A log-scaled upper threshold must be positive")

        # cc_thickness returns level paths in canonical anterior-to-posterior
        # order, matching thickness_profile in the JSON output and --values_file.
        levelpaths, *_ = self.create_levelpaths(num_points=len(plot_values), inplace=False)
        if len(levelpaths) != len(plot_values):
            raise ValueError(
                f"Requested {len(plot_values)} level paths, but thickness estimation returned {len(levelpaths)}"
            )

        outside_contour = self.points.T

        # Create a grid of points covering the contour area with higher resolution
        x_min, x_max = np.min(outside_contour[0]), np.max(outside_contour[0])
        y_min, y_max = np.min(outside_contour[1]), np.max(outside_contour[1])
        margin = 1
        resolution = 0.05  # Higher resolution for smoother interpolation
        x_grid, y_grid = np.meshgrid(
            np.arange(x_min - margin, x_max + margin, resolution), np.arange(y_min - margin, y_max + margin, resolution)
        )

        # Create a path from the outside contour
        contour_path = matplotlib.path.Path(np.column_stack([outside_contour[0], outside_contour[1]]))

        # Check which points are inside the contour
        points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
        mask = contour_path.contains_points(points).reshape(x_grid.shape)

        # Collect all levelpath points and their corresponding values
        # Extend each levelpath at both ends to improve extrapolation
        all_level_points_x = []
        all_level_points_y = []
        all_level_values = []

        for path, plot_value in zip(levelpaths, plot_values, strict=True):

            # add third dimension to path
            path = np.column_stack([path, np.zeros(len(path))])

            if len(path) == 1:
                all_level_points_x.append(path[0][0])
                all_level_points_y.append(path[0][1])
                all_level_values.append(plot_value)
                continue

            # make levelpath
            path = lapy.Polygon(path).resample(1000).points

            # Extend at the beginning: add point in direction opposite to first segment
            first_segment = path[1] - path[0]
            # standardize length of first segment
            first_segment = first_segment / np.linalg.norm(first_segment) * 10
            extension_start = path[0] - first_segment
            all_level_points_x.append(extension_start[0])
            all_level_points_y.append(extension_start[1])
            all_level_values.append(plot_value)

            # Add original path points
            for point in path:
                all_level_points_x.append(point[0])
                all_level_points_y.append(point[1])
                all_level_values.append(plot_value)

            # Extend at the end: add point in direction of last segment
            last_segment = path[-1] - path[-2]
            # standardize length of last segment
            last_segment = last_segment / np.linalg.norm(last_segment) * 10
            extension_end = path[-1] + last_segment
            all_level_points_x.append(extension_end[0])
            all_level_points_y.append(extension_end[1])
            all_level_values.append(plot_value)

        # Convert to numpy arrays
        all_level_points_x = np.array(all_level_points_x)
        all_level_points_y = np.array(all_level_points_y)
        all_level_values = np.array(all_level_values)

        # Linear interpolation does not cover the tips beyond the outermost
        # level paths. Fill those locations from the nearest level path before
        # smoothing; filling with zero would create false neutral-colored areas
        # when a diverging colormap is used.
        grid_values = _interpolate_and_smooth_values(
            all_level_points_x,
            all_level_points_y,
            all_level_values,
            x_grid,
            y_grid,
        )

        # Apply the mask to only show values inside the contour
        masked_values = np.where(mask, grid_values, np.nan)

        if colormap is not None:
            cmap = get_colormap(colormap)
        elif mode == "p-value":
            # Sample colormaps
            colors1 = plt.cm.binary([0.4] * 128)
            colors2 = plt.cm.hot(np.linspace(0.8, 0.1, 128))
        elif mode == "icc":
            colors1 = plt.cm.Blues(np.linspace(0, 1, 128))
            colors2 = plt.cm.binary([0.4] * 128)
        elif mode == "thickness":
            # Blue to red colormap for thickness values
            cmap = plt.cm.coolwarm

        # Combine the color samples for p-value and icc modes
        if colormap is None and mode != "thickness":
            colors = np.vstack((colors2, colors1))
            # Create a new colormap
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_colormap", colors)

        if mode == "thickness" or color_range is not None:
            default_vmin, default_vmax = resolve_color_range(plot_values, color_range)
        else:
            default_vmin, default_vmax = 0.0, 0.10 if mode == "p-value" else 1.0
        plot_vmax = upper_threshold if upper_threshold is not None else default_vmax
        if not np.isfinite(plot_vmax):
            raise ValueError("The upper threshold must be finite")
        if log_scale:
            plot_vmin = float(np.min(finite_values)) if color_range is None else default_vmin
            if plot_vmin <= 0:
                raise ValueError("A log color range must be positive")
            if plot_vmin >= plot_vmax:
                raise ValueError(f"Log color range must increase, got {plot_vmin} to {plot_vmax}")
            norm = matplotlib.colors.LogNorm(vmin=plot_vmin, vmax=plot_vmax)
        else:
            if default_vmin >= plot_vmax:
                raise ValueError("The upper threshold must exceed the color range minimum")
            norm = matplotlib.colors.Normalize(vmin=default_vmin, vmax=plot_vmax)
        if threshold is not None:
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("Threshold must be a finite, non-negative value")
            if log_scale:
                if threshold > norm.vmin:
                    # LogNorm cannot normalize zero or negative thresholds.
                    # Its domain is positive, so only the upper boundary matters.
                    cmap = with_neutral_range(cmap, (0.0, 1.0), (0.0, float(norm(threshold))))
            else:
                cmap = with_neutral_range(cmap, (norm.vmin, norm.vmax), (-threshold, threshold))
        if upper_threshold is not None:
            # Apply the over color to the underlying map, not the neutral interval.
            cmap = cmap.with_extremes(over=threshold_color)

        # Plot CC contour with levelsets
        fig = plt.figure(figsize=(10, 3))
        # Apply a 10-degree rotation to the entire plot
        base = plt.gca().transData
        transform = matplotlib.transforms.Affine2D().rotate_deg(10)
        transform = transform + base

        # Plot the filled contour with interpolated colors.
        image = plt.imshow(
            masked_values,
            extent=(x_min - margin, x_max + margin, y_min - margin, y_max + margin),
            origin="lower",
            cmap=cmap,
            alpha=1,
            interpolation="bilinear",
            norm=norm,
            transform=transform,
        )

        if colorbar:
            # Add a colorbar
            cbar = plt.colorbar(image, aspect=15, extend="max" if upper_threshold is not None else "neither")
            if mode == "p-value":
                if log_scale:
                    pvalue_ticks = [plot_vmin, 0.005, 0.01, plot_vmax]
                    pvalue_ticks = sorted({tick for tick in pvalue_ticks if plot_vmin <= tick <= plot_vmax})
                    cbar.set_ticks(pvalue_ticks)
                    cbar.set_ticklabels([f"{tick:g}" for tick in pvalue_ticks])
                    cbar.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                elif colormap is None and color_range is None and upper_threshold is None:
                    cbar.ax.set_ylim(0.001, 0.054)
                    cbar.ax.set_yticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
                else:
                    cbar.ax.set_ylim(default_vmin, plot_vmax)
                cbar.set_label(colorbar_label or ("p-value (log scale)" if log_scale else "p-value"))
            elif mode == "icc":
                if not log_scale and color_range is None and upper_threshold is None:
                    cbar.ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                cbar.set_label(colorbar_label or "Intraclass correlation coefficient")
            elif mode == "thickness":
                cbar.ax.set_ylim(norm.vmin, norm.vmax)
                cbar.set_label(colorbar_label or legend or "Thickness (mm)")

        # Plot the outside contour on top for clear boundary
        plt.plot(outside_contour[0], outside_contour[1], "k-", linewidth=2, label="CC Contour", transform=transform)

        plt.axis("equal")
        plt.title(title, fontsize=14, fontweight="bold")
        # plt.legend(loc='best')
        plt.gca().invert_xaxis()
        plt.axis("off")
        if save_path is not None:
            self.__make_parent_folder(save_path)
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
        return fig

    @staticmethod
    def __make_parent_folder(filename: Path | str) -> None:
        """Create the parent folder for a file if it doesn't exist.

        Parameters
        ----------
        filename : Path, str
            Path to the file whose parent folder should be created.

        Notes
        -----
        Creates parent directory with parents=False to avoid creating multiple levels of directories unintentionally.
        """
        Path(filename).parent.mkdir(parents=False, exist_ok=True)

    def save_contour(self, output_path: Path | str) -> None:
        """Save the contours to a CSV file.

        Parameters
        ----------
        output_path : Path, str
            Path to save the CSV file.

        Notes
        -----
        The function saves contours in CSV format with:
        - Header: slice_idx,x,y.
        - Special lines indicating new contours with endpoint indices.
        - Each point gets its own row with slice index and coordinates.
        """
        self.__make_parent_folder(output_path)
        logger.info(f"Saving contours to CSV file: {output_path}")
        with open(output_path, "w") as f:
            
            f.write(
                f"New contour, anterior_endpoint_idx={self.endpoint_idxs[0]}, "
                f"posterior_endpoint_idx={self.endpoint_idxs[1]}\n"
            )
            f.write("x,y\n")
            for point in self.points:
                f.write(f"{point[0]},{point[1]}\n")

    @classmethod
    def from_contour_file(
            cls: type[Self],
            input_path: str | Path,
            thickness_values_path: str | Path,
            lr_position: float = 0.0,
    ) -> Self:
        """Load contour from a CSV file.

        Parameters
        ----------
        input_path : str, Path
            Path to the CSV file containing the contours.
        thickness_values_path : str, Path
            Path to the CSV file containing the thickness_values.
        lr_position : float, default=0.0
            The distance to the midslice (in fsaverage space).

        Raises
        ------
        ValueError
            If the file format doesn't match expected structure.

        Notes
        -----
        The function:
        1. Reads CSV file with format matching save_contours output.
        2. Processes special lines for endpoint indices.
        3. Reconstructs contours and endpoint indices for each slice.
        4. Converts lists to fixed-size arrays with None padding.
        """
        current_points = []
        endpoint_idxs = []

        with open(input_path) as f:
            header = next(f).strip()
            # Parse endpoint indices from header
            anterior_match = re.search(r'anterior_endpoint_idx=(\d+)', header)
            posterior_match = re.search(r'posterior_endpoint_idx=(\d+)', header)
            assert anterior_match and posterior_match, "Header does not contain endpoint indices"

            anterior_idx = int(anterior_match.group(1))
            posterior_idx = int(posterior_match.group(1))
            endpoint_idxs = (anterior_idx, posterior_idx)

            # Skip column names
            next(f)

            for line in f:
                x, y = line.strip().split(",")
                current_points.append([float(x), float(y)])
        contour = np.array(current_points)
        if thickness_values_path:
            thickness_values = cls._load_thickness_values(contour, None, thickness_values_path)
        else:
            thickness_values = None
        return CCContour(contour, thickness_values, endpoint_idxs, lr_position=lr_position)

    def save_thickness_values(self, output_path: Path | str) -> None:
        """Save thickness values to a CSV file.

        Parameters
        ----------
        output_path : Path, str
            Path to save the CSV file.

        Notes
        -----
        The function saves thickness values in CSV format with:
        - Header: thickness.
        - Each thickness value gets its own row with slice index.
        - Skips slices with no thickness values.
        """
        self.__make_parent_folder(output_path)
        logger.info(f"Saving thickness data to CSV file: {output_path}")
        with open(output_path, "w") as f:
            f.write("thickness\n")
            for value in self.thickness_values:
                f.write(f"{value}\n")

    def load_thickness_values(
        self,
        input_path: str | Path,
    ) -> None:
        """Load thickness values from a CSV file.

        Parameters
        ----------
        input_path : Path, str
            Path to the CSV file containing thickness values.

        Raises
        ------
        ValueError
            If number of thickness values doesn't match measurement points
            or if number of slices is inconsistent.
        """
        self.thickness_values = self._load_thickness_values(self.points, self.original_thickness_vertices, input_path)

    @classmethod
    def _load_thickness_values(
        cls,
        contour: Points2dType,
        original_thickness_vertices: np.ndarray[tuple[int], np.dtype[np.signedinteger]] | None,
        input_path: str | Path,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """See load_thickness_values.

        Ignore shape of thickness values if original_thickness_vertices is None.

        Notes
        -----
        The function:
        1. Reads thickness values from CSV file.
        2. Groups values by slice index.
        3. Optionally associates values with specific vertices.
        4. Handles both full contour and profile measurements.
        """
        data = np.loadtxt(input_path, delimiter=",", skiprows=1)
        if data.ndim == 0:
            values = np.array([float(data)])
        elif data.ndim == 1:
            values = data.astype(float)
        else:
            raise ValueError("Thickness values file must contain a single column")

        if len(values) == len(contour):
            # Perfect match - use values directly
            new_values = values
        elif original_thickness_vertices is None:
            # No original vertices specified, use values as-is (may differ in length)
            new_values = values
        elif np.sum(~np.isnan(values)) == len(original_thickness_vertices):
            # Values match the number of measurement points, map them to the contour
            new_values = np.full(len(contour), np.nan)
            new_values[original_thickness_vertices] = values[~np.isnan(values)]
        else:
            raise ValueError(
                f"Number of thickness values {len(values)} does not match number of points in the contour "
                f"{len(contour)} and current number of measurement points {len(original_thickness_vertices)} does "
                f"not match the number of set thickness values {np.sum(~np.isnan(values))}."
            )

        return new_values

    @classmethod
    def from_mask_and_acpc(
            cls: type[Self],
            cc_mask: Mask2d,
            ac_2d: Vector2d,
            pc_2d: Vector2d,
            slice_vox2ras: AffineMatrix4x4,
            contour_smoothing: int = 5
    ) -> Self:
        """Extracts the contour of the CC using marching squares, smooth and transform to RAS coordinates.

        Parameters
        ----------
        cc_mask : np.ndarray of shape (H, W) and type bool
            Binary mask of the corpus callosum.
        ac_2d : np.ndarray of shape (2,) and type float
            2D voxel coordinates of the anterior commissure.
        pc_2d : np.ndarray of shape (2,) and type float
            2D voxel coordinates of the posterior commissure.
        slice_vox2ras : AffineMatrix4x4
            Transformation matrix from slice-voxel space to RAS-coordinates.
        contour_smoothing : int, default=5
            Number of Laplace smoothing iterations of the contour polygon (lapy Polygon.smooth_laplace).

        Returns
        -------
        contour : CCContour
            The contour object.

        Notes
        -----
        Expects LIA orientation.
        """
        import skimage.measure
        from nibabel.affines import apply_affine

        _contour: Points2dType = skimage.measure.find_contours(cc_mask, level=0.5)[0]

        # remove last, duplicate point
        _contour = _contour[:-1]
        polygon = lapy.Polygon(np.concatenate([np.zeros_like(_contour[:, :1]), _contour], axis=1), closed=True)
        polygon.smooth_laplace(n=contour_smoothing, inplace=True)
        polygon.resample(700, inplace=True)
        contour_ras = apply_affine(slice_vox2ras, polygon.points)

        ac_pc_3d = np.concatenate([np.zeros((2, 1), like=ac_2d), np.stack([ac_2d, pc_2d], axis=0)], axis=1) # (2, 3)
        ac_ras, pc_ras = apply_affine(slice_vox2ras, ac_pc_3d)
        endpoint_idx = find_cc_endpoints(contour_ras[:, 1:].T, ac_ras[1:], pc_ras[1:])

        return cls(contour_ras[:, 1:], None, endpoint_idx, lr_position=slice_vox2ras[0, 3])


def calculate_volume(contours: list[CCContour], width: float = CC_ANALYSIS_WIDTH_MM) -> float:
    """Calculate the volume of the corpus callosum.

    This method calculates the volume of a slab of the CC centered on the midplane.
    It multiplies the area of each cross-sectional slice by the width it
    represents within the slab. It assumes equally spaced contours centered
    around the midplane (lr_position=0).

    Parameters
    ----------
    contours : list[CCContour]
        Contours of equally spaced slices, ordered by their left-right position (lr_position).
    width : float, default=CC_ANALYSIS_WIDTH_MM
        The width of the slab centered on the midplane to calculate the volume for (in mm).

    Returns
    -------
    float
        The volume of the CC in cubic millimeters.
    """
    if len(contours) < 2:
        return 0.0

    # left-right positions of the contours (in mm relative to the midplane)
    lr_coords = [contour.lr_position for contour in contours]
    areas = [contour.area for contour in contours]

    contour_widths = np.diff(lr_coords)

    # check that all widths are the same
    if not np.allclose(contour_widths, contour_widths[0]):
        raise ValueError("Contours must be equally spaced to calculate CC volume")

    contour_width_mm = abs(contour_widths[0])

    # Define the slab boundaries centered on the midplane (LR=0)
    lr_min, lr_max = -width / 2.0, width / 2.0

    volume = 0.0
    for i, lr_coord in enumerate(lr_coords):
        # Each contour represents a slab of contour_width_mm
        # centered at its LR position.
        start = lr_coord - contour_width_mm / 2.0
        end = lr_coord + contour_width_mm / 2.0

        # Intersection of [start, end] and [lr_min, lr_max]
        effective_start = max(start, lr_min)
        effective_end = min(end, lr_max)

        effective_width = max(0.0, effective_end - effective_start)
        volume += areas[i] * effective_width

    return volume


def contours_for_analysis_width(contours: list[CCContour]) -> list[CCContour]:
    """Return contour copies with LR positions spanning a fixed display width.

    The CC segmentation mask is sampled on image slices, so its realized width
    depends on the voxel size and the odd number of slices needed to cover the
    requested analysis slab. The surface representation, however, is the
    continuous slab itself. For meshes we therefore place the available contours
    evenly across that physical width.

    The contour list follows increasing slab voxel index order. In the upright
    CC volume, the slab voxel axis has a negative RAS left/right direction, so
    increasing slice indices correspond to decreasing mesh LR RAS coordinates.
    Assign the display positions from ``+CC_ANALYSIS_WIDTH_MM / 2`` down to
    ``-CC_ANALYSIS_WIDTH_MM / 2`` to preserve that orientation.

    This function intentionally returns copies. Measurement code can continue to
    use the original slice-center coordinates, while visualization/surface
    output gets the fixed physical slab width expected by Freeview inspection.
    """
    if not contours:
        return []

    surface_contours = [contour.copy() for contour in contours]
    if len(surface_contours) == 1:
        surface_contours[0].lr_position = 0.0
        return surface_contours

    half_width = CC_ANALYSIS_WIDTH_MM / 2.0
    for contour, lr_position in zip(
        surface_contours,
        np.linspace(half_width, -half_width, len(surface_contours)),
        strict=True,
    ):
        contour.lr_position = float(lr_position)
    return surface_contours
