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

from pathlib import Path
from typing import TypeVar

import lapy
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from lapy import TriaMesh
from plotly.io import write_html as plotly_write_html

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.shape.contour import CCContour
from CorpusCallosum.utils.colormaps import as_plotly_colorscale, get_colormap, resolve_color_range, with_neutral_range
from FastSurferCNN.utils import AffineMatrix4x4, nibabelHeader, nibabelImage
from FastSurferCNN.utils.common import update_docstring

try:
    from pyrr import Matrix44
    HAS_PYRR = True
except ImportError:
    HAS_PYRR = False
    class Matrix44(np.ndarray):
        pass

logger = logging.get_logger(__name__)


def _snapshot_color_options(
    vertex_values: np.ndarray | None,
    *,
    colormap: str | None = None,
    color_range: tuple[float, float] | None = None,
    legend: str | None = None,
    threshold: float | None = None,
) -> dict:
    """Translate CC controls into supported WhipperSnapPy native heat options."""
    if threshold is not None and not np.isscalar(threshold):
        raise ValueError("Snapshot threshold must be a finite, non-negative scalar.")
    fthresh = 0.0 if threshold is None else float(threshold)
    if not np.isfinite(fthresh) or fthresh < 0:
        raise ValueError("Snapshot threshold must be a finite, non-negative scalar.")

    options = dict(
        overlay=vertex_values, fthresh=fthresh, fmax=None, colorbar=True,
        caption="Corpus Callosum thickness (mm)" if legend is None else legend,
    )
    if colormap is not None:
        get_colormap(colormap)
        if colormap != "diverging":
            logger.warning(
                "WhipperSnapPy snapshots cannot use colormap '%s'; using native heat colors. "
                "PNG and Plotly HTML still use the selected colormap.", colormap,
            )
    if color_range is not None:
        data_min, data_max = resolve_color_range(np.asarray([]), color_range)
        if data_min == -data_max:
            options["fmax"] = data_max
        else:
            logger.warning(
                "WhipperSnapPy snapshots require a symmetric color_range (-M, M); ignoring %s "
                "for the snapshot and using automatic magnitude limits. PNG and Plotly HTML retain it.", color_range,
            )

    if vertex_values is not None:
        values = np.ma.asarray(vertex_values, dtype=float).filled(np.nan)
        if values.ndim != 1 or np.isinf(values).any():
            raise ValueError("Snapshot overlay must be one-dimensional and cannot contain infinite values.")
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError("Snapshot overlay must contain at least one finite value.")
        options["overlay"] = values
        if not np.any(np.abs(finite_values) > fthresh):
            # WhipperSnapPy rejects overlays with no sign left after thresholding.
            # Render the explicitly all-neutral result as its plain gray mesh.
            options.update(overlay=None, colorbar=False)
            logger.info(
                "All snapshot values are zero or at/below the threshold; rendering a gray mesh without a colorbar."
            )
        elif options["fmax"] is not None and options["fmax"] <= fthresh:
            logger.warning(
                "WhipperSnapPy snapshot saturation must exceed its threshold; ignoring color_range %s "
                "for the snapshot and using automatic magnitude limits.", color_range,
            )
            options["fmax"] = None
    return options


def make_triangles_between_contours(contour1: np.ndarray, contour2: np.ndarray) -> np.ndarray:
    """Create a triangular mesh between two contours using a robust method.

    Parameters
    ----------
    contour1 : np.ndarray
        First contour points of shape (N, 2).
    contour2 : np.ndarray
        Second contour points of shape (M, 2).

    Returns
    -------
    np.ndarray
        Array of triangle indices of shape (K, 3) where K is the number of triangles.

    Notes
    -----
    The function:
    1. Finds closest point on contour2 to first point of contour1
    2. Creates triangles by connecting corresponding points
    3. Handles contours with different numbers of points
    4. Creates two triangles to form a quad between each pair of points
    """
    start_idx_c1 = 0
    # get closest point on contour2 to contour1[0]
    start_idx_c2 = np.argmin(np.linalg.norm(contour2 - contour1[0], axis=1))

    triangles = []
    n1 = len(contour1)
    n2 = len(contour2)

    for i in range(n1):
        # Current and next indices for contour1
        c1_curr = (start_idx_c1 + i) % n1
        c1_next = (start_idx_c1 + i + 1) % n1

        # Current and next indices for contour2, offset by n1 to account for vertex stacking
        c2_curr = ((start_idx_c2 + i) % n2) + n1
        c2_next = ((start_idx_c2 + i + 1) % n2) + n1

        # Create two triangles to form a quad between the contours
        triangles.append([c1_curr, c2_curr, c1_next])
        triangles.append([c2_curr, c2_next, c1_next])

    return np.array(triangles)


Self = TypeVar('Self', bound='type[CCMesh]')


class CCMesh(lapy.TriaMesh):
    """A class for representing and manipulating corpus callosum (CC) meshes.

    This class extends lapy.TriaMesh to provide specialized functionality for working with
    corpus callosum meshes, including contour management, thickness measurements, and
    visualization capabilities.

    The mesh can be constructed from a series of 2D contours representing slices of the
    corpus callosum, with optional thickness measurements at various points along these
    contours.

    Attributes
    ----------
    v : np.ndarray
        Vertex coordinates of the mesh.
    t : np.ndarray
        Triangle indices of the mesh.
    mesh_vertex_colors : np.ndarray
        Vertex values for each vertex (CC thickness values)
    """

    def __init__(
            self,
            vertices: list | np.ndarray,
            faces: list | np.ndarray,
            vertex_values: list | np.ndarray | None = None,
    ):
        """Initialize a CC_Mesh object.

        Parameters
        ----------
        vertices : list or numpy.ndarray
            List of vertex coordinates or array of shape (N, 3).
        faces : list or numpy.ndarray
            List of face indices or array of shape (M, 3).
        vertex_values : list or numpy.ndarray, optional
            Vertex values for each vertex (CC thickness values)
        """
        super().__init__(np.vstack(vertices), np.vstack(faces))
        self.mesh_vertex_colors = vertex_values
   

    def plot_mesh(
        self,
        output_path: Path | str | None = None,
        colormap: str = "red_to_yellow",
        thickness_overlay: bool = True,
        show_grid: bool = False,
        color_range: tuple[float, float] | None = None,
        legend: str = "",
        threshold: float | tuple[float, float] | None = None,
    ):
        """Plot the mesh using Plotly for better performance and interactivity.

        Creates an interactive 3D visualization of the mesh with optional features like
        thickness overlay, contour display, and grid visualization.

        Parameters
        ----------
        output_path : Path, str, optional
            Path to save the plot. If None, displays the plot interactively.
        colormap : str, optional
            Which colormap to use, by default "red_to_yellow".
            Accepts the predefined FastSurfer colormaps, including the
            WhipperSnapPy/FreeSurfer-style ``"diverging"`` heat map, or any
            registered Matplotlib colormap name.
        thickness_overlay : bool, optional
            Whether to overlay thickness values on the mesh, by default True.
        show_contours : bool, optional
            Whether to show the contours, by default False.
        show_grid : bool, optional
            Whether to show the grid, by default False.
        color_range : tuple[float, float], optional
            Fixed range (min, max) for the colorbar, by default None.
        show_mesh_edges : bool, optional
            Whether to show the mesh edges, by default False.
        legend : str, optional
            Legend text for the colorbar, by default "".
        threshold : float or tuple of float, optional
            A scalar shows values between its negative and positive magnitude
            in gray. A tuple explicitly defines the gray interval.

        Notes
        -----
        The plot can be saved to an HTML file or displayed in a web browser.
        """
        assert self.v is not None and self.t is not None, "Mesh has not been created yet"

        if len(self.v) == 0:
            logger.warning("Warning: No vertices in mesh to plot")
            return

        if len(self.t) == 0:
            logger.warning("Warning: No faces in mesh to plot")
            return

        selected_cmap = get_colormap(colormap)
        data_min: float | None = None
        data_max: float | None = None
        if thickness_overlay and self.mesh_vertex_colors is not None:
            data_min, data_max = resolve_color_range(np.asarray(self.mesh_vertex_colors), color_range)
        if threshold is not None and thickness_overlay and self.mesh_vertex_colors is not None:
            assert data_min is not None and data_max is not None
            neutral_range = (-threshold, threshold) if np.isscalar(threshold) else threshold
            selected_cmap = with_neutral_range(selected_cmap, (data_min, data_max), neutral_range)
        selected_colormap = as_plotly_colorscale(selected_cmap)

        # Calculate data ranges and center
        xyz_min = self.v.min(axis=0)
        xyz_max = self.v.max(axis=0)
        xyz_range = xyz_max - xyz_min
        max_range = xyz_range.max()
        center = (xyz_max + xyz_min) / 2

        # Create mesh plot
        fig = go.Figure()

        # Add the mesh as a surface
        mesh_args = {
            "x": self.v[:, 0],
            "y": self.v[:, 1],
            "z": self.v[:, 2],
            "i": self.t[:, 0],  # First vertex of each triangle
            "j": self.t[:, 1],  # Second vertex
            "k": self.t[:, 2],  # Third vertex
            "hoverinfo": "skip",
            "lighting": dict(ambient=0.9, diffuse=0.1, roughness=0.3),
        }

        if thickness_overlay and self.mesh_vertex_colors is not None:
            mesh_args.update(
                {
                    "intensity": self.mesh_vertex_colors,  # Add intensity values for colorbar
                    "showscale": True,
                    "colorbar": dict(
                        title=dict(
                            text=legend,
                            font=dict(size=35, color="white"),  # Increase title font size and make white
                            side="right",  # Place title on right side
                        ),
                        len=0.55,  # Make colorbar shorter
                        thickness=35,  # Make colorbar wider
                        tickfont=dict(size=30, color="white"),  # Increase tick font size and make white
                        tickformat=".1f",  # Show one decimal place
                    ),
                    "opacity": 1,
                    "colorscale": selected_colormap,
                }
            )

            # Set the colorbar range
            mesh_args["cmin"] = data_min
            mesh_args["cmax"] = data_max
        else:
            mesh_args["color"] = "lightsteelblue"

        fig.add_trace(go.Mesh3d(**mesh_args))

        # Calculate axis ranges to maintain equal aspect ratio
        ranges = []
        for i in range(3):
            axis_range = [center[i] - max_range / 2, center[i] + max_range / 2]
            ranges.append(axis_range)

        # Configure axes and grid visibility
        axis_config = dict(
            showgrid=show_grid,
            showline=show_grid,
            zeroline=show_grid,
            showbackground=show_grid,
            showticklabels=show_grid,
            gridcolor="white",
            tickfont=dict(color="white"),
            title=dict(font=dict(color="white")),
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=ranges[0], **{**axis_config, "title": "LR" if show_grid else ""}),
                yaxis=dict(range=ranges[1], **{**axis_config, "title": "AP" if show_grid else ""}),
                zaxis=dict(range=ranges[2], **{**axis_config, "title": "SI" if show_grid else ""}),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1), up=dict(x=0, y=0, z=1)),
                aspectmode="cube",  # Force equal aspect ratio
                aspectratio=dict(x=1, y=1, z=1),
                bgcolor="black",
                dragmode="orbit",  # Enable orbital rotation by default
            ),
            showlegend=False,
            margin=dict(l=0, r=100, t=0, b=0),  # Increased right margin for colorbar
            paper_bgcolor="black",
            plot_bgcolor="black",
        )

        if output_path is not None:
            self.__make_parent_folder(output_path)
            plotly_write_html(fig, output_path, include_plotlyjs="cdn")  # Save as interactive HTML
        else:
            # For non-interactive display, save to a temporary HTML and open in browser
            import tempfile
            import webbrowser

            temp_path = Path(tempfile.gettempdir()) / "cc_mesh_plot.html"
            plotly_write_html(fig, temp_path, include_plotlyjs="cdn")
            webbrowser.open(f"file://{temp_path}")


    @staticmethod
    def __create_cc_viewmat() -> "Matrix44":
        """Create the view matrix for a nice view of the corpus callosum.

        Returns
        -------
        Matrix44
            4x4 view matrix that provides a standard view of the corpus callosum (from pyrr).

        Notes
        -----
        The function:
        1. Creates a base view matrix looking from the left with top up
        2. Applies a series of rotations:
            - -10 degrees around x-axis
            - 35 degrees around y-axis
            - -8 degrees around z-axis
        3. Adds a small translation for better centering
        """
        from whippersnappy import ViewType, get_view_matrix

        if not HAS_PYRR:
            raise ImportError("Pyrr not installed, install pyrr with `pip install pyrr`.")

        viewLeft = get_view_matrix(ViewType.LEFT)  # left w top up // right
        transl = Matrix44.from_translation((0, 0, 0.4))
        viewmat = transl * viewLeft

        # rotate 10 degrees around x axis
        rot = Matrix44.from_x_rotation(np.deg2rad(-10))
        viewmat = viewmat * rot

        # rotate 35 degrees around y axis
        rot = Matrix44.from_y_rotation(np.deg2rad(35))
        viewmat = viewmat * rot

        # rotate 10 degrees around z axis
        rot = Matrix44.from_z_rotation(np.deg2rad(-8))
        viewmat = viewmat * rot

        return viewmat

    def snap_cc_picture(
        self,
        output_path: Path | str,
        ref_header: Path | str | nibabelHeader | None = None,
        *,
        colormap: str | None = None,
        color_range: tuple[float, float] | None = None,
        legend: str | None = None,
        threshold: float | None = None,
    ) -> None:
        """Snap a picture of the corpus callosum mesh.

        Parameters
        ----------
        output_path : Path, str
            Path where to save the snapshot image.
        ref_header : Path, str, nibabelHeader, optional
            Path to reference image header to use for tkr creation. If None, ignores the file for saving.
        colormap : str, optional
            ``diverging`` uses the matching native heat palette. Other valid
            colormap names produce a warning and retain native snapshot colors.
        color_range : tuple of float, optional
            A symmetric range ``(-M, M)`` sets native saturation to ``M``.
            Unsupported ranges produce a warning and use automatic native limits.
        legend : str, optional
            Snapshot caption. Defaults to "Corpus Callosum thickness (mm)".
            An empty string hides the caption.
        threshold : float, optional
            Non-negative native absolute-value threshold; defaults to zero.
            Unlike PNG/HTML neutrality, native thresholding also rescales colors
            above the cutoff and uses the native background gray below it.

        Raises
        ------
        ImportError
            If whippersnappy is not installed or if the version is too old.
        ValueError
            If the mesh has no faces, or the overlay or numeric settings are invalid.
        """
        from packaging.version import parse
        try:
            import whippersnappy
        except ImportError as e:
            # whippersnappy not installed
            raise ImportError(
                f"The snap_cc_picture method of CCMesh requires {e.name}, but {e.name} was not found. "
                f"Please install {e.name}!",
                name=e.name, path=e.path
            ) from None
        from nibabel.affines import apply_affine

        if parse(whippersnappy.__version__) < parse("2.1.0"):
            raise ImportError(
                f"The snap_cc_picture method of CCMesh requires whippersnappy>=2.1, but version "
                f"{whippersnappy.__version__} is installed. Please upgrade whippersnappy to version 2.1 or higher!",
                name="whippersnappy", path=None
            )
        # Skip snapshot if there are no faces
        if len(self.t) == 0:
            raise ValueError("Cannot create snapshot - no faces in mesh")

        if self.mesh_vertex_colors is not None and np.shape(self.mesh_vertex_colors) != (len(self.v),):
            raise ValueError("Snapshot overlay must contain exactly one value per mesh vertex.")
        color_options = _snapshot_color_options(
            self.mesh_vertex_colors, colormap=colormap, color_range=color_range, legend=legend, threshold=threshold,
        )
        self.__make_parent_folder(output_path)

        if ref_header is None:
            v = self.v
        else:
            from nibabel.freesurfer.mghformat import MGHHeader

            # if header is a file, load its header from the file
            if isinstance(ref_header, str | Path):
                ref_header = nib.load(ref_header).header
            # if header is not already an MGHHeader, convert it to MGHHeader, so we have the get_vox2ras_tkr function
            mgh_header = ref_header if isinstance(ref_header, MGHHeader) else MGHHeader.from_header(ref_header)
            v = apply_affine(mgh_header.get_vox2ras_tkr(), self.v)

        whippersnappy.snap1(
            mesh=(v, self.t),
            **color_options,
            view=self.__create_cc_viewmat(),
            width=3 * 500,
            height=3 * 300,
            outpath=str(output_path),
            ambient=0.6,
            colorbar_scale=0.5,
            colorbar_y=0.88,
            colorbar_x=0.19,
            scale=2.1,
            caption_y=0.85,
            caption_x=0.17,
            caption_scale=0.5,
        )

    def smooth_(self, iterations: int = 1) -> None:
        """Smooth the mesh while preserving the z-coordinates.

        Parameters
        ----------
        iterations : int, optional
            Number of smoothing iterations, by default 1.

        Notes
        -----
        The function:
        1. Stores original z-coordinates.
        2. Applies Laplacian smoothing to x and y coordinates.
        3. Restores original z-coordinates to maintain slice structure.
        """
        z_values = self.v[:, 0]
        super().smooth_(iterations)
        self.v[:, 0] = z_values


    @staticmethod
    def __make_parent_folder(filename: Path | str) -> None:
        """Create the parent folder for a file if it doesn't exist.

        Parameters
        ----------
        filename : Path, str
            Path to the file whose parent folder should be created.

        Notes
        -----
        Creates parent directory with parents=False to avoid creating
        multiple levels of directories unintentionally.
        """
        Path(filename).parent.mkdir(parents=False, exist_ok=True)

    def to_vox_coordinates(
        self: Self,
        mesh_ras2vox: AffineMatrix4x4,
    ) -> Self:
        """Convert mesh coordinates to FreeSurfer coordinate system.

        Parameters
        ----------
        mesh_ras2vox : AffineMatrix4x4
            Transformation matrix from midplane mesh space (RAS centered on midplane) to voxel coordinates.

        Returns
        -------
        CCMesh
            A CCMesh object with vertices reoriented to FreeSurfer coordinates.

        Notes
        -----
        Mesh coordinates are in ASR (Anterior-Superior-Right) orientation, with the coordinate system origin on
        *the* midslice. The function *first* transforms from midslice ASR to LIA vox coordinates.
        """
        from copy import copy
        new_object = copy(self)

        # tkrRAS = Torig*[C R S 1]'
        # Torig: mri_info --vox2ras-tkr orig.mgz
        # https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems

        new_object.v = (mesh_ras2vox[:3, :3] @ self.v.T).T + mesh_ras2vox[None, :3, 3]
        return new_object

    @update_docstring(parent_doc=TriaMesh.write_fssurf.__doc__)
    def write_fssurf(self, filename: Path | str, image: str | nibabelImage | nibabelHeader | None = None) -> None:
        """{parent_doc}
        Also creates parent directory if needed before writing the file.
        """
        from os import environ
        # make sure the process has a username, so nibabel does not crash in write_geometry
        environ.setdefault("USERNAME", "UNKNOWN")

        self.__make_parent_folder(filename)
        return super().write_fssurf(filename, image=image)

    @update_docstring(parent_doc=TriaMesh.write_vtk.__doc__)
    def write_vtk(self, filename: Path | str) -> None:
        """{parent_doc}
        Also creates parent directory if needed before writing the file.
        """
        self.__make_parent_folder(filename)
        return super().write_vtk(filename)

    def write_morph_data(self, filename: Path | str) -> None:
        """Write the thickness values as a FreeSurfer overlay file.

        Parameters
        ----------
        filename : Path, str
            Path where to save the overlay file.

        Notes
        -----
        Creates parent directory if needed before writing the file.
        """
        self.__make_parent_folder(filename)
        return nib.freesurfer.write_morph_data(filename, self.mesh_vertex_colors)

    @classmethod
    def from_contours(
            cls: type[Self],
            contours: list[CCContour],
            *,
            smooth: int = 0,
    ) -> Self:
        """Create a surface mesh by triangulating between consecutive contours.

        Parameters
        ----------
        contours : list[CCContour]
            List of CCContour objects to create mesh from.
        smooth : int, default=0
            Number of smoothing iterations to apply.

        Returns
        -------
        CCMesh
            The joined CCMesh object.

        Raises
        ------
        ValueError
            If contours have duplicate slice positions or different numbers of points.

        Notes
        -----
        Each contour's z_position determines its left-right coordinate. The mesh
        connects consecutive contours, leaves both slab ends open, and optionally
        smooths the surface while preserving those coordinates.
        """
        # Check that all contours have distinct slice positions.
        z_coordinates = np.array([contour.z_position for contour in contours])
        same_z_position = np.isclose(z_coordinates[:, None], z_coordinates[None, :])
        # filter for diagonal and duplicates
        unique_same_z_position = np.logical_and(same_z_position, np.tri(z_coordinates.shape[0], k=-1, dtype=bool).T)
        if np.any(unique_same_z_position):
            raise ValueError(
                f"All contours must have different z_positions, but {np.array(np.where(unique_same_z_position)).T} "
                f"have similar z_positions."
            )

        # vertices list with z-coordinates and collect thickness values
        vertices = []
        faces = []
        vertex_values_list = []
        current_index = 0
        previous_contour: CCContour | None = None

        for contour in contours:
            vertices.append(np.hstack([np.full((len(contour.points), 1), contour.z_position), contour.points]))
            if contour.thickness_values is not None:
                vertex_values_list.append(contour.thickness_values)

            # Check if there's a next valid contour to connect to
            if previous_contour is not None:
                if len(previous_contour.points) != len(contour.points):
                    raise ValueError("The number of points of multiple contours must be the same!")
                faces_between = make_triangles_between_contours(previous_contour.points, contour.points)
                faces.append(faces_between + current_index)

                current_index += len(previous_contour.points)

            previous_contour = contour

        vertex_values = None
        if len(vertex_values_list) == len(contours):
            vertex_values = np.concatenate(vertex_values_list)
        elif len(vertex_values_list) > 0:
            logger.warning("Some contours have thickness values while others don't; skipping thickness overlay")

        if smooth > 0:
            tmp_mesh = CCMesh(vertices, faces, vertex_values=vertex_values)
            tmp_mesh.smooth_(smooth)
            vertices = tmp_mesh.v
            faces = tmp_mesh.t
            vertex_values = tmp_mesh.mesh_vertex_colors

        return cls(vertices, faces, vertex_values=vertex_values)
