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

from contextlib import nullcontext
from pathlib import Path

import numpy as np

from CorpusCallosum.utils.types import ContourList, Polygon2dType
from FastSurferCNN.utils import AffineMatrix4x4, Image3d, Vector2d


def plot_contours(
    slice_or_slab: Image3d,
    split_contours: ContourList | None = None,
    midline_equidistant: Polygon2dType | None = None,
    levelpaths: list[Polygon2dType] | None = None,
    output_path: str | Path | list[Path | str] | None = None,
    ac_coords_vox: Vector2d | None = None,
    pc_coords_vox: Vector2d | None = None,
    vox2ras: AffineMatrix4x4 | None = None,
    title: str = "",
) -> None:
    """Creates a figure of the contours (shape) and the subdivisions of the corpus callosum.

    Parameters
    ----------
    slice_or_slab : np.ndarray
        Intensities of the current slice, midslice or midslab (will plot middle slice).
    split_contours : list[np.ndarray], optional
        List of contour arrays for each subdivision (ignore contours on None) in upright AS coordinates each with shape
        (N, 2).
    midline_equidistant : np.ndarray, optional
        Midline points at equidistant spacing (ignore midline on None) in upright AS coordinates with shape (2, N).
    levelpaths : list[np.ndarray], optional
        List of level paths for visualization (ignore level paths on None) in upright AS coordinates each with shape
        (2, N).
    output_path : str or Path or list of Paths, optional
        Path to save the plot (show and do not save on None).
    ac_coords_vox : np.ndarray, optional
        AC coordinates for visualization (ignore AC on None) in LIA voxel coordinates.
    pc_coords_vox : np.ndarray, optional
        PC coordinates for visualization (ignore PC on None) in LIA voxel coordinates.
    vox2ras : AffineMatrix4x4, optional
        Slice vox2ras transformation matrix.
    title : str, default=""
        Title for the plot.

    Notes
    -----
    Creates a visualization of the corpus callosum contours and their subdivisions.
    If output_path is provided, saves the plot to that location.
    """
    from functools import partial

    from nibabel.affines import apply_affine

    from FastSurferCNN.utils.plotting import backend

    if vox2ras is None and None in (split_contours, midline_equidistant, levelpaths):
        raise ValueError("vox_size must be provided if split_contours, midline_equidistant, or levelpaths are given.")
    
    _backend_context = nullcontext if output_path is None else partial(backend, 'agg')  # Use non-GUI backend

    # convert vox_size from LIA to AS
    ras2vox = partial(apply_affine, np.linalg.inv(vox2ras)[1:, 1:])

    # scale contour data by vox_size to convert from AS to AS-aligned voxel space
    _split_contours = [] if split_contours is None else [ras2vox(sp.T).T for sp in split_contours]
    _midline_equi = np.zeros((0, 2)) if midline_equidistant is None else ras2vox(midline_equidistant)
    _levelpaths = [] if levelpaths is None else [ras2vox(lp) for lp in levelpaths]

    has_first_plot = not (len(_split_contours) == 0 and ac_coords_vox is None and pc_coords_vox is None)
    num_plots = 1 + int(has_first_plot)

    with _backend_context():
        # import here to have the correct backend set for non-GUI environments
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, num_plots, sharex=True, sharey=True, figsize=(15, 10))

        # NOTE: For all plots imshow shows y inverted
        current_plot = 0

        if _split_contours:
            reference_contour = _split_contours[-1]

        # This visualization uses voxel coordinates in fsaverage space...
        if has_first_plot:
            ax[current_plot].imshow(slice_or_slab[slice_or_slab.shape[0] // 2], cmap="gray")
            ax[current_plot].set_title(title)
        if _split_contours:
            for this_contour in _split_contours:
                ax[current_plot].fill(this_contour[1, :], this_contour[0, :], color="steelblue", alpha=0.25)
                kwargs = {"color": "mediumblue", "linewidth": 0.7, "linestyle": "solid"}
                ax[current_plot].plot(this_contour[1, :], this_contour[0, :], **kwargs)
        if ac_coords_vox is not None:
            ax[current_plot].scatter(ac_coords_vox[1], ac_coords_vox[0], color="red", marker="x")
        if pc_coords_vox is not None:
            ax[current_plot].scatter(pc_coords_vox[1], pc_coords_vox[0], color="blue", marker="x")
        current_plot += int(has_first_plot)

        ax[current_plot].imshow(slice_or_slab[slice_or_slab.shape[0] // 2], cmap="gray")
        for this_path in _levelpaths:
            ax[current_plot].plot(this_path[:, 1], this_path[:, 0], color="brown", linewidth=0.8)
        ax[current_plot].set_title("Midline & Levelpaths")
        if _midline_equi.shape[0] > 0:
            ax[current_plot].plot(_midline_equi[:, 1], _midline_equi[:, 0], color="red")
        if _split_contours:
            ax[current_plot].plot(reference_contour[1, :], reference_contour[0, :], color="red", linewidth=0.5)

        padding = 30
        for a in ax.flatten():
            a.set_aspect("equal", adjustable="box")
            a.axis("off")
            if _split_contours:
                # get bounding box of contours
                a.set_xlim(reference_contour[1, :].min() - padding, reference_contour[1, :].max() + padding)
                a.set_ylim((reference_contour[0, :]).max() + padding, (reference_contour[0, :]).min() - padding)

        if output_path is None:
            return plt.show()
        for _output_path in (output_path if isinstance(output_path, list | tuple) else [output_path]):
            Path(_output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(_output_path, dpi=300, bbox_inches="tight")
    return None
