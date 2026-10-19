import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from neuroreg import LTA

from CorpusCallosum.data.constants import CC_TEMPLATE_METADATA_FILENAME
from CorpusCallosum.data.fsaverage_cc_template import load_fsaverage_cc_template
from CorpusCallosum.shape.contour import CCContour, contours_for_analysis_width
from CorpusCallosum.shape.mesh import CCMesh
from CorpusCallosum.utils.colormaps import PREDEFINED_COLORMAPS, get_colormap
from FastSurferCNN.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def colormap_name(value: str) -> str:
    """Validate a command-line colormap name."""
    try:
        get_colormap(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def nonnegative_float(value: str) -> float:
    """Parse a non-negative command-line float."""
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("Threshold must be a finite, non-negative value.")
    return parsed


def make_parser() -> argparse.ArgumentParser:
    """Create a command line parser for the visualization pipeline."""
    parser = argparse.ArgumentParser(description="Visualize corpus callosum from template files.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--template_dir",
        type=str,
        help=(
            "Path to a template directory containing per-slice files named "
            "thickness_values_<idx>.txt, and optionally contour_<idx>.txt "
            "and thickness_measurement_points_<idx>.txt. If contour_<idx>.txt "
            "and thickness_measurement_points_<idx>.txt are not provided, "
            "uses fsaverage template. For FreeSurfer surfaces in orig.mgz "
            "reference space, also provide mri/orig.mgz and "
            "mri/transforms/cc_up.lta in this directory."
        ),
        metavar="TEMPLATE_DIR",
        default=None,
    )
    input_group.add_argument(
        "--values_file",
        type=str,
        default=None,
        metavar="VALUES_FILE",
        help=(
            "One-column CSV containing values ordered anterior to posterior. "
            "The first row is treated as a header. Uses the bundled fsaverage "
            "contour and generates a 2D plot; no template directory is required."
        ),
    )
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Directory for output files. Writes: "
                             "cc_mesh.html - Interactive 3D mesh visualization (HTML file) "
                             "midslice_2d.png - 2D midslice visualization of the corpus callosum "
                             "cc_mesh.vtk - VTK mesh file format "
                             "cc_mesh.fssurf - FreeSurfer surface file "
                             "cc_mesh_overlay.curv - FreeSurfer curvature overlay file "
                             "cc_mesh_snap.png - Screenshot/snapshot of the 3D mesh (requires whippersnappy>=2.1). "
                             "If template_dir does not contain orig.mgz and cc_up.lta, "
                             "<output_dir>/mri/upright.mgz is used as the fallback reference when available; "
                             "otherwise FreeSurfer surfaces are written without a reference space.",
                        metavar="OUTPUT_DIR"
                        )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Legacy spacing in mm used when template contour files do not store slice positions.",
        metavar="RESOLUTION"
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=5,
        help="Window size for smoothing the contour.",
        metavar="SMOOTHING_WINDOW"
    )
    parser.add_argument(
        "--colormap",
        type=colormap_name,
        default="red_to_yellow",
        metavar="COLORMAP",
        help=(
            "Colormap progression from lower to higher values. "
            f"Predefined options: {', '.join(PREDEFINED_COLORMAPS)}. "
            "Any registered Matplotlib colormap name is also accepted for PNG/Plotly HTML. "
            "WhipperSnapPy snapshots use native heat colors, matching the diverging preset."
        ),
    )
    parser.add_argument(
        "--color_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        required=False,
        help=("Specify colorbar limits (min max); defaults to automatic choice. "
              "Native snapshots support symmetric limits (-M M) only."),
    )
    parser.add_argument(
        "--legend",
        type=str,
        default=None,
        help="Override the colorbar label and native snapshot caption.",
        metavar="LEGEND")
    parser.add_argument(
        "--mode",
        choices=["thickness", "p-value", "icc"],
        default="thickness",
        help="Value type for --values_file (default: thickness).",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Use logarithmic color normalization for --values_file.",
    )
    parser.add_argument(
        "--upper_threshold",
        type=float,
        default=None,
        metavar="VALUE",
        help="Color values above this limit with --threshold_color.",
    )
    parser.add_argument(
        "--threshold_color",
        default="gray",
        help="Matplotlib color for values above --upper_threshold (default: gray).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title for a 2D values plot.",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        metavar="FILENAME",
        help="Filename for the 2D PNG within --output_dir.",
    )
    parser.add_argument(
        "--threshold",
        type=nonnegative_float,
        default=None,
        metavar="VALUE",
        help=("Show values with absolute magnitude below VALUE in neutral gray. "
              "Native snapshots also rescale colors above this absolute-value cutoff."),
    )
    parser.add_argument(
        "--twoD",
        action="store_true",
        help="Generate 2D visualization instead of 3D mesh.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log debug output as well.",
    )
    return parser


def options_parse() -> argparse.Namespace:
    """Parse command line arguments for the pipeline."""
    parser = make_parser()
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args


def load_plot_values(values_file: str | Path) -> np.ndarray:
    """Load a one-column CSV with one header row."""
    values = np.loadtxt(values_file, delimiter=",", skiprows=1, dtype=float, ndmin=2)
    if values.shape[1] != 1:
        raise ValueError(f"Values file must contain exactly one column, found shape {values.shape}")
    values = values[:, 0]
    if len(values) < 2:
        raise ValueError("Values file must contain at least two values")
    return values


def _indexed_thickness_files(template_dir: Path) -> list[tuple[int, Path]]:
    """Return valid thickness files sorted by their numeric slice suffix."""
    indexed_files: list[tuple[int, Path]] = []
    for thickness_file in template_dir.glob("thickness_values_*.txt"):
        try:
            idx = int(thickness_file.stem.split("_")[-1])
        except ValueError:
            continue
        indexed_files.append((idx, thickness_file))
    return sorted(indexed_files, key=lambda item: item[0])


def _load_template_metadata(template_dir: Path) -> dict | None:
    """Load optional metadata written by the FastSurfer CC pipeline."""
    metadata_path = template_dir / CC_TEMPLATE_METADATA_FILENAME
    if not metadata_path.exists():
        return None
    with metadata_path.open() as file:
        metadata = json.load(file)
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported CC template metadata schema in {metadata_path}")
    return metadata


def _template_midslice_position(template_dir: Path, two_d: bool) -> int:
    """Return the selected contour position and reject incomplete 3D templates."""
    indices = [idx for idx, _ in _indexed_thickness_files(template_dir)]
    metadata = _load_template_metadata(template_dir)
    if metadata is None:
        return len(indices) // 2

    requested = [int(idx) for idx in metadata["requested_slice_indices"]]
    successful = [int(idx) for idx in metadata["successful_slice_indices"]]
    selected = int(metadata["selected_slice_index"])
    if successful != indices:
        raise ValueError(
            "CC template metadata does not match the available thickness files: "
            f"metadata lists {successful}, files contain {indices}."
        )
    if selected not in indices:
        raise ValueError(
            f"The selected anatomical slice {selected} is missing from this incomplete CC template; "
            "a neighboring slice will not be substituted."
        )
    if not two_d and successful != requested:
        missing = [idx for idx in requested if idx not in successful]
        raise ValueError(
            f"Cannot create a full-slab 3D visualization from an incomplete CC template; missing slices: {missing}."
        )
    return indices.index(selected)


def load_contours_from_template_dir(
        template_dir: Path, resolution: float, smoothing_window: int
) -> list[CCContour]:
    """Load all contours and thickness data from a template directory."""
    indexed_thickness_files = _indexed_thickness_files(template_dir)

    if not indexed_thickness_files:
        raise FileNotFoundError(
            f"No thickness files found in template directory {template_dir}. "
            "Expected files named thickness_values_<idx>.txt and "
            "optionally contour_<idx>.txt and thickness_measurement_points_<idx>.txt."
        )

    fsaverage_contour = None
    contours: list[CCContour] = []
    indices = [idx for idx, _ in indexed_thickness_files]
    metadata = _load_template_metadata(template_dir)
    stored_positions = {} if metadata is None else {
        int(idx): float(position) for idx, position in metadata.get("slice_positions_mm", {}).items()
    }

    # Calculate lr_positions centered around the middle slice
    num_slices = len(indices)
    middle_idx = num_slices // 2

    for idx, thickness_file in indexed_thickness_files:
        # Calculate lr_position: use the index offset from middle, scaled by resolution
        lr_position = stored_positions.get(idx, (idx - indices[middle_idx]) * resolution)

        contour_file = template_dir / f"contour_{idx}.txt"

        if not contour_file.exists():
            # get length of thickness values
            thickness_values = np.loadtxt(thickness_file, dtype=str)
            # get the non nan thickness values (excluding header), so we know how many points to sample
            num_thickness_values = np.sum(~np.isnan(np.array(thickness_values[1:], dtype=float)))
            if fsaverage_contour is None:
                fsaverage_contour = load_fsaverage_cc_template()
                # create measurement points (points = 2 x levelpaths) according to number of thickness values
                fsaverage_contour.create_levelpaths(num_points=num_thickness_values // 2, inplace=True)
            current_contour = fsaverage_contour.copy()
            current_contour.lr_position = lr_position
            current_contour.load_thickness_values(thickness_file)

        else:
            current_contour = CCContour.from_contour_file(contour_file, thickness_file, lr_position=lr_position)

        if smoothing_window > 0:
            current_contour.smooth_contour(window_size=smoothing_window)

        current_contour.fill_thickness_values()
        contours.append(current_contour)

    if not contours:
        raise ValueError(f"No valid contours could be loaded from {template_dir}")
    return contours


def _upright_reference_from_template(
        template_dir: Path,
        output_dir: Path,
) -> tuple[nib.spatialimages.SpatialHeader | None, np.ndarray | None]:
    """Return the display header and RAS-to-voxel transform for CC mesh output.

    ``CCMesh.from_contours`` creates vertices in the CC upright/fsaverage RAS
    coordinate system. FreeSurfer surface files, however, store vertices in
    surface RAS (tkRAS). Lapy's ``write_fssurf(..., image=header)`` performs
    that voxel-to-tkRAS conversion and stamps the surface with the supplied
    header's volume_info. Therefore callers must first convert mesh RAS to voxel
    coordinates in the same volume whose header is passed to ``write_fssurf``.

    For Freeview inspection with orig.mgz, the preferred path uses orig.mgz plus
    ``cc_up.lta``. The LTA maps orig scanner RAS to upright/fsaverage RAS, so
    ``inv(LTA @ orig.affine)`` maps mesh RAS directly to orig voxel coordinates.
    If only an upright volume is available, the fallback writes a surface for
    that upright volume instead.
    """
    orig_image = template_dir / "mri" / "orig.mgz"
    upright_lta = template_dir / "mri" / "transforms" / "cc_up.lta"
    if orig_image.exists() and upright_lta.exists():
        image = nib.load(orig_image)
        upright_vox2ras = LTA.read(upright_lta).r2r() @ image.affine
        return image.header, np.linalg.inv(upright_vox2ras)

    upright_image = output_dir / "mri" / "upright.mgz"
    if upright_image.exists():
        image = nib.load(upright_image)
        return image.header, np.linalg.inv(image.affine)

    return None, None


def main(
        template_dir: str | Path | None,
        output_dir: str | Path,
        resolution: float = 1.0,
        smoothing_window: int = 5,
        colormap: str = "red_to_yellow",
        color_range: tuple[float, float] | None = None,
        legend: str | None = None,
        twoD: bool = False,
        values_file: str | Path | None = None,
        mode: str = "thickness",
        log_scale: bool = False,
        upper_threshold: float | None = None,
        threshold_color: str = "gray",
        title: str | None = None,
        output_name: str | None = None,
        threshold: float | None = None,
) -> Literal[0] | str:
    """Visualize corpus callosum templates in 2D or 3D."""
    if values_file is None and mode != "thickness":
        raise ValueError("--mode is only supported with --values_file")
    if values_file is None and (log_scale or upper_threshold is not None):
        raise ValueError("--log_scale and --upper_threshold require --values_file")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    color_range = tuple(color_range) if color_range is not None else None

    if values_file is not None:
        plot_values = load_plot_values(values_file)
        contour = load_fsaverage_cc_template()
        if smoothing_window > 0:
            contour.smooth_contour(window_size=smoothing_window)
        output_name = output_name or f"cc_{mode.replace('-', '_')}_2d.png"
        output_path = output_dir / output_name
        logger.info(f"Writing output to {output_path}")
        figure = contour.plot_contour_colorfill(
            plot_values=plot_values,
            title=title,
            save_path=str(output_path),
            colorbar=True,
            mode=mode,
            colormap=colormap,
            log_scale=log_scale,
            upper_threshold=upper_threshold,
            threshold_color=threshold_color,
            colorbar_label=legend,
            color_range=color_range,
            threshold=threshold,
        )
        plt.close(figure)
        return 0

    if template_dir is None:
        raise ValueError("template_dir is required when values_file is not provided")
    template_dir = Path(template_dir)

    contours = load_contours_from_template_dir(
        template_dir, resolution=resolution, smoothing_window=smoothing_window,
    )

    # 2D visualization
    mid_contour = contours[_template_midslice_position(template_dir, twoD)]

    logger.info(f"Writing output to {output_dir / 'cc_thickness_2d.png'}")

    raw_thickness_values = mid_contour.get_thickness_profile()
    figure = mid_contour.plot_contour_colorfill(
        plot_values=raw_thickness_values,
        title=None,
        save_path=str(output_dir / "cc_thickness_2d.png"),
        colorbar=True,
        mode="thickness",
        colormap=colormap,
        color_range=color_range,
        legend=legend,
        threshold=threshold,
    )
    plt.close(figure)
    if twoD:
        return 0
    if len(contours) < 2:
        logger.warning(
            f"Skipping 3D outputs: the template directory contains only {len(contours)} slice, but at least 2 slices "
            "are required for a surface (save the template with --slice_selection all).",
        )
        return 0

    # 3D visualization
    surface_mesh = CCMesh.from_contours(contours_for_analysis_width(contours), smooth=0)
    header, mesh_ras2vox = _upright_reference_from_template(template_dir, output_dir)

    if mesh_ras2vox is not None:
        # Convert from CC upright/fsaverage RAS into the voxel coordinates of
        # the same volume whose header is passed below.
        # write_fssurf/snap_cc_picture then use the header's vox2ras-tkr to
        # write/display vertices in the FreeSurfer coordinate frame expected by
        # Freeview. Skipping this step treats millimeter RAS coordinates as voxel
        # indices and shifts the surface far away from the MRI.
        freeview_mesh = surface_mesh.to_vox_coordinates(mesh_ras2vox)
    else:
        logger.warning(
            "Writing FreeSurfer surface outputs without a reference space. "
            "The surface vertices remain in CC upright RAS coordinates and may not align with an MRI in Freeview. "
            "Provide template_dir/mri/orig.mgz with template_dir/mri/transforms/cc_up.lta, "
            "or output_dir/mri/upright.mgz, to write the surface in a reference space.",
        )
        freeview_mesh = surface_mesh

    plot_kwargs = dict(
        colormap=colormap,
        color_range=color_range,
        thickness_overlay=True,
        legend=legend or "",
        threshold=threshold,
    )
    surface_mesh.plot_mesh(**plot_kwargs)
    surface_mesh.plot_mesh(output_path=str(output_dir / "cc_mesh.html"), **plot_kwargs)

    logger.info(f"Writing vtk file to {output_dir / 'cc_mesh.vtk'}")
    surface_mesh.write_vtk(str(output_dir / "cc_mesh.vtk"))
    logger.info(f"Writing freesurfer surface file to {output_dir / 'cc_mesh.fssurf'}")
    freeview_mesh.write_fssurf(str(output_dir / "cc_mesh.fssurf"), image=header)
    logger.info(f"Writing freesurfer overlay file to {output_dir / 'cc_mesh_overlay.curv'}")
    surface_mesh.write_morph_data(str(output_dir / "cc_mesh_overlay.curv"))
    freeview_mesh.snap_cc_picture(
        str(output_dir / "cc_mesh_snap.png"), ref_header=header,
        colormap=colormap, color_range=color_range, legend=legend, threshold=threshold,
    )
    logger.info(f"Writing 3D snapshot image to {output_dir / 'cc_mesh_snap.png'}")
    return 0


if __name__ == "__main__":
    options = options_parse()

    # INFO by default, as in fastsurfer_cc.py; this tool has no log file, only stdout
    setup_logging(None, "DEBUG" if options.verbose else None)

    sys.exit(main(
        template_dir=options.template_dir,
        output_dir=options.output_dir,
        resolution=options.resolution,
        smoothing_window=options.smoothing_window,
        colormap=options.colormap,
        color_range=options.color_range,
        legend=options.legend,
        threshold=options.threshold,
        twoD=options.twoD,
        values_file=options.values_file,
        mode=options.mode,
        log_scale=options.log_scale,
        upper_threshold=options.upper_threshold,
        threshold_color=options.threshold_color,
        title=options.title,
        output_name=options.output_name,
    ))
