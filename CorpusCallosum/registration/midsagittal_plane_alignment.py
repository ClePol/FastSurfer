from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np
from neuroreg.segreg import segreg
from scipy.ndimage import affine_transform, binary_fill_holes, distance_transform_edt
from skimage.morphology import convex_hull_image

from CorpusCallosum.data.constants import FSAVERAGE_MIDPLANE_MM, FSAVERAGE_REGISTRATION_LABELS, FSAVERAGE_TARGET_PATH
from CorpusCallosum.data.read_write import MGHHeaderDict, convert_numpy_to_json_serializable, init_mgh_header
from CorpusCallosum.registration.landmarks import (
    _affine_from_rotation_and_center,
    _rotation_from_vectors,
    adjust_midplane_to_landmarks,
)
from CorpusCallosum.shape.postprocessing import offset_affine
from FastSurferCNN.utils import AffineMatrix4x4, Shape3d, logging, nibabelImage
from FastSurferCNN.utils.brainvolstats import hemi_masks_from_aseg

logger = logging.get_logger(__name__)

# FreeSurfer aseg left/right label pairs used for LR symmetry scoring.
# Restricted to subcortical structures that are clearly unilateral (present on one side
# only in any given yz-slab).
_ASEG_LR_PAIRS: tuple[tuple[int, int], ...] = (
    (2, 41),  # Cerebral-White-Matter
    (4, 43),  # Lateral-Ventricle
    (5, 44),  # Inf-Lat-Vent
    (10, 49),  # Thalamus
    (11, 50),  # Caudate
    (12, 51),  # Putamen
    (13, 52),  # Pallidum
    (17, 53),  # Hippocampus
    (18, 54),  # Amygdala
    (26, 58),  # Accumbens-Area
    (28, 60),  # VentralDC
)


@dataclass(frozen=True)
class MidplaneRefinementResult:
    """Container for distance-map refinement outputs.

    Attributes
    ----------
    updated_vox2vox : AffineMatrix4x4
        Updated subject-to-fsaverage voxel transform.
    center_shift_vox : float
        Estimated LR center shift in fsaverage voxels.
    diagnostics : dict[str, object]
        JSON-serializable diagnostics and rejection reasons.
    """

    updated_vox2vox: AffineMatrix4x4
    center_shift_vox: float
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class MidplaneTransformResult:
    """Result returned by :func:`find_midplane_transform`.

    Attributes
    ----------
    orig2fsavg_vox2vox : AffineMatrix4x4
        Subject-to-fsaverage voxel transform after optional refinement.
    fsavg_vox2ras : AffineMatrix4x4
        fsaverage voxel-to-RAS transform used for downstream mapping.
    fsavg_header_dict : MGHHeaderDict
        Header metadata for the fsaverage-aligned volume.
    fsavg_shape : Shape3d
        Shape of the fsaverage-aligned target grid.
    base_middle_vox : float
        Baseline midsagittal x-coordinate in fsaverage voxel units.
    midline_shift_vox : float
        Applied LR shift (or estimated center shift) in voxel units.
    midline_shift_diagnostics : dict[str, object]
        Method-specific diagnostics for refinement decisions.
    """

    orig2fsavg_vox2vox: AffineMatrix4x4
    fsavg_vox2ras: AffineMatrix4x4
    fsavg_header_dict: MGHHeaderDict
    fsavg_shape: Shape3d
    base_middle_vox: float
    midline_shift_vox: float
    midline_shift_diagnostics: dict[str, object]


def resample_segmentation_to_fsavg(
        seg_data: np.ndarray,
        orig2fsavg_vox2vox: AffineMatrix4x4,
        fsavg_shape: Shape3d,
) -> np.ndarray:
    """Resample a segmentation into fsaverage voxel space.

    Parameters
    ----------
    seg_data : np.ndarray
        Input segmentation array in subject voxel space.
    orig2fsavg_vox2vox : AffineMatrix4x4
        Subject-to-fsaverage voxel transform.
    fsavg_shape : Shape3d
        Output shape in fsaverage voxel coordinates.

    Returns
    -------
    np.ndarray
        Resampled integer segmentation in fsaverage space.

    Notes
    -----
    Nearest-neighbor interpolation is used to preserve discrete labels.
    """
    return affine_transform(
        seg_data.astype(np.int32),
        np.linalg.inv(orig2fsavg_vox2vox),
        output_shape=fsavg_shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(np.int32)


def _make_fsavg_convex_hull_mask(brain_mask: np.ndarray) -> np.ndarray:
    """Approximate a whole-brain hull mask in fsaverage voxel space.

    Parameters
    ----------
    brain_mask : np.ndarray
        Binary brain mask in fsaverage space.

    Returns
    -------
    np.ndarray
        Boolean hull mask spanning valid x extents for each y-z location.
    """
    yz_projection = brain_mask.any(axis=0)
    yz_hull = convex_hull_image(yz_projection)

    x_min = np.full(yz_projection.shape, brain_mask.shape[0], dtype=np.int32)
    x_max = np.full(yz_projection.shape, -1, dtype=np.int32)
    x_coords, y_coords, z_coords = np.where(brain_mask)
    np.minimum.at(x_min, (y_coords, z_coords), x_coords)
    np.maximum.at(x_max, (y_coords, z_coords), x_coords)

    valid_yz = x_max >= x_min
    hull_valid = yz_hull & valid_yz
    if not hull_valid.any():
        return brain_mask.copy()

    nearest_valid = distance_transform_edt(~valid_yz, return_distances=False, return_indices=True)
    hull_mask = np.zeros_like(brain_mask, dtype=bool)
    for y_idx, z_idx in np.argwhere(yz_hull):
        if valid_yz[y_idx, z_idx]:
            x0 = int(x_min[y_idx, z_idx])
            x1 = int(x_max[y_idx, z_idx])
        else:
            src_y = int(nearest_valid[0, y_idx, z_idx])
            src_z = int(nearest_valid[1, y_idx, z_idx])
            x0 = int(x_min[src_y, src_z])
            x1 = int(x_max[src_y, src_z])
        if x1 >= x0:
            hull_mask[x0: x1 + 1, y_idx, z_idx] = True
    return hull_mask


def _clean_partitioned_hemi_masks(
        seg_fsavg: np.ndarray,
        sampling: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate robust, hole-free left/right hemisphere masks.

    Parameters
    ----------
    seg_fsavg : np.ndarray
        Segmentation labels in fsaverage voxel space.
    sampling : np.ndarray, optional
        Relative voxel spacing per axis for the distance transforms (default: isotropic).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Left and right boolean masks that jointly cover the brain mask.
    """
    brain_mask = seg_fsavg > 0
    left_vote, right_vote = hemi_masks_from_aseg(seg_fsavg)
    left_core = np.logical_and(left_vote, brain_mask)
    right_core = np.logical_and(right_vote, brain_mask)

    undecided = np.logical_and(brain_mask, np.logical_not(np.logical_or(left_core, right_core)))
    left_core_dt = distance_transform_edt(np.logical_not(left_core), sampling=sampling)
    right_core_dt = distance_transform_edt(np.logical_not(right_core), sampling=sampling)
    left_mask = np.logical_or(left_core, np.logical_and(undecided, left_core_dt <= right_core_dt))
    right_mask = np.logical_and(brain_mask, np.logical_not(left_mask))

    left_mask = np.logical_and(binary_fill_holes(left_mask), brain_mask)
    right_mask = np.logical_and(binary_fill_holes(right_mask), brain_mask)

    overlap = np.logical_and(left_mask, right_mask)
    if np.any(overlap):
        left_overlap = left_core_dt <= right_core_dt
        left_mask = np.logical_or(
            np.logical_and(left_mask, np.logical_not(overlap)),
            np.logical_and(overlap, left_overlap),
        )
        right_mask = np.logical_and(brain_mask, np.logical_not(left_mask))

    remainder = np.logical_and(brain_mask, np.logical_not(np.logical_or(left_mask, right_mask)))
    if np.any(remainder):
        left_mask = np.logical_or(left_mask, np.logical_and(remainder, left_core_dt <= right_core_dt))
        right_mask = np.logical_and(brain_mask, np.logical_not(left_mask))

    return left_mask, right_mask


def _prepare_lr_pair_labels(seg_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create matched left/right aseg label arrays available in an input segmentation.

    Parameters
    ----------
    seg_data : np.ndarray
        FreeSurfer-style labeled segmentation volume.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of matched left and right label IDs present in ``seg_data``.
    """
    label_set = set(np.unique(seg_data.astype(np.int32)).tolist())
    label_set.discard(0)

    left_ids = [lbl for lbl, r in _ASEG_LR_PAIRS if lbl in label_set and r in label_set]
    right_ids = [r for lbl, r in _ASEG_LR_PAIRS if lbl in label_set and r in label_set]

    return np.asarray(left_ids, dtype=np.int32), np.asarray(right_ids, dtype=np.int32)


def _score_midline_shift(
        seg_in_fsavg: np.ndarray,
        left_ids: np.ndarray,
        right_ids: np.ndarray,
        base_middle_vox: float,
        shift_vox: int,
        max_pairs_per_slice: int = 64,
        slab_half_vox: int = 12,
) -> tuple[float, int]:
    """Score a candidate LR shift by mirrored label consistency near the midline.

    Parameters
    ----------
    seg_in_fsavg : np.ndarray
        Segmentation labels in fsaverage voxel space.
    left_ids : np.ndarray
        Label IDs considered left-hemisphere structures.
    right_ids : np.ndarray
        Label IDs considered right-hemisphere structures.
    base_middle_vox : float
        Baseline midsagittal x-coordinate in voxel units.
    shift_vox : int
        Candidate shift to evaluate relative to ``base_middle_vox``.
    max_pairs_per_slice : int, default=64
        Maximum sampled right-side voxels per z-slice for distance matching.
    slab_half_vox : int, default=12
        Half-width of the LR slab around the candidate midline that is compared, in voxels.

    Returns
    -------
    tuple[float, int]
        Median mirrored y-distance score and number of matched samples.
        Lower scores indicate better LR consistency.
    """
    if left_ids.size == 0 or right_ids.size == 0:
        return float("inf"), 0

    x_mid = int(np.round(base_middle_vox + shift_vox))
    if x_mid <= 1 or x_mid >= seg_in_fsavg.shape[0] - 2:
        return float("inf"), 0

    x0 = max(0, x_mid - slab_half_vox)
    x1 = min(seg_in_fsavg.shape[0], x_mid + slab_half_vox + 1)
    slab = seg_in_fsavg[x0:x1]

    left_mask = np.isin(slab, left_ids)
    right_mask = np.isin(slab, right_ids)
    if not left_mask.any() or not right_mask.any():
        return float("inf"), 0

    distances: list[float] = []
    for z_idx in range(slab.shape[2]):
        left_slice = left_mask[:, :, z_idx]
        right_slice = right_mask[:, :, z_idx]
        if not left_slice.any() or not right_slice.any():
            continue

        right_x, right_y = np.where(right_slice)
        if right_x.size == 0:
            continue

        if right_x.size > max_pairs_per_slice:
            sample_idx = np.linspace(0, right_x.size - 1, max_pairs_per_slice).astype(int)
            right_x = right_x[sample_idx]
            right_y = right_y[sample_idx]

        for rx, ry in zip(right_x, right_y, strict=False):
            mirror_x = int(np.clip(2 * (x_mid - x0) - rx, 0, slab.shape[0] - 1))
            left_row = left_slice[mirror_x]
            if not left_row.any():
                continue
            left_y = np.where(left_row)[0]
            distances.append(float(np.min(np.abs(left_y - ry))))

    if not distances:
        return float("inf"), 0

    return float(np.median(distances)), len(distances)


def refine_midline_lr_shift(
        orig2fsavg_vox2vox: AffineMatrix4x4,
        aseg_data: np.ndarray,
        fsavg_shape: Shape3d,
        base_middle_vox: float,
        max_shift_mm: float = 6.0,
        step_vox: int = 1,
        slab_half_mm: float = 12.0,
        min_improvement_mm: float = 0.5,
        vox_size: np.ndarray | None = None,
) -> tuple[AffineMatrix4x4, int, float, dict[str, object]]:
    """Refine midsagittal alignment by searching small LR translations.

    Parameters
    ----------
    orig2fsavg_vox2vox : AffineMatrix4x4
        Initial subject-to-fsaverage voxel transform.
    aseg_data : np.ndarray
        Subject segmentation labels in native voxel space.
    fsavg_shape : Shape3d
        Shape of the fsaverage target grid.
    base_middle_vox : float
        Baseline midsagittal x-coordinate in fsaverage voxels.
    max_shift_mm : float, default=6.0
        Maximum absolute LR shift explored in mm (rounded to full voxels).
    step_vox : int, default=1
        Integer step size for the LR search.
    slab_half_mm : float, default=12.0
        Half-width of the LR slab around the candidate midline that is compared, in mm (rounded to full voxels).
    min_improvement_mm : float, default=0.5
        Minimum improvement of the score (a median distance in mm) over the unshifted midline to accept a shift.
    vox_size : np.ndarray, optional
        Voxel size of the fsaverage grid per voxel axis (default: 1 mm isotropic). The thresholds are converted from
        mm to voxels, so the behavior is consistent across resolutions (and unchanged for 1 mm grids).

    Returns
    -------
    tuple[AffineMatrix4x4, int, float, dict[str, object]]
        Updated transform, selected shift, selected score, and diagnostics.

    Notes
    -----
    Candidate shifts are regularized and validated with conservative rejection
    criteria (boundary hits, low support, and weak improvements over baseline).
    """
    scale = np.ones(3, dtype=float) if vox_size is None else np.asarray(vox_size, dtype=float)

    def _mm_to_vox(mm: float, axis: int) -> int:
        # round half up with a tolerance for float noise in the voxel size (e.g. 6 mm / 0.8 mm -> 8 voxels)
        return int(np.floor(mm / scale[axis] + 0.5 + 1e-6))

    max_shift_vox = _mm_to_vox(max_shift_mm, 0)
    slab_half_vox = _mm_to_vox(slab_half_mm, 0)
    if max_shift_vox < 0 or step_vox <= 0:
        return orig2fsavg_vox2vox, 0, float("inf"), {}

    left_ids, right_ids = _prepare_lr_pair_labels(aseg_data)
    if left_ids.size == 0:
        logger.warning("Midline refinement skipped: no left/right label pairs found.")
        return orig2fsavg_vox2vox, 0, float("inf"), {}

    seg_fsavg = resample_segmentation_to_fsavg(aseg_data, orig2fsavg_vox2vox, fsavg_shape)

    candidate_scores: list[dict[str, int | float | None]] = []
    zero_shift_score = float("inf")
    zero_shift_support = 0
    best_shift = 0
    best_adjusted_score = float("inf")
    best_raw_score = float("inf")
    best_support = 0
    # scores are median IS distances in voxels of axis 1, shifts are in voxels of axis 0; the penalty is 0.25 mm of
    # score per mm of shift, and the minimal improvement is converted to voxels of axis 1
    shift_penalty = float(0.25 * scale[0] / scale[1])
    min_improvement = float(min_improvement_mm / scale[1])
    min_support = 32

    for shift in range(-max_shift_vox, max_shift_vox + 1, step_vox):
        raw_score, support = _score_midline_shift(
            seg_fsavg,
            left_ids,
            right_ids,
            base_middle_vox=base_middle_vox,
            shift_vox=-shift,
            slab_half_vox=slab_half_vox,
        )
        adjusted_score = raw_score + shift_penalty * abs(shift)
        candidate_scores.append(
            {
                "shift_vox": int(shift),
                "raw_score": float(raw_score) if np.isfinite(raw_score) else None,
                "adjusted_score": float(adjusted_score) if np.isfinite(adjusted_score) else None,
                "support": int(support),
            }
        )
        if shift == 0:
            zero_shift_score = raw_score
            zero_shift_support = support
        if adjusted_score < best_adjusted_score:
            best_adjusted_score = adjusted_score
            best_raw_score = raw_score
            best_shift = shift
            best_support = support

    diagnostics: dict[str, object] = {
        "candidate_scores": candidate_scores,
        "zero_shift_score": float(zero_shift_score) if np.isfinite(zero_shift_score) else None,
        "zero_shift_support": int(zero_shift_support),
        "selected_shift_raw_score": float(best_raw_score) if np.isfinite(best_raw_score) else None,
        "selected_shift_adjusted_score": float(best_adjusted_score) if np.isfinite(best_adjusted_score) else None,
        "selected_shift_support": int(best_support),
        "shift_penalty": shift_penalty,
        "min_improvement": min_improvement,
        "min_improvement_mm": float(min_improvement_mm),
        "min_support": min_support,
        "max_shift_vox": max_shift_vox,
        "max_shift_mm": float(max_shift_mm),
        "slab_half_vox": slab_half_vox,
        "slab_half_mm": float(slab_half_mm),
    }

    at_boundary = abs(best_shift) == max_shift_vox
    insufficient_support = best_support < min_support
    no_baseline = not np.isfinite(zero_shift_score)
    small_improvement = (
            np.isfinite(best_raw_score)
            and np.isfinite(zero_shift_score)
            and (zero_shift_score - best_raw_score) < min_improvement
    )
    reject_shift = (
            best_shift == 0
            or not np.isfinite(best_raw_score)
            or at_boundary
            or insufficient_support
            or (not no_baseline and small_improvement)
    )

    diagnostics["no_baseline"] = no_baseline
    diagnostics["rejected_boundary_shift"] = at_boundary
    diagnostics["rejected_low_support"] = insufficient_support
    diagnostics["rejected_small_improvement"] = not no_baseline and small_improvement

    if reject_shift:
        logger.info(
            "Midline refinement applied no LR correction (best_shift=%d, raw_score=%s, zero_score=%s, support=%d).",
            best_shift,
            f"{best_raw_score:.3f}" if np.isfinite(best_raw_score) else "inf",
            f"{zero_shift_score:.3f}" if np.isfinite(zero_shift_score) else "inf",
            best_support,
        )
        return orig2fsavg_vox2vox, 0, zero_shift_score, diagnostics

    updated_transform = offset_affine([best_shift, 0, 0]) @ orig2fsavg_vox2vox
    logger.info(
        "Midline refinement applied %+d vox LR correction in fsaverage space "
        "(raw_score=%.3f, zero_score=%.3f, support=%d).",
        best_shift,
        best_raw_score,
        zero_shift_score,
        best_support,
    )
    return updated_transform, best_shift, best_raw_score, diagnostics


def register_centroids_to_fsavg(
        aseg_nib: nibabelImage,
) -> tuple[AffineMatrix4x4, AffineMatrix4x4, AffineMatrix4x4, MGHHeaderDict]:
    """Estimate a rigid subject-to-fsaverage alignment from aseg centroids.

    Parameters
    ----------
    aseg_nib : nibabelImage
        Input aseg image in subject space.

    Returns
    -------
    tuple[AffineMatrix4x4, AffineMatrix4x4, AffineMatrix4x4, MGHHeaderDict]
        ``(aseg2fsavg_vox2vox, aseg2fsavg_ras2ras, fsavg_hires_vox2ras, fsavg_header)``.
    """
    logger.info("Starting centroid registration")
    registration = segreg(
        aseg_nib,
        centroids=FSAVERAGE_TARGET_PATH,
        dof=6,
        labels=list(FSAVERAGE_REGISTRATION_LABELS),
    )
    if registration.target_geometry is None:
        raise ValueError("Pinned fsaverage target did not provide target geometry.")

    aseg2fsaverage_ras2ras = np.asarray(registration.r2r, dtype=np.float64)
    atlas_header = registration.target_geometry
    fsavg_header: MGHHeaderDict = {
        "dims": [int(v) for v in atlas_header["dims"]],
        "delta": [float(v) for v in atlas_header["delta"]],
        "Mdc": np.asarray(atlas_header["Mdc"], dtype=np.float64).copy(),
        "Pxyz_c": np.asarray(atlas_header["Pxyz_c"], dtype=np.float64).copy(),
    }
    # Downstream CC mapping expects FreeSurfer's MGH/tkregister affine convention.
    fsaverage_vox2ras = np.asarray(init_mgh_header(aseg_nib.header, fsavg_header).get_vox2ras(), dtype=np.float64)

    # Resample the fsaverage grid to the subject resolution: the zooms of the subject image (in RAS order) are
    # permuted into the voxel axis order of fsaverage (LIA), i.e. each fsaverage voxel axis gets the zoom of the RAS
    # axis it points along.
    aseg_zooms_ras = np.asarray(nib.as_closest_canonical(aseg_nib).header.get_zooms()[:3], dtype=np.float64)
    fsavg_axis_to_ras = np.argmax(np.abs(fsaverage_vox2ras[:3, :3]), axis=0)
    fsavg_delta = np.linalg.norm(fsaverage_vox2ras[:3, :3], axis=0)
    hires_delta = aseg_zooms_ras[fsavg_axis_to_ras]
    # The scaling must act on the voxel axes (columns of vox2ras), i.e. right-multiply; voxel 0 (the corner of the
    # fsaverage volume) stays at the same RAS position.
    resolution_trans: AffineMatrix4x4 = np.diagflat(np.append(hires_delta / fsavg_delta, [1])).astype(float)
    fsavg_hires_vox2ras: AffineMatrix4x4 = fsaverage_vox2ras @ resolution_trans

    fsavg_header["delta"] = hires_delta
    fsavg_header["dims"] = (
        np.ceil(np.asarray(fsavg_header["dims"], dtype=np.float64) / np.diag(resolution_trans)[:3])
        .astype(int)
        .tolist()
    )
    # Pxyz_c must be consistent with fsavg_hires_vox2ras (center voxel is at dims / 2 in the MGH convention)
    fsavg_header["Pxyz_c"] = (fsavg_hires_vox2ras @ np.append(np.asarray(fsavg_header["dims"]) / 2, [1]))[:3]

    aseg2fsavg_vox2vox: AffineMatrix4x4 = np.linalg.inv(fsavg_hires_vox2ras) @ aseg2fsaverage_ras2ras @ aseg_nib.affine
    logger.info(
        "Centroid registration successful via neuroreg using %d pinned labels (%d matched)!",
        len(FSAVERAGE_REGISTRATION_LABELS),
        len(registration.labels),
    )
    return aseg2fsavg_vox2vox, aseg2fsaverage_ras2ras, fsavg_hires_vox2ras, fsavg_header


def refine_midplane_with_distance_maps(
        orig2fsavg_vox2vox: AffineMatrix4x4,
        aseg_data: np.ndarray,
        fsavg_shape: Shape3d,
        base_middle_vox: float,
        zero_band_vox: float = 0.5,
        support_distance_mm: float = 24.0,
        fit_band_mm: float = 20.0,
        max_tilt_deg: float = 7.5,
        max_center_shift_mm: float = 8.0,
        vox_size: np.ndarray | None = None,
) -> MidplaneRefinementResult:
    """Fit a midsagittal plane from LR distance-map symmetry in fsaverage space.

    Parameters
    ----------
    orig2fsavg_vox2vox : AffineMatrix4x4
        Initial subject-to-fsaverage voxel transform.
    aseg_data : np.ndarray
        Subject segmentation labels in native voxel space.
    fsavg_shape : Shape3d
        Shape of the fsaverage target grid.
    base_middle_vox : float
        Baseline midsagittal x-coordinate in fsaverage voxels.
    zero_band_vox : float, default=0.5
        Absolute distance-difference threshold for candidate midplane voxels, in LR voxels. This selects a sheet of
        about one voxel thickness around the zero crossing, i.e. it is a sampling parameter and stays in voxels.
    support_distance_mm : float, default=24.0
        Maximum distance to either hemisphere mask for candidate support, in mm.
    fit_band_mm : float, default=20.0
        LR range around baseline center used for plane fitting, in mm.
    max_tilt_deg : float, default=7.5
        Maximum accepted angular tilt away from the LR axis.
    max_center_shift_mm : float, default=8.0
        Maximum accepted LR center shift from baseline, in mm.
    vox_size : np.ndarray, optional
        Voxel size of the fsaverage grid per voxel axis (default: 1 mm isotropic). Distances are measured in units of
        the LR voxel size, and the plane tilt and rotation are computed in mm, so the update is rigid in mm. The mm
        thresholds are converted to LR voxels (unchanged for 1 mm grids).

    Returns
    -------
    MidplaneRefinementResult
        Updated transform, applied center shift, and diagnostics.
    """
    scale = np.ones(3, dtype=float) if vox_size is None else np.asarray(vox_size, dtype=float)
    # distances in units of LR voxels (identical to plain voxel distances for isotropic grids)
    edt_sampling = scale / scale[0]
    support_distance_vox = float(support_distance_mm / scale[0])
    fit_band_vox = float(fit_band_mm / scale[0])
    max_center_shift_vox = float(max_center_shift_mm / scale[0])
    if not np.any(aseg_data > 0):
        logger.warning("Distance-map midplane refinement skipped: segmentation is empty.")
        return MidplaneRefinementResult(
            updated_vox2vox=orig2fsavg_vox2vox,
            center_shift_vox=0.0,
            diagnostics={},
        )

    seg_fsavg = resample_segmentation_to_fsavg(aseg_data, orig2fsavg_vox2vox, fsavg_shape)
    left_mask, right_mask = _clean_partitioned_hemi_masks(seg_fsavg, sampling=edt_sampling)
    brain_mask = seg_fsavg > 0
    hull_mask = _make_fsavg_convex_hull_mask(brain_mask)

    left_distance = distance_transform_edt(~left_mask, sampling=edt_sampling)
    right_distance = distance_transform_edt(~right_mask, sampling=edt_sampling)
    distance_diff = left_distance - right_distance
    candidate_mask = (
            hull_mask
            & (np.abs(distance_diff) <= zero_band_vox)
            & (np.minimum(left_distance, right_distance) <= support_distance_vox)
            & (np.abs(np.arange(fsavg_shape[0])[:, np.newaxis, np.newaxis] - base_middle_vox) <= fit_band_vox)
    )
    candidate_coords = np.argwhere(candidate_mask)
    diagnostics: dict[str, object] = {
        "candidate_count": int(candidate_coords.shape[0]),
        "zero_band_vox": float(zero_band_vox),
        "support_distance_vox": float(support_distance_vox),
        "support_distance_mm": float(support_distance_mm),
        "fit_band_vox": float(fit_band_vox),
        "fit_band_mm": float(fit_band_mm),
        "left_mask_voxels": int(np.count_nonzero(left_mask)),
        "right_mask_voxels": int(np.count_nonzero(right_mask)),
        "brain_mask_voxels": int(np.count_nonzero(brain_mask)),
    }
    if candidate_coords.shape[0] < 200:
        logger.warning(
            "Distance-map midplane refinement skipped: too few candidate voxels (%d).",
            candidate_coords.shape[0],
        )
        diagnostics["rejected_reason"] = "too_few_candidates"
        return MidplaneRefinementResult(
            updated_vox2vox=orig2fsavg_vox2vox,
            center_shift_vox=0.0,
            diagnostics=diagnostics,
        )

    x = candidate_coords[:, 0].astype(float)
    y = candidate_coords[:, 1].astype(float)
    z = candidate_coords[:, 2].astype(float)
    design = np.column_stack([y, z, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(design, x, rcond=None)
    a_coef, b_coef, c_coef = (float(c) for c in coeffs)
    fitted_x = design @ coeffs
    rmse = float(np.sqrt(np.mean((fitted_x - x) ** 2)))

    # plane x = a*y + b*z + c (voxel coordinates) -> normal in mm: (1/sx, -a/sy, -b/sz)
    plane_normal = np.array([1.0, -a_coef, -b_coef], dtype=float) / scale
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    tilt_deg = float(np.degrees(np.arccos(np.clip(np.dot(plane_normal, x_axis), -1.0, 1.0))))

    yz_center = np.asarray(fsavg_shape[1:], dtype=float) / 2.0
    plane_center = np.array(
        [
            a_coef * yz_center[0] + b_coef * yz_center[1] + c_coef,
            yz_center[0],
            yz_center[1],
        ],
        dtype=float,
    )
    center_shift_vox = float(plane_center[0] - base_middle_vox)

    diagnostics.update(
        {
            "plane_coefficients_xyz": [1.0, -a_coef, -b_coef, -c_coef],
            "plane_center_vox": plane_center.tolist(),
            "plane_fit_rmse_vox": rmse,
            "plane_tilt_deg": tilt_deg,
            "center_shift_vox": center_shift_vox,
            "max_tilt_deg": float(max_tilt_deg),
            "max_center_shift_vox": float(max_center_shift_vox),
            "max_center_shift_mm": float(max_center_shift_mm),
        }
    )

    if abs(center_shift_vox) > max_center_shift_vox:
        logger.info(
            "Distance-map midplane refinement rejected: center shift %.2f vox exceeds %.2f.",
            center_shift_vox,
            max_center_shift_vox,
        )
        diagnostics["rejected_reason"] = "center_shift_too_large"
        return MidplaneRefinementResult(
            updated_vox2vox=orig2fsavg_vox2vox,
            center_shift_vox=0.0,
            diagnostics=diagnostics,
        )
    if tilt_deg > max_tilt_deg:
        logger.info(
            "Distance-map midplane refinement rejected: tilt %.2f deg exceeds %.2f deg.",
            tilt_deg,
            max_tilt_deg,
        )
        diagnostics["rejected_reason"] = "tilt_too_large"
        return MidplaneRefinementResult(
            updated_vox2vox=orig2fsavg_vox2vox,
            center_shift_vox=0.0,
            diagnostics=diagnostics,
        )

    rotation = _rotation_from_vectors(plane_normal, x_axis)
    rotate_affine = _affine_from_rotation_and_center(rotation, plane_center, scale)
    translate_affine = offset_affine([base_middle_vox - plane_center[0], 0, 0])
    update_affine = translate_affine @ rotate_affine
    diagnostics["update_affine"] = convert_numpy_to_json_serializable(update_affine)
    diagnostics["rejected_reason"] = None
    logger.info(
        "Distance-map midplane refinement applied center shift %.2f vox with tilt %.2f deg (RMSE %.3f, candidates %d).",
        center_shift_vox,
        tilt_deg,
        rmse,
        candidate_coords.shape[0],
    )
    return MidplaneRefinementResult(
        updated_vox2vox=update_affine @ orig2fsavg_vox2vox,
        center_shift_vox=center_shift_vox,
        diagnostics=diagnostics,
    )


def find_midplane_transform(
        orig: nibabelImage,
        aseg_img: nibabelImage,
        midplane_method: str,
        ac_coords_orig: np.ndarray | None = None,
        pc_coords_orig: np.ndarray | None = None,
) -> MidplaneTransformResult:
    """Resolve the fsaverage midplane transform for a selected refinement method.

    Parameters
    ----------
    orig : nibabelImage
        Input conformed anatomical image.
    aseg_img : nibabelImage
        Subject aseg segmentation image.
    midplane_method : str
        Midplane strategy. Supported values are ``"center"``, ``"fsaverage"``,
        ``"fsaverage_distance_map"``, and ``"fsaverage_symmetry"``.
    ac_coords_orig : np.ndarray, optional
        AC voxel coordinates in orig (requires pc_coords_orig). If passed, the midplane is adjusted to pass through
        AC and PC.
    pc_coords_orig : np.ndarray, optional
        PC voxel coordinates in orig (requires ac_coords_orig).

    Returns
    -------
    MidplaneTransformResult
        Transform bundle and method-specific diagnostics for downstream CC steps.
    """
    aseg_data = np.asarray(aseg_img.dataobj)

    def _grid_zooms(vox2ras: AffineMatrix4x4) -> np.ndarray:
        """Voxel size per voxel axis of the target (fsaverage-aligned) grid."""
        return np.linalg.norm(np.asarray(vox2ras, dtype=float)[:3, :3], axis=0)

    if (ac_coords_orig is None) != (pc_coords_orig is None):
        raise ValueError("AC and PC coordinates must be supplied together.")

    def _adjust_with_landmarks(result: MidplaneTransformResult) -> MidplaneTransformResult:
        if ac_coords_orig is None or pc_coords_orig is None:
            return result
        adjusted_transform, landmark_diagnostics = adjust_midplane_to_landmarks(
            result.orig2fsavg_vox2vox,
            ac_coords_orig,
            pc_coords_orig,
            result.base_middle_vox,
            vox_size=_grid_zooms(result.fsavg_vox2ras),
        )
        diagnostics = dict(result.midline_shift_diagnostics)
        diagnostics["landmark_adjustment"] = landmark_diagnostics
        return MidplaneTransformResult(
            orig2fsavg_vox2vox=adjusted_transform,
            fsavg_vox2ras=result.fsavg_vox2ras,
            fsavg_header_dict=result.fsavg_header_dict,
            fsavg_shape=result.fsavg_shape,
            base_middle_vox=result.base_middle_vox,
            midline_shift_vox=result.midline_shift_vox,
            midline_shift_diagnostics=diagnostics,
        )

    if midplane_method == "center":
        # the center method uses orig's voxel grid directly and assumes LIA orientation (axis 0 = LR, slab
        # extracted along axis 0, axes 1/2 = IS/AP)
        orig_axcodes = nib.aff2axcodes(orig.affine)
        if tuple(orig_axcodes) != ("L", "I", "A"):
            raise ValueError(
                f"midplane_method 'center' requires an LIA-oriented (conformed) input image, but the image orientation "
                f"is {''.join(orig_axcodes)}. Conform the image to LIA or use a fsaverage-based midplane method."
            )
        return _adjust_with_landmarks(MidplaneTransformResult(
            orig2fsavg_vox2vox=np.eye(4),
            fsavg_vox2ras=orig.affine,
            fsavg_header_dict={"dims": list(orig.shape[:3])},
            fsavg_shape=orig.shape[:3],
            base_middle_vox=orig.shape[0] / 2.0,
            midline_shift_vox=0.0,
            midline_shift_diagnostics={},
        ))

    orig2fsavg_vox2vox, _, fsavg_vox2ras, fsavg_header_dict = register_centroids_to_fsavg(aseg_img)
    fsavg_shape = tuple(fsavg_header_dict["dims"])
    base_middle_vox = FSAVERAGE_MIDPLANE_MM / float(_grid_zooms(fsavg_vox2ras)[0])

    if midplane_method == "fsaverage_distance_map":
        dm_result = refine_midplane_with_distance_maps(
            orig2fsavg_vox2vox=orig2fsavg_vox2vox,
            aseg_data=aseg_data,
            fsavg_shape=fsavg_shape,
            base_middle_vox=base_middle_vox,
            vox_size=_grid_zooms(fsavg_vox2ras),
        )
        return _adjust_with_landmarks(MidplaneTransformResult(
            orig2fsavg_vox2vox=dm_result.updated_vox2vox,
            fsavg_vox2ras=fsavg_vox2ras,
            fsavg_header_dict=fsavg_header_dict,
            fsavg_shape=fsavg_shape,
            base_middle_vox=base_middle_vox,
            midline_shift_vox=dm_result.center_shift_vox,
            midline_shift_diagnostics=dm_result.diagnostics,
        ))

    if midplane_method == "fsaverage_symmetry":
        orig2fsavg_vox2vox, lr_shift, _, diagnostics = refine_midline_lr_shift(
            orig2fsavg_vox2vox=orig2fsavg_vox2vox,
            aseg_data=aseg_data,
            fsavg_shape=fsavg_shape,
            base_middle_vox=base_middle_vox,
            vox_size=_grid_zooms(fsavg_vox2ras),
        )
        return _adjust_with_landmarks(MidplaneTransformResult(
            orig2fsavg_vox2vox=orig2fsavg_vox2vox,
            fsavg_vox2ras=fsavg_vox2ras,
            fsavg_header_dict=fsavg_header_dict,
            fsavg_shape=fsavg_shape,
            base_middle_vox=base_middle_vox,
            midline_shift_vox=float(lr_shift),
            midline_shift_diagnostics=diagnostics,
        ))

    if midplane_method == "fsaverage":
        return _adjust_with_landmarks(MidplaneTransformResult(
            orig2fsavg_vox2vox=orig2fsavg_vox2vox,
            fsavg_vox2ras=fsavg_vox2ras,
            fsavg_header_dict=fsavg_header_dict,
            fsavg_shape=fsavg_shape,
            base_middle_vox=base_middle_vox,
            midline_shift_vox=0.0,
            midline_shift_diagnostics={},
        ))

    raise ValueError(f"Unsupported midplane_method: {midplane_method!r}")
