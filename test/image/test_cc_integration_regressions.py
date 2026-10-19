"""Regression tests for corpus callosum pipeline integration."""

import inspect
import json
from concurrent.futures import Future
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import CorpusCallosum.cc_visualization as cc_visualization
import CorpusCallosum.fastsurfer_cc as cc_pipeline
import CorpusCallosum.shape.postprocessing as postprocessing
from CorpusCallosum.cc_visualization import load_contours_from_template_dir
from CorpusCallosum.data.constants import THIRD_VENTRICLE_LABEL


def test_single_row_csv_with_multiple_columns_is_rejected(tmp_path) -> None:
    """A two-column row is not an anterior-to-posterior value profile."""
    path = tmp_path / "values.csv"
    path.write_text("first,second\n1,2\n")
    with pytest.raises(ValueError, match="exactly one column"):
        cc_visualization.load_plot_values(path)


@pytest.mark.parametrize("kwargs", [{"log_scale": True}, {"upper_threshold": 0.05}])
def test_values_only_options_cannot_be_silently_ignored(tmp_path, kwargs) -> None:
    """Unsupported template options fail before producing misleading output."""
    with pytest.raises(ValueError, match="require --values_file"):
        cc_visualization.main(tmp_path, tmp_path / "output", **kwargs)


class ImmediateExecutor:
    """Executor that exposes failures through Future.result()."""

    def submit(self, function, *args, **kwargs) -> Future:
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future


class NoOutputSubjectDirectory:
    """Minimal subject directory for postprocessing tests."""

    @staticmethod
    def has_attribute(_attribute: str) -> bool:
        return False

    @staticmethod
    def filename_by_attribute(attribute: str) -> Path:
        return Path(attribute)


def _run_multi_slice_reconstruction(monkeypatch: pytest.MonkeyPatch, worker, subject_dir=None, slice_selection="all"):
    executor = ImmediateExecutor()
    monkeypatch.setattr(postprocessing, "process_executor", lambda: executor)
    monkeypatch.setattr(postprocessing, "thread_executor", lambda: executor)
    monkeypatch.setattr(postprocessing, "recon_cc_surf_measure", worker)
    return postprocessing.recon_cc_surf_measures_multi(
        segmentation=np.zeros((5, 2, 2), dtype=int),
        slice_selection=slice_selection,
        orig_header=None,
        fsavg2midslab_vox2vox=np.eye(4),
        fsavg_vox2ras=np.eye(4),
        orig2fsavg_vox2vox=np.eye(4),
        midslices=np.stack([np.full((2, 2), idx) for idx in range(13)]),
        ac_coords_vox=np.zeros(2),
        pc_coords_vox=np.ones(2),
        num_thickness_points=2,
        subdivisions=[0.5],
        subdivision_method="shape",
        contour_smoothing=0,
        subject_dir=subject_dir or NoOutputSubjectDirectory(),
    )


def test_slice_failure_does_not_discard_later_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Independent futures allow reconstruction to continue after one failure."""

    def worker(_segmentation, slice_idx, _affine, **_kwargs):
        if slice_idx == 2:
            raise RuntimeError("synthetic middle-slice failure")
        return {"slice_index": slice_idx}, object()

    results, _io_futures, contours, failed, indices = _run_multi_slice_reconstruction(monkeypatch, worker)

    assert [result["slice_index"] if result else None for result in results] == [0, 1, None, 3, 4]
    assert indices == [0, 1, 2, 3, 4]
    assert len(contours) == 4
    assert failed == 1


def test_failed_middle_slice_is_not_replaced() -> None:
    """Midslice output stays absent if the anatomical middle slice failed."""
    results = [{"slice_index": idx} for idx in (0, 1, 3, 4)]

    ordered, selected = cc_pipeline.select_requested_slice_results(results, 5, "all")

    assert [result["slice_index"] if result else None for result in ordered] == [0, 1, None, 3, 4]
    assert selected is None


def test_incomplete_slice_set_has_no_contour_volume() -> None:
    """Partial recovery must not calculate or publish a slab volume."""
    assert cc_pipeline.calculate_complete_contour_volume([object(), object()], num_failed_slices=1) is None


def test_missing_selected_slice_fails_subdivision_dependent_output() -> None:
    """The pipeline must not silently pass a generic label to downstream CC painting."""
    error = cc_pipeline.selected_slice_output_error(None, orig_segmentation_requested=True)

    assert error is not None
    assert "FreeSurfer-compatible subdivided CC segmentation" in error
    assert cc_pipeline.selected_slice_output_error(None, orig_segmentation_requested=False) is None


def test_middle_slice_qc_uses_matching_context_slice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """QC for the middle segmentation slice must use the corresponding MRI slice."""
    captured_slices = []

    class QCSubjectDirectory(NoOutputSubjectDirectory):
        @staticmethod
        def has_attribute(attribute: str) -> bool:
            return attribute == "cc_qc_image"

        @staticmethod
        def filename_by_attribute(_attribute: str) -> Path:
            return tmp_path / "qc.png"

    def worker(_segmentation, slice_idx, _affine, **_kwargs):
        measures = {
            "slice_index": slice_idx,
            "split_contours": [],
            "midline_equidistant": np.empty((0, 2)),
            "levelpaths": [],
        }
        return measures, object()

    monkeypatch.setattr(
        postprocessing,
        "plot_contours",
        lambda **kwargs: captured_slices.append(kwargs["slice_or_slab"].copy()),
    )
    _run_multi_slice_reconstruction(monkeypatch, worker, QCSubjectDirectory(), slice_selection="middle")

    np.testing.assert_array_equal(captured_slices[0], np.full((1, 2, 2), 6))


def test_template_files_are_sorted_by_numeric_suffix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Double-digit slice indices must stay in anatomical order."""

    class FakeContour:
        def __init__(self):
            self.loaded_index = None
            self.lr_position = 0.0

        def create_levelpaths(self, **_kwargs):
            return None

        def copy(self):
            return FakeContour()

        def load_thickness_values(self, path: Path):
            self.loaded_index = int(path.stem.split("_")[-1])

        def smooth_contour(self, **_kwargs):
            return None

        def fill_thickness_values(self):
            return None

    for idx in range(11):
        (tmp_path / f"thickness_values_{idx}.txt").write_text("thickness\n1\n1\n")
    monkeypatch.setattr("CorpusCallosum.cc_visualization.load_fsaverage_cc_template", FakeContour)

    contours = load_contours_from_template_dir(tmp_path, resolution=1.0, smoothing_window=0)

    assert [contour.loaded_index for contour in contours] == list(range(11))
    assert contours[5].lr_position == 0


def test_incomplete_template_does_not_substitute_midslice(tmp_path: Path) -> None:
    """Metadata prevents neighboring slices from replacing a failed selected slice."""
    indices = [0, 1, 3, 4]
    for idx in indices:
        (tmp_path / f"thickness_values_{idx}.txt").touch()
    metadata = {
        "schema_version": 1,
        "requested_slice_indices": list(range(5)),
        "successful_slice_indices": indices,
        "selected_slice_index": 2,
        "slice_positions_mm": {},
    }
    (tmp_path / "template_metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="selected anatomical slice 2 is missing"):
        cc_visualization._template_midslice_position(tmp_path, two_d=True)


def test_incomplete_template_is_rejected_for_3d(tmp_path: Path) -> None:
    """A partial slice set must not be redistributed across the full slab."""
    indices = [0, 1, 2, 4]
    for idx in indices:
        (tmp_path / f"thickness_values_{idx}.txt").touch()
    metadata = {
        "schema_version": 1,
        "requested_slice_indices": list(range(5)),
        "successful_slice_indices": indices,
        "selected_slice_index": 2,
        "slice_positions_mm": {},
    }
    (tmp_path / "template_metadata.json").write_text(json.dumps(metadata))

    assert cc_visualization._template_midslice_position(tmp_path, two_d=True) == 2
    with pytest.raises(ValueError, match="full-slab 3D visualization"):
        cc_visualization._template_midslice_position(tmp_path, two_d=False)


def test_visualization_main_preserves_twod_positional_argument() -> None:
    """Appending threshold must not change the legacy eighth positional argument."""
    parameters = list(inspect.signature(cc_visualization.main).parameters)

    assert parameters.index("twoD") < parameters.index("threshold")


def test_third_ventricle_uses_freesurfer_label() -> None:
    """The localization anchor must use the LUT's third-ventricle label."""
    assert THIRD_VENTRICLE_LABEL == 14


def test_missing_third_ventricle_uses_aligned_center(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing label 14 should produce a finite, explicit crop fallback."""
    executor = ImmediateExecutor()
    captured_centers = []
    monkeypatch.setattr(cc_pipeline, "thread_executor", lambda: executor)
    monkeypatch.setattr(
        cc_pipeline,
        "affine_transform",
        lambda _data, _affine, output_shape, **_kwargs: np.zeros(output_shape),
    )
    monkeypatch.setattr(
        cc_pipeline.localization_inference,
        "run_inference_on_slice",
        lambda _model, _image, center: (captured_centers.append(center.copy()) or (np.zeros(2), np.ones(2))),
    )
    aseg = nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.int16), np.eye(4))

    cc_pipeline.localize_ac_pc(
        orig_data=np.zeros((3, 3, 3)),
        aseg_nib=aseg,
        orig2midslab_vox2vox=np.eye(4),
        model_localization=object(),
        resample_shape=(5, 9, 11),
    )

    np.testing.assert_array_equal(captured_centers[0], [4.0, 5.0])


def test_template_metadata_keeps_physical_positions_after_partial_failure(monkeypatch, tmp_path):
    """Exported indices and LR positions survive a missing middle slice."""
    class TemplateSubjectDirectory(NoOutputSubjectDirectory):
        @staticmethod
        def has_attribute(attribute):
            return attribute == "save_template_dir"

        @staticmethod
        def filename_by_attribute(_attribute):
            return tmp_path

    class SavedContour:
        def __init__(self, position):
            self.lr_position = position

        @staticmethod
        def save_contour(path):
            path.write_text("contour\n")

        @staticmethod
        def save_thickness_values(path):
            path.write_text("thickness\n1\n")

    def worker(_segmentation, slice_idx, affine, **_kwargs):
        if slice_idx == 2:
            raise RuntimeError("synthetic middle-slice failure")
        return {"slice_index": slice_idx}, SavedContour(float(affine[0, 3]))

    _, futures, _, failed, _ = _run_multi_slice_reconstruction(monkeypatch, worker, TemplateSubjectDirectory())
    for future in futures:
        future.result()
    metadata = json.loads((tmp_path / "template_metadata.json").read_text())
    assert failed == 1
    assert metadata["requested_slice_indices"] == [0, 1, 2, 3, 4]
    assert metadata["successful_slice_indices"] == [0, 1, 3, 4]
    assert metadata["selected_slice_index"] == 2
    assert metadata["slice_positions_mm"] == {"0": -2., "1": -1., "3": 1., "4": 2.}
    assert not (tmp_path / "contour_2.txt").exists()
    with pytest.raises(ValueError, match="selected anatomical slice 2 is missing"):
        cc_visualization._template_midslice_position(tmp_path, two_d=True)
