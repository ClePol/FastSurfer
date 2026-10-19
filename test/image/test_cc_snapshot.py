"""Tests for supported color-control forwarding to native WhipperSnapPy snapshots."""

import inspect
import logging
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

import CorpusCallosum.cc_visualization as visualization
from CorpusCallosum.shape.mesh import CCMesh, _snapshot_color_options


def test_native_snapshot_maps_supported_controls() -> None:
    """Only native API parameters, rather than Matplotlib names, reach the renderer."""
    values = np.array([-1., -0.1, 0.1, 1.])
    original = values.copy()
    options = _snapshot_color_options(
        values, colormap="diverging", color_range=(-1.5, 1.5), threshold=0.25, legend="Artificial correlation",
    )
    assert options["fthresh"] == 0.25
    assert options["fmax"] == 1.5
    assert options["caption"] == "Artificial correlation"
    assert options["colorbar"] is True
    assert "colormap" not in options
    np.testing.assert_array_equal(options["overlay"], original)
    np.testing.assert_array_equal(values, original)


def test_native_snapshot_defaults_preserve_pipeline_calls() -> None:
    """The existing thickness snapshot caller keeps its native defaults."""
    options = _snapshot_color_options(np.array([1., 2.]))
    assert options["fthresh"] == 0
    assert options["fmax"] is None
    assert options["caption"] == "Corpus Callosum thickness (mm)"
    assert options["colorbar"] is True
    assert _snapshot_color_options(None, legend="")["caption"] == ""


@pytest.mark.parametrize("name", ["viridis", "RdBu_r", "red_to_yellow"])
def test_unsupported_snapshot_palette_is_explicit(name, caplog) -> None:
    """Valid arbitrary/legacy palettes cannot be silently claimed as native colors."""
    options = _snapshot_color_options(np.array([-1., 1.]), colormap=name)
    assert options["overlay"] is not None
    assert "cannot use colormap" in caplog.text
    assert name in caplog.text
    assert "PNG and Plotly HTML" in caplog.text


@pytest.mark.parametrize("color_range", [(0., 1.), (-0.5, 1.), (2., 5.), (-5., -2.)])
def test_unsupported_snapshot_range_uses_native_auto_with_warning(color_range, caplog) -> None:
    """Asymmetric display limits must not be silently converted to a different range."""
    options = _snapshot_color_options(np.array([-1., 1.]), color_range=color_range)
    assert options["fmax"] is None
    assert "symmetric color_range" in caplog.text
    assert "PNG and Plotly HTML retain it" in caplog.text


@pytest.mark.parametrize("threshold", [-0.1, np.nan, np.inf, (-0.2, 0.2)])
def test_native_snapshot_rejects_invalid_threshold(threshold) -> None:
    """The native cutoff is a finite, non-negative magnitude, not a gray-band tuple."""
    with pytest.raises(ValueError, match="finite, non-negative scalar"):
        _snapshot_color_options(np.array([-1., 1.]), threshold=threshold)


@pytest.mark.parametrize("color_range", [(1., -1.), (0., 0.), (-np.inf, np.inf), (np.nan, 1.)])
def test_native_snapshot_rejects_invalid_range(color_range) -> None:
    """Invalid ranges fail rather than being treated as unsupported but valid limits."""
    with pytest.raises(ValueError, match="Color range"):
        _snapshot_color_options(np.array([-1., 1.]), color_range=color_range)


def test_native_snapshot_rejects_unknown_palette() -> None:
    """Misspelled colormaps still receive the shared actionable validation error."""
    with pytest.raises(ValueError, match="Unknown colormap"):
        _snapshot_color_options(np.array([-1., 1.]), colormap="invalid-map")


@pytest.mark.parametrize("values, threshold", [
    ([0., 0.], None), ([0., 0.], 0.), ([0.1, -0.1, np.nan], 0.2), ([0.25, -0.25], 0.25),
])
def test_all_neutral_snapshot_has_gray_geometry_without_colorbar(values, threshold, caplog) -> None:
    """All-neutral data avoid native no-sign errors and a misleading colorbar."""
    with caplog.at_level(logging.INFO):
        options = _snapshot_color_options(np.array(values), threshold=threshold)
    assert options["overlay"] is None
    assert options["colorbar"] is False
    assert "gray mesh without a colorbar" in caplog.text


@pytest.mark.parametrize("values", [[3.5, 3.5], [-3.5, -3.5]])
def test_nonzero_constant_snapshot_retains_overlay(values) -> None:
    """Constant positive and negative values remain valid native overlays."""
    options = _snapshot_color_options(np.array(values))
    np.testing.assert_array_equal(options["overlay"], values)
    assert options["colorbar"] is True


def test_threshold_beyond_saturation_warns_and_keeps_visible_data(caplog) -> None:
    """Do not invert normalization or turn visible values gray when fmax <= fthresh."""
    options = _snapshot_color_options(np.array([-1., 1.]), threshold=0.3, color_range=(-0.2, 0.2))
    assert options["fmax"] is None
    assert options["fthresh"] == 0.3
    assert options["overlay"] is not None
    assert "saturation must exceed its threshold" in caplog.text


@pytest.mark.parametrize("values", [[np.nan, np.nan], [np.inf, 1.], [[1., 2.], [3., 4.]]])
def test_native_snapshot_rejects_invalid_overlay(values) -> None:
    """Unusable scalar arrays must not reach native threshold estimation."""
    with pytest.raises(ValueError, match="Snapshot overlay"):
        _snapshot_color_options(np.array(values))


def test_native_snapshot_preserves_masked_values_as_missing() -> None:
    """Masked values cannot contribute to saturation or native color selection."""
    values = np.ma.array([1, 500], mask=[False, True])
    options = _snapshot_color_options(values)
    assert options["overlay"][0] == 1.0
    assert np.isnan(options["overlay"][1])
    assert values[0] == 1


@pytest.fixture
def snapshot_renderer(monkeypatch):
    """Capture native calls without requiring WhipperSnapPy or an OpenGL context."""
    calls = []
    fake = SimpleNamespace(__version__="2.1.0", snap1=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setitem(sys.modules, "whippersnappy", fake)
    monkeypatch.setattr(CCMesh, "_CCMesh__create_cc_viewmat", lambda self: np.eye(4))
    return calls, fake


def _mesh(values=None):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return CCMesh(vertices, faces, np.array([-1., -0.1, 0.1, 1.]) if values is None else values)


def test_snapshot_method_forwards_native_options(snapshot_renderer, tmp_path) -> None:
    """Exercise the actual method so argument translation cannot be bypassed."""
    calls, _ = snapshot_renderer
    mesh = _mesh()
    mesh.snap_cc_picture(tmp_path / "snapshot.png", colormap="diverging", color_range=(-1.5, 1.5),
                         threshold=0.25, legend="Artificial correlation")
    assert len(calls) == 1
    assert calls[0]["fthresh"] == 0.25
    assert calls[0]["fmax"] == 1.5
    assert calls[0]["caption"] == "Artificial correlation"
    np.testing.assert_array_equal(calls[0]["overlay"], mesh.mesh_vertex_colors)
    np.testing.assert_array_equal(calls[0]["mesh"][0], mesh.v)


def test_snapshot_method_legacy_positional_call(snapshot_renderer, tmp_path) -> None:
    """Adding settings cannot change the existing output/header positional API."""
    calls, _ = snapshot_renderer
    _mesh().snap_cc_picture(tmp_path / "snapshot.png", None)
    assert calls[0]["caption"] == "Corpus Callosum thickness (mm)"
    assert calls[0]["fthresh"] == 0
    assert inspect.signature(CCMesh.snap_cc_picture).parameters["colormap"].kind == inspect.Parameter.KEYWORD_ONLY


def test_snapshot_method_all_neutral_forwards_no_scalar_overlay(snapshot_renderer, tmp_path) -> None:
    """The actual renderer receives plain geometry for an all-zero overlay."""
    calls, _ = snapshot_renderer
    _mesh(np.zeros(4)).snap_cc_picture(tmp_path / "snapshot.png")
    assert calls[0]["overlay"] is None
    assert calls[0]["colorbar"] is False


def test_snapshot_method_rejects_wrong_vertex_count(snapshot_renderer, tmp_path) -> None:
    """Wrong-size overlays fail even if all their values would be neutral."""
    calls, _ = snapshot_renderer
    with pytest.raises(ValueError, match="one value per mesh vertex"):
        _mesh(np.zeros(3)).snap_cc_picture(tmp_path / "snapshot.png")
    assert calls == []


def test_snapshot_method_keeps_minimum_version_check(snapshot_renderer, tmp_path) -> None:
    """Forwarding needs no old-version fallback or dependency-floor change."""
    calls, fake = snapshot_renderer
    fake.__version__ = "2.0.0"
    with pytest.raises(ImportError, match="whippersnappy>=2.1"):
        _mesh().snap_cc_picture(tmp_path / "snapshot.png")
    assert calls == []


def test_visualization_main_forwards_snapshot_options(monkeypatch, tmp_path) -> None:
    """The CLI's main path must forward the same settings used by PNG/HTML."""
    snapshot_calls = []
    figure = plt.figure()
    contour = SimpleNamespace(get_thickness_profile=lambda: np.array([1., -1.]),
                              plot_contour_colorfill=lambda *_args, **_kwargs: figure)
    mesh = SimpleNamespace(
        plot_mesh=lambda **kwargs: None, write_vtk=lambda *_args: None,
        write_fssurf=lambda *_args, **_kwargs: None, write_morph_data=lambda *_args: None,
        snap_cc_picture=lambda *args, **kwargs: snapshot_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(visualization, "load_contours_from_template_dir", lambda *_args, **_kwargs: [contour, contour])
    monkeypatch.setattr(visualization, "_template_midslice_position", lambda *_args: 0)
    monkeypatch.setattr(visualization, "contours_for_analysis_width", lambda contours: contours)
    monkeypatch.setattr(visualization.CCMesh, "from_contours", lambda *_args, **_kwargs: mesh)
    monkeypatch.setattr(visualization, "_upright_reference_from_template", lambda *_args: (None, None))
    assert visualization.main(tmp_path, tmp_path / "output", colormap="diverging", color_range=(-1.5, 1.5),
                              threshold=0.25, legend="Artificial correlation") == 0
    assert snapshot_calls[0][1] == dict(ref_header=None, colormap="diverging", color_range=(-1.5, 1.5),
                                      threshold=0.25, legend="Artificial correlation")
