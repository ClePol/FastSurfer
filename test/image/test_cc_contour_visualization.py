"""Regression tests for corpus callosum contour visualization."""

import numpy as np
import pytest

import CorpusCallosum.shape.contour as contour_module
from CorpusCallosum.data.fsaverage_cc_template import load_fsaverage_cc_template
from CorpusCallosum.shape.contour import CCContour, _interpolate_and_smooth_values
from CorpusCallosum.shape.thickness import insert_point_with_thickness


def test_interpolation_does_not_add_neutral_edges() -> None:
    """Values outside the linear hull inherit the nearest measured value."""
    points_x = np.array([0.0, 1.0, 0.0])
    points_y = np.array([0.0, 0.0, 1.0])
    values = np.ones(3)
    x_grid, y_grid = np.meshgrid(np.linspace(-1.0, 2.0, 25), np.linspace(-1.0, 2.0, 25))

    interpolated = _interpolate_and_smooth_values(points_x, points_y, values, x_grid, y_grid)

    np.testing.assert_allclose(interpolated, 1.0)


def test_plot_profile_uses_only_original_measurement_vertices() -> None:
    """Interpolated contour values must not be mistaken for measurements."""
    contour = CCContour(
        points=np.zeros((8, 2)),
        thickness_values=np.array([np.nan, -1.0, np.nan, 1.0, np.nan, 1.0, np.nan, -1.0]),
    )
    contour.fill_thickness_values()

    profile = contour.get_thickness_profile()

    np.testing.assert_array_equal(profile, [-1.0, 1.0])


def test_matching_endpoint_records_one_measurement() -> None:
    """An exact endpoint match must not duplicate its thickness value."""
    contour = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    thickness = np.full(4, np.nan)

    contour, thickness = insert_point_with_thickness(contour, thickness, np.array([1.0, 0.0]), 1.0)
    contour, thickness = insert_point_with_thickness(contour, thickness, np.array([0.0, 0.5]), 1.0)

    assert len(contour) == 6
    assert np.count_nonzero(~np.isnan(thickness)) == 2
    profile = CCContour(points=contour, thickness_values=thickness).get_thickness_profile()
    np.testing.assert_array_equal(profile, [1.0])


def test_colorfill_uses_one_levelpath_per_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every supplied profile value must be assigned to a regenerated level path."""
    requested_counts = []
    interpolated_values = []

    class FakeContour:
        points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])

        @staticmethod
        def create_levelpaths(num_points: int, inplace: bool):
            requested_counts.append(num_points)
            paths = [np.array([[0.5 * idx, 0.5]]) for idx in range(num_points)]
            return paths, None, None, None, None, None, None

    def fake_interpolate(_x, _y, values, x_grid, _y_grid):
        interpolated_values.extend(values)
        return np.zeros_like(x_grid)

    monkeypatch.setattr(contour_module, "_interpolate_and_smooth_values", fake_interpolate)
    monkeypatch.setattr(contour_module.plt, "show", lambda: None)

    figure = CCContour.plot_contour_colorfill(
        FakeContour(),
        plot_values=np.array([1.0, 2.0, 3.0]),
        colorbar=False,
        mode="thickness",
    )

    assert requested_counts == [3]
    np.testing.assert_array_equal(interpolated_values, [1.0, 2.0, 3.0])
    contour_module.plt.close(figure)


def _plot_without_remeshing(monkeypatch, **kwargs):
    """Exercise normalization and colorbars without expensive mesh construction."""
    contour = CCContour(np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.]]), None)
    monkeypatch.setattr(
        contour, "create_levelpaths",
        lambda num_points, inplace: ([np.array([[float(idx), 0.5]]) for idx in range(num_points)],),
    )
    monkeypatch.setattr(contour_module, "_interpolate_and_smooth_values", lambda _x, _y, _v, x, _gy: np.ones_like(x))
    monkeypatch.setattr(contour_module.plt, "show", lambda: None)
    return contour.plot_contour_colorfill(**kwargs)


def test_log_and_neutral_thresholds_preserve_upper_color(monkeypatch) -> None:
    """Log-normalized neutral bands and upper-threshold colors use data values."""
    figure = _plot_without_remeshing(
        monkeypatch, plot_values=np.array([0.001, 0.01, 0.1]), mode="p-value", colormap="viridis",
        log_scale=True, color_range=(0.001, 0.1), threshold=0.01,
        upper_threshold=0.05, threshold_color="lime",
    )
    image = figure.axes[0].images[0]
    np.testing.assert_allclose(image.cmap(image.norm(0.005))[:3], (150 / 255,) * 3)
    np.testing.assert_allclose(image.cmap(image.norm(0.06))[:3], (0., 1., 0.))
    assert not np.allclose(image.cmap(image.norm(0.02))[:3], (150 / 255,) * 3)
    assert image.norm.vmin == 0.001
    assert image.norm.vmax == 0.05
    contour_module.plt.close(figure)


@pytest.mark.parametrize("value", [0.0, 3.5])
def test_constant_contour_has_valid_color_range(monkeypatch, value) -> None:
    """Constant 2D overlays retain the non-degenerate automatic color range."""
    figure = _plot_without_remeshing(
        monkeypatch, plot_values=np.full(3, value), mode="thickness", colormap="diverging", threshold=0.2,
    )
    image = figure.axes[0].images[0]
    assert image.norm.vmin < value < image.norm.vmax
    contour_module.plt.close(figure)


@pytest.mark.parametrize("kwargs, message", [
    ({"log_scale": True, "color_range": (-1., 1.)}, "positive"),
    ({"upper_threshold": float("nan")}, "finite"),
    ({"threshold": -1.}, "non-negative"),
    ({"color_range": (1., 0.)}, "smaller"),
])
def test_invalid_normalization_has_clear_error(monkeypatch, kwargs, message) -> None:
    """Invalid combinations cannot silently render a misleading color scale."""
    with pytest.raises(ValueError, match=message):
        _plot_without_remeshing(monkeypatch, plot_values=np.array([0.1, 0.2, 0.3]), mode="thickness", **kwargs)


@pytest.mark.parametrize("smoothing", [0, 5])
def test_signed_values_keep_anatomical_ends(smoothing: int, tmp_path) -> None:
    """Canonical profiles remain red anterior and blue posterior after remeshing."""
    contour = load_fsaverage_cc_template()
    contour.create_levelpaths(num_points=12, inplace=True)
    profile = np.linspace(1.0, -1.0, 12)
    contour.set_thickness_values(np.concatenate((profile, profile[::-1])), use_measurement_points=True)
    if smoothing:
        contour.smooth_contour(window_size=smoothing)
    contour.fill_thickness_values()
    np.testing.assert_allclose(contour.get_thickness_profile(), profile)
    figure = contour.plot_contour_colorfill(
        contour.get_thickness_profile(), mode="thickness", colormap="diverging",
        color_range=(-1.0, 1.0), save_path=tmp_path / "signed.png",
    )
    image = figure.axes[0].images[0]
    data = image.get_array().filled(np.nan)
    x_min, x_max, _, _ = image.get_extent()
    x = x_min + np.arange(data.shape[1]) * (x_max - x_min) / data.shape[1]
    assert np.nanmean(data[:, x > 20]) > 0.5
    assert np.nanmean(data[:, x < -30]) < -0.5
    assert np.nanmean(contour.thickness_values[contour.points[:, 0] > 20]) > 0.5
    assert np.nanmean(contour.thickness_values[contour.points[:, 0] < -30]) < -0.5
    contour_module.plt.close(figure)
