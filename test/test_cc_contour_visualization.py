"""Regression tests for corpus callosum contour visualization."""

from types import SimpleNamespace

import numpy as np
import pytest

import CorpusCallosum.shape.contour as contour_module
from CorpusCallosum.cc_visualization import thickness_profile_for_plot
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
    contour = SimpleNamespace(
        thickness_values=np.array([99.0, -1.0, 99.0, -0.5, 99.0, 0.5, 99.0, 1.0]),
        original_thickness_vertices=np.array([1, 3, 5, 7]),
    )

    profile = thickness_profile_for_plot(contour)

    np.testing.assert_array_equal(profile, [0.5, 1.0])


def test_matching_endpoint_records_one_measurement() -> None:
    """An exact endpoint match must not duplicate its thickness value."""
    contour = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    thickness = np.full(4, np.nan)

    contour, thickness = insert_point_with_thickness(contour, thickness, np.array([1.0, 0.0]), 1.0)
    contour, thickness = insert_point_with_thickness(contour, thickness, np.array([0.0, 0.5]), 1.0)

    assert len(contour) == 6
    assert np.count_nonzero(~np.isnan(thickness)) == 2
    profile = thickness_profile_for_plot(
        SimpleNamespace(thickness_values=thickness, original_thickness_vertices=np.where(~np.isnan(thickness))[0])
    )
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

    CCContour.plot_contour_colorfill(
        FakeContour(),
        plot_values=np.array([1.0, 2.0, 3.0]),
        colorbar=False,
        mode="thickness",
    )

    assert requested_counts == [3]
    np.testing.assert_array_equal(interpolated_values, [3.0, 2.0, 1.0])
