"""Regression tests for corpus callosum contour visualization."""

from types import SimpleNamespace

import numpy as np

from CorpusCallosum.cc_visualization import thickness_profile_for_plot
from CorpusCallosum.shape.contour import _interpolate_and_smooth_values


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
