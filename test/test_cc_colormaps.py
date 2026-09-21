"""Tests for the shared corpus callosum colormap resolver."""

import numpy as np
import pytest

from CorpusCallosum.utils.colormaps import as_plotly_colorscale, get_colormap, with_neutral_range


def test_diverging_colormap_matches_whippersnappy_heat_colors() -> None:
    """The diverging preset should use WhipperSnapPy's overlay colors."""
    cmap = get_colormap("diverging")

    np.testing.assert_allclose(cmap(0.0)[:3], (0.0, 1.0, 1.0), atol=0.02)
    np.testing.assert_allclose(cmap(1.0 / 3.0)[:3], (0.0, 0.0, 1.0), atol=0.02)
    np.testing.assert_allclose(cmap(0.499)[:3], (0.0, 0.0, 0.5625), atol=0.02)
    np.testing.assert_allclose(cmap(0.5)[:3], (0.5625, 0.0, 0.0), atol=0.02)
    np.testing.assert_allclose(cmap(2.0 / 3.0)[:3], (1.0, 0.0, 0.0), atol=0.02)
    np.testing.assert_allclose(cmap(1.0)[:3], (1.0, 1.0, 0.0), atol=0.02)


@pytest.mark.parametrize("name", ["red_to_blue", "blue_to_red", "red_to_yellow", "yellow_to_red"])
def test_legacy_colormap_presets_remain_available(name: str) -> None:
    """Existing command-line values must remain valid."""
    assert get_colormap(name).name == name


def test_registered_matplotlib_colormap_is_accepted() -> None:
    """Users can pass a Matplotlib name in addition to FastSurfer presets."""
    assert get_colormap("viridis").name == "viridis"


def test_colormap_converts_to_plotly_scale() -> None:
    """The 3D renderer receives a complete Plotly colorscale."""
    scale = as_plotly_colorscale("diverging", samples=5)

    assert [entry[0] for entry in scale] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert all(str(entry[1]).startswith("#") for entry in scale)


def test_unknown_colormap_has_actionable_error() -> None:
    """Invalid names should identify both supported naming schemes."""
    with pytest.raises(ValueError, match="registered Matplotlib colormap"):
        get_colormap("not-a-real-colormap")


def test_neutral_range_replaces_only_thresholded_values() -> None:
    """Thresholding should add a gray band without rescaling outer colors."""
    base = get_colormap("diverging")
    thresholded = with_neutral_range(base, (-1.0, 1.0), (-0.2, 0.2), samples=1001)

    np.testing.assert_allclose(thresholded(0.5)[:3], (150 / 255, 150 / 255, 150 / 255), atol=0.01)
    np.testing.assert_allclose(thresholded(0.0), base(0.0), atol=0.01)
    np.testing.assert_allclose(thresholded(1.0), base(1.0), atol=0.01)
