"""Tests for the shared corpus callosum colormap resolver."""

import numpy as np
import pytest

from CorpusCallosum.shape.mesh import CCMesh
from CorpusCallosum.utils.colormaps import as_plotly_colorscale, get_colormap, resolve_color_range, with_neutral_range


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


def test_narrow_neutral_range_is_exact_in_matplotlib_and_plotly() -> None:
    """Thresholds narrower than the sampled color-table spacing must remain visible."""
    thresholded = with_neutral_range("diverging", (-1.0, 1.0), (-0.001, 0.001))

    assert thresholded(0.5)[:3] == pytest.approx((150 / 255,) * 3)
    scale = as_plotly_colorscale(thresholded, samples=5)
    assert [0.5, "#969696"] in scale


def test_neutral_range_covering_data_has_gray_plotly_endpoints() -> None:
    """A threshold covering the complete range must not restore base colors at its edges."""
    thresholded = with_neutral_range("diverging", (0.0, 1.0), (-2.0, 2.0))

    scale = as_plotly_colorscale(thresholded, samples=5)
    assert {color for _position, color in scale} == {"#969696"}


def test_zero_width_neutral_range_is_disabled_consistently() -> None:
    """Threshold zero represents an empty interval in both renderers."""
    base = get_colormap("diverging")
    thresholded = with_neutral_range(base, (-1.0, 1.0), (0.0, 0.0))

    np.testing.assert_allclose(thresholded(0.5), base(0.5))
    assert "#969696" not in {color for _position, color in as_plotly_colorscale(thresholded, samples=5)}


@pytest.mark.parametrize("value", [0.0, 3.5])
def test_constant_overlay_gets_non_degenerate_automatic_range(value: float) -> None:
    """Constant overlays need valid limits for both renderers."""
    color_range = resolve_color_range(np.full(3, value))

    assert color_range[0] < value < color_range[1]


def test_thresholded_constant_mesh_renders(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A constant signed overlay should not fail when thresholding is enabled."""
    figures = []
    monkeypatch.setattr(
        "CorpusCallosum.shape.mesh.plotly_write_html",
        lambda figure, *_args, **_kwargs: figures.append(figure),
    )
    mesh = CCMesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
        vertex_values=np.zeros(3),
    )

    mesh.plot_mesh(output_path=tmp_path / "constant.html", colormap="diverging", threshold=0.2)

    trace = figures[0].data[0]
    assert trace.cmin < 0 < trace.cmax


def test_neutral_range_preserves_masked_and_over_colors() -> None:
    """Neutral bands cannot hide missing values or upstream's upper threshold."""
    cmap = with_neutral_range("diverging", (0.0, 1.0), (-2.0, 2.0)).with_extremes(over="lime")
    colors = cmap(np.ma.array([0.5, 0.5, 1.1], mask=[False, True, False]))
    np.testing.assert_allclose(colors[0, :3], (150 / 255,) * 3)
    assert colors[1, 3] == 0.0
    np.testing.assert_allclose(colors[2, :3], (0.0, 1.0, 0.0))


def test_neutral_range_applies_outside_fixed_limits_without_over_override() -> None:
    """Fixed display limits cannot silently disable an absolute-value threshold."""
    cmap = with_neutral_range("diverging", (-1., 1.), (-2., 2.))
    np.testing.assert_allclose(cmap(1.1)[:3], (150 / 255,) * 3)
