"""Tests for the shared corpus callosum colormap resolver."""

from copy import copy

import numpy as np
import pytest
from matplotlib.colors import to_rgba

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
    from CorpusCallosum.shape.mesh import CCMesh

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


def test_thresholded_colormap_public_queries_and_preview() -> None:
    """Matplotlib's query and notebook-preview APIs require a complete lookup table."""
    cmap = with_neutral_range("diverging", (-1., 1.), (-0.2, 0.2))
    np.testing.assert_allclose(cmap.get_bad(), (0., 0., 0., 0.))
    np.testing.assert_allclose(cmap.get_under(), cmap(-0.1))
    np.testing.assert_allclose(cmap.get_over(), cmap(1.1))
    assert not cmap.is_gray()
    assert "<img" in cmap._repr_html_()
    assert cmap == cmap.copy()
    for neutral_range in ((0., 0.1), (0.9, 1.)):
        edge_map = with_neutral_range("diverging", (0., 1.), neutral_range)
        np.testing.assert_allclose(edge_map.get_under(), edge_map(-0.1))
        np.testing.assert_allclose(edge_map.get_over(), edge_map(1.1))


def test_thresholded_colormap_grayscale_queries_include_neutral_band() -> None:
    """A neutral interval covering the full domain really is grayscale."""
    cmap = with_neutral_range("diverging", (0., 1.), (-2., 2.))
    assert cmap.is_gray()
    np.testing.assert_allclose(cmap.get_under(), cmap(-0.1))
    np.testing.assert_allclose(cmap.get_over(), cmap(1.1))


@pytest.mark.parametrize("extreme, value", [("bad", np.nan), ("under", -0.1), ("over", 1.1)])
def test_thresholded_colormap_setters_affect_colors_and_queries(extreme: str, value: float) -> None:
    """Mutable special colors must affect mapping before and after initialization."""
    cmap = with_neutral_range("diverging", (0., 1.), (-2., 2.))
    for color in ("lime", "magenta"):
        cmap.get_bad()  # Exercise updating an already-initialized lookup table.
        getattr(cmap, f"set_{extreme}")(color, alpha=0.4)
        np.testing.assert_allclose(cmap(value), to_rgba(color, alpha=0.4))
        np.testing.assert_allclose(getattr(cmap, f"get_{extreme}")(), cmap(value))


@pytest.mark.parametrize("copy_map", [copy, lambda cmap: cmap.copy(), lambda cmap: cmap.with_extremes()])
def test_thresholded_colormap_copies_are_independent(copy_map) -> None:
    """Changing a copy cannot mutate its source or the supplied base colormap."""
    base = get_colormap("diverging")
    cmap = with_neutral_range(base, (0., 1.), (-2., 2.))
    cmap.get_bad()
    copied = copy_map(cmap)
    copied.set_extremes(bad="lime", under="red", over="blue")
    copied.neutral_rgba[0] = 0.0
    np.testing.assert_allclose(cmap(0.5), to_rgba("#969696"))
    np.testing.assert_allclose(cmap(np.nan), base(np.nan))
    np.testing.assert_allclose(copied(np.nan), to_rgba("lime"))
    np.testing.assert_allclose(copied(-0.1), to_rgba("red"))
    np.testing.assert_allclose(copied(1.1), to_rgba("blue"))
    np.testing.assert_allclose(base(-0.1), get_colormap("diverging")(-0.1))


def test_thresholded_colormap_preserves_existing_base_extremes() -> None:
    """Explicit colors on a supplied map take precedence over a wide neutral band."""
    base = get_colormap("diverging").with_extremes(bad="magenta", under="lime", over="blue")
    cmap = with_neutral_range(base, (0., 1.), (-2., 2.))
    np.testing.assert_allclose(cmap([-0.1, np.nan, 1.1]), base([-0.1, np.nan, 1.1]))


@pytest.mark.parametrize("samples", [5, 256, 1001])
def test_thresholded_colormap_integer_indices_match_sampled_domain(samples: int) -> None:
    """Integers index the wrapper's own lookup table, rather than normalized values."""
    base = get_colormap("diverging")
    cmap = with_neutral_range(base, (0., 1.), (0.4, 0.6), samples=samples)
    positions = np.linspace(0., 1., samples)
    expected = base(positions)
    expected[(positions >= 0.4) & (positions <= 0.6)] = to_rgba("#969696")
    np.testing.assert_allclose(cmap(np.arange(samples)), expected)
    np.testing.assert_allclose(cmap([samples, -1]), cmap([1.1, -0.1]))


@pytest.mark.parametrize("alpha", [0.3, -0.5, 1.5, [0.1, 0.3, 0.8]])
@pytest.mark.parametrize("bytes", [False, True])
def test_thresholded_colormap_alpha_and_bytes_match_matplotlib(alpha, bytes: bool) -> None:
    """Thresholded colors follow standard alpha clipping, shapes, and byte conversion."""
    values = np.array([0.45, 0.5, 0.55])
    cmap = with_neutral_range("diverging", (0., 1.), (0.4, 0.6))
    expected = get_colormap("gray").with_extremes(under="#969696")(values * 0 - 1, alpha=alpha, bytes=bytes)
    np.testing.assert_allclose(cmap(values, alpha=alpha, bytes=bytes), expected)
    if bytes:
        assert cmap(values, alpha=alpha, bytes=True).dtype == np.uint8


def test_thresholded_colormap_scalar_and_masked_alpha_contract() -> None:
    """Scalar colors are tuples and missing data remain transparent even with alpha."""
    cmap = with_neutral_range("diverging", (0., 1.), (0.4, 0.6))
    assert isinstance(cmap(0.5), tuple)
    colors = cmap(np.ma.array([0.5, 0.5, np.nan], mask=[False, True, False]), alpha=[0.2, 0.4, 0.6])
    np.testing.assert_allclose(colors[0], to_rgba("#969696", alpha=0.2))
    np.testing.assert_allclose(colors[1:], np.zeros((2, 4)))
    with pytest.raises(ValueError, match="shape"):
        cmap([0.5, 0.6], alpha=[0.1, 0.2, 0.3])


def test_thresholded_colormap_resampling_keeps_exact_band() -> None:
    """Resampling changes table size without widening or dropping the exact threshold."""
    cmap = with_neutral_range("diverging", (0., 1.), (0.4999, 0.5001)).with_extremes(over="lime")
    resampled = cmap.resampled(5)
    assert resampled.N == 5
    assert resampled.name == cmap.name
    np.testing.assert_allclose(resampled(0.5), to_rgba("#969696"))
    np.testing.assert_allclose(resampled(1.1), to_rgba("lime"))
    np.testing.assert_allclose(resampled(0.49), get_colormap("diverging").resampled(5)(0.49))
    assert [0.4999, "#969696"] in as_plotly_colorscale(resampled, samples=5)


def test_thresholded_colormap_reversal_moves_band_and_swaps_extremes() -> None:
    """An asymmetric neutral interval must reverse along with the underlying colors."""
    cmap = with_neutral_range("diverging", (0., 1.), (0.1, 0.2)).with_extremes(under="lime", over="magenta")
    reversed_map = cmap.reversed()
    assert reversed_map.name == f"{cmap.name}_r"
    assert cmap.reversed("custom").name == "custom"
    np.testing.assert_allclose(reversed_map(0.85), to_rgba("#969696"))
    np.testing.assert_allclose(reversed_map(0.15), cmap.base.reversed()(0.15))
    np.testing.assert_allclose(reversed_map([-0.1, 1.1]), cmap([1.1, -0.1]))
    np.testing.assert_allclose(reversed_map.reversed()(np.linspace(0., 1., 9)), cmap(np.linspace(0., 1., 9)))


def test_thresholded_colormap_uniform_alpha() -> None:
    """Uniform transparency affects the neutral, missing, and extreme colors on a copy."""
    cmap = with_neutral_range("diverging", (0., 1.), (0.4, 0.6))
    translucent = cmap.with_alpha(0.25)
    np.testing.assert_allclose(translucent([-0.1, 0.2, 0.5, 1.1, np.nan])[:, 3], 0.25)
    np.testing.assert_allclose(translucent.get_bad()[3], 0.25)
    assert cmap(0.5)[3] == 1.0
    assert cmap(np.nan)[3] == 0.0
    with pytest.raises(ValueError, match="between 0 and 1"):
        cmap.with_alpha(1.5)
    with pytest.raises(TypeError, match="numeric"):
        cmap.with_alpha("invalid")


def test_thresholded_colormap_non_neutral_bytes_match_base() -> None:
    """Byte conversion outside the threshold must not change the base palette."""
    base = get_colormap("viridis")
    cmap = with_neutral_range(base, (0., 1.), (0.4, 0.6))
    np.testing.assert_array_equal(cmap([0.2, 0.8], alpha=0.3, bytes=True), base([0.2, 0.8], alpha=0.3, bytes=True))


@pytest.mark.parametrize("name", ["diverging", "tab20"])
@pytest.mark.parametrize("samples", [5, 1001])
def test_thresholded_colormap_keeps_base_resolution_for_float_values(name: str, samples: int) -> None:
    """The query-table size must not resample an arbitrarily selected palette."""
    base = get_colormap(name)
    cmap = with_neutral_range(base, (0., 1.), (0.4, 0.6), samples=samples)
    values = np.array([0.11, 0.23, 0.71, 0.87])
    np.testing.assert_array_equal(cmap(values), base(values))


def test_thresholded_colormap_uniform_alpha_keeps_base_resolution() -> None:
    """Changing transparency cannot implicitly resample a low-resolution palette."""
    cmap = with_neutral_range("tab20", (0., 1.), (0.4, 0.6), samples=5)
    values = np.array([0.11, 0.23, 0.5, 0.71, 0.87])
    np.testing.assert_array_equal(cmap.with_alpha(0.25)(values)[:, :3], cmap(values)[:, :3])


@pytest.mark.filterwarnings("ignore:The set_bad function:PendingDeprecationWarning")
def test_thresholded_colormap_does_not_follow_base_mutations() -> None:
    """Owning a copy keeps cached queries consistent with subsequent mapping."""
    base = get_colormap("diverging")
    cmap = with_neutral_range(base, (0., 1.), (0.4, 0.6))
    base.set_bad("lime")
    np.testing.assert_allclose(cmap.get_bad(), cmap(np.nan))
    np.testing.assert_allclose(cmap(np.nan), (0., 0., 0., 0.))
