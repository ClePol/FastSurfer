"""Shared colormap helpers for corpus callosum visualizations."""

from collections.abc import Sequence

import matplotlib
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, to_hex, to_rgba

_PREDEFINED_COLORS: dict[str, Sequence[tuple[float, str]]] = {
    "red_to_blue": (
        (0.0, "#ff0000"),
        (0.25, "#ffa500"),
        (0.5, "#969696"),
        (0.75, "#add8e6"),
        (1.0, "#0000ff"),
    ),
    "blue_to_red": (
        (0.0, "#0000ff"),
        (0.25, "#add8e6"),
        (0.5, "#969696"),
        (0.75, "#ffa500"),
        (1.0, "#ff0000"),
    ),
    "red_to_yellow": (
        (0.0, "#ff0000"),
        (0.33, "#ff5500"),
        (0.66, "#ffaa00"),
        (1.0, "#ffff00"),
    ),
    "yellow_to_red": (
        (0.0, "#ffff00"),
        (0.33, "#ffaa00"),
        (0.66, "#ff5500"),
        (1.0, "#ff0000"),
    ),
}

PREDEFINED_COLORMAPS = (*_PREDEFINED_COLORS, "diverging")

# Exact normalized color anchors from WhipperSnapPy's ``heat_color`` overlay
# mapping, expressed on Matplotlib's [0, 1] colormap domain.
_WHIPPERSNAPPY_HEAT_COLORS: Sequence[tuple[float, tuple[float, float, float]]] = (
    (0.0, (0.0, 1.0, 1.0)),
    (1.0 / 3.0, (0.0, 0.0, 1.0)),
    (0.5, (0.0, 0.0, 0.5625)),
    (0.5, (0.5625, 0.0, 0.0)),
    (2.0 / 3.0, (1.0, 0.0, 0.0)),
    (1.0, (1.0, 1.0, 0.0)),
)


def get_colormap(name: str) -> Colormap:
    """Return a predefined or Matplotlib colormap.

    ``diverging`` matches WhipperSnapPy's FreeSurfer-style heat colors:
    negative values run cyan to dark blue and positive values run dark red to
    yellow. Any registered Matplotlib colormap name is also accepted.
    """
    if name == "diverging":
        return LinearSegmentedColormap.from_list(name, _WHIPPERSNAPPY_HEAT_COLORS)
    if name in _PREDEFINED_COLORS:
        return LinearSegmentedColormap.from_list(name, _PREDEFINED_COLORS[name])
    try:
        return matplotlib.colormaps[name]
    except KeyError as error:
        presets = ", ".join(PREDEFINED_COLORMAPS)
        raise ValueError(
            f"Unknown colormap '{name}'. Use a predefined colormap ({presets}) "
            "or a registered Matplotlib colormap name."
        ) from error


def as_plotly_colorscale(colormap: str | Colormap, samples: int = 256) -> list[list[float | str]]:
    """Convert a Matplotlib colormap into a Plotly colorscale."""
    if samples < 2:
        raise ValueError("A Plotly colorscale requires at least two samples.")
    cmap = get_colormap(colormap) if isinstance(colormap, str) else colormap
    positions = np.linspace(0.0, 1.0, samples)
    return [[float(position), to_hex(cmap(position), keep_alpha=False)] for position in positions]


def with_neutral_range(
    colormap: str | Colormap,
    color_range: tuple[float, float],
    neutral_range: tuple[float, float],
    neutral_color: str = "#969696",
    samples: int = 256,
) -> Colormap:
    """Return a sampled colormap with a neutral interval in data coordinates."""
    data_min, data_max = color_range
    neutral_min, neutral_max = neutral_range
    if data_min >= data_max:
        raise ValueError("Color range minimum must be smaller than its maximum.")
    if neutral_min > neutral_max:
        raise ValueError("Neutral range minimum must not exceed its maximum.")
    if samples < 2:
        raise ValueError("A thresholded colormap requires at least two samples.")

    cmap = get_colormap(colormap) if isinstance(colormap, str) else colormap
    positions = np.linspace(0.0, 1.0, samples)
    data_values = data_min + positions * (data_max - data_min)
    colors = cmap(positions)
    colors[(data_values >= neutral_min) & (data_values <= neutral_max)] = to_rgba(neutral_color)
    return ListedColormap(colors, name=f"{cmap.name}_thresholded")
