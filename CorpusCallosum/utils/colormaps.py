"""Shared colormap helpers for corpus callosum visualizations."""

from collections.abc import Sequence

import matplotlib
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_hex, to_rgba

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


class NeutralRangeColormap(Colormap):
    """Colormap wrapper that applies an exact neutral interval."""

    def __init__(
        self,
        base: Colormap,
        neutral_min: float,
        neutral_max: float,
        neutral_color: str,
        samples: int,
    ) -> None:
        super().__init__(name=f"{base.name}_thresholded", N=samples)
        self.base = base
        self.neutral_min = neutral_min
        self.neutral_max = neutral_max
        self.neutral_rgba = np.asarray(to_rgba(neutral_color))
        self._preserve_over = False
        self._preserve_under = False

    def __call__(self, values, alpha=None, bytes=False):
        """Map normalized values, replacing the neutral interval exactly."""
        normalized = np.asarray(values)
        colors = np.array(self.base(values, alpha=alpha, bytes=False), copy=True)
        neutral = self.neutral_rgba.copy()
        if alpha is not None and np.isscalar(alpha):
            neutral[3] = alpha
        mask = (
            (normalized >= self.neutral_min)
            & (normalized <= self.neutral_max)
            & ~np.ma.getmaskarray(values)
        )
        if self._preserve_over:
            mask &= normalized <= 1.0
        if self._preserve_under:
            mask &= normalized >= 0.0
        if colors.ndim == 1:
            if bool(mask):
                colors = neutral
        else:
            colors[mask] = neutral
        if bytes:
            return np.asarray(np.round(colors * 255), dtype=np.uint8)
        return colors

    def with_extremes(self, *, bad=None, under=None, over=None):
        """Preserve special colors when wrapping the base colormap."""
        result = NeutralRangeColormap(
            self.base.with_extremes(bad=bad, under=under, over=over),
            self.neutral_min, self.neutral_max, self.neutral_rgba, self.N,
        )
        result._preserve_over = self._preserve_over or over is not None
        result._preserve_under = self._preserve_under or under is not None
        return result


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
    scale = [[float(position), to_hex(cmap(position), keep_alpha=False)] for position in positions]
    if isinstance(cmap, NeutralRangeColormap):
        lower = max(0.0, cmap.neutral_min)
        upper = min(1.0, cmap.neutral_max)
        if lower <= upper:
            neutral = to_hex(cmap.neutral_rgba, keep_alpha=False)
            if 0.0 < lower < 1.0:
                scale.extend(
                    [
                        [lower, to_hex(cmap.base(lower), keep_alpha=False)],
                        [lower, neutral],
                    ]
                )
            if 0.0 < upper < 1.0:
                scale.extend(
                    [
                        [upper, neutral],
                        [upper, to_hex(cmap.base(upper), keep_alpha=False)],
                    ]
                )
            scale.sort(key=lambda entry: float(entry[0]))
    return scale


def resolve_color_range(
    values: np.ndarray,
    color_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return finite, non-degenerate color limits for an overlay."""
    if color_range is not None:
        data_min, data_max = map(float, color_range)
        if not np.isfinite((data_min, data_max)).all():
            raise ValueError("Color range values must be finite.")
        if data_min >= data_max:
            raise ValueError("Color range minimum must be smaller than its maximum.")
        return data_min, data_max

    finite_values = np.asarray(values)[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot determine a color range without finite overlay values.")

    data_min = float(np.min(finite_values))
    data_max = float(np.max(finite_values))
    if data_min == data_max:
        half_width = max(abs(data_min), 1.0) * 1e-6
        data_min -= half_width
        data_max += half_width
    return data_min, data_max


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
    if neutral_min == neutral_max:
        return cmap
    normalized_min = (neutral_min - data_min) / (data_max - data_min)
    normalized_max = (neutral_max - data_min) / (data_max - data_min)
    return NeutralRangeColormap(cmap, normalized_min, normalized_max, neutral_color, samples)
