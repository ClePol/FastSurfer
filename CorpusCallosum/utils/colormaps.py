"""Shared colormap helpers for corpus callosum visualizations."""

from collections.abc import Sequence
from numbers import Real

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


class NeutralRangeColormap(Colormap):
    """Matplotlib colormap with an exact, rather than quantized, neutral interval."""

    def __init__(
        self,
        base: Colormap,
        neutral_min: float,
        neutral_max: float,
        neutral_color: str,
        samples: int,
    ) -> None:
        super().__init__(name=f"{base.name}_thresholded", N=samples)
        self.base = base.copy()
        self.neutral_min = neutral_min
        self.neutral_max = neutral_max
        self.neutral_rgba = np.asarray(to_rgba(neutral_color))
        # Matplotlib distinguishes explicit extremes from endpoint defaults only
        # through these attributes. Explicit colors take precedence over neutrality.
        self._preserve_over = base._rgba_over is not None
        self._preserve_under = base._rgba_under is not None
        self.colorbar_extend = base.colorbar_extend

    def _init(self):
        """Initialize the table used by inherited queries, equality, and previews."""
        self._lut = np.empty((self.N + 3, 4))
        self._lut[:self.N] = self(np.linspace(0.0, 1.0, self.N))
        self._lut[self._i_bad] = self.base.get_bad()
        self._lut[self._i_under] = self.base.get_under() if self._preserve_under else self(np.nextafter(0.0, -np.inf))
        self._lut[self._i_over] = self.base.get_over() if self._preserve_over else self(np.nextafter(1.0, np.inf))
        self._isinit = True

    def __call__(self, values, alpha=None, bytes=False):
        """Map normalized values, replacing the neutral interval exactly."""
        normalized = np.asarray(values)
        indexed = normalized.dtype.kind in "biu"
        mask_under = normalized < 0
        mask_over = normalized >= self.N if indexed else normalized > 1.0
        if indexed:
            normalized = normalized.astype(float) / max(self.N - 1, 1)
            base_values = np.where(mask_over, np.maximum(normalized, np.nextafter(1.0, np.inf)), normalized)
        else:
            base_values = values
        mask_bad = np.ma.getmaskarray(values) | np.isnan(normalized)
        colors = np.array(self.base(np.ma.array(base_values, mask=mask_bad), alpha=alpha), copy=True)
        mask = (
            (normalized >= self.neutral_min)
            & (normalized <= self.neutral_max)
            & ~mask_bad
        )
        if self._preserve_over:
            mask &= ~mask_over
        if self._preserve_under:
            mask &= ~mask_under
        colors[..., :3][mask] = self.neutral_rgba[:3]
        if alpha is None:
            colors[..., 3][mask] = self.neutral_rgba[3]
        if bytes:
            colors = (colors * 255).astype(np.uint8)
        return colors if np.iterable(values) else tuple(colors)

    def __copy__(self):
        """Keep the base map and neutral color independent when copying."""
        result = super().__copy__()
        result.base = self.base.copy()
        result.neutral_rgba = self.neutral_rgba.copy()
        return result

    def set_bad(self, color="k", alpha=None):
        """Set the missing-data color on this map only."""
        self.set_extremes(bad=to_rgba(color, alpha))

    def set_under(self, color="k", alpha=None):
        """Set an explicit below-range color, overriding the neutral band."""
        self.set_extremes(under=to_rgba(color, alpha))

    def set_over(self, color="k", alpha=None):
        """Set an explicit above-range color, overriding the neutral band."""
        self.set_extremes(over=to_rgba(color, alpha))

    def set_extremes(self, *, bad=None, under=None, over=None):
        """Update special colors and invalidate the cached query table."""
        self.base = self.base.with_extremes(bad=bad, under=under, over=over)
        self._preserve_over |= over is not None
        self._preserve_under |= under is not None
        self._isinit = False

    def with_extremes(self, *, bad=None, under=None, over=None):
        """Preserve special colors when wrapping the base colormap."""
        result = self.copy()
        result.set_extremes(bad=bad, under=under, over=over)
        return result

    def resampled(self, lutsize):
        """Resample the base map without quantizing the neutral interval."""
        result = NeutralRangeColormap(
            self.base.resampled(lutsize), self.neutral_min, self.neutral_max, self.neutral_rgba, lutsize,
        )
        result.name = self.name
        result._preserve_over = self._preserve_over
        result._preserve_under = self._preserve_under
        result.colorbar_extend = self.colorbar_extend
        return result

    def reversed(self, name=None):
        """Reverse the base colors, neutral interval, and explicit extremes."""
        result = NeutralRangeColormap(
            self.base.reversed(), 1.0 - self.neutral_max, 1.0 - self.neutral_min, self.neutral_rgba, self.N,
        )
        result.name = f"{self.name}_r" if name is None else name
        result._preserve_over = self._preserve_under
        result._preserve_under = self._preserve_over
        result.colorbar_extend = self.colorbar_extend
        return result

    def with_alpha(self, alpha):
        """Return a copy with uniform transparency, including neutral and bad colors."""
        if not isinstance(alpha, Real):
            raise TypeError("'alpha' must be numeric")
        if not 0 <= alpha <= 1:
            raise ValueError("'alpha' must be between 0 and 1, inclusive")
        result = self.copy()
        colors = self.base(np.linspace(0.0, 1.0, self.base.N))
        colors[:, 3] = alpha
        result.base = ListedColormap(colors, name=self.base.name).with_extremes(
            bad=to_rgba(self.base.get_bad(), alpha),
            under=to_rgba(self.base.get_under(), alpha),
            over=to_rgba(self.base.get_over(), alpha),
        )
        result.neutral_rgba[3] = alpha
        result._isinit = False
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
    """Return a colormap with an exact neutral interval in data coordinates."""
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
