"""Color palettes, color conversion, and perceptual manipulation utilities."""

from __future__ import annotations

import colorsys
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np


# ---------------------------------------------


def get_cmap_safe(cmap: Union[str, mpl.colors.Colormap]) -> mpl.colors.Colormap:
    """Retrieve a colormap object from the current Matplotlib registry.

    Modern alias of ``mpl.colormaps[cmap]`` that also accepts a Colormap
    instance directly. Kept for API compatibility.
    """
    if isinstance(cmap, mpl.colors.Colormap):
        return cmap
    return mpl.colormaps[cmap]


DEFAULT_PALETTES: Dict[str, List[str]] = {
    # Wong (2011) Nature Methods – gold standard for colorblind-friendly plots
    "wong": [
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    ],
    # Identical to wong; alias used by Okabe & Ito (2008)
    "okabe": [
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    ],
    # Paul Tol muted qualitative
    "tol": [
        "#332288", "#88CCEE", "#44AA99", "#117733",
        "#999933", "#DDCC77", "#CC6677", "#882255",
        "#AA4499", "#DDDDDD",
    ],
    # Paul Tol bright
    "tol_bright": [
        "#4477AA", "#EE6677", "#228833", "#CCBB44",
        "#66CCEE", "#AA3377", "#BBBBBB",
    ],
    # IBM Carbon accessible palette (5 colors)
    "ibm": ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"],
    # seaborn colorblind-10
    "colorblind": [
        "#0173B2", "#DE8F05", "#029E73", "#D55E00",
        "#CC78BC", "#CA9161", "#FBAFE4", "#949494",
        "#ECE133", "#56B4E9",
    ],
    # seaborn deep-10
    "deep": [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
        "#CCB974", "#64B5CD",
    ],
    # seaborn muted-10
    "muted": [
        "#4878D0", "#EE854A", "#6ACC65", "#D65F5F",
        "#956CB4", "#8C613C", "#DC7EC0", "#797979",
        "#D5BB67", "#82C6E2",
    ],
    # Standard Matplotlib cycles
    "tableau": list(mcolors.TABLEAU_COLORS.values()),
    "classic": [
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
        "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
        "#BCBD22", "#17BECF",
    ],
    # Journal-style palettes
    "nature": [
        "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
        "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
        "#7E6148", "#B09C85",
    ],
    "science": [
        "#3B4992", "#EE0000", "#008B45", "#631879",
        "#008280", "#BB0021", "#5F559B", "#A20056",
        "#808180", "#1B1919",
    ],
    "pastel": [
        "#AEC6CF", "#FFD1DC", "#B5EAD7", "#FFDAC1",
        "#C7CEEA", "#E2F0CB", "#F8C8D4", "#D4E6F1",
    ],
    "sunset": [
        "#364B9A", "#4A7BB7", "#98CAE1", "#EAECCC",
        "#FEDA8B", "#FDB366", "#F67E4B", "#DD3D2D", "#A50026",
    ],
}


def palette(name: str = "tableau", n: Optional[int] = None) -> List[str]:
    """Return a named color palette as a list of hex strings.

    Parameters
    ----------
    name : str, default='tableau'
        Palette name (wong, okabe, tol, tol_bright, ibm, colorblind, deep,
        muted, tableau, classic, nature, science, pastel, sunset).
    n : int, optional
        Return exactly `n` colors. Cycles when `n > len(palette)`.
    """
    p = DEFAULT_PALETTES.get(name)
    if p is None:
        raise ValueError(f"Unknown palette '{name}'. Available: {sorted(DEFAULT_PALETTES.keys())}")
    if n is None:
        return list(p)
    return [p[i % len(p)] for i in range(n)]


def palette_cycle(name: str = "tableau") -> itertools.cycle:
    """Return an infinite cycle iterator over a named palette."""
    return itertools.cycle(palette(name))


def to_rgba(color: Any, alpha: Optional[float] = None) -> Tuple[float, float, float, float]:
    """Convert any Matplotlib-compatible color spec to an (r, g, b, a) tuple."""
    r, g, b, a = mcolors.to_rgba(color)
    if alpha is not None:
        a = float(alpha)
    return (r, g, b, a)


def to_hex(color: Any, keep_alpha: bool = False) -> str:
    """Convert any Matplotlib-compatible color spec to a hex string."""
    return mcolors.to_hex(color, keep_alpha=keep_alpha)


def adjust_color(color: Any, *, lighten: float = 0.0, darken: float = 0.0, saturate: float = 0.0, desaturate: float = 0.0, alpha: Optional[float] = None) -> Tuple[float, float, float, float]:
    """Perceptually adjust a color in HLS space.

    Parameters
    ----------
    color : color spec
        Any Matplotlib-compatible color.
    lighten : float, default=0.0
        Push lightness toward 1 (white).
    darken : float, default=0.0
        Push lightness toward 0 (black).
    saturate : float, default=0.0
        Push saturation toward 1.
    desaturate : float, default=0.0
        Push saturation toward 0 (grey).
    alpha : float, optional
        Override alpha channel (0–1).
    """
    r, g, b, a = mcolors.to_rgba(color)
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)

    lightness = lightness + (1.0 - lightness) * max(0.0, min(1.0, float(lighten)))
    lightness = lightness * (1.0 - max(0.0, min(1.0, float(darken))))
    s = s + (1.0 - s) * max(0.0, min(1.0, float(saturate)))
    s = s * (1.0 - max(0.0, min(1.0, float(desaturate))))

    r2, g2, b2 = colorsys.hls_to_rgb(h, lightness, s)
    return (r2, g2, b2, float(alpha) if alpha is not None else a)


def lighten(color: Any, amount: float = 0.3) -> Tuple[float, float, float, float]:
    """Return a lightened version of `color` (push lightness toward white)."""
    return adjust_color(color, lighten=amount)


def darken(color: Any, amount: float = 0.3) -> Tuple[float, float, float, float]:
    """Return a darkened version of `color` (push lightness toward black)."""
    return adjust_color(color, darken=amount)


def desaturate(color: Any, amount: float = 0.5) -> Tuple[float, float, float, float]:
    """Return a desaturated (greyed-out) version of `color`."""
    return adjust_color(color, desaturate=amount)


def with_alpha(color: Any, a: float = 0.5) -> Tuple[float, float, float, float]:
    """Return `color` with a modified alpha channel."""
    return to_rgba(color, alpha=a)


def blend(c1: Any, c2: Any, t: float = 0.5, *, n: Optional[int] = None) -> Union[Tuple[float, float, float, float], List[Tuple[float, float, float, float]]]:
    """Linearly interpolate between two colors in linear RGB space."""
    a = np.array(mcolors.to_rgba(c1))
    b = np.array(mcolors.to_rgba(c2))
    if n is not None:
        ts = np.linspace(0.0, 1.0, int(n))
        return [tuple((1.0 - s) * a + s * b) for s in ts]
    return tuple((1.0 - float(t)) * a + float(t) * b)


def n_colors(n: int, cmap: Union[str, mpl.colors.Colormap] = "viridis", vmin: float = 0.0, vmax: float = 1.0, *, as_hex: bool = False) -> List[Any]:
    """Sample `n` evenly-spaced colors from a colormap."""
    cmap_obj    = get_cmap_safe(cmap)
    values      = np.linspace(float(vmin), float(vmax), int(n))
    if as_hex:
        return [mcolors.to_hex(cmap_obj(v)) for v in values]
    return [cmap_obj(v) for v in values]


def cmap(name: Union[str, mpl.colors.Colormap], vmin: float = 0.0, vmax: float = 1.0, n: int = 256) -> mpl.colors.Colormap:
    """Return `name` restricted to the [vmin, vmax] segment of the colormap.

    The returned Colormap maps the normalized interval [0, 1] onto the sampled
    segment, so values can be passed directly (e.g. ``cmap(norm(x))``). This is
    the continuous counterpart of `n_colors`, which samples the same segment.
    """
    base        = get_cmap_safe(name)
    samples     = base(np.linspace(float(vmin), float(vmax), int(n)))
    return mcolors.LinearSegmentedColormap.from_list(f"{base.name}_{vmin:g}_{vmax:g}", samples)


# ---------------------------------------------
#! EOF
# ---------------------------------------------
