"""Tests for color manipulation and palette helpers."""

import matplotlib as mpl
import numpy as np

from general_plot.colors import blend, cmap, darken, desaturate, get_cmap_safe, lighten, n_colors, palette, palette_cycle, to_hex, to_rgba, with_alpha


def test_conversions_and_adjustments():
    """RGBA/hex conversion and perceptual adjustment helpers behave sanely."""
    assert to_rgba("#FF0000", alpha=0.5) == (1.0, 0.0, 0.0, 0.5)
    assert to_hex((1.0, 0.0, 0.0, 1.0)).lower() == "#ff0000"

    base        = "#0000FF"  # pure blue
    lighter     = lighten(base, 0.5)
    darker      = darken(base, 0.5)
    assert lighter[0] > 0.0 or lighter[1] > 0.0  # shifted toward white
    assert darker[2] < 1.0  # shifted toward black
    assert with_alpha(base, 0.3)[3] == 0.3
    assert desaturate(base, 0.5) is not None


def test_palettes_blend_and_cmaps():
    """Palette lookup, blending, colormap sampling, and safe cmap retrieval."""
    assert len(palette("wong")) == 8
    assert len(palette("nature", n=4)) == 4
    assert next(palette_cycle("science")) is not None

    assert isinstance(blend("red", "blue", t=0.5), tuple)
    assert len(blend("red", "blue", n=5)) == 5

    samples     = n_colors(7, cmap="viridis", as_hex=True)
    assert len(samples) == 7
    assert all(c.startswith("#") for c in samples)

    assert isinstance(get_cmap_safe("plasma"), mpl.colors.Colormap)


def test_cmap_truncation():
    """cmap() restricts a colormap to [vmin, vmax] and stays callable."""
    truncated   = cmap("inferno", vmin=0.35, vmax=0.85)
    assert isinstance(truncated, mpl.colors.Colormap)

    # Normalized 0..1 map onto the sampled [vmin, vmax] segment endpoints.
    base        = mpl.colormaps["inferno"]
    assert np.allclose(truncated(0.0), base(0.35))
    assert np.allclose(truncated(1.0), base(0.85))

    # Interior samples interpolate within the segment (monotone in value).
    vals        = [truncated(t)[0] for t in np.linspace(0.0, 1.0, 11)]
    refs        = [base(v)[0] for v in np.linspace(0.35, 0.85, 11)]
    assert np.allclose(vals, refs, atol=2e-2)


# ---------------------------------------------
#! EOF
# ---------------------------------------------
