"""Tests for style configuration and line/style cycles."""

import matplotlib as mpl

from general_plot.style import configure_style, get_color_cycle, get_linestyle_cycle, reset_color_cycles, linestyleNorm, markerNorm


def test_configure_style_presets_and_overrides():
    """Style presets update rcParams and accept underscore kwargs."""
    configure_style("publication", font_size=9)
    assert mpl.rcParams["font.size"] == 9
    assert mpl.rcParams["axes.spines.top"] is True

    configure_style("minimal", font_size=10, axes_linewidth=2.5)
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.linewidth"] == 2.5



def test_configure_style_twice_gives_the_same_rcparams_and_keeps_the_backend():
    """A second call with the same arguments changes nothing and does not reset the backend (a notebook calls it after the import)."""
    backend = mpl.rcParams["backend"]
    configure_style(font_size=8)
    first   = {key: str(value) for key, value in mpl.rcParams.items()}
    configure_style(font_size=8)
    assert {key: str(value) for key, value in mpl.rcParams.items()} == first
    assert mpl.rcParams["backend"] == backend
    assert mpl.rcParams["axes.labelsize"] == 8 and mpl.rcParams["xtick.labelsize"] == 7

def test_cycles_and_normalizers():
    """Color/linestyle cycles iterate and string normalizers map aliases."""
    assert next(get_color_cycle("TABLEAU")) is not None
    reset_color_cycles("TABLEAU")
    assert next(get_color_cycle("Plastic")) is not None

    assert next(get_linestyle_cycle("Normal")) == "-"
    assert markerNorm("M0") == "o"
    assert linestyleNorm("L1") == "--"


# ---------------------------------------------
#! EOF
# ---------------------------------------------
