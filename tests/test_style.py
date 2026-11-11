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



def test_configure_style_twice_gives_the_same_rcparams_and_keeps_the_session_keys():
    """A second call with the same arguments changes nothing and keeps the session keys backend and interactive."""
    backend = mpl.rcParams["backend"]
    mpl.interactive(True)                                       # as in a Jupyter session; regression: the reset set it to False and hid the figures
    configure_style(font_size=8)
    first   = {key: str(value) for key, value in mpl.rcParams.items()}
    configure_style(font_size=8)
    assert {key: str(value) for key, value in mpl.rcParams.items()} == first
    assert mpl.rcParams["backend"] == backend and mpl.is_interactive()
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


def test_printed_sizes_are_accessible_and_apply_to_new_figures():
    import numpy as np
    from general_plot import DEFAULTS, Plotter
    before = dict(DEFAULTS)
    settings = configure_style()
    assert settings.one_column == 246 / 70
    assert settings.two_column == 510 / 70
    assert settings.width == settings.one_column
    assert settings.height == 0.5 * settings.width
    assert (settings.font_size, settings.legend_size, settings.letter_size, settings.inset_size) == (9, 6.5, 7.5, 7)
    assert mpl.rcParams['legend.fontsize'] == settings.legend_size
    fig, _ = Plotter.get_subplots()
    assert np.allclose(fig.get_size_inches(), [settings.width, settings.height])
    assert DEFAULTS == before


def test_printed_sizes_follow_overrides_without_leaking_between_calls():
    import numpy as np
    from general_plot import Plotter
    settings = configure_style(font_size=10, column_pt=250, text_pt=520, points_per_inch=72.27,
                               width=5, legend_size=8, letter_size=9, inset_size=7)
    assert settings.one_column == 250 / 72.27
    assert settings.two_column == 520 / 72.27
    assert settings.width == 5 and settings.height == 2.5
    assert (settings.legend_size, settings.letter_size, settings.inset_size) == (8, 9, 7)
    assert mpl.rcParams['font.size'] == 10 and mpl.rcParams['legend.fontsize'] == 8
    fig, _ = Plotter.get_subplots(sizex=6, sizey=3)
    assert np.allclose(fig.get_size_inches(), [6, 3])
    assert configure_style().font_size == 9
    assert configure_style().width == 246 / 70
