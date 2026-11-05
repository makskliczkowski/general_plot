"""Tests for the shared figure defaults."""

import pytest

from general_plot import DEFAULTS, Defaults, Plotter, default


def test_defaults_read_as_items_and_attributes_and_overrides_do_not_leak():
    D = default()
    assert isinstance(D, Defaults) and D["width"] == D.width == DEFAULTS["width"] == 7.0
    assert D.cmap == "viridis" and D.floor == 1e-12 and D.formats == ("png", "pdf")

    narrow = default(width=3.5)
    narrow.height = 2.0
    assert narrow.width == 3.5 and narrow["height"] == 2.0
    assert DEFAULTS["width"] == 7.0 and DEFAULTS["height"] == 2.8 and default().height == 2.8  # the shared table is untouched
    assert narrow.copy() == narrow and narrow.copy() is not narrow


def test_unknown_names_fail_loudly():
    with pytest.raises(KeyError, match="unknown defaults"):
        default(widht=3.0)
    with pytest.raises(AttributeError, match="unknown default"):
        default().widht


def test_defaults_feed_the_plotting_helpers():
    D = default()
    assert len(Plotter.n_colors(4, D.cmap, *D.cmap_range)) == 4
    assert str(Plotter.mask_below([1e-3, 1e-15], D.floor)) == "[0.001   nan]"
