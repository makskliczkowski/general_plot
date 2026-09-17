"""Tests for AxesList indexing, named panels, and IgnoredAxis proxying."""

import matplotlib.pyplot as plt

from general_plot import AxesList, IgnoredAxis, Plotter


def test_axes_list_indexing_and_panels():
    """1D/2D indexing, named panels, and select/rename work on AxesList."""
    fig, axes   = Plotter.get_subplots(2, 2, named_panels={"TL": (0, 0), "BR": (1, 1)})
    assert isinstance(axes, AxesList)
    assert axes.shape == (2, 2)
    assert axes[0, 0] is axes[0]
    assert axes[1, 1] is axes[3]
    assert axes["TL"] is axes[0, 0]

    axes.rename_panel("TL", "TopLeft")
    assert axes.has_panel("TopLeft")
    assert len(axes.select("TopLeft", "BR")) == 2

    plt.close(fig)


def test_ignored_axis_noop():
    """Disabling an axis yields an IgnoredAxis whose method calls are safe no-ops."""
    fig, axes   = Plotter.get_subplots(2, 2)
    axes.disable((0, 1))

    disabled    = axes[0, 1]
    assert isinstance(disabled, IgnoredAxis)
    assert disabled.is_disabled
    assert not bool(disabled)

    # Any method call is a silent no-op, chains included
    disabled.plot([1, 2], [3, 4]).set_xlabel("X")
    disabled.tick_params(axis="both")

    plt.close(fig)


def test_axes_list_batch_apply():
    """Methods applied to AxesList propagate to every panel."""
    fig, axes   = Plotter.get_subplots(1, 3)
    axes.set_title("Universal Title")
    for ax in axes:
        title_text = ax.get_title(loc="left") or ax.get_title(loc="center") or ax.get_title(loc="right")
        assert title_text == "Universal Title"

    plt.close(fig)


# ---------------------------------------------
#! EOF
# ---------------------------------------------
