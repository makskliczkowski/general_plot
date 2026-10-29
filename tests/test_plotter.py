"""Tests for Plotter core plotting methods and layouts."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from general_plot import Plotter


def test_plotting_primitives():
    """Core 1D plotting functions render onto axes."""
    fig, axes   = Plotter.get_subplots(2, 2)
    x           = np.linspace(0, 10, 50)
    y           = np.sin(x)

    Plotter.plot(axes[0], x, y, color="C0", label="Sin")
    Plotter.scatter(axes[1], x, y, s=10, color="C1")
    Plotter.errorbar(axes[2], x[:10], y[:10], yerr=0.1, color="C2")
    Plotter.semilogy(axes[3], x, np.exp(x / 5), color="C3")

    assert len(axes[0].get_lines()) == 1
    assert len(axes[1].collections) == 1
    assert axes[3].get_yscale() == "log"

    plt.close(fig)


def test_set_ax_params_and_power_law_guide():
    """Axis styling and log-log power-law guides render correctly."""
    fig, axes   = Plotter.get_subplots(1, 1)
    ax          = axes[0]

    Plotter.set_ax_params(
        ax,
        xlabel="Time (s)",
        ylabel="Amplitude",
        title="Oscillator",
        xlim=(0, 10),
        ylim=(-1.5, 1.5),
        grid=True,
    )
    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_xlim() == (0.0, 10.0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    line    = Plotter.power_law_guide(ax, x_range=(1.0, 10.0), exponent=-2.0, anchor=(1.0, 10.0))
    assert line is not None
    assert len(ax.get_lines()) == 1

    plt.close(fig)


def test_colorbar_and_grid_builder():
    """add_colorbar links to the axes; GridBuilder lays out multi-row panels."""
    fig, axes   = Plotter.get_subplots(1, 1)
    im          = axes[0].imshow(np.random.rand(10, 10), cmap="viridis")
    cbar, cax   = Plotter.add_colorbar(fig, pos=[0.92, 0.15, 0.02, 0.7], mappable=im, label="Intensity")
    assert cbar.ax is cax
    plt.close(fig)

    builder     = Plotter.GridBuilder(figsize=(8, 6))
    builder.add_row(ncols=1, height_ratio=1.0)
    builder.add_row(ncols=2, height_ratio=2.0)
    fig, axes   = builder.build()
    assert len(axes) == 2
    assert len(axes[1]) == 2
    plt.close(fig)


def test_save_fig(tmp_path):
    """save_fig writes the figure to disk in the requested format."""
    fig, axes   = Plotter.get_subplots(1, 1)
    Plotter.plot(axes[0], [1, 2, 3], [4, 5, 6])
    Plotter.save_fig(str(tmp_path), "out.pdf", format="pdf")
    assert (tmp_path / "out.pdf").exists()
    plt.close(fig)


def test_default_color_auto_cycles():
    """Curve primitives default to None color so matplotlib advances the cycle."""
    fig, axes   = Plotter.get_subplots(1, 1)
    ax          = axes[0]
    x           = np.linspace(0, 1, 10)

    Plotter.plot(ax, x, x)
    Plotter.plot(ax, x, x**2)
    Plotter.plot(ax, x, x**3, color="C0")

    colors  = [line.get_color() for line in ax.get_lines()]
    assert len(colors) == 3
    assert colors[0] != colors[1], "plot() should auto-advance the color cycle"
    assert colors[2] == "C0", "explicit color='C0' must be honored"

    fig2, axes2  = Plotter.get_subplots(1, 1)
    ax2          = axes2[0]
    Plotter.scatter(ax2, x, x)
    Plotter.scatter(ax2, x, x**2, c="C1")
    assert ax2.collections[0].get_facecolor() is not None
    assert np.allclose(np.asarray(ax2.collections[1].get_facecolor())[0], Plotter.to_rgba("C1")[:4])

    plt.close(fig)
    plt.close(fig2)


def test_maxelems_thins_curves_and_errorbars():
    """maxelems caps the number of drawn samples, keeping arrays aligned."""
    fig, axes   = Plotter.get_subplots(1, 1)
    ax          = axes[0]
    x           = np.linspace(0, 10, 1000)
    y           = np.sin(x)

    Plotter.plot(ax, x, y, maxelems=50)
    lines       = ax.get_lines()
    assert len(lines[-1].get_xdata()) == 50
    assert len(lines[-1].get_ydata()) == 50

    Plotter.semilogy(ax, x, np.exp(x / 10), maxelems=30)
    assert len(ax.get_lines()[-1].get_xdata()) == 30

    Plotter.loglog(ax, np.geomspace(1, 1000, 500), np.geomspace(1, 100, 500), maxelems=25)
    assert len(ax.get_lines()[-1].get_xdata()) == 25

    # errorbars thin their arrays, including asymmetric (2, N) errors
    Plotter.errorbar(
        ax, x[:200], y[:200],
        yerr=np.vstack([np.full(200, 0.1), np.full(200, 0.2)]),
        maxelems=40,
    )
    n_pts   = len(ax.get_lines()[-1].get_xdata())
    assert 0 < n_pts <= 40

    # without maxelems nothing is thinned
    Plotter.plot(ax, x, y)
    assert len(ax.get_lines()[-1].get_xdata()) == 1000

    plt.close(fig)


# ---------------------------------------------
#! EOF
# ---------------------------------------------


def test_tick_style_shared_between_axes_and_colorbar():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from general_plot import Plotter
    fig, ax = plt.subplots()
    mesh    = Plotter.pcolormesh(ax, np.arange(4), np.arange(3), np.random.rand(3, 4), scale="linear")
    Plotter.set_ax_params(ax)
    cbar, _ = Plotter.add_colorbar(fig, [0.9, 0.1, 0.02, 0.8], mappable=mesh)
    tx, tc  = ax.xaxis.get_major_ticks()[0].tick1line, cbar.ax.yaxis.get_major_ticks()[0].tick2line
    assert tx.get_markersize() == tc.get_markersize() == Plotter.TICK_LENGTH_MAJOR
    assert tx.get_markeredgewidth() == tc.get_markeredgewidth() == Plotter.TICK_WIDTH
    plt.close(fig)


def test_pcolormesh_band_bar_smoke():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from general_plot import Plotter
    fig, ax = plt.subplots()
    z       = np.random.rand(5, 7) + 0.1
    mesh    = Plotter.pcolormesh(ax, np.arange(7), np.linspace(0, 1, 5), z, scale="log")
    assert mesh.get_array().shape == z.shape and mesh.norm.vmin > 0
    Plotter.band(ax, np.arange(7), np.random.rand(20, 7))
    Plotter.bar(ax, [1, 2, 3], [0.5, 0.2, 0.1], log=True)
    plt.close(fig)


def test_add_colorbar_placement_and_mappable_forms():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from general_plot import Plotter
    fig, axs = plt.subplots(1, 2)
    norm     = mcolors.LogNorm(1e-3, 1)
    c1, _    = Plotter.add_colorbar(fig, ax=axs, mappable=norm, cmap="Blues", label="a")          # norm only, next to axes
    c2, cax  = Plotter.add_colorbar(fig, [0.95, 0.1, 0.02, 0.8], norm=norm)                       # norm kwarg, explicit position
    c3, _    = Plotter.add_colorbar(fig, ax=axs[0], mappable=np.random.rand(3, 3))                # array
    assert c1.norm is norm and c2.ax is cax
    try:
        Plotter.add_colorbar(fig, mappable=norm)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError without pos and ax")
    plt.close(fig)


def test_set_ticks_keeps_the_limits():
    """A tick (or a replaced label) outside the view must not widen it; the autoscale state is kept."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from general_plot import Plotter
    fig, ax = plt.subplots()
    ax.set_ylim(-2.6, -2.0)
    Plotter.set_ticks(ax, 'y', replace={-1.3: "E/N"})
    assert ax.get_ylim() == (-2.6, -2.0)
    Plotter.set_ticks(ax, 'y', [-2.5, -1.0, 3.0])
    assert ax.get_ylim() == (-2.6, -2.0)
    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [0, 1])
    Plotter.set_ticks(ax2, 'x', [0.0, 0.5])
    assert ax2.get_autoscalex_on()
    plt.close(fig)
    plt.close(fig2)


def test_set_ticks_every_replace_hide():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from general_plot import Plotter
    fig, ax = plt.subplots()
    ax.set_xlim(0, 0.5)
    t, lab = Plotter.set_ticks(ax, 'x', step=0.1, every=2, replace={0.4: "x"})
    assert np.allclose(t, np.arange(6) * 0.1) and lab == ["0", "", "0.2", "", "x", ""]
    _, lab = Plotter.set_ticks(ax, 'x', [0, .1, .2], hide=[.1], replace={0.2: "z"})
    assert lab == ["0", "", "z"]
    Plotter.set_ax_params(ax, xtick_opts=dict(step=0.25), xlabel="a", xlabel_coords=(0.5, -0.1))
    assert [l.get_text() for l in ax.get_xticklabels()] == ["0", "0.25", "0.5"]
    assert ax.xaxis.get_label().get_position()[1] == -0.1
    plt.close(fig)


def test_colormap_result_feeds_colorbar_and_pcolormesh():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from general_plot import Plotter
    cm          = Plotter.get_colormap(vmin=1e-3, vmax=1, cmap="Blues", scale="log")
    getcolor, colors, norm = cm                                            # unpacking is unchanged
    assert len(cm) == 3 and cm(0.1) == getcolor(0.1)
    fig, ax = plt.subplots()
    mesh    = Plotter.pcolormesh(ax, np.arange(3), np.arange(2), np.full((2, 3), 0.1), cmap=cm)
    assert mesh.norm is norm
    cbar, _ = Plotter.add_colorbar(fig, ax=ax, mappable=cm)
    assert cbar.norm is norm
    plt.close(fig)


def test_letter_density_inset_log_grid():
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from general_plot import Plotter
    fig, axs = Plotter.get_subplots(1, 2, sizex=6, sizey=3, constrained_layout=True)
    Plotter.label_panels(axs, ["A", "B"])
    assert axs[0].texts[0].get_text() == "(a) A" and axs[1].texts[0].get_text() == "(b) B"
    dens, c = Plotter.density_2d(np.random.rand(5, 100), np.linspace(0, 1, 11))
    assert dens.shape == (10, 5) and np.allclose(dens.sum(axis=0), 1) and len(c) == 10
    ins = Plotter.get_inset(axs[0], [0.5, 0.5, 0.4, 0.4])
    assert ins.get_position().width < axs[0].get_position().width
    Plotter.set_ax_params(axs[1], yscale="log", ylim=(1e-8, 1e4), ydecade_step=4, grid=dict(axis="y", style=":"))
    assert len(axs[1].get_yticks()) >= 3
    fig2, ax2 = plt.subplots(layout="tight")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Plotter.add_colorbar(fig2, ax=ax2, norm=matplotlib.colors.Normalize(0, 1))
    assert any("tight_layout" in str(x.message) for x in w)
    plt.close("all")


def test_get_inset_background_style():
    """The inset background accepts an explicit face color and opacity."""
    fig, ax = Plotter.get_subplots(1, 1)
    ins     = Plotter.get_inset(ax, [0.5, 0.5, 0.4, 0.4], facecolor="red", facealpha=0.25)
    assert ins.get_facecolor()[:3] == (1.0, 0.0, 0.0)
    assert ins.patch.get_alpha() == 0.25
    plt.close(fig)


def test_set_ax_params_keeps_unset_properties():
    """A later call changes only what it names: grid, tick size, spines, log formatter and minor locator survive."""
    fig, ax = plt.subplots()
    Plotter.set_ax_params(ax, yscale="log", tick_length_major=7, spine_width=2.0, grid=True)
    fmt, loc = ax.yaxis.get_major_formatter(), ax.yaxis.get_minor_locator()

    Plotter.set_ax_params(ax, xlabel="a")
    assert ax.xaxis._major_tick_kw["grid_alpha"] == 0.3 and ax.xaxis.get_gridlines()[0].get_visible()
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 7
    assert all(sp.get_linewidth() == 2.0 for sp in ax.spines.values())
    assert ax.yaxis.get_major_formatter() is fmt and ax.yaxis.get_minor_locator() is loc

    Plotter.set_ax_params(ax, grid=False, which="y", spine_color="red")
    assert not ax.yaxis.get_gridlines()[0].get_visible()
    assert ax.spines["left"].get_edgecolor()[:3] == (1.0, 0.0, 0.0) and ax.spines["left"].get_linewidth() == 2.0
    plt.close(fig)


def test_helpers_keep_unset_properties():
    """Tick, label, spine and axis helpers change only what they are asked to."""
    fig, ax = plt.subplots()
    ax.set_xlabel("x", fontsize=17)
    Plotter.set_xlabel(ax, "y")
    assert ax.xaxis.get_label().get_fontsize() == 17

    Plotter.set_tickparams(ax, maj_tick_l=9, direction="out")                   # direction passed twice used to raise
    Plotter.set_tickparams(ax, labelsize=7)
    t = ax.xaxis.get_major_ticks()[0]
    assert t.tick1line.get_markersize() == 9 and ax.xaxis._major_tick_kw["tickdir"] == "out"

    Plotter.unset_ticks(ax, xticks=True)
    Plotter.configure_axes(ax, xlabel="a")                                      # tick labels stay hidden
    assert not ax.xaxis.get_major_ticks()[0].label1.get_visible()

    Plotter.unset_spines(ax, left=True)
    Plotter.unset_spines(ax, top=True)
    assert not ax.spines["left"].get_visible() and not ax.spines["top"].get_visible()
    plt.close(fig)


def test_data_limits_and_mask_below():
    """data_limits uses the finite values (above the floor on a log axis); mask_below blanks values under the floor."""
    series = [np.array([1e-3, 1e-1, np.nan]), np.array([1e-15, 1.0])]
    assert np.allclose(Plotter.data_limits(series, log=True, margin=10.0, floor=1e-12), (1e-4, 10.0))
    assert np.allclose(Plotter.data_limits(series, log=True, margin=10.0), (1e-16, 10.0))
    assert np.allclose(Plotter.data_limits([np.array([0.0, 1.0])], log=False), (-0.08, 1.08))
    assert Plotter.data_limits([np.array([np.nan])]) is None
    masked = Plotter.mask_below([1e-15, 1e-3, np.nan], 1e-12)
    assert np.isnan(masked[0]) and masked[1] == 1e-3 and np.isnan(masked[2])


def test_suptitle_leaves_room_and_save_fig_returns_the_path(tmp_path):
    """suptitle sets the title and the top margin; save_fig(adjust=False) keeps that margin and returns the file."""
    fig, axes   = Plotter.get_subplots(1, 2, constrained_layout=False)
    title       = Plotter.suptitle(fig, "system", top=0.8)
    assert title.get_text() == "system" and fig.subplotpars.top == 0.8
    path        = Plotter.save_fig(tmp_path, "figure.pdf", fig=fig, adjust=False)
    assert path == tmp_path / "figure.pdf" and path.exists() and fig.subplotpars.top == 0.8
    plt.close(fig)


def test_get_subplots_width_and_height_ratios():
    """width_ratios and height_ratios reach the grid (they were overwritten by None before)."""
    fig, axes   = Plotter.get_subplots(2, 3, sizex=5.4, sizey=3.5, width_ratios=(1.0, 0.1, 1.0), height_ratios=(2.0, 1.0), constrained_layout=False)
    gridspec    = axes[0, 0].get_subplotspec().get_gridspec()
    assert list(gridspec.get_width_ratios()) == [1.0, 0.1, 1.0] and list(gridspec.get_height_ratios()) == [2.0, 1.0]
    widths      = [axes[0, col].get_position().width for col in range(3)]
    assert np.isclose(widths[1] / widths[0], 0.1)
    plt.close(fig)


def test_place_grid_sets_absolute_gaps_and_save_fig_keeps_the_size(tmp_path):
    """place_grid puts margins, per-column gaps and row gap in inches; save_fig(bbox_inches=None) keeps the figure size."""
    fig, axes   = Plotter.get_subplots(2, 3, sizex=6.0, sizey=4.0, constrained_layout=False)
    Plotter.place_grid(fig, axes, width_ratios=(1.0, 0.5, 1.0), col_gaps=(0.2, 0.6), row_gap=0.3, margins=(0.7, 0.1, 0.4, 0.2))
    inches      = lambda ax: np.array(ax.get_position().bounds) * np.array([6.0, 4.0, 6.0, 4.0])
    first, slim, last, below = (inches(axes[0]), inches(axes[1]), inches(axes[2]), inches(axes[3]))
    assert np.isclose(first[0], 0.7) and np.isclose(slim[0] - (first[0] + first[2]), 0.2) and np.isclose(last[0] - (slim[0] + slim[2]), 0.6)
    assert np.isclose(last[0] + last[2], 6.0 - 0.1) and np.isclose(slim[2] / first[2], 0.5)
    assert np.isclose(4.0 - 0.2, first[1] + first[3]) and np.isclose(first[1] - (below[1] + below[3]), 0.3) and np.isclose(below[1], 0.4)
    path        = Plotter.save_fig(tmp_path, "grid.png", format="png", dpi=100, fig=fig, adjust=False, bbox_inches=None)
    assert plt.imread(path).shape[:2] == (400, 600)
    tight       = Plotter.save_fig(tmp_path, "tight.png", format="png", dpi=100, fig=fig, adjust=False)
    assert plt.imread(tight).shape[:2] != (400, 600)                    # the default crops the page to the content
    with pytest.raises(ValueError):
        Plotter.place_grid(fig, axes, width_ratios=(1.0, 1.0, 1.0), col_gaps=3.0)
    plt.close(fig)


def test_whitespace_reports_the_blank_margins_in_inches():
    """whitespace measures the distance of the content from the figure edges; a placed grid with generous margins leaves some."""
    fig, axes   = Plotter.get_subplots(1, 2, sizex=6.0, sizey=3.0, constrained_layout=False)
    Plotter.place_grid(fig, axes, width_ratios=(1.0, 1.0), col_gaps=0.8, margins=(1.0, 1.0, 0.8, 0.8))
    left, right, bottom, top = Plotter.whitespace(fig)
    assert 0.0 < left < 1.0 and 0.0 < right < 1.0 and 0.0 < bottom < 0.8 and 0.0 < top <= 0.8
    Plotter.place_grid(fig, axes, width_ratios=(1.0, 1.0), col_gaps=0.8, margins=(0.0, 0.0, 0.0, 0.0))
    assert Plotter.whitespace(fig)[0] < 0.0                              # the tick labels reach beyond the page
    plt.close(fig)
