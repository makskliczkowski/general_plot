"""Tests for Plotter core plotting methods and layouts."""

import matplotlib.pyplot as plt
import numpy as np

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
