"""Tests for configuration dataclasses, help, and data loading."""

from general_plot import FigureConfig, KSpaceConfig, PlotStyle, Plotter, SpectralConfig, filter_results


def test_config_dataclasses():
    """Configuration dataclasses can be built via Plotter factories and serialized."""
    ps  = Plotter.plot_style(style="presentation", font_size=14)
    assert isinstance(ps, PlotStyle)
    assert ps.style == "presentation"
    assert ps.to_dict()["font_size"] == 14

    fc  = Plotter.figure_config(width=7.0, height=4.0)
    assert isinstance(fc, FigureConfig)
    assert fc.width == 7.0

    assert Plotter.kspace_config(num_points=200).num_points == 200
    assert Plotter.spectral_config(broadening=0.01).broadening == 0.01


def test_help_topics(capsys):
    """Plotter.help() renders known topics and errors on unknown ones."""
    Plotter.help()
    assert "Scientific Plotting Utilities" in capsys.readouterr().out

    Plotter.help("plot")
    assert "Plotting Primitives" in capsys.readouterr().out

    Plotter.help("nonexistent_topic")
    assert "Unknown topic" in capsys.readouterr().out


def test_filter_results():
    """filter_results narrows result lists by exact parameter matches."""
    runs = [
        {"system_size": 16, "temperature": 0.5, "energy": -1.2},
        {"system_size": 32, "temperature": 0.5, "energy": -1.1},
        {"system_size": 16, "temperature": 1.0, "energy": -0.8},
    ]

    assert len(filter_results(runs, {"system_size": 16})) == 2
    assert len(filter_results(runs, {"system_size": 16, "temperature": 0.5})) == 1


# ---------------------------------------------
#! EOF
# ---------------------------------------------
