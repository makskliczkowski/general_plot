"""Tests for tick label formatters."""

import matplotlib.pyplot as plt

from general_plot.formatters import CustomFormatter, MathTextSciFormatter, PercentFormatter, set_formatter


def test_formatters():
    """Each formatter renders numbers with the requested representation."""
    assert CustomFormatter("{x:.1f} sec")(3.1415) == "3.1 sec"
    assert PercentFormatter(decimals=1)(75.5) == "75.5%"

    sci     = MathTextSciFormatter("%.1e")(1.23e-4)
    assert r"10^{-4}" in sci or r"10^{-04}" in sci
    assert r"\times" in sci


def test_set_formatter_on_axis():
    """set_formatter attaches a major formatter to the requested axis."""
    fig, ax     = plt.subplots()
    set_formatter(ax, formatter_type="sci", fmt="%1.1e", axis="xy")
    assert ax.xaxis.get_major_formatter() is not None
    assert ax.yaxis.get_major_formatter() is not None
    plt.close(fig)


# ---------------------------------------------
#! EOF
# ---------------------------------------------
