"""Formatters for publication-quality axis tick labels."""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.ticker as mticker


class CustomFormatter(mticker.Formatter):
    """Matplotlib formatter backed by a Python format string.

    Parameters
    ----------
    fmt : str, default="{x:.2f}"
        Format string using `{x}` placeholder.
    """

    def __init__(self, fmt: str = "{x:.2f}"):
        self.fmt = fmt

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
        return self.fmt.format(x=x)


class PercentFormatter(mticker.PercentFormatter):
    """Percent formatter with concise defaults for publication plots."""

    def __init__(self, xmax: float = 100.0, decimals: int = 2, symbol: str = "%"):
        super().__init__(xmax=xmax, decimals=decimals, symbol=symbol)

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
        if self.axis is None:
            fmt = f"{{:.{self.decimals}f}}{{}}" if self.decimals is not None else "{:g}{}"
            val = x / self.xmax * 100.0 if self.xmax != 0 else x
            return fmt.format(val, self.symbol or "")
        return super().__call__(x, pos=pos)


class MathTextSciFormatter(mticker.Formatter):
    """Scientific-notation formatter that renders exponents as math text.

    Examples:
        `1.20e-03` -> `$1.2\\times10^{-3}$`
        `0.00e+00` -> `$0$`
    """

    def __init__(self, fmt: str = "%1.2e"):
        self.fmt = fmt

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
        s               = self.fmt % x
        decimal_point   = "."
        positive_sign   = "+"

        if "e" not in s and "E" not in s:
            return f"${s}$"

        sep         = "e" if "e" in s else "E"
        tup         = s.split(sep)
        significand = tup[0].rstrip(decimal_point)
        sign        = tup[1][0].replace(positive_sign, "")
        exponent    = tup[1][1:].lstrip("0")

        if exponent:
            exponent_str = f"10^{{{sign}{exponent}}}"
        else:
            exponent_str = ""

        if significand and exponent_str:
            res = rf"{significand}{{\times}}{exponent_str}"
        elif exponent_str:
            res = exponent_str
        else:
            res = significand or "0"

        return f"${res}$"


def set_formatter(ax              : Any,
                  formatter_type  : str = "sci",
                  fmt             : str = "%1.2e",
                  axis            : str = "xy") -> None:
    """Set tick label formatter for the specified axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object on which to set the formatter.
    formatter_type : str, default='sci'
        Type of formatter: 'sci', 'custom', or 'percent'.
    fmt : str, default='%1.2e'
        Format string for the axis labels.
    axis : str, default='xy'
        Which axis to apply the formatter to ('x', 'y', or 'xy').
    """
    if formatter_type == "sci":
        formatter = MathTextSciFormatter(fmt)
    elif formatter_type == "custom":
        formatter = CustomFormatter(fmt)
    elif formatter_type == "percent":
        formatter = PercentFormatter()
    else:
        raise ValueError("Unsupported formatter type. Choose from 'sci', 'custom', 'percent'.")

    if "y" in axis:
        ax.yaxis.set_major_formatter(formatter)
    if "x" in axis:
        ax.xaxis.set_major_formatter(formatter)


# ---------------------------------------------
#! EOF
# ---------------------------------------------
