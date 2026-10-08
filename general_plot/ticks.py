"""Tick methods of `Plotter` (mixed into the class in plotter.py)."""

from __future__             import annotations

import warnings
from typing                 import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors      import ListedColormap, LogNorm, Normalize, SymLogNorm
from matplotlib.ticker      import FixedLocator, LogFormatterMathtext, LogLocator, NullFormatter

from .colors                import get_cmap_safe

# Set by plotter.py once `Plotter` exists. The methods below are static and call each other through `Plotter.`.
Plotter = None

class TickMixin:
    """Tick positions, labels, style and log-axis ticks. Mixed into `Plotter`."""

    @staticmethod
    def _set_ticks_labelled(ax, which: str, ticks, labels):
        """Set tick positions and labels, matching Matplotlib's modern idiom.

        Uses the combined ``set_*ticks(positions, labels=...)`` form when
        ``ticks`` and ``labels`` have equal length (or when ``ticks`` is None
        and the current tick count matches). Mismatched counts fall back to
        the historical ``set_*ticks`` + ``set_*ticklabels`` sequence, whose
        behavior (warning or error) is delegated to Matplotlib itself.
        """
        set_ticks, get_ticks = (ax.set_xticks, ax.get_xticks) if which == 'x' else (ax.set_yticks, ax.get_yticks)
        set_labels = ax.xaxis.set_ticklabels if which == 'x' else ax.yaxis.set_ticklabels

        if labels is None:
            set_ticks(ticks)
        elif ticks is not None and len(ticks) == len(labels):
            set_ticks(ticks, labels=labels)
        elif ticks is None:
            # labelled ticks without explicit positions: on current positions when matched
            current = get_ticks()
            if len(current) == len(labels):
                set_ticks(current, labels=labels)
            else:
                set_labels(labels)
        else:
            set_ticks(ticks)
            set_labels(labels)

    @staticmethod
    def set_ticks(ax, which: str = 'x', ticks=None, *, step=None, n=None, every=None, offset: int = 0, labels=None, replace=None, hide=None, fmt: str = '{:g}', tol: float = 1e-9):
        """
        Set tick positions and labels in one call, without writing the label list by hand.

        Positions: `ticks`; else multiples of `step` inside the current limits; else `n` equally spaced ticks
        over the limits; else the current ticks inside the limits.
        Labels: `labels` if given, else `fmt.format(tick)`. Then `every`, `replace` and `hide` act on them.

        Parameters
        ----------
        which : {'x', 'y'}
        ticks : array-like, optional
            Tick positions.
        step : float, optional
            Spacing of ticks, anchored at zero.
        n : int, optional
            Number of equally spaced ticks, ends included.
        every : int, optional
            Label only every `every`-th tick, counted from `offset`. The others keep the tick but get an empty label.
        replace : dict, optional
            {position: label}. Replaces the label of the tick at that position, for example {0.5: "1/2"}.
        hide : array-like, optional
            Positions whose labels are emptied. The ticks stay.
        labels : list of str, optional
            Explicit labels, same length as the ticks.
        fmt : str
            Format of the automatic labels.

        Returns
        -------
        ticks, labels : np.ndarray, list of str

        Examples
        --------
        >>> Plotter.set_ticks(ax, 'x', step=0.1, every=2)                 # 0, 0.2, 0.4, ... labelled
        >>> Plotter.set_ticks(ax, 'x', [0, .1, .2, .3, .4, .5], hide=[.2, .3])
        >>> Plotter.set_ticks(ax, 'y', step=6, replace={0: r"$0$"})
        """
        ax          = Plotter.ax(ax)
        axis        = ax.xaxis if which == 'x' else ax.yaxis
        lo, hi      = sorted(ax.get_xlim() if which == 'x' else ax.get_ylim())
        if ticks is None:
            if step is not None:
                ticks   = np.arange(np.ceil(lo / step - tol), np.floor(hi / step + tol) + 1) * step
            elif n is not None:
                ticks   = np.linspace(lo, hi, n)
            else:
                cur     = np.asarray(axis.get_majorticklocs())
                ticks   = cur[(cur >= lo - tol * abs(hi - lo)) & (cur <= hi + tol * abs(hi - lo))]
        ticks       = np.asarray(ticks, dtype=float) + 0.0      # +0.0 turns -0.0 into 0.0
        out         = list(labels) if labels is not None else [fmt.format(t) for t in ticks]
        if len(out) != len(ticks):
            raise ValueError(f"{len(out)} labels for {len(ticks)} ticks.")
        if every is not None:
            out     = [l if (i - offset) % every == 0 else '' for i, l in enumerate(out)]
        scale       = max(abs(lo), abs(hi), 1e-300)
        near        = lambda t, v: abs(t - v) <= tol * scale + 1e-12 * abs(v)
        for pos in (hide if hide is not None else []):
            out     = ['' if near(t, pos) else l for t, l in zip(ticks, out)]
        for pos, text in (replace or {}).items():
            out     = [text if near(t, pos) else l for t, l in zip(ticks, out)]
        (ax.set_xticks if which == 'x' else ax.set_yticks)(ticks, labels=out)
        return ticks, out

    @staticmethod
    def set_tickparams( ax,
                        labelsize       =   None,
                        left            =   True,
                        right           =   True,
                        top             =   True,
                        bottom          =   True,
                        xticks          =   None,
                        yticks          =   None,
                        xticklabels     =   None,
                        yticklabels     =   None,
                        maj_tick_l      =   4,
                        min_tick_l      =   2,
                        **kwargs
                        ):
        '''
        Sets tickparams to the desired ones.
        - ax        :   axis to use
        - labelsize :   fontsize; None keeps the current size
        - left      :   whether to show the left side
        - right     :   whether to show the right side
        - top       :   whether to show the top side
        - bottom    :   whether to show the bottom side
        - xticks    :   list of xticks
        - yticks    :   list of yticks
        '''
        ax = Plotter.ax(ax)

        # labelsize=None keeps the current tick-label size; passing None to tick_params would reset it to the rcParams default.
        ax.tick_params(axis='both', which='major', left=left, right=right, top=top, bottom=bottom, **({} if labelsize is None else dict(labelsize=labelsize)))
        ax.tick_params(axis="both", which='major', left=left, right=right,
                        top=top, bottom=bottom, direction="in",length=maj_tick_l, **kwargs)
        ax.tick_params(axis="both", which='minor', left=left, right=right,
                        top=top, bottom=bottom, direction="in",length=min_tick_l, **kwargs)

        if xticks is not None or xticklabels is not None:
            Plotter._set_ticks_labelled(ax, 'x', xticks, xticklabels)
        if yticks is not None or yticklabels is not None:
            Plotter._set_ticks_labelled(ax, 'y', yticks, yticklabels)

    @staticmethod
    def _setup_log_axis(axis, limits=(), decade_step=4):
        """Apply decade-aligned major/minor ticks for a log-scaled axis.

        *axis* is a matplotlib Axis (e.g. ``ax.xaxis``). Parameters match the
        public :meth:`setup_log_x` / :meth:`setup_log_y`.
        """
        lo, hi  = np.log10(limits[0]), np.log10(limits[1])
        start   = int(np.ceil(lo / decade_step) * decade_step)
        stop    = int(np.floor(hi / decade_step) * decade_step)
        majors  = 10.0 ** np.arange(start, stop + 1, decade_step, dtype=float)

        axis.set_major_locator(FixedLocator(majors))
        axis.set_major_formatter(LogFormatterMathtext(base=10))  # shows 10^{n}
        # minors at 2..9 within each decade
        axis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
        axis.set_minor_formatter(NullFormatter())

        axis.axes.tick_params(axis=axis.axis_name, which='major', length=Plotter.TICK_LENGTH_MAJOR, width=Plotter.TICK_WIDTH)
        axis.axes.tick_params(axis=axis.axis_name, which='minor', length=Plotter.TICK_LENGTH_MINOR, width=Plotter.TICK_WIDTH)

    @staticmethod
    def setup_log_y(ax: plt.Axes, ylims=(1e-12, 1e6), decade_step=4):
        """Configure clean log-scale y ticks at powers of 10 with LaTeX-like labels."""
        ax.set_yscale('log')
        ax.set_ylim(*ylims)
        Plotter._setup_log_axis(ax.yaxis, ylims, decade_step)

    @staticmethod
    def setup_log_x(ax: plt.Axes, xlims=(1e-12, 1e6), decade_step=4):
        """Configure clean log-scale x ticks at powers of 10 with LaTeX-like labels."""
        ax.set_xscale('log')
        ax.set_xlim(*xlims)
        Plotter._setup_log_axis(ax.xaxis, xlims, decade_step)
