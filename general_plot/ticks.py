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
        limits = ax.get_xlim() if which == 'x' else ax.get_ylim()

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
        TickMixin._restore_limits(ax, which, limits)

    @staticmethod
    def _restore_limits(ax, which: str, limits):
        """Put the view limits back after ticks were set: Matplotlib's ``set_*ticks`` widens the view to include a tick outside it.

        ``auto=None`` leaves the autoscale state as it was, so data plotted later still rescales an axis whose limits were never set.
        """
        (ax.set_xlim if which == 'x' else ax.set_ylim)(*limits, auto=None)

    @staticmethod
    def set_ticks(ax, which: str = 'x', ticks=None, *, step=None, n=None, every=None, offset: int = 0, labels=None, replace=None, replace_rotation=None, hide=None, hide_ticks=None, fmt: str = '{:g}', tol: float = 1e-9):
        """
        Set tick positions and labels in one call, without writing the label list by hand.

        Positions: `ticks`; else multiples of `step` inside the current limits; else `n` equally spaced ticks
        over the limits; else the current ticks inside the limits. `hide_ticks` removes selected major tick
        positions. Labels: `labels` if given, else `fmt.format(tick)`. Then `every`, `replace` and `hide`
        act on the labels; `hide` keeps the tick mark. The axis limits never change: a tick outside them
        (Matplotlib would widen the view to show it) is set but not visible.

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
            {position: label}. Replaces the label of the tick at that position, for example {0.5: "1/2"}. A position
            that is not a tick yet becomes one (unless `labels` are given), so an axis label can stand in for a tick
            label: replace={-1.5: r"$E/N$"}. Put the position inside the limits, or the label is not visible.
        replace_rotation : float, optional
            Rotation in degrees of the replaced labels only (e.g. 90 for an axis label set along a y axis).
        hide : array-like, optional
            Positions whose labels are emptied. The ticks stay.
        hide_ticks : array-like, optional
            Major tick positions to remove, along with their labels. Minor ticks are controlled separately
            by `show_minor_ticks` in `set_ax_params`.
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
        scale       = max(abs(lo), abs(hi), 1e-300)
        if ticks is None:
            if step is not None:
                ticks   = np.arange(np.ceil(lo / step - tol), np.floor(hi / step + tol) + 1) * step
            elif n is not None:
                ticks   = np.linspace(lo, hi, n)
            else:
                cur     = np.asarray(axis.get_majorticklocs())
                ticks   = cur[(cur >= lo - tol * abs(hi - lo)) & (cur <= hi + tol * abs(hi - lo))]
        else:
            ticks = np.asarray(ticks, dtype=float)
            if step is not None:
                # Filter provided ticks to those matching multiples of step (or within tolerance)
                ticks = ticks[np.isclose(np.fmod(np.abs(ticks), step), 0, atol=tol * scale) |
                              np.isclose(np.fmod(np.abs(ticks), step), step, atol=tol * scale)]
            elif n is not None and len(ticks) > n:
                idx = np.round(np.linspace(0, len(ticks) - 1, n)).astype(int)
                ticks = ticks[idx]
        ticks       = np.asarray(ticks, dtype=float) + 0.0      # +0.0 turns -0.0 into 0.0
        if replace and labels is None:
            # a replaced position that is not a tick yet is added, so its label can be placed anywhere on the axis
            missing = [pos for pos in replace if not np.any(np.isclose(ticks, pos, rtol=1e-9, atol=tol * scale))]
            ticks   = np.sort(np.concatenate([ticks, np.asarray(missing, dtype=float)]))
        if labels is not None:
            out     = list(labels)
            if len(out) != len(ticks):
                # If labels matched original ticks before filtering, filter labels identically
                raise ValueError(f"{len(out)} labels for {len(ticks)} ticks.")
        else:
            is_log  = (ax.get_xscale() == 'log') if which == 'x' else (ax.get_yscale() == 'log')
            fmt_obj = axis.get_major_formatter()
            if is_log and fmt == '{:g}' and fmt_obj is not None and not isinstance(fmt_obj, mticker.NullFormatter):
                out = [fmt_obj(t) for t in ticks]
            else:
                out = [fmt.format(t) for t in ticks]
        if len(out) != len(ticks):
            raise ValueError(f"{len(out)} labels for {len(ticks)} ticks.")
        if every is not None and every > 1:
            out     = [l if (i - offset) % every == 0 else '' for i, l in enumerate(out)]
        is_log_scale= (ax.get_xscale() == 'log') if which == 'x' else (ax.get_yscale() == 'log')
        if is_log_scale:
            near    = lambda t, v: abs(t - v) <= 1e-5 * max(abs(t), abs(v), 1e-300)
        else:
            near    = lambda t, v: abs(t - v) <= tol * scale + 1e-12 * abs(v)
        for pos in (hide_ticks if hide_ticks is not None else []):
            keep    = np.asarray([not near(t, pos) for t in ticks], dtype=bool)
            ticks   = ticks[keep]
            out     = [label for label, keep_tick in zip(out, keep) if keep_tick]
        for pos in (hide if hide is not None else []):
            out     = ['' if near(t, pos) else l for t, l in zip(ticks, out)]
        for pos, text in (replace or {}).items():
            out     = [text if near(t, pos) else l for t, l in zip(ticks, out)]
        limits      = ax.get_xlim() if which == 'x' else ax.get_ylim()
        (ax.set_xticks if which == 'x' else ax.set_yticks)(ticks, labels=out)
        TickMixin._restore_limits(ax, which, limits)
        if replace and replace_rotation is not None:
            for tick, text in zip(ticks, (ax.get_xticklabels() if which == 'x' else ax.get_yticklabels())):
                if any(near(tick, pos) for pos in replace):
                    text.set_rotation(replace_rotation)
        return ticks, out

    @staticmethod
    def set_tickparams( ax,
                        labelsize       =   None,
                        left            =   None,
                        right           =   None,
                        top             =   None,
                        bottom          =   None,
                        xticks          =   None,
                        yticks          =   None,
                        xticklabels     =   None,
                        yticklabels     =   None,
                        maj_tick_l      =   None,
                        min_tick_l      =   None,
                        # aliases
                        majl            =   None,
                        minl            =   None,
                        **kwargs
                        ):
        """
        Configure tick parameters and positions.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axis.
        labelsize : float, optional
            Font size for tick labels; None preserves the current size.
        left, right, top, bottom : bool, optional
            Whether to show ticks on the corresponding side; None preserves the current state.
        xticks, yticks : list, optional
            Tick positions.
        xticklabels, yticklabels : list, optional
            Tick labels.
        maj_tick_l, min_tick_l : float, optional
            Major and minor tick lengths. None preserves/adapts to axis type (smaller for insets).
        **kwargs
            Additional keyword arguments forwarded to `ax.tick_params`.
        """
        ax                  = Plotter.ax(ax)
        is_inset            = getattr(ax, '_is_inset', False) or (getattr(ax, '_axes_locator', None) is not None)

        # Library lengths and direction apply on the first tick styling of an axis only; later calls change what they name.
        first               = not getattr(ax, '_gp_ticks_styled', False)
        ax._gp_ticks_styled = True
        major_kw, minor_kw  = {}, {}
        if majl is not None:
            warnings.warn("`majl` overrides `maj_tick_l` (deprecated).", DeprecationWarning, stacklevel=2)
        if minl is not None:
            warnings.warn("`minl` overrides `min_tick_l` (deprecated).", DeprecationWarning, stacklevel=2)
        maj_tick_l          = majl if majl is not None else maj_tick_l
        min_tick_l          = minl if minl is not None else min_tick_l
        
        if maj_tick_l is not None or first:
            major_kw['length'] = maj_tick_l if maj_tick_l is not None else (2.5 if is_inset else Plotter.TICK_LENGTH_MAJOR)
        if min_tick_l is not None or first:
            minor_kw['length'] = min_tick_l if min_tick_l is not None else (1.2 if is_inset else Plotter.TICK_LENGTH_MINOR)
        if first:
            major_kw['direction'] = minor_kw['direction'] = Plotter.TICK_DIRECTION

        side_kw = {k: v for k, v in [('left', left), ('right', right), ('top', top), ('bottom', bottom)] if v is not None}

        # labelsize=None keeps the current tick-label size; passing None to tick_params would reset it to the rcParams default.
        ax.tick_params(axis='both', which='major', **side_kw, **({} if labelsize is None else dict(labelsize=labelsize)))
        ax.tick_params(axis="both", which='major', **{**major_kw, **side_kw, **kwargs})
        ax.tick_params(axis="both", which='minor', **{**minor_kw, **side_kw, **kwargs})

        if xticks is not None or xticklabels is not None:
            Plotter._set_ticks_labelled(ax, 'x', xticks, xticklabels)
        if yticks is not None or yticklabels is not None:
            Plotter._set_ticks_labelled(ax, 'y', yticks, yticklabels)

    @staticmethod
    def _setup_log_axis(axis, limits=(), decade_step=4, length_major=None, length_minor=None, width=None):
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

        l_maj = Plotter.TICK_LENGTH_MAJOR if length_major is None else length_major
        l_min = Plotter.TICK_LENGTH_MINOR if length_minor is None else length_minor
        w = Plotter.TICK_WIDTH if width is None else width

        axis.axes.tick_params(axis=axis.axis_name, which='major', length=l_maj, width=w)
        axis.axes.tick_params(axis=axis.axis_name, which='minor', length=l_min, width=w)

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
