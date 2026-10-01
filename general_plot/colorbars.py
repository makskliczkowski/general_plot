"""Colormap, norm and colorbar methods of `Plotter` (mixed into the class in plotter.py)."""

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

class ColormapResult(tuple):
    """
    Return type of `Plotter.get_colormap`: unpacks as (getcolor, colors, norm) or (getcolor, colors, norm, mappable).

    The same object can be passed directly as `mappable` to `Plotter.add_colorbar` and as `cmap` to `Plotter.pcolormesh`,
    so the colormap and the norm are given only once. Attributes: `getcolor`, `cmap`, `norm`, `mappable`.
    """
    def __new__(cls, getcolor, cmap, norm, mappable, with_mappable: bool = False):
        self            = super().__new__(cls, (getcolor, cmap, norm, mappable) if with_mappable else (getcolor, cmap, norm))
        self.getcolor, self.cmap, self.norm, self.mappable = getcolor, cmap, norm, mappable
        return self

    def __call__(self, x):
        return self.getcolor(x)


class ColorbarMixin:
    """Colormaps, norms and colorbars. Mixed into `Plotter`."""

    @staticmethod
    def add_colorbar(fig                : mpl.figure.Figure,
                    pos                 : Optional[List[float]] = None,
                    mappable            : Union[np.ndarray, list, mpl.cm.ScalarMappable, mpl.colors.Normalize, None] = None,
                    cmap                : Union[str, mpl.colors.Colormap] = 'viridis',
                    norm                : Optional[mpl.colors.Normalize] = None,
                    vmin                : Optional[float] = None,
                    vmax                : Optional[float] = None,
                    scale               : str = 'linear',
                    orientation         : str = 'vertical',
                    label               : str = '',
                    label_kwargs        : dict = None,
                    title               : str = '',
                    title_kwargs        : dict = None,
                    ticks               : Optional[Union[List, np.ndarray]] = None,
                    ticklabels          : Optional[List[str]] = None,
                    tick_location       : str = 'auto',
                    tick_params         : dict = None,
                    extend              : str = None,
                    format              : Optional[Union[str, mpl.ticker.Formatter]] = None,
                    discrete            : Union[bool, int] = False,
                    boundaries          : List[float] = None,
                    invert              : bool = False,
                    remove_pdf_lines    : bool = True,
                    ax                  = None,
                    pad                 : float = 0.02,
                    shrink              : float = 1.0,
                    **kwargs) -> Tuple[mpl.colorbar.Colorbar, mpl.axes.Axes]:
        """
        Add a fully customizable colorbar to the figure, at an explicit position or next to given axes.

        The mappable can be given in any of these forms:
        a `ScalarMappable` (for example the return value of `Plotter.pcolormesh`, `imshow` or `scatter`),
        an array of values (limits and `scale` are taken from it), a `Normalize` instance (with `cmap`),
        or None (then `norm` is required).

        Placement: `pos=[left, bottom, width, height]` in figure coordinates, or `pos=None` with `ax`
        (an axis or a list of axes) to steal space from those axes, with `pad` and `shrink` as in
        `Figure.colorbar`. In the second case call this before `fig.tight_layout`.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Parent figure onto which the colorbar axis is added.
        pos : list[float] | tuple[float, float, float, float]
            [left, bottom, width, height] in figure coordinates (0..1).
        mappable : array-like | matplotlib.cm.ScalarMappable | ColormapResult
            - If array-like: a new ScalarMappable is built from `cmap`/`norm` (and `scale`, `vmin`, `vmax`).
            - If ScalarMappable: it is used directly. `vmin`/`vmax` update its clim; `norm` is taken from it
            when not provided. Note: in this case `discrete`/`boundaries` resampling is not applied.
        cmap : str | Colormap, default='viridis'
            Colormap name or object. If `mappable` is a ScalarMappable, its cmap is used unless `cmap`
            is explicitly different from the default and a new mappable is constructed (array-like path).
        norm : matplotlib.colors.Normalize, optional
            Normalization to map data to 0-1. Ignored if `mappable` is ScalarMappable and `norm` is None
            (then the mappable's norm is used).
        vmin, vmax : float, optional
            Data limits. When `scale='log'`, non-positive `vmin` is clamped internally.
        scale : {'linear', 'log', 'symlog'}, default='linear'
            Creates a suitable Normalize when `mappable` is array-like and `norm` is None.
            - 'linear'  -> Normalize
            - 'log'     -> LogNorm (vmin<=0 clamped to ~1e-10)
            - 'symlog'  -> SymLogNorm with linthresh=0.1
        orientation : {'vertical', 'horizontal'}, default='vertical'
            Colorbar orientation.
        label : str, default=''
            Axis label along the long side of the colorbar.
        label_kwargs : dict, optional
            Passed to ColorbarBase.set_label (e.g., dict(fontsize=..., labelpad=...)).
        title : str, default=''
            Title text set at the end/top of the colorbar. For horizontal bars, the title is placed to the side.
        title_kwargs : dict, optional
            Text properties for the title (e.g., dict(fontsize=..., pad=...)).
        ticks : list[float] | np.ndarray, optional
            Explicit major tick locations.
        ticklabels : list[str], optional
            Custom labels for the ticks (same length as `ticks`).
        tick_location : {'auto','left','right','top','bottom'}, default='auto'
            Side on which to draw ticks/labels (respects `orientation`).
        tick_params : dict, optional
            Passed to cbar.ax.tick_params (e.g., dict(length=4, width=1, direction='in')).
        extend : {'neither','both','min','max','neutral'}, default='neutral'
            Colorbar extension behavior. Standard Matplotlib values are 'neither', 'both', 'min', 'max'.
            'neutral' is treated as a pass-through here and may behave like 'neither' depending on Matplotlib.
        format : str | matplotlib.ticker.Formatter, optional
            Tick formatting. If str (e.g., '%.2e'), uses FormatStrFormatter.
        discrete : bool | int, default=False
            Discretize colormap when building from array-like:
            - True  -> 10 bins
            - int N -> N bins
            Ignored when `mappable` is a ScalarMappable.
        boundaries : list[float], optional
            Discrete bin edges. Enables BoundaryNorm and passes `boundaries` to fig.colorbar
            (default spacing='proportional', overridable via kwargs['spacing']).
        invert : bool, default=False
            If True, invert the colorbar axis direction.
        remove_pdf_lines : bool, default=True
            Set solids edgecolor to 'face' to avoid white hairlines in vector exports (PDF/SVG).
        **kwargs :
            Additional arguments forwarded to `fig.colorbar`, e.g.:
            - alpha, spacing ('uniform'|'proportional'), fraction, pad, shrink, aspect, drawedges, etc.

        Returns
        -------
        (cbar, cax) : tuple[matplotlib.colorbar.Colorbar, matplotlib.axes.Axes]
            The created colorbar and its axes.

        Notes
        -----
        - When `mappable` is a ScalarMappable, this helper does not modify its colormap discretization.
            To use `discrete`/`boundaries`, pass raw data (array-like) instead.
        - For 'log' scale, ensure your data are strictly positive (this function clamps vmin if needed).

        Examples
        --------
        # Vertical, linear scale from raw data
        cbar, cax = Plotter.add_colorbar(fig, [0.92, 0.15, 0.02, 0.7], data, label='Mz')

        # Horizontal, log scale with sci formatting and extensions
        cbar, cax = Plotter.add_colorbar(
            fig, [0.2, 0.9, 0.6, 0.03], data,
            scale='log', orientation='horizontal',
            format='%.0e', extend='both',
            tick_location='top', label='Conductance'
        )

        # Discrete categorical-like bar with custom tick labels
        cbar, cax = Plotter.add_colorbar(
            fig, [0.85, 0.1, 0.03, 0.8], [0, 1, 2],
            cmap='Set1', discrete=3,
            ticklabels=['Insulator', 'Metal', 'SC']
        )

        # Non-uniform boundaries
        cbar, cax = Plotter.add_colorbar(
            fig, [0.86, 0.15, 0.02, 0.7], data,
            boundaries=[0, 0.5, 2.0, 10.0], spacing='proportional'
        )
        """
        # 1. Handle Normalization and Mappable
        if isinstance(mappable, ColormapResult):
            mappable = mappable.mappable
            
        if isinstance(mappable, mcolors.Normalize):
            mappable, norm = None, mappable
            
        # Get the mappable...
        if mappable is None:
            if norm is None:
                raise ValueError("add_colorbar needs a mappable or a norm.")
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
        elif isinstance(mappable, mpl.cm.ScalarMappable):
            sm = mappable
            if vmin is not None or vmax is not None:
                sm.set_clim(vmin, vmax)
            # If a mappable is passed, we might need to extract the cmap/norm for modifications
            if norm is None:
                norm = sm.norm
            if cmap == 'viridis':
                cmap = sm.cmap  # Only override default if specific one not passed
        else:
            data    = np.asarray(mappable)
            _vmin   = vmin if vmin is not None else np.nanmin(data)
            _vmax   = vmax if vmax is not None else np.nanmax(data)

            # Discrete Boundaries (e.g., for Phase Diagrams)
            if boundaries is not None:
                cmap_obj    = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
                norm        = mcolors.BoundaryNorm(boundaries, cmap_obj.N, clip=True)

            # Standard Scales
            elif norm is None:
                if scale == 'log':
                    if _vmin <= 0:
                        _vmin = 1e-10
                    norm = mcolors.LogNorm(vmin=_vmin, vmax=_vmax)
                elif scale == 'symlog':
                    norm = mcolors.SymLogNorm(linthresh=0.1, vmin=_vmin, vmax=_vmax)
                else:
                    norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

            # Discretize Colormap (e.g. 10 distinct colors)
            if discrete:
                cmap_obj    = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
                n_bins      = discrete if isinstance(discrete, int) and discrete > 1 else 10
                cmap        = cmap_obj.resampled(n_bins)

            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

        # Formatting
        if isinstance(format, str):
            format = mticker.FormatStrFormatter(format)

        # Create Axes and Colorbar
        # Pass boundaries to colorbar if they exist (ensures spacing is correct)
        if pos is not None:
            cax     = fig.add_axes(pos)
        elif ax is not None:
            if isinstance(fig.get_layout_engine(), mpl.layout_engine.TightLayoutEngine):
                warnings.warn("add_colorbar(ax=...) with a tight_layout figure: the bar may overlap the axes. "
                              "Use constrained_layout=True, or pass `pos`.", stacklevel=2)
            cax     = None
            kwargs  = dict(kwargs, ax=ax if isinstance(ax, mpl.axes.Axes) else list(ax), pad=pad, shrink=shrink)
        else:
            raise ValueError("add_colorbar needs `pos` or `ax`.")
        cbar_kwargs = kwargs.copy()
        if boundaries is not None:
            cbar_kwargs['boundaries']   = boundaries
            cbar_kwargs['spacing']      = kwargs.get('spacing', 'proportional')

        cbar        = fig.colorbar(sm, cax=cax, orientation=orientation, extend=extend, format=format, **cbar_kwargs)
        cax         = cbar.ax

        # Labels and Titles
        if label:
            l_kwargs = label_kwargs or {}
            cbar.set_label(label, **l_kwargs)

        if title:
            # Smart default padding for title
            t_kwargs    = title_kwargs or {}
            pad         = t_kwargs.pop('pad', 10)
            if orientation == 'vertical':
                cbar.ax.set_title(title, pad=pad, **t_kwargs)
            else:
                # For horizontal, title usually makes sense as a side label or top label
                cbar.ax.text(1.05, 0.5, title, transform=cbar.ax.transAxes, va='center', ha='left', **t_kwargs)

        # Tick Customization
        if ticks is not None:
            cbar.set_ticks(ticks)
        if ticklabels is not None:
            cbar.set_ticklabels(ticklabels)

        # Tick Location (Top/Bottom/Left/Right)
        if tick_location != 'auto':
            if orientation == 'horizontal':
                cax.xaxis.set_ticks_position(tick_location)
                cax.xaxis.set_label_position(tick_location)
            else:
                cax.yaxis.set_ticks_position(tick_location)
                cax.yaxis.set_label_position(tick_location)

        # Same tick style as `set_ax_params`; `tick_params` overrides it.
        cbar.ax.tick_params(which='major', length=Plotter.TICK_LENGTH_MAJOR, width=Plotter.TICK_WIDTH,
                            direction=Plotter.TICK_DIRECTION, labelsize=Plotter.default_labelsize())
        cbar.ax.tick_params(which='minor', length=Plotter.TICK_LENGTH_MINOR, width=Plotter.TICK_WIDTH, direction=Plotter.TICK_DIRECTION)
        if tick_params:
            cbar.ax.tick_params(**tick_params)

        # Utilities
        if invert:
            cbar.ax.invert_axis()

        if scale == 'log' and ticks is None and boundaries is None:
            cbar.ax.minorticks_on()

        # PDF Export Fix (Removes white lines between colors)
        if remove_pdf_lines:
            cbar.solids.set_edgecolor("face")

        return cbar, cax

    @staticmethod
    def get_colormap(values: Optional[np.ndarray] = None, vmin=None, vmax=None, *,
            cmap='PuBu', elsecolor='blue', get_mappable: bool = False, return_mappable: Optional[bool] = None,
            norm=None, scale='linear', **kwargs):
        """
        Get a colormap for the given values.
        
        Parameters:
        - values (array-like): The values to map to colors.
        - cmap (str, optional): The colormap to use. Defaults to 'PuBu'.
        - elsecolor (str, optional): The color to use if there is only one value. Defaults to 'blue'.
        - get_mappable (bool, optional): If True, also return a ScalarMappable as
          the 4th item, ready to pass into `Plotter.add_colorbar(..., mappable=...)`.
        - return_mappable (bool, optional): Alias for `get_mappable`.
        
        Returns:
        - `ColormapResult`, a tuple that unpacks as (getcolor, colors, norm[, mappable]) and is itself callable
          (`result(x)` is `getcolor(x)`). Pass it as `mappable` to `Plotter.add_colorbar`, or as `cmap` to
          `Plotter.pcolormesh`, so that cmap and norm are given once.
          - getcolor (function): A function that maps a value to a color.
          - colors (Colormap): The colormap object.
          - norm (Normalize): The normalization object.
          - mappable (ScalarMappable): Unpacked as 4th item only when `get_mappable=True`; always an attribute.
        
        Example:
        >>> getcolor, colors, norm = Plotter.get_colormap([1, 2, 3], cmap='viridis')
        >>> color = getcolor(2.5)
        >>> getcolor, colors, norm, mappable = Plotter.get_colormap(
        ...     [1, 2, 3], cmap='viridis', return_mappable=True
        ... )
        """
        if return_mappable is not None:
            get_mappable = bool(return_mappable)

        # Resolve vmin/vmax
        if values is None and (vmin is None or vmax is None):
            raise ValueError("Either 'values' or both 'vmin' and 'vmax' must be provided.")

        if vmin is None:
            vmin = np.nanmin(values)
        if vmax is None:
            vmax = np.nanmax(values)

        # Create Norm (if not provided externally)
        if norm is None:
            if scale == 'log':
                if vmin <= 0:
                    vmin = 1e-10
                norm = LogNorm(vmin=vmin, vmax=vmax)
            elif scale == 'symlog':
                norm = SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)

        colors = get_cmap_safe(cmap)

        # Create getcolor function
        if vmin == vmax:
            def getcolor(x):
                return elsecolor
        else:
            def getcolor(x):
                return colors(norm(x))

        # Create Mappable for downstream colorbar reuse.
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colors)
        if values is not None:
            mappable.set_array(np.asarray(values))
        else:
            mappable.set_array(np.asarray([vmin, vmax], dtype=float))

        return ColormapResult(getcolor, colors, norm, mappable, with_mappable=get_mappable)

    @staticmethod
    def get_norm(scale: str = 'linear', vmin=None, vmax=None, *, data=None, linthresh: float = 0.1):
        """
        Normalization for colormaps.

        Parameters
        ----------
        scale : {'linear', 'log', 'symlog'}
            Mapping from value to colour.
        vmin, vmax : float, optional
            Limits. None takes the limits of `data`. For 'log', non-positive `vmin` is replaced by 1e-10.
        data : array-like, optional
            Used only to fill missing limits.
        linthresh : float
            Linear range of 'symlog'.

        Returns
        -------
        matplotlib.colors.Normalize
        """
        if data is not None:
            vmin = np.nanmin(data) if vmin is None else vmin
            vmax = np.nanmax(data) if vmax is None else vmax
        if scale == 'log':
            return LogNorm(vmin=vmin if vmin is not None and vmin > 0 else 1e-10, vmax=vmax)
        if scale == 'symlog':
            return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
        if scale == 'linear':
            return Normalize(vmin=vmin, vmax=vmax)
        raise ValueError(f"Unknown scale {scale!r}; use 'linear', 'log' or 'symlog'.")

    @staticmethod
    def apply_colormap(ax, data, cmap='PuBu', colorbar=True, **kwargs):
        """
        Apply a colormap to the given data and plot it on the provided axis.
        
        Parameters:
        - ax (object): The axis object to plot on.
        - data (array-like): The data to plot.
        - cmap (str, optional): The colormap to use. Defaults to 'PuBu'.
        - colorbar (bool, optional): Whether to add a colorbar. Defaults to True.
        
        Returns:
        - img (AxesImage): The image object.
        """
        norm    = Normalize(np.min(data), np.max(data))
        img     = ax.imshow(data, cmap=cmap, norm=norm, **kwargs)
        if colorbar:
            plt.colorbar(img, ax=ax)
        return img

    @staticmethod
    def discrete_colormap(N, base_cmap=None):
        """
        Create an N-bin discrete colormap from the specified input map.
        
        Parameters:
        - N (int): Number of discrete colors.
        - base_cmap (str or Colormap, optional): The base colormap to use. Defaults to None.
        
        Returns:
        - cmap (Colormap): The discrete colormap.
        """
        base        = get_cmap_safe(base_cmap)
        color_list  = base(np.linspace(0, 1, N))
        cmap_name   = base.name + str(N)
        return ListedColormap(color_list, name=cmap_name)
