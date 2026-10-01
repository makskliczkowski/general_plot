"""Help reference text for general_plot.Plotter."""

from __future__ import annotations

PLOTTER_HELP: dict[str, str] = {
    "overview": """
Plotter: Scientific Plotting Utilities
======================================
Comprehensive toolkit for creating publication-quality Matplotlib figures.

Available Help Topics:
  - 'style'         : Publication presets (Nature, Science, PRL) and rcParams
  - 'plot'          : 1D/2D plotting primitives (plot, errorbar, scatter, semilogy, etc.)
  - 'axis'          : Setting labels, limits, scales, ticks, and spines (set_ax_params)
  - 'color'         : Palettes, color manipulation, colorbars, colormaps
  - 'layout'        : Multi-panel subplot creation (get_subplots, AxesList)
  - 'grid'          : GridSpec-based complex layouts (make_grid, GridBuilder)
  - 'legend'        : Clean, customizable figure and axis legends
  - 'annotate'      : Panel lettering ((a), (b)), callouts, highlight boxes/circles
  - 'plotters'      : Configuration dataclasses (PlotStyle, FigureConfig, etc.)
  - 'save'          : Exporting publication figures and numeric artifacts (PlotterSave)

Usage:
  >>> from general_plot import Plotter
  >>> Plotter.help('plot')
  >>> Plotter.help('axis')
""",
    "style": """
Style Management (Plotter / configure_style)
--------------------------------------------
- configure_style(style='publication', font_size=10, use_latex=False, dpi=150, **overrides)
  Presets available:
    * 'publication'  : Clean Nature/Science style with compact font hierarchy
    * 'presentation' : High-contrast, larger font size and linewidths for talks
    * 'poster'       : Scaled typography and lines for large conference posters
    * 'minimal'      : Stripped spines and minimal decorations
    * 'default'      : Reset rcParams back to Matplotlib defaults

- reset_color_cycles(palette='nature')
- get_color_cycle()
- reset_linestyles()
""",
    "plot": """
Plotting Primitives
-------------------
- Plotter.plot(ax, x, y, ..., color='C0', label='Data')
- Plotter.errorbar(ax, x, y, yerr=..., xerr=..., capsize=2, ...)
- Plotter.scatter(ax, x, y, s=..., c=..., marker='o', ...)
- Plotter.semilogy(ax, x, y, ...)
- Plotter.semilogx(ax, x, y, ...)
- Plotter.loglog(ax, x, y, ...)
- Plotter.fill_between(ax, x, y1, y2=0, alpha=0.2, ...)
- Plotter.histogram(ax, data, bins=30, density=True, ...)
- Plotter.bar(ax, x, height, width=0.8, orientation='vertical', log=False, ...)
- Plotter.pcolormesh(ax, x, y, z, cmap='viridis', scale='linear', vmin=None, vmax=None, ...)  (x, y: cell centers)
- Plotter.band(ax, x, y, q=(5, 95), axis=0, ...)  (median line + percentile band)
- Plotter.set_ticks(ax, 'x', step=0.1, every=2, replace={0.5: '1/2'}, hide=[...])  (also via set_ax_params(xtick_opts=dict(...)))
- Plotter.get_colormap(...) result can be passed directly to add_colorbar(mappable=...) and pcolormesh(cmap=...)
- set_ax_params(xlabel_coords=(x, y), ylabel_coords=(x, y))  (label position in axes fractions)
- Plotter.get_norm(scale='linear', vmin, vmax, data=None)
- Plotter.tripcolor_field(ax, x, y, z, cmap='viridis', ...)
- Plotter.power_law_guide(ax, x_range, exponent, anchor=..., ...)
- Plotter.hline(ax, val, ...), Plotter.vline(ax, val, ...)
""",
    "axis": """
Axis Formatting & Parameters (set_ax_params)
--------------------------------------------
Fine-grained control over all axis properties in a single call:

>>> Plotter.set_ax_params(
...     ax,
...     xlabel=r'$x$ (m)',
...     ylabel=r'$E$ (eV)',
...     title='Band Structure',
...     xlim=(0, 10),
...     ylim=(-2, 2),
...     xscale='linear',
...     yscale='log',
...     grid=True,
...     show_minor_ticks=True,
...     tick_direction='in',
...     spine_color='black',
...     spine_width=1.0,
... )

Additional axis helpers:
- Plotter.set_xlabel(ax, label, ...)
- Plotter.set_ylabel(ax, label, ...)
- Plotter.set_smart_lim(ax, x_data, y_data, pad=0.05)
- Plotter.unset_spines(ax, ['top', 'right'])
- Plotter.unset_ticks(ax, xticks=True)
- Plotter.set_formatter(ax, formatter='%.1e', axis='xy')
""",
    "color": """
Colors, Palettes & Colorbars
----------------------------
- Plotter.palette('nature') / 'science' / 'prl' / 'vibrant' / 'muted' / 'tableau'
- Plotter.add_colorbar(fig, cax_or_rect, mappable_or_data, cmap='viridis', scale='linear', ...)
- Color adjustments:
  * Plotter.lighten(color, amount=0.2)
  * Plotter.darken(color, amount=0.2)
  * Plotter.desaturate(color, amount=0.2)
  * Plotter.with_alpha(color, alpha=0.5)
  * Plotter.blend(color1, color2, weight=0.5)
  * Plotter.n_colors(n, cmap='viridis')
""",
    "layout": """
Subplot Management & AxesList
-----------------------------
- Plotter.get_subplots(nrows=1, ncols=1, sizex=10, sizey=8, ...)
  Returns (fig, axes) where axes is an `AxesList`.

Features of AxesList:
  * 1D and 2D indexing: `axes[0]` or `axes[row, col]`
  * Named panel indexing: `axes['panel_a']`
  * Grid slicing: `axes.row(0)`, `axes.col(1)`, `axes.span(0, 2, 0, 2)`
  * Batch operations: `axes.apply(lambda ax: ax.grid(True))`
  * Disable unused panels cleanly: `axes.disable('panel_c')` or `axes[1, 2].disable()`
""",
    "grid": """
Complex Grids & GridBuilder
---------------------------
- Plotter.make_grid(nrows, ncols, width_ratios=..., height_ratios=...)
- Plotter.GridBuilder(figsize=(10, 8))
  Fluent API for building heterogeneous multi-row layouts:
  >>> builder = Plotter.GridBuilder(figsize=(12, 6))
  >>> builder.add_row(ncols=1, height_ratio=1.0)
  >>> builder.add_row(ncols=3, height_ratio=2.0)
  >>> fig, axes = builder.build()
""",
    "legend": """
Legends
-------
- Plotter.set_legend(ax, loc='best', frameon=False, style='publication', ...)
- Plotter.set_legend_custom(ax, handles, labels, ...)
""",
    "annotate": """
Annotations & Visual Accents
----------------------------
- Plotter.set_annotate_letter(ax, letter='(a)', loc='top_left')
- Plotter.set_annotate(ax, text, xy=(x, y), ...)
- Plotter.set_arrow(ax, start=(x0, y0), end=(x1, y1), ...)
- Plotter.highlight_box(ax, x_range=(x1, x2), y_range=(y1, y2), alpha=0.2)
- Plotter.highlight_circle(ax, center=(x, y), radius=r, alpha=0.2)
""",
    "plotters": """
Plotters & Configuration Dataclasses
------------------------------------
Access configuration classes for reproducible plots:
- Plotter.plot_style(**kwargs) -> PlotStyle
- Plotter.figure_config(**kwargs) -> FigureConfig
- Plotter.kspace_config(**kwargs) -> KSpaceConfig
- Plotter.kpath_config(**kwargs) -> KPathConfig
- Plotter.spectral_config(**kwargs) -> SpectralConfig
""",
    "save": """
Figure and Data Export (Plotter.save_fig / PlotterSave)
------------------------------------------------------
- Plotter.save_fig(directory, filename, format='pdf', dpi=300, ...)
- PlotterSave:
  * PlotterSave.dict2json(directory, filename, data)
  * PlotterSave.json2dict(directory, filename)
  * PlotterSave.singleColumnData(directory, filename, y, typ='.npy'|'.txt')
  * PlotterSave.twoColumnsData(directory, filename, x, y, typ='.npy'|'.txt'|'.dat')
  * PlotterSave.matrixData(directory, filename, x, y, typ='.npy'|'.txt'|'.dat')
  * PlotterSave.app_df(df, colname, y, fill_value=np.nan)
  * PlotterSave.app_array(arr, y)
""",
}


# ---------------------------------------------
#! EOF
# ---------------------------------------------
