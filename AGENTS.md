# Agent guide

Task-first index of the public `general_plot` API. Read this before writing plotting code, so a helper is used
instead of a copy of it. `Plotter.help(topic)` prints the same topics at runtime. Every `Plotter` method is static.

## Quick start

```python
from general_plot import Plotter, configure_style, cmap, thin

configure_style("publication", font_size=9)
fig, axes = Plotter.get_subplots(1, 2, sizex=7.0, sizey=3.0, share_y=True)
Plotter.plot(axes[0], x, y, label="data")
Plotter.set(axes[0], xlabel=r"$t$", ylabel=r"$E$", xscale="log", yscale="log")
Plotter.set_legend(axes[0])
Plotter.save_fig("figures", "name", format=("png", "pdf"), fig=fig, log_ticks=True)
```

`Plotter.set` is the short name of `Plotter.set_ax_params`. `format` may be one suffix or a sequence. With
`log_ticks=True`, `save_fig` sets readable ticks on every logarithmic axis before writing.

## Building and saving figures

| Function | Use |
| --- | --- |
| `configure_style(style, ...)` | Apply `publication`, `presentation`, `poster`, `minimal` or `default`; return the size table. |
| `default(...)` / `DEFAULTS` | Figure defaults without applying a style. |
| `Plotter.get_subplots(nrows, ncols, sizex, sizey, ...)` | Regular grid as `(fig, AxesList)`. Use `single_if_1=True` for one axis. |
| `Plotter.make_grid`, `Plotter.get_grid`, `Plotter.get_grid_ax` | GridSpec layouts and one panel from a grid. |
| `Plotter.get_inset(ax, ...)` | Inset axis inside an existing panel. |
| `Plotter.twin_axis(ax, ...)` | Second y or x axis with its own label and scale. |
| `Plotter.suptitle(fig, text, ...)` | Figure title with room reserved above the panels. |
| `Plotter.save_fig(directory, filename, format=..., fig=..., log_ticks=...)` | Save one or several formats; returns a `Path` or a list of paths. |
| `PlotterSave` | JSON, arrays and matrix data next to a figure. |

`AxesList` (returned by `get_subplots`) supports `axes[i]`, `axes[row, col]`, `axes["name"]`, `axes.first()`,
`axes.row(i)`, `axes.col(j)`, `axes.span(rows, cols)`, `axes.as_grid()`, `axes.apply(fn)`, and
`axes.disable(target)`. Disabled panels become `IgnoredAxis` no-ops.

## Plotting primitives

| Function | Use |
| --- | --- |
| `Plotter.plot(ax, x, y, ...)` | Line; accepts `maxelems` to thin long curves. |
| `Plotter.scatter(ax, x, y, ...)` | Markers. |
| `Plotter.errorbar(ax, x, y, yerr=..., xerr=...)` | Points with error bars; `maxelems` thins bars with the curve. |
| `Plotter.semilogy`, `semilogx`, `loglog` | Curve on logarithmic axes. |
| `Plotter.fill_between(ax, x, y1, y2=...)` | Shaded region. |
| `Plotter.band(ax, x, y, q=(5, 95))` | Median line and percentile band. |
| `Plotter.histogram`, `Plotter.bar` | Distributions and bars. |
| `Plotter.pcolormesh`, `Plotter.contourf`, `Plotter.tripcolor_field` | 2D fields and unstructured data. |
| `Plotter.density_2d(values, bins)` | Smoothed 2D density. |
| `Plotter.hline(ax, val)`, `Plotter.vline(ax, val)` | Reference lines. |
| `Plotter.power_law_guide(ax, x_range, exponent, anchor=...)` | Guide line for $y=A x^{\gamma}$. |
| `Plotter.phase_spans(ax, phases, ...)` | Shade labelled phase regions. |
| `Plotter.plot_with_uncertainty(ax, x, y, error)` | Line with a shaded uncertainty. |
| `Plotter.plot_fit(ax, funct, x, ...)` | Draw a fitted callable. |

## Axes, ticks and formatters

| Function | Use |
| --- | --- |
| `Plotter.set_ax_params` / `Plotter.set` | Labels, limits, scales, title, grid, spines, ticks, legend in one call. |
| `Plotter.set_ticks(ax, which, ticks, step=, every=, replace=, hide=)` | Tick positions and labels without building label lists. |
| `Plotter.set_log_ticks(ax, which, ...)` | Major and minor ticks on a log axis. |
| `Plotter.log_ticks(axis, low, high)` | Few readable ticks on a narrow log span; `save_fig(log_ticks=True)` applies it. |
| `Plotter.set_xlabel`, `Plotter.set_ylabel`, `Plotter.set_ax_labels` | Label helpers. |
| `Plotter.set_formatter(ax, formatter, axis)` | `CustomFormatter`, `MathTextSciFormatter`, `PercentFormatter`, `set_formatter`. |
| `Plotter.set_symlog(ax, ...)` | Symmetric log scale for values around zero. |
| `Plotter.set_smart_lim`, `Plotter.data_limits`, `Plotter.auto_scale` | Limits and scale from data. |
| `Plotter.unify_limits(axes, which)` | Shared limits across panels. |
| `Plotter.set_margins`, `Plotter.whitespace`, `Plotter.place_grid` | Spacing and absolute placement. |
| `Plotter.unset_spines`, `Plotter.unset_ticks`, `Plotter.unset_all`, `Plotter.disable_axis` | Remove decorations. |

## Legends, letters and annotations

| Function | Use |
| --- | --- |
| `Plotter.set_legend(ax, ...)` | Publication legend. |
| `Plotter.set_legend_custom(ax, conditions, ...)` | Legend filtered by label predicates. |
| `Plotter.figure_legend(fig, axes, ...)` | One figure legend collected from several axes; drops duplicate labels. |
| `Plotter.letter(ax, index, text)`, `Plotter.label_panels(axes)` | Panel letters `(a)`, `(b)`, ... |
| `Plotter.set_annotate`, `Plotter.callout`, `Plotter.set_arrow` | Text, callouts and arrows. |
| `Plotter.highlight_box`, `Plotter.highlight_circle` | Highlight regions. |
| `Plotter.set_transparency(ax, alpha)` | Fade a panel. |

## Colors and colormaps

| Function | Use |
| --- | --- |
| `Plotter.palette(name, n)`, `palette(name)`, `palette_cycle(name)` | Journal palettes: `nature`, `science`, `prl`, `vibrant`, `muted`, `tableau`. |
| `Plotter.colorsN`, `Plotter.colorsC`, `Plotter.set_color_cycle`, `Plotter.apply_palette` | Color cycles. |
| `Plotter.get_norm`, `Plotter.get_colormap`, `Plotter.add_colorbar` | Normalization and colorbars. |
| `cmap(name, vmin, vmax)` | Colormap restricted to a segment. |
| `Plotter.n_colors(n, cmap)`, `Plotter.cmap_colors(...)` | Discrete colors from a colormap. |
| `Plotter.lighten`, `darken`, `desaturate`, `with_alpha`, `blend`, `adjust_color` | Perceptual color changes. |
| `Plotter.to_rgba`, `to_hex` | Color conversion. |

## Data, fitting and array helpers

| Function | Use |
| --- | --- |
| `Fitter(x, y)` / `Fitter.fitLinear`, `fitExp`, `fitPower`, `fitAny`, `fit_histogram` | Curve fits; `FitterParams` holds `popt`, `pcov` and is callable. |
| `fit_power_scaling`, `fit_inverse_power_scaling`, `fit_ipr_scaling`, `aggregate` | Scaling fits and means. |
| `thin`, `thin_log` | Subsample aligned arrays. |
| `Plotter.mask_below(values, floor)` | Hide values below a floor on a log axis. |
| `Plotter.log_bin_median`, `Plotter.data_limits`, `Plotter.auto_scale` | Binning, limits, scale choice. |
| `Plotter.filter_results`, `filter_results`, `Between` | Selection by parameter values. |
| `next_power`, `prev_power`, `mod_euc`, `mod_ceil`, `mod_floor`, `mod_round`, `mod_trunc` | Numeric helpers. |
| `find_nearest_idx`, `find_nearest_val`, `find_maximum_idx` | Index helpers. |

## Run data

`general_plot.data` requires the `data` extra (HDF5 and pandas).

| Function | Use |
| --- | --- |
| `RunCatalog.scan(root, pattern=..., path_for=..., loader=...)` | Catalog of runs from metadata only. |
| `RunCatalog.select(predicate=...)`, `.select(mapping, parameters=...)` | Filter runs. |
| `RunEntry.load()` | Lazy mapping of one `history.h5`. |
| `LazyHDF5(path, pattern=...)` | Browse and read datasets on demand. |
| `write_figure_data`, `read_figure_data`, `read_figure_metadata`, `figure_metadata` | One small file per figure. |
| `source_info`, `source_fields`, `with_provenance` | Provenance for saved data. |
| `HDF5Manager`, `LazyRunMapping` | Direct HDF5 access and lazy run mappings. |
| `discover_files`, `parse_parameters`, `format_parameters`, `format_number` | Run-folder discovery and parameter names. |
| `parameter_key`, `parameter_differences`, `Between` | Compare and filter parameters. |

## Remaining utilities

| Function | Use |
| --- | --- |
| `Plotter.ax`, `Plotter.ax_off`, `Plotter.disable` | Normalize an axis argument, hide it, or disable a panel. |
| `Plotter.markers`, `markersC`, `Plotter.colors`, `Plotter.linestyles`, `linestylesC` | Built-in marker, color and line cycles. |
| `Plotter.grid`, `Plotter.set_margins`, `Plotter.whitespace` | Grid on a panel, margins and blank-margin measurement. |
| `Plotter.get_figsize`, `Plotter.ensure_list`, `Plotter.get_grid_map` | Size from column width and small argument/grid helpers. |
| `Plotter.get_grid_subplot`, `Plotter.app_grid_subplot`, `Plotter.configure_axes` | One panel from a grid, append a panel, configure many axes. |
| `Plotter.set_standard_formatter`, `Plotter.set_formater` | Standard numeric tick format (historical spelling kept). |
| `Plotter.plot_heatmaps` | Row of heatmaps from DataFrames. |
| `get_cmap_safe`, `get_color_cycle`, `get_linestyle_cycle`, `get_rcparams_summary` | Style and colormap access. |
| `Plotter.get_norm`, `Plotter.cmap_colors`, `Plotter.n_colors` | Normalization and discrete colors. |

## Rules for this package

- Use `Plotter` helpers before writing Matplotlib code directly.
- Keep `plot_*` scripts thin: extract data, then draw from the small figure-data file.
- Save both `png` and `pdf` with `Plotter.save_fig(..., format=("png", "pdf"))`.
- Prefer `Plotter.set` over a sequence of `ax.set_*` calls.
- Run `ruff check .` and `pytest` after a change.
