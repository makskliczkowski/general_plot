"""general_plot: Publication-quality plotting tools for scientific computing."""

from __future__     import annotations
from importlib      import import_module

# Export all public symbols from submodules on demand. Style modules are configured on first access.
_EXPORTS = {
    **dict.fromkeys(('AxesList', 'IgnoredAxis'), 'axes'),
    **dict.fromkeys(('DEFAULT_PALETTES', 'adjust_color', 'blend', 'cmap', 'darken', 'desaturate', 'get_cmap_safe', 'lighten', 'n_colors', 'palette', 'palette_cycle', 'to_hex', 'to_rgba', 'with_alpha'), 'colors'),
    **dict.fromkeys(('FigureConfig', 'KPathConfig', 'KSpaceConfig', 'PlotStyle', 'SpectralConfig'), 'config'),
    **dict.fromkeys(('Between', 'filter_results'), 'data_loader'),
    **dict.fromkeys(('DEFAULTS', 'Defaults', 'default'), 'defaults'),
    **dict.fromkeys(('Fitter', 'FitterParams', 'find_maximum_idx', 'find_nearest_idx', 'find_nearest_val', 'mod_ceil', 'mod_euc', 'mod_floor', 'mod_round', 'mod_trunc', 'next_power', 'prev_power', 'thin', 'thin_log'), 'fitting'),
    **dict.fromkeys(('CustomFormatter', 'MathTextSciFormatter', 'PercentFormatter', 'set_formatter'), 'formatters'),
    **dict.fromkeys(('PLOTTER_HELP',), 'help'),
    **dict.fromkeys(('MatrixPrinter', 'PlotterSave'), 'io'),
    **dict.fromkeys(('ColormapResult',), 'colorbars'),
    **dict.fromkeys(('Plotter',), 'plotter'),
    **dict.fromkeys(('configure_style', 'get_color_cycle', 'get_linestyle_cycle', 'get_rcparams_summary', 'reset_color_cycles', 'reset_linestyles'), 'style'),
}
_STYLE_MODULES  = {"axes", "colors", "config", "defaults", "formatters", "help", "io", "colorbars", "plotter", "style"}
_style_ready    = False


def __getattr__(name):
    """Load a public export on access and configure style for plotting exports only."""
    global _style_ready
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _EXPORTS[name]
    if module in _STYLE_MODULES and not _style_ready:
        import_module(".style", __name__).configure_style()
        _style_ready = True
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__version__         = "0.5.0"
__all__             = [*_EXPORTS, "__version__"]

# ---------------------------------------------
#! EOF
# ---------------------------------------------
