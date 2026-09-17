"""general_plot: Publication-quality plotting tools for scientific computing."""

from __future__     import annotations

from .axes          import AxesList, IgnoredAxis
from .colors        import DEFAULT_PALETTES, adjust_color, blend, cmap, darken, desaturate, get_cmap_safe, lighten, n_colors, palette, palette_cycle, to_hex, to_rgba, with_alpha
from .config        import FigureConfig, KPathConfig, KSpaceConfig, PlotStyle, SpectralConfig
from .data_loader   import filter_results
from .fitting       import Fitter, FitterParams, find_maximum_idx, find_nearest_idx, find_nearest_val, mod_ceil, mod_euc, mod_floor, mod_round, mod_trunc, next_power, prev_power, thin
from .formatters    import CustomFormatter, MathTextSciFormatter, PercentFormatter, set_formatter
from .help          import PLOTTER_HELP
from .io            import MatrixPrinter, PlotterSave
from .plotter       import Plotter
from .style         import configure_style, get_color_cycle, get_linestyle_cycle, get_rcparams_summary, reset_color_cycles, reset_linestyles

# Configure the default style on import
configure_style()

__version__         = "0.1.0"
__all__             = [
                        "AxesList",
                        "CustomFormatter",
                        "DEFAULT_PALETTES",
                        "FigureConfig",
                        "Fitter",
                        "FitterParams",
                        "IgnoredAxis",
                        "KPathConfig",
                        "KSpaceConfig",
                        "MathTextSciFormatter",
                        "MatrixPrinter",
                        "PLOTTER_HELP",
                        "PercentFormatter",
                        "PlotStyle",
                        "Plotter",
                        "PlotterSave",
                        "SpectralConfig",
                        "__version__",
                        "adjust_color",
                        "blend",
                        "cmap",
                        "configure_style",
                        "darken",
                        "desaturate",
                        "find_maximum_idx",
                        "find_nearest_idx",
                        "find_nearest_val",
                        "filter_results",
                        "get_cmap_safe",
                        "get_color_cycle",
                        "get_linestyle_cycle",
                        "get_rcparams_summary",
                        "lighten",
                        "mod_ceil",
                        "mod_euc",
                        "mod_floor",
                        "mod_round",
                        "mod_trunc",
                        "n_colors",
                        "next_power",
                        "palette",
                        "palette_cycle",
                        "prev_power",
                        "reset_color_cycles",
                        "reset_linestyles",
                        "set_formatter",
                        "thin",
                        "to_hex",
                        "to_rgba",
                        "with_alpha",
                    ]

# ---------------------------------------------
#! EOF
# ---------------------------------------------
