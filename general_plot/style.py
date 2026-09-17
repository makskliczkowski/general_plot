"""Style configuration, rcParams presets, and cycle management for publication plots."""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Dict, List, Optional

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Check optional scienceplots availability cleanly
try:
    import scienceplots  # noqa: F401
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False

# Check optional labellines availability cleanly
try:
    from labellines import labelLines
    HAS_LABELLINES  = True
except ImportError:
    labelLines      = None
    HAS_LABELLINES  = False

# Typography scales
SMALL_SIZE      = 12
MEDIUM_SIZE     = 14
BIGGER_SIZE     = 16

ADDITIONAL_LINESTYLES: Dict[str, tuple[int, tuple[int, ...]]] = {
    "loosely dotted"            : (0, (1, 5)),
    "dotted"                    : (0, (1, 1)),
    "densely dotted"            : (0, (1, 1)),
    "loosely dashed"            : (0, (2, 5)),
    "dashed"                    : (0, (5, 5)),
    "densely dashed"            : (0, (5, 1)),
    "loosely dashdotted"        : (0, (3, 10, 1, 10)),
    "dashdotted"                : (0, (3, 5, 1, 5)),
    "densely dashdotted"        : (0, (3, 1, 1, 1)),
    "dashdotdotted"             : (0, (3, 5, 1, 5, 1, 5)),
    "loosely dashdotdotted"     : (0, (3, 10, 1, 10, 1, 10)),
    "densely dashdotdotted"     : (0, (3, 1, 1, 1, 1, 1)),
    "solid"                     : (0, ()),
    "loosely long dashed"       : (0, (5, 10)),
    "long dashed"               : (0, (5, 15)),
    "densely long dashed"       : (0, (5, 1)),
    "loosely spaced dots"       : (0, (1, 10)),
    "spaced dots"               : (0, (1, 15)),
    "densely spaced dots"       : (0, (1, 1)),
    "loosely spaced dashes"     : (0, (5, 10)),
    "spaced dashes"             : (0, (5, 15)),
    "densely spaced dashes"     : (0, (5, 1)),
}

colorsList              = list(mcolors.TABLEAU_COLORS)
colorsCycle             = itertools.cycle(colorsList)
colorsCyclePlastic      = itertools.cycle(["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"])
colorsCycleBright       = itertools.cycle(list(mcolors.CSS4_COLORS))
colorsCycleDark         = itertools.cycle(list(mcolors.BASE_COLORS))
colorsCyclePastel       = itertools.cycle(list(mcolors.XKCD_COLORS))

markersList             = ["o", "s", "v", "+", "o", "*", "D", "h", "H", "p", "P", "X", "d", "|", "_"]
markersCycle            = itertools.cycle(["4", "2", "3", "1", "+", "x", "."] + markersList)

linestylesList          = ["-", "--", "-.", ":"]
linestylesCycle         = itertools.cycle(["-", "--", "-.", ":"])
linestylesCycleExtended = itertools.cycle(["-", "--", "-.", ":"] + list(ADDITIONAL_LINESTYLES.keys()))


def markerNorm(x: Any) -> Any:
    """Normalize marker string representation (e.g., 'M0' -> 'o')."""
    if x is not None and isinstance(x, str) and x.startswith("M") and x[1:].isdigit():
        idx = int(x[1:])
        if 0 <= idx < len(markersList):
            return markersList[idx]
    return x


def linestyleNorm(x: Any) -> Any:
    """Normalize linestyle string representation (e.g., 'L0' -> '-')."""
    if x is not None and isinstance(x, str):
        if x.startswith("L") and x[1:].isdigit():
            idx = int(x[1:])
            if 0 <= idx < len(linestylesList):
                return linestylesList[idx]
        if x in ADDITIONAL_LINESTYLES:
            return ADDITIONAL_LINESTYLES[x]
    return x


def _reset_cycle(which: Optional[str], cycle_name: str, colors: list, cycle2take: Any) -> Any:
    if which is None or which == cycle_name:
        new_cycle = itertools.cycle(colors)
        return new_cycle if cycle2take is None else cycle2take
    return cycle2take


def reset_color_cycles(which: Optional[str] = None):
    """Reset the color cycles to their initial states.

    Parameters
    ----------
    which : str, optional
        Cycle to reset: 'TABLEAU', 'Plastic', 'Bright', 'Dark', or 'Pastel'.
        If None, all cycles are reset.
    """
    global colorsCycle, colorsCyclePlastic, colorsCycleBright, colorsCycleDark, colorsCyclePastel
    cycle2take          = None
    colorsCycle         = itertools.cycle(list(mcolors.TABLEAU_COLORS)) if (which is None or which == "TABLEAU") else colorsCycle
    colorsCyclePlastic  = itertools.cycle(["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) if (which is None or which == "Plastic") else colorsCyclePlastic
    colorsCycleBright   = itertools.cycle(list(mcolors.CSS4_COLORS)) if (which is None or which == "Bright") else colorsCycleBright
    colorsCycleDark     = itertools.cycle(list(mcolors.BASE_COLORS)) if (which is None or which == "Dark") else colorsCycleDark
    colorsCyclePastel   = itertools.cycle(list(mcolors.XKCD_COLORS)) if (which is None or which == "Pastel") else colorsCyclePastel

    if which == "TABLEAU":
        return colorsCycle
    if which == "Plastic":
        return colorsCyclePlastic
    if which == "Bright":
        return colorsCycleBright
    if which == "Dark":
        return colorsCycleDark
    if which == "Pastel":
        return colorsCyclePastel
    return colorsCycle


def get_color_cycle(which: Optional[str] = None):
    """Get the active color cycle iterator."""
    global colorsCycle, colorsCyclePlastic, colorsCycleBright, colorsCycleDark, colorsCyclePastel
    if which is None or which == "TABLEAU":
        return colorsCycle
    if which == "Plastic":
        return colorsCyclePlastic
    if which == "Bright":
        return colorsCycleBright
    if which == "Dark":
        return colorsCycleDark
    if which == "Pastel":
        return colorsCyclePastel
    return colorsCycle


def reset_linestyles(which: Optional[str] = None):
    """Reset linestyle cycles to initial states ('Normal' or 'Extended')."""
    global linestylesCycle, linestylesCycleExtended
    cycle2take = None
    if which is None or which == "Normal":
        linestylesCycle = itertools.cycle(["-", "--", "-.", ":"])
        cycle2take = linestylesCycle
    if which is None or which == "Extended":
        linestylesCycleExtended = itertools.cycle(["-", "--", "-.", ":"] + list(ADDITIONAL_LINESTYLES.keys()))
        cycle2take = linestylesCycleExtended if cycle2take is None else cycle2take
    return cycle2take


def get_linestyle_cycle(which: Optional[str] = None):
    """Get the active linestyle cycle iterator."""
    global linestylesCycle, linestylesCycleExtended
    if which is None or which == "Normal":
        return linestylesCycle
    if which == "Extended":
        return linestylesCycleExtended
    return linestylesCycle


def configure_style(
    style: str = "publication",
    font_size: int = 10,
    use_latex: bool = False,
    dpi: int = 150,
    **overrides: Any,
) -> None:
    """Configure matplotlib rcParams for publication-quality figures.

    Parameters
    ----------
    style : str, default='publication'
        Style preset to apply:
        - 'publication'  : compact Nature/Science-like defaults
        - 'presentation' : larger text and strokes for talks
        - 'poster'       : scaled fonts and strokes for posters
        - 'minimal'      : stripped-down axes visuals
        - 'default'      : reset to Matplotlib defaults
    font_size : int, default=10
        Base typographic scale.
    use_latex : bool, default=False
        If True, tries LaTeX-backed rendering via scienceplots.
    dpi : int, default=150
        Screen/display DPI.
    **overrides : dict
        Additional rcParams overrides. Underscores are converted to dots.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)

    if style != "default":
        try:
            if use_latex:
                plt.style.use(["science", "nature"])
            else:
                plt.style.use(["science", "no-latex"])
        except Exception:
            pass  # Fall back gracefully to manual configuration

    base_config = {
        # Figure
        "figure.dpi": dpi,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.autolayout": False,
        "figure.constrained_layout.use": True,
        # Saving
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.unicode_minus": True,
        "axes.axisbelow": True,
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.titlelocation": "left",
        "axes.titleweight": "regular",
        "axes.formatter.limits": (-3, 4),
        "axes.formatter.use_mathtext": True,
        "axes.formatter.useoffset": False,
        "axes.prop_cycle": mpl.cycler(
            color=["#1F77B4", "#D62728", "#2CA02C", "#FF7F0E", "#9467BD", "#8C564B", "#17BECF"]
        ),
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # Grid
        "grid.color": "#D9D9D9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        # Lines
        "lines.linewidth": 1.25,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.75,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        # Scatter / Error bars / Hist
        "scatter.marker": "o",
        "errorbar.capsize": 2.5,
        "hist.bins": 40,
        # Images
        "image.cmap": "viridis",
        "image.interpolation": "nearest",
        "image.origin": "lower",
        # Patches
        "patch.linewidth": 0.8,
        # Rendering behavior
        "path.simplify": True,
        "path.simplify_threshold": 0.0,
        "agg.path.chunksize": 20000,
        # Fonts
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times", "Computer Modern Roman"],
        "mathtext.fontset": "stix",
        # Legend
        "legend.frameon": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "none",
        "legend.fancybox": False,
        "legend.borderaxespad": 0.4,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
    }

    style_configs = {
        "publication": {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "axes.linewidth": 0.9,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "lines.linewidth": 1.3,
            "lines.markersize": 4.2,
        },
        "presentation": {
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2,
            "legend.fontsize": font_size - 2,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
        },
        "poster": {
            "font.size": font_size,
            "axes.titlesize": font_size + 4,
            "axes.labelsize": font_size + 2,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "lines.linewidth": 2.5,
            "lines.markersize": 10,
        },
        "minimal": {
            "font.size": font_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.top": False,
            "ytick.right": False,
        },
        "default": {},
    }

    if style != "default":
        for key, value in base_config.items():
            try:
                mpl.rcParams[key] = value
            except (KeyError, ValueError):
                pass

        if style in style_configs:
            for key, value in style_configs[style].items():
                try:
                    mpl.rcParams[key] = value
                except (KeyError, ValueError):
                    pass

    for key, value in overrides.items():
        key = key.replace("_", ".")
        try:
            mpl.rcParams[key] = value
        except (KeyError, ValueError):
            warnings.warn(f"Invalid rcParam '{key}'", UserWarning, stacklevel=2)


def get_rcparams_summary() -> dict:
    """Get a summary of current rcParams relevant to plotting."""
    keys = [
        "font.size", "axes.labelsize", "axes.titlesize",
        "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
        "lines.linewidth", "lines.markersize",
        "scatter.marker", "errorbar.capsize",
        "image.cmap", "image.interpolation",
        "axes.linewidth", "xtick.major.size", "ytick.major.size",
        "axes.formatter.limits", "axes.formatter.use_mathtext", "axes.formatter.useoffset",
        "figure.dpi", "savefig.dpi",
        "axes.spines.top", "axes.spines.right",
        "savefig.transparent", "pdf.fonttype", "svg.fonttype",
    ]
    return {k: mpl.rcParams.get(k, "N/A") for k in keys}


# ---------------------------------------------
#! EOF
# ---------------------------------------------
