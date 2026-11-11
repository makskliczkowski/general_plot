r"""Figure defaults in one place, read as ``D["width"]`` or ``D.width``.

A script or notebook asks once, ``D = default()``, and uses ``D.width``, ``D.cmap``, ``D.floor`` ... instead of redefining them. ``default(width=5.0)``
returns a copy with some values changed; the shared table ``DEFAULTS`` itself is never modified by that. The values are for quick looks and
printed figures; they are data, not behavior, so they combine with the existing functions:

- ``Plotter.n_colors(n, D.cmap, *D.cmap_range)``: ``n`` sequential colors for a swept number, in sorted order;
- ``Plotter.mask_below(y, D.floor)``: no spikes on a log axis;
- ``ax.margins(y=D.margin)``: headroom for legends and panel letters;
- ``Plotter.save_fig(folder, name + "." + fmt, format=fmt, dpi=D.dpi)`` for ``fmt`` in ``D.formats``.

-----------------------------------------------
Author            : Maks Kliczkowski
File              : defaults.py
-----------------------------------------------
"""

from __future__ import annotations
from typing     import Any
import math

# ---------------------------------------------

DEFAULTS: dict[str, Any] = {
    "column_pt"         : 246.0,            # printed column width in points
    "text_pt"           : 510.0,            # printed text width in points
    "points_per_inch"   : 70.0,             # effective conversion used for printed figures
    "font_size"         : 9.0,              # base font size in points
    "cmap"              : "viridis",        # sequential colormap for swept numbers
    "cmap_range"        : (0.0, 0.9),       # segment of the colormap that is sampled; the pale end is hard to see on white
    "floor"             : 1e-12,            # values below this are not drawn on log axes
    "margin"            : 0.25,             # fractional headroom for legends and letters (axis margins)
    "dpi"               : 150,              # resolution of saved rasters
    "formats"           : ("png", "pdf"),   # file formats of a saved figure
}


class Defaults(dict):
    """A dictionary whose keys are also attributes: ``D["width"]`` and ``D.width`` are the same, and ``D.width = 5`` sets it."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"unknown default '{name}'; available: {sorted(self)}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def copy(self) -> "Defaults":
        return Defaults(self)


def default(**overrides: Any) -> Defaults:
    """Resolve printed figure dimensions and font sizes in a detached copy.

    Parameters
    ----------
    **overrides : keyword arguments
        Any DEFAULTS key. Column/text widths default to 246/510 points, converted
        with 70 points per inch. Font size defaults to 9. Derived sizes are
        recomputed per call unless overridden. Width defaults to one_column,
        height to half the resolved width. Legend, letter and inset fonts are
        font_size minus 2.5, 1.5 and 2 points. DEFAULTS is not changed.

    Returns
    -------
    Defaults
        Settings accessible as attributes or mapping entries. Dimensions are
        inches and font sizes are points. Use configure_style to also apply them.

    Raises
    ------
    KeyError
        If an override is not a known default, so a misspelled name is not silently ignored.
    """
    values = Defaults({**DEFAULTS, **overrides})
    for key in ("column_pt", "text_pt", "points_per_inch", "font_size"):
        if not math.isfinite(values[key]) or values[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    
    # Compute derived values, but do not overwrite any overrides.
    derived = {
        "one_column"    : values.column_pt / values.points_per_inch,
        "two_column"    : values.text_pt / values.points_per_inch,
        "legend_size"   : values.font_size - 2.5,
        "letter_size"   : values.font_size - 1.5,
        "inset_size"    : values.font_size - 2.0,
    }
    derived["width"]    = overrides.get("width", derived["one_column"])
    derived["height"]   = 0.5 * derived["width"]
    unknown             = sorted(set(overrides) - (set(DEFAULTS) | set(derived)))
    
    if unknown:
        raise KeyError(f"unknown defaults {unknown}; available: {sorted(set(DEFAULTS) | set(derived))}")
    
    values.update({key: value for key, value in derived.items() if key not in overrides})
    for key in ("width", "height", "legend_size", "letter_size", "inset_size"):
        if not math.isfinite(values[key]) or values[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    return values


DEFAULTS.update(default())

# ---------------------------------------------
#! EOF
# ---------------------------------------------
