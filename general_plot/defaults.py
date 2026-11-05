r"""Figure defaults in one place, read as ``D["width"]`` or ``D.width``.

A script or notebook asks once, ``D = default()``, and uses ``D.width``, ``D.cmap``, ``D.floor`` ... instead of redefining them. ``default(width=5.0)``
returns a copy with some values changed; the shared table ``DEFAULTS`` itself is never modified by that. The values are for quick looks and
two-column figures; they are data, not behavior, so they combine with the existing functions:

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

# ---------------------------------------------

DEFAULTS: dict[str, Any] = {
    "width"      : 7.0,             # figure width in inches (two columns)
    "height"     : 2.8,             # height of one row of panels in inches
    "cmap"       : "viridis",       # sequential colormap for swept numbers
    "cmap_range" : (0.0, 0.9),      # segment of the colormap that is sampled; the pale end is hard to see on white
    "floor"      : 1e-12,           # values below this are not drawn on log axes
    "margin"     : 0.25,            # fractional headroom for legends and letters (axis margins)
    "dpi"        : 150,             # resolution of saved rasters
    "formats"    : ("png", "pdf"),  # file formats of a saved figure
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
    """A copy of ``DEFAULTS`` with ``overrides`` applied.

    Raises
    ------
    KeyError
        If an override is not a known default, so a misspelled name is not silently ignored.
    """
    unknown = sorted(set(overrides) - set(DEFAULTS))
    if unknown:
        raise KeyError(f"unknown defaults {unknown}; available: {sorted(DEFAULTS)}")
    return Defaults({**DEFAULTS, **overrides})


# ---------------------------------------------
#! EOF
# ---------------------------------------------
