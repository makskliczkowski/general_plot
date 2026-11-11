r"""Opt-in timing and memory reports for data operations.

-------------------------------------------
Author: Maksymilian Kliczkowski
File: general_plot/data/_diagnostics.py
-------------------------------------------
"""
from __future__ import annotations

import logging
import sys
from contextlib     import contextmanager
from dataclasses    import is_dataclass
from time           import perf_counter

import numpy as np

# The logger is configured by general_plot.data.__init__.py. It is not a library logger.
LOGGER = logging.getLogger("general_plot.data")

# -------------------------------------------

def _array_bytes(value):
    """Count logical NumPy buffers without materializing lazy mappings or properties."""
    seen = set()
    found = False

    def count(item):
        nonlocal found
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item, (np.ndarray, np.generic)):
            found = True
            return item.nbytes
        if is_dataclass(item) and not isinstance(item, type) and hasattr(item, "__dict__"):
            item = vars(item)
        if isinstance(item, dict):
            return sum(count(part) for part in item.values())
        if isinstance(item, (list, tuple)):
            return sum(count(part) for part in item)
        return 0

    total = count(value)
    return total if found else None


def _peak_rss():
    """Return process peak RSS in bytes on Linux/macOS, otherwise unknown."""
    if sys.platform not in ("linux", "darwin"):
        return None
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


@contextmanager
def measured(operation: str, verbose: bool):
    """Emit one INFO record on exit without changing return values or exceptions."""
    details = {}
    if not verbose:
        yield details
        return
    started = perf_counter()
    try:
        yield details
    finally:
        elapsed = perf_counter() - started
        buffers = _array_bytes(details.get("result"))
        peak    = _peak_rss()
        LOGGER.info("%s status=%s wall_s=%.6f array_bytes=%s peak_rss_bytes=%s workspace_bytes=%s items=%s cache_hit=%s", operation, "error" if sys.exc_info()[0] is not None else "ok", elapsed, buffers if buffers is not None else "n/a", peak if peak is not None else "n/a", details.get("workspace_bytes", "n/a"), details.get("items", "n/a"), details.get("cache_hit", "n/a"))


# -------------------------------------------
#! EOF
# -------------------------------------------
