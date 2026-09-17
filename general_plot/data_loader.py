"""Data loading and filtering helpers for experiment and simulation results."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Callable, List, Optional, Tuple


def filter_results(results         : Iterable[Any],
                   filters         : Optional[Mapping[str, Any]] = None,
                   get_params_fun  : Optional[Callable[[Any], Mapping[str, Any]]] = None,
                   *,
                   tol             : float = 1e-9) -> List[Any]:
    """Filter experiment or simulation results by matching parameter values.

    Parameters
    ----------
    results : iterable
        Collection of result objects or dictionaries to filter.
    filters : dict, optional
        Key-value mapping of parameters that each result must satisfy.
        If numeric, matching uses absolute tolerance `tol`.
        If the target is a set/list/tuple, membership is checked.
    get_params_fun : callable, optional
        Function `f(item) -> dict` returning parameters for each result item.
        Defaults to inspecting dictionary keys or object attributes.
    tol : float, default=1e-9
        Tolerance for floating-point equality comparisons.

    Returns
    -------
    list
        Filtered list of matching items.
    """
    if not filters:
        return list(results)

    filtered: List[Any] = []

    def _get_val(item: Any, key: str) -> Tuple[bool, Any]:
        if get_params_fun is not None:
            params = get_params_fun(item)
            if isinstance(params, Mapping) and key in params:
                return True, params[key]
            if hasattr(params, key):
                return True, getattr(params, key)
            return False, None

        if isinstance(item, Mapping) and key in item:
            return True, item[key]
        if hasattr(item, "params") and isinstance(item.params, Mapping) and key in item.params:
            return True, item.params[key]
        if hasattr(item, key):
            return True, getattr(item, key)
        return False, None

    def _matches(actual: Any, target: Any) -> bool:
        if isinstance(target, (int, float)) and isinstance(actual, (int, float)):
            return math.isclose(float(actual), float(target), abs_tol=tol, rel_tol=tol)
        if isinstance(target, (set, list, tuple)):
            return actual in target
        return actual == target

    for item in results:
        match = True
        for key, target in filters.items():
            found, val = _get_val(item, key)
            if not found or not _matches(val, target):
                match = False
                break
        if match:
            filtered.append(item)

    return filtered


# ---------------------------------------------
#! EOF
# ---------------------------------------------
