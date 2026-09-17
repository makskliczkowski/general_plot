"""Subplot container and ignored-axis proxy objects for grid layouts."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence

from matplotlib.gridspec import GridSpec
import numpy as np


# ---------------------------------------------


class _IgnoredAxisAttr:
    """Chainable no-op attribute proxy used by disabled axes."""

    def __init__(self, owner: Any, path: str):
        self._owner = owner
        self._path = str(path)

    def __call__(self, *args: Any, **kwargs: Any) -> _IgnoredAxisAttr:
        self._owner._warn(self._path)
        return self

    def __getattr__(self, name: str) -> _IgnoredAxisAttr:
        return _IgnoredAxisAttr(self._owner, f"{self._path}.{name}")

    def __getitem__(self, key: Any) -> _IgnoredAxisAttr:
        if isinstance(key, int):
            raise IndexError(key)
        self._owner._warn(f"{self._path}[{key!r}]")
        return self

    def __setitem__(self, key: Any, value: Any) -> None:
        self._owner._warn(f"{self._path}[{key!r}]=...")


class IgnoredAxis:
    """No-op stand-in for a disabled axis.

    Any call or chained attribute access is ignored by default.
    Optionally, warnings can be emitted to make ignored operations explicit.
    """

    def __init__(
        self,
        *,
        index: Optional[int] = None,
        row_col: Optional[tuple[int, int]] = None,
        names: Optional[Sequence[str]] = None,
        warn: bool = False,
        reason: Optional[str] = None,
    ):
        self._qes_axis_disabled = True
        self._qes_axis_index = index
        self._qes_axis_row_col = row_col
        self._qes_axis_names = list(names or [])
        self._qes_axis_warn = bool(warn)
        self._qes_axis_reason = str(reason) if reason else None

    @property
    def is_disabled(self) -> bool:
        """Return True for compatibility with regular axes checks."""
        return True

    def set_warn(self, enabled: bool = True) -> IgnoredAxis:
        """Enable or disable warnings for ignored axis operations."""
        self._qes_axis_warn = bool(enabled)
        return self

    def description(self) -> str:
        """Return a compact description of the disabled axis target."""
        bits = []
        if self._qes_axis_names:
            bits.append(f"name={self._qes_axis_names}")
        if self._qes_axis_row_col is not None:
            bits.append(f"position={self._qes_axis_row_col}")
        if self._qes_axis_index is not None:
            bits.append(f"index={self._qes_axis_index}")
        if self._qes_axis_reason:
            bits.append(f"reason='{self._qes_axis_reason}'")
        return ", ".join(bits) if bits else "unknown axis"

    def _warn(self, op: str) -> None:
        if not self._qes_axis_warn:
            return
        warnings.warn(
            f"Ignored call '{op}' on disabled axis ({self.description()}).",
            RuntimeWarning,
            stacklevel=3,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> IgnoredAxis:
        self._warn("__call__")
        return self

    def __getattr__(self, name: str) -> Any:
        if str(name).startswith("__array"):
            raise AttributeError(name)
        return _IgnoredAxisAttr(self, str(name))

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"IgnoredAxis({self.description()})"


class AxesList(list):
    """List-like container for subplot axes with grid-aware helpers.

    Behaviors:
    - Inherits from list.
    - Supports 2D indexing when grid metadata is available: axes[row, col].
    - Supports named panel access: axes['panel_name'].
    - Forwards unknown attribute access to the first axis.
    """

    def __init__(
        self,
        axes: Sequence[Any],
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        panel_map: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(list(axes))
        self.nrows = int(nrows) if nrows is not None else None
        self.ncols = int(ncols) if ncols is not None else None
        self._panel_map = dict(panel_map or {})

    @property
    def shape(self) -> Optional[tuple[int, int]]:
        """Grid shape (nrows, ncols) when known, otherwise None."""
        if self.nrows is None or self.ncols is None:
            return None
        return (self.nrows, self.ncols)

    def first(self) -> Any:
        """Return the first axis or raise if the container is empty."""
        if len(self) == 0:
            raise IndexError("AxesList is empty.")
        return self[0]

    @property
    def panel_names(self) -> List[str]:
        """Names of registered semantic panels."""
        return list(self._panel_map.keys())

    def has_panel(self, name: str) -> bool:
        """Return whether a semantic panel name is registered."""
        return str(name) in self._panel_map

    def panel(self, name: str) -> Any:
        """Return the axis or nested axes registered for `name`."""
        key = str(name)
        if key not in self._panel_map:
            raise KeyError(f"Unknown panel '{name}'. Available: {self.panel_names}")
        return self._panel_map[key]

    def panels(self) -> Dict[str, Any]:
        """Return a copy of the semantic panel map."""
        return dict(self._panel_map)

    def rename_panel(self, old: str, new: str) -> AxesList:
        """Rename a semantic panel while preserving its mapped axes."""
        old_k, new_k = str(old), str(new)
        if old_k not in self._panel_map:
            raise KeyError(f"Unknown panel '{old}'.")
        if new_k in self._panel_map and new_k != old_k:
            raise KeyError(f"Panel '{new}' already exists.")
        self._panel_map[new_k] = self._panel_map.pop(old_k)
        return self

    def select(self, *names: str) -> AxesList:
        """Return an AxesList containing only the named panels."""
        selected = [self.panel(name) for name in names]
        flat = []
        for entry in selected:
            if isinstance(entry, (AxesList, list)):
                flat.extend(entry)
            else:
                flat.append(entry)
        sub_map = {str(name): self.panel(name) for name in names}
        return AxesList(flat, panel_map=sub_map)

    def as_grid(self) -> np.ndarray:
        """Return axes as a 2D rectangular object array using stored grid shape."""
        if self.shape is None:
            raise ValueError("Grid shape is unknown for this AxesList.")
        expected = int(self.nrows) * int(self.ncols)
        if len(self) != expected:
            raise ValueError(
                f"AxesList grid shape mismatch: len={len(self)} but nrows*ncols={expected}."
            )
        return np.asarray(self, dtype=object).reshape(self.nrows, self.ncols)

    def at(self, row: int, col: int) -> Any:
        """Return the axis at grid location (row, col)."""
        if self.shape is None:
            raise ValueError("Grid shape is unknown for this AxesList.")
        return self[row, col]

    def span(self, rows: Any, cols: Any) -> Any:
        """Return axes in a rectangular grid window."""
        return self[rows, cols]

    def row(self, row: int) -> AxesList:
        """Return one grid row as an AxesList."""
        if self.shape is None or self.ncols is None:
            raise ValueError("Grid shape is unknown for this AxesList.")
        start = row * self.ncols
        end = start + self.ncols
        return AxesList(self[start:end], nrows=1, ncols=self.ncols)

    def col(self, col: int) -> AxesList:
        """Return one grid column as an AxesList."""
        if self.shape is None or self.nrows is None:
            raise ValueError("Grid shape is unknown for this AxesList.")
        return AxesList([self[r, col] for r in range(self.nrows)], nrows=self.nrows, ncols=1)

    def set_title(self, title: str, **kwargs: Any) -> AxesList:
        """Set title on all axes."""
        return self.apply(lambda ax: ax.set_title(title, **kwargs))

    def apply(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> AxesList:
        """Apply `fn(ax, *args, **kwargs)` to each axis and return self."""
        for ax in self:
            fn(ax, *args, **kwargs)
        return self

    def _subset(self, items: Sequence[Any]) -> AxesList:
        items_list = list(items)
        kept_ids = {id(ax) for ax in items_list}
        sub_map = {name: ax for name, ax in self._panel_map.items() if id(ax) in kept_ids}
        return AxesList(items_list, panel_map=sub_map)

    def disable(
        self,
        target: Any,
        *,
        warn: bool = False,
        hide: bool = True,
        reason: Optional[str] = None,
    ) -> AxesList:
        """Disable one or more axes by replacing them with IgnoredAxis placeholders."""

        def _panel_names_for_axis(ax_obj: Any) -> List[str]:
            return [name for name, ref in self._panel_map.items() if ref is ax_obj]

        def _row_col_for_index(index: int) -> Optional[tuple[int, int]]:
            if self.shape is None or self.ncols in (None, 0):
                return None
            return (int(index // self.ncols), int(index % self.ncols))

        def _disable_index(index: int) -> None:
            if index < 0 or index >= len(self):
                raise IndexError(f"Axis index out of range: {index}")

            axis_obj = super(AxesList, self).__getitem__(index)
            if isinstance(axis_obj, IgnoredAxis):
                axis_obj.set_warn(warn)
                return

            if hide:
                try:
                    axis_obj.set_axis_off()
                except Exception:
                    pass

            names = _panel_names_for_axis(axis_obj)
            proxy = IgnoredAxis(
                index=index,
                row_col=_row_col_for_index(index),
                names=names,
                warn=warn,
                reason=reason,
            )
            super(AxesList, self).__setitem__(index, proxy)

            for name, ref in list(self._panel_map.items()):
                if ref is axis_obj:
                    self._panel_map[name] = proxy

        if isinstance(target, tuple) and len(target) == 2:
            if self.shape is None:
                raise ValueError("Tuple indexing requires known grid shape.")
            row, col = target
            flat_idx = np.ravel_multi_index((int(row), int(col)), (self.nrows, self.ncols))
            _disable_index(int(flat_idx))
            return self

        if isinstance(target, (list, tuple, np.ndarray)):
            for item in list(np.asarray(target, dtype=object).ravel()):
                self.disable(item, warn=warn, hide=hide, reason=reason)
            return self

        if isinstance(target, str):
            item = self.panel(target)
            if isinstance(item, (AxesList, list, tuple, np.ndarray)):
                self.disable(item, warn=warn, hide=hide, reason=reason)
                return self
            try:
                idx = self.index(item)
            except ValueError as exc:
                raise KeyError(f"Panel '{target}' is not mapped to a direct axis.") from exc
            _disable_index(int(idx))
            return self

        if isinstance(target, (int, np.integer)):
            _disable_index(int(target))
            return self

        try:
            idx = self.index(target)
        except ValueError as exc:
            raise TypeError(
                "Unsupported target for AxesList.disable. Use index, (row,col), panel name, axis, or list-like."
            ) from exc
        _disable_index(int(idx))
        return self

    def collapse(self, *, redraw: bool = True) -> AxesList:
        """Collapse rows that are fully disabled and reflow remaining axes."""
        if self.shape is None:
            raise ValueError("Grid shape is unknown for this AxesList.")

        expected = int(self.nrows) * int(self.ncols)
        if len(self) != expected:
            raise ValueError(
                f"AxesList grid shape mismatch: len={len(self)} but nrows*ncols={expected}."
            )

        def _is_disabled(ax_obj: Any) -> bool:
            return isinstance(ax_obj, IgnoredAxis) or bool(getattr(ax_obj, "_qes_axis_disabled", False))

        grid = np.asarray(self, dtype=object).reshape(self.nrows, self.ncols)
        disabled_rows = [r for r in range(self.nrows) if all(_is_disabled(grid[r, c]) for c in range(self.ncols))]

        if not disabled_rows:
            return self

        if len(disabled_rows) == int(self.nrows):
            warnings.warn("All rows are disabled; collapse() skipped.", RuntimeWarning, stacklevel=2)
            return self

        kept_rows = [r for r in range(self.nrows) if r not in disabled_rows]
        fig = None
        for r in kept_rows:
            for c in range(self.ncols):
                candidate = grid[r, c]
                if not _is_disabled(candidate):
                    fig = getattr(candidate, "figure", None)
                    if fig is not None:
                        break
            if fig is not None:
                break

        if fig is None:
            return self

        sp = fig.subplotpars
        new_gs = GridSpec(
            len(kept_rows),
            self.ncols,
            figure=fig,
            left=sp.left,
            right=sp.right,
            bottom=sp.bottom,
            top=sp.top,
            wspace=sp.wspace,
            hspace=sp.hspace,
        )

        for new_r, old_r in enumerate(kept_rows):
            for c in range(self.ncols):
                ax = grid[old_r, c]
                if _is_disabled(ax):
                    continue
                slot = new_gs[new_r, c]
                ax.set_position(slot.get_position(fig))
                if hasattr(ax, "set_subplotspec"):
                    try:
                        ax.set_subplotspec(slot)
                    except Exception:
                        pass

        new_items = [grid[r, c] for r in kept_rows for c in range(self.ncols)]
        self[:] = new_items
        self.nrows = len(kept_rows)

        kept_ids = {id(ax) for ax in new_items}
        self._panel_map = {name: ax for name, ax in self._panel_map.items() if id(ax) in kept_ids}

        if redraw:
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass

        return self

    def adjust(
        self,
        same: str = "xy",
        *,
        hide: str = "both",
        keep_x: str = "bottom",
        keep_y: str = "left",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlabel_kwargs: Optional[dict] = None,
        ylabel_kwargs: Optional[dict] = None,
        x_label_position: Optional[str] = None,
        y_label_position: Optional[str] = None,
        x_label_coords: Optional[tuple[float, float]] = None,
        y_label_coords: Optional[tuple[float, float]] = None,
        x_label_coords_system: str = "axes",
        y_label_coords_system: str = "axes",
        x_tick_params: Optional[dict] = None,
        y_tick_params: Optional[dict] = None,
        interior_x_tick_params: Optional[dict] = None,
        interior_y_tick_params: Optional[dict] = None,
    ) -> AxesList:
        """Remove duplicated axis labels/ticklabels for multi-panel layouts."""
        key = str(same).lower().strip()
        do_x = "x" in key
        do_y = "y" in key
        hide_key = str(hide).lower().strip()
        hide_labels = hide_key in {"both", "label", "labels"}
        hide_ticklabels = hide_key in {"both", "tick", "ticks", "ticklabel", "ticklabels"}
        xlabel_kwargs = dict(xlabel_kwargs or {})
        ylabel_kwargs = dict(ylabel_kwargs or {})
        x_tick_params = dict(x_tick_params or {})
        y_tick_params = dict(y_tick_params or {})
        interior_x_tick_params = dict(interior_x_tick_params or {})
        interior_y_tick_params = dict(interior_y_tick_params or {})

        keep_x = str(keep_x).lower().strip()
        keep_y = str(keep_y).lower().strip()
        if keep_x not in {"bottom", "top", "all", "vbottom", "vtop"}:
            raise ValueError("keep_x must be one of: 'bottom', 'top', 'all', 'vbottom', 'vtop'.")
        if keep_y not in {"left", "right", "all", "vleft", "vright"}:
            raise ValueError("keep_y must be one of: 'left', 'right', 'all', 'vleft', 'vright'.")

        force_x_side = None
        if keep_x in {"vbottom", "vtop"}:
            force_x_side = "bottom" if keep_x == "vbottom" else "top"
            keep_x = force_x_side
        force_y_side = None
        if keep_y in {"vleft", "vright"}:
            force_y_side = "left" if keep_y == "vleft" else "right"
            keep_y = force_y_side

        def _edge_flags(ax: Any, idx: int) -> dict[str, bool]:
            try:
                ss = ax.get_subplotspec()
                return {
                    "first_row": bool(ss.is_first_row()),
                    "last_row": bool(ss.is_last_row()),
                    "first_col": bool(ss.is_first_col()),
                    "last_col": bool(ss.is_last_col()),
                }
            except Exception:
                pass

            if self.shape is not None and self.ncols is not None and self.ncols > 0:
                row = idx // self.ncols
                col = idx % self.ncols
                return {
                    "first_row": row == 0,
                    "last_row": row == (self.nrows - 1),
                    "first_col": col == 0,
                    "last_col": col == (self.ncols - 1),
                }
            return {"first_row": True, "last_row": True, "first_col": True, "last_col": True}

        for idx, ax in enumerate(self):
            flags = _edge_flags(ax, idx)

            if do_x:
                keep_this_x = (
                    True
                    if keep_x == "all"
                    else flags["last_row"] if keep_x == "bottom"
                    else flags["first_row"]
                )
                if not keep_this_x:
                    if hide_ticklabels:
                        ax.tick_params(axis="x", which="both", labelbottom=False, labeltop=False)
                    if interior_x_tick_params:
                        ax.tick_params(axis="x", which="both", **interior_x_tick_params)
                    if hide_labels:
                        ax.set_xlabel("")
                else:
                    if xlabel is not None:
                        ax.set_xlabel(xlabel, **xlabel_kwargs)
                    if x_label_position in {"top", "bottom"}:
                        ax.xaxis.set_label_position(x_label_position)
                    elif force_x_side is not None:
                        ax.xaxis.set_label_position(force_x_side)
                    if x_label_coords is not None:
                        x_t = ax.transData if str(x_label_coords_system).lower() == "data" else ax.transAxes
                        ax.xaxis.set_label_coords(float(x_label_coords[0]), float(x_label_coords[1]), transform=x_t)
                    if force_x_side is not None:
                        if force_x_side == "top":
                            ax.tick_params(axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False)
                        else:
                            ax.tick_params(axis="x", which="both", top=False, bottom=True, labeltop=False, labelbottom=True)
                    if x_tick_params:
                        ax.tick_params(axis="x", which="both", **x_tick_params)

            if do_y:
                keep_this_y = (
                    True
                    if keep_y == "all"
                    else flags["first_col"] if keep_y == "left"
                    else flags["last_col"]
                )
                if not keep_this_y:
                    if hide_ticklabels:
                        ax.tick_params(axis="y", which="both", labelleft=False, labelright=False)
                    if interior_y_tick_params:
                        ax.tick_params(axis="y", which="both", **interior_y_tick_params)
                    if hide_labels:
                        ax.set_ylabel("")
                else:
                    if ylabel is not None:
                        ax.set_ylabel(ylabel, **ylabel_kwargs)
                    if y_label_position in {"left", "right"}:
                        ax.yaxis.set_label_position(y_label_position)
                    elif force_y_side is not None:
                        ax.yaxis.set_label_position(force_y_side)
                    if y_label_coords is not None:
                        y_t = ax.transData if str(y_label_coords_system).lower() == "data" else ax.transAxes
                        ax.yaxis.set_label_coords(float(y_label_coords[0]), float(y_label_coords[1]), transform=y_t)
                    if force_y_side is not None:
                        if force_y_side == "right":
                            ax.tick_params(axis="y", which="both", right=True, left=False, labelright=True, labelleft=False)
                        else:
                            ax.tick_params(axis="y", which="both", right=False, left=True, labelright=False, labelleft=True)
                    if y_tick_params:
                        ax.tick_params(axis="y", which="both", **y_tick_params)

        return self

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self.panel(key)

        if isinstance(key, tuple):
            if self.shape is None:
                raise ValueError("Tuple indexing requires known grid shape.")

            if len(key) != 2:
                raise IndexError("Use axes[row, col] for 2D indexing.")
            row, col = key

            if isinstance(row, (int, np.integer)) and isinstance(col, (int, np.integer)):
                flat_idx = int(np.ravel_multi_index((int(row), int(col)), (self.nrows, self.ncols)))
                return super().__getitem__(flat_idx)

            grid = self.as_grid()
            result = grid[row, col]
            if isinstance(result, np.ndarray):
                flat = result.ravel().tolist()
                return self._subset(flat)
            return result

        if isinstance(key, slice):
            return self._subset(super().__getitem__(key))

        if isinstance(key, (list, tuple, np.ndarray)):
            arr = np.asarray(key)
            if arr.dtype == bool:
                if arr.size != len(self):
                    raise IndexError("Boolean index length must match AxesList length.")
                idx = np.flatnonzero(arr)
                return self._subset([super().__getitem__(int(i)) for i in idx])
            return self._subset([super().__getitem__(int(i)) for i in arr.ravel()])

        return super().__getitem__(key)

    def __getattr__(self, name: str) -> Any:
        if len(self) == 0:
            raise AttributeError(name)
        return getattr(self[0], name)


# ---------------------------------------------
#! EOF
# ---------------------------------------------
