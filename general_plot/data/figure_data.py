r"""Store detached figure series and caller-supplied provenance in HDF5.

Files contain named groups with x and y arrays. Writers replace the destination
atomically after validation. h5py is imported only when these functions are used.

-------------------------------------------
Author: Maksymilian Kliczkowski
File: general_plot/data/figure_data.py
-------------------------------------------
"""
from __future__ import annotations

from pathlib    import Path
from typing     import Any, Mapping, Optional, Union

import numpy    as np

from .hdf5      import HDF5Manager


def write_figure_data(path: Union[str, Path], series: Mapping[str, tuple], *, metadata: Optional[Mapping[str, Any]] = None, verbose: bool = False) -> Path:
    """Atomically replace an HDF5 figure-data file with named (x, y) series.

    Parameters
    ----------
    path : path-like
        Output file. Parent directories are created. Existing files are replaced.
    series : mapping
        Relative group paths to (x, y) array pairs with equal first dimensions.
        Group paths must be unique, normalized and cannot contain one another.
    metadata : mapping, optional
        Caller-owned JSON provenance. Stored as the 'metadata' file attribute.
        
        Missing writer provenance is filled with safe local host facts. Supplied
        source records are preserved. No repository or project semantics are inferred.
    verbose : bool
        Emit INFO timing and memory reports through general_plot.data.

    Returns
    -------
    Path
        Written file. Failed writes leave the previous destination intact.
    """
    arrays = {}
    
    # Validate series names and array shapes, and convert to numpy arrays.
    for name, (x, y) in series.items():
        if not isinstance(name, str) or not name or any(part in ("", ".", "..") for part in name.split("/")):
            raise ValueError(f"invalid series group path: {name!r}")
        
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim == 0 or y.ndim == 0 or len(x) != len(y):
            raise ValueError(f"{name}: x and y need matching first dimensions")
        arrays[name] = (x, y)
        
    # Validate that no series paths overlap, and prepare the datasets dictionary.
    for name in arrays:
        parts = name.split("/")
        if any("/".join(parts[:i]) in arrays for i in range(1, len(parts))):
            raise ValueError(f"series paths overlap: {name!r}")
    
    # Prepare the datasets dictionary with keys like "group/x" and "group/y".
    datasets = {f"{name}/{axis}": values for name, pair in arrays.items() for axis, values in zip(("x", "y"), pair)}
    return HDF5Manager.write(path, datasets, metadata=metadata, verbose=verbose)


def read_figure_data(path: Union[str, Path], *, verbose: bool = False) -> dict:
    """Read named (x, y) arrays, closing the HDF5 file before returning.

    Parameters
    ----------
    path : path-like
        Figure-data file. Also reads older files with separate provenance attrs.

    verbose : bool
        Emit INFO timing and memory reports through general_plot.data.

    Returns
    -------
    dict
        Group paths mapped to detached (x, y) arrays.
    """
    keys    = set(HDF5Manager.keys(path))
    groups  = sorted(key[:-2] for key in keys if key.endswith("/x") and key[:-1] + "y" in keys)
    arrays  = HDF5Manager.read(path, [f"{name}/{axis}" for name in groups for axis in ("x", "y")], verbose=verbose)
    return {name: (arrays[f"{name}/x"], arrays[f"{name}/y"]) for name in groups}



read_figure_metadata = HDF5Manager.metadata


# -------------------------------------------
#! EOF
# -------------------------------------------
