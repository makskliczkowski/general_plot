r"""Read detached HDF5 datasets, cache individual datasets and write files atomically.

The manager owns file lifetimes. No open h5py objects escape its methods.
h5py is optional and imported only when a file operation is requested.

-------------------------------------------
Author: Maksymilian Kliczkowski
File: general_plot/data/hdf5.py
-------------------------------------------
"""
from __future__         import annotations

import json
import os
import tempfile
from collections.abc    import Mapping
from pathlib            import Path
from fnmatch            import fnmatchcase
from typing             import Any, Callable, Iterable, Optional, Union

import numpy            as np

# Import provenance functions for metadata handling
from .provenance        import with_provenance
from ._diagnostics      import measured

PathLike                = Union[str, Path]


def _dataset_keys(file) -> tuple:
    """Walk dataset names without touching array contents."""
    import h5py
    names = []
    file.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
    return tuple(names)


def _read_dataset(dataset, selection, transform):
    """Decode UTF-8 text and apply a caller transform after the selected read."""
    import h5py
    text = h5py.check_string_dtype(dataset.dtype)
    values = dataset.asstr()[selection] if text is not None and text.encoding == "utf-8" else dataset[selection]
    return transform(values) if transform is not None else values


def _write_dataset(file, name, values):
    """Store numeric/structured arrays unchanged and text as variable-length UTF-8."""
    import h5py
    array = np.asarray(values)
    if array.dtype.kind == "U" or (array.dtype.kind == "O" and all(isinstance(value, str) for value in array.flat)):
        file.create_dataset(name, data=array.astype(object), dtype=h5py.string_dtype("utf-8"))
    elif array.dtype.kind == "O" and all(isinstance(value, bytes) for value in array.flat):
        file.create_dataset(name, data=array, dtype=h5py.string_dtype("ascii"))
    elif array.dtype.kind == "O":
        raise TypeError(f"{name}: arbitrary Python objects are unsupported; supply a numeric, structured or text array")
    else:
        file.create_dataset(name, data=array)


class HDF5Manager:
    """Detached HDF5 I/O with explicit dataset paths and JSON metadata.

    Dataset paths are relative to the file root. Missing datasets and invalid
    files raise errors. Writers replace complete files rather than editing inputs.
    """

    @staticmethod
    def keys(path: PathLike, pattern: str = "*") -> tuple:
        """List dataset paths without reading array contents.

        Parameters
        ----------
        path : path-like
            Existing HDF5 file.
        pattern : str
            Glob matching complete dataset paths. Defaults to all paths.

        Returns
        -------
        tuple of str
            Sorted paths relative to the file root.
        """
        import h5py
        with h5py.File(path, "r") as file:
            return tuple(name for name in _dataset_keys(file) if fnmatchcase(name, pattern))

    @staticmethod
    def describe(path: PathLike, pattern: str = "*") -> dict:
        """Browse shapes, dtypes and compression without reading array contents.

        Parameters
        ----------
        path : path-like
            Existing HDF5 file.
        pattern : str
            Glob matching complete dataset paths, for example 'history/*'.

        Returns
        -------
        dict
            Dataset names mapped to shape, dtype, compression and text encoding.
        """
        import h5py
        with h5py.File(path, "r") as file:
            descriptions = {}
            for name in _dataset_keys(file):
                if fnmatchcase(name, pattern):
                    dataset = file[name]
                    text = h5py.check_string_dtype(dataset.dtype)
                    descriptions[name] = {"shape": dataset.shape, "dtype": str(dataset.dtype),
                                          "compression": dataset.compression, "encoding": text.encoding if text else None}
            return descriptions

    @staticmethod
    def read(path: PathLike, keys: Optional[Iterable[str]] = None, *, selection=(), transform: Optional[Callable[[Any], Any]] = None, verbose: bool = False) -> dict:
        """Read selected dataset elements and optionally transform their NumPy values.

        Parameters
        ----------
        path : path-like
            Existing HDF5 file.
        keys : iterable of str, optional
            Dataset paths. None reads every dataset. An empty iterable reads none.
        selection : slice or index tuple
            h5py indexing, applied to every selected dataset before materialization.
            The default () reads complete datasets. Shapes must support the selection.
        transform : callable, optional
            Receives each detached selected value. Use NumPy functions or a lambda
            for magnitudes, reshaping, reductions or value filters. Source data is unchanged.

        verbose : bool
            Emit timing, returned NumPy-buffer bytes and process peak RSS through
            the general_plot.data logger at INFO. Defaults to quiet.

        Returns
        -------
        dict
            Paths mapped to numeric arrays/scalars, decoded UTF-8 text, or transformed
            values. Fixed-length byte arrays retain their byte dtype.
        """
        with measured("hdf5.read", verbose) as report:
            import h5py
        
            # Open the HDF5 file, determine which keys to read, and return a dictionary of selected datasets with optional transformations.
            with h5py.File(path, "r") as file:
                if keys is None:
                    keys = _dataset_keys(file)
                if isinstance(keys, str):
                    raise TypeError("keys must be an iterable of dataset paths, not a string")
                result = {key: _read_dataset(file[key], selection, transform) for key in keys}
                report.update(result=result, items=len(result))
                return result
    @staticmethod
    def read_dataset(path: PathLike, key: str, *, selection=(),
                     transform: Optional[Callable[[Any], Any]] = None, verbose: bool = False) -> Any:
        """Read one dataset or hyperslab and optionally transform its selected values.

        Parameters
        ----------
        path : path-like
            Existing HDF5 file.
        key : str
            Dataset path.
        selection : slice or index tuple
            h5py indexing. Only selected elements are read. Defaults to the whole dataset.
        transform : callable, optional
            NumPy operation on the detached selection. It never edits the source file.

        verbose : bool
            Emit timing, returned NumPy-buffer bytes and process peak RSS through
            the general_plot.data logger at INFO. Defaults to quiet.

        Returns
        -------
        object
            Selected values or the transform result, with no open file handle.
        """
        with measured("hdf5.read_dataset", verbose) as report:
            import h5py
            with h5py.File(path, "r") as file:
                result = _read_dataset(file[key], selection, transform)
                report.update(result=result, items=1)
                return result
    @staticmethod
    def read_record(path: PathLike, keys: Iterable[str] = (), *, groups: Iterable[str] = (), verbose: bool = False) -> tuple[dict, dict]:
        """Read JSON metadata and selected datasets in one file opening.

        Parameters
        ----------
        path : path-like
            HDF5 file with the JSON 'metadata' attribute.
        keys : iterable of str
            Required dataset paths. Missing datasets raise KeyError.
        groups : iterable of str
            Optional group paths. Read all datasets below each existing group.
            Missing groups contribute no datasets. Existing non-groups raise
            TypeError. Unselected datasets are not materialized.
        verbose : bool
            Emit an INFO timing and memory report through general_plot.data.

        Returns
        -------
        tuple of dict
            Metadata and detached arrays keyed by full dataset paths. Overlapping
            selections read each dataset once. No project schema or saved Python
            objects are inferred. Missing or invalid metadata raises an error.
        """
        if isinstance(keys, str) or isinstance(groups, str):
            raise TypeError("keys and groups must be iterables of paths, not strings")
        with measured("hdf5.read_record", verbose) as report:
            import h5py
            with h5py.File(path, "r") as file:
                metadata = json.loads(file.attrs["metadata"])
                if not isinstance(metadata, dict):
                    raise TypeError("metadata must be a JSON object")
                selected = {}
                for name in keys:
                    dataset = file[name]
                    if not isinstance(dataset, h5py.Dataset):
                        raise TypeError(f"expected HDF5 dataset: {name!r}")
                    selected[dataset.name.lstrip("/")] = None
                for name in groups:
                    if name in file:
                        group = file[name]
                        if not isinstance(group, h5py.Group):
                            raise TypeError(f"expected HDF5 group: {name!r}")
                        selected.update(dict.fromkeys(f"{group.name}/{key}".lstrip("/") for key in _dataset_keys(group)))
                arrays = {name: _read_dataset(file[name], (), None) for name in selected}
                report.update(result=arrays, items=len(arrays))
                return metadata, arrays

    @staticmethod
    def metadata(path: PathLike) -> dict:
        """Read the JSON metadata attribute without reading datasets.

        Parameters
        ----------
        path : path-like
            Existing HDF5 file with a JSON 'metadata' attribute.

        Returns
        -------
        dict
            Decoded metadata. Missing attributes raise KeyError.
        """
        import h5py
        with h5py.File(path, "r") as file:
            return json.loads(file.attrs["metadata"])

    @staticmethod
    def write(path: PathLike, datasets: Mapping[str, Any], *, metadata: Optional[Mapping[str, Any]] = None, verbose: bool = False) -> Path:
        """Atomically replace a complete file with datasets and JSON metadata.

        Parameters
        ----------
        path : path-like
            Destination. Parent directories are created. Existing files are replaced.
        datasets : mapping
            Normalized relative dataset paths to array-like or scalar values.
            Paths cannot contain one another. Real, complex, integer, boolean and
            structured arrays retain their dtypes and shapes, including scalars and
            empty arrays. Unicode and string arrays use UTF-8. Arbitrary Python
            object arrays are rejected; no pickle serialization is used.
        metadata : mapping, optional
            JSON provenance supplied by the caller. Missing provenance.writer is filled
            with safe local host facts. Supplied writer/producer records are preserved.

        verbose : bool
            Emit timing, returned NumPy-buffer bytes and process peak RSS through
            the general_plot.data logger at INFO. Defaults to quiet.

        Returns
        -------
        Path
            Written file. On failure the previous destination remains intact.
        """
        with measured("hdf5.write", verbose) as report:
            import h5py
            encoded = json.dumps(with_provenance(metadata))
            for name in datasets:
                if not isinstance(name, str) or not name or any(part in ("", ".", "..") for part in name.split("/")):
                    raise ValueError(f"invalid dataset path: {name!r}")
            
                parts = name.split("/")
                if any("/".join(parts[:i]) in datasets for i in range(1, len(parts))):
                    raise ValueError(f"dataset paths overlap: {name!r}")
            
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            os.close(descriptor)
        
            # Write to a temporary file and replace the destination atomically. Clean up the temporary file on failure.
            try:
                with h5py.File(temporary, "w") as file:
                    for name, values in datasets.items():
                        _write_dataset(file, name, values)
                    file.attrs["metadata"] = encoded
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            report.update(result=datasets, items=len(datasets))
            return path


class LazyHDF5(Mapping):
    """Browse and filter dataset headers, caching full datasets on first access.

    Parameters
    ----------
    path : path-like
        Existing HDF5 file. Headers are discovered once. Treat the source file as
        unchanged for this object's lifetime. No file handles are retained.
    pattern : str
        Glob selecting dataset paths, for example 'history/*'. Iteration, membership
        and describe read no arrays. Index access caches a complete dataset.
    verbose : bool
        Emit INFO reports for reads and cache hits. Defaults to quiet.
    """
    def __init__(self, path: PathLike, pattern: str = "*", *, verbose: bool = False):
        self.path       = Path(path).resolve()
        self._info      = HDF5Manager.describe(self.path, pattern)
        self._cache     = {}
        self.verbose    = verbose

    def __iter__(self):
        return iter(self._info)

    def __len__(self):
        return len(self._info)

    def __contains__(self, key):
        return key in self._info

    def __repr__(self):
        return f"LazyHDF5(path={str(self.path)!r}, datasets={len(self)}, cached={len(self._cache)})"

    def __getitem__(self, key):
        with measured("hdf5.cached_read", self.verbose) as report:
            if key not in self._info:
                raise KeyError(key)
            report["cache_hit"] = key in self._cache
            if key not in self._cache:
                self._cache[key] = HDF5Manager.read_dataset(self.path, key)
            report.update(result=self._cache[key], items=1)
            return self._cache[key]


    def describe(self) -> dict:
        """Return the selected dataset headers without materializing arrays."""
        return {key: dict(value) for key, value in self._info.items()}

    def read(self, key: str, *, selection=(), transform: Optional[Callable[[Any], Any]] = None, verbose: Optional[bool] = None) -> Any:
        """Read selected elements of a visible dataset, bypassing the full-array cache.

        Parameters
        ----------
        key : str
            A dataset included by this view's path pattern.
        selection : slice or index tuple
            h5py indexing. Only the selection is materialized.
        transform : callable, optional
            NumPy operation applied after reading. The source and cache are unchanged.

        verbose : bool, optional
            Override this object's verbosity for this partial read.

        Returns
        -------
        object
            The detached selection or transformed value.
        """
        if key not in self._info:
            raise KeyError(key)
        return HDF5Manager.read_dataset(self.path, key, selection=selection, transform=transform, verbose=self.verbose if verbose is None else verbose)

    def clear_cache(self) -> None:
        """Release full-array references while retaining the header snapshot."""
        self._cache.clear()


# -------------------------------------------
#! EOF
# -------------------------------------------
