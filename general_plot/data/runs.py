r"""Discover, select and lazily load runs described by metadata files.

Identity and ranking belong to the caller. Discovery reads metadata only.
Entries keep loaded payloads until clear_cache is called. Selections share entries.

-------------------------------------------
Author: Maksymilian Kliczkowski
File: general_plot/data/runs.py
-------------------------------------------
"""
from __future__         import annotations

import json
from collections.abc    import Mapping
from dataclasses        import dataclass, field
from pathlib            import Path
from typing             import Any, Callable, Iterable, Optional, Union

import numpy as np

from ..data_loader      import filter_results
from .directories       import discover_files
from ._diagnostics      import measured


# -------------------------------------------


def parameter_key(parameters: Mapping[str, Any]) -> str:
    """Encode JSON parameters for exact identity comparisons.

    Parameters
    ----------
    parameters : mapping
        JSON metadata defining a run. Mapping order is ignored. List order,
        numeric representation and all supplied fields are significant.

    Returns
    -------
    str
        Canonical JSON. The caller must exclude outcomes and source locations.
    """
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def parameter_differences(*parameters: Mapping[str, Any]) -> dict:
    """Compare nested JSON parameter mappings without loading payloads.

    Parameters
    ----------
    parameters : mappings
        Selected runs' parameters, in comparison order.

    Returns
    -------
    dict
        Differing leaf paths (tuples of keys) mapped to {input_index: value}.
        An absent index means the field is missing, distinct from JSON null.
        Lists are compared as whole values. Empty mappings remain leaves.
    """
    def leaves(mapping, prefix=()):
        result = {}
        for key, value in mapping.items():
            path = (*prefix, key)
            if isinstance(value, Mapping) and value:
                result.update(leaves(value, path))
            else:
                result[path] = value
        return result

    # Flatten each parameter mapping to leaf paths, then compare values across inputs.
    flattened   = [leaves(mapping) for mapping in parameters]
    paths       = dict.fromkeys(path for mapping in flattened for path in mapping)
    different   = {}
    for path in paths:
        # Compare values across inputs, using JSON encoding to distinguish NaN and -0.0.
        values  = {i: mapping[path] for i, mapping in enumerate(flattened) if path in mapping}
        encoded = {json.dumps(value, sort_keys=True) for value in values.values()}
        if len(values) != len(parameters) or len(encoded) > 1:
            different[path] = values
    return different


# -------------------------------------------


@dataclass(eq=False)
class RunEntry:
    """A metadata record with an explicitly cached, lazy payload.

    Parameters
    ----------
    path, root, key : Path, Path, str
        Payload location, discovery root and metadata path relative to that root.
    metadata : dict
        Caller-defined JSON metadata. Treat it as read only after discovery.
    modified : float
        Metadata modification time in seconds since the Unix epoch.
    loader : callable
        Receives the payload path. It must return detached data, not an open file.
    """
    path        : Path
    root        : Path
    key         : str
    metadata    : dict
    modified    : float
    loader      : Callable[[Path], Any] = field(repr=False)
    _cache      : dict = field(default_factory=dict, init=False, repr=False)

    def load(self, *, verbose: bool = False) -> Any:
        """Load once and cache the payload; verbose emits an INFO resource report."""
        with measured("run.load", verbose) as report:
            report["cache_hit"] = "data" in self._cache
            if "data" not in self._cache:
                self._cache["data"] = self.loader(self.path)
            report.update(result=self._cache["data"], items=1)
            return self._cache["data"]

    def clear_cache(self) -> None:
        """Release this entry's payload reference so the next access reloads it."""
        self._cache.clear()

    @property
    def loaded(self) -> bool:
        """Whether a successful payload load is cached, including a None result."""
        return "data" in self._cache


class LazyRunMapping(Mapping):
    """A unique-key view that loads a run only when its value is accessed.

    Parameters
    ----------
    entries : mapping
        Unique labels to RunEntry objects. The entries property exposes metadata
        without loading data. keys, len and membership are also metadata-only.
    verbose : bool
        Emit resource reports when payloads are accessed. values and items load
        their payloads as iterated. No attribute forwarding is performed.
    """
    def __init__(self, entries: Mapping, *, verbose: bool = False):
        self.entries = dict(entries)
        self.verbose = verbose

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __contains__(self, key):
        return key in self.entries

    def __getitem__(self, key):
        return self.entries[key].load(verbose=self.verbose)

    def __repr__(self):
        return f"LazyRunMapping(entries={len(self)}, loaded={sum(entry.loaded for entry in self.entries.values())})"


class RunCatalog:
    """A metadata collection supporting lazy selection and explicit copy policies.

    Parameters
    ----------
    entries : iterable of RunEntry
        Entries are retained by reference, including their payload caches.
    """
    def __init__(self, entries: Iterable[RunEntry]):
        self.entries = tuple(entries)

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    @classmethod
    def scan(cls, roots: Union[str, Path, Iterable[Union[str, Path]]], *, pattern: str,
             loader: Callable[[Path], Any], path_for: Callable[[Path], Path],
             read_metadata: Optional[Callable[[Path], Mapping[str, Any]]] = None, verbose: bool = False) -> RunCatalog:
        """Discover metadata once per file across possibly overlapping roots.

        Parameters
        ----------
        roots : path or iterable of paths
            Existing directories. Missing roots raise FileNotFoundError.
        pattern : str
            Path.glob pattern identifying metadata files, such as '**/*.json'.
        loader, path_for : callables
            Payload reader and metadata-path to payload-path mapping.
        read_metadata : callable, optional
            Metadata reader. Defaults to JSON. No payload reader is called here.

        verbose : bool
            Emit INFO timing and memory reports. Defaults to quiet.

        Returns
        -------
        RunCatalog
            Entries in root order and sorted file order. Overlapping roots keep
            the first occurrence of the same resolved metadata file.
        """
        with measured("catalog.scan", verbose) as report:
            entries = []
            for root, path in discover_files(roots, pattern):
                metadata = read_metadata(path) if read_metadata else json.loads(path.read_text())
                if not isinstance(metadata, Mapping):
                    raise TypeError(f"metadata must be a mapping: {path}")
            
                # Create a RunEntry for each discovered metadata file, using the provided loader and path mapping.
                entries.append(RunEntry(Path(path_for(path)), root, str(path.relative_to(root)), dict(metadata), path.stat().st_mtime, loader))
            report["items"] = len(entries)
            return cls(entries)
    # -------------------------------------------


    def select(self, filters: Optional[Mapping[str, Any]] = None, *, parameters: Optional[Callable[[RunEntry], Mapping[str, Any]]] = None, predicate: Optional[Callable[[RunEntry], bool]] = None, tol: float = 1e-9) -> RunCatalog:
        """Filter metadata using filter_results, then an optional entry predicate.

        Parameters
        ----------
        filters : mapping, optional
            Scalar, membership, Between or callable field filters as in filter_results.
        parameters : callable, optional
            Filter fields for an entry. Defaults to its metadata.
        predicate : callable, optional
            Additional arbitrary rule. It receives an entry without loading data.
        tol : float
            Relative and absolute tolerance for numeric filters. Identity matching
            in best is exact and independent of this filter tolerance.

        Returns
        -------
        RunCatalog
            Matching entries with shared caches.
        """
        chosen = filter_results(self.entries, filters, parameters or (lambda entry: entry.metadata), tol=tol)
        return type(self)(entry for entry in chosen if predicate is None or predicate(entry))

    def index_by(self, key: Callable[[RunEntry], Any], *, verbose: bool = False) -> LazyRunMapping:
        """Build a unique metadata index before loading any payloads.

        Parameters
        ----------
        key : callable
            A hashable label from each entry's metadata.
        verbose : bool
            Report resources when an indexed payload is requested.

        Returns
        -------
        LazyRunMapping
            A metadata-only index. Duplicate labels raise ValueError with both
            paths and differing field names. Use group_by for legitimate variants,
            or best for genuine copies, before requesting a unique index.
        """
        indexed = {}
        for entry in self:
            label = key(entry)
            if label in indexed:
                previous    = indexed[label]
                differences = list(parameter_differences(previous.metadata, entry.metadata))
                raise ValueError(f"duplicate index key {label!r}: {previous.path}, {entry.path}; differing fields: {differences}")
            indexed[label] = entry
        return LazyRunMapping(indexed, verbose=verbose)

    def group_by(self, *fields: str, parameters: Optional[Callable[[RunEntry], Mapping[str, Any]]] = None) -> dict:
        """Group runs by seed, realization or other caller-selected scalar fields.

        Parameters
        ----------
        fields : str
            One or more field names in the parameter view. No field roles are guessed.
        parameters : callable, optional
            Entry-to-fields mapping. Defaults to metadata. Use this to expose nested
            seeds or disorder IDs without imposing a project-specific schema.

        Returns
        -------
        dict
            Scalar keys for one field, tuple keys for several fields, mapped to
            catalogs sharing entries and caches. Uses ordinary Python key equality.
            Missing fields raise KeyError. No payloads are loaded.
        """
        if not fields:
            raise ValueError("group_by needs at least one field")
        groups = {}
        for entry in self:
            values = parameters(entry) if parameters else entry.metadata
            missing = [name for name in fields if name not in values]
            if missing:
                raise KeyError(f"{entry.path}: missing grouping fields {missing}")
            key = tuple(values[name] for name in fields)
            key = key[0] if len(fields) == 1 else key
            try:
                hash(key)
            except TypeError as error:
                raise TypeError(f"grouping fields must be hashable: {fields}") from error
            groups.setdefault(key, []).append(entry)
        return {key: type(self)(entries) for key, entries in groups.items()}

    def pair_with(self, other: RunCatalog, *, parameters: Callable[[RunEntry], Mapping[str, Any]]) -> tuple:
        """Pair matching random realizations across two selections without loading data.

        Parameters
        ----------
        other : RunCatalog
            The second selection, for example another method or device.
        parameters : callable
            Exact JSON matching fields, including the system and the seed roles or
            realization IDs that define the comparison. Exclude the compared setting.

        Returns
        -------
        tuple of (RunEntry, RunEntry)
            Pairs in this catalog's order. Caches are shared. Duplicate matching
            keys or unequal key sets raise ValueError instead of dropping runs.
        """
        def indexed(catalog):
            entries = {}
            for entry in catalog:
                key = parameter_key(parameters(entry))
                if key in entries:
                    raise ValueError(f"ambiguous pairing key {key}: {entries[key].path}, {entry.path}")
                entries[key] = entry
            return entries

        left, right = indexed(self), indexed(other)
        if left.keys() != right.keys():
            raise ValueError(f"unmatched pairing keys: left-only={sorted(left.keys() - right.keys())}, right-only={sorted(right.keys() - left.keys())}")
        return tuple((entry, right[key]) for key, entry in left.items())

    def aggregate(self, values: Callable[[RunEntry], Any], *, statistic: Callable = np.mean,
                  transform: Optional[Callable[[Any], Any]] = None, axis=0, verbose: bool = False, **kwargs) -> Any:
        """Apply a NumPy statistic to explicitly selected per-run values.

        Parameters
        ----------
        values : callable
            Receives an entry and extracts the required array or HDF5 slice. Select
            small windows here to avoid reading full datasets.
        statistic : callable
            NumPy-style statistic accepting axis and keyword arguments. Defaults to mean.
        transform : callable, optional
            Applied to each extracted array before stacking, for example np.square.
        axis : int, tuple of int or None
            Reduction axes after a new run axis is prepended. Default 0 reduces runs.
        **kwargs
            Passed to the statistic, such as ddof for std or q for quantile.

        verbose : bool
            Emit INFO timing and memory reports. Defaults to quiet.

        Returns
        -------
        object
            Statistic result. Equal-shaped arrays are stacked in catalog order.
            Empty selections or incompatible shapes raise ValueError. No interpolation,
            seed deduplication or normalization is performed. Selected arrays are held
            in memory; use the extraction callback to reduce them first when needed.
            Extracted values may come from caches; transforms should not mutate them.
        """
        with measured("catalog.aggregate", verbose) as report:
            if not self.entries:
                raise ValueError("cannot aggregate an empty catalog")
            arrays = []
            for entry in self:
                array = values(entry)
                arrays.append(np.asarray(transform(array) if transform is not None else array))
            shape = arrays[0].shape
            if any(array.shape != shape for array in arrays):
                raise ValueError(f"aggregation requires equal shapes, found {[array.shape for array in arrays]}")
            
            # Stack arrays along a new run axis, then apply the statistic. Report workspace memory usage.
            stacked                     = np.stack(arrays, axis=0)
            report["workspace_bytes"]   = stacked.nbytes
            result                      = statistic(stacked, axis=axis, **kwargs)
            report.update(result=result, items=len(arrays))
            return result
    def best(self, *, parameters: Callable[[RunEntry], Mapping[str, Any]], rank: Callable[[RunEntry], Any]) -> RunCatalog:
        """Keep the highest-ranked copy for each exact parameter identity.

        Parameters
        ----------
        parameters : callable
            Complete JSON identity, including seeds when they distinguish runs.
        rank : callable
            Comparable score or tuple. Larger wins. Equal scores keep the first.

        Returns
        -------
        RunCatalog
            Winners in first-identity appearance order. No payloads are loaded.
        """
        winners = {}
        for entry in self:
            identity    = parameter_key(parameters(entry))
            score       = rank(entry)
            if identity not in winners or score > winners[identity][0]:
                winners[identity] = (score, entry)
        return type(self)(entry for _, entry in winners.values())

    
    # -------------------------------------------
    
    
    def differences(self, *, parameters: Optional[Callable[[RunEntry], Mapping[str, Any]]] = None) -> dict:
        """Compare selected entries' metadata or a caller-defined parameter view.

        Parameters
        ----------
        parameters : callable, optional
            Parameter view. Defaults to complete metadata.

        Returns
        -------
        dict
            parameter_differences output indexed by this catalog's entry order.
        """
        return parameter_differences(*(parameters(entry) if parameters else entry.metadata for entry in self))


    # -------------------------------------------


    def to_frame(self, fields: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None):
        """Build a pandas table without loading payloads.

        Parameters
        ----------
        fields : callable, optional
            Convert metadata to table columns. Defaults to unchanged metadata.
            Source fields path, root, key, modified and entry are reserved.

        Returns
        -------
        pandas.DataFrame
            Metadata plus source fields and lazy entry objects. Requires pandas.
        """
        import pandas as pd
        rows = []
        for entry in self:
            values = dict(fields(entry.metadata) if fields else entry.metadata)
            reserved = {"path", "root", "key", "modified", "entry"}.intersection(values)
            if reserved:
                raise ValueError(f"metadata fields collide with source columns: {sorted(reserved)}")
            
            # Add a row with metadata and reserved source fields, including the RunEntry object itself for lazy access.
            rows.append({**values, "path": entry.path, "root": entry.root, "key": entry.key, "modified": entry.modified, "entry": entry})
        return pd.DataFrame(rows)


# -------------------------------------------
#! EOF
# -------------------------------------------
