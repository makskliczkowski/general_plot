# Run data and HDF5

`general_plot.data` discovers runs from metadata, selects copies and loads arrays on demand. The project supplies its metadata schema, identity and ranking rules. The library handles file discovery, detached reads and atomic writes.

## Browse arrays before loading them

```python
import numpy as np
from general_plot.data import HDF5Manager, LazyHDF5

HDF5Manager.write("runs/example.h5", {"history/state": np.ones((8, 4, 3), dtype=complex), "history/energy": np.arange(8.0), "labels": ["up", "down"]}, metadata={"parameters": {"seed": 11}, "units": {"energy": "J"}})

print(HDF5Manager.describe("runs/example.h5", "history/*"))
data = LazyHDF5("runs/example.h5", pattern="history/*")
print(list(data))                  # dataset names, no array reads
print(data.describe())             # shapes, dtypes, compression and text encoding
energy = data["history/energy"]    # loads and caches this dataset
```

Filter headers with normal Python when selection depends on shape rather than name.

```python
headers     = HDF5Manager.describe("runs/example.h5")
matrices    = [name for name, info in headers.items() if len(info["shape"]) == 2]
values      = HDF5Manager.read("runs/example.h5", matrices)
```

Dataset paths are relative to the file root. `pattern` is a glob applied to complete paths. `keys`, `describe`, membership and iteration read headers only. `LazyHDF5` is a mapping, so `keys()`, `get()`, `values()` and `items()` use normal Python mapping behavior. Iterating values or items loads the corresponding arrays. Index access caches complete datasets. `clear_cache()` releases those array references. Neither the manager nor the lazy mapping retains open HDF5 handles. A lazy mapping is a snapshot of dataset headers. Recreate it after replacing the source file.

## Select elements and apply NumPy operations

```python
magnitude       = data.read("history/state", selection=np.s_[2:6, ::2, :], transform=np.abs)
mean_magnitude  = HDF5Manager.read_dataset("runs/example.h5", "history/state", selection=np.s_[2:6, :, :], transform=lambda values: np.abs(values).mean(axis=-1))
positive        = data.read("history/energy", transform=lambda values: values[values > 0])
```

`selection` uses h5py indexing and reads only the selected elements. Standard slices and index tuples work. h5py's constraints still apply to fancy indexing, including increasing index order and positive slice strides. For arbitrary NumPy indexing, load the array and index it with NumPy. `transform` receives the detached selection and can return an array, scalar or other computed value. It does not write to the source. Value filtering runs after the selected data enters memory. It is not an HDF5 query engine.

`LazyHDF5.read` bypasses the full-dataset cache. Its result is consistent whether or not index access has already cached that dataset. `HDF5Manager.read(path, keys, selection=..., transform=...)` applies the same selection and transform to every requested dataset in one file open. Supply datasets with compatible shapes. An empty key list reads no arrays. Missing dataset paths raise `KeyError`.

## Supported data

| Input | Storage and returned values |
| --- | --- |
| Real, complex, integer and boolean scalars or arrays | Native HDF5 types, preserving dtype and shape |
| Matrices, tensors and empty arrays | The same array interface, retaining the original shape |
| Structured NumPy arrays with HDF5-compatible fields | Compound datasets |
| Python strings, Unicode arrays and object arrays of strings | UTF-8 text, returned as Python strings or arrays of strings |
| Byte strings and fixed-length byte arrays | Bytes remain bytes |
| Arbitrary Python objects or mixed object arrays | Rejected with an error; no pickle serialization |

JSON provenance is a separate `metadata` attribute. Missing writer source information is added as described below. It must contain JSON-compatible values. Complex values belong in datasets. HDF5 string limitations, such as embedded null characters, still apply. Binary blobs can be represented as `uint8` arrays.

`HDF5Manager.write` replaces a complete file atomically after writing a temporary sibling. Failed writes preserve the previous destination. Dataset paths must be normalized and cannot contain another dataset path. Parent directories are created. Reads never create directories or modify inputs.

## Discover and filter runs

```python
from general_plot.data import RunCatalog

runs = RunCatalog.scan(
    ["results/local", "results/cluster"], pattern="**/*.json",
    path_for=lambda path: path.with_suffix(".h5"),
    loader=LazyHDF5,
)
selected = runs.select(
    {"seed": [11, 12], "method": "solver_a"},
    parameters=lambda entry: entry.metadata["parameters"],
    predicate=lambda entry: entry.metadata["progress"]["steps"] >= 100,
)
```

This example's sidecars have `parameters` and `progress` groups. JSON is the default metadata reader. Supply `read_metadata(path)` for another format. `path_for` maps metadata files to payloads, and `loader` controls what a selected entry loads. `entry.load()` caches that payload until `entry.clear_cache()`. With `LazyHDF5` as the loader, the payload itself loads datasets on demand.

`select` reuses `filter_results` for scalar and membership filters. `parameters` supplies the fields used by those filters. `predicate` accepts an additional arbitrary rule. Numeric filter tolerance is separate from exact identity matching. Selection shares entries and caches with the original catalog.

`discover_files(roots, pattern, predicate=...)` provides the underlying directory traversal. It yields absolute `(root, file)` pairs in root order and sorted file order. Overlapping roots discover each resolved physical file once. Missing roots raise errors. List only the roots available in the current environment. Use `pathlib.Path` for other directory operations.

## Comparison rules

Dictionary scalars are equality filters. Callables receive a field value and can use normal Python comparisons.

```python
selected = runs.select(
    {"seed": lambda x: x == 11,
     "coupling": lambda x: 0.2 <= x < 0.5,
     "status": lambda x: x != "failed"},
    parameters=lambda entry: entry.metadata["parameters"],
)
```

Missing fields do not match and their callable is not invoked. An explicit None value is passed through to the callable; handle it there when needed. Predicate exceptions propagate. Callable comparisons are exact Python operations, independent of the tolerance used by scalar equality filters. Compound rules across fields can use the catalog's entry-level `predicate` argument.

## Between rules

```python
from general_plot.data import Between

selected = runs.select(
    {"seed": Between(4, 7), "coupling": Between(0.2, 0.5, closed="left")},
    parameters=lambda entry: entry.metadata["parameters"],
)
inside = Between(4, 7)(values)       # NumPy mask for values, not index positions
mean_square = np.mean(values[inside]**2)
```

`closed` is `both` (the default), `left`, `right` or `neither`. `None` leaves a bound open. None and NaN values do not match. Comparisons use exact ordering, independent of equality-filter tolerance. Lists and tuples still mean membership. Index cuts such as `np.s_[:, 4:7]` use Python's upper-exclusive indexing and are separate from value intervals.

## Unique indexes and legitimate variants

```python
by_method = selected.index_by(lambda entry: entry.metadata["parameters"]["method"])
print(list(by_method))                 # keys only, no data reads
print(by_method.entries["solver_a"].metadata)
result = by_method["solver_a"]          # loads and caches this payload
```

`index_by` requires one run per key and validates that requirement before any I/O. A collision reports both paths and the differing metadata fields. Real copies can first be resolved with `best`. Distinct controls are legitimate variants. Group by their configuration before building a unique method index, or keep the groups as collections. Do not silently overwrite a run or choose the best-performing control to resolve an indexing conflict.

A lazy index exposes its underlying `RunEntry` objects through `entries`. Keys, length and membership do not load data. Iterating `values()` or `items()` does load their values, as with `LazyHDF5`. Use the entry mapping for metadata checks and lightweight scoring. `index_by(..., verbose=True)` reports individual payload accesses.

## Group and pair random realizations

```python
def fields(entry):
    return entry.metadata["parameters"]

by_seed = runs.group_by("seed", parameters=fields)
by_realization = runs.group_by("seed", "disorder_id", parameters=fields)
by_method = runs.group_by("method", parameters=fields)
pairs = by_method["solver_a"].pair_with(
    by_method["solver_b"],
    parameters=lambda entry: {name: fields(entry)[name] for name in ("system", "seed", "disorder_id")},
)
```

Grouping returns catalogs that share entries and caches. One field gives scalar dictionary keys; multiple fields give tuple keys. It uses normal Python key equality. Missing fields raise errors. Use distinct names for initialization, sampler and disorder seeds. A seed value alone does not define a physical realization or establish independence.

Pairing uses the same exact JSON identity comparisons as copy selection. Include all fixed physical parameters and realization fields. Exclude the quantity being compared, such as method or device. Duplicate matching keys and unequal key sets raise errors. No run is silently dropped or selected from an ambiguous pair.

## Statistics on selected windows

For a dataset `y[realization, state]`, a middle-spectrum index window can be selected before reading. This assumes the caller knows the ordering of the state axis.

```python
per_realization = data.read(
    "y", selection=np.s_[:, 4:7],
    transform=lambda y: np.mean(y**2, axis=1),
)
```

Here the mean is over the three selected states, returning one mean-square value per realization. The same form handles tensors, for example `np.s_[:, 6:10, :]`. Complex `y**2` is an algebraic square. Use `np.abs(y)**2` when the squared magnitude is intended.

For a different window in each row, NumPy can select within a coarse slice without reading the full state axis.

```python
indices = np.array([[4, 5, 6], [6, 7, 8]])
per_realization = data.read(
    "y", selection=np.s_[:, 4:9],
    transform=lambda y: np.take_along_axis(y, indices - 4, axis=1)**2,
)
```

`aggregate` applies a NumPy statistic across selected runs. It stacks equally shaped extracted arrays along a new leading run axis. The default reduction axis is 0.

```python
mean_square = selected.aggregate(
    lambda entry: entry.load().read("y", selection=np.s_[:, 4:7]),
    transform=np.square, statistic=np.mean, axis=None,
)

# Standard deviation across seeds of each realization's mean-square window value.
std = selected.aggregate(
    lambda entry: entry.load().read(
        "y", selection=np.s_[:, 4:7],
        transform=lambda y: np.mean(y**2, axis=1),
    ),
    statistic=np.std, axis=0, ddof=1,
)
```

The first example reduces runs, realizations and selected states into one mean. `axis=0` would keep the realization and state axes. Other NumPy statistics, such as median or quantile, use the same interface. Additional keywords, such as `ddof` or `q`, are passed to the statistic. Group or deduplicate the catalog before aggregating when required. The library does not infer independent realizations or statistical error bars.

Only extracted arrays are collected, but they and their stack are held in memory. Read small cuts or compute per-run summaries in the extraction callback for large Hilbert spaces. Shapes must match and the caller must align coordinates. No interpolation is performed. Extraction can return cached values, so use transforms that do not mutate their inputs.

## Compare parameters and select copies

```python
print(selected.differences(parameters=lambda entry: entry.metadata["parameters"]))
best = selected.best(parameters=lambda entry: entry.metadata["parameters"], rank=lambda entry: (entry.metadata["progress"]["steps"], entry.modified))
```

The identity callback must include every setting and seed that distinguishes a run. Exclude outcomes, timing and source paths. `best` compares canonical JSON exactly. Mapping order is ignored. List order and numeric representation remain significant, so `1` and `1.0` are distinct identities. Approximate filters never merge run identities. Rank tuples are compared in order with larger values winning. Equal scores keep the first entry.

`parameter_differences` and `RunCatalog.differences` compare nested parameter leaves. Results map tuples of field names to `{input_index: value}`. An absent index means the field is missing, distinct from an explicit null. Lists remain whole values. No payloads are loaded for these comparisons.

## Browse a metadata table

```python
table = best.to_frame(fields=lambda metadata: metadata["parameters"])
print(table[["method", "seed", "path"]])
rows = table.query("seed == 11 and method == 'solver_a'")
for entry in rows["entry"]:
    arrays = entry.load()
    print(arrays.describe())
```

`to_frame` needs pandas. `fields` converts project metadata into table columns. `path`, `root`, `key`, `modified` and `entry` are reserved source columns. Querying or sorting this table loads no payloads. `key` records the metadata path relative to its discovery root. It is a source identifier, separate from the caller-defined run identity.

## Source devices and writer provenance

```python
from general_plot.data import source_fields, source_info

sources = runs.to_frame(fields=source_fields)
print(sources[["path", "writer_host_id", "producer_host_id", "writer_devices"]])
by_writer = runs.group_by("writer_host_id", parameters=lambda entry: source_fields(entry.metadata))
```

Savers add `metadata.provenance.writer` only if it is missing. The record contains an opaque truncated hash of the local hostname, OS, architecture and explicitly known devices. Generic HDF5 saves do not probe devices, so their device list is empty unless supplied. No accelerator backend is initialized for provenance. Raw hostnames, usernames, IP/MAC addresses, hardware serials, paths and environment values are not automatically stored. The host ID distinguishes recorded hostnames, not persistent physical hardware.

A writer describes where an artifact was saved. It does not establish where imported arrays were computed. Supply a separate `producer` record when it is known, for example `{"label": "cluster_a", **source_info(devices=known_devices)}`. Existing writer and producer records are preserved. Unknown producer fields in older files remain None; paths are not used to guess origins. Optional source labels are exposed as `writer_label` and `producer_label`.

`source_fields` exposes provenance for browsing and grouping. Keep it out of scientific run identity. Different writer or device records do not create new random realizations. Device-dependent numerical settings such as precision still belong in the scientific configuration and must remain part of matching parameters.

## Save reproducible figure data

```python
from general_plot.data import write_figure_data, read_figure_data, read_figure_metadata

write_figure_data("figures/convergence.h5", {"panel_a/solver_a": (steps, error)}, metadata={"selection": {"seed": 11}, "sources": [str(entry.path) for entry in best]})
series      = read_figure_data("figures/convergence.h5")
provenance  = read_figure_metadata("figures/convergence.h5")
```

Figure-data helpers add a named `(x, y)` series contract to the same HDF5 manager. Arrays in a pair need matching first dimensions. Each group contains `x` and `y` datasets. Older files with this layout remain readable. Selections, units, dates and repository commits are supplied explicitly by the caller. Missing writer provenance is added automatically. `Plotter.save_fig` handles rendered images.

Install `general-plot[data]` for the optional h5py and pandas dependencies. The discovery and selection logic uses the standard library. The discovery and lazy-cache boundaries draw on the [general_python common modules](https://github.com/makskliczkowski/general_python/tree/main/common). Identity and ranking remain explicit because a shared library cannot infer a project's run semantics. The APIs do not coordinate concurrent cache loads or automatically aggregate trajectories with different grids or units.


## Verbose timing and memory

```python
import logging
logging.basicConfig(level=logging.INFO)

data = LazyHDF5("run.h5", verbose=True)
part = data.read("y", selection=np.s_[:, 4:7], transform=np.square)
run = entry.load(verbose=True)
mean = selected.aggregate(extract, statistic=np.mean, verbose=True)
```

Verbose mode emits INFO records through the standard `general_plot.data` logger. The application configures its handlers and level. Quiet mode emits no reports. Reports cover HDF5 reads and writes, run loads, metadata scans and aggregation. They include elapsed wall seconds, logical bytes of returned NumPy buffers, process peak RSS, item count and cache-hit status. Aggregation also reports the bytes of its temporary stack. Exceptions are reported and propagated unchanged. No measurements are inserted into saved metadata or scientific identity.

`array_bytes` counts NumPy buffers found in ordinary dictionaries, lists, tuples and dataclass storage, counting a repeated array object once. Views can share allocations, and object-dtype buffers omit the Python objects they point to. This is returned data volume, not an allocator measurement. Unknown or lazy payloads report `n/a`; measuring them never loads arrays or evaluates properties.

`peak_rss_bytes` is the process-wide resident-memory high-water mark since process start. It is available on Linux and macOS, with OS units converted to bytes. Other platforms report `n/a`. It is neither current RSS, GPU memory nor the incremental memory cost of one operation. Timing includes the selected read and requested transform or statistic, but resource-report formatting is excluded. Cached reads are explicitly marked.
