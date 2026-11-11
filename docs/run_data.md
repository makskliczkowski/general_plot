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

JSON provenance is a separate `metadata` attribute. It must contain JSON-compatible values. Complex values belong in datasets. HDF5 string limitations, such as embedded null characters, still apply. Binary blobs can be represented as `uint8` arrays.

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

## Save reproducible figure data

```python
from general_plot.data import write_figure_data, read_figure_data, read_figure_metadata

write_figure_data("figures/convergence.h5", {"panel_a/solver_a": (steps, error)}, metadata={"selection": {"seed": 11}, "sources": [str(entry.path) for entry in best]})
series      = read_figure_data("figures/convergence.h5")
provenance  = read_figure_metadata("figures/convergence.h5")
```

Figure-data helpers add a named `(x, y)` series contract to the same HDF5 manager. Arrays in a pair need matching first dimensions. Each group contains `x` and `y` datasets. Older files with this layout remain readable. Selections, units, dates and repository commits are supplied explicitly by the caller. `Plotter.save_fig` handles rendered images.

Install `general-plot[data]` for the optional h5py and pandas dependencies. The discovery and selection logic uses the standard library. The discovery and lazy-cache boundaries draw on the [general_python common modules](https://github.com/makskliczkowski/general_python/tree/main/common). Identity and ranking remain explicit because a shared library cannot infer a project's run semantics. The APIs do not coordinate concurrent cache loads or automatically aggregate trajectories with different grids or units.
