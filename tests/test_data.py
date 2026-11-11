"""Run selection and figure-data contracts with independent temporary inputs."""
import json
from pathlib import Path

import numpy as np
import pytest

from general_plot.data import RunCatalog, parameter_differences, parameter_key
from general_plot.data import read_figure_data, read_figure_metadata, write_figure_data


def test_catalog_selection_is_lazy_and_keeps_parameter_distinctions(tmp_path):
    calls = []
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    inputs = [(first / "a.json", {"parameters": {"seed": 4, "rate": 0.1}, "steps": 3}),
              (second / "b.json", {"parameters": {"seed": 4, "rate": 0.1}, "steps": 8}),
              (second / "c.json", {"parameters": {"seed": 4, "rate": 0.2}, "steps": 5}),
              (second / "d.json", {"parameters": {"seed": 5, "rate": 0.1}, "steps": 6})]
    for path, metadata in inputs:
        path.write_text(json.dumps(metadata))

    def load(path):
        calls.append(path)
        return {"detached": path.stem}

    catalog = RunCatalog.scan([first, second, tmp_path], pattern="**/*.json", loader=load,
                              path_for=lambda path: path.with_suffix(".h5"))
    assert len(catalog) == 4
    assert calls == []
    winners = catalog.best(parameters=lambda entry: entry.metadata["parameters"],
                           rank=lambda entry: (entry.metadata["steps"], entry.modified))
    assert [entry.path.stem for entry in winners] == ["b", "c", "d"]
    selected = winners.select({"seed": 4}, parameters=lambda entry: entry.metadata["parameters"],
                              predicate=lambda entry: entry.metadata["steps"] > 6)
    assert calls == []
    entry = next(iter(selected))
    assert entry.load() is entry.load()
    assert calls == [second / "b.h5"]
    assert next(iter(winners)).loaded
    entry.clear_cache()
    assert not entry.loaded
    entry.load()
    assert len(calls) == 2
    assert winners.differences(parameters=lambda entry: entry.metadata["parameters"]) == {
        ("rate",): {0: 0.1, 1: 0.2, 2: 0.1}, ("seed",): {0: 4, 1: 4, 2: 5}}


def test_exact_identity_and_missing_values():
    assert parameter_key({"a": 1, "b": [2]}) == parameter_key({"b": [2], "a": 1})
    assert parameter_key({"a": 1}) != parameter_key({"a": 1.0})
    assert parameter_differences({"x": {"y": None}}, {"x": {}}) == {
        ("x", "y"): {0: None}, ("x",): {1: {}}}


def test_missing_root_and_metadata_errors_are_visible(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunCatalog.scan(tmp_path / "missing", pattern="*.json", loader=dict, path_for=lambda path: path)
    (tmp_path / "bad.json").write_text("[]")
    with pytest.raises(TypeError, match="mapping"):
        RunCatalog.scan(tmp_path, pattern="*.json", loader=dict, path_for=lambda path: path)


def test_rank_ties_use_caller_policy(tmp_path):
    import os
    for index, name in enumerate(("a", "b")):
        path = tmp_path / f"{name}.json"
        path.write_text('{"seed": 1}')
        os.utime(path, (100 + index, 100 + index))
    catalog = RunCatalog.scan(tmp_path, pattern="*.json", loader=dict, path_for=lambda path: path)
    assert next(iter(catalog.best(parameters=lambda entry: entry.metadata, rank=lambda entry: entry.modified))).path.stem == "b"
    assert next(iter(catalog.best(parameters=lambda entry: entry.metadata, rank=lambda entry: 0))).path.stem == "a"


def test_figure_roundtrip_and_failed_write_preserves_file(tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "figure.h5"
    x = np.arange(5)
    write_figure_data(path, {"panel/line": (x, x**2)}, metadata={"query": {"seed": [1, 2]}})
    metadata = read_figure_metadata(path)
    assert metadata["query"] == {"seed": [1, 2]}
    assert "writer" in metadata["provenance"]
    assert "producer" not in metadata["provenance"]
    np.testing.assert_array_equal(read_figure_data(path)["panel/line"][1], x**2)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="dimensions"):
        write_figure_data(path, {"bad": (x, x[:2])})
    with pytest.raises(ValueError, match="overlap"):
        write_figure_data(path, {"a": (x, x), "a-b": (x, x), "a/b": (x, x)})
    with pytest.raises(TypeError):
        write_figure_data(path, {"bad": (np.array([object()]), np.array([object()]))})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_load_retries_and_none_is_cached(tmp_path):
    from general_plot.data import RunEntry
    calls = []
    def load(path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError('temporary read failure')
        return None
    entry = RunEntry(tmp_path / 'result', tmp_path, 'result.json', {}, 0, load)
    with pytest.raises(OSError):
        entry.load()
    assert not entry.loaded
    assert entry.load() is None
    assert entry.load() is None
    assert entry.loaded
    assert len(calls) == 2


def test_frame_conversion_preserves_lazy_entry_and_rejects_collisions(tmp_path):
    pytest.importorskip('pandas')
    path = tmp_path / 'result.json'
    path.write_text('{"parameters":{"method":"alpha"}}')
    catalog = RunCatalog.scan(tmp_path, pattern='*.json', loader=dict, path_for=lambda path: path)
    frame = catalog.to_frame(lambda metadata: metadata['parameters'])
    assert frame['method'].iloc[0] == 'alpha'
    assert not frame['entry'].iloc[0].loaded
    with pytest.raises(ValueError, match='collide'):
        catalog.to_frame(lambda metadata: {'path': 'shadowed'})


def test_discovery_filters_overlapping_roots_without_opening_files(tmp_path):
    from general_plot.data import discover_files
    nested = tmp_path / 'nested'
    nested.mkdir()
    keep = nested / 'keep.bin'
    keep.write_bytes(b'not metadata')
    (nested / 'skip.bin').write_bytes(b'ignored')
    found = list(discover_files([tmp_path, nested], '**/*.bin', predicate=lambda path: path.stem == 'keep'))
    assert found == [(tmp_path.resolve(), keep.resolve())]
    assert list(discover_files(tmp_path, '*.missing')) == []


def test_hdf5_selected_reads_and_dataset_cache(tmp_path, monkeypatch):
    pytest.importorskip('h5py')
    from general_plot.data import HDF5Manager, LazyHDF5
    path = tmp_path / 'arrays.h5'
    HDF5Manager.write(path, {'history/x': np.arange(4), 'history/y': np.ones(4), 'scalar': 3}, metadata={'unit': 's'})
    assert HDF5Manager.keys(path) == ('history/x', 'history/y', 'scalar')
    metadata = HDF5Manager.metadata(path)
    assert metadata['unit'] == 's'
    assert 'host_id' in metadata['provenance']['writer']
    assert HDF5Manager.read(path, []) == {}
    assert list(HDF5Manager.read(path, ['scalar'])) == ['scalar']
    with pytest.raises(KeyError):
        HDF5Manager.read(path, ['absent'])
    with pytest.raises(TypeError, match='iterable'):
        HDF5Manager.read(path, 'scalar')
    reads = []
    read = HDF5Manager.read_dataset
    def counted(path, key, **kwargs):
        reads.append(key)
        return read(path, key, **kwargs)
    monkeypatch.setattr(HDF5Manager, 'read_dataset', counted)
    lazy = LazyHDF5(path)
    assert len(lazy) == 3
    assert 'history/y' in lazy
    assert list(lazy) == ['history/x', 'history/y', 'scalar']
    assert reads == []
    assert lazy['history/x'] is lazy['history/x']
    assert reads == ['history/x']
    lazy.clear_cache()
    np.testing.assert_array_equal(lazy['history/x'], np.arange(4))
    assert len(reads) == 2
    with pytest.raises(KeyError):
        lazy['absent']


def test_hdf5_failed_write_keeps_previous_file(tmp_path):
    pytest.importorskip('h5py')
    from general_plot.data import HDF5Manager
    path = tmp_path / 'arrays.h5'
    HDF5Manager.write(path, {'value': 8})
    original = path.read_bytes()
    with pytest.raises(ValueError, match='overlap'):
        HDF5Manager.write(path, {'a': 1, 'a/b': 2})
    with pytest.raises(ValueError, match='invalid'):
        HDF5Manager.write(path, {'../bad': 1})
    with pytest.raises(TypeError):
        HDF5Manager.write(path, {'bad': object()})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('values', [
    1.25, 2 + 3j, True, np.int64(7),
    np.array([1., np.nan, np.inf], dtype=np.float32),
    np.array([1+2j, 3-4j], dtype=np.complex64),
    np.arange(12, dtype=np.int16).reshape(3, 4),
    (np.arange(24).reshape(2, 3, 4) + 1j).astype(np.complex128),
    np.array([True, False]), np.empty((0, 3), dtype=np.float64),
    np.array([(1., 2+3j)], dtype=[('real', 'f8'), ('complex', 'c16')]),
])
def test_hdf5_numeric_shapes_and_dtypes(tmp_path, values):
    pytest.importorskip('h5py')
    from general_plot.data import HDF5Manager
    path = tmp_path/'numeric.h5'
    HDF5Manager.write(path, {'data': values})
    loaded = HDF5Manager.read_dataset(path, 'data')
    assert np.shape(loaded) == np.shape(values)
    assert np.asarray(loaded).dtype == np.asarray(values).dtype
    np.testing.assert_array_equal(loaded, values)


@pytest.mark.parametrize('values', [
    'spin ↑', '', np.array(['α', 'longer text', '🙂']),
    np.array([['up', 'down'], ['β', 'γ']], dtype=object),
    np.array([], dtype='U1'), b'raw', np.array([b'a', b'bb'], dtype='S2'),
    np.array([b'a', b'bb'], dtype=object),
])
def test_hdf5_unicode_and_byte_text(tmp_path, values):
    pytest.importorskip('h5py')
    from general_plot.data import HDF5Manager
    path = tmp_path/'text.h5'
    HDF5Manager.write(path, {'text': values})
    loaded = HDF5Manager.read_dataset(path, 'text')
    np.testing.assert_array_equal(loaded, values)
    assert np.shape(loaded) == np.shape(values)
    if isinstance(values, str):
        assert isinstance(loaded, str)


def test_browse_filter_slice_and_numpy_transform_do_not_edit_source(tmp_path, monkeypatch):
    pytest.importorskip('h5py')
    from general_plot.data import HDF5Manager, LazyHDF5
    path = tmp_path/'tensor.h5'
    tensor = (np.arange(60).reshape(3,4,5) + 2j).astype('complex128')
    HDF5Manager.write(path, {'history/state': tensor, 'history/energy': np.arange(3.), 'labels': ['up','down']})
    assert HDF5Manager.keys(path, 'history/*') == ('history/energy', 'history/state')
    view = LazyHDF5(path, pattern='history/state')
    assert list(view) == ['history/state']
    assert view.describe()['history/state']['shape'] == (3,4,5)
    assert 'cached=0' in repr(view)
    assert HDF5Manager.describe(path, 'labels')['labels']['encoding'] == 'utf-8'
    selected = np.s_[1:, ::2, 2:]
    result = view.read('history/state', selection=selected, transform=lambda values: np.abs(values).mean(axis=-1))
    np.testing.assert_allclose(result, np.abs(tensor[selected]).mean(axis=-1))
    assert 'cached=0' in repr(view)
    np.testing.assert_array_equal(HDF5Manager.read_dataset(path,'history/state'), tensor)
    np.testing.assert_array_equal(HDF5Manager.read(path,['history/state'],selection=selected)['history/state'],tensor[selected])
    with pytest.raises(KeyError):
        view.read('history/energy')


def test_group_and_pair_realizations_without_loading_payloads(tmp_path):
    from general_plot.data import RunEntry
    def forbidden(path):
        raise AssertionError('metadata operation loaded a payload')
    entries = [RunEntry(tmp_path/f'{method}_{seed}',tmp_path,f'{method}_{seed}.json',
                        {'parameters':{'method':method,'seed':seed,'disorder':3}},0,forbidden)
               for method in ('a','b') for seed in (11,12)]
    runs = RunCatalog(entries)
    def fields(entry):
        return entry.metadata['parameters']
    assert {key:len(group) for key,group in runs.group_by('seed',parameters=fields).items()} == {11:2,12:2}
    assert list(runs.group_by('seed','disorder',parameters=fields)) == [(11,3),(12,3)]
    methods = runs.group_by('method',parameters=fields)
    def match(entry):
        return {key:value for key,value in fields(entry).items() if key != 'method'}
    pairs = methods['a'].pair_with(methods['b'],parameters=match)
    assert [(fields(left)['seed'],fields(right)['seed']) for left,right in pairs] == [(11,11),(12,12)]
    with pytest.raises(ValueError,match='ambiguous'):
        runs.pair_with(methods['a'],parameters=match)
    with pytest.raises(ValueError,match='unmatched'):
        methods['a'].pair_with(RunCatalog([entries[-1]]),parameters=match)
    with pytest.raises(KeyError,match='missing grouping'):
        runs.group_by('unknown')
    with pytest.raises(ValueError,match='at least one'):
        runs.group_by()
    assert all(not entry.loaded for entry in runs)


def test_catalog_statistics_over_selected_spectral_windows(tmp_path):
    pytest.importorskip('h5py')
    from general_plot.data import HDF5Manager, LazyHDF5, RunEntry
    arrays = [np.arange(24.).reshape(2,12), np.arange(24.).reshape(2,12)+2]
    entries = []
    for seed,array in enumerate(arrays):
        path = tmp_path/f'{seed}.h5'
        HDF5Manager.write(path,{'y':array})
        entries.append(RunEntry(path,tmp_path,str(seed),{'seed':seed},0,LazyHDF5))
    runs = RunCatalog(entries)
    def extract(entry):
        return entry.load().read('y',selection=np.s_[:,4:7])
    expected = np.stack([array[:,4:7]**2 for array in arrays])
    np.testing.assert_allclose(runs.aggregate(extract,transform=np.square),expected.mean(axis=0))
    np.testing.assert_allclose(runs.aggregate(extract,transform=np.square,statistic=np.std,ddof=1),expected.std(axis=0,ddof=1))
    np.testing.assert_allclose(runs.aggregate(extract,transform=np.square,statistic=np.quantile,q=0.5),np.quantile(expected,0.5,axis=0))
    assert runs.aggregate(extract,transform=np.square,axis=None) == expected.mean()
    indices = np.array([[3,4],[6,7]])
    np.testing.assert_allclose(runs.aggregate(lambda entry:entry.load().read('y'),
        transform=lambda y:np.take_along_axis(y,indices,axis=1)**2),
        np.stack([np.take_along_axis(array,indices,axis=1)**2 for array in arrays]).mean(axis=0))
    with pytest.raises(ValueError,match='empty'):
        RunCatalog([]).aggregate(extract)
    with pytest.raises(ValueError,match='equal shapes'):
        runs.aggregate(lambda entry:np.ones((entry.metadata['seed']+1,)))


def test_safe_source_metadata_preserves_producer_and_does_not_mutate_inputs(tmp_path,monkeypatch):
    import hashlib
    from general_plot.data import HDF5Manager, source_info, source_fields, with_provenance
    monkeypatch.setattr('general_plot.data.provenance.platform.node',lambda:'private-hostname')
    info = source_info()
    assert info['host_id'] == hashlib.sha256(b'private-hostname').hexdigest()[:16]
    assert 'private-hostname' not in str(info)
    assert info['devices'] == []
    metadata = {'provenance':{'producer':{'host_id':'origin','devices':[{'platform':'gpu','kind':'example'}]}}}
    enriched = with_provenance(metadata)
    assert 'writer' not in metadata['provenance']
    assert enriched['provenance']['producer'] == metadata['provenance']['producer']
    path=tmp_path/'source.h5'
    HDF5Manager.write(path,{'x':[1,2]},metadata=metadata)
    loaded=HDF5Manager.metadata(path)
    assert loaded['provenance']==enriched['provenance']
    assert source_fields(loaded)['producer_host_id']=='origin'
    assert source_fields({})['writer_host_id'] is None
    explicit={'provenance':{'writer':{'host_id':'known'}}}
    assert with_provenance(explicit)==explicit


@pytest.mark.parametrize(('closed','expected'),[('both',[False,True,True,True,False,False]), ('left',[False,True,True,False,False,False]), ('right',[False,False,True,True,False,False]), ('neither',[False,False,True,False,False,False])])
def test_between_metadata_and_array_masks(closed,expected,tmp_path):
    from general_plot.data import Between, RunEntry
    rule        = Between(4,7,closed=closed)
    values      = np.array([3.,4.,5.,7.,8.,np.nan])
    np.testing.assert_array_equal(rule(values),expected)
    runs        = RunCatalog(RunEntry(tmp_path/str(i),tmp_path,str(i),{'seed':value},0,dict) for i,value in enumerate(values))
    selected    = runs.select({'seed':rule},tol=1.)
    assert [entry.metadata['seed'] for entry in selected] == values[np.array(expected)].tolist()
    assert not rule(None)
    assert Between(upper=4)(4)
    assert Between(lower=4)(5)
    assert not Between()(np.nan)
    assert not Between(4,7)(7.00001)
    # Tuples retain their membership meaning, rather than becoming intervals.
    assert [entry.metadata['seed'] for entry in runs.select({'seed':(4,7)})] == [4.,7.]


def test_invalid_between_bounds():
    from general_plot.data import Between
    with pytest.raises(ValueError,match='lower'):
        Between(7,4)
    with pytest.raises(ValueError,match='closed'):
        Between(4,7,closed='unknown')
    with pytest.raises(ValueError,match='NaN'):
        Between(np.nan,7)


def test_python_comparison_field_filters_are_shared_with_plotter(tmp_path):
    from general_plot import Plotter
    from general_plot.data import RunEntry
    metadata    = [{'seed':11,'coupling':0.2},{'seed':11,'coupling':0.5},{'seed':12,'coupling':0.3},{'seed':11}]
    rules       = {'seed':lambda x:x==11,'coupling':lambda x:0.2<=x<0.5}
    runs        = RunCatalog(RunEntry(tmp_path/str(i),tmp_path,str(i),item,0,dict) for i,item in enumerate(metadata))
    assert [entry.metadata for entry in runs.select(rules)] == [metadata[0]]
    assert Plotter.filter_results(metadata,rules) == [metadata[0]]
    calls=[]
    assert list(RunCatalog([list(runs)[-1]]).select({'missing':lambda value:calls.append(value)})) == []
    assert calls==[]
    def invalid(value):
        raise RuntimeError('visible predicate failure')
    with pytest.raises(RuntimeError,match='visible'):
        runs.select({'seed':invalid})


def test_verbose_resource_reports_do_not_force_reads_or_change_results(tmp_path,caplog):
    import logging
    from general_plot.data import HDF5Manager, LazyHDF5, RunEntry
    pytest.importorskip('h5py')
    path=tmp_path/'resources.h5'
    values=np.arange(12.,dtype='float64')
    with caplog.at_level(logging.INFO,logger='general_plot.data'):
        HDF5Manager.write(path,{'x':values})
        HDF5Manager.read_dataset(path,'x')
        assert not caplog.records
        np.testing.assert_array_equal(HDF5Manager.read_dataset(path,'x',selection=np.s_[2:5],verbose=True),values[2:5])
        assert 'array_bytes=24' in caplog.records[-1].getMessage()
        view=LazyHDF5(path,verbose=True)
        assert 'cached=0' in repr(view)
        assert view['x'] is view['x']
        assert 'cache_hit=True' in caplog.records[-1].getMessage()
        entry=RunEntry(path,tmp_path,'r',{},0,LazyHDF5)
        payload=entry.load(verbose=True)
        assert 'cached=0' in repr(payload)
        assert 'array_bytes=n/a' in caplog.records[-1].getMessage()
        with pytest.raises(KeyError):
            HDF5Manager.read_dataset(path,'missing',verbose=True)
        assert 'status=error' in caplog.records[-1].getMessage()
        assert all('wall_s=' in record.getMessage() and 'peak_rss_bytes=' in record.getMessage() for record in caplog.records)


def test_memory_report_counts_buffers_once_and_normalizes_peak_rss(monkeypatch):
    import sys
    from types import SimpleNamespace
    from general_plot.data._diagnostics import _array_bytes, _peak_rss
    resource=pytest.importorskip("resource")
    values=np.arange(4.)
    assert _array_bytes({'a':values,'again':values})==values.nbytes
    monkeypatch.setattr(resource,'getrusage',lambda _:SimpleNamespace(ru_maxrss=1024))
    monkeypatch.setattr(sys,'platform','linux')
    assert _peak_rss()==1024*1024
    monkeypatch.setattr(sys,'platform','darwin')
    assert _peak_rss()==1024
    monkeypatch.setattr(sys,'platform','win32')
    assert _peak_rss() is None


def test_unique_lazy_index_validates_before_loading(tmp_path):
    from general_plot.data import RunEntry
    calls=[]
    def load(path):
        calls.append(path)
        return path.stem
    entries=[RunEntry(tmp_path/name,tmp_path,name,{'scheme':name,'seed':11},0,load) for name in ('a','b')]
    index=RunCatalog(entries).index_by(lambda entry:entry.metadata['scheme'])
    assert list(index)==['a','b'] and 'a' in index and len(index)==2
    assert calls==[]
    assert index['a']=='a' and index['a']=='a'
    assert len(calls)==1
    with pytest.raises(ValueError,match='duplicate index key'):
        RunCatalog(entries).index_by(lambda entry:entry.metadata['seed'])
    assert len(calls)==1
