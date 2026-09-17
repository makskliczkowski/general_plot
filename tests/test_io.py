"""Tests for data serialization helpers in io module."""

import numpy as np
import pandas as pd

from general_plot.io import MatrixPrinter, PlotterSave


def test_json_roundtrip(tmp_path):
    """dict2json / json2dict round-trips data."""
    data     = {"alpha": 1.5, "beta": [1, 2, 3]}
    PlotterSave.dict2json(str(tmp_path), "experiment", data)
    assert PlotterSave.json2dict(str(tmp_path), "experiment") == data


def test_save_and_load_columns(tmp_path):
    """Single and two-column data survive a save/load cycle."""
    PlotterSave.singleColumnData(str(tmp_path), "single.txt", [1.0, 2.0, 3.0], typ=".txt")
    assert np.allclose(np.loadtxt(tmp_path / "single.txt"), [1.0, 2.0, 3.0])

    PlotterSave.twoColumnsData(str(tmp_path), "two.npy", [1, 2], [10, 20], typ=".npy")
    assert np.load(tmp_path / "two.npy").shape == (2, 2)


def test_dataframe_and_matrix_helpers(tmp_path):
    """app_df pads columns and matrixData prepends an x column."""
    df      = pd.DataFrame({"x": [1, 2, 3, 4]})
    PlotterSave.app_df(df, "y", [10, 20], fill_value=np.nan)
    assert len(df["y"]) == 4
    assert df["y"][0] == 10 and df["y"][1] == 20
    assert np.isnan(df["y"][2])

    PlotterSave.matrixData(str(tmp_path), "mat.npy", [0.1, 0.2], np.array([[1.0, 2.0], [3.0, 4.0]]), typ=".npy")
    loaded  = np.load(tmp_path / "mat.npy")
    assert loaded.shape == (2, 3)
    assert np.allclose(loaded[:, 0], [0.1, 0.2])


def test_matrix_printer(capsys):
    """MatrixPrinter prints without error (console fallback)."""
    MatrixPrinter().print_matrix(np.array([[1, 2], [3, 4]]))
    out     = capsys.readouterr().out
    assert "1" in out or "Matrix" in out


# ---------------------------------------------
#! EOF
# ---------------------------------------------
