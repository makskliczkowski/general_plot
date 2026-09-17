"""Data output helpers, serialization, and matrix display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Optional IPython / sympy for notebook matrix display
try:
    from IPython.display    import display
    from sympy              import Matrix, init_printing
    HAS_SYMPY               = True
except ImportError:
    HAS_SYMPY               = False
    display                 = None
    Matrix                  = None
    init_printing           = None


class PlotterSave:
    """File-output helpers for saving plot-adjacent data artifacts."""

    @staticmethod
    def _prepare_path(directory: Union[str, Path], filename: str, ext: str = "") -> Path:
        """Helper to resolve path with directory creation and extension safety."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        if ext and not filename.endswith(ext):
            filename = f"{filename}{ext}"
        return dir_path / filename

    @staticmethod
    def dict2json(directory: Union[str, Path], fileName: str, data: Any) -> None:
        """Save dictionary or serializable object to a JSON file."""
        filepath = PlotterSave._prepare_path(directory, fileName, ".json")
        with open(filepath, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

    @staticmethod
    def json2dict(directory: Union[str, Path], fileName: str) -> dict:
        """Load dictionary from a JSON file."""
        filepath = PlotterSave._prepare_path(directory, fileName, ".json")
        with open(filepath, "r", encoding="utf-8") as readfile:
            return json.load(readfile)

    @staticmethod
    def json2dict_multiple(directory: Union[str, Path], keys: Sequence[str]) -> Dict[str, Any]:
        """Load multiple JSON files corresponding to a list of keys."""
        data2plot = {}
        for key in keys:
            data2plot[key] = PlotterSave.json2dict(directory, key)
        return data2plot

    @staticmethod
    def singleColumnData(
        directory: Union[str, Path],
        fileName: str,
        y: Any,
        typ: str = ".npy",
    ) -> None:
        """Store a single 1D vector as .npy or .txt."""
        to_save = np.asarray(y)
        if typ == ".npy":
            filepath = PlotterSave._prepare_path(directory, fileName, ".npy")
            np.save(filepath, to_save)
        elif typ in (".txt", ".dat"):
            filepath = PlotterSave._prepare_path(directory, fileName, typ)
            np.savetxt(filepath, to_save)
        else:
            raise ValueError(f"Unsupported file type '{typ}'. Use '.npy', '.txt', or '.dat'.")

    @staticmethod
    def twoColumnsData(
        directory: Union[str, Path],
        fileName: str,
        x: Any,
        y: Any,
        typ: str = ".npy",
    ) -> None:
        """Store x and y vectors in 2D format (columns: x, y)."""
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        if len(x_arr) != len(y_arr):
            raise ValueError(f"Sizes incompatible: len(x)={len(x_arr)} != len(y)={len(y_arr)}.")

        to_save = np.column_stack((x_arr, y_arr))

        if typ == ".npy":
            filepath = PlotterSave._prepare_path(directory, fileName, ".npy")
            np.save(filepath, to_save)
        elif typ in (".txt", ".dat"):
            filepath = PlotterSave._prepare_path(directory, fileName, typ)
            np.savetxt(filepath, to_save)
        else:
            raise ValueError(f"Unsupported file type '{typ}'. Use '.npy', '.txt', or '.dat'.")

    @staticmethod
    def matrixData(
        directory: Union[str, Path],
        fileName: str,
        x: Any,
        y: Any,
        typ: str = ".npy",
    ) -> None:
        """Store x coordinates along with a 2D matrix y (prepend x as first column)."""
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        if len(x_arr) != len(y_arr):
            raise ValueError(f"Sizes incompatible: len(x)={len(x_arr)} != len(y)={len(y_arr)}.")

        if y_arr.ndim == 1:
            to_save = np.column_stack((x_arr, y_arr))
        else:
            to_save = np.column_stack((x_arr, y_arr))

        if typ == ".npy":
            filepath = PlotterSave._prepare_path(directory, fileName, ".npy")
            np.save(filepath, to_save)
        else:
            filepath = PlotterSave._prepare_path(directory, fileName, typ)
            np.savetxt(filepath, to_save)

    @staticmethod
    def app_df(df: Any, colname: str, y: Any, fill_value: Any = np.nan) -> None:
        """Append a column to a DataFrame, matching length by padding or truncation."""
        y_arr = np.asarray(y)
        original_len = len(y_arr)
        df_len = len(df)

        if original_len < df_len:
            pad_shape = (df_len,) + y_arr.shape[1:]
            padded = np.full(pad_shape, fill_value, dtype=y_arr.dtype if not np.isnan(fill_value) else float)
            padded[:original_len] = y_arr
            df[colname] = padded
        elif original_len > df_len:
            df[colname] = y_arr[:df_len]
        else:
            df[colname] = y_arr

    @staticmethod
    def app_array(arr: np.ndarray, y: Any) -> np.ndarray:
        """Append data along axis 0 to a numpy array."""
        return np.append(arr, y, axis=0)


class MatrixPrinter:
    """Pretty-display matrices and vectors in Jupyter notebooks or text."""

    def __init__(self):
        if HAS_SYMPY and init_printing is not None:
            try:
                init_printing()
            except Exception:
                pass

    @staticmethod
    def print_matrix(matrix: np.ndarray) -> None:
        """Print or display a matrix."""
        if HAS_SYMPY and display is not None and Matrix is not None:
            display(Matrix(matrix))
        else:
            print("Matrix:")
            print(matrix)

    @staticmethod
    def print_vector(vector: np.ndarray) -> None:
        """Print or display a vector."""
        if HAS_SYMPY and display is not None and Matrix is not None:
            display(Matrix(vector))
        else:
            print("Vector:")
            print(vector)

    @staticmethod
    def print_matrices(matrices: Sequence[np.ndarray]) -> None:
        """Print or display multiple matrices."""
        for matrix in matrices:
            MatrixPrinter.print_matrix(matrix)

    @staticmethod
    def print_vectors(vectors: Sequence[np.ndarray]) -> None:
        """Print or display multiple vectors."""
        for vector in vectors:
            MatrixPrinter.print_vector(vector)


# ---------------------------------------------
#! EOF
# ---------------------------------------------
