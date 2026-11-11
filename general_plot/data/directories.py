r"""Discover files across roots with stable ordering and no duplicate physical files.

-------------------------------------------
Author: Maksymilian Kliczkowski
File: general_plot/data/directories.py
-------------------------------------------
"""
from __future__     import annotations

import math
import re
from numbers        import Real
from pathlib        import Path
from typing         import Callable, Iterable, Iterator, Mapping, Optional, Tuple, Union
from urllib.parse   import quote, unquote


# -------------------------------------------


def format_number(value: Real) -> str:
    """Format a finite number as a compact folder label.

    Parameters
    ----------
    value : real number
        Integers below one million retain all digits. Other values use four
        significant digits. This display format is not an exact run identity.

    Returns
    -------
    str
        Decimal or scientific notation. Non-finite values raise ValueError.
    """
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("folder numbers must be finite")
    return str(int(value)) if value.is_integer() and abs(value) < 1e6 else f"{value:.4g}"


def format_parameters(parameters: Mapping[str, Union[str, Real, None]]) -> str:
    """Encode ordered parameter fields as one comma-separated folder name.

    Parameters
    ----------
    parameters : mapping
        Nonempty text keys mapped to text, real numbers or None. None writes
        a bare flag. Empty text writes key_. Numbers use format_number and
        booleans use true/false. Mapping order determines display order.

    Returns
    -------
    str
        key_value tokens joined with commas. Percent encoding protects path
        separators, commas, percent signs and underscores in keys. Value
        underscores remain readable. Empty mappings and dot paths are rejected.
        Numeric formatting can lose precision. Store exact identity in metadata.
    """
    tokens = []
    for key, value in parameters.items():
        if not isinstance(key, str) or not key:
            raise ValueError("folder parameter keys must be nonempty strings")
        name = quote(key, safe="").replace("_", "%5F")
        if value is not None:
            if isinstance(value, bool):
                value = str(value).lower()
            elif isinstance(value, Real):
                value = format_number(value)
            elif not isinstance(value, str):
                raise TypeError(f"unsupported folder value for {key!r}: {type(value).__name__}")
            name += "_" + quote(value, safe="")
        tokens.append(name)
    folder = ",".join(tokens)
    if folder in ("", ".", ".."):
        raise ValueError("folder parameters must form a nonempty path component")
    return folder


def parse_parameters(folder: str) -> dict[str, Optional[str]]:
    """Decode a parameter folder without guessing scientific types or defaults.

    Parameters
    ----------
    folder : str
        One comma-separated path component using key_value tokens and bare
        flags. Each valued token splits at its first underscore. Percent escapes
        must be valid UTF-8. Existing unescaped value underscores are supported.

    Returns
    -------
    dict
        Ordered text keys and values. Bare flags map to None. Numeric text,
        auto, none and booleans stay strings. Invalid names, empty tokens and
        duplicate decoded keys raise ValueError. No files are read or created.
    """
    if not isinstance(folder, str):
        raise TypeError("folder must be a string path component")
    
    if folder in ("", ".", "..") or any(char in folder for char in ("/", "\\", "\x00")):
        raise ValueError(f"invalid parameter folder: {folder!r}")
    
    if re.search(r"%(?![0-9a-fA-F]{2})", folder):
        raise ValueError(f"invalid percent escape in folder: {folder!r}")
    
    fields = {}
    for token in folder.split(","):
        key, separator, value   = token.partition("_")
        key                     = unquote(key, errors="strict")
        if not key or key in fields:
            raise ValueError(f"empty or duplicate parameter key: {key!r}")
        fields[key] = unquote(value, errors="strict") if separator else None
    return fields


def discover_files(roots: Union[str, Path, Iterable[Union[str, Path]]], pattern: str, *, predicate: Optional[Callable[[Path], bool]] = None) -> Iterator[Tuple[Path, Path]]:
    """Yield (root, file) pairs once per resolved file across overlapping roots.

    Parameters
    ----------
    roots : path or iterable of paths
        Existing directories. Missing roots raise FileNotFoundError.
    pattern : str
        Path.glob pattern. Files are sorted within each root.
    predicate : callable, optional
        Additional file selection without opening its contents.

    Yields
    ------
    tuple of Path
        Absolute root and file paths. First root wins when roots overlap.
        Discovery never creates directories or alters files.
    """
    roots   = [roots] if isinstance(roots, (str, Path)) else roots
    seen    = set()
    
    # Iterate over roots in the order given, yielding files in sorted order.
    for root in roots:
        root = Path(root).resolve()
        
        # Check that the root is a directory before globbing. This avoids creating a directory if it doesn't exist.
        if not root.is_dir():
            raise FileNotFoundError(f"run root is not a directory: {root}")
        
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or (predicate is not None and not predicate(path)):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield root, path


# -------------------------------------------
#! EOF
# -------------------------------------------
