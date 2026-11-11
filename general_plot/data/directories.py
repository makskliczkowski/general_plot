r"""Discover files across roots with stable ordering and no duplicate physical files.

-------------------------------------------
Author: Maksymilian Kliczkowski
File: general_plot/data/directories.py
-------------------------------------------
"""
from __future__ import annotations

from pathlib    import Path
from typing     import Callable, Iterable, Iterator, Optional, Tuple, Union


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
