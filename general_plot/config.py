"""Configuration dataclasses for general_plot."""

from __future__     import annotations

from dataclasses    import asdict, dataclass, field
from typing         import Any, Dict, List, Optional, Tuple


@dataclass
class PlotStyle:
    """Style configuration options for publication figures."""

    style       : str               = "publication"
    font_size   : int               = 10
    use_latex   : bool              = False
    dpi         : int               = 150
    palette     : str               = "nature"
    overrides   : Dict[str, Any]    = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FigureConfig:
    """Figure sizing and layout configuration."""

    figsize             : Optional[Tuple[float, float]] = None
    width               : Optional[float]               = 3.5
    height              : Optional[float]               = 2.5
    dpi                 : int                           = 300
    constrained_layout  : bool                          = True
    tight_layout        : bool                          = False
    facecolor           : str                           = "white"
    edgecolor           : str                           = "white"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KSpaceConfig:
    """Reciprocal space / Brillouin zone plotting parameters."""

    gamma_point : Tuple[float, float]   = (0.0, 0.0)
    k_min       : float                 = -1.0
    k_max       : float                 = 1.0
    num_points  : int                   = 100
    symmetric   : bool                  = True
    labels      : List[str]             = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KPathConfig:
    """High-symmetry path configuration for band structures."""

    points      : List[Tuple[float, float, float]]  = field(default_factory=list)
    labels      : List[str]                         = field(default_factory=list)
    num_points  : int                               = 300
    ticks       : List[float]                       = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpectralConfig:
    """Spectral function / DOS configuration."""

    energy_min      : float = -5.0
    energy_max      : float = 5.0
    broadening      : float = 0.05
    num_energies    : int   = 500
    normalize       : bool  = False
    colormap        : str   = "viridis"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------
#! EOF
# ---------------------------------------------
