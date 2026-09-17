"""Curve fitting and scaling analysis.

Implementation follows the ``Fitter`` design of the standalone scientific
toolkit referenced by this project (see ``maths.math_utils`` in that repo):
a class holding ``(x, y)`` samples, producing a ``FitterParams`` result
(optimal parameters, covariance, callable) via ``scipy.optimize.curve_fit``.

The module is self-contained: it depends only on ``numpy``, ``scipy``, and
optionally ``pandas`` for DataFrame-aware helpers. No external package is
imported for the fitting logic.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit as _curve_fit

try:
    import pandas as pd
except ImportError:                                                             # pragma: no cover - optional dependency
    pd = None

PI     = math.pi
PIHALF = math.pi / 2
TWOPI  = math.pi * 2


def find_maximum_idx(x):
    """Return the index of the maximum element along axis 1.

    Supports a DataFrame, a 2D numpy array, or a JAX array. A DataFrame is
    reduced via ``idxmax(axis=1)``; an array via ``argmax(axis=1)``.
    """
    if pd is not None and isinstance(x, pd.DataFrame):
        return x.idxmax(axis=1)
    if isinstance(x, np.ndarray):
        return np.argmax(x, axis=1)
    raise TypeError("Input must be a DataFrame, numpy array, or JAX array")


def find_nearest_val(x, val, col=None):
    """Return the value in ``x`` closest to ``val``.

    ``col`` selects a column for a DataFrame. Numpy arrays are searched flat.
    """
    if pd is not None and isinstance(x, pd.DataFrame):
        if col is None:
            raise ValueError("Column name must be provided for DataFrame.")
        idx = (x[col] - val).abs().idxmin()
        return x.at[idx, col]
    if isinstance(x, np.ndarray):
        idx = int((np.abs(x - val)).argmin())
        return x.flat[idx]
    raise TypeError("Input must be a DataFrame, numpy array, or JAX array")


def find_nearest_idx(x, val, col=None):
    """Return the index of the value in ``x`` closest to ``val``.

    ``col`` selects a column for a DataFrame. Numpy arrays are searched flat.
    """
    if pd is not None and isinstance(x, pd.DataFrame):
        if col is None:
            raise ValueError("Column name must be provided for DataFrame.")
        return (x[col] - val).abs().idxmin()
    if isinstance(x, np.ndarray):
        return int((np.abs(x - val)).argmin())
    raise TypeError("Input must be a DataFrame, numpy array, or JAX array")


class FitterParams:
    """Container for a fitted model.

    Attributes
    ----------
    popt : np.ndarray
        Optimized fit parameters.
    pcov : np.ndarray
        Estimated covariance matrix of the parameters.
    funct : callable
        Fitted callable mapping x to predicted values.
    """

    def __init__(self, funct, popt, pcov):
        self._popt = popt
        self._pcov = pcov
        self._funct = funct

    def get_popt(self):
        """Optimized fit parameters."""
        return self._popt

    def get_pcov(self):
        """Estimated covariance matrix of the fitted parameters."""
        return self._pcov

    def get_fun(self):
        """Fitted callable."""
        return self._funct

    @property
    def popt(self):
        """Optimized fit parameters."""
        return self._popt

    @property
    def pcov(self):
        """Estimated covariance matrix of the fitted parameters."""
        return self._pcov

    @property
    def funct(self):
        """Fitted callable."""
        return self._funct

    def __call__(self, x):
        """Evaluate the fitted function at ``x``."""
        return self._funct(x)

    def __str__(self):
        return f"FitterParams: {self._popt} {self._pcov}"


class Fitter:
    """Fit ``(x, y)`` sample pairs to functional models.

    The class is used either as a stateful fitter holding the data, or through
    its static methods that operate on explicit ``x`` and ``y`` arrays.

    The ``fit_*`` instance methods store the result in ``self._fitter``; the
    ``fit*`` static methods return a ``FitterParams`` directly.

    - ``apply(x)`` evaluates the current fitted function.
    - ``skip(x, y, skipF, skipL)`` trims leading/trailing samples before fitting.
    """

    _MEAN_ALIASES = {
        "arithmetic": {"arithmetic", "arith", "avg", "average", "mean"},
        "typical":    {"typical", "typ", "geometric", "geo", "log-mean", "logmean"},
        "median":     {"median", "med"},
        "harmonic":   {"harmonic", "harm"},
    }

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """Initialize the fitter with the sample arrays.

        Parameters
        ----------
        x : np.ndarray
            Arguments.
        y : np.ndarray
            Values.
        """
        self._x      = x
        self._y      = y
        self._fitter = FitterParams(None, None, None)

    def apply(self, x: np.ndarray):
        """Evaluate the current fitted function at ``x``."""
        return self._fitter.funct(x)

    @staticmethod
    def skip(x, y, skipF: int = 0, skipL: int = 0):
        """Trim ``skipF`` leading and ``skipL`` trailing elements.

        Parameters
        ----------
        x, y : array-like
            Arguments and values to trim.
        skipF : int
            Number of first elements to skip.
        skipL : int
            Number of last elements to skip.
        """
        xfit = x[skipF if skipF != 0 else None : -skipL if skipL != 0 else None]
        yfit = y[skipF if skipF != 0 else None : -skipL if skipL != 0 else None]
        return xfit, yfit

    # ------------------------------------------------------------------
    # Robust helpers for scaling analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_mean(mean_type: str) -> str:
        key = str(mean_type).strip().lower()
        for canon, aliases in Fitter._MEAN_ALIASES.items():
            if key in aliases:
                return canon
        raise ValueError(
            f"Unknown mean_type '{mean_type}'. Use one of: arithmetic/avg/mean, typical/typ/geometric, median, harmonic."
        )

    @staticmethod
    def _as_1d(a, dtype=float):
        return np.asarray(a, dtype=dtype).ravel()

    @staticmethod
    def _prepare_xy(x, y, *, require_positive_x: bool = False, require_positive_y: bool = False, eps: float = 1e-300, dtype=float):
        """Return filtered finite 1D x and y arrays with equal length."""
        x_arr = Fitter._as_1d(x, dtype=dtype)
        y_arr = Fitter._as_1d(y, dtype=dtype)
        if x_arr.size != y_arr.size:
            raise ValueError(f"x and y must have same length, got {x_arr.size} and {y_arr.size}.")

        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if require_positive_x and x_arr.dtype in (np.float32, np.float64):
            mask &= x_arr > 0.0
        if require_positive_y and y_arr.dtype in (np.float32, np.float64):
            mask &= y_arr > eps

        x_ok = x_arr[mask]
        y_ok = y_arr[mask]
        if x_ok.size < 2:
            raise ValueError("Need at least 2 valid points after filtering.")
        return x_ok, y_ok

    @staticmethod
    def aggregate(values, *, mean_type: str = "arithmetic", weights=None, trim_fraction: float = 0.0, eps: float = 1e-300):
        """Aggregate samples with a configurable averaging convention.

        Parameters
        ----------
        values : array-like
            Input samples.
        mean_type : str
            One of arithmetic/avg/mean, typical/typ/geometric, median, harmonic.
        weights : array-like, optional
            Optional weights for arithmetic averaging only.
        trim_fraction : float, default=0.0
            Fraction trimmed from each tail for arithmetic mean (0 <= trim < 0.5).
        eps : float, default=1e-300
            Positivity cutoff for logarithmic/harmonic aggregations.
        """
        arr  = Fitter._as_1d(values, dtype=float)
        arr  = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan

        mode = Fitter._canonical_mean(mean_type)

        if mode == "median":
            return float(np.median(arr))

        if mode == "typical":
            pos = arr[arr > eps]
            if pos.size == 0:
                return np.nan
            return float(np.exp(np.mean(np.log(pos))))

        if mode == "harmonic":
            pos = arr[arr > eps]
            if pos.size == 0:
                return np.nan
            return float(pos.size / np.sum(1.0 / pos))

        # arithmetic
        trim = float(trim_fraction)
        if trim < 0.0 or trim >= 0.5:
            raise ValueError("trim_fraction must satisfy 0 <= trim_fraction < 0.5.")

        if trim > 0.0 and arr.size >= 3:
            arr = np.sort(arr)
            k = int(trim * arr.size)
            if 2 * k < arr.size:
                arr = arr[k : arr.size - k]

        if weights is None:
            return float(np.mean(arr))

        values_arr = Fitter._as_1d(values, dtype=float)
        w = Fitter._as_1d(weights, dtype=float)
        if w.size != values_arr.size:
            raise ValueError("weights must have the same length as values.")

        mask = np.isfinite(values_arr) & np.isfinite(w)
        v    = values_arr[mask]
        wv   = w[mask]
        s    = np.sum(wv)
        if s == 0.0:
            return np.nan
        return float(np.dot(v, wv) / s)

    @staticmethod
    def fit_loglog_linear(x, y):
        """Fit ``log(y) = a + b log(x)`` and return ``(a, b, r2)``."""
        x_ok, y_ok = Fitter._prepare_xy(x, y, require_positive_x=True, require_positive_y=True)
        lx         = np.log(x_ok)
        ly         = np.log(y_ok)
        b, a       = np.polyfit(lx, ly, 1)
        yhat       = a + b * lx
        ss_res     = float(np.sum((ly - yhat) ** 2))
        ss_tot     = float(np.sum((ly - np.mean(ly)) ** 2))
        r2         = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
        return float(a), float(b), float(r2)

    @staticmethod
    def fit_power_scaling(x, y):
        """Fit ``y = A x^beta`` and return ``(A, beta, r2_log)``."""
        a, b, r2 = Fitter.fit_loglog_linear(x, y)
        return float(np.exp(a)), float(b), float(r2)

    @staticmethod
    def fit_inverse_power_scaling(x, y):
        """Fit ``y = A x^{-alpha}`` and return ``(A, alpha, r2_log)``."""
        A, beta, r2 = Fitter.fit_power_scaling(x, y)
        return float(A), float(-beta), float(r2)

    @staticmethod
    def fit_ipr_scaling(dimensions, iprs, q: float = 2.0):
        r"""Fit ``IPR(N) = A N^{-alpha}`` and infer ``D_q = alpha/(q-1)``.

        Returns
        -------
        tuple
            ``(A, alpha, D_q, r2_log)``.
        """
        A, alpha, r2      = Fitter.fit_inverse_power_scaling(dimensions, iprs)
        Dq                = alpha / (q - 1.0) if q != 1.0 else math.inf
        return float(A), float(alpha), float(Dq), float(r2)

    #################### F I T S ! ####################

    def fit_linear(self, skipF: int = 0, skipL: int = 0):
        """Fit a linear model ``y = a * x + b`` and store the result.

        Parameters
        ----------
        skipF : int
            Number of first points to skip.
        skipL : int
            Number of last points to skip.
        """
        xfit, yfit = Fitter.skip(self._x, self._y, skipF, skipL)
        a, b       = np.polyfit(xfit, yfit, 1)
        self._fitter = FitterParams(lambda x: a * x + b, [a, b], [])

    @staticmethod
    def fitLinear(x, y, skipF: int = 0, skipL: int = 0):
        """Fit a linear model ``y = a * x + b`` and return a ``FitterParams``."""
        xfit, yfit = Fitter.skip(x, y, skipF, skipL)
        a, b       = np.polyfit(xfit, yfit, 1)
        return FitterParams(lambda x: a * x + b, [a, b], [])

    #############

    def fit_exp(self, skipF: int = 0, skipL: int = 0):
        """Fit ``y = a * exp(-b * x)`` and store the result."""
        xfit, yfit     = Fitter.skip(self._x, self._y, skipF, skipL)

        def funct(x, a, b):
            return a * np.exp(-b * x)

        popt, pcov     = _curve_fit(funct, xfit, yfit)
        self._fitter   = FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)

    @staticmethod
    def fitExp(x, y, skipF: int = 0, skipL: int = 0):
        """Fit ``y = a * exp(-b * x)`` and return a ``FitterParams``."""
        xfit, yfit  = Fitter.skip(x, y, skipF, skipL)

        def funct(x, a, b):
            return a * np.exp(-b * x)

        popt, pcov  = _curve_fit(funct, xfit, yfit)
        return FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)

    #############

    def fit_x_plus_x2(self, skipF: int = 0, skipL: int = 0):
        """Fit ``y = a * x + b * x**2`` and store the result."""
        xfit, yfit   = Fitter.skip(self._x, self._y, skipF, skipL)

        def funct(x, a, b):
            return (a * x) + (b * x**2)

        popt, pcov   = _curve_fit(funct, xfit, yfit)
        self._fitter = FitterParams(lambda x: popt[0] * x + popt[1] * x**2, popt, pcov)

    @staticmethod
    def fitXPlusX2(x, y, skipF: int = 0, skipL: int = 0):
        """Fit ``y = a * x + b * x**2`` and return a ``FitterParams``."""
        xfit, yfit  = Fitter.skip(x, y, skipF, skipL)

        def funct(x, a, b):
            return (a * x) + (b * x**2)

        popt, pcov  = _curve_fit(funct, xfit, yfit)
        return FitterParams(lambda x: popt[0] * x + popt[1] * x**2, popt, pcov)

    #############

    def fit_power(self, skipF: int = 0, skipL: int = 0):
        """Fit ``y = a * x**b`` and store the result."""
        xfit, yfit   = Fitter.skip(self._x, self._y, skipF, skipL)

        def funct(x, a, b):
            return a * x**b

        popt, pcov   = _curve_fit(funct, xfit, yfit)
        self._fitter = FitterParams(lambda x: popt[0] * (x**popt[1]), popt, pcov)

    @staticmethod
    def fitPower(x, y, skipF: int = 0, skipL: int = 0):
        """Fit ``y = a * x**b`` and return a ``FitterParams``."""
        xfit, yfit  = Fitter.skip(x, y, skipF, skipL)

        def funct(x, a, b):
            return a * x**b

        popt, pcov  = _curve_fit(funct, xfit, yfit)
        return FitterParams(lambda x: popt[0] * (x**popt[1]), popt, pcov)

    #################### G E N E R A L ####################

    def fit_any(self, funct, skipF: int = 0, skipL: int = 0):
        """Fit an arbitrary ``funct(x, *params)`` and store the result."""
        xfit, yfit   = Fitter.skip(self._x, self._y, skipF, skipL)
        popt, pcov   = _curve_fit(funct, xfit, yfit)
        self._fitter = FitterParams(lambda x: funct(x, *popt), popt, pcov)

    @staticmethod
    def fitAny(x, y, funct, skipF: int = 0, skipL: int = 0, bounds=[]):
        """Fit an arbitrary ``funct(x, *params)`` and return a ``FitterParams``.

        Parameters
        ----------
        x, y : array-like
            Arguments and values to fit.
        funct : callable
            Model ``funct(x, *params)``.
        skipF : int
            Number of first points to skip.
        skipL : int
            Number of last points to skip.
        bounds : tuple or list
            Parameter bounds passed to ``curve_fit``; use ``[]`` for none.
        """
        xfit, yfit = Fitter.skip(x, y, skipF, skipL)
        if bounds == []:
            popt, pcov = _curve_fit(funct, xfit, yfit)
            return FitterParams(lambda x: funct(x, *popt), popt, pcov)
        popt, pcov = _curve_fit(funct, xfit, yfit, bounds=bounds)
        return FitterParams(lambda x: funct(x, *popt), popt, pcov)

    #################### D I S T R I B U T I O N S ####################

    @staticmethod
    def gen_cauchy(x, v=1.0, gamma=1.0, alpha=1.0, beta=1.0):
        r"""Generalized Cauchy distribution.

        - v is the normalization factor,
        - alpha is the stability (shape) parameter,
        - beta is the scale parameter,
        - gamma is a scale parameter related to the width.
        """
        return v * gamma * (1 + (x * gamma / beta) ** 2) ** (-(alpha + 1) / 2)

    @staticmethod
    def cauchy(x, x0=0.0, gamma=1.0, v=1.0):
        """Cauchy distribution centered at ``x0``."""
        return v / ((x - x0) ** 2 + gamma**2)

    @staticmethod
    def pareto(x, v=1.0, alpha=1.0, xm=1.0, mu=0.0):
        """Pareto-like distribution with exponent controlled by ``xm``."""
        return v * np.power(1.0 + alpha * (np.abs(x - mu)), -1.0 / xm - 1.0)

    @staticmethod
    def poisson(x, lambd=1.0, v=1.0):
        """Poisson-like decaying distribution ``v * exp(-lambd * |x|)``."""
        return v * np.exp(-lambd * (np.abs(x)))

    @staticmethod
    def chi2(x, k=1.0, v=1.0, z=1.0):
        """Chi-squared-like density with shape ``k`` and scale ``z``."""
        return v * np.exp(-np.abs(z * x) / 2) * (np.abs(x) ** (k / 2 - 1)) / (2 ** (k / 2) * math.gamma(k / 2))

    @staticmethod
    def gaussian(x, mu=0.0, sigma=1.0):
        """Gaussian density."""
        return norm_pdf(x, mu, sigma)

    @staticmethod
    def laplace(x, lambd=1.0, v=1.0, mu=0.0):
        """Laplace-like cumulative distribution scaled by ``v``."""
        return (0.5 + 0.5 * np.sign(x - mu) * (1.0 - np.exp(-np.abs(x - mu) / lambd))) / v

    @staticmethod
    def exponential(x, lambd, sigma):
        """Exponential-like decaying distribution ``sigma * exp(-lambd * |x|)``."""
        return sigma * np.exp(-lambd * np.abs(x))

    @staticmethod
    def lorentzian(x, v=1.0, g=1.0):
        """Lorentzian centered at zero with width ``g`` and amplitude ``v``."""
        return np.abs(v) * g / (x**2 + g**2)

    @staticmethod
    def two_lorentzian(x, v=1.0, g1=1.0, g2=1.0, v2=1.0):
        """Sum of two Lorentzians; returns a large value for unphysical widths."""
        if g1 < 0 or g2 < 0 or v < 0 or v2 < 0:
            return 1.0e10
        return Fitter.lorentzian(x, v, g1) + Fitter.lorentzian(x, v2, g2)

    @staticmethod
    def lorentzian_system_size(param):
        """Return a Lorentzian model whose width is scaled by ``param[0]``."""
        def lorentzian(x, g=1.0, v=1.0):
            return v * (g / param[0]) / (x**2 + (g / param[0]) ** 2)
        return lorentzian

    #################### H I S T O G R A M ####################

    @staticmethod
    def fit_histogram(edges, counts, typek="gaussian", skipF: int = 0, skipL: int = 0, centers=[], params=[], bounds=None):
        """Fit a parametric distribution to histogram bin edges and counts.

        Parameters
        ----------
        edges : array-like
            Histogram bin edges.
        counts : array-like
            Bin counts (or densities).
        typek : str
            Distribution name: gaussian, poisson, laplace, exponential, pareto,
            cauchy, gen_cauchy, chi2, lorentzian, two_lorentzian, or
            lorentzian_system_size.
        skipF : int
            Number of first bins to skip.
        skipL : int
            Number of last bins to skip.
        centers : array-like
            Explicit bin centers; computed from edges when empty.
        params : list
            Extra parameters, required for ``lorentzian_system_size``.
        bounds : tuple, optional
            Parameter bounds passed to ``curve_fit``.
        """
        model_names = {
            "gaussian": Fitter.gaussian,
            "poisson": Fitter.poisson,
            "laplace": Fitter.laplace,
            "exponential": Fitter.exponential,
            "pareto": Fitter.pareto,
            "cauchy": Fitter.cauchy,
            "gen_cauchy": Fitter.gen_cauchy,
            "chi2": Fitter.chi2,
            "lorentzian": Fitter.lorentzian,
            "two_lorentzian": Fitter.two_lorentzian,
            "lorentzian_system_size": Fitter.lorentzian_system_size(params),
        }
        if typek not in model_names:
            raise ValueError("Type not recognized: " + typek)
        funct = model_names[typek]

        if len(centers) == 0:
            if len(edges) <= 1:
                raise ValueError("Edges must have at least 2 elements")
            centers = (edges[:-1] + edges[1:]) / 2

        centers, counts = Fitter.skip(centers, counts, skipF, skipL)
        popt, pcov      = _curve_fit(funct, centers, counts)
        return FitterParams(lambda x: funct(x, *popt), popt, pcov)

    @staticmethod
    def get_histogram(typek="gaussian"):
        """Return the distribution model callable for ``typek``."""
        if typek == "gaussian":
            return Fitter.gaussian
        if typek == "poisson":
            return Fitter.poisson
        if typek == "laplace":
            return Fitter.laplace
        if typek == "exponential":
            return Fitter.exponential
        if typek == "pareto":
            return Fitter.pareto
        if typek == "cauchy":
            return Fitter.cauchy
        raise ValueError("Type not recognized: " + typek)


def norm_pdf(x, mu=0.0, sigma=1.0):
    """Gaussian probability density without scipy.stats.

    Returns the normalized density for the given mean and standard deviation.
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(TWOPI))


def next_power(x: float, base: int = 2):
    """Return the smallest power of ``base`` greater than ``x``."""
    return base ** math.ceil(math.log(x) / math.log(base))


def prev_power(x: float, base: int = 2):
    """Return the largest power of ``base`` smaller than ``x``."""
    return base ** math.floor(math.log(x) / math.log(base))


def mod_euc(a: int, b: int) -> int:
    """Return the Euclidean remainder of ``a`` divided by ``b`` (result takes the sign of ``b``)."""
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
    m = a % b
    if m < 0:
        m = m - b if b < 0 else m + b
    return m


def mod_floor(a: int, b: int) -> int:
    """Return ``a`` floor-divided by ``b`` (result rounded toward negative infinity)."""
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
    m = a // b
    if (a < 0) != (b < 0) and a % b != 0:
        m -= 1
    return m


def mod_ceil(a: int, b: int) -> int:
    """Return ``a`` ceil-divided by ``b`` (result rounded toward positive infinity)."""
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
    m = a // b
    if (a < 0) == (b < 0) and a % b != 0:
        m += 1
    return m


def mod_trunc(a: int, b: int) -> int:
    """Return ``a`` truncated-divided by ``b`` (result rounded toward zero)."""
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
    return a // b


def mod_round(a: int, b: int) -> int:
    """Return ``a`` rounded-divided by ``b`` (result rounded to the nearest integer)."""
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
    m = a / b
    if m < 0:
        m = m - 1 if a % b < 0 else m + 1
    return int(m)


def thin(*arrays, max_points=None) -> tuple:
    """Subsample all arrays together to at most `max_points` evenly-spaced rows.

    Parameters
    ----------
    arrays : sequence of array-like
        Aligned trajectories of equal leading length (e.g. steps, energy).
    max_points : int, optional
        Maximum number of samples to keep after thinning. When None or when
        the input is already shorter, the arrays are returned unchanged.

    Returns
    -------
    tuple
        One thinned array per input, all sliced with the same stride so they
        stay mutually aligned after subsampling.
    """
    if not arrays:
        return ()
    n   = len(arrays[0])
    if not max_points or n <= max_points:
        return arrays
    stride  = max(1, int(np.ceil(n / max_points)))
    return tuple(np.asarray(a)[::stride] for a in arrays)

# ---------------------------------------------
#! EOF
# ---------------------------------------------
