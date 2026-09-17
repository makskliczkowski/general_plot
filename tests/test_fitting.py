"""Tests for the essential Fitter functionality: parameter recovery, apply() binding, and scaling helpers."""

import numpy as np
import pytest

from general_plot.fitting import Fitter, FitterParams


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


def test_fitter_params_container(rng):
    """FitterParams holds popt/pcov/funct and stays callable."""
    fp      = FitterParams(lambda x: 2 * x, [2.0], np.array([[1.0]]))
    assert fp.get_popt() == [2.0]
    assert fp.get_pcov().shape == (1, 1)
    assert fp(3.0) == 6.0
    assert "FitterParams" in str(fp)


def test_static_fits_recover_params(rng):
    """Static fits recover known parameters for linear, exponential, and power models."""
    x       = np.linspace(0.0, 5.0, 60)
    y_lin   = 2.5 * x + 1.0 + rng.normal(0, 0.05, x.size)
    y_exp   = 4.0 * np.exp(-1.5 * x) + rng.normal(0, 0.02, x.size)

    a, b    = Fitter.fitLinear(x, y_lin).get_popt()
    assert np.isclose(a, 2.5, atol=0.1)
    assert np.isclose(b, 1.0, atol=0.1)

    a, b    = Fitter.fitExp(x, y_exp).get_popt()
    assert np.isclose(a, 4.0, rtol=0.1)
    assert np.isclose(b, 1.5, rtol=0.1)

    xp      = np.linspace(1.0, 5.0, 60)
    y_pow   = 3.0 * xp**2.0 + rng.normal(0, 0.02, xp.size)
    a, b    = Fitter.fitPower(xp, y_pow).get_popt()
    assert np.isclose(a, 3.0, rtol=0.1)
    assert np.isclose(b, 2.0, rtol=0.1)


def test_instance_fits_and_apply(rng):
    """Instance fits store parameters and apply() evaluates the bound model."""
    x       = np.linspace(0.0, 5.0, 60)
    y_lin   = 2.5 * x + 1.0 + rng.normal(0, 0.05, x.size)
    f       = Fitter(x, y_lin)
    f.fit_linear()

    a, b    = f._fitter.get_popt()
    assert np.isclose(a, 2.5, atol=0.1)
    assert np.isclose(b, 1.0, atol=0.1)
    assert np.isclose(f.apply(3.0), 2.5 * 3.0 + 1.0, atol=0.2)

    # Power fit: apply() must evaluate the bound power-law (regression: raw model was stored)
    xp, yp  = np.linspace(1.0, 5.0, 60), 3.0 * np.linspace(1.0, 5.0, 60) ** 2.0
    fp      = Fitter(xp, yp)
    fp.fit_power()
    assert np.isclose(fp.apply(2.0), 12.0, rtol=0.2)


def test_fit_any_arbitrary_model(rng):
    """fitAny fits a user-supplied model, with and without bounds."""
    x       = np.linspace(0.0, 3.0, 60)
    y       = 4.0 * np.exp(-1.5 * x) + rng.normal(0, 0.02, x.size)

    def model(t, a, b):
        return a * np.exp(-b * t)

    a, b    = Fitter.fitAny(x, y, model).get_popt()
    assert np.isclose(a, 4.0, rtol=0.2)
    assert np.isclose(b, 1.5, rtol=0.2)

    a, b    = Fitter.fitAny(x, y, model, bounds=([0.0, 0.0], [10.0, 10.0])).get_popt()
    assert np.isclose(a, 4.0, rtol=0.2)
    assert np.isclose(b, 1.5, rtol=0.2)


def test_power_scaling_helpers():
    """Log-log scaling regression recovers exact power-law parameters and r2."""
    x       = np.logspace(0, 3, 50)
    y       = 2.0 * x**-1.5
    a, b, r2 = Fitter.fit_loglog_linear(x, y)
    assert np.isclose(a, np.log(2.0), rtol=1e-9)
    assert np.isclose(b, -1.5, rtol=1e-9)
    assert np.isclose(r2, 1.0)

    A, alpha, Dq, r2i = Fitter.fit_ipr_scaling(x, 5.0 * x**-0.8, q=2.0)
    assert np.isclose(A, 5.0, rtol=1e-9)
    assert np.isclose(alpha, 0.8, rtol=1e-9)
    assert np.isclose(Dq, 0.8, rtol=1e-9)
    assert np.isclose(r2i, 1.0)


def test_aggregate_and_histogram(rng):
    """Aggregate handles the distinct mean conventions; fit_histogram recovers gaussian moments."""
    v       = np.array([1.0, 2.0, 4.0])
    assert np.isclose(Fitter.aggregate(v, mean_type="mean"), 7.0 / 3.0)
    assert np.isclose(Fitter.aggregate(v, mean_type="geometric"), np.cbrt(8.0))
    assert np.isclose(Fitter.aggregate(v, mean_type="median"), 2.0)
    assert np.isclose(Fitter.aggregate(v, mean_type="harmonic"), 3.0 / (1 + 0.5 + 0.25))

    data    = rng.normal(0.0, 1.0, 2000)
    counts, edges = np.histogram(data, bins=40, density=True)
    mu, sigma = Fitter.fit_histogram(edges, counts, typek="gaussian").get_popt()[:2]
    assert np.isclose(mu, 0.0, atol=0.1)
    assert np.isclose(sigma, 1.0, atol=0.1)


def test_input_validation():
    """Mismatched or insufficient data raises clear errors."""
    with pytest.raises(ValueError, match="same length"):
        Fitter._prepare_xy([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="at least 2 valid points"):
        Fitter._prepare_xy([1.0], [1.0])
    with pytest.raises(ValueError, match="mean_type"):
        Fitter.aggregate([1.0, 2.0], mean_type="bogus")


# ---------------------------------------------
#! EOF
# ---------------------------------------------
