"""Pytest configuration and shared fixtures for general_plot tests."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

# Ensure headless non-GUI backend during all tests
mpl.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_gp_test")


@pytest.fixture(autouse=True)
def clean_mpl_state():
    """Ensure all Matplotlib figures are closed and rcParams reset for test isolation."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    yield
    plt.close("all")
    mpl.rcParams.update(mpl.rcParamsDefault)


# ---------------------------------------------
#! EOF
# ---------------------------------------------
