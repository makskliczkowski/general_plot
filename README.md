# general_plot

Scientific visualization toolkit based on Matplotlib. The module provides standardized configuration presets, multi-panel axis management, tick formatters, color palette manipulation, and curve fitting for publication figures.

## Installation

### Submodule

To embed as a Git submodule in an existing repository, run:

```bash
git submodule add <repository-url> tools/general_plot
```

Scripts import from the submodule path directly:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "tools" / "general_plot"))

from general_plot import Plotter, configure_style
```

### Standalone Package

Install in editable mode:

```bash
pip install -e .
```

To install with optional dependencies (such as `scienceplots`, `pandas`, `scipy`):

```bash
pip install -e ".[all]"
```

## Architecture

The library is organized into dedicated functional modules under `general_plot`:

| Module | Contents |
|---|---|
| `general_plot.style` | Style presets (`publication`, `nature`, `science`, `poster`), cycle iterators, and normalizers |
| `general_plot.axes` | `AxesList` wrapper with 1D and 2D indexing, named panels, and `IgnoredAxis` proxy |
| `general_plot.colors` | Colormaps, journal palettes, conversions, and perceptual transforms |
| `general_plot.formatters` | Numeric tick formatters (`CustomFormatter`, `PercentFormatter`, `MathTextSciFormatter`) |
| `general_plot.fitting` | Model regressions via `Fitter`/`FitterParams` (linear, exponential, power, polynomial, arbitrary, histogram distributions) |
| `general_plot.plotter` | Primary `Plotter` interface and `GridBuilder` multi-row layout manager |
| `general_plot.config` | Dataclasses for plot, figure, and reciprocal space configuration |
| `general_plot.data_loader` | Result filtering with parameter tolerances via `filter_results` |
| `general_plot.io` | Matrix printing and data serialization via `PlotterSave` |
| `general_plot.help` | Terminal documentation and layout guidelines via `Plotter.help()` |

## Usage

### Multi-Panel Figures

The `Plotter.get_subplots` method constructs figures wrapped in an `AxesList`. Individual panels support linear indexing, coordinate indexing, or named panel keys.

```python
import numpy as np
import matplotlib.pyplot as plt
from general_plot import Plotter, configure_style

configure_style("nature")

fig, axes = Plotter.get_subplots(
    nrows=2,
    ncols=2,
    sizex=3.5,
    sizey=2.5,
    named_panels={"signal": (0, 0), "spectrum": (0, 1), "scaling": (1, 0)},
)

axes.disable(1, 1)

t = np.linspace(0, 10, 200)
axes["signal"].plot(t, np.sin(2 * np.pi * 0.5 * t) * np.exp(-t / 4), color="C0")
Plotter.set_ax_params(axes["signal"], xlabel="Time (s)", ylabel="Amplitude", title="Signal")

freqs = np.linspace(0.1, 5, 100)
axes["spectrum"].plot(freqs, 1.0 / (1.0 + (freqs - 0.5)**2), color="C1")
Plotter.set_ax_params(axes["spectrum"], xlabel="Frequency (Hz)", ylabel="Power", title="Spectrum")

x = np.logspace(1, 4, 30)
y = 50.0 * (x ** -1.5)
axes["scaling"].set_xscale("log")
axes["scaling"].set_yscale("log")
Plotter.scatter(axes["scaling"], x, y, color="C2", s=15)
Plotter.power_law_guide(axes["scaling"], x_range=(50, 2000), exponent=-1.5, anchor=(100, y[5]))
Plotter.set_ax_params(axes["scaling"], xlabel=r"System Size $N$", ylabel=r"Variance $\sigma^2$")

Plotter.save_fig("./figures", "multipanel.pdf", format="pdf")
plt.close(fig)
```

### Power-Law Guides

Scaling behavior on logarithmic axes follows

$$y = A x^{\gamma}\;.$$

The guide line passes through an optional coordinate anchor $(x_0, y_0)$, fixing the prefactor $A = y_0 x_0^{-\gamma}$.

```python
Plotter.power_law_guide(ax, x_range=(10.0, 1000.0), exponent=-2.0, anchor=(10.0, 1.0))
```

### Curve Fitting

The `Fitter` class provides standard functional fits backed by `scipy.optimize.curve_fit`. Each fit returns a `FitterParams` object holding the optimal parameters (`popt`), the covariance matrix (`pcov`), and the fitted callable.

```python
import numpy as np
from general_plot import Fitter

x = np.linspace(0, 5, 50)
y = 2.5 * x + 1.0

# Stateful API
fitter = Fitter(x, y)
fitter.fit_linear()
slope, intercept = fitter._fitter.popt   # access via apply() or FitterParams
y_fit = fitter.apply(x)

# Static API returns a FitterParams directly
fp_linear   = Fitter.fitLinear(x, y)
slope, intc = fp_linear.get_popt()
y_pred      = fp_linear(x)

fp_power    = Fitter.fitPower(x, 3.0 * x**2)
A, b        = fp_power.get_popt()

fp_exp      = Fitter.fitExp(x, y, skipF=2, skipL=2)       # y = a*exp(-b*x)
a_exp, b_exp = fp_exp.get_popt()

def model(t, a, b):
    return a * np.exp(-b * t)

fp_any      = Fitter.fitAny(x, y, model)                  # arbitrary curve_fit
```

Supporting helpers include `fit_power_scaling` (log-log `y = A x^beta`), `fit_inverse_power_scaling`, `fit_ipr_scaling`, `aggregate` (arithmetic/geometric/median/harmonic means), `fit_histogram`, and module-level `next_power`, `prev_power`, `mod_euc`, `find_nearest_val`.

### Tick Formatters

Formatters enforce consistent decimal and exponent representations without floating-point artifacts.

```python
from general_plot.formatters import CustomFormatter, MathTextSciFormatter, set_formatter

set_formatter(ax, formatter_type="sci", fmt="%1.2e", axis="y")
set_formatter(ax, formatter_type="custom", fmt="{x:.2f}", axis="x")
```

## Testing

Run the test suite:

```bash
pytest
```

Execute code verification:

```bash
ruff check .
```
