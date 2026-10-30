# Choose the fit function of the convergence

!!! info "Only for the convergence"
    The fit function belongs to the **automated convergence** (`convergence_parameters` / `parameters_space` with
    `ConvParam`, `CoupledConvParam`, `KpointMeshConvParam`). A **scan** (`scan_points`) runs fixed points and never
    fits anything: see [Run fixed points](scan-points.md).

## What the fit does

For each parameter being converged, once the calculations of an iteration are done:

1. every candidate function (see [the list](#built-in-fit-functions)) is fitted to the results; the fit gives more
   weight to the large values of the parameter;
2. the candidate with the smallest error is kept;
3. the fitted curve is evaluated up to ten times the largest value computed, and the **converged value** is the first
   one where the curve is within `conv_thr` of its value at the end of that range (`conv_thr` is in percent of that
   value by default, `conv_thr_units='%'`; use `'eV'` for an absolute threshold);
4. that point is computed explicitly and checked against the fit. If the fit is poor (error larger than 10 times
   `conv_thr`, 30 times for coupled parameters) or the point is beyond `max`, a new grid is built and the iteration
   repeats, up to `max_iterations`.

So the choice of the function matters when the data are not well described by the default; with a good choice the
convergence needs fewer iterations.

## Built-in fit functions

The **default** is `power_law`, for every parameter. List what is available in your installation (it includes the
functions you or other packages registered):

```python
from aiida_yambo.workflows.utils.fit_models import available_fit_models

available_fit_models(1)   # ['exponential', 'power_law']   one variable
available_fit_models(2)   # ['power_law']                   two coupled variables
```

| Name | Dimension | Function | Options (`fit_options`) | Default candidates |
| --- | --- | --- | --- | --- |
| `power_law` (**default**) | 1: `ConvParam`, `KpointMeshConvParam` | `a / x**alpha + b`; the converged value tends to `b` | `alphas`: list of exponents to try | `alphas=[1, 2, 3]` (`[1, 2]` for `kpoint_mesh`) |
| `exponential` | 1 | `a * exp(-x / L) + b`; the converged value tends to `b` | none | one model |
| `power_law` (**default**) | 2: `CoupledConvParam` | `(a / x**alpha + b) * (c / y**beta + d)`, with `x`, `y` the two variables in the order of `var` | `alphas`, `betas`: lists of exponents | `alphas=[1, 2]`, `betas=[1, 2]` (4 combinations) |

For the k-points the `x` seen by the function is not the raw value: it is the number of k-points for `kpoint_mesh`,
and `(1/distance)**3` for `kpoint_density`.

### What `alpha` and `beta` are

They are the **exponents** of the power laws: `alpha` for the first variable and `beta` for the second one of a
coupled parameter (for example `BndsRnXp` and `NGsBlkXp`). A larger exponent means that the fitted curve flattens
faster. You give **lists** (`alphas`, `betas`): the function is fitted once for each value (for each pair in 2D) and
the one with the smallest error is kept, so you rarely need to set them. Restrict the lists if you know the behaviour
(`'alphas': [1]`), or to avoid over-fitting when you have few points; widen them if the default exponents fit badly.

## How to use them

The name goes in `fit_function` and the options in `fit_options`, per parameter. Three equivalent ways.

**1. In `get_builder_from_protocol`**, with `fit_functions`: a dictionary whose keys are the variable names of the
parameters (for a coupled parameter, any of its variables; for the k-points `kpoint_density`)
and whose values are the name, or a dictionary with the name and the options:

```python
builder = YamboConvergence.get_builder_from_protocol(
    ...,
    fit_functions={
        'FFTGvecs': 'exponential',                                  # one variable
        'BndsRnXp': {'fit_function': 'power_law',                   # the coupled bands/screening parameter
                     'fit_options': {'alphas': [1, 2], 'betas': [1]}},
    },
)
```

**2. On the parameter object** (as in the [automated convergence tutorial](../tutorials/convergence.md)):

```python
from aiida_yambo.workflows.utils.parameters import ConvParam, CoupledConvParam

fft = ConvParam('FFTGvecs', start=10, stop=30, delta=5, max=50, units='Ry', conv_thr=0.5,
                fit_function='exponential')
bands = CoupledConvParam(['BndsRnXp', 'NGsBlkXp'], start=[10, 2], stop=[30, 6], delta=[10, 2], max=[50, 10],
                         units=['', 'Ry'], conv_thr=10, mirror_values={'BndsRnXp': ['GbndRnge']},
                         fit_function='power_law', fit_options={'alphas': [1, 2], 'betas': [1]})
manager.parameters_space = [fft, bands]
```

**3. Leave the default**: nothing to set.

The names, not the functions, are stored in the provenance. An unknown name fails as soon as the parameter is defined,
and the error lists the available ones.

### Write your own model

A model describes the function and what the predictor needs: the weights of the fit, the derivative (to find
the plateau; numerical if you do not give it) and the value at infinity.

```python
import numpy as np
from aiida_yambo.workflows.utils.fit_models import FitModel1D, register_fit_model


class Saturation(FitModel1D):
    n_parameters = 3                       # a, L, b

    def __call__(self, x, a, L, b):
        return b - a / (x + L)

    def asymptote(self, popt):             # value for x -> infinity: the "extra" value
        return popt[2]

    def p0(self, x, y):                    # starting point of the fit
        return [1.0, 1.0, float(np.max(y))]


# a *factory* returns the list of models to compare (the one with the smallest error is kept)
register_fit_model('saturation', lambda options, variable: [Saturation()], dimension=1)
```

For coupled (2D) parameters subclass `FitModel2D` (`__call__(X, *p)` with `X[0]`, `X[1]` the two variables;
`derivatives(X, *p)` returns `df/dx, df/dy, d2f/dxdy`) and register it with `dimension=2`.

`register_fit_model` works in the process where it is called. The **daemon runs the workflows**, so it must be
able to find your model too: publish it as an entry point of your package and install the package in the
environment of the daemon (then restart it):

```toml
[project.entry-points."aiida_yambo.fit_models_1d"]
saturation = "my_package.fits:saturation_factory"

[project.entry-points."aiida_yambo.fit_models_2d"]
my_surface = "my_package.fits:surface_factory"
```

Notes:

- A model with no `asymptote` (returns `None`) is allowed: the last fitted value is then the reference.
- For `kpoint_density` and `kpoint_mesh` the `x` axis seen by the model is the proxy used by the predictor
  ((1/distance)^3 and the number of k-points), not the raw value.
- The 2D power law keeps the derivatives it has always had in the plateau search (the factor of `y` assumes an
  exponent 1 in `df/dx`, and the factor of `x` in `df/dy`); they only locate the plateau.
