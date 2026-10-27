# Scan fixed values and choose the fit function

Two options of `YamboConvergence` for when the default automated convergence is not what you need.

## Run a list of parameter sets (no convergence)

Sometimes you just need several calculations: to run tests, or to collect data for your own post-processing.
Give `scan_points` instead of `convergence_parameters`: one dictionary `{variable: value}` for each calculation.

```python
from aiida_yambo.workflows.yamboconvergence import YamboConvergence

builder = YamboConvergence.get_builder_from_protocol(
    pw_code=pw_code, preprocessing_code=p2y_code, code=yambo_code,
    structure=structure,
    scan_points=[
        {'FFTGvecs': 10, 'BndsRnXp': 100, 'NGsBlkXp': 2},
        {'FFTGvecs': 15, 'BndsRnXp': 200, 'NGsBlkXp': 4},
        {'FFTGvecs': 20, 'BndsRnXp': 300, 'NGsBlkXp': 6},
    ],
)
```

- Every set must define the same variables. Units default to the usual ones (`Ry` for the cutoffs, none for the
  bands, `1/A` for `kpoint_density`) and `GbndRnge` follows `BndsRnXp`; to change them, build the
  `ScanParam` yourself (`ScanParam.from_dicts(points, units=..., mirror_values=...)`) and pass it in `parameters_space`.
- The calculations are submitted `max_concurrent_steps` at a time (default 4). Calculations already done in the
  same group are reused.
- **Nothing is evaluated.** When all the points are done the workflow ends. A calculation which fails does *not*
  stop the others: it is flagged `failed` in the story. The workflow fails (`CALCS_FAILED`) only if *all* of
  them failed.
- A `ScanParam` must be the only entry of `parameters_space`.

The results are in the `collection` output (`story`: one row per calculation, with the values of the variables,
the quantities in `workflow_settings['what']`, and the `failed` flag) and in `convergence_summary['scan']`
(number of points and of failures).

```python
story = node.outputs.collection.get_dict()['story']
```

## Choose the function fitted to the results

By default the predictors fit a power law `a / x**alpha + b` (1D) or a product of two power laws (coupled
parameters), choosing the exponents with the smallest error. You can select another function *by name*, per
parameter:

```python
builder = YamboConvergence.get_builder_from_protocol(
    ...,
    fit_functions={
        'FFTGvecs': 'exponential',                                   # built-in 1D model: a*exp(-x/L) + b
        'BndsRnXp': {'fit_function': 'power_law',                  # the coupled bands/G-vectors parameter
                     'fit_options': {'alphas': [1, 2], 'betas': [1]}},
    },
)
```

The names are stored in the provenance (`fit_function`, `fit_options` of each parameter), not the function itself.
An unknown name fails as soon as the parameter is defined, listing the available ones.

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
