"""Write, register and test a fit model on data, without AiiDA and without a cluster.

A fit model is the function fitted to the results of a convergence, with what the predictor needs: the weights, the
derivative (to find the plateau) and the value at infinity. Here: a model ``a * exp(-x / L) + b * x**-2 + c`` is
written and compared with the built-in power law on synthetic data of a gap against the number of bands (replace
``X`` and ``Y`` by the ``story`` of a finished convergence to use your data).

To use a model in a convergence, name it in ``fit_functions`` (see 05_convergence/05_fit_functions.py); the
daemon must find it: register it from a package which declares the entry point of the group
``aiida_yambo.fit_models_1d`` (or ``_2d``).

Status: runs anywhere (it is part of the tests). Needs numpy and scipy; the plot needs matplotlib.
"""
import numpy as np
from scipy.optimize import curve_fit

from aiida_yambo.workflows.utils.fit_models import FitModel1D, get_fit_models, register_fit_model


class ExpPlusPower(FitModel1D):
    """``a * exp(-x / L) + b / x**2 + c``: the value at infinity is ``c``."""

    n_parameters = 4  # a, L, b, c

    def __call__(self, x, a, L, b, c):
        return a * np.exp(-x / L) + b / x ** 2 + c

    def asymptote(self, popt):
        return popt[3]

    def p0(self, x, y):
        return [y[0] - y[-1], (x.max() - x.min()) / 2, 0.0, y[-1]]

    def bounds(self):
        return [-np.inf, 1e-9, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]


register_fit_model('exp_plus_power', lambda options, variable: [ExpPlusPower()], dimension=1)

# Synthetic data: the gap (eV) against the number of bands, with a little noise
rng = np.random.default_rng(0)
X = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0])
Y = 3.2 + 4.0 * np.exp(-X / 80.0) + 0.02 * rng.standard_normal(X.size)


def fit(model):
    """Fit the model to the data: the parameters, the value at infinity and the largest residual."""
    p0 = model.p0(X, Y)
    popt, _ = curve_fit(model, X, Y, p0=p0, bounds=model.bounds(), sigma=model.sigma(X), maxfev=20000)
    residual = np.abs(model(X, *popt) - Y).max()
    asymptote = model.asymptote(popt)
    return popt, (asymptote if asymptote is not None else Y[-1]), residual


results = {}
for name in ('power_law', 'exp_plus_power'):
    for model in get_fit_models(name, dimension=1, variable='BndsRnXp'):
        try:
            results[repr(model)] = fit(model)
        except RuntimeError:  # the fit did not converge
            print(f'{model!r}: the fit failed')

best = min(results, key=lambda key: results[key][2])
for key, (popt, asymptote, residual) in results.items():
    print(f'{key:32s} value at infinity = {asymptote:.4f} eV   largest residual = {residual:.4f} eV')
print(f'The best fit of this data: {best}')

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = np.linspace(X.min(), X.max(), 200)
    plt.plot(X, Y, 'ko', label='data')
    for name in ('power_law', 'exp_plus_power'):
        for model in get_fit_models(name, dimension=1, variable='BndsRnXp'):
            if repr(model) in results:
                plt.plot(grid, model(grid, *results[repr(model)][0]), label=repr(model))
    plt.xlabel('BndsRnXp')
    plt.ylabel('gap (eV)')
    plt.legend()
    plt.savefig('fit_models.png', dpi=120)
    print('plot written: fit_models.png')
except ImportError:
    pass
