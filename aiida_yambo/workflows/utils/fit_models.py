"""Fit models used by the predictors of the convergence workflow.

A *fit model* is the function which is fitted to the results of a convergence (1D) or of a coupled
convergence (2D), together with what the predictors need to know about it: the weights of the fit, the
derivatives (to look for the plateau), and the value at infinity (the "extra" value).

Models are referred to *by name* in the convergence parameters (``fit_function`` and ``fit_options``), so
that the provenance stores only plain data. A name resolves, in this order, to:

1. a model registered with :func:`register_fit_model` (scripts, tests);
2. an entry point of the group ``aiida_yambo.fit_models_1d`` or ``aiida_yambo.fit_models_2d``
   (models of other packages: the daemon finds them if the package is installed in its environment);
3. the built-in models: ``power_law`` and ``exponential`` (1D only).

A registered object is a *factory*: ``factory(options, variable) -> list of models``, where ``options`` is the
``fit_options`` dictionary and ``variable`` the name of the converged variable (e.g. 'kpoint_mesh'). The
predictor fits all the models of the list and keeps the one with the smallest error (e.g. the power laws with
different exponents).

Example of a 1D model::

    class Exponential(FitModel1D):
        n_parameters = 3                                  # a, L, b
        def __call__(self, x, a, L, b):
            return a * np.exp(-x / L) + b
        def asymptote(self, popt):
            return popt[2]

    register_fit_model('my_exponential', lambda options, variable: [Exponential()], dimension=1)
"""
from importlib.metadata import entry_points

import numpy as np

ENTRY_POINT_GROUPS = {1: 'aiida_yambo.fit_models_1d', 2: 'aiida_yambo.fit_models_2d'}

_REGISTRY = {1: {}, 2: {}}


class FitModel1D:
    """A function ``y = f(x, *p)`` fitted to the results of a 1D convergence."""

    #: number of parameters of the function
    n_parameters = 2

    def __call__(self, x, *p):
        raise NotImplementedError

    def derivative(self, x, *p):
        """df/dx: used to look for the plateau. By default, numerical."""
        x = np.asarray(x, dtype=float)
        h = 1e-6 * np.maximum(np.abs(x), 1.0)
        return (self(x + h, *p) - self(x - h, *p)) / (2 * h)

    def asymptote(self, popt):
        """The value of the function for x -> infinity, or None if there is none (the last value is then used)."""
        return None

    def p0(self, x, y):
        """Starting parameters of the fit (None: the default of scipy)."""
        return None

    def bounds(self):
        """Bounds of the parameters of the fit."""
        return [-np.inf] * self.n_parameters, [np.inf] * self.n_parameters

    def sigma(self, x):
        """Uncertainties of the data, i.e. the weights of the fit (None: all the same)."""
        return None

    def __repr__(self):
        return f'{type(self).__name__}()'


class FitModel2D:
    """A function ``z = f((x, y), *p)`` fitted to the results of a coupled (2D) convergence.

    ``X`` is an array of shape (2, N): the first row are the values of the first variable, the second of the second.
    """

    n_parameters = 4

    def __call__(self, X, *p):
        raise NotImplementedError

    def derivatives(self, X, *p):
        """(df/dx, df/dy, d2f/dxdy) on the points X (shape (2, ...)): used to look for the plateau. By default, numerical."""
        X = np.asarray(X, dtype=float)
        hx = 1e-6 * np.maximum(np.abs(X[0]), 1.0)
        hy = 1e-6 * np.maximum(np.abs(X[1]), 1.0)

        def shifted(dx, dy):
            return self(np.array((X[0] + dx, X[1] + dy)), *p)

        fx = (shifted(hx, 0) - shifted(-hx, 0)) / (2 * hx)
        fy = (shifted(0, hy) - shifted(0, -hy)) / (2 * hy)
        fxy = (shifted(hx, hy) - shifted(hx, -hy) - shifted(-hx, hy) + shifted(-hx, -hy)) / (4 * hx * hy)
        return fx, fy, fxy

    def asymptote(self, popt):
        """The value of the function for x, y -> infinity, or None if there is none (the last value is then used)."""
        return None

    def p0(self, X, z):
        return None

    def bounds(self):
        return [-np.inf] * self.n_parameters, [np.inf] * self.n_parameters

    def sigma(self, X):
        return None

    def gradient_angle(self, popt, last_value):
        """Sign-like quantity used by the 2D predictor to orient the plateau (see predictor_2D)."""
        return 1.0

    def __repr__(self):
        return f'{type(self).__name__}()'


############################### built-in models ###############################

class PowerLaw1D(FitModel1D):
    """``a / x**alpha + b``, the asymptote is ``b``. The weights give more importance to the high values."""

    n_parameters = 2

    def __init__(self, alpha=1):
        self.alpha = alpha

    def __call__(self, x, a, b):
        return a / x ** self.alpha + b

    def derivative(self, x, a, b=None):
        return -self.alpha * a / (x ** (self.alpha + 1))

    def asymptote(self, popt):
        return popt[1]

    def sigma(self, x):
        return 1 / x ** (3 - self.alpha)

    def __repr__(self):
        return f'PowerLaw1D(alpha={self.alpha})'


class Exponential1D(FitModel1D):
    """``a * exp(-x / L) + b``, the asymptote is ``b``."""

    n_parameters = 3

    def __call__(self, x, a, L, b):
        return a * np.exp(-x / L) + b

    def derivative(self, x, a, L, b=None):
        return -a / L * np.exp(-x / L)

    def asymptote(self, popt):
        return popt[2]

    def p0(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return [y[0] - y[-1] if y[0] != y[-1] else 1.0, (np.max(x) - np.min(x)) / 2 or 1.0, y[-1]]

    def bounds(self):
        return [-np.inf, 1e-12, -np.inf], [np.inf, np.inf, np.inf]


class PowerLawProduct2D(FitModel2D):
    """``(a / x**alpha + b) * (c / y**beta + d)``, the asymptote is ``b * d``.

    NOTE: the derivatives are kept as they have always been in the predictor: ``fx`` assumes beta = 1 in the
    factor of y and ``fy`` alpha = 1 in the factor of x. They are used only to look for the plateau.
    """

    n_parameters = 4

    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, X, a, b, c, d):
        return (a / X[0] ** self.alpha + b) * (c / X[1] ** self.beta + d)

    def derivatives(self, X, a, b, c, d):
        alpha, beta = self.alpha, self.beta
        fx = -alpha * a / (X[0] ** (alpha + 1)) * (c / X[1] + d)
        fy = (a / X[0] + b) * (-beta * c / X[1] ** (beta + 1))
        fxy = -alpha * a / (X[0] ** (alpha + 1)) * (-beta * c / X[1] ** (beta + 1))
        return fx, fy, fxy

    def asymptote(self, popt):
        return popt[1] * popt[3]

    def sigma(self, X):
        return 1 / (X[0] * X[1])

    def gradient_angle(self, popt, last_value):
        return popt[0] * popt[2] * np.sign(last_value)

    def __repr__(self):
        return f'PowerLawProduct2D(alpha={self.alpha}, beta={self.beta})'


def _power_law_1d(options, variable):
    alphas = options.get('alphas') or ([1, 2] if variable == 'kpoint_mesh' else [1, 2, 3])
    return [PowerLaw1D(alpha) for alpha in alphas]


def _power_law_2d(options, variable):
    alphas = options.get('alphas') or [1, 2]
    betas = options.get('betas') or [1, 2]
    return [PowerLawProduct2D(alpha, beta) for alpha in alphas for beta in betas]


def _exponential_1d(options, variable):
    return [Exponential1D()]


_BUILTIN = {
    1: {'power_law': _power_law_1d, 'exponential': _exponential_1d},
    2: {'power_law': _power_law_2d},
}

DEFAULT_FIT_FUNCTION = 'power_law'


################################### registry ##################################

def register_fit_model(name, factory, dimension=1):
    """Register a factory ``factory(options, variable) -> list of models`` under a name (see the module docstring)."""
    if dimension not in _REGISTRY:
        raise ValueError(f'dimension must be 1 or 2, got {dimension!r}')
    _REGISTRY[dimension][name] = factory


def unregister_fit_model(name, dimension=1):
    _REGISTRY[dimension].pop(name, None)


def _entry_point_factories(dimension):
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUPS[dimension])}


def available_fit_models(dimension=1):
    """Names of the fit models for the given dimension."""
    return sorted({*_BUILTIN[dimension], *_REGISTRY[dimension], *_entry_point_factories(dimension)})


def _factory(name, dimension):
    if name in _REGISTRY[dimension]:
        return _REGISTRY[dimension][name]
    entry_points_ = _entry_point_factories(dimension)
    if name in entry_points_:
        return entry_points_[name].load()
    if name in _BUILTIN[dimension]:
        return _BUILTIN[dimension][name]
    raise ValueError(
        f"Unknown {dimension}D fit function {name!r}. Available: {available_fit_models(dimension)}. "
        f"A model of another package must be installed (with its entry point in the group "
        f"'{ENTRY_POINT_GROUPS[dimension]}') in the environment where this is evaluated, daemon included."
    )


def get_fit_models(name=DEFAULT_FIT_FUNCTION, dimension=1, options=None, variable=None):
    """The models (to be compared by the predictor) of the fit function `name`."""
    models = _factory(name, dimension)(dict(options or {}), variable)
    models = list(models)
    if not models:
        raise ValueError(f'The fit function {name!r} gave no model to fit.')
    return models


def check_fit_function(name, dimension=1, options=None, variable=None):
    """Raise a ValueError if the fit function cannot be resolved (used when the parameters are defined)."""
    get_fit_models(name, dimension, options, variable)
