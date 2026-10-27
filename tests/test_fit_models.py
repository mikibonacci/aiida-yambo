"""Fit models and their registry."""
import numpy as np
import pytest
from scipy.optimize import curve_fit

from aiida_yambo.workflows.utils import fit_models
from aiida_yambo.workflows.utils.fit_models import (
    Exponential1D, FitModel1D, FitModel2D, PowerLaw1D, PowerLawProduct2D, available_fit_models, check_fit_function,
    get_fit_models, register_fit_model, unregister_fit_model,
)


@pytest.mark.parametrize('alpha', [1, 2, 3])
def test_power_law_analytic_derivative_matches_the_numerical_one(alpha):
    model = PowerLaw1D(alpha)
    x = np.array([10., 20., 50., 100.])

    assert model.derivative(x, 3., 5.) == pytest.approx(FitModel1D.derivative(model, x, 3., 5.), rel=1e-3)


def test_power_law_is_what_the_predictor_always_fitted():
    model = PowerLaw1D(2)
    x = np.array([10., 20.])

    assert model(x, 3., 5.) == pytest.approx(3. / x ** 2 + 5.)
    assert model.asymptote([3., 5.]) == 5.
    assert model.sigma(x) == pytest.approx(1 / x)  # 1 / x**(3 - alpha)
    assert model.bounds() == ([-np.inf, -np.inf], [np.inf, np.inf])


def test_default_power_laws_depend_on_the_variable():
    assert [m.alpha for m in get_fit_models('power_law', 1, variable='FFTGvecs')] == [1, 2, 3]
    assert [m.alpha for m in get_fit_models('power_law', 1, variable='kpoint_mesh')] == [1, 2]
    assert [m.alpha for m in get_fit_models('power_law', 1, {'alphas': [2]})] == [2]


def test_default_2d_power_laws():
    models = get_fit_models('power_law', 2)

    assert [(m.alpha, m.beta) for m in models] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert [(m.alpha, m.beta) for m in get_fit_models('power_law', 2, {'alphas': [1], 'betas': [3]})] == [(1, 3)]


def test_2d_power_law_values_and_asymptote():
    model = PowerLawProduct2D(1, 2)
    X = np.array([[10., 20.], [2., 4.]])
    a, b, c, d = 2., 1., 3., 0.5

    assert model(X, a, b, c, d) == pytest.approx((a / X[0] + b) * (c / X[1] ** 2 + d))
    assert model.asymptote([a, b, c, d]) == b * d
    assert model.sigma(X) == pytest.approx(1 / (X[0] * X[1]))


def test_numerical_2d_derivatives_of_a_smooth_function():
    class Smooth(FitModel2D):
        n_parameters = 1

        def __call__(self, X, k):
            return k * X[0] ** 2 * X[1]

    X = np.array([[3., 4.], [5., 6.]])

    fx, fy, fxy = Smooth().derivatives(X, 2.)

    assert fx == pytest.approx(2 * 2. * X[0] * X[1], rel=1e-5)
    assert fy == pytest.approx(2. * X[0] ** 2, rel=1e-5)
    assert fxy == pytest.approx(2 * 2. * X[0], rel=1e-4)


def test_exponential_model_recovers_its_parameters():
    model = Exponential1D()
    x = np.array([10., 15., 20., 25., 30., 40.])
    y = model(x, 2.0, 10.0, 5.0)

    popt, _ = curve_fit(model, x, y, p0=model.p0(x, y), bounds=model.bounds())

    assert popt == pytest.approx([2.0, 10.0, 5.0], rel=1e-4)
    assert model.asymptote(popt) == pytest.approx(5.0, rel=1e-4)
    assert model.derivative(x, *popt) == pytest.approx(FitModel1D.derivative(model, x, *popt), rel=1e-4)


@pytest.fixture
def cleanup():
    registered = []
    yield registered
    for name, dimension in registered:
        unregister_fit_model(name, dimension)


def test_a_registered_model_is_found_by_name(cleanup):
    cleanup.append(('mine', 1))
    register_fit_model('mine', lambda options, variable: [Exponential1D()], dimension=1)

    assert 'mine' in available_fit_models(1)
    assert 'mine' not in available_fit_models(2)
    assert isinstance(get_fit_models('mine', 1)[0], Exponential1D)
    check_fit_function('mine', 1)


def test_unknown_model_error_lists_the_available_ones():
    with pytest.raises(ValueError, match=r"Unknown 1D fit function 'nope'.*power_law"):
        check_fit_function('nope', 1)
    with pytest.raises(ValueError, match='Unknown 2D'):
        check_fit_function('exponential', 2)  # a 1D model only


def test_entry_points_provide_models(monkeypatch):
    class EntryPoint:
        name = 'from_a_package'

        @staticmethod
        def load():
            return lambda options, variable: [Exponential1D()]

    monkeypatch.setattr(fit_models, '_entry_point_factories', lambda dimension: {'from_a_package': EntryPoint} if dimension == 1 else {})

    assert 'from_a_package' in available_fit_models(1)
    assert isinstance(get_fit_models('from_a_package', 1)[0], Exponential1D)


def test_a_factory_without_models_is_an_error(cleanup):
    cleanup.append(('empty', 1))
    register_fit_model('empty', lambda options, variable: [], dimension=1)

    with pytest.raises(ValueError, match='no model'):
        get_fit_models('empty', 1)
