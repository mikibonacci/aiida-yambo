"""The fit function is a parameter of the convergence (by name), validated when the parameter is defined."""
import pytest
from pydantic import ValidationError

from aiida_yambo.workflows.utils.convergence import ConvergenceAnalyzer
from aiida_yambo.workflows.utils.fit_models import FitModel1D, FitModel2D, register_fit_model, unregister_fit_model
from aiida_yambo.workflows.utils.parameters import (
    ConvParam, CoupledConvParam, KpointMeshConvParam, create_parameter_space_inputs, load_parameter_space,
)

import numpy as np
import pandas as pd


def _conv(**kwargs):
    return ConvParam('FFTGvecs', start=10, stop=25, delta=5, max=60, units='Ry', conv_thr=1.0, steps=4, **kwargs)


def _coupled(**kwargs):
    return CoupledConvParam(var=['BndsRnXp', 'NGsBlkXp'], start=[100, 2], stop=[300, 6], delta=[100, 2], max=[1000, 20],
                            units=['', 'Ry'], **kwargs)


def test_parameters_use_the_power_law_by_default():
    assert (_conv().fit_function, _conv().fit_options) == ('power_law', {})
    assert _coupled().fit_function == 'power_law'


def test_parameters_without_fit_function_load_as_before():
    """The parameters stored by older workflows have no fit_function."""
    raw = create_parameter_space_inputs([_conv()])
    del raw[0]['fit_function'], raw[0]['fit_options']

    (loaded,) = load_parameter_space(raw)

    assert loaded.fit_function == 'power_law'


@pytest.mark.parametrize('build', [
    lambda: _conv(fit_function='nope'),
    lambda: _coupled(fit_function='exponential'),  # a 1D function
    lambda: KpointMeshConvParam(var='kpoint_density', start=1.0, stop=.25, delta=-.25, max=1.0, units='1/A',
                                fit_function='nope'),
])
def test_an_unknown_fit_function_is_refused_when_the_parameter_is_defined(build):
    with pytest.raises(ValidationError, match='Unknown'):
        build()


def test_fit_function_and_options_survive_the_round_trip():
    parameters = _conv(fit_function='exponential')
    options = _coupled(fit_options={'alphas': [1], 'betas': [2]})

    loaded = load_parameter_space(create_parameter_space_inputs([parameters, options]))

    assert loaded == [parameters, options]
    assert loaded[1].fit_options == {'alphas': [1], 'betas': [2]}


@pytest.fixture
def cleanup():
    registered = []
    yield registered
    for name, dimension in registered:
        unregister_fit_model(name, dimension)


class _Saturation(FitModel1D):
    """y = b - a / (x + L), the asymptote is b."""

    n_parameters = 3

    def __call__(self, x, a, L, b):
        return b - a / (x + L)

    def asymptote(self, popt):
        return popt[2]

    def p0(self, x, y):
        return [1.0, 1.0, float(np.max(y))]


def _data(f):
    x = np.array([10., 15., 20., 25.])
    return pd.DataFrame({'FFTGvecs': [10, 15, 20, 25], 'pk': [1, 2, 3, 4], 'uuid': list('abcd'),
                         'failed': [False] * 4, 'gap_': f(x)})


def _predictor(parameters, frame):
    """The predictor the analysis uses, to look at what it fitted."""
    from aiida_yambo.workflows.utils.predictor_1D import The_Predictor_1D
    from aiida_yambo.workflows.utils.predictor_2D import The_Predictor_2D

    calc_dict = parameters.model_dump()
    calc_dict['delta'] = calc_dict['delta'] if isinstance(calc_dict['delta'], list) else [calc_dict['delta']]
    grid = {name: frame[name].tolist() for name in parameters.var}
    predictor_class = The_Predictor_2D if len(parameters.var) == 2 else The_Predictor_1D
    return predictor_class(grid=grid, result=frame, bande=0, r=frame['gap_'].to_numpy(), calc_dict=calc_dict, k='gap_',
                           what='gap_')


def test_a_custom_model_is_used_by_the_predictor(cleanup):
    """Data following a custom law are fitted by it: its asymptote is found."""
    cleanup.append(('saturation', 1))
    register_fit_model('saturation', lambda options, variable: [_Saturation()], dimension=1)
    parameters = _conv(fit_function='saturation')
    frame = _data(lambda x: 5.0 - 2.0 / (x + 30.0))

    predictor = _predictor(parameters, frame)
    predictor.analyse()

    assert predictor.extra == pytest.approx(5.0, abs=1e-3)
    assert ConvergenceAnalyzer(parameters, 'gap_').analyze_dataframe(frame)['status'] in ('CONTINUE', 'CONVERGED')


def test_the_power_law_does_not_find_the_asymptote_of_other_data(cleanup):
    """Control of the previous test: the default model gives another extrapolation."""
    frame = _data(lambda x: 5.0 - 2.0 / (x + 30.0))

    predictor = _predictor(_conv(), frame)
    predictor.analyse()

    assert predictor.extra != pytest.approx(5.0, abs=1e-3)


def test_the_options_of_the_default_model_select_the_exponents():
    frame = _data(lambda x: 5 + 0.5 / x)

    predictor = _predictor(_conv(fit_options={'alphas': [1]}), frame)
    predictor.analyse()

    assert predictor.extra == pytest.approx(5.0, abs=1e-6)


def test_a_custom_2d_model_is_used_by_the_predictor(cleanup):
    class Plane(FitModel2D):
        n_parameters = 3

        def __call__(self, X, a, b, c):
            return a + b / X[0] + c / X[1]

        def asymptote(self, popt):
            return popt[0]

    cleanup.append(('plane', 2))
    register_fit_model('plane', lambda options, variable: [Plane()], dimension=2)
    parameters = _coupled(fit_function='plane', conv_thr=1.0)
    points = parameters.build_diagonal_corners()
    frame = pd.DataFrame({'BndsRnXp': [p[0] for p in points], 'NGsBlkXp': [p[1] for p in points],
                          'GbndRnge': [p[0] for p in points], 'pk': range(1, 7), 'uuid': list('abcdef'),
                          'failed': [False] * 6, 'gap_': [2 + 20. / p[0] + 0.02 / p[1] for p in points]})

    predictor = _predictor(parameters, frame)
    predictor.analyse()

    assert predictor.extra == pytest.approx(2.0, abs=1e-3)
