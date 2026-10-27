"""Synthetic convergence data, to compare the output of the predictors before/after refactoring them.

The scales are realistic (a GW gap in eV changing by ~0.05 eV over the grid) so that the plateau conditions of the
predictors (|f'| < thr_fx...) are met inside the extrapolation range.
"""
import math
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd

from aiida_yambo.workflows.utils.convergence import ConvergenceAnalyzer
from aiida_yambo.workflows.utils.parameters import ConvParam, CoupledConvParam, KpointMeshConvParam


def _frame(columns, ys, what='gap_'):
    n = len(ys)
    data = dict(columns)
    data.update({'pk': list(range(1, n + 1)), 'uuid': [f'u{i}' for i in range(1, n + 1)],
                 'failed': [False] * n, what: list(ys)})
    return pd.DataFrame(data)


def _fft(f, thr=1.0):
    x = np.array([10, 15, 20, 25], dtype=float)
    parameters = ConvParam('FFTGvecs', start=10, stop=25, delta=5, max=60, units='Ry', conv_thr=thr, steps=4)
    return parameters, _frame({'FFTGvecs': [10, 15, 20, 25]}, [f(v) for v in x]), None


def _bands():
    x = np.array([100, 200, 300, 400.])
    parameters = ConvParam('BndsRnXp', start=100, stop=400, delta=100, max=1000, units='', conv_thr=1.0, steps=4,
                           mirror_values=['GbndRnge'])
    return parameters, _frame({'BndsRnXp': [100, 200, 300, 400], 'GbndRnge': [100, 200, 300, 400]}, list(3 + 5 / x)), None


def _kdensity():
    d = np.array([1.0, .75, .5, .25])
    parameters = KpointMeshConvParam(var='kpoint_density', start=1.0, stop=.25, delta=-.25, max=1.0, units='1/A',
                                     conv_thr=1.0, steps=4)
    return parameters, _frame({'kpoint_density': list(d)}, list(4 + 0.5 / (1 / d) ** 3)), None


def _kmesh():
    meshes = [[4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10]]
    parameters = KpointMeshConvParam(start=[4, 4, 4], stop=[10, 10, 10], delta=[2, 2, 2], max=[30, 30, 30], units='mesh',
                                     conv_thr=1.0, steps=4)
    return parameters, _frame({'kpoint_mesh': meshes}, [4 + 20 / (m[0] ** 3) for m in meshes]), None


def _coupled(a, b, c, d, alpha, beta, thr=1.0):
    parameters = CoupledConvParam(var=['BndsRnXp', 'NGsBlkXp'], start=[100, 2], stop=[300, 6], delta=[100, 2],
                                  max=[1000, 20], units=['', 'Ry'], conv_thr=thr, mirror_values={'BndsRnXp': ['GbndRnge']})
    points = parameters.build_diagonal_corners()
    ys = [(a / x ** alpha + b) * (c / y ** beta + d) for x, y in points]
    return parameters, _frame({'BndsRnXp': [p[0] for p in points], 'NGsBlkXp': [p[1] for p in points],
                               'GbndRnge': [p[0] for p in points]}, ys), None


def _second_pass():
    """The point predicted by the first analysis is computed: the analysis verifies it."""
    parameters, frame, _ = _fft(lambda x: 5 + 0.5 / x)
    hints = run(parameters, frame, None)['hints']
    new = int(round(hints['FFTGvecs']))
    extra = _frame({'FFTGvecs': [new]}, [5 + 0.5 / new])
    frame = pd.concat([frame, extra], ignore_index=True)
    frame['pk'] = range(1, len(frame) + 1)
    frame['uuid'] = [f'u{i}' for i in frame['pk']]
    return parameters, frame, hints


def scenarios():
    return {
        'fft_power1': lambda: _fft(lambda x: 5 + 0.5 / x),
        'fft_power2': lambda: _fft(lambda x: 5 + 5 / x ** 2),
        'fft_power3_noise': lambda: _fft(lambda x: 5 + 100 / x ** 3 + 1e-5 * np.sin(x)),
        'fft_unconverged': lambda: _fft(lambda x: 5 + 8 / x, thr=0.5),
        'fft_tight_threshold': lambda: _fft(lambda x: 5 + 0.5 / x, thr=0.01),
        'fft_wide_threshold': lambda: _fft(lambda x: 5 + 0.5 / x, thr=5.0),
        'fft_second_pass': _second_pass,
        'bands_power1': _bands,
        'kdensity': _kdensity,
        'kmesh': _kmesh,
        'coupled_power1': lambda: _coupled(20., 2., 0.02, 1., 1, 1),
        'coupled_power2': lambda: _coupled(400., 2., 0.2, 1., 2, 2),
        'coupled_unconverged': lambda: _coupled(300., 2., 0.5, 1., 1, 1, thr=0.01),
    }


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else round(value, 6)
    return value


@contextmanager
def fake_pw_parent():
    """The 1D predictor reads the number of k-points of the PW calculations: we have no calculations."""
    from aiida_yambo.workflows.utils import predictor_1D

    parent = SimpleNamespace(outputs=SimpleNamespace(output_parameters=SimpleNamespace(
        get_dict=lambda: {'number_of_k_points': 10})))
    original = predictor_1D.load_node, predictor_1D.find_pw_parent
    predictor_1D.load_node, predictor_1D.find_pw_parent = (lambda uuid: uuid), (lambda node: parent)
    try:
        yield
    finally:
        predictor_1D.load_node, predictor_1D.find_pw_parent = original


def run(parameters, frame, hints, what='gap_'):
    """What the workflow gets from the analysis of the data, as plain python data."""
    analyzer = ConvergenceAnalyzer(parameters, what, hints=hints)
    with fake_pw_parent():
        decision = analyzer.analyze_dataframe(frame)
    return _clean({'status': decision['status'], 'new_grid': decision.get('new_grid'), 'hints': decision['hints'],
                   'check_single_point': analyzer.check_single_point, 'inquired_point': analyzer.inquired_point})


def run_all():
    results = {}
    for name, build in scenarios().items():
        parameters, frame, hints = build()
        results[name] = run(parameters, frame, hints)
    return results
