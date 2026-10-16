"""Pure-data tests for convergence analysis state transitions."""

import pandas as pd
import pytest

from aiida_yambo.workflows.utils.convergence import ConvergenceAnalyzer
from aiida_yambo.workflows.utils.parameters import ConvParam, CoupledConvParam


def test_empty_dataframe_is_rejected():
    analyzer = ConvergenceAnalyzer(ConvParam('FFTGvecs', start=1, stop=3, delta=1, max=5, units='Ry'), 'gap_eV')
    with pytest.raises(ValueError, match='No dataframe'):
        analyzer.analyze_dataframe(pd.DataFrame())


def test_failed_rows_stop_analysis_before_predictor():
    analyzer = ConvergenceAnalyzer(ConvParam('FFTGvecs', start=1, stop=3, delta=1, max=5, units='Ry'), 'gap_eV')
    frame = pd.DataFrame([{'FFTGvecs': 2, 'gap_eV': None, 'pk': 1, 'uuid': 'u', 'failed': True}])
    result = analyzer.analyze_dataframe(frame)
    assert result['status'] == 'FAILED_CALCULATION'
    assert analyzer.last_succeeded_pk is None


def test_space_bounds_are_reported():
    analyzer = ConvergenceAnalyzer(ConvParam('FFTGvecs', start=1, stop=6, delta=1, max=5, units='Ry'), 'gap_eV')
    frame = pd.DataFrame([{'FFTGvecs': 2, 'gap_eV': 1.0, 'pk': 1, 'uuid': 'u', 'failed': False}])
    result = analyzer.analyze_dataframe(frame)
    assert result['status'] == 'SPACE_EXCEEDED'


def test_missing_parameter_rows_are_filtered():
    parameter = CoupledConvParam(var=['BndsRnXp', 'NGsBlkXp'], start=[1, 1], stop=[3, 3], delta=[1, 1], max=[5, 5], units=['', 'Ry'])
    analyzer = ConvergenceAnalyzer(parameter, 'gap_eV')
    frame = pd.DataFrame([
        {'BndsRnXp': 2, 'NGsBlkXp': 2, 'gap_eV': 1.0, 'pk': 1, 'uuid': 'u1', 'failed': False},
        {'BndsRnXp': None, 'NGsBlkXp': 2, 'gap_eV': 2.0, 'pk': 2, 'uuid': 'u2', 'failed': False},
    ])
    analyzer.history = frame
    result = analyzer.analyze_dataframe(frame)
    assert len(analyzer.history) == 1
    assert result['status'] in {'CONTINUE', 'CONVERGED'}
