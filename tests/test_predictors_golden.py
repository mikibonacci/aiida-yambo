"""The predictors give the same answers they gave before they were refactored (synthetic data, see predictor_scenarios).

To regenerate the reference (only if the change of the results is intended):
    REGEN_PREDICTOR_GOLDEN=1 pytest tests/test_predictors_golden.py -k regenerate
"""
import json
import os
import pathlib

import pytest

from predictor_scenarios import run_all, scenarios

GOLDEN = pathlib.Path(__file__).parent / 'data' / 'predictor_golden.json'


def _approx(value):
    if isinstance(value, dict):
        return {k: _approx(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_approx(v) for v in value]
    if isinstance(value, float):
        return pytest.approx(value, abs=2e-6)
    return value


@pytest.mark.skipif(not os.environ.get('REGEN_PREDICTOR_GOLDEN'), reason='only to regenerate the reference')
def test_regenerate_the_reference(aiida_profile):
    GOLDEN.write_text(json.dumps(run_all(), indent=1, sort_keys=True) + '\n')


@pytest.fixture(scope='module')
def results(aiida_profile):
    return run_all()


@pytest.mark.parametrize('name', sorted(scenarios()))
def test_the_predictors_do_not_change_their_answer(results, name):
    expected = json.loads(GOLDEN.read_text())[name]

    assert results[name] == _approx(expected)


def test_the_scenarios_cover_the_decisions_of_the_workflow(results):
    statuses = {(r['status'], r['check_single_point'], r['new_grid']) for r in results.values()}

    assert ('CONVERGED', True, False) in statuses  # the predicted point is verified
    assert ('CONTINUE', False, True) in statuses  # a new grid is needed
    assert ('CONTINUE', True, False) in statuses  # a point to compute
