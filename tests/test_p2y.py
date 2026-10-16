"""Small, deterministic tests for the Yambo CalcJob specification.

Running a real p2y process belongs in an integration test with a Yambo
executable and retrieved fixtures.  The old test was an unfinished template
and prevented the suite from being a reliable smoke test.
"""

from aiida_yambo.calculations.yambo import YamboCalculation


def test_yambo_spec_declares_p2y_inputs():
    ports = YamboCalculation.spec().inputs
    assert 'preprocessing_code' in ports
    assert 'precode_parameters' in ports
    assert ports['settings'].valid_type is not None
