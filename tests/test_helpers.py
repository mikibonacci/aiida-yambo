"""Helpers shared by the workflows."""
import pytest


@pytest.fixture
def helpers(aiida_profile):
    from aiida_yambo.workflows.utils import helpers_yambowf
    return helpers_yambowf


def _manager(nscf_nbnd):
    from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager

    return YamboInputManager(
        scf={'parameters': {'SYSTEM': {'ecutwfc': 30}}},
        nscf={'parameters': {'SYSTEM': {'ecutwfc': 30, 'nbnd': nscf_nbnd}}},
        qp={'parameters': {'arguments': ['gw0'], 'variables': {'BndsRnXp': [[1, 20], ''], 'GbndRnge': [[1, 30], '']}}},
    )


def test_input_validator_asks_for_more_nscf_bands(helpers):
    scf, nscf, redo_nscf, gwbands, messages = helpers.quantumespresso_input_validator(_manager(nscf_nbnd=10))

    assert gwbands == 30
    assert redo_nscf is True
    assert nscf['SYSTEM']['nbnd'] == 30
    assert scf['SYSTEM']['ecutwfc'] == 30


def test_input_validator_keeps_enough_nscf_bands(helpers):
    _, nscf, redo_nscf, gwbands, _ = helpers.quantumespresso_input_validator(_manager(nscf_nbnd=50))

    assert (gwbands, redo_nscf, nscf['SYSTEM']['nbnd']) == (30, False, 50)
