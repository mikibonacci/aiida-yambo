"""ScanParam: an explicit list of parameter sets, with no convergence evaluation."""
import pytest
from pydantic import ValidationError

from aiida_yambo.workflows.utils.parameters import ScanParam, load_parameter_space, create_parameter_space_inputs

POINTS = [
    {'FFTGvecs': 10, 'BndsRnXp': 100, 'NGsBlkXp': 2},
    {'FFTGvecs': 15, 'BndsRnXp': 200, 'NGsBlkXp': 4},
    {'FFTGvecs': 20, 'BndsRnXp': 300, 'NGsBlkXp': 6},
]


def test_from_dicts_builds_the_sets_in_the_order_of_the_variables():
    scan = ScanParam.from_dicts(POINTS)

    assert scan.var == ['FFTGvecs', 'BndsRnXp', 'NGsBlkXp']
    assert scan.build_points() == [[10, 100, 2], [15, 200, 4], [20, 300, 6]]
    assert scan.steps == 3  # one calculation for each set
    assert scan.convergence_algorithm == 'scan'


def test_units_and_mirrors_have_defaults_for_the_usual_variables():
    scan = ScanParam.from_dicts(POINTS)

    assert scan.units == ['Ry', '', 'Ry']
    assert scan.mirror_values == {'BndsRnXp': ['GbndRnge']}


def test_units_and_mirrors_can_be_given():
    scan = ScanParam.from_dicts(POINTS, units={'NGsBlkXp': 'RL'}, mirror_values={})

    assert scan.units == ['Ry', '', 'RL']
    assert scan.mirror_values == {}


def test_a_mirrored_variable_in_the_scan_is_not_mirrored_again():
    scan = ScanParam.from_dicts([{'BndsRnXp': 100, 'GbndRnge': 150}, {'BndsRnXp': 200, 'GbndRnge': 250}])

    assert scan.mirror_values == {}


def test_max_of_each_variable():
    scan = ScanParam.from_dicts(POINTS + [{'FFTGvecs': 5, 'BndsRnXp': 500, 'NGsBlkXp': 1}])

    assert scan.max == [20, 500, 6]


def test_k_points_meshes_have_no_scalar_maximum_and_are_accepted():
    scan = ScanParam.from_dicts([{'FFTGvecs': 10, 'kpoint_mesh': [4, 4, 4]}, {'FFTGvecs': 10, 'kpoint_mesh': [6, 6, 6]}])

    assert scan.max == [10, None]
    assert scan.units == ['Ry', 'mesh']


def test_the_space_has_no_bounds_to_exceed():
    assert ScanParam.from_dicts(POINTS).check_bounds()[0] is True


@pytest.mark.parametrize('points, message', [
    ([], 'at least one point'),
    ([{'FFTGvecs': 10}, {'FFTGvecs': 10, 'BndsRnXp': 100}], 'same variables'),
    ([{'FFTGvecs': float('nan')}], 'finite'),
    ([{'FFTGvecs': 'ten'}], 'finite'),
    ([{'FFTGvecs': True}], 'finite'),
])
def test_invalid_sets_are_rejected(points, message):
    with pytest.raises(ValueError, match=message):
        ScanParam.from_dicts(points)


@pytest.mark.parametrize('kwargs', [
    {'var': ['a', 'a'], 'points': [[1, 2]]},  # repeated variable
    {'var': ['a', 'b'], 'points': [[1]]},  # a point with a missing value
    {'var': ['a'], 'points': [[1]], 'units': ['Ry', 'Ry']},  # units not matching the variables
    {'var': [], 'points': [[]]},
])
def test_invalid_scan_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        ScanParam(**kwargs)


def test_dict_round_trip_through_the_loader():
    scan = ScanParam.from_dicts(POINTS, max_concurrent_steps=2)

    raw = create_parameter_space_inputs([scan])
    (loaded,) = load_parameter_space(raw)

    assert isinstance(loaded, ScanParam)
    assert loaded == scan
    assert loaded.max_concurrent_steps == 2


def test_a_scan_of_a_single_variable_is_still_a_scan():
    (loaded,) = load_parameter_space(create_parameter_space_inputs([ScanParam.from_dicts([{'FFTGvecs': 10}, {'FFTGvecs': 15}])]))

    assert isinstance(loaded, ScanParam)
    assert loaded.var == ['FFTGvecs']


def test_input_manager_keeps_the_scan():
    from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager

    scan = ScanParam.from_dicts(POINTS)
    manager = YamboInputManager(parameters_space=create_parameter_space_inputs([scan]))

    assert manager.parameters_space == [scan]
