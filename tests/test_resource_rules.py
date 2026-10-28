"""Tests of the resource rules of the convergence (pure logic) and of their use through `updater`."""
import pytest

from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.utils.resource_rules import (
    apply_resource_rules,
    current_values,
    rule_matches,
    snapshot_base,
    validate_resource_rules,
)


def make_manager(bands=100, gcut=10):
    manager = YamboInputManager()
    manager.qp.parameters = {'arguments': ['gw0'], 'variables': {
        'BndsRnXp': [[1, bands], ''], 'GbndRnge': [[1, bands], ''], 'NGsBlkXp': [gcut, 'Ry'], 'FFTGvecs': [20, 'Ry']}}
    return manager


RULES = [
    {'when': {'BndsRnXp': {'min': 200}}, 'for': ['qp'], 'use': {'num_machines': 4, 'max_wallclock_seconds': 7200}},
    {'when': {'BndsRnXp': {'max': 199}}, 'use': {'num_machines': 1, 'num_mpiprocs_per_machine': 8,
                                                 'num_cores_per_mpiproc': 2}},
]


@pytest.mark.parametrize('rules, message', [
    ('no', 'list of rules'),
    ([['when']], 'dictionary'),
    ([{'use': {'num_machines': 1}, 'wen': {}}], 'unknown keys'),
    ([{'when': {}}], "'use' is required"),
    ([{'use': {'nodes': 2}}], "unknown keys in 'use'"),
    ([{'use': {'num_machines': 0}}], 'positive integer'),
    ([{'use': {'num_machines': 1}, 'for': ['p2y']}], "'for'"),
    ([{'use': {'parallelism': 'auto'}, 'for': ['scf']}], 'only for the yambo steps'),
    ([{'use': {'parallelism': 'balanced'}}], "'auto' or a dictionary"),
    ([{'use': {'num_machines': 1}, 'when': {'BndsRnXp': {'minimum': 3}}}], "'min' and/or 'max'"),
    ([{'use': {'num_machines': 1}, 'when': {'BndsRnXp': {'min': 5, 'max': 2}}}], "'min' is larger"),
])
def test_validation_rejects(rules, message):
    with pytest.raises(ValueError, match=message):
        validate_resource_rules(rules)


def test_validation_accepts_valid_rules():
    validate_resource_rules(RULES + [{'use': {'parallelism': 'auto', 'queue_name': 'long'}},
                                     {'when': {'kpoint_mesh': [4, 4, 4]}, 'for': ['qp', 'bse'],
                                      'use': {'parallelism': {'X_CPU': '1 1 4 1 1'}, 'max_walltime': 3600}}])


def test_current_values():
    manager = make_manager(bands=150)
    values = current_values(manager, 'qp', ['BndsRnXp', 'NGsBlkXp', 'bands', 'kpoints', 'missing'])
    assert values['BndsRnXp'] == 150
    assert values['NGsBlkXp'] == 10
    assert values['bands'] == 150
    assert values['missing'] is None


def test_point_values_win_over_the_manager():
    manager = make_manager()
    assert current_values(manager, 'qp', ['kpoint_mesh'], {'kpoint_mesh': [[2, 2, 2]]})['kpoint_mesh'] == [2, 2, 2]


@pytest.mark.parametrize('when, values, expected', [
    ({}, {}, True),
    ({'A': {'min': 3}}, {'A': 3}, True),
    ({'A': {'min': 3}}, {'A': 2}, False),
    ({'A': {'max': 3}}, {'A': 3}, True),
    ({'A': {'min': 1, 'max': 3}}, {'A': 4}, False),
    ({'A': 5}, {'A': 5}, True),
    ({'A': [4, 4, 4]}, {'A': [4, 4, 4]}, True),
    ({'A': [4, 4, 4]}, {'A': [2, 2, 2]}, False),
    ({'A': {'min': 1}}, {'A': None}, False),  # an unknown name never matches
    ({'A': {'min': 1}, 'B': 2}, {'A': 5, 'B': 3}, False),  # all the conditions must hold
])
def test_rule_matches(when, values, expected):
    assert rule_matches({'when': when}, values) is expected


def test_first_matching_rule_per_step_is_applied():
    manager = make_manager(bands=300)
    applied = apply_resource_rules(manager, RULES, point_values={}, base=snapshot_base(manager))

    assert applied == {'qp': 0}  # the second rule does not match 300 bands; bse has no rule matching
    assert manager.qp.metadata.options.resources.num_machines == 4
    assert manager.qp.metadata.options.max_wallclock_seconds == 7200


def test_rules_for_other_steps_and_omp():
    manager = make_manager(bands=100)
    applied = apply_resource_rules(manager, RULES, base=snapshot_base(manager))

    assert applied['qp'] == 1 and applied['bse'] == 1
    for step in ('qp', 'bse'):
        options = getattr(manager, step).metadata.options
        assert (options.resources.num_mpiprocs_per_machine, options.resources.num_cores_per_mpiproc) == (8, 2)
        assert 'export OMP_NUM_THREADS=2' in options.prepend_text


def test_nothing_is_left_from_the_previous_point():
    manager = make_manager(bands=300)
    manager.qp.metadata.options.prepend_text = 'module load yambo\n'
    base = snapshot_base(manager)

    apply_resource_rules(manager, RULES, base=base)
    assert manager.qp.metadata.options.max_wallclock_seconds == 7200

    manager.qp.parameters['variables']['BndsRnXp'] = [[1, 50], '']  # the next point matches the other rule
    apply_resource_rules(manager, RULES, base=base)
    options = manager.qp.metadata.options
    assert options.max_wallclock_seconds == 86400  # the default, back
    assert options.resources.num_machines == 1
    assert options.prepend_text.count('OMP_NUM_THREADS') == 1 and 'module load yambo' in options.prepend_text

    manager.qp.parameters['variables']['BndsRnXp'] = [[1, 300], '']
    apply_resource_rules(manager, [{'when': {'BndsRnXp': {'min': 1000}}, 'use': {'num_machines': 9}}], base=base)
    assert manager.qp.metadata.options.resources.num_machines == 1  # no rule matches: the base


def test_parallelism_dictionary_replaces_the_previous_one_and_is_reset():
    manager = make_manager()
    manager.qp.parameters['variables']['X_CPU'] = '1 1 1 1 1'
    base = snapshot_base(manager)
    rules = [{'when': {'BndsRnXp': {'min': 100}}, 'for': ['qp'],
              'use': {'parallelism': {'SE_CPU': '1 2 1', 'SE_ROLEs': 'q qp b'}, 'max_walltime': 3600}}]

    apply_resource_rules(manager, rules, base=base)
    variables = manager.qp.parameters['variables']
    assert variables['SE_CPU'] == '1 2 1' and 'X_CPU' not in variables
    assert manager.qp.max_walltime == 3600

    manager.qp.parameters['variables']['BndsRnXp'] = [[1, 10], '']
    apply_resource_rules(manager, rules, base=base)
    variables = manager.qp.parameters['variables']
    assert 'SE_CPU' not in variables and variables['X_CPU'] == '1 1 1 1 1'
    assert not hasattr(manager.qp, 'max_walltime')


def test_scf_and_nscf_get_resources_but_no_parallelism():
    manager = make_manager()
    rules = [{'use': {'num_machines': 3, 'parallelism': {'X_CPU': '1'}}}]

    apply_resource_rules(manager, rules, base=snapshot_base(manager))

    assert manager.scf.metadata.options.resources.num_machines == 3
    assert manager.nscf.metadata.options.resources.num_machines == 3
    assert 'X_CPU' not in manager.scf.parameters


def test_updater_applies_the_rules_to_each_point(aiida_profile, monkeypatch):
    from types import SimpleNamespace

    from aiida_yambo.workflows.utils import helpers_aiida_yambo as helpers
    from aiida_yambo.workflows.utils.parameters import ConvParam

    monkeypatch.setattr(helpers, 'search_in_group', lambda *args, **kwargs: (False, None, None))
    manager = make_manager(bands=10)
    base = snapshot_base(manager)
    parameter = ConvParam('BndsRnXp', start=100, stop=300, delta=200, max=300, units='')
    evaluator = SimpleNamespace(parameters=parameter)
    values = [[100], [300]]

    for index, nodes in ((0, 1), (1, 4)):
        helpers.updater(manager, evaluator, [100, 300], None, index, None, calc_type='qp',
                        resource_rules=RULES, resources_base=base)
        assert manager.qp.metadata.options.resources.num_machines == nodes, values[index]


def test_the_convergence_port_validates_the_rules(aiida_profile):
    from aiida import orm

    from aiida_yambo.workflows.yamboconvergence import YamboConvergence

    port = YamboConvergence.spec().inputs['resource_rules']
    assert port.validate(orm.List([{'use': {'num_machines': 2}}])) is None
    assert "unknown keys in 'use'" in port.validate(orm.List([{'use': {'nodes': 2}}])).message


def test_rules_add_prepend_text_and_scheduler_commands_to_the_base_without_accumulating():
    manager = make_manager()
    manager.qp.metadata.options.prepend_text = 'module load yambo'
    manager.qp.set_scheduler_commands('#SBATCH --exclusive')
    base = snapshot_base(manager)
    rules = [{'for': ['qp'], 'use': {'prepend_text': 'ulimit -s unlimited', 'custom_scheduler_commands': '#SBATCH -C cpu'}}]

    for _ in range(2):  # two points: the text is not repeated
        apply_resource_rules(manager, rules, base=base)
        options = manager.qp.metadata.options
        assert options.prepend_text == 'module load yambo\nulimit -s unlimited'
        assert options.custom_scheduler_commands == '#SBATCH --exclusive\n#SBATCH -C cpu'
