"""Regression checks for input editing and AiiDA conversion boundaries."""
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager, YamboRestartInputManager
from aiida_yambo.workflows.utils.parameters import (
    ConvParam, CoupledConvParam, KpointMeshConvParam, load_parameter_space,
)

pytest_plugins = ['aiida.tools.pytest_fixtures']


def test_yambo_setters():
    manager = YamboInputManager(
        qp={'parameters': {'arguments': ['gw0'], 'variables': {'NGsBlkXp': [2, 'Ry']}}},
        bse={'parameters': {'arguments': ['bse'], 'variables': {'NGsBlkXs': [2, 'Ry'], 'BSENGBlk': [2, 'Ry']}}},
    )
    manager.set_Gcut(value=6, units='Ry')
    assert manager.qp.get_Gcut() == 6
    assert manager.bse.parameters['variables'] == {'NGsBlkXs': [6, 'Ry'], 'BSENGBlk': [6, 'Ry']}
    manager.bse.set_BSEGcut(value=8, units='RL')
    assert manager.bse.parameters['variables']['BSENGBlk'] == [8, 'RL']
    manager.remove_arguments(['gw0', 'missing'])
    assert manager.qp.parameters['arguments'] == []
    assert manager.bse.parameters['arguments'] == ['bse']
    manager.bse.set_BSEbands([2, 8])
    assert manager.get_BSEbands() == [[2, 8], '']
    # QP-only workflows must not acquire a BSE namespace when changing a cutoff.
    manager.bse = YamboRestartInputManager()
    manager.set_Gcut(value=10)
    assert manager.bse.parameters == {}


def test_disable_soc():
    manager = YamboInputManager(scf={'parameters': {'SYSTEM': {}}}, nscf={'parameters': {'SYSTEM': {}}})
    manager.set_SOC(True)
    manager.set_SOC(False)
    for pw in (manager.scf, manager.nscf):
        assert pw.parameters['SYSTEM'] == {'noncolin': False, 'lspinorb': False}


@pytest.mark.parametrize('change', [
    {'steps': 0}, {'max_iterations': -1}, {'max_concurrent_steps': 0},
    {'var': []}, {'var': ['a', 'b']}, {'delta': 0}, {'delta': -1},
    {'start': 5}, {'stop': float('nan')},
])
def test_invalid_scalar_space(change):
    kwargs = dict(var='FFTGvecs', start=1, stop=3, delta=1, max=5, units='Ry')
    kwargs.update(change)
    with pytest.raises(ValidationError):
        ConvParam(**kwargs)


@pytest.mark.parametrize('change', [
    {'units': ['Ry']}, {'steps': 0}, {'points': 3}, {'delta': [1, 0]},
    {'var': ['a', 'a']}, {'stop': [0, 4]},
])
def test_invalid_coupled_space(change):
    kwargs = dict(var=['a', 'b'], start=[1, 2], stop=[3, 4], delta=[1, 1], max=[5, 6], units=['', 'Ry'])
    kwargs.update(change)
    with pytest.raises(ValidationError):
        CoupledConvParam(**kwargs)


def test_kpoint_spaces_and_round_trip():
    density = KpointMeshConvParam(var='kpoint_density', start=.8, stop=.2, delta=-.2, max=.8, units='1/A')
    assert density.build_stepped_space() == pytest.approx([.8, .6, .4, .2])
    rebuilt = load_parameter_space([density.to_dict()])[0]
    assert isinstance(rebuilt, KpointMeshConvParam)
    assert rebuilt.build_stepped_space() == density.build_stepped_space()
    mesh = KpointMeshConvParam(start=[2, 2, 1], stop=[8, 8, 1], delta=[2, 2, 0], max=[12, 12, 1], units='mesh')
    assert mesh.build_linear_space() == [[2, 4, 6, 8], [2, 4, 6, 8], [1]]
    assert mesh.build_stepped_space()[-1] == [8, 8, 1]
    for change in ({'delta': .2}, {'stop': 0}, {'start': .1}, {'max': float('inf')}):
        with pytest.raises(ValidationError):
            KpointMeshConvParam(**{**density.to_dict(), **change})
    # Exceeding a convergence cap remains a runtime stopping condition.
    assert not ConvParam('x', start=1, stop=6, delta=1, max=5, units='').check_bounds()[0]


def test_kpoint_density_min_is_the_finest_distance_allowed():
    kwargs = dict(var='kpoint_density', start=.8, stop=.4, delta=-.2, max=.8, units='1/A')
    density = KpointMeshConvParam(min=.3, **kwargs)

    assert density.check_bounds()[0] and density.build_stepped_space() == pytest.approx([.8, .6667, .5333, .4], abs=1e-3)
    # a window shifted finer than `min` is refused, as a grid beyond `max` is (SPACE_TOO_SMALL in the workflow)
    shifted = density.with_shift(-.2, -.2)
    assert shifted.min == .3 and shifted.build_stepped_space() == ['space_exceeds_max']
    assert shifted.check_bounds()[0] is False
    assert all(x >= .3 for x in KpointMeshConvParam(min=.3, **{**kwargs, 'stop': .3}).build_linear_space())
    assert load_parameter_space([density.to_dict()])[0].min == .3
    # no `min`: only positivity limits it, as before
    assert KpointMeshConvParam(**kwargs).min is None
    assert KpointMeshConvParam(min=.5, **kwargs).check_bounds()[0] is False  # the grid itself is finer than `min`
    for change in ({'min': 0}, {'min': -1}):  # not positive
        with pytest.raises(ValidationError):
            KpointMeshConvParam(**{**kwargs, **change})
    with pytest.raises(ValidationError):  # for a mesh the limit is `max`
        KpointMeshConvParam(start=[2, 2, 2], stop=[4, 4, 4], delta=[1, 1, 1], max=[8, 8, 8], min=1.0, units='mesh')


@pytest.fixture
def restart_inputs(aiida_profile, aiida_localhost):
    from aiida import orm
    from aiida.common import AttributeDict
    code = orm.InstalledCode(label=f'test-yambo-{uuid4().hex}', computer=aiida_localhost, filepath_executable='/bin/true').store()
    parent = orm.RemoteData(computer=aiida_localhost, remote_path='/tmp/parent').store()
    return AttributeDict(dict(
        code=code, parent_folder=parent,
        parameters=orm.Dict({'arguments': ['gw0'], 'variables': {'NGsBlkXp': [2, 'Ry'], 'X_CPU': '1 2'}}),
        settings=orm.Dict({'INITIALISE': False}),
        metadata={'label': 'calculation', 'options': {
            'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 4, 'num_cores_per_mpiproc': 1},
            'max_wallclock_seconds': 100, 'queue_name': 'test', 'prepend_text': 'module load yambo',
        }},
    ))


@pytest.mark.parametrize('what', ['YamboRestart', 'YamboCalculation'])
def test_standalone_round_trip(restart_inputs, what):
    from aiida import orm
    from aiida_yambo.workflows.yamborestart import YamboRestart
    from aiida_yambo.calculations.yambo import YamboCalculation
    process_class = YamboRestart if what == 'YamboRestart' else YamboCalculation
    builder = process_class.get_builder()
    if what == 'YamboRestart':
        builder.parent_folder = restart_inputs.parent_folder
        builder.max_iterations = orm.Int(7)
        builder.max_walltime = orm.Int(200)
        builder.code_version = orm.Str('5.x')
        leaf = builder.yambo
    else:
        leaf = builder
        leaf.parent_folder = restart_inputs.parent_folder
    for key, value in restart_inputs.items():
        if key != 'parent_folder':
            leaf[key] = value
    manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what=what, mode='builder'))
    exported = manager.generate_AiiDA_inputs(what)
    rebuilt = process_class.get_builder()
    rebuilt._update(exported)
    result_leaf = rebuilt.yambo if what == 'YamboRestart' else rebuilt
    assert result_leaf.parameters.get_dict() == restart_inputs.parameters.get_dict()
    assert result_leaf.settings.get_dict() == restart_inputs.settings.get_dict()
    assert result_leaf.code.uuid == restart_inputs.code.uuid
    assert result_leaf.metadata.options.queue_name == 'test'
    assert rebuilt.parent_folder.uuid == restart_inputs.parent_folder.uuid
    if what == 'YamboRestart':
        assert rebuilt.max_iterations.value == 7
        assert rebuilt.max_walltime.value == 200
    assert 'scf' not in exported and 'qp' not in exported
    # Check the stored-process import path as well as the live builder path.
    from aiida.engine.utils import instantiate_process
    from aiida.manage import get_manager
    process = instantiate_process(get_manager().get_runner(), process_class, **exported)
    try:
        from_node = YamboInputManager(**YamboInputManager.get_nested_inputs(process.node, what=what))
        assert from_node.qp.parameters == manager.qp.parameters
        assert from_node.qp.metadata.options.queue_name == 'test'
        assert from_node.parent_folder.uuid == manager.parent_folder.uuid
    finally:
        process.close()


@pytest.fixture
def restart_harness(restart_inputs):
    from aiida import orm
    from aiida.common import AttributeDict
    from aiida_yambo.workflows.yamborestart import YamboRestart

    class Harness:
        _edit_inputs = YamboRestart._edit_inputs
        _apply_parallelism_fix = staticmethod(YamboRestart._apply_parallelism_fix)
        report_error_handled = YamboRestart.report_error_handled
        report = lambda self, message: None

    harness = Harness()
    harness.ctx = AttributeDict(dict(inputs=restart_inputs, iteration=2))
    harness.inputs = AttributeDict(dict(max_walltime=orm.Int(250), max_number_of_nodes=orm.Int(4)))
    harness.exit_codes = YamboRestart.exit_codes
    return harness


def test_restart_edits_preserve_inputs_and_rollback(restart_harness):
    harness = restart_harness
    original = dict(harness.ctx.inputs)
    with harness._edit_inputs() as manager:
        manager.settings['COPY_SAVE'] = True
    assert harness.ctx.inputs.settings.get_dict()['COPY_SAVE'] is True
    for key in ('parameters', 'metadata', 'code', 'parent_folder'):
        assert harness.ctx.inputs[key] is original[key]
    with pytest.raises(RuntimeError), harness._edit_inputs() as manager:
        manager.set_variable('NGsBlkXp', [99, 'Ry'])
        raise RuntimeError('abort edit')
    assert harness.ctx.inputs.parameters is original['parameters']
    with harness._edit_inputs() as manager:
        assert manager.settings['COPY_SAVE'] is True


@pytest.mark.parametrize('handler', ['_handle_walltime_error', '_handle_parallelism_error', '_handle_memory_error', '_handle_unknown_error', '_handle_variable_NOT_DEFINED'])
@pytest.mark.parametrize('wrote_dbs', [False, True])
def test_restart_handlers(restart_harness, monkeypatch, handler, wrote_dbs):
    from aiida import orm
    from aiida.common import AttributeDict
    from aiida_yambo.workflows import yamborestart as module
    harness = restart_harness
    original_code = harness.ctx.inputs.code
    original_parent = harness.ctx.inputs.parent_folder
    new_parent = orm.RemoteData(computer=original_parent.computer, remote_path='/tmp/restart').store()
    calculation = SimpleNamespace(
        process_label='YamboCalculation', pk=123, exit_status=500, exit_message='test failure',
        outputs=AttributeDict(dict(remote_folder=new_parent, output_parameters=orm.Dict({
            'yambo_wrote_dbs': wrote_dbs, 'corrupted_fragment': 'ndb.pp_fragment_1',
        }))),
    )
    resources = {'num_machines': 2, 'num_mpiprocs_per_machine': 2, 'num_cores_per_mpiproc': 2}
    fix = lambda *args: ({'PAR_def_mode': 'balanced'}, resources, ['X_CPU'])
    monkeypatch.setattr(module, 'fix_parallelism', fix)
    monkeypatch.setattr(module, 'fix_memory', fix)
    result = getattr(module.YamboRestart, handler).__wrapped__(harness, calculation)
    assert result.do_break and result.exit_code.status == 0
    inputs = harness.ctx.inputs
    assert inputs.settings.get_dict()['ITERATION'] == 2
    assert inputs.code is original_code
    assert inputs.metadata.options.queue_name == 'test'
    if handler == '_handle_walltime_error':
        assert inputs.metadata.options.max_wallclock_seconds == 250
    if handler in ('_handle_memory_error', '_handle_parallelism_error'):
        assert inputs.parameters.get_dict()['variables'] == {'NGsBlkXp': [2, 'Ry'], 'PAR_def_mode': 'balanced'}
        assert inputs.metadata.options.resources == resources
        assert inputs.metadata.options.prepend_text == 'module load yambo\nexport OMP_NUM_THREADS=2'
    if handler == '_handle_unknown_error':
        assert inputs.settings.get_dict()['COPY_SAVE'] is True
    else:
        assert inputs.settings.get_dict().get('COPY_DBS', False) is wrote_dbs
    expect_new_parent = handler == '_handle_walltime_error' or (wrote_dbs and handler != '_handle_unknown_error')
    assert inputs.parent_folder is (new_parent if expect_new_parent else original_parent)


def test_restart_does_not_inject_scheduler_defaults(restart_harness):
    from aiida.common import AttributeDict
    harness = restart_harness
    harness.ctx.inputs.metadata = AttributeDict({'options': {'resources': {'tot_num_mpiprocs': 8}}})
    with harness._edit_inputs() as manager:
        manager.settings['COPY_SAVE'] = True
        manager.set_prepend_text('module load yambo')
    assert harness.ctx.inputs.metadata.options == {
        'resources': {'tot_num_mpiprocs': 8}, 'prepend_text': 'module load yambo',
    }


@pytest.mark.parametrize('handler, helper', [
    ('_handle_memory_error', 'fix_memory'), ('_handle_parallelism_error', 'fix_parallelism'),
])
def test_restart_unresolved_parallelism(restart_harness, monkeypatch, handler, helper):
    from aiida_yambo.workflows import yamborestart as module
    original = dict(restart_harness.ctx.inputs)
    monkeypatch.setattr(module, helper, lambda *args: (False, False, False))
    result = getattr(module.YamboRestart, handler).__wrapped__(restart_harness, SimpleNamespace(exit_status=500))
    assert result.exit_code == module.YamboRestart.exit_codes.ERROR_UNRECOVERABLE_FAILURE
    assert all(restart_harness.ctx.inputs[key] is value for key, value in original.items())
