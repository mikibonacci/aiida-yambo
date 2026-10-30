"""Control flow of YamboConvergence: predicates are pure, state changes happen in the steps."""
from types import SimpleNamespace

import pytest

from aiida_yambo.workflows.utils.convergence import ConvergenceEvaluator
from aiida_yambo.workflows.utils.parameters import ConvParam


@pytest.fixture
def yamboconvergence(aiida_profile):
    from aiida_yambo.workflows import yamboconvergence as module
    return module.YamboConvergence


def _evaluator(**state):
    parameters = ConvParam('FFTGvecs', start=10, stop=25, delta=5, max=40, units='Ry', conv_thr=0.5, steps=4,
                           max_iterations=3)
    return ConvergenceEvaluator(parameters=parameters, what='gap_', **state)


def _workflow(evaluator, remaining=()):
    return SimpleNamespace(
        ctx=SimpleNamespace(current_convergence=evaluator, none_encountered=False,
                            input_manager=SimpleNamespace(parameters_space=list(remaining), workflow_settings={})),
        report=lambda *args: None,
    )


def test_has_to_continue_does_not_advance_the_iteration(yamboconvergence):
    evaluator = _evaluator(concurrent_steps_count=4)  # the batch is complete, no convergence yet
    wf = _workflow(evaluator)

    assert yamboconvergence.has_to_continue(wf) is True
    assert yamboconvergence.has_to_continue(wf) is True  # evaluating twice (e.g. on reload) changes nothing
    assert evaluator.current_iteration == 0


def test_update_next_step_advances_the_iteration_of_a_new_grid(yamboconvergence):
    evaluator = _evaluator(concurrent_steps_count=4)
    wf = _workflow(evaluator)

    yamboconvergence.update_next_step(wf)

    assert evaluator.current_iteration == 1


@pytest.mark.parametrize('state', [
    {'concurrent_steps_count': 1},  # the batch of calculations is not complete yet
    {'concurrent_steps_count': 3, 'check_single_point': True},  # verification of a point is not a new grid
])
def test_update_next_step_does_not_advance_otherwise(yamboconvergence, state):
    evaluator = _evaluator(**state)

    yamboconvergence.update_next_step(_workflow(evaluator))

    assert evaluator.current_iteration == 0


def test_has_to_continue_stops_when_the_iterations_are_exceeded(yamboconvergence):
    evaluator = _evaluator(concurrent_steps_count=4, current_iteration=3)

    assert yamboconvergence.has_to_continue(_workflow(evaluator)) is False


def test_has_to_continue_when_converged(yamboconvergence):
    evaluator = _evaluator(concurrent_steps_count=0, success=True)

    assert yamboconvergence.has_to_continue(_workflow(evaluator, remaining=[object()])) is True
    assert yamboconvergence.has_to_continue(_workflow(evaluator)) is False


def test_p2y_needed_is_a_pure_predicate(yamboconvergence):
    wf = SimpleNamespace(ctx=SimpleNamespace(run_p2y=True),
                         _evaluate_p2y_needed=lambda: pytest.fail('the predicate must not evaluate anything'))

    assert yamboconvergence.p2y_needed(wf) is True


@pytest.mark.parametrize('mode, required, expected', [
    ('all-at-once', 300, 300),  # the default: no cap
    ('full-step', 300, 300),
    (200, 300, 200),
    (500, 300, 300),
    (200, 0, 0),
    (True, 300, 300),  # a bool is not a number of bands
])
def test_nscf_bands_cap(yamboconvergence, mode, required, expected):
    wf = SimpleNamespace(ctx=SimpleNamespace(bands_mode=mode))

    assert yamboconvergence._capped_nscf_bands(wf, required) == expected


def test_failed_precalculation_still_returns_the_story(generate_workchain, yc_inputs):
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence

    process = generate_workchain(YamboConvergence, yc_inputs)
    try:
        process.setup()
        process.ctx.PRE = SimpleNamespace(is_finished_ok=False)

        exit_code = process.prepare_calculations()

        assert exit_code == process.exit_codes.PRECALC_FAILED
        assert {'collection', 'convergence_summary'} <= set(process.outputs)
        assert process.outputs['collection'].get_dict()['story'] == []
    finally:
        process.close()


def test_consumed_precalculation_is_forgotten(generate_workchain, yc_inputs):
    from aiida import orm
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence

    process = generate_workchain(YamboConvergence, yc_inputs)
    try:
        process.setup()
        remote = SimpleNamespace(pk=1)
        process.ctx.PRE = SimpleNamespace(is_finished_ok=True, outputs=SimpleNamespace(remote_folder=remote))
        process.ctx.input_manager.set_parent_folder = lambda node: setattr(process.ctx, 'parent', node)

        assert process.prepare_calculations() is None

        assert process.ctx.parent is remote
        assert process.ctx.PRE is None
        assert process.prepare_calculations() is None  # the next lap does not set the stale folder again
    finally:
        process.close()


def _setup_with_group_label(generate_workchain, yc_inputs, label):
    from aiida import orm
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence

    process = generate_workchain(YamboConvergence, {**yc_inputs, 'group_label': orm.Str(label)})
    reports = []
    process.report = lambda message, *args: reports.append(message)
    process.setup()
    return process, reports


@pytest.mark.parametrize('label', ['convergence/silicon/new', '123456', 'cafe'])
def test_setup_creates_the_group_if_it_does_not_exist_and_reports_it(generate_workchain, yc_inputs, label):
    from aiida import orm

    process, reports = _setup_with_group_label(generate_workchain, yc_inputs, label)
    try:
        group = orm.load_group(label=label)
        assert process.ctx.group.pk == group.pk and group.is_stored
        assert any('did not exist: created it' in report and label in report for report in reports), reports
    finally:
        process.close()


def test_setup_reuses_an_existing_group_and_reports_it(generate_workchain, yc_inputs):
    from aiida import orm

    existing = orm.Group(label='convergence/silicon/existing').store()
    existing.add_nodes(orm.Int(1).store())

    process, reports = _setup_with_group_label(generate_workchain, yc_inputs, 'convergence/silicon/existing')
    try:
        assert process.ctx.group.pk == existing.pk
        assert any('using the existing group' in report and '1 nodes' in report for report in reports), reports
        assert orm.Group.collection.find(filters={'label': 'convergence/silicon/existing'}).__len__() == 1
    finally:
        process.close()


@pytest.fixture
def module(yamboconvergence):
    import sys
    return sys.modules[yamboconvergence.__module__]


def _params():
    from aiida_yambo.workflows.utils.parameters import ConvParam, CoupledConvParam

    fft = ConvParam('FFTGvecs', start=10, stop=30, delta=5, max=50, units='Ry')
    bands = CoupledConvParam(['BndsRnXp', 'NGsBlkXp'], start=[10, 2], stop=[30, 6], delta=[10, 2], max=[50, 10],
                             units=['', 'Ry'])
    return fft, bands


def test_max_bands_reach(module):
    fft, bands = _params()

    assert module._max_bands([fft, bands]) == 50  # what the algorithm may reach
    assert module._max_bands([fft, bands], reach='stop') == 30  # the grid which is built
    assert module._max_bands([fft], reach='stop') == 0


def test_nscf_bands_follow_the_mode(yamboconvergence):
    from types import SimpleNamespace

    fft, bands = _params()
    for mode, current, expected in [('all-at-once', fft, 30), ('full-step', fft, 0), ('full-step', bands, 30)]:
        wf = SimpleNamespace(ctx=SimpleNamespace(bands_mode=mode, input_manager=SimpleNamespace(
            parameters_space=[fft, bands])))

        result = yamboconvergence._nscf_bands_for(wf, SimpleNamespace(parameters=current))

        assert result == expected, mode


@pytest.mark.parametrize('settings, valid', [
    ({}, True),
    ({'bands_nscf_update': 'all-at-once'}, True),
    ({'bands_nscf_update': 'full-step'}, True),
    ({'bands_nscf_update': 200}, True),
    ({'bands_nscf_update': 0}, False),
    ({'bands_nscf_update': 'everything'}, False),
    ({'bands_nscf_update': True}, False),
    ({'max_nscf_bands': 5000}, True),
    ({'max_nscf_bands': 0}, False),
    ({'max_nscf_bands': 'many'}, False),
])
def test_bands_settings_are_validated(module, settings, valid):
    assert (module.check_bands_settings(settings) is None) is valid


def _check_wf(yamboconvergence, mode, bands_needed, settings, grid):
    from types import SimpleNamespace

    exit_codes = SimpleNamespace(TOO_MANY_BANDS='too many')
    wf = SimpleNamespace(
        ctx=SimpleNamespace(bands_mode=mode, max_bands_needed=bands_needed,
                            input_manager=SimpleNamespace(workflow_settings=settings)),
        report=lambda message: wf.messages.append(message), messages=[], exit_codes=exit_codes)
    parameters = SimpleNamespace(var=['BndsRnXp'], stop=grid, max=grid)
    return yamboconvergence._check_nscf_bands(wf, SimpleNamespace(parameters=parameters)), wf.messages


def test_the_nscf_bands_are_reported_and_allowed(yamboconvergence):
    result, messages = _check_wf(yamboconvergence, 'all-at-once', 300, {}, 300)

    assert result is None and 'with 300 bands' in messages[0]


def test_too_many_nscf_bands_stop_the_workflow_unless_confirmed(yamboconvergence):
    result, messages = _check_wf(yamboconvergence, 'all-at-once', 3204, {}, 3204)
    assert result == 'too many' and 'max_nscf_bands' in messages[-1]

    result, _ = _check_wf(yamboconvergence, 'all-at-once', 3204, {'max_nscf_bands': 4000}, 3204)
    assert result is None


def test_a_grid_above_the_integer_cap_stops_the_workflow(yamboconvergence):
    result, messages = _check_wf(yamboconvergence, 200, 200, {}, 300)

    assert result == 'too many' and 'more than the 200' in messages[-1]
