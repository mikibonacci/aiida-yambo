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
    (0, 300, 300),  # default: no cap
    ('all-at-once', 300, 300),
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
