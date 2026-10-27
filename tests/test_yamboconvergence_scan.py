"""YamboConvergence in scan mode: one calculation for each given set of values, no convergence evaluation."""
from types import SimpleNamespace

import pandas as pd
import pytest

from aiida_yambo.workflows.utils.parameters import ConvParam, ScanParam, create_parameter_space_inputs

POINTS = [
    {'FFTGvecs': 10, 'BndsRnXp': 20, 'NGsBlkXp': 2},
    {'FFTGvecs': 15, 'BndsRnXp': 30, 'NGsBlkXp': 4},
    {'FFTGvecs': 20, 'BndsRnXp': 40, 'NGsBlkXp': 6},
]


@pytest.fixture
def YamboConvergence(aiida_profile):
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence
    return YamboConvergence


@pytest.fixture
def scan_inputs(yc_inputs):
    from aiida import orm

    scan = ScanParam.from_dicts(POINTS, max_concurrent_steps=2)
    return {**yc_inputs, 'parameters_space': orm.List(create_parameter_space_inputs([scan]))}


@pytest.mark.parametrize('space, valid', [
    ([ScanParam.from_dicts(POINTS)], True),
    ([ConvParam('FFTGvecs', start=10, stop=25, delta=5, max=40, units='Ry', conv_thr=0.5)], True),
    ([], True),
    ([ScanParam.from_dicts(POINTS), ConvParam('FFTGvecs', start=10, stop=25, delta=5, max=40, units='Ry')], False),
])
def test_a_scan_must_be_the_only_entry(YamboConvergence, space, valid):
    from aiida import orm

    error = YamboConvergence._validate_parameters_space(orm.List(create_parameter_space_inputs(space)), None)

    assert (error is None) is valid


def test_an_invalid_space_is_refused_at_submission(YamboConvergence):
    from aiida import orm

    error = YamboConvergence._validate_parameters_space(orm.List([{'var': ['FFTGvecs'], 'start': 10}]), None)

    assert 'invalid parameters space' in error


def test_a_workchain_cannot_mix_a_scan_with_a_convergence(generate_workchain, YamboConvergence, scan_inputs):
    from aiida import orm

    space = create_parameter_space_inputs([ScanParam.from_dicts(POINTS), ConvParam(
        'FFTGvecs', start=10, stop=25, delta=5, max=40, units='Ry', conv_thr=0.5)])

    with pytest.raises(ValueError, match='only entry'):
        generate_workchain(YamboConvergence, {**scan_inputs, 'parameters_space': orm.List(space)})


def test_the_scan_is_submitted_in_batches_with_the_values_of_each_point(generate_workchain, YamboConvergence, scan_inputs,
                                                                         monkeypatch):
    from aiida import orm

    process = generate_workchain(YamboConvergence, scan_inputs)
    submitted = []

    def submit(process_class, **inputs):
        submitted.append(inputs['qp']['yambo']['parameters'].get_dict()['variables'])
        return orm.Int(len(submitted)).store()  # any stored node: we need a pk and a uuid

    monkeypatch.setattr(process, 'submit', submit)
    monkeypatch.setattr('aiida_yambo.workflows.yamboconvergence.ToContext', lambda *args, **kwargs: None)
    try:
        process.setup()
        laps = 0
        while process.has_to_continue():
            laps += 1
            assert laps < 10, 'the loop does not end'
            process.update_next_step()
            assert process.prepare_grid() is None
            process.perform_next_step()
            if process.ctx.current_convergence.cannot_stop is False:
                break

        assert laps == 2  # 3 points, 2 at a time
        assert [(v['FFTGvecs'], v['BndsRnXp'], v['NGsBlkXp'], v['GbndRnge']) for v in submitted] == [
            ([10, 'Ry'], [[1, 20], ''], [2, 'Ry'], [[1, 20], '']),
            ([15, 'Ry'], [[1, 30], ''], [4, 'Ry'], [[1, 30], '']),
            ([20, 'Ry'], [[1, 40], ''], [6, 'Ry'], [[1, 40], '']),
        ]
        assert list(process.ctx.df['FFTGvecs']) == [10, 15, 20]
        assert process.ctx.current_convergence.parameters.steps == 3
    finally:
        process.close()


def _scan_workflow(generate_workchain, YamboConvergence, scan_inputs, monkeypatch, failed):
    """A scan whose three calculations were 'submitted'; `failed` says which of them failed."""
    process = generate_workchain(YamboConvergence, scan_inputs)
    process.setup()
    process.ctx.df = pd.DataFrame({
        'pk': [101, 102, 103], 'uuid': ['u1', 'u2', 'u3'], 'global_step': [0, 1, 2],
        'FFTGvecs': [10, 15, 20], 'BndsRnXp': [20, 30, 40], 'NGsBlkXp': [2, 4, 6],
        'parameters_studied': ['FFTGvecs, BndsRnXp, NGsBlkXp'] * 3,
        'failed': False, 'useful': False, 'converged': False, 'gap_GG': False,
    })
    process.ctx.current_batch_pks = [101, 102, 103]
    process.ctx.current_convergence.concurrent_steps_count = 3

    def take_quantities(df, what):
        df['failed'] = [pk in failed for pk in df['pk']]
        df['useful'] = ~df['failed']
        for quantity in what:
            df[quantity] = [1.0 if not f else False for f in df['failed']]

    monkeypatch.setattr('aiida_yambo.workflows.yamboconvergence.take_quantities', take_quantities)
    return process


@pytest.mark.parametrize('failed, summary_failed', [([], 0), ([102], 1)])
def test_failed_calculations_do_not_stop_the_scan(generate_workchain, YamboConvergence, scan_inputs, monkeypatch, failed,
                                                  summary_failed):
    process = _scan_workflow(generate_workchain, YamboConvergence, scan_inputs, monkeypatch, failed)
    try:
        assert process.data_analysis() is None

        assert process.ctx.current_convergence.success is True
        assert process.ctx.none_encountered is False
        assert process.has_to_continue() is False
        scan = process.ctx.convergence_summary['scan']
        assert (scan['points'], scan['failed']) == (3, summary_failed)
        assert list(process.ctx.df['failed']) == [pk in failed for pk in (101, 102, 103)]
        assert process.ctx.current_convergence.last_succeeded_uuid == 'u3'
        assert process.report_wf() is None  # the run is successful
    finally:
        process.close()


def test_the_scan_fails_if_every_calculation_failed(generate_workchain, YamboConvergence, scan_inputs, monkeypatch):
    process = _scan_workflow(generate_workchain, YamboConvergence, scan_inputs, monkeypatch, [101, 102, 103])
    try:
        process.data_analysis()

        assert process.ctx.convergence_summary['scan']['failed'] == 3
        assert process.has_to_continue() is False
        assert process.report_wf() == process.exit_codes.CALCS_FAILED
        assert len(process.outputs['collection'].get_dict()['story']) == 3  # the story is kept
    finally:
        process.close()
