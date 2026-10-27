"""Regression tests for the bug fixes in yambowf.py and yamboconvergence.py.

The AiiDA-facing pieces are replaced by small fakes (the profile is only needed to import the workflows).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from aiida import orm

from aiida_yambo.workflows.utils.parameters import ConvParam, CoupledConvParam, KpointMeshConvParam


@pytest.fixture
def yambowf(aiida_profile):
    # importing the workflows loads the AiiDA profile
    from aiida_yambo.workflows import yambowf as module
    return module


@pytest.fixture
def yamboconvergence(aiida_profile):
    from aiida_yambo.workflows import yamboconvergence as module
    return module


def _calcjob(pk, finished_ok):
    node = MagicMock(spec=orm.CalcJobNode)
    node.pk = pk
    node.is_finished_ok = finished_ok
    return node


def test_clean_reports_cleaned_calculations(yambowf):
    failed_1, failed_2, ok = _calcjob(11, False), _calcjob(12, False), _calcjob(13, True)
    node = SimpleNamespace(called_descendants=[failed_1, ok, failed_2])

    assert yambowf.clean(node) == 'cleaned remote folders of calculations: 11, 12'
    failed_1.outputs.remote_folder._clean.assert_called_once()
    ok.outputs.remote_folder._clean.assert_not_called()


def test_clean_returns_none_if_nothing_to_clean(yambowf):
    assert yambowf.clean(SimpleNamespace(called_descendants=[_calcjob(1, True)])) is None


@pytest.mark.parametrize('nelectrons, soc, expected', [(8, False, (4, 5)), (7, False, (4, 5)), (8, True, (8, 9))])
def test_valence_conduction_bands(yambowf, nelectrons, soc, expected):
    assert yambowf.valence_conduction_bands(nelectrons, soc) == expected


def test_qp_list_merger_skips_qp_present_in_any_database(yambowf, monkeypatch):
    computed = {1: {(1, 1)}, 2: {(1, 2)}}  # db pk -> {(k, b)}
    monkeypatch.setattr(yambowf, 'check_already_computed', lambda db, b, k: (k, b) in computed[db])

    groups = yambowf.QP_list_merger([[1, 1, 1, 3]], qp_per_subset=10, already_computed_QP_db=[1, 2])

    assert groups == [[[1, 1, 3, 3]]]


def test_qp_list_merger_without_databases(yambowf):
    assert yambowf.QP_list_merger([[1, 1, 1, 2]], qp_per_subset=1) == [[[1, 1, 1, 1]], [[1, 1, 2, 2]]]


def _fake_pw(yambowf, monkeypatch, bands, nelectrons=8, soc=False):
    bands = np.array(bands)
    pw = SimpleNamespace(outputs=SimpleNamespace(
        output_parameters=SimpleNamespace(get_dict=lambda: {
            'fermi_energy': 0., 'spin_orbit_calculation': soc, 'number_of_electrons': nelectrons}),
        output_band=SimpleNamespace(get_kpoints=lambda: np.zeros((len(bands), 3)), get_bands=lambda: bands),
    ))
    monkeypatch.setattr(yambowf, 'find_pw_parent', lambda ywfl: pw)


BANDS = [[-3, -2, -0.9, -0.5, 0.5, 0.9]] * 2  # valence = band 4, conduction = band 5, mid-gap = 0


def test_qp_mapper_enlarges_range_and_forwards_spectrum_tol(yambowf, monkeypatch):
    _fake_pw(yambowf, monkeypatch, BANDS)

    QP, scissored = yambowf.QP_mapper(None, tol=0.1, spectrum_tol=0.6)

    assert QP == [[1, 1, 4, 4], [1, 1, 5, 5], [2, 2, 4, 4], [2, 2, 5, 5]]
    # spectrum_tol=0.6 must survive the enlargements of tol: only bands 4 and 5 (0-based 3, 4)
    assert scissored == [3, 4]


def test_qp_mapper_gives_up_instead_of_recursing_forever(yambowf, monkeypatch):
    _fake_pw(yambowf, monkeypatch, BANDS)

    with pytest.raises(ValueError):
        yambowf.QP_mapper(None, tol=0.01, max_iterations=2)


def _workflow(**ctx):
    ctx = SimpleNamespace(**ctx)
    return SimpleNamespace(ctx=ctx, report=lambda *args: None)


def test_perform_next_returns_the_post_process_exit_code(yambowf):
    wf = _workflow(all_calcs_to_do=['merge QP'], merge_QP_needed=True, calc_to_do=None,
                   calc=SimpleNamespace(is_finished_ok=True))
    wf.run_post_process = lambda: 'EXIT_CODE'

    assert yambowf.YamboWorkflow.perform_next(wf) == 'EXIT_CODE'
    assert wf.ctx.merge_QP_needed is False


def test_perform_next_continues_if_post_process_succeeds(yambowf):
    wf = _workflow(all_calcs_to_do=['merge QP'], merge_QP_needed=True, calc_to_do=None,
                   calc=SimpleNamespace(is_finished_ok=True))
    wf.run_post_process = lambda: None

    assert yambowf.YamboWorkflow.perform_next(wf) is None


@pytest.mark.parametrize('has_valence, has_conduction, run_bse, extend_db, fails', [
    (True, True, True, True, False),
    (True, False, False, False, False),  # QP-only run: not an error
    (False, True, False, False, False),  # QP-only run: not an error (it used to abort)
    (True, False, True, False, True),
    (False, True, True, False, True),
    (True, False, False, True, True),
])
def test_run_post_process_missing_band_region(yambowf, monkeypatch, has_valence, has_conduction, run_bse, extend_db, fails):
    fake_orm = SimpleNamespace(Str=lambda v: SimpleNamespace(value=v), Int=lambda v: SimpleNamespace(value=v),
                               Dict=lambda dict=None: dict, List=lambda v: v)
    merge_result = {'QP_db': 'QP_db', 'has_valence': SimpleNamespace(value=has_valence),
                    'has_conduction': SimpleNamespace(value=has_conduction)}
    monkeypatch.setattr(yambowf, 'orm', fake_orm)
    monkeypatch.setattr(yambowf, 'store_List', lambda value: value)
    monkeypatch.setattr(yambowf, 'merge_QP', lambda *args, **kwargs: merge_result)
    monkeypatch.setattr(yambowf, 'extend_QP', lambda *args, **kwargs: 'extended')
    monkeypatch.setattr(yambowf, 'QP_analyzer', lambda *args: {})

    wf = _workflow(splitted_QP=[], calc=SimpleNamespace(pk=1), mapping={}, should_run_bse=run_bse,
                   input_manager=SimpleNamespace(QP_subsets_dict={'extend_db': extend_db}))
    wf.inputs = {}
    wf.out = lambda *args: None
    wf.exit_codes = SimpleNamespace(ERROR_MISSING_BAND_REGION='MISSING', ERROR_SPLITTED_QP_FAILED='SPLIT')

    result = yambowf.YamboWorkflow.run_post_process(wf)

    assert result == ('MISSING' if fails else None)


def _parameter_space():
    fft = ConvParam('FFTGvecs', start=10, stop=30, delta=5, max=50, units='Ry', conv_thr=0.5)
    kpoints = KpointMeshConvParam(var='kpoint_density', start=0.5, stop=0.2, delta=-0.1, max=0.5, units='1/A', conv_thr=0.5)
    coupled = CoupledConvParam(var=['BndsRnXp', 'NGsBlkXp'], start=[10, 2], stop=[30, 6], delta=[10, 2], max=[50, 10],
                               units=['', 'Ry'], conv_thr=1.0, mirror_values={'BndsRnXp': ['GbndRnge']})
    return fft, kpoints, coupled


def test_select_convergence_parameters_follows_the_requested_order(yamboconvergence):
    fft, kpoints, coupled = _parameter_space()

    selected = yamboconvergence.select_convergence_parameters(
        [fft, kpoints, coupled], [['BndsRnXp', 'GbndRnge', 'NGsBlkXp'], 'FFTGvecs'])

    assert selected == [coupled, fft]


def test_select_convergence_parameters_rejects_unknown_names(yamboconvergence):
    with pytest.raises(ValueError):
        yamboconvergence.select_convergence_parameters(list(_parameter_space()), ['EXXRLvcs'])


def _write_qp_db(path, eo, qp_e):
    import xarray

    n_qp = len(eo)
    xarray.Dataset({
        'QP_table': (('three', 'n_qp'), np.array([list(range(1, n_qp + 1)), [1] * n_qp, [1] * n_qp])),
        'QP_Eo': (('n_qp',), np.array(eo, dtype=float)),
        'QP_E': (('n_qp', 'comp'), np.stack([np.array(qp_e, dtype=float), np.zeros(n_qp)], axis=1)),
    }).to_netcdf(path)


def test_sanity_check_qp_fits_the_outliers(yambowf, tmp_path):
    import xarray

    eo = [0.0, 0.1, 0.5, 0.6, 0.7]
    qp_e = [1.1 * e + 0.01 for e in eo]
    qp_e[4] = eo[4] + 1.0  # 27 eV correction: not reliable
    _write_qp_db(tmp_path / 'in.nc', eo, qp_e)

    output, fit_v, fit_c = yambowf.sanity_check_QP(2, 3, str(tmp_path / 'in.nc'), str(tmp_path / 'out.nc'))

    assert fit_c == pytest.approx([1.1, 0.01])
    with xarray.open_dataset(output) as fixed:
        # outlier replaced by the fit, then everything aligned to the valence band maximum
        assert fixed.QP_E[4, 0] == pytest.approx(1.1 * 0.7 + 0.01 - (1.1 * 0.1 + 0.01))
        assert fixed.QP_E[1, 0] == pytest.approx(0.0)


def test_sanity_check_qp_without_enough_conduction_states(yambowf, tmp_path):
    import xarray

    eo = [0.0, 0.1, 0.5, 0.6]
    qp_e = [e + 0.01 for e in eo]
    qp_e[3] = eo[3] + 1.0
    _write_qp_db(tmp_path / 'in.nc', eo, qp_e)

    output, fit_v, fit_c = yambowf.sanity_check_QP(2, 3, str(tmp_path / 'in.nc'), str(tmp_path / 'out.nc'))

    assert fit_v is not None and fit_c is None  # a single reliable conduction state: no fit
    with xarray.open_dataset(output) as fixed:
        assert fixed.QP_E[3, 0] == pytest.approx(eo[3] + 1.0 - 0.11)  # left untouched (only aligned)


def test_sanity_check_qp_threshold_is_configurable(yambowf, tmp_path):
    eo = [0.0, 0.1, 0.5, 0.6, 0.7]
    _write_qp_db(tmp_path / 'in.nc', eo, [e + 0.01 for e in eo])

    # 0.01 Ha = 0.27 eV: with a 0.1 eV threshold all the states are "wrong", so there is no reliable fit
    _, fit_v, fit_c = yambowf.sanity_check_QP(2, 3, str(tmp_path / 'in.nc'), str(tmp_path / 'out.nc'), max_correction_eV=0.1)

    assert fit_v is None and fit_c is None


def _fake_merge_environment(yambowf, monkeypatch, tmp_path, run):
    pw = SimpleNamespace(outputs=SimpleNamespace(output_parameters=SimpleNamespace(get_dict=lambda: {
        'spin_orbit_calculation': False, 'number_of_electrons': 4})))
    monkeypatch.setattr(yambowf, 'find_pw_parent', lambda node: pw)
    monkeypatch.setattr(yambowf, 'load_node', lambda pk: object())
    monkeypatch.setattr(yambowf.subprocess, 'run', run)


def test_merge_qp_runs_yambopy_and_checks_the_result(yambowf, monkeypatch, tmp_path):
    from aiida import orm

    eo = [0.0, 0.1, 0.5, 0.6]

    def run(command, check):
        assert command[:2] == ['yambopy', 'mergeqp'] and check is True
        _write_qp_db(command[command.index('-o') + 1], eo, [1.1 * e + 0.01 for e in eo])

    _fake_merge_environment(yambowf, monkeypatch, tmp_path, run)

    result = yambowf.merge_QP(orm.List([]), orm.Str('ndb.QP_merged'), orm.Int(1), orm.Dict({}))

    assert result['has_valence'].value and result['has_conduction'].value
    assert result['QP_db'].filename == 'ndb.QP_fixed'


def test_merge_qp_fails_if_yambopy_fails(yambowf, monkeypatch, tmp_path):
    import subprocess

    from aiida import orm

    def run(command, check):
        raise subprocess.CalledProcessError(1, command)

    _fake_merge_environment(yambowf, monkeypatch, tmp_path, run)

    with pytest.raises(subprocess.CalledProcessError):
        yambowf.merge_QP(orm.List([]), orm.Str('ndb.QP_merged'), orm.Int(1), orm.Dict({}))


def test_qp_splitter_options_are_applied_to_the_qp_manager(yambowf):
    from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager

    manager = YamboInputManager(qp={'parameters': {'arguments': ['gw0'], 'variables': {}}})

    yambowf.set_QP_splitter_options(manager.qp, {'resources': {'num_machines': 2, 'num_mpiprocs_per_machine': 8},
                                                  'prepend_text': 'module load yambo'})

    assert manager.qp.metadata.options.resources.num_machines == 2
    assert manager.qp.metadata.options.resources.num_mpiprocs_per_machine == 8
    assert manager.qp.metadata.options.prepend_text == 'module load yambo'
    assert manager.bse.metadata.options.prepend_text == ''  # nothing leaks to the BSE step


def test_qp_splitter_options_are_optional(yambowf):
    from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager

    manager = YamboInputManager(qp={'parameters': {'arguments': ['gw0'], 'variables': {}}})
    before = manager.qp.metadata.options.model_dump()

    yambowf.set_QP_splitter_options(manager.qp, {'parallel_runs': 1})

    assert manager.qp.metadata.options.model_dump() == before


def test_setup_with_an_empty_parent_starts_from_scratch(yambowf, generate_workchain, ywfl_qp_inputs, aiida_localhost,
                                                        monkeypatch):
    from aiida import orm

    parent_folder = orm.RemoteData(computer=aiida_localhost, remote_path='/tmp/empty').store()
    parent = SimpleNamespace(process_type='aiida.calculations:quantumespresso.pw',
                             outputs=SimpleNamespace(remote_folder=SimpleNamespace(is_empty=True)))
    monkeypatch.setattr(yambowf, 'take_calc_from_remote', lambda *args, **kwargs: parent)

    process = generate_workchain(yambowf.YamboWorkflow, {**ywfl_qp_inputs, 'parent_folder': parent_folder})
    try:
        process.setup()
        assert process.ctx.all_calcs_to_do == ['scf', 'nscf', 'qp']
        assert process.ctx.input_manager.parent_folder is None
        assert 'calc' not in process.ctx
    finally:
        process.close()
