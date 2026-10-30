"""The examples of the folders 01_dft ... : run with test nodes, the inputs they build must be valid."""
import pytest

from example_harness import EXAMPLES, install_submit, make_environment, run_example


@pytest.fixture
def environment(aiida_profile, aiida_localhost, monkeypatch):
    environment = make_environment(aiida_localhost)
    install_submit(monkeypatch, environment.submitted)
    return environment


# (script, settings to override): the default of the script and the variations it offers
RUNS = [
    ('01_dft/01_scf_nscf.py', {}),
    ('01_dft/01_scf_nscf.py', {'SCF_PK': '{SCF_PK}'}),
    ('02_gw/01_gw_from_scratch.py', {}),
    ('02_gw/02_gw_from_parent.py', {}),
    ('02_gw/04_qp_already_computed.py', {}),
    ('03_bse/01_bse_scissor.py', {}),
    ('03_bse/01_bse_scissor.py', {'PARENT_PK': '{PARENT_PK}'}),
    ('03_bse/02_bse_at_gw.py', {}),
    ('03_bse/02_bse_at_gw.py', {'MODE': 'range_QP'}),
    ('03_bse/02_bse_at_gw.py', {'PARENT_PK': '{PARENT_PK}'}),
    ('04_options/01_resources_and_parallelism.py', {}),
    ('04_options/02_initialise_only.py', {}),
    ('04_options/02_initialise_only.py', {'PARENT_PK': '{PARENT_PK}'}),
    ('04_options/03_cleaning_and_pseudos.py', {'CASE': 'clean_failed'}),
    ('04_options/03_cleaning_and_pseudos.py', {'CASE': 'lda'}),
    ('04_options/03_cleaning_and_pseudos.py', {'CASE': 'few_bands'}),
    ('04_options/04_molecule_and_metals.py', {'CASE': 'molecule'}),
    ('04_options/04_molecule_and_metals.py', {'CASE': 'metal_check'}),
    ('05_convergence/01_qp_convergence.py', {}),
    ('06_low_level/01_p2y.py', {}),
    ('06_low_level/02_gw_calculation.py', {}),
    ('06_low_level/03_ip_rpa.py', {}),
    ('06_low_level/04_yambo_restart.py', {}),
    ('08_ypp/01_ypp_wannier.py', {}),
    ('05_convergence/02_bse_convergence.py', {}),
    ('05_convergence/03_bse_at_gw_convergence.py', {}),
    ('05_convergence/04_scan.py', {}),
    ('05_convergence/05_fit_functions.py', {}),
    ('05_convergence/07_resource_rules.py', {}),
    ('05_convergence/08_qp_convergence_kpoints_mesh.py', {}),
    ('05_convergence/09_qp_convergence_kpoints_density.py', {}),
    ('04_options/01_resources_and_parallelism.py', {'QUEUE': 'debug', 'ACCOUNT': 'project'}),
] + [('02_gw/03_qp_splitter.py', {'MODE': mode}) for mode in
     ['boundaries', 'k_range', 'range_QP', 'full_bands', 'explicit', 'extend_db']]


@pytest.mark.parametrize('script, overrides', RUNS, ids=[f'{s}-{o}' for s, o in RUNS])
def test_example_builds_valid_inputs(environment, script, overrides):
    overrides = {k: environment.settings[v.strip('{}')] if isinstance(v, str) and v.startswith('{') else v
                 for k, v in overrides.items()}

    run_example(script, environment.settings, overrides)

    assert len(environment.submitted) == 1


def test_the_qp_splitter_modes_set_the_subsets(environment):
    run_example('02_gw/03_qp_splitter.py', environment.settings, {'MODE': 'range_QP'})

    (_, inputs), = environment.submitted
    assert inputs['QP_subsets_dict'].get_dict() == {'parallel_runs': 1, 'qp_per_subset': 2, 'range_QP': 5.0}


def test_the_qp_splitter_resources_are_passed_to_the_splitted_runs(environment):
    resources = {'num_machines': 1, 'num_mpiprocs_per_machine': 4}

    run_example('02_gw/03_qp_splitter.py', environment.settings, {'SPLIT_RESOURCES': resources})

    (_, inputs), = environment.submitted
    assert inputs['QP_subsets_dict'].get_dict()['resources'] == resources


def test_the_qp_splitter_walltime_and_auto_parallelism_are_passed(environment):
    run_example('02_gw/03_qp_splitter.py', environment.settings,
                {'SPLIT_WALLTIME': 7200, 'SPLIT_PARALLELISM': 'auto'})

    (_, inputs), = environment.submitted
    subsets = inputs['QP_subsets_dict'].get_dict()
    assert subsets['max_wallclock_seconds'] == 7200 and subsets['parallelism'] == 'auto'


def test_the_resources_example_sets_the_options_of_each_step(environment):
    run_example('04_options/01_resources_and_parallelism.py', environment.settings, {'QUEUE': 'debug'})

    (_, inputs), = environment.submitted
    qp = inputs['qp']
    assert qp['yambo']['metadata']['options']['resources']['num_cores_per_mpiproc'] == 2
    assert qp['yambo']['metadata']['options']['queue_name'] == 'debug'
    assert qp['max_walltime'].value == 4 * 3600 and qp['max_number_of_nodes'].value == 2
    assert inputs['nscf']['pw']['metadata']['options']['resources']['num_mpiprocs_per_machine'] == 4


def test_the_resource_rules_example_passes_the_rules_to_the_convergence(environment):
    run_example('05_convergence/07_resource_rules.py', environment.settings)

    (process_class, inputs), = environment.submitted
    rules = inputs['resource_rules'].get_list()
    assert [rule['use'].get('num_machines') for rule in rules] == [4, 2, 2]


def test_the_scan_example_has_one_scan_parameter(environment):
    run_example('05_convergence/04_scan.py', environment.settings)

    (_, inputs), = environment.submitted
    assert len(inputs['parameters_space'].get_list()) == 1


def test_the_fit_functions_example_keeps_the_bands_small(environment):
    """The protocol grid of the bands (thousands, times log10 of the volume) made the NSCF compute 3204 bands."""
    from aiida_yambo.workflows.yamboconvergence import _max_bands

    run_example('05_convergence/05_fit_functions.py', environment.settings)

    (_, inputs), = environment.submitted
    parameters = inputs['parameters_space'].get_list()
    from aiida_yambo.workflows.utils.parameters import CoupledConvParam
    coupled = [CoupledConvParam(**p) for p in parameters if len(p['var']) == 2]
    assert coupled and _max_bands(coupled) < 200


def test_the_offline_fit_example_runs_and_prefers_the_matching_model(tmp_path, monkeypatch, capsys):
    from aiida_yambo.workflows.utils.fit_models import unregister_fit_model

    monkeypatch.chdir(tmp_path)
    try:
        run_example('05_convergence/06_fit_models_offline.py', {})
    finally:
        unregister_fit_model('exp_plus_power')

    output = capsys.readouterr().out
    assert 'The best fit of this data: ExpPlusPower()' in output


def _stored_workflow(**outputs):
    from aiida import orm
    from aiida.common.links import LinkType

    workflow = orm.WorkflowNode()
    workflow.store()
    for label, node in outputs.items():
        node.store()
        node.base.links.add_incoming(workflow, link_type=LinkType.RETURN, link_label=label)
    return workflow


def _example_namespace(script):
    import runpy

    return runpy.run_path(str(EXAMPLES / script), run_name='example')


def test_the_workflow_analysis_functions_read_the_outputs(aiida_profile):
    import numpy as np
    from aiida import orm

    chi = orm.ArrayData()
    chi.set_array('eV', np.linspace(0, 5, 11))
    chi.set_array('eps_2', np.linspace(0, 1, 11))
    workflow = _stored_workflow(output_ywfl_parameters=orm.Dict({'gap_': 1.2}), array_chi=chi)
    functions = _example_namespace('07_analysis/01_read_workflow_results.py')

    summary = functions['summary'](workflow)
    assert summary['parameters'] == {'gap_': 1.2}
    assert 'array_chi' in summary['outputs']
    energy, epsilon2 = functions['spectrum'](workflow)
    assert len(energy) == len(epsilon2) == 11
    assert functions['qp_corrections'](workflow) is None


def test_the_convergence_analysis_functions_read_the_story(aiida_profile, tmp_path, monkeypatch):
    from aiida import orm

    rows = [{'FFTGvecs': 10, 'gap_': 1.0, 'failed': False, 'converged': False, 'pk': 1},
            {'FFTGvecs': 15, 'gap_': 1.1, 'failed': False, 'converged': True, 'pk': 2},
            {'FFTGvecs': 20, 'gap_': False, 'failed': True, 'converged': False, 'pk': 3}]
    workflow = _stored_workflow(collection=orm.Dict({'story': rows}),
                                convergence_summary=orm.Dict({'FFTGvecs': {'converged': 15}}))
    functions = _example_namespace('07_analysis/02_read_convergence_results.py')

    table = functions['story'](workflow)
    assert list(functions['converged_points'](table)['pk']) == [2]
    assert list(functions['failed_points'](table)['pk']) == [3]
    assert functions['summary'](workflow) == {'FFTGvecs': {'converged': 15}}
    monkeypatch.chdir(tmp_path)
    assert (tmp_path / functions['plot_story'](table)).exists()
