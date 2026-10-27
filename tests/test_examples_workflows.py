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
