"""get_builder_from_protocol of YamboConvergence / YamboWorkflow with fake codes and pseudos."""
import copy
import io
import pathlib
import tempfile
from uuid import uuid4

import pytest

FAMILY_LABEL = 'SSSP/1.3/PBEsol/efficiency'  # default pseudo family of the `moderate` PW protocol


@pytest.fixture(scope='session')
def pseudo_family(aiida_profile):
    from aiida import orm
    from aiida_pseudo.groups.family import SsspFamily

    try:
        return orm.load_group(FAMILY_LABEL)
    except Exception:
        pass
    content = '<UPF version="2.0.1"><PP_HEADER\nelement="Si"\nz_valence="4.0"\n/></UPF>\n'
    with tempfile.TemporaryDirectory() as directory:
        (pathlib.Path(directory) / 'Si.upf').write_text(content)
        family = SsspFamily.create_from_folder(pathlib.Path(directory), FAMILY_LABEL)
    family.set_cutoffs({'Si': {'cutoff_wfc': 30.0, 'cutoff_rho': 240.0}}, 'standard', unit='Ry')
    return family


@pytest.fixture
def codes(aiida_localhost, pseudo_family):
    from aiida import orm

    def _code(name):
        return orm.InstalledCode(label=f'{name}-{uuid4().hex}', computer=aiida_localhost,
                                 filepath_executable='/bin/true').store()

    return {'pw_code': _code('pw'), 'preprocessing_code': _code('p2y'), 'code': _code('yambo')}


@pytest.fixture
def silicon():
    from aiida import orm

    param = 5.43
    structure = orm.StructureData(cell=[[param / 2, param / 2, 0], [param / 2, 0, param / 2], [0, param / 2, param / 2]])
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Si', name='Si')
    structure.append_atom(position=(param / 4, param / 4, param / 4), symbols='Si', name='Si')
    return structure


def _space(builder):
    return [parameters['var'] for parameters in builder.parameters_space.get_list()]


@pytest.mark.parametrize('protocol, threshold', [('fast', 10), ('moderate', 5), ('precise', 1)])
def test_convergence_protocols_build_with_their_thresholds(aiida_profile, codes, silicon, protocol, threshold):
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence

    builder = YamboConvergence.get_builder_from_protocol(structure=silicon, protocol=protocol, **codes)

    assert _space(builder) == [['FFTGvecs'], ['kpoint_density'], ['BndsRnXp', 'NGsBlkXp']]
    assert {p['conv_thr'] for p in builder.parameters_space.get_list()} == {threshold}


def test_convergence_parameters_select_and_order(aiida_profile, codes, silicon):
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence

    builder = YamboConvergence.get_builder_from_protocol(
        structure=silicon, convergence_parameters=[['BndsRnXp', 'GbndRnge', 'NGsBlkXp'], 'FFTGvecs'], **codes)

    assert _space(builder) == [['BndsRnXp', 'NGsBlkXp'], ['FFTGvecs']]


def test_the_overrides_of_the_caller_are_not_modified(aiida_profile, codes, silicon):
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence

    overrides = {'ywfl': {'clean_workdir': True, 'scf': {'clean_workdir': True}}}
    original = copy.deepcopy(overrides)

    YamboConvergence.get_builder_from_protocol(structure=silicon, overrides=overrides, **codes)

    assert overrides == original


def test_workflow_settings_of_the_user_update_the_protocol_ones(aiida_profile, codes, silicon):
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence

    builder = YamboConvergence.get_builder_from_protocol(structure=silicon, workflow_settings={'type': 'heavy'}, **codes)

    settings = builder.workflow_settings.get_dict()
    assert settings['type'] == 'heavy'
    assert settings['what'] == ['gap_GG']  # from the protocol
    assert settings['bands_nscf_update'] == 'all-at-once'


def test_bse_converges_the_excitons_unless_told_otherwise(aiida_profile, codes, silicon):
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence

    default = YamboConvergence.get_builder_from_protocol(structure=silicon, calc_type='bse', **codes)
    chosen = YamboConvergence.get_builder_from_protocol(
        structure=silicon, calc_type='bse', workflow_settings={'what': ['gap_']}, **codes)

    assert default.workflow_settings.get_dict()['what'] == ['lowest_exciton', 'brightest_exciton']
    assert chosen.workflow_settings.get_dict()['what'] == ['gap_']


def test_workflow_builder_accepts_codes_by_label_and_keeps_overrides(aiida_profile, codes, silicon):
    from aiida_yambo.workflows.yambowf import YamboWorkflow

    overrides = {'scf': {'clean_workdir': True}}

    builder = YamboWorkflow.get_builder_from_protocol(
        pw_code=codes['pw_code'].label, preprocessing_code=codes['preprocessing_code'].label,
        code=codes['code'].label, structure=silicon, overrides=overrides)

    assert overrides == {'scf': {'clean_workdir': True}}
    assert builder.scf['pw']['code'].uuid == codes['pw_code'].uuid
    assert builder.qp['yambo']['code'].uuid == codes['code'].uuid


def _workflow_builder(codes, silicon, **kwargs):
    from aiida_yambo.workflows.yambowf import YamboWorkflow

    return YamboWorkflow.get_builder_from_protocol(structure=silicon, calc_type='bse@qp', **codes, **kwargs)


def test_workflow_overrides_of_qp_and_bse_are_independent(aiida_profile, codes, silicon):
    overrides = {'qp': {'yambo': {'parameters': {'variables': {'EXXRLvcs': [10, 'Ry']}}}},
                 'bse': {'yambo': {'parameters': {'variables': {'BSEQptR': [[1, 1], '']}}}}}

    builder = _workflow_builder(codes, silicon, overrides=overrides)

    qp_variables = builder.qp['yambo']['parameters'].get_dict()['variables']
    bse_variables = builder.bse['yambo']['parameters'].get_dict()['variables']
    assert qp_variables['EXXRLvcs'] == [10, 'Ry'] and 'BSEQptR' not in qp_variables
    assert bse_variables['BSEQptR'] == [[1, 1], ''] and 'EXXRLvcs' not in bse_variables


def test_workflow_yres_overrides_still_apply_to_both_but_are_deprecated(aiida_profile, codes, silicon):
    overrides = {'yres': {'yambo': {'parameters': {'variables': {'EXXRLvcs': [10, 'Ry']}}}}}

    with pytest.warns(DeprecationWarning, match='yres'):
        builder = _workflow_builder(codes, silicon, overrides=overrides)

    for namespace in (builder.qp, builder.bse):
        assert namespace['yambo']['parameters'].get_dict()['variables']['EXXRLvcs'] == [10, 'Ry']
