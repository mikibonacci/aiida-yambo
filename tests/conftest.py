"""Shared AiiDA pytest fixtures.

The previous file was named ``conftests.py`` and therefore was ignored by
pytest's fixture discovery.
"""

pytest_plugins = ['aiida.tools.pytest_fixtures']

# The pinned Yambopy/AiiDA dependency set still imports this NumPy 1.x alias.
# Keep the compatibility shim in the test harness rather than changing runtime
# package code or pinning NumPy for users.
import numpy as np

if not hasattr(np, 'cumproduct'):
    np.cumproduct = np.cumprod


import pytest


@pytest.fixture
def generate_workchain():
    """Instantiate (without running) a process, so its outline steps can be called one by one.

    Same approach as aiida-quantumespresso: set ``process.ctx`` by hand and call the step
    methods, no daemon needed. The caller must ``process.close()`` when done.
    """

    def _generate(process_class, inputs):
        from aiida.engine.utils import instantiate_process
        from aiida.manage import get_manager

        return instantiate_process(get_manager().get_runner(), process_class, **inputs)

    return _generate


@pytest.fixture
def yambo_code(aiida_profile, aiida_localhost):
    from uuid import uuid4

    from aiida import orm

    return orm.InstalledCode(
        label=f'test-yambo-{uuid4().hex}', computer=aiida_localhost, filepath_executable='/bin/true'
    ).store()


@pytest.fixture
def pw_inputs(aiida_profile, aiida_localhost):
    """Inputs of a PwBaseWorkChain (silicon, fake pseudo): enough to instantiate the process."""
    import io
    from uuid import uuid4

    from aiida import orm
    from aiida_pseudo.data.pseudo import UpfData

    code = orm.InstalledCode(
        label=f'test-pw-{uuid4().hex}', computer=aiida_localhost, filepath_executable='/bin/true'
    ).store()
    param = 5.43
    structure = orm.StructureData(cell=[[param / 2, param / 2, 0], [param / 2, 0, param / 2], [0, param / 2, param / 2]])
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Si', name='Si')
    structure.append_atom(position=(param / 4, param / 4, param / 4), symbols='Si', name='Si')
    content = '<UPF version="2.0.1"><PP_HEADER\nelement="Si"\nz_valence="4.0"\n/></UPF>\n'
    pseudo = UpfData(io.BytesIO(content.encode()), filename='Si.upf')
    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh([2, 2, 2])

    def _pw_inputs(calculation):
        return {
            'pw': {
                'code': code,
                'structure': structure,
                'pseudos': {'Si': pseudo},
                'parameters': orm.Dict({'CONTROL': {'calculation': calculation},
                                        'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 120.0}}),
                'metadata': {'options': {'resources': {'num_machines': 1}, 'max_wallclock_seconds': 100}},
            },
            'kpoints': kpoints,
        }

    return _pw_inputs


@pytest.fixture
def ywfl_qp_inputs(yambo_code, pw_inputs):
    """Minimal YamboWorkflow inputs: scf + nscf + a QP step, starting from scratch (no parent folder)."""
    from aiida import orm

    return {
        'scf': pw_inputs('scf'),
        'nscf': pw_inputs('nscf'),
        'qp': {
            'yambo': {
                'code': yambo_code,
                'parameters': orm.Dict({'arguments': ['gw0', 'HF_and_locXC', 'ppa'],
                                        'variables': {'NGsBlkXp': [2, 'Ry'], 'BndsRnXp': [[1, 20], '']}}),
                'settings': orm.Dict({}),
                'metadata': {'options': {
                    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
                    'max_wallclock_seconds': 100,
                }},
            },
        },
    }
