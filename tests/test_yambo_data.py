"""YamboLatticeData and YamboExcitonData, the files retrieved temporarily and the parser which creates the nodes.

`tests/data/bse_without_table` has the `ns.db1` and `ndb.BS_diago_Q1` of a BSE run (no `WRbsWF`: no `BS_TABLE`);
`tests/data/bse_calculation` the other files that run retrieved (reports, logs, o- files).
"""
import pathlib
import shutil

import numpy as np
import pytest

from aiida import orm
from aiida.common.links import LinkType

DATA = pathlib.Path(__file__).parent / 'data'
DBS = DATA / 'bse_without_table'
CALCULATION = DATA / 'bse_calculation'


@pytest.fixture
def lattice_node(aiida_profile):
    from aiida_yambo.data import YamboLatticeData

    return YamboLatticeData.from_netcdf(DBS / 'ns.db1')


def _with_table_and_eigenvectors(tmp_path, neigs=8, ntransitions=8):
    """A copy of the BSE database with the transitions table and (random, orthonormal) eigenvectors, as `WRbsWF` writes."""
    from netCDF4 import Dataset

    folder = tmp_path / 'wf'
    folder.mkdir()
    shutil.copy(DBS / 'ndb.BS_diago_Q1', folder)
    shutil.copy(DBS / 'ns.db1', folder)
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(neigs, ntransitions)) + 1j * rng.normal(size=(neigs, ntransitions)))
    with Dataset(folder / 'ndb.BS_diago_Q1', 'a') as database:
        database.createDimension('four', 4)
        database.createDimension('ntransitions', ntransitions)
        database.createDimension('neigs', neigs)
        database.createDimension('two', 2)
        table = database.createVariable('BS_TABLE', 'f8', ('four', 'ntransitions'))
        table[:] = np.array([[k + 1, 4, 5, 1] for k in range(ntransitions)]).T
        eigenstates = database.createVariable('BS_EIGENSTATES', 'f4', ('neigs', 'ntransitions', 'two'))
        eigenstates[:] = np.stack([q.real, q.imag], axis=-1)
    return folder, q


def test_lattice_data_round_trip(lattice_node):
    from yambopy.dbs.latticedb import YamboLatticeDB

    lattice_node.store()
    expected = YamboLatticeDB.from_db_file(filename=str(DBS / 'ns.db1'))
    lattice = orm.load_node(lattice_node.pk).to_yambopy()

    for name in ('lat', 'alat', 'sym_car', 'iku_kpoints', 'car_atomic_positions', 'atomic_numbers'):
        np.testing.assert_allclose(getattr(lattice, name), getattr(expected, name), err_msg=name)
    assert (lattice.time_rev, lattice.nelectrons, lattice.spinor_components) == \
        (expected.time_rev, expected.nelectrons, expected.spinor_components)
    # the irreducible k-points are stored, the expansion is done when the lattice is read back
    assert len(lattice_node.get_array('iku_kpoints')) == expected.ibz_nkpoints


def test_exciton_data_without_table_round_trip(lattice_node):
    from aiida_yambo.data import YamboExcitonData
    from aiida_yambo.parsers.utils import BS_arrays, read_BS

    q, original = read_BS(str(DBS) + '/', 'ndb.BS_diago_Q1', str(DBS))
    node = YamboExcitonData.from_yambopy(original).store()
    node = orm.load_node(node.pk)

    assert (q, node.base.attributes.get('Qpt'), node.base.attributes.get('has_table'),
            node.base.attributes.get('has_eigenvectors')) == ('Q1', '1', False, False)
    assert set(node.get_arraynames()) >= {'eigenvalues', 'l_residual', 'r_residual'}
    assert not {'table', 'eigenvectors'} & set(node.get_arraynames())

    chi, excitons = BS_arrays(node.to_yambopy(lattice=lattice_node))
    expected_chi, expected_excitons = BS_arrays(original)
    np.testing.assert_allclose(excitons['energies'], expected_excitons['energies'])
    np.testing.assert_allclose(excitons['intensities'], expected_excitons['intensities'])
    np.testing.assert_allclose(chi['eps_2'], expected_chi['eps_2'])


@pytest.mark.parametrize('store, neigs, expected', [(False, None, None), (True, None, 8), (True, 3, 3)])
def test_exciton_data_eigenvectors_are_optional_truncated_and_single_precision(tmp_path, lattice_node, store, neigs, expected):
    from aiida_yambo.data import YamboExcitonData
    from aiida_yambo.parsers.utils import read_BS

    folder, eigenvectors = _with_table_and_eigenvectors(tmp_path)
    _, original = read_BS(str(folder) + '/', 'ndb.BS_diago_Q1', str(folder), load_eigenvectors=True)
    node = YamboExcitonData.from_yambopy(original, store_eigenvectors=store, neigs=neigs).store()

    assert node.base.attributes.get('has_table') is True
    assert node.base.attributes.get('has_eigenvectors') is (expected is not None)
    if expected is None:
        assert 'eigenvectors' not in node.get_arraynames()
        return
    stored = node.get_array('eigenvectors')
    assert stored.dtype == np.complex64 and stored.shape == (expected, 8)
    np.testing.assert_allclose(stored, eigenvectors[:expected], atol=1e-6)
    np.testing.assert_array_equal(node.to_yambopy(lattice=lattice_node).table, original.table)


def test_the_raw_databases_are_retrieved_temporarily_unless_asked():
    from aiida_yambo.calculations.yambo import split_retrieve_list

    paths = ['SAVE/ns.db1', 'aiida.out/ndb.BS_diago_Q*', 'aiida.out/ndb.HF_and_locXC', 'aiida.out/ndb.QP', ['x', 'y', 0]]

    permanent, temporary = split_retrieve_list(paths)
    assert temporary == ['SAVE/ns.db1', 'aiida.out/ndb.BS_diago_Q*', 'aiida.out/ndb.HF_and_locXC']
    assert permanent == ['aiida.out/ndb.QP', ['x', 'y', 0]]

    permanent, temporary = split_retrieve_list(paths, keep_raw_dbs=True)
    assert (permanent, temporary) == (paths, [])


def test_additional_retrieve_list_entries():
    from aiida.common.exceptions import InputValidationError

    from aiida_yambo.calculations.yambo import additional_retrieve_entries

    assert additional_retrieve_entries(['r-*', 'LOG/l-*01', ['aiida.out/a', 'a_local', 0]]) == \
        ['r-*', 'LOG/l-*01', ['aiida.out/a', 'a_local', 0]]
    assert additional_retrieve_entries([]) == []
    for wrong in ('r-*', [1], [['a', 'b']], [['a', 'b', 'c']]):
        with pytest.raises(InputValidationError):
            additional_retrieve_entries(wrong)


@pytest.fixture
def bse_calculation(aiida_localhost, monkeypatch):
    """The `CalcJobNode` of the BSE run, its `retrieved` folder (without the databases) and a folder with the databases."""
    from types import SimpleNamespace

    from aiida_yambo.parsers import parsers

    parameters = {'arguments': ['em1s', 'bse', 'bss', 'optics', 'rim_cut', 'dipoles'], 'variables': {}}
    monkeypatch.setattr(parsers, 'find_pw_parent', lambda calculation: SimpleNamespace(
        inputs=SimpleNamespace(structure=SimpleNamespace(cell=np.eye(3).tolist()))))

    def create(settings=None):
        calculation = orm.CalcJobNode(computer=aiida_localhost, process_type='aiida.calculations:yambo.yambo')
        calculation.set_option('max_wallclock_seconds', 3600)
        for label, value in (('parameters', parameters), ('settings', settings or {})):
            calculation.base.links.add_incoming(orm.Dict(value).store(), LinkType.INPUT_CALC, label)
        calculation.store()
        retrieved = orm.FolderData(tree=str(CALCULATION))
        retrieved.base.links.add_incoming(calculation, LinkType.CREATE, 'retrieved')
        retrieved.store()
        return calculation, retrieved

    return create


def _parse(calculation, retrieved, temporary_folder):
    from aiida_yambo.parsers.parsers import YamboParser

    parser = YamboParser(calculation)
    exit_code = parser.parse(retrieved=retrieved, retrieved_temporary_folder=str(temporary_folder))
    return exit_code, parser.outputs


def test_parser_creates_the_lattice_and_exciton_nodes_from_the_temporary_folder(bse_calculation):
    calculation, retrieved = bse_calculation()

    exit_code, outputs = _parse(calculation, retrieved, DBS)

    assert exit_code is None
    assert 'ns.db1' not in retrieved.list_object_names() and 'ndb.BS_diago_Q1' not in retrieved.list_object_names()
    assert type(outputs['lattice']).__name__ == 'YamboLatticeData'
    exciton = outputs['exciton_db.Q1']
    assert type(exciton).__name__ == 'YamboExcitonData' and not exciton.base.attributes.get('has_eigenvectors')
    assert {'array_chi', 'array_excitonic_states', 'array_eps', 'array_eels'} <= set(outputs)


def test_parser_stores_the_eigenvectors_when_asked(bse_calculation, tmp_path):
    folder, eigenvectors = _with_table_and_eigenvectors(tmp_path)
    calculation, retrieved = bse_calculation({'STORE_EXCITON_EIGENVECTORS': True, 'N_EXCITON_EIGENVECTORS': 2})

    exit_code, outputs = _parse(calculation, retrieved, folder)

    assert exit_code is None
    stored = outputs['exciton_db.Q1'].get_array('eigenvectors')
    assert stored.dtype == np.complex64 and stored.shape == (2, 8)


def test_the_new_outputs_reach_the_workflows(aiida_profile):
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence
    from aiida_yambo.workflows.yambowf import YamboWorkflow
    from aiida_yambo.workflows.yamborestart import YamboRestart

    for process in (YamboRestart, YamboWorkflow, YamboConvergence):
        outputs = process.spec().outputs
        assert 'lattice' in outputs and 'exciton_db' in outputs, process.__name__
