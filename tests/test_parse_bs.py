"""`parse_BS` reads the BSE database also when yambo did not write the transitions table.

`tests/data/bse_without_table` is a BSE run (silicon, 8 eigenvalues, no `WRbsWF`): `ndb.BS_diago_Q1` has no `BS_TABLE`
and no eigenvectors, and yambopy's `YamboExcitonDB.from_db_file` raises `KeyError: 'BS_TABLE'` on it.
"""
import pathlib

import numpy as np
import pytest

DATA = pathlib.Path(__file__).parent / 'data' / 'bse_without_table'


@pytest.fixture
def parsed():
    from aiida_yambo.parsers.utils import parse_BS

    return parse_BS(str(DATA) + '/', 'ndb.BS_diago_Q1', str(DATA))


def test_parse_bs_without_the_transitions_table(parsed):
    q, chi, excitons = parsed

    assert q == 'Q1'
    np.testing.assert_allclose(excitons['energies'],
                               [2.963, 3.198, 3.230, 3.250, 4.021, 4.049, 4.066, 39.373], atol=2e-3)
    assert excitons['intensities'].shape == (8,)
    assert excitons['intensities'].max() == pytest.approx(1.0)
    assert set(chi) == {'eV', 'eps_2', 'eps_1'}
    assert len(chi['eV']) == len(chi['eps_2']) == len(chi['eps_1'])


def test_other_keyerrors_are_not_hidden(monkeypatch):
    from yambopy.dbs.excitondb import YamboExcitonDB

    from aiida_yambo.parsers import utils

    def broken(*args, **kwargs):
        raise KeyError('BS_Energies')

    monkeypatch.setattr(YamboExcitonDB, 'from_db_file', broken)
    with pytest.raises(KeyError):
        utils.parse_BS(str(DATA) + '/', 'ndb.BS_diago_Q1', str(DATA))
