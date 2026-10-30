# Analyse the Yambo databases with yambopy

The databases of a calculation that yambopy reads for the analysis are kept in AiiDA as two data nodes, so that you
can work on them in `verdi shell` or in a notebook without keeping the raw netCDF files:

| Node | Created from | Contains | Back to yambopy |
| --- | --- | --- | --- |
| `YamboLatticeData` (entry point `yambo.lattice`) | `ns.db1` | lattice, atoms, symmetries, k-points of the irreducible zone | `to_yambopy()` gives a `YamboLatticeDB` |
| `YamboExcitonData` (entry point `yambo.exciton`) | `ndb.BS_diago_Q*` | exciton energies, left and right residuals; the transitions table; the eigenvectors (optional) | `to_yambopy()` gives a `YamboExcitonDB` |

The nodes are outputs of the `YamboCalculation` and are propagated by the restart layer and the workflows:

- `lattice`: one `YamboLatticeData` for each calculation;
- `exciton_db.Q1`, `exciton_db.Q2`...: one `YamboExcitonData` for each Q point of a BSE calculation.

Which raw files are kept and which settings control them: [Retrieved files and settings](../reference/retrieved-files.md).

!!! note "After updating the plugin"
    The nodes are plugin data types: reinstall the plugin (`pip install -e .` or your usual command) and restart the
    daemon (`verdi daemon restart`) so that the entry points `yambo.lattice` and `yambo.exciton` are registered.

## Get the nodes

```python
from aiida import orm

node = orm.load_node(PK)  # YamboWorkflow, YamboRestart or YamboCalculation: the outputs have the same names

lattice = node.outputs.lattice
exciton = node.outputs.exciton_db.Q1

exciton.base.attributes.all  # Qpt, spin_pol, Lkind, has_table, has_eigenvectors, number_of_excitons, yambopy_version
exciton.get_arraynames()      # eigenvalues, l_residual, r_residual, (table), (eigenvectors)
```

For a `YamboWorkflow` or a `YamboConvergence` the outputs are the ones of the last calculation, as for the other
outputs (the BSE for a `bse` or `bse@qp` workflow).

## Response function and exciton energies (like `array_chi`)

The arrays `array_chi` and `array_excitonic_states` are still created. To compute the response function again, with
another broadening or energy range, use the yambopy object:

```python
ydb = exciton.to_yambopy()     # the lattice is the `lattice` output of the same calculation; or pass lattice=...

energies, chi = ydb.get_chi(emin=0, emax=10, estep=0.01, broad=0.1)   # broadening in eV
eps_2, eps_1 = chi.imag, chi.real

ydb.eigenvalues.real           # exciton energies (eV)
ydb.get_intensities()          # intensities, normalised to the brightest exciton
```

!!! warning "Check the kind of response"
    `get_chi` uses the residuals stored in the database, that is the response of the kernel of the run (`Lkind` in
    `exciton.base.attributes.get('Lkind')`). On the test run used for this feature (`Lkind='full'`) the spectrum did
    not reproduce the `eps` file written by yambo itself, which is also retrieved (`o-*eps*`). For the absorption
    spectrum use `Lkind='bar'`, and compare with the `o-*` files the first time.

## Exciton composition

The composition needs the transitions table and the eigenvectors. Yambo writes them only if `WRbsWF` is requested
(add it to the `arguments` of the BSE), and the eigenvectors are stored only if you ask for them (they are the
largest part: [sizes](../reference/retrieved-files.md#sizes)):

```python
manager.bse.settings.update({'STORE_EXCITON_EIGENVECTORS': True, 'N_EXCITON_EIGENVECTORS': 20})  # high-level API
# or on a YamboWorkflow builder: builder.bse.yambo.settings = orm.Dict({...}); on a YamboCalculation: builder.settings
```

```python
import numpy as np

exciton = node.outputs.exciton_db.Q1
assert exciton.base.attributes.get('has_table') and exciton.base.attributes.get('has_eigenvectors')

ydb = exciton.to_yambopy()
i = 0                                          # the first exciton (the eigenvalues are in the order of the file)
weights = np.abs(ydb.eigenvectors[i]) ** 2     # one weight for each transition
largest = np.argsort(weights)[::-1][:10]
for t in largest:
    k, v, c, spin = ydb.table[t]               # k-point, valence band, conduction band, spin (1-based)
    print(f'k={k} v={v} c={c} spin={spin}  weight={weights[t]:.3f}')
```

The eigenvectors are stored in single precision (`complex64`), the lowest one, and only for the first
`N_EXCITON_EIGENVECTORS` excitons if you set it. Other yambopy methods that need more than this (for example the
real-space exciton wavefunctions need the Kohn-Sham wavefunctions of the `SAVE` folder) are not covered by the nodes.

## Keep the raw files and use them later

By default the raw `ns.db1`, `ndb.BS_diago_Q*` and `ndb.HF_and_locXC` are only retrieved temporarily, to create the
nodes above. To keep them in the `retrieved` folder of the calculation, set `KEEP_RAW_DBS`:

```python
manager.bse.settings.update({'KEEP_RAW_DBS': True})  # or builder.bse.yambo.settings = orm.Dict({'KEEP_RAW_DBS': True})
```

Then, in `verdi shell` or in a notebook, copy them to a folder and use yambopy directly:

```python
import tempfile
from aiida import orm
from yambopy.dbs.excitondb import YamboExcitonDB, YamboLatticeDB

calculation = orm.load_node(CALCULATION_PK)         # the YamboCalculation (not the workflow)
retrieved = calculation.outputs.retrieved
print(retrieved.list_object_names())                 # ns.db1, ndb.BS_diago_Q1, ...

folder = tempfile.mkdtemp()
retrieved.base.repository.copy_tree(folder)          # or, in a terminal: verdi node repo dump PK FOLDER

lattice = YamboLatticeDB.from_db_file(f'{folder}/ns.db1')
ydb = YamboExcitonDB.from_db_file(lattice, filename='ndb.BS_diago_Q1', folder=folder)
```

A single file can be read without copying: `retrieved.get_object_content('ndb.BS_diago_Q1', mode='rb')`.

### Calculations run before this feature

Their `retrieved` folder already has the raw `ns.db1` and `ndb.BS_diago_Q*`. Create the nodes from them (they are
new nodes, not linked to the calculation):

```python
from aiida_yambo.data import YamboExcitonData, YamboLatticeData
from aiida_yambo.parsers.utils import read_BS

lattice_node = YamboLatticeData.from_netcdf(f'{folder}/ns.db1')
q, ydb = read_BS(folder + '/', 'ndb.BS_diago_Q1', folder, load_eigenvectors=True)
exciton_node = YamboExcitonData.from_yambopy(ydb, store_eigenvectors=True)
exciton_node.to_yambopy(lattice=lattice_node)
```
