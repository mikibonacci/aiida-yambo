# Prerequisites: structure, pseudopotentials, and DFT starting point

Before running `YamboWorkflow`, prepare three things in AiiDA:

1. a `StructureData` node;
2. a norm-conserving pseudopotential family;
3. an SCF/NSCF Quantum ESPRESSO starting point, unless the workflow will create it for you.

This page expands the original `tutorials-aiida-yambo/prerequisites/` notebooks into one reproducible Markdown tutorial. It uses silicon as a small example; the same steps apply to hBN, Cu, or any structure supported by Quantum ESPRESSO.

!!! note
    You need a configured AiiDA profile, a `pw.x` code, and a running daemon. The YAMBO interface requires norm-conserving pseudopotentials.

## 1. Create and store a structure

### Read a Quantum ESPRESSO input with ASE

```python
from ase.io.espresso import read_espresso_in

silicon = read_espresso_in('tutorials-aiida-yambo/data/silicon.in')
print(silicon)
```

Convert the ASE object to an AiiDA node and store it:

```python
from aiida import load_profile, orm

load_profile()
structure = orm.StructureData(ase=silicon)
structure.store()
print(structure.pk, structure.uuid)
```

You can also create a structure from CIF, XYZ, or another supported format using ASE or pymatgen. The final object passed to the workflow is `orm.StructureData`.

### Organise structures in a group

```python
group, _ = orm.Group.collection.get_or_create('tutorial/Silicon/bulk')
group.add_nodes(structure)
```

Query it later without remembering a primary key:

```python
qb = orm.QueryBuilder()
qb.append(orm.Group, filters={'label': 'tutorial/Silicon/bulk'}, tag='group')
qb.append(orm.StructureData, with_group='group')
structure = qb.all(flat=True)[0]
```

## 2. Install norm-conserving pseudopotentials

List families already available in the profile:

```bash
verdi data core.upf listfamilies
```

The examples use:

```text
PseudoDojo/0.4/PBE/SR/standard/upf
```

Install a compatible family with `aiida-pseudo` if it is not present. For custom UPF files, consult the [aiida-pseudo guide](https://aiida-pseudo.readthedocs.io/en/latest/howto.html), then verify coverage:

```python
family = orm.load_group('PseudoDojo/0.4/PBE/SR/standard/upf')
pseudos = family.get_pseudos(structure=structure)
print(pseudos.keys())
```

## 3. Run the SCF calculation

The SCF and NSCF calculations are provided by `aiida-quantumespresso`:

```python
from aiida.engine import submit
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

pw = orm.load_code('pw')
resources = {
    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 4},
    'withmpi': True,
}

scf = PwBaseWorkChain.get_builder_from_protocol(
    code=pw, structure=structure, protocol='fast',
    electronic_type='insulator',
    pseudo_family='PseudoDojo/0.4/PBE/SR/standard/upf',
)
scf.pw.metadata.options = resources
scf.clean_workdir = orm.Bool(False)
scf_run = submit(scf)
print(scf_run.pk)
```

Keeping `clean_workdir=False` is useful while developing because the remote folder is needed by the NSCF step.

## 4. Run the NSCF calculation

The NSCF step supplies wavefunctions and empty states for YAMBO. Set the SCF remote folder as its parent and increase `nbnd`:

```python
nscf = PwBaseWorkChain.get_builder_from_protocol(
    code=pw, structure=structure, protocol='fast',
    electronic_type='insulator',
    pseudo_family='PseudoDojo/0.4/PBE/SR/standard/upf',
    overrides={'pw': {'parameters': {
        'CONTROL': {
            'calculation': 'nscf', 'restart_mode': 'restart',
            'wf_collect': True,
        },
        'SYSTEM': {'nbnd': 200},
    }}},
)
nscf.pw.parent_folder = scf_run.outputs.remote_folder
nscf.pw.metadata.options = resources
nscf.clean_workdir = orm.Bool(False)
nscf_run = submit(nscf)
print(nscf_run.pk)
```

For a denser mesh, replace the protocol-generated k-points:

```python
kpoints = orm.KpointsData()
kpoints.set_kpoints_mesh([4, 4, 4])
nscf.kpoints = kpoints
```

The NSCF number of bands must be at least as large as the largest YAMBO band window.

## 5. Inspect and monitor runs

```bash
verdi process list
verdi process show <pk>
verdi process report <pk>
```

```python
print(scf_run.is_finished_ok)
print(nscf_run.is_finished_ok)
if nscf_run.is_finished_ok:
    print(nscf_run.outputs.output_parameters.get_dict())
```

Do not pass a failed or unfinished parent to a YAMBO workflow. Fix the DFT calculation first, or let `YamboWorkflow` create a fresh starting point.

## 6. Continue with the high-level YAMBO API

Once the structure, codes, and optional NSCF parent are available, continue with the [high-level input API](high-level-api.md):

```python
from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.yambowf import YamboWorkflow

builder = YamboWorkflow.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code('yambo'),
    preprocessing_code=orm.load_code('p2y'),
    pw_code=pw,
    pseudo_family='PseudoDojo/0.4/PBE/SR/standard/upf',
    protocol='fast', calc_type='qp',
    parent_folder=nscf_run.outputs.remote_folder,
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(
    builder, what='YamboWorkflow', mode='builder'))
manager.set_parent_folder(nscf_run.outputs.remote_folder)
```

From here, configure `manager.qp`, `manager.bse`, resources, QP subsets, and parsed quantities without manually constructing nested AiiDA data nodes.
