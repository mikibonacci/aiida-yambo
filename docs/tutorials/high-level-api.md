# Prepare inputs with the high-level API

The recommended user interface is `YamboInputManager`. It sits between the AiiDA builder and your scientific choices: protocols create valid defaults, while the manager exposes convenient Python methods for k-point meshes, band counts, cutoffs, resources, parent runs, QP subsets, and parsed quantities.

You only need AiiDA data nodes for the objects that must come from the database: the structure, codes, and optional parent folders. You do not need to manually assemble nested `orm.Dict`, `KpointsData`, or scheduler metadata objects for ordinary workflows.

## Create a manager from a protocol builder

There are two layers in this example. `get_builder_from_protocol` knows the workflow topology and supplies tested defaults. `YamboInputManager` then gives you a small, typed interface for changing the physics and resources. The conversion back to AiiDA happens only immediately before submission.

```python
from aiida import load_profile, orm
from aiida.engine import submit
from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.yambowf import YamboWorkflow

load_profile()

# The structure and codes are the few database objects we still need.
builder = YamboWorkflow.get_builder_from_protocol(
    structure=orm.load_node(STRUCTURE_PK),
    code=orm.load_code('yambo'),
    preprocessing_code=orm.load_code('p2y'),
    pw_code=orm.load_code('pw'),
    pseudo_family='PseudoDojo/0.4/PBE/SR/standard/upf',
    protocol='fast',
    calc_type='qp',
)

# Flatten the nested workflow inputs into the user-facing manager.
inputs = YamboInputManager.get_nested_inputs(
    builder, what='YamboWorkflow', mode='builder'
)
manager = YamboInputManager(**inputs)
```

`manager.scf`, `manager.nscf`, `manager.qp`, and `manager.bse` are typed helpers. Changes remain ordinary Python data until the final conversion:

```python
# DFT settings: the SCF and NSCF meshes should normally match.
manager.scf.set_kmesh([2, 2, 2])
manager.nscf.set_kmesh([2, 2, 2])

# Resource settings are expressed as simple integers.
manager.scf.set_resources(mpi=2)
manager.scf.set_npools(2)

# YAMBO needs enough empty states in the NSCF calculation.
manager.set_nscf_nbnd(76)
manager.qp.set_Nb(value=75)
manager.qp.set_Gcut(value=1)

# Ask the parser to expose quantities useful for the next analysis step.
manager.additional_parsing = ['gap_', 'homo', 'lumo']

aiida_inputs = manager.generate_AiiDA_inputs(what='YamboWorkflow')
run = submit(YamboWorkflow, **aiida_inputs)
```

!!! tip
    This is the pattern of the scripts in `examples/` (settings on top, protocol builder, manager, submit). Keep the protocol builder and manager construction in one place, then add small variations for each scientific case.

## Resources without nested AiiDA metadata

The manager accepts compact resource settings:

```python
manager.scf.set_resources(nodes=1, mpi=4, threads=2, time=7200)
manager.qp.set_resources(nodes=2, mpi=8, threads=2, time=14400)
manager.qp.set_npools(4)
```

Use `set_prepend_text` or `set_scheduler_commands` when the site requires modules or scheduler directives. Apply the settings to each namespace that launches a calculation; the manager does not silently copy resource settings between SCF, NSCF, QP, and BSE.

## Parent calculations

With no parent, the workflow creates its DFT starting point. To skip already completed stages, pass the remote folder owner through the manager:

```python
manager.set_parent_folder(orm.load_node(NSCF_PK).outputs.remote_folder)
```

Use `manager.remove_parent_folder()` to return to a from-scratch workflow. The supported practical modes are:

| Parent | Work skipped |
| --- | --- |
| none | SCF, NSCF, preprocessing, and MBPT are created |
| SCF remote folder | SCF is reused; NSCF and MBPT are created |
| NSCF remote folder | DFT is reused; preprocessing and MBPT are created |
| previous YAMBO remote folder | Continue YAMBO work from the existing database |

## The methods you will use most

| Goal | Manager method |
| --- | --- |
| Change QE k-points | `manager.scf.set_kmesh`, `manager.nscf.set_kmesh` |
| Set NSCF bands | `manager.set_nscf_nbnd` |
| Set GW bands/cutoff | `manager.qp.set_Nb`, `manager.qp.set_Gcut` |
| Set BSE bands/cutoff | `manager.bse.set_BSENb`, `manager.bse.set_BSEGcut` |
| Select BSE transitions | `manager.bse.set_BSEbands([valence, conduction])` |
| Select parsed values | `manager.additional_parsing = [...]` |
| Split QP calculations | `manager.set_QP_subsets_dict(...)` |
| Inspect consistency | `manager.has_enough_nbnd`, `manager.Nb`, `manager.Gcut` |

When a feature is not wrapped yet, use `set_variable(name, value)` as the escape hatch. For example, `manager.qp.set_variable('FFTGvecs', [24, 'Ry'])` changes one YAMBO variable while preserving the rest of the protocol. Direct manipulation of the final AiiDA inputs should be the last resort.
