# GW with `YamboWorkflow`

This is the main user-facing workflow. Start from a structure and let the protocol create the DFT and YAMBO inputs; use `YamboInputManager` for scientific choices.

The example below is intentionally small: it is suitable for checking that the installation works, not for production convergence. Increase the k-point mesh, NSCF bands, and YAMBO cutoffs only after this first run finishes successfully.

```python
from aiida import load_profile, orm
from aiida.engine import submit
from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.yambowf import YamboWorkflow

load_profile()
builder = YamboWorkflow.get_builder_from_protocol(
    structure=orm.load_node(STRUCTURE_PK),
    code=orm.load_code('yambo'), preprocessing_code=orm.load_code('p2y'),
    pw_code=orm.load_code('pw'),
    pseudo_family='PseudoDojo/0.4/PBE/SR/standard/upf',
    protocol='fast', calc_type='qp',
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(
    builder, what='YamboWorkflow', mode='builder'))

# DFT starting point: SCF and NSCF on the same mesh.
manager.scf.set_kmesh([2, 2, 2])
manager.nscf.set_kmesh([2, 2, 2])
# These settings are examples; use resources appropriate for your computer.
manager.scf.set_resources(mpi=2)
manager.scf.set_npools(2)
# The NSCF must contain at least all bands requested by GW.
manager.set_nscf_nbnd(76)
manager.qp.set_Nb(75)
manager.qp.set_Gcut(1)
# Request values that are parsed into output_ywfl_parameters.
manager.additional_parsing = ['gap_', 'gap_GG', 'homo', 'lumo']

run = submit(YamboWorkflow, **manager.generate_AiiDA_inputs('YamboWorkflow'))
print(run.pk)
```

## Start from scratch or reuse a parent

```python
manager.remove_parent_folder()  # SCF -> NSCF -> p2y -> GW
manager.set_parent_folder(orm.load_node(PARENT_PK).outputs.remote_folder)
```

The parent can be a completed SCF, NSCF, or previous YAMBO run. The workflow detects which stages are missing and submits only those stages.

This is the main advantage over manually chaining calculations: the provenance remains the same, but the user describes the available starting point rather than reconstructing every intermediate input.

## Common variations

```python
manager.scf.set_kmesh([4, 4, 4])
manager.nscf.set_kmesh([4, 4, 4])
manager.set_nscf_nbnd(120)
manager.qp.set_Nb(110)
manager.qp.set_Gcut(4)
manager.set_pseudo_family('PseudoDojo/0.4/LDA/SR/standard/upf')
print(manager.Nb, manager.nscf_nbnd, manager.has_enough_nbnd)
```

Inspect results with:

```python
run.outputs.output_ywfl_parameters.get_dict()
run.outputs.bands_quasiparticle
```

Monitor with `verdi process report <pk>`. If a run fails, inspect the report before changing physics parameters; many failures are scheduler, memory, or parent-folder problems.
