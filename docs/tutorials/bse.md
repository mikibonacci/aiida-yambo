# BSE and BSE@GW

Use the same high-level interface as for GW. Change `calc_type` and customise the `bse` or `qp` namespace.

There are two scientifically different routes. A scissor correction is inexpensive and useful for testing optical-input choices. BSE@GW explicitly computes quasiparticle corrections first and is the route to use when the optical spectrum must be based on calculated GW energies.

## BSE with a scissor correction

Here `KfnQP_E` shifts and stretches the DFT energies before the BSE calculation. The indices in `BSEBands` are the valence and conduction bands included in the excitonic problem; choose them for the material and the energy window of interest.

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
    protocol='fast', calc_type='bse')
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(
    builder, what='YamboWorkflow', mode='builder'))
manager.bse.set_BSENb(value=75)
manager.bse.set_BSEGcut(value=4)
manager.bse.set_BSEbands([4, 5])
manager.bse.set_variable('KfnQP_E', [[0.5, 1, 1], ''])
manager.additional_parsing = ['lowest_exciton', 'brightest_exciton']
run = submit(YamboWorkflow, **manager.generate_AiiDA_inputs('YamboWorkflow'))
```

## BSE on explicit GW corrections

`bse@qp` computes QP corrections, splits them, merges the databases, and then runs BSE:

```python
builder = YamboWorkflow.get_builder_from_protocol(
    structure=orm.load_node(STRUCTURE_PK),
    code=orm.load_code('yambo'), preprocessing_code=orm.load_code('p2y'),
    pw_code=orm.load_code('pw'),
    pseudo_family='PseudoDojo/0.4/PBE/SR/standard/upf',
    protocol='fast', calc_type='bse@qp')
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(
    builder, what='YamboWorkflow', mode='builder'))
manager.qp.set_Nb(value=75)
manager.qp.set_Gcut(value=1)
manager.bse.set_BSENb(value=75)
manager.bse.set_BSEbands([4, 5])
manager.set_QP_subsets_dict(ki=1, kf=5, bi=3, bf=6,
                             qp_per_subset=8, parallel_runs=1)
manager.additional_parsing = ['gap_', 'lowest_exciton']
run = submit(YamboWorkflow, **manager.generate_AiiDA_inputs('YamboWorkflow'))
```

Other supported QP selections include energy windows and explicit QP lists:

```python
manager.set_QP_subsets_dict(qp_per_subset=8, parallel_runs=1, range_QP=5.0)
manager.set_QP_subsets_dict(qp_per_subset=8, parallel_runs=1,
                            explicit=[[1, 1, 1, 2], [1, 1, 2, 2]])
```

!!! tip "More on the QP splitter"
    The QP selection (`boundaries`, `range_QP`, `explicit`, `consider_only`), the resources and the walltime of the
    splitted runs, `parallelism='auto'`, the cleaning of the splitted folders and `extend_db` are explained in the
    how-to [Split the GW calculation (QP splitter)](../how-to/qp-splitter.md). The complete list of the
    `QP_subsets_dict` keys is in [Options and exit codes](../reference/options-and-exit-codes.md#qp_subsets_dict).

The resulting spectrum is available through `run.outputs.array_chi`, while requested exciton values are in `output_ywfl_parameters`.

For a first check, inspect the energy grid and imaginary dielectric function:

```python
energy = run.outputs.array_chi.get_array('eV')
epsilon2 = run.outputs.array_chi.get_array('eps_2')
print(energy[:5], epsilon2[:5])
```
