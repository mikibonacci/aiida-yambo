# Automated convergence

`YamboConvergence` repeatedly calls `YamboWorkflow`, stores every trial, fits the convergence space, and records the selected point. The high-level manager keeps this setup readable.

Converge one physical observable at a time. Typical choices are the indirect gap (`gap_`), the Gamma gap (`gap_GG`), the lowest exciton, or the brightest exciton. A coupled parameter entry is used when two numerical parameters must be increased together.

```python
from aiida import load_profile, orm
from aiida.engine import submit
from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.yamboconvergence import YamboConvergence
from aiida_yambo.workflows.utils.parameters import ConvParam, CoupledConvParam

load_profile()
builder = YamboConvergence.get_builder_from_protocol(
    structure=orm.load_node(STRUCTURE_PK),
    code=orm.load_code('yambo'), preprocessing_code=orm.load_code('p2y'),
    pw_code=orm.load_code('pw'),
    pseudo_family='PseudoDojo/0.4/PBE/SR/standard/upf',
    protocol='fast', calc_type='qp')
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(
    builder, what='YamboConvergence', mode='builder'))
manager.set_nscf_nbnd(76)
manager.qp.set_Nb(75)
manager.qp.set_Gcut(1)
manager.parameters_space = [
    # First sample the FFT grid in a one-dimensional scan.
    ConvParam('FFTGvecs', start=10, stop=30, delta=5, max=50,
              units='Ry', conv_thr=0.5, max_concurrent_steps=2),
    # Then fit the coupled empty-state and screening-cutoff space.
    CoupledConvParam(['BndsRnXp', 'NGsBlkXp'], start=[10, 2], stop=[30, 6],
        delta=[10, 2], max=[50, 10], units=['', 'Ry'], conv_thr=10,
        mirror_values={'BndsRnXp': ['GbndRnge']}, max_iterations=2,
        max_concurrent_steps=2),
]
manager.workflow_settings = {
    'type': 'heavy', 'what': ['gap_'],
    'bands_nscf_update': 'all-at-once', 'skip_pre': False}
manager.additional_parsing = ['gap_', 'homo', 'lumo']
manager.group_label = 'convergence/silicon/qp'
run = submit(YamboConvergence,
             **manager.generate_AiiDA_inputs('YamboConvergence'))
```

For BSE target `lowest_exciton` and use `manager.bse.set_BSENb(...)`; for BSE@GW use `calc_type='bse@qp'` and configure `manager.set_QP_subsets_dict(...)` first.

The `start`, `stop`, and `delta` values define the initial grid. `max` prevents the algorithm from exploring unreasonably expensive calculations, while `conv_thr` defines the tolerance in the observable's units. Start with a small grid for a test run, then widen it for production.

```python
import pandas as pd
infos = run.outputs.infos.get_dict()
history = pd.DataFrame(run.outputs.history.get_dict())
converged = history[history['useful'] == True]
print(infos)
print(converged[['parameters_studied', 'gap_']])
```

Use `heavy` when the final calculation should already contain converged values, and `cheap` for less expensive intermediate trials.
