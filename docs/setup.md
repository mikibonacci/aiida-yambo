# Configure AiiDA and the codes

The plugin uses three code entry points:

| Code | Entry point | Role |
| --- | --- | --- |
| Quantum ESPRESSO | `quantumespresso.pw` | SCF and NSCF DFT starting point |
| `p2y` | `yambo.yambo` | Convert Quantum ESPRESSO wavefunctions to a YAMBO database |
| YAMBO | `yambo.yambo` | GW, BSE, and post-processing |

Register the executables with `verdi code create` (or `verdi code setup` on older AiiDA versions):

```bash
verdi code create core.code.installed \
  --label yambo --computer cluster \
  --filepath /path/to/bin/yambo --input-plugin yambo.yambo

verdi code create core.code.installed \
  --label p2y --computer cluster \
  --filepath /path/to/bin/p2y --input-plugin yambo.yambo
```

Configure the Quantum ESPRESSO code with the `quantumespresso.pw` entry point. Use `verdi code show yambo` to verify the executable and module/prepend text.

## Minimal Python configuration

```python
from aiida import load_profile, orm
from aiida_yambo.workflows.yambowf import YamboWorkflow

load_profile()
structure = orm.load_node(STRUCTURE_PK)
builder = YamboWorkflow.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code('yambo'),
    preprocessing_code=orm.load_code('p2y'),
    pw_code=orm.load_code('pw'),
    pseudo_family='PseudoDojo/0.4/PBE/SR/standard/upf',
    protocol='fast',
    calc_type='qp',
)
```

Set scheduler resources on every nested calculation:

```python
resources = {
    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 4},
    'withmpi': True,
    'queue_name': 'compute',
}
builder.scf.pw.metadata.options = resources
builder.nscf.pw.metadata.options = resources
builder.qp.yambo.metadata.options = resources
```

Inspect the builder before submission: the exact nested namespace depends on the selected workflow and plugin version.
