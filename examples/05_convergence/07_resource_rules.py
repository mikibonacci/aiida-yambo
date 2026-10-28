"""Resources which depend on the values of the parameters, in a convergence: ``resource_rules``.

In a convergence the cost of the calculations grows with the parameters: few cores for the small cutoffs, many
nodes for the large numbers of bands. Each rule says WHEN (the values of the parameters), FOR which steps
(scf, nscf, qp, bse) and USE which resources; for each step the first rule which matches is used, and the options go
back to the ones set on the manager at every point. See the docs ("Resources and parallelism") for all the keys.

Here the default parameters of the protocol are converged (FFTGvecs, the k-points density, then the coupled bands and
cutoff) with three rules. The rules are checked when the workflow is submitted.

Status: tested here (the inputs and the rules are valid); not yet run on a cluster.
Edit: the code labels and the numbers (they must fit your machine and queue).
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.yamboconvergence import YamboConvergence

load_profile()

# ---------------------------------------------------------------------------
# 1. Settings: edit them
# ---------------------------------------------------------------------------
YAMBO_CODE = 'yambo@mycluster'
P2Y_CODE = 'p2y@mycluster'
PW_CODE = 'pw@mycluster'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'
GROUP_LABEL = 'convergence/silicon/qp-resources'

structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

builder = YamboConvergence.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code(YAMBO_CODE),
    preprocessing_code=orm.load_code(P2Y_CODE),
    pw_code=orm.load_code(PW_CODE),
    pseudo_family=PSEUDO_FAMILY,
    calc_type='qp',
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboConvergence', mode='builder'))
manager.scf.set_kmesh([2, 2, 2])
manager.nscf.set_kmesh([2, 2, 2])
manager.set_nscf_nbnd(value=76)
manager.qp.set_Nb(value=75)
manager.qp.set_Gcut(value=1)
manager.additional_parsing = ['gap_', 'homo', 'lumo']
manager.group_label = GROUP_LABEL

# What every step uses if no rule matches: the values set on the manager.
manager.scf.set_resources(nodes=1, mpi=4)
manager.nscf.set_resources(nodes=1, mpi=4)
manager.qp.set_resources(nodes=1, mpi=8, time=3600)

# --8<-- [start:rules]
# ---------------------------------------------------------------------------
# 2. The rules. WHEN: the Yambo variables (BndsRnXp, NGsBlkXp, FFTGvecs...), `bands`, `gcut`, `kpoints`,
#    `kpoint_density`, `kpoint_mesh`; a value (must be equal) or {'min': .., 'max': ..}. No `when`: always.
# ---------------------------------------------------------------------------
manager.resource_rules = [
    # many bands: the GW step needs more nodes and time, and the MPI tasks distributed by the plugin
    {'when': {'BndsRnXp': {'min': 200}},
     'for': ['qp'],
     'use': {'num_machines': 4, 'num_mpiprocs_per_machine': 32, 'max_wallclock_seconds': 4 * 3600,
             'parallelism': 'auto', 'max_walltime': 8 * 3600}},
    # intermediate numbers of bands
    {'when': {'BndsRnXp': {'min': 100, 'max': 199}},
     'for': ['qp'],
     'use': {'num_machines': 2, 'num_mpiprocs_per_machine': 16, 'max_wallclock_seconds': 2 * 3600}},
    # a dense k-points mesh: the DFT steps on more nodes
    {'when': {'kpoints': {'min': 27}},
     'for': ['scf', 'nscf'],
     'use': {'num_machines': 2}},
]
# --8<-- [end:rules]

# ---------------------------------------------------------------------------
# 3. Submit
# ---------------------------------------------------------------------------
node = submit(YamboConvergence, **manager.generate_AiiDA_inputs(what='YamboConvergence'))
print(f'YamboConvergence with resource rules submitted, pk = {node.pk}')
print(f'See what each calculation used:  verdi process show <pk of a YamboWorkflow of {node.pk}>')
