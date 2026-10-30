"""SCF and NSCF with Quantum ESPRESSO: a starting point (parent) for Yambo calculations.

You do not need this to run Yambo: ``YamboWorkflow`` performs the SCF and NSCF itself when it starts from a
structure (see 02_gw/01_gw_from_scratch.py). Use this script when you want the DFT calculations on their own,
for instance to reuse them as ``parent_folder`` of several Yambo runs (see 02_gw/02_gw_from_parent.py).

Flow, in two runs of this script:

    1. SCF_PK = None   ->  submits the SCF
    2. SCF_PK = <pk>   ->  submits the NSCF on top of the finished SCF

Status: the same calculations, as made inside ``YamboWorkflow``, ran on a cluster (silicon, PseudoDojo PBE).
Edit: PW_CODE, PSEUDO_FAMILY and the scheduler options.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from ase.build import bulk

load_profile()

# ---------------------------------------------------------------------------
# 1. Settings: edit them
# ---------------------------------------------------------------------------
PW_CODE = 'qe-7.5-gcc-spack@corvina-direct'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'  # norm-conserving: `aiida-pseudo install pseudo-dojo`
KMESH = [2, 2, 2]  # tiny, to test: use a denser mesh for production
NSCF_NBND = 76  # the NSCF must contain all the bands the Yambo calculation will use
SCF_PK = 39768  # None: submit the SCF. The pk of the finished SCF: submit the NSCF

# ---------------------------------------------------------------------------
# 2. Structure (silicon, from ASE)
# ---------------------------------------------------------------------------
structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

# ---------------------------------------------------------------------------
# 3. Inputs
# ---------------------------------------------------------------------------
kpoints = orm.KpointsData()
kpoints.set_kpoints_mesh(KMESH)

# Yambo needs the symmetries of the Hamiltonian to be symmorphic.
overrides = {'pseudo_family': PSEUDO_FAMILY, 'pw': {'parameters': {'SYSTEM': {'force_symmorphic': True}}}}
if SCF_PK is not None:
    overrides['pw']['parameters']['CONTROL'] = {'calculation': 'nscf', 'wf_collect': True}
    overrides['pw']['parameters']['SYSTEM']['nbnd'] = NSCF_NBND

builder = PwBaseWorkChain.get_builder_from_protocol(orm.load_code(PW_CODE), structure, overrides=overrides)
del builder.kpoints_distance  # we give the mesh explicitly
builder.kpoints = kpoints

if SCF_PK is not None:
    # the NSCF reads the density of the SCF
    builder.pw.parent_folder = orm.load_node(SCF_PK).outputs.remote_folder

# ---------------------------------------------------------------------------
# 4. Scheduler options: edit them to match your cluster
# ---------------------------------------------------------------------------
builder.pw.metadata.options.resources = {'num_machines': 1, 'num_mpiprocs_per_machine': 2}
builder.pw.metadata.options.max_wallclock_seconds = 3600
# builder.pw.metadata.options.queue_name = 'my_queue'
# builder.pw.metadata.options.account = 'my_account'
builder.pw.parallelization = orm.Dict({'npool': 2})

# ---------------------------------------------------------------------------
# 5. Submit
# ---------------------------------------------------------------------------
node = submit(builder)
print(f"{'SCF' if SCF_PK is None else 'NSCF'} submitted, pk = {node.pk}")
print(f'Monitor with:  verdi process status {node.pk}')

# ---------------------------------------------------------------------------
# 6. After it finishes
# ---------------------------------------------------------------------------
# node = orm.load_node(<PK>)
# print(node.is_finished_ok, node.exit_status)
# print(node.outputs.output_parameters.get_dict()['fermi_energy'])
# node.outputs.remote_folder        # the folder to give as `parent_folder` to the Yambo examples
