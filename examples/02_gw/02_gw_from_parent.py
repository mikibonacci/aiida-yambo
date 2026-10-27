"""GW with ``YamboWorkflow``, starting from a calculation which is already done.

The parent is a finished calculation with a remote folder; the workflow runs only the steps which are missing:

    PARENT_KIND = 'scf'    ->  nscf -> p2y -> GW
    PARENT_KIND = 'nscf'   ->  p2y -> GW
    PARENT_KIND = 'yambo'  ->  GW   (the parent is a previous Yambo run: p2y is not repeated)

If the NSCF of the parent has fewer bands than the GW needs (NSCF_NBND < QP_NB), the workflow notices it and
repeats the NSCF with enough bands, starting from the SCF.

Status: the three kinds of parent ran on a cluster (silicon, PseudoDojo PBE).
Edit: the code labels, PARENT_KIND and PARENT_PK (the pk of the finished SCF/NSCF ``PwBaseWorkChain``, or of
a ``YamboWorkflow``/``YamboRestart``/``YamboCalculation``), and the scheduler options.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.yambowf import YamboWorkflow

load_profile()

# ---------------------------------------------------------------------------
# 1. Settings: edit them
# ---------------------------------------------------------------------------
YAMBO_CODE = 'yambo@mycluster'
P2Y_CODE = 'p2y@mycluster'
PW_CODE = 'pw@mycluster'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'
PARENT_KIND = 'nscf'  # 'scf', 'nscf' or 'yambo': only for your reference, the workflow detects it
PARENT_PK = 1234  # pk of the finished parent (it must have a `remote_folder` output)
KMESH = [2, 2, 2]  # must be the mesh of the parent, if it is an NSCF or a Yambo run
NSCF_NBND = 76
QP_NB = 75

# ---------------------------------------------------------------------------
# 2. Structure: the one of the parent
# ---------------------------------------------------------------------------
structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

# ---------------------------------------------------------------------------
# 3. Inputs
# ---------------------------------------------------------------------------
builder = YamboWorkflow.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code(YAMBO_CODE),
    preprocessing_code=orm.load_code(P2Y_CODE),
    pw_code=orm.load_code(PW_CODE),
    pseudo_family=PSEUDO_FAMILY,
    calc_type='qp',
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboWorkflow', mode='builder'))
manager.scf.set_kmesh(KMESH)
manager.nscf.set_kmesh(KMESH)
manager.set_nscf_nbnd(value=NSCF_NBND)
manager.qp.set_Nb(value=QP_NB)
manager.qp.set_Gcut(value=1)
manager.additional_parsing = ['gap_', 'homo', 'lumo']

# the parent: the workflow looks at what it is and skips the steps it already contains
manager.set_parent_folder(orm.load_node(PARENT_PK).outputs.remote_folder)

# ---------------------------------------------------------------------------
# 4. Scheduler options (see 01_gw_from_scratch.py)
# ---------------------------------------------------------------------------
manager.scf.set_resources(mpi=2)
manager.scf.set_npools(value=2)

# ---------------------------------------------------------------------------
# 5. Submit
# ---------------------------------------------------------------------------
node = submit(YamboWorkflow, **manager.generate_AiiDA_inputs(what='YamboWorkflow'))
print(f'GW from a {PARENT_KIND} parent submitted, pk = {node.pk}')
print(f'Monitor with:  verdi process report {node.pk}')

# ---------------------------------------------------------------------------
# 6. After it finishes
# ---------------------------------------------------------------------------
# node = orm.load_node(<PK>)
# print(node.outputs.output_ywfl_parameters.get_dict())
