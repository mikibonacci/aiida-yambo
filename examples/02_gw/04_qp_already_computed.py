"""QP splitter skipping the quasiparticles you already have.

If you already computed some quasiparticles (``ndb.QP`` databases, e.g. the ``merged_QP`` output of a previous
run) the splitter does not repeat them: it builds the subsets only from the missing (k, band) pairs and merges
the new databases with yours.

Status: never run on a cluster (experimental).
Edit: the code labels, PARENT_PK and ALREADY_COMPUTED_QP_DB (pks of ``SinglefileData`` holding an ``ndb.QP``).
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
PARENT_PK = 1234  # a finished NSCF
ALREADY_COMPUTED_QP_DB = [5678]  # pks of the QP databases you have
KMESH = [2, 2, 2]
NSCF_NBND = 76
QP_NB = 75

# ---------------------------------------------------------------------------
# 2. Structure and inputs
# ---------------------------------------------------------------------------
structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))
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
manager.set_parent_folder(orm.load_node(PARENT_PK).outputs.remote_folder)

# The subsets are built from the QP which are not in the databases below.
manager.set_QP_subsets_dict(bi=1, bf=4, qp_per_subset=2, parallel_runs=1)
manager.already_computed_QP_db = ALREADY_COMPUTED_QP_DB

# ---------------------------------------------------------------------------
# 3. Submit
# ---------------------------------------------------------------------------
node = submit(YamboWorkflow, **manager.generate_AiiDA_inputs(what='YamboWorkflow'))
print(f'GW with already computed QP submitted, pk = {node.pk}')

# After it finishes: node.outputs.merged_QP contains your QP and the new ones.
