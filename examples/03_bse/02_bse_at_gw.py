"""BSE on explicit GW corrections (BSE@GW), with ``YamboWorkflow``.

Flow:  scf -> nscf -> p2y + GW -> [ GW on subsets (QP splitter) ] -> merge -> BSE with the merged ``ndb.QP``

``calc_type='bse@qp'``: the quasiparticles you choose are computed with the QP splitter (see
02_gw/03_qp_splitter.py for all the ways to choose them and to set their resources), merged, and used by the BSE
(``KfnQPdb``). ``BSEBands`` follow the QP computed with ``range_QP`` unless you set them.

Status: the same mode ran on a cluster (from scratch, from an SCF, an NSCF and a Yambo parent). Needs ``yambopy``.
Edit: the code labels, PARENT_PK (None: from the structure), MODE (the way the QP are chosen).
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
YAMBO_CODE = 'yambo-5.3.0-gcc@corvina-direct'
P2Y_CODE = 'p2y-5.3.0-gcc@corvina-direct'
PW_CODE = 'qe-7.5-gcc-spack@corvina-direct'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'
PARENT_PK = None  # a finished SCF, NSCF or Yambo run; None: from the structure
KMESH = [2, 2, 2]
NSCF_NBND = 76
QP_NB = 75
BSE_BANDS = [4, 5]
MODE = 'boundaries'  # or 'range_QP': the QP within an energy window around mid-gap (BSEBands follow it)

MODES = {
    'boundaries': dict(ki=1, kf=2, bi=3, bf=6),
    'range_QP': dict(range_QP=5.0),
}

structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

# ---------------------------------------------------------------------------
# 2. Inputs
# ---------------------------------------------------------------------------
builder = YamboWorkflow.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code(YAMBO_CODE),
    preprocessing_code=orm.load_code(P2Y_CODE),
    pw_code=orm.load_code(PW_CODE),
    pseudo_family=PSEUDO_FAMILY,
    calc_type='bse@qp',
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboWorkflow', mode='builder'))
manager.scf.set_kmesh(KMESH)
manager.nscf.set_kmesh(KMESH)
manager.set_nscf_nbnd(value=NSCF_NBND)
manager.qp.set_Nb(value=QP_NB)
manager.qp.set_Gcut(value=1)
manager.bse.set_BSENb(value=QP_NB)
manager.bse.set_BSEGcut(value=4)
if MODE != 'range_QP':
    manager.bse.set_BSEbands(BSE_BANDS)
manager.set_QP_subsets_dict(qp_per_subset=8, parallel_runs=1, **MODES[MODE])
manager.additional_parsing = ['gap_', 'lowest_exciton']
if PARENT_PK is not None:
    manager.set_parent_folder(orm.load_node(PARENT_PK).outputs.remote_folder)

manager.scf.set_resources(mpi=2)
manager.scf.set_npools(value=2)

# ---------------------------------------------------------------------------
# 3. Submit
# ---------------------------------------------------------------------------
node = submit(YamboWorkflow, **manager.generate_AiiDA_inputs(what='YamboWorkflow'))
print(f'BSE@GW ({MODE}) submitted, pk = {node.pk}')

# ---------------------------------------------------------------------------
# 4. After it finishes
# ---------------------------------------------------------------------------
# node = orm.load_node(<PK>)
# node.outputs.merged_QP                                   # the QP database used by the BSE
# print(node.outputs.output_ywfl_parameters.get_dict())    # gap_, lowest_exciton
# node.outputs.array_chi                                   # the spectrum
