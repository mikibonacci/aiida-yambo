"""Run only the pre-calculation (p2y and Yambo initialization), to get a ``SAVE`` folder to reuse.

Flow:  [scf -> nscf ->] p2y + yambo initialization -> done  (no GW, no BSE)

With ``INITIALISE`` in the settings of the first yambo step, ``YamboWorkflow`` stops after p2y. The
``remote_folder`` of the result is a Yambo parent: give it as the parent folder of a GW or BSE workflow (see
02_gw/02_gw_from_parent.py) to skip the DFT, p2y and initialization steps.

Status: the same mode ran on a cluster (from an NSCF parent).
Edit: the code labels, PARENT_PK (None: from the structure).
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
PARENT_PK = None  # a finished NSCF (or SCF); None: scf -> nscf first
NSCF_NBND = 76

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
manager.scf.set_kmesh([2, 2, 2])
manager.nscf.set_kmesh([2, 2, 2])
manager.set_nscf_nbnd(value=NSCF_NBND)
manager.qp.set_Nb(value=75)
manager.qp.set_Gcut(value=1)
if PARENT_PK is not None:
    manager.set_parent_folder(orm.load_node(PARENT_PK).outputs.remote_folder)

manager.qp.settings['INITIALISE'] = True  # only p2y and the initialization

node = submit(YamboWorkflow, **manager.generate_AiiDA_inputs(what='YamboWorkflow'))
print(f'Initialization submitted, pk = {node.pk}')
# After it finishes:  node.outputs.remote_folder  is the parent folder of the next workflows.
