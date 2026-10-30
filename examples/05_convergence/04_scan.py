"""Run a list of parameter sets with ``YamboConvergence``, without converging anything.

Flow: one ``YamboWorkflow`` for each set of parameters (scf -> nscf -> p2y -> GW), at most ``MAX_CONCURRENT`` at a time.

Use it for tests, or to collect data for your own analysis. Nothing is evaluated: the workflow ends when all the
points are done, and a failed calculation does not stop the others (it is flagged ``failed`` in the story).

Status: tested here (the inputs are valid); not yet run on a cluster.
Edit: the code labels, the points.
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
YAMBO_CODE = 'yambo-5.3.0-gcc@corvina-direct'
P2Y_CODE = 'p2y-5.3.0-gcc@corvina-direct'
PW_CODE = 'qe-7.5-gcc-spack@corvina-direct'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'
GROUP_LABEL = 'scan/silicon/qp'

# One dictionary {variable: value} for each calculation; all the sets define the same variables.
# GbndRnge follows BndsRnXp. Units: Ry for the cutoffs, none for the bands.
POINTS = [
    {'FFTGvecs': 10, 'BndsRnXp': 20, 'NGsBlkXp': 2},
    {'FFTGvecs': 15, 'BndsRnXp': 30, 'NGsBlkXp': 3},
    {'FFTGvecs': 20, 'BndsRnXp': 40, 'NGsBlkXp': 4},
]

structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

# ---------------------------------------------------------------------------
# 2. Inputs
# ---------------------------------------------------------------------------
builder = YamboConvergence.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code(YAMBO_CODE),
    preprocessing_code=orm.load_code(P2Y_CODE),
    pw_code=orm.load_code(PW_CODE),
    pseudo_family=PSEUDO_FAMILY,
    calc_type='qp',
    scan_points=POINTS,
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboConvergence', mode='builder'))
manager.scf.set_kmesh([2, 2, 2])
manager.nscf.set_kmesh([2, 2, 2])
manager.set_nscf_nbnd(value=76)
manager.qp.set_Nb(value=75)
manager.qp.set_Gcut(value=1)
manager.workflow_settings = {**manager.workflow_settings, 'what': ['gap_']}  # the quantities collected in the story
manager.additional_parsing = ['gap_', 'homo', 'lumo']
manager.group_label = GROUP_LABEL

# ---------------------------------------------------------------------------
# 3. Submit
# ---------------------------------------------------------------------------
node = submit(YamboConvergence, **manager.generate_AiiDA_inputs(what='YamboConvergence'))
print(f'Scan of {len(POINTS)} points submitted, pk = {node.pk}')

# ---------------------------------------------------------------------------
# 4. After it finishes
# ---------------------------------------------------------------------------
# import pandas as pd
# node = orm.load_node(<PK>)
# story = pd.DataFrame(node.outputs.collection.get_dict()['story'])  # values, quantities and the `failed` flag
# print(node.outputs.convergence_summary.get_dict()['scan'])          # number of points and of failures
