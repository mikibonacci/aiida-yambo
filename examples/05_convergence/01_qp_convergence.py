"""Automated convergence of a GW gap with ``YamboConvergence``.

Flow: many ``YamboWorkflow`` runs (scf -> nscf -> p2y -> GW) with growing parameters, a fit of the gap, the
converged point. Here: first the FFT cutoff (``FFTGvecs``), then the number of bands and the screening cutoff
together (``BndsRnXp`` mirrored on ``GbndRnge``, with ``NGsBlkXp``).

The settings are small, to test: a real convergence needs a denser k-mesh and larger ranges, and it is expensive.
How the algorithm works: docs, "Convergence". Resources which depend on the parameters: 07_resource_rules.py.

Status: the QP convergence of silicon ran on a cluster with this kind of inputs.
Edit: the code labels, the ranges, ``conv_thr`` (eV), ``GROUP_LABEL``.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.utils.parameters import ConvParam, CoupledConvParam
from aiida_yambo.workflows.yamboconvergence import YamboConvergence

load_profile()

# ---------------------------------------------------------------------------
# 1. Settings: edit them
# ---------------------------------------------------------------------------
YAMBO_CODE = 'yambo-5.3.0-gcc@corvina-direct'
P2Y_CODE = 'p2y-5.3.0-gcc@corvina-direct'
PW_CODE = 'qe-7.5-gcc-spack@corvina-direct'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'
GROUP_LABEL = 'convergence/silicon/qp'  # the calculations are kept here, and reused if you run again
WHAT = ['gap_']  # the quantity to converge: 'gap_' (indirect), 'gap_GG', 'homo', 'lumo'...
CONV_THR = 0.1  # eV

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
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboConvergence', mode='builder'))
manager.scf.set_kmesh([2, 2, 2])
manager.nscf.set_kmesh([2, 2, 2])
manager.set_nscf_nbnd(value=76)
manager.qp.set_Nb(value=75)
manager.qp.set_Gcut(value=1)

# The parameters, converged one after the other. The coupled one increases the bands and the cutoff together.
manager.parameters_space = [
    ConvParam('FFTGvecs', start=10, stop=30, delta=5, max=50, units='Ry', conv_thr=CONV_THR, max_concurrent_steps=2),
    CoupledConvParam(['BndsRnXp', 'NGsBlkXp'], start=[10, 2], stop=[30, 6], delta=[10, 2], max=[50, 10],
                     units=['', 'Ry'], conv_thr=CONV_THR, mirror_values={'BndsRnXp': ['GbndRnge']},
                     max_iterations=2, max_concurrent_steps=2),
]
# 'heavy': the converged values are kept for the next parameters; 'cheap': every parameter starts from the initial inputs
manager.workflow_settings = {'type': 'heavy', 'what': WHAT, 'bands_nscf_update': 'all-at-once', 'skip_pre': False}
manager.additional_parsing = ['gap_', 'homo', 'lumo']
manager.group_label = GROUP_LABEL

# ---------------------------------------------------------------------------
# 3. Submit
# ---------------------------------------------------------------------------
node = submit(YamboConvergence, **manager.generate_AiiDA_inputs(what='YamboConvergence'))
print(f'YamboConvergence submitted, pk = {node.pk}')
print(f'Monitor with:  verdi process status {node.pk}   or   verdi process report {node.pk}')

# ---------------------------------------------------------------------------
# 4. After it finishes (see 07_analysis/ to read the results)
# ---------------------------------------------------------------------------
# import pandas as pd
# node = orm.load_node(<PK>)
# print(node.outputs.convergence_summary.get_dict())               # the converged value of each parameter
# story = pd.DataFrame(node.outputs.collection.get_dict()['story'])  # one row per calculation
# print(story[['parameters_studied', 'gap_', 'converged', 'failed']])
