"""Convergence of the lowest exciton of a BSE@GW.  EXPERIMENTAL: the last cluster run failed.

Flow: many ``YamboWorkflow`` runs (scf -> nscf -> p2y + GW -> QP splitter -> merge -> BSE) with growing parameters.
The QP are chosen with the QP splitter (a cheap window here: only the Gamma point; see 02_gw/03_qp_splitter.py).

Status: EXPERIMENTAL. The inputs are valid (tested here) but the same convergence on a cluster ended with
``CALCS_FAILED``: check the failed calculations (``verdi process report``) before trusting it. Needs ``yambopy``.
Edit: the code labels, the ranges, the QP window.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.utils.parameters import ConvParam, CoupledConvParam
from aiida_yambo.workflows.yamboconvergence import YamboConvergence

load_profile()

YAMBO_CODE = 'yambo@mycluster'
P2Y_CODE = 'p2y@mycluster'
PW_CODE = 'pw@mycluster'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'
GROUP_LABEL = 'convergence/silicon/bse-at-gw'
WHAT = ['lowest_exciton']

structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

builder = YamboConvergence.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code(YAMBO_CODE),
    preprocessing_code=orm.load_code(P2Y_CODE),
    pw_code=orm.load_code(PW_CODE),
    pseudo_family=PSEUDO_FAMILY,
    calc_type='bse@qp',
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboConvergence', mode='builder'))
manager.scf.set_kmesh([2, 2, 2])
manager.nscf.set_kmesh([2, 2, 2])
manager.set_nscf_nbnd(value=76)
manager.qp.set_Nb(value=75)
manager.qp.set_Gcut(value=1)
manager.bse.set_BSENb(value=75)
manager.bse.set_BSEbands([4, 5])
# QP of the Gamma point only, bands 3 to 6 (2 valence and 2 conduction bands of silicon): the cheapest choice
manager.set_QP_subsets_dict(ki=1, kf=1, bi=3, bf=6, qp_per_subset=8, parallel_runs=1)

manager.parameters_space = [
    ConvParam('FFTGvecs', start=10, stop=30, delta=5, max=50, units='Ry', conv_thr=0.5, max_concurrent_steps=2),
    CoupledConvParam(['BndsRnXs', 'NGsBlkXs'], start=[10, 2], stop=[30, 6], delta=[10, 2], max=[50, 10],
                     units=['', 'Ry'], conv_thr=10, mirror_values={'NGsBlkXs': ['BSENGBlk']},
                     max_iterations=2, max_concurrent_steps=2),
]
manager.workflow_settings = {'type': 'heavy', 'what': WHAT, 'bands_nscf_update': 'all-at-once', 'skip_pre': False}
manager.additional_parsing = WHAT
manager.group_label = GROUP_LABEL

node = submit(YamboConvergence, **manager.generate_AiiDA_inputs(what='YamboConvergence'))
print(f'BSE@GW convergence (experimental) submitted, pk = {node.pk}')
