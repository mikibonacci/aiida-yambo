"""Automated convergence of the lowest exciton of a BSE (with a scissor), with ``YamboConvergence``.

Flow: many ``YamboWorkflow`` runs (scf -> nscf -> p2y -> BSE) with growing parameters: first ``FFTGvecs``, then the
screening of the BSE, ``BndsRnXs`` with ``NGsBlkXs`` (mirrored on ``BSENGBlk``), together.

Status: the BSE convergence of silicon ran on a cluster with these inputs.
Edit: the code labels, the ranges, ``GROUP_LABEL``.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.utils.parameters import ConvParam, CoupledConvParam
from aiida_yambo.workflows.yamboconvergence import YamboConvergence

load_profile()

YAMBO_CODE = 'yambo-5.3.0-gcc@corvina-direct'
P2Y_CODE = 'p2y-5.3.0-gcc@corvina-direct'
PW_CODE = 'qe-7.5-gcc-spack@corvina-direct'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'
GROUP_LABEL = 'convergence/silicon/bse'
WHAT = ['lowest_exciton']  # or 'brightest_exciton'

structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

builder = YamboConvergence.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code(YAMBO_CODE),
    preprocessing_code=orm.load_code(P2Y_CODE),
    pw_code=orm.load_code(PW_CODE),
    pseudo_family=PSEUDO_FAMILY,
    calc_type='bse',
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboConvergence', mode='builder'))
manager.scf.set_kmesh([2, 2, 2])
manager.nscf.set_kmesh([2, 2, 2])
manager.set_nscf_nbnd(value=76)
manager.bse.set_BSENb(value=75)
manager.bse.set_BSEbands([4, 5])

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
print(f'BSE convergence submitted, pk = {node.pk}')
# After it finishes: see 01_qp_convergence.py and 07_analysis/02_read_convergence_results.py
