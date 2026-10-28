"""Small variations of a GW run: cleaning after failures, another pseudopotential family, too few NSCF bands.

Flow:  scf -> nscf -> p2y -> GW  (as in 02_gw/01_gw_from_scratch.py), with the option chosen by CASE:

    'clean_failed'  ``clean_failed=True``: at the end, the remote folders of the failed calculations are cleaned
    'lda'           PseudoDojo LDA pseudopotentials instead of PBE
    'few_bands'     the NSCF has fewer bands than the GW needs: the workflow detects it and redoes the NSCF

Status: all the cases ran on a cluster (silicon).
Edit: the code labels, CASE.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.yambowf import YamboWorkflow

load_profile()

YAMBO_CODE = 'yambo@mycluster'
P2Y_CODE = 'p2y@mycluster'
PW_CODE = 'pw@mycluster'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'
CASE = 'clean_failed'

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
manager.set_nscf_nbnd(value=76)
manager.qp.set_Nb(value=75)
manager.qp.set_Gcut(value=1)
manager.additional_parsing = ['gap_', 'homo', 'lumo']

if CASE == 'clean_failed':
    manager.clean_failed = True
elif CASE == 'lda':
    manager.set_pseudo_family(value='PseudoDojo/0.4/LDA/SR/standard/upf')
elif CASE == 'few_bands':
    manager.nscf.set_pw_parameter(namelist='SYSTEM', key='nbnd', value=4)  # less than the 75 of the GW
else:
    raise ValueError(f'unknown CASE {CASE!r}')

node = submit(YamboWorkflow, **manager.generate_AiiDA_inputs(what='YamboWorkflow'))
print(f'GW ({CASE}) submitted, pk = {node.pk}')
