"""A molecule (no periodicity) and the metal check (``stop_if_metal``).  EXPERIMENTAL: never run on a cluster.

Flow:  scf -> nscf -> p2y -> GW

CASE = 'molecule':  an isolated molecule in a box, gamma point only. Check the box size and the Coulomb cutoff
                    (``RIM_cut`` is in the protocol) for your molecule before trusting the results.
CASE = 'metal_check': a crystal with ``stop_if_metal=True``: if the NSCF bands predict a metal, the workflow stops
                    with the exit code 303 before Yambo. The flag is not stored in the provenance, and
                    ``YamboConvergence`` does not forward it.

Status: experimental, tested here only (the inputs are valid).
Edit: the code labels, CASE.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase import Atoms
from ase.build import bulk

from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.yambowf import YamboWorkflow

load_profile()

YAMBO_CODE = 'yambo@mycluster'
P2Y_CODE = 'p2y@mycluster'
PW_CODE = 'pw@mycluster'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'
CASE = 'molecule'

if CASE == 'molecule':
    atoms = Atoms('Si2', positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.3]])  # the Si2 dimer; any molecule of ASE works
    atoms.set_cell([10.0, 10.0, 10.0])
    atoms.center()
    atoms.pbc = False
    kmesh = [1, 1, 1]
else:
    atoms = bulk('Si', 'diamond', a=5.43)
    kmesh = [2, 2, 2]
structure = orm.StructureData(ase=atoms)

builder = YamboWorkflow.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code(YAMBO_CODE),
    preprocessing_code=orm.load_code(P2Y_CODE),
    pw_code=orm.load_code(PW_CODE),
    pseudo_family=PSEUDO_FAMILY,
    calc_type='qp',
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboWorkflow', mode='builder'))
manager.scf.set_kmesh(kmesh)
manager.nscf.set_kmesh(kmesh)
manager.set_nscf_nbnd(value=40 if CASE == 'molecule' else 76)
manager.qp.set_Nb(value=30 if CASE == 'molecule' else 75)
manager.qp.set_Gcut(value=1)
manager.additional_parsing = ['gap_', 'homo', 'lumo']

inputs = manager.generate_AiiDA_inputs(what='YamboWorkflow')
if CASE == 'metal_check':
    inputs['stop_if_metal'] = True

node = submit(YamboWorkflow, **inputs)
print(f'GW ({CASE}) submitted, pk = {node.pk}')
