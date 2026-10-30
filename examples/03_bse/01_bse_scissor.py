"""Bethe-Salpeter equation (BSE) with a scissor correction, with ``YamboWorkflow``.

Flow:  scf -> nscf -> p2y -> BSE   (or, from a parent: only the missing steps)

``calc_type='bse'``: the DFT energies are shifted and stretched by ``KfnQP_E = [shift (eV), stretch (valence), stretch
(conduction)]`` before the BSE. It is cheap, and good to test the optical inputs; for a spectrum based on calculated
GW energies see 02_bse_at_gw.py. ``BSEBands`` are the valence and conduction bands of the excitonic problem.

Status: the same mode ran on a cluster (silicon, PseudoDojo PBE).
Edit: the code labels, PARENT_PK (None: start from the structure), the BSE settings.
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
PARENT_PK = None  # a finished SCF, NSCF or Yambo run; None: scf -> nscf -> p2y -> BSE
KMESH = [2, 2, 2]
NSCF_NBND = 76
BSE_NB = 75  # bands of the screening of the BSE (BndsRnXs)
BSE_GCUT = 4  # Ry, cutoff of the screening (NGsBlkXs)
BSE_BANDS = [4, 5]  # valence and conduction bands of the excitonic problem (Si: 4 is the highest valence band)
SCISSOR = [0.5, 1.0, 1.0]  # eV shift, and stretch of the valence and conduction bands

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
    calc_type='bse',
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboWorkflow', mode='builder'))
manager.scf.set_kmesh(KMESH)
manager.nscf.set_kmesh(KMESH)
manager.set_nscf_nbnd(value=NSCF_NBND)
manager.bse.set_BSENb(value=BSE_NB)
manager.bse.set_BSEGcut(value=BSE_GCUT)
manager.bse.set_BSEbands(BSE_BANDS)
manager.bse.set_variable('KfnQP_E', [SCISSOR, ''])
manager.additional_parsing = ['lowest_exciton', 'brightest_exciton']
if PARENT_PK is not None:
    manager.set_parent_folder(orm.load_node(PARENT_PK).outputs.remote_folder)

# ---------------------------------------------------------------------------
# 3. Scheduler options (see 04_options/01_resources_and_parallelism.py)
# ---------------------------------------------------------------------------
manager.scf.set_resources(mpi=2)
manager.scf.set_npools(value=2)

# ---------------------------------------------------------------------------
# 4. Submit
# ---------------------------------------------------------------------------
node = submit(YamboWorkflow, **manager.generate_AiiDA_inputs(what='YamboWorkflow'))
print(f'BSE (scissor) submitted, pk = {node.pk}')

# ---------------------------------------------------------------------------
# 5. After it finishes
# ---------------------------------------------------------------------------
# node = orm.load_node(<PK>)
# print(node.outputs.output_ywfl_parameters.get_dict())   # lowest_exciton, brightest_exciton
# energy = node.outputs.array_chi.get_array('eV')          # the spectrum
# epsilon2 = node.outputs.array_chi.get_array('eps_2')     # imaginary part of the dielectric function
