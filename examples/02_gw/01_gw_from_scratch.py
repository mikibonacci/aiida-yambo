"""GW quasiparticle calculation with ``YamboWorkflow``, starting from a structure.

Flow:  scf -> nscf -> p2y -> GW -> done

``YamboWorkflow`` runs the DFT starting point (Quantum ESPRESSO SCF and NSCF), converts it for Yambo (p2y), and
runs the GW (``calc_type='qp'``). The settings below are small, to test the installation: for production
increase the k-point mesh, the NSCF bands, the number of bands and the cutoffs of the GW (and converge them,
see 05_convergence/).

Status: the same inputs ran on a cluster (silicon, PseudoDojo PBE norm-conserving pseudopotentials).
Edit: the code labels, PSEUDO_FAMILY and the scheduler options.
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
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'  # norm-conserving: `aiida-pseudo install pseudo-dojo`
KMESH = [2, 2, 2]  # tiny, to test
NSCF_NBND = 76  # bands of the NSCF: it must be >= the bands used by Yambo (QP_NB)
QP_NB = 75  # bands of the GW (BndsRnXp and GbndRnge)
QP_GCUT = 1  # Ry, cutoff of the screening (NGsBlkXp)

# ---------------------------------------------------------------------------
# 2. Structure (silicon, from ASE)
# ---------------------------------------------------------------------------
structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

# ---------------------------------------------------------------------------
# 3. Inputs: the protocol fills everything, the manager edits what we want
# ---------------------------------------------------------------------------
builder = YamboWorkflow.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code(YAMBO_CODE),
    preprocessing_code=orm.load_code(P2Y_CODE),
    pw_code=orm.load_code(PW_CODE),
    pseudo_family=PSEUDO_FAMILY,
    calc_type='qp',  # 'qp' = GW; see 03_bse/ for the Bethe-Salpeter equation
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboWorkflow', mode='builder'))

manager.scf.set_kmesh(KMESH)
manager.nscf.set_kmesh(KMESH)
manager.set_nscf_nbnd(value=NSCF_NBND)
manager.qp.set_Nb(value=QP_NB)
manager.qp.set_Gcut(value=QP_GCUT)

# Quantities parsed from the results into `output_ywfl_parameters`:
# the indirect gap, the HOMO and LUMO (see the reference of the outputs for more).
manager.additional_parsing = ['gap_', 'homo', 'lumo']

# ---------------------------------------------------------------------------
# 4. Scheduler options: edit them to match your cluster
# ---------------------------------------------------------------------------
manager.scf.set_resources(mpi=2)  # nodes=, mpi= (per node), threads=, time= (seconds)
manager.scf.set_npools(value=2)
# manager.qp.set_resources(nodes=1, mpi=4, threads=1, time=3600)
# manager.qp.metadata.options.queue_name = 'my_queue'
# manager.qp.metadata.options.account = 'my_account'
# The same for manager.nscf. How to choose resources: see 04_options/01_resources_and_parallelism.py.

# ---------------------------------------------------------------------------
# 5. Submit
# ---------------------------------------------------------------------------
node = submit(YamboWorkflow, **manager.generate_AiiDA_inputs(what='YamboWorkflow'))
print(f'YamboWorkflow submitted, pk = {node.pk}')
print(f'Monitor with:  verdi process status {node.pk}   or   verdi process report {node.pk}')

# ---------------------------------------------------------------------------
# 6. After it finishes
# ---------------------------------------------------------------------------
# node = orm.load_node(<PK>)
# print(node.is_finished_ok, node.exit_status)
# print(node.outputs.output_ywfl_parameters.get_dict())   # gap_, homo, lumo (GW) and gap_dft, homo_dft, lumo_dft
# node.outputs.bands_quasiparticle  # or node.outputs.array_ndb: the quasiparticle corrections
# node.outputs.remote_folder        # reuse it as parent_folder: see 02_gw_from_parent.py
