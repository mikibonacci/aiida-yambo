"""Resources, queue and parallelism of a ``YamboWorkflow``: where each of them is set.

Flow:  scf -> nscf -> p2y + GW   (the same as 02_gw/01_gw_from_scratch.py, with the scheduler options spelled out)

Every step has its own options in the manager: ``manager.scf``, ``manager.nscf`` (Quantum ESPRESSO),
``manager.qp`` and ``manager.bse`` (Yambo, the first one also runs p2y). For the splitted runs of the QP splitter see
02_gw/03_qp_splitter.py (``SPLIT_*``), for the resources which depend on the values of the parameters in a
convergence see 05_convergence/07_resource_rules.py. The explanation is in the docs ("Resources and parallelism").

Status: tested here (the inputs are valid); the options are the ones used on a cluster.
Edit: the code labels, the numbers (they must fit your machine and queue).
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
QUEUE = None  # e.g. 'debug'
ACCOUNT = None  # e.g. 'my_project'

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

# --8<-- [start:resources]
# ---------------------------------------------------------------------------
# 2. Resources of each step: nodes=, mpi= (per node), threads= (OpenMP, per MPI task), time= (seconds)
# ---------------------------------------------------------------------------
manager.scf.set_resources(nodes=1, mpi=4, time=1800)
manager.scf.set_npools(value=2)  # Quantum ESPRESSO: k-points pools (-npool)
manager.nscf.set_resources(nodes=1, mpi=4, time=3600)
manager.nscf.set_npools(value=2)
manager.qp.set_resources(nodes=1, mpi=4, threads=2, time=7200)  # threads: OMP_NUM_THREADS is exported for you

# ---------------------------------------------------------------------------
# 3. Queue, account, scheduler text, modules
# ---------------------------------------------------------------------------
for step in (manager.scf, manager.nscf, manager.qp):
    if QUEUE:
        step.metadata.options.queue_name = QUEUE
    if ACCOUNT:
        step.metadata.options.account = ACCOUNT
# manager.qp.set_prepend_text('module load yambo')            # added to the job script, before the executable
# manager.qp.set_scheduler_commands('#SBATCH --constraint=cpu')  # extra scheduler lines

# ---------------------------------------------------------------------------
# 4. Yambo parallelism: the default is a balanced distribution; set the roles by hand only if you need it
# ---------------------------------------------------------------------------
manager.qp.set_variable('PAR_def_mode', 'balanced')
# manager.qp.set_variable('X_and_IO_CPU', '1 1 2 2 1')   # MPI tasks on each role of the screening: q k g c v
# manager.qp.set_variable('X_and_IO_ROLEs', 'q k g c v')
# manager.qp.set_variable('SE_CPU', '1 2 2')             # ... and of the self-energy: q qp b
# manager.qp.set_variable('SE_ROLEs', 'q qp b')

# ---------------------------------------------------------------------------
# 5. Limits of the automatic restarts (YamboRestart): after a timeout the walltime grows up to max_walltime,
#    after a memory error the nodes can grow up to max_number_of_nodes (0: they are not increased)
# ---------------------------------------------------------------------------
manager.qp.max_walltime = 4 * 3600
manager.qp.max_number_of_nodes = 2
# --8<-- [end:resources]

# ---------------------------------------------------------------------------
# 6. Submit
# ---------------------------------------------------------------------------
node = submit(YamboWorkflow, **manager.generate_AiiDA_inputs(what='YamboWorkflow'))
print(f'YamboWorkflow submitted, pk = {node.pk}')
print(f'Check what was used:  verdi process show {node.pk}   and   verdi calcjob inputcat <pk of a calculation>')
