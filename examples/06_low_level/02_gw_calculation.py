"""A single GW (plasmon-pole, G0W0) ``YamboCalculation``.

Flow: from the parent (p2y done), dipoles, exchange and correlation of the quasiparticle (k-point 1, bands 8 and 9).
No restart on failure: for that use ``YamboRestart`` (04_yambo_restart.py) or, better, ``YamboWorkflow``.

Status: rewritten from the old yambo_gw example, which ran on a cluster; this version is tested here (valid inputs).
Edit: the code labels, PARENT_PK (an NSCF with p2y, or a Yambo run), the bands and k-points of ``QPkrange``.
"""
from aiida import load_profile, orm
from aiida.engine import submit

from aiida_yambo.calculations.yambo import YamboCalculation

load_profile()

# ---------------------------------------------------------------------------
# 1. Settings: edit them
# ---------------------------------------------------------------------------
YAMBO_CODE = 'yambo@mycluster'
P2Y_CODE = 'p2y@mycluster'
PARENT_PK = 1234  # a finished calculation with a remote folder: an NSCF (with p2y done) or a Yambo run
NODES, MPI, THREADS = 1, 4, 1
WALLTIME = 3600  # seconds
QUEUE = None
ACCOUNT = None

# ---------------------------------------------------------------------------
# 2. The calculation: the yambo input is a dictionary of the runlevel arguments and the variables
# ---------------------------------------------------------------------------
parameters = {
    'arguments': ['dipoles', 'HF_and_locXC', 'gw0', 'ppa'],
    'variables': {
        'Chimod': 'hartree',
        'DysSolver': 'n',
        'GTermKind': 'BG',
        'NGsBlkXp': [2, 'Ry'],
        'BndsRnXp': [[1, 50], ''],
        'GbndRnge': [[1, 50], ''],
        'QPkrange': [[[1, 1, 8, 9]], ''],  # [[k1, k2, b1, b2], ...]
    },
}

builder = YamboCalculation.get_builder()
builder.code = orm.load_code(YAMBO_CODE)
builder.preprocessing_code = orm.load_code(P2Y_CODE)
builder.parent_folder = orm.load_node(PARENT_PK).outputs.remote_folder
builder.parameters = orm.Dict(parameters)
builder.precode_parameters = orm.Dict({})  # extra options of p2y
builder.settings = orm.Dict({'INITIALISE': False, 'COPY_DBS': False})

# ---------------------------------------------------------------------------
# 3. Options of the job
# ---------------------------------------------------------------------------
builder.metadata.options.resources = {
    'num_machines': NODES, 'num_mpiprocs_per_machine': MPI, 'num_cores_per_mpiproc': THREADS}
builder.metadata.options.max_wallclock_seconds = WALLTIME
builder.metadata.options.prepend_text = f'export OMP_NUM_THREADS={THREADS}'
if QUEUE:
    builder.metadata.options.queue_name = QUEUE
if ACCOUNT:
    builder.metadata.options.account = ACCOUNT

# ---------------------------------------------------------------------------
# 4. Submit
# ---------------------------------------------------------------------------
node = submit(builder)
print(f'YamboCalculation submitted, pk = {node.pk}')
print(f'Monitor with:  verdi process status {node.pk}')
