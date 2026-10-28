"""Independent-particle optical absorption (IP-RPA) with a single ``YamboCalculation``.

Status: rewritten from the old yambo_ip_rpa example, which ran on a cluster; this version is tested here (valid inputs).
Edit: the code labels, PARENT_PK, the energy range and the direction of the field.
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
    'arguments': ['optics', 'optics'],
    'variables': {
        'Chimod': 'IP',
        'DysSolver': 'n',
        'GTermKind': 'BG',
        'NGsBlkXp': [4, 'Ry'],
        'QpntsRXd': [[1, 1], ''],
        'BndsRnXp': [[1, 100], ''],
        'EnRngeXd': [[0.0, 10.0], 'eV'],
        'DmRngeXd': [[0.15, 0.3], 'eV'],
        'ETStpsXd': [1000, ''],
        'LongDrXd': [[1.0, 0.0, 0.0], ''],
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
