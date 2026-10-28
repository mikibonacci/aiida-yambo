"""A GW calculation with automatic restarts: ``YamboRestart``.

``YamboRestart`` wraps a ``YamboCalculation`` and restarts it after the usual failures: walltime exhausted (the walltime
grows), memory and parallelization errors (the MPI/OpenMP balance and the parallelism change), corrupted databases.
``YamboWorkflow`` uses it for every Yambo step; use it directly when you want the restarts without the DFT part.

Status: rewritten from the old yambo_restart example, which ran on a cluster; this version is tested here (valid inputs).
Edit: the code labels, PARENT_PK, the options and the limits.
"""
from aiida import load_profile, orm
from aiida.engine import submit

from aiida_yambo.workflows.yamborestart import YamboRestart

load_profile()

YAMBO_CODE = 'yambo@mycluster'
P2Y_CODE = 'p2y@mycluster'
PARENT_PK = 1234  # a finished calculation with a remote folder (an NSCF with p2y, or a Yambo run)
NODES, MPI, THREADS = 1, 4, 1
WALLTIME = 3600  # seconds, it grows at each restart after a timeout...
MAX_WALLTIME = 4 * 3600  # ... up to this
MAX_NODES = 2  # the nodes can grow after a memory error up to this (0: not increased)
MAX_ITERATIONS = 5  # restarts

parameters = {
    'arguments': ['dipoles', 'HF_and_locXC', 'gw0', 'ppa'],
    'variables': {
        'Chimod': 'hartree',
        'DysSolver': 'n',
        'GTermKind': 'BG',
        'NGsBlkXp': [2, 'Ry'],
        'BndsRnXp': [[1, 50], ''],
        'GbndRnge': [[1, 50], ''],
        'QPkrange': [[[1, 1, 8, 9]], ''],
    },
}

builder = YamboRestart.get_builder()
builder.yambo.code = orm.load_code(YAMBO_CODE)
builder.yambo.preprocessing_code = orm.load_code(P2Y_CODE)
builder.yambo.parameters = orm.Dict(parameters)
builder.yambo.settings = orm.Dict({'INITIALISE': False, 'COPY_DBS': False})
builder.yambo.metadata.options.resources = {
    'num_machines': NODES, 'num_mpiprocs_per_machine': MPI, 'num_cores_per_mpiproc': THREADS}
builder.yambo.metadata.options.max_wallclock_seconds = WALLTIME
builder.yambo.metadata.options.prepend_text = f'export OMP_NUM_THREADS={THREADS}'
builder.parent_folder = orm.load_node(PARENT_PK).outputs.remote_folder
builder.max_iterations = orm.Int(MAX_ITERATIONS)
builder.max_walltime = orm.Int(MAX_WALLTIME)
builder.max_number_of_nodes = orm.Int(MAX_NODES)

node = submit(builder)
print(f'YamboRestart submitted, pk = {node.pk}')
print(f'Monitor with:  verdi process status {node.pk}   (each restart is a new child)')
