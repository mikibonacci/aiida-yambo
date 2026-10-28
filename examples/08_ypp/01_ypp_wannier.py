"""``ypp`` post-processing with ``YppCalculation``: the Wannier90 interface.  EXPERIMENTAL.

The ypp support of the plugin is unfinished: there is no workflow for it, only the CalcJob and ``YppRestart``.
This script runs ``ypp -w`` (``wannier``) on a finished Yambo calculation.

Status: experimental, tested here only (the inputs are valid).
Edit: the code label, PARENT_PK (a finished Yambo calculation), the options.
"""
from aiida import load_profile, orm
from aiida.engine import submit

from aiida_yambo.calculations.ypp import YppCalculation

load_profile()

YPP_CODE = 'ypp@mycluster'
PARENT_PK = 1234
NODES, MPI, THREADS = 1, 4, 1
WALLTIME = 3600

builder = YppCalculation.get_builder()
builder.code = orm.load_code(YPP_CODE)
builder.parent_folder = orm.load_node(PARENT_PK).outputs.remote_folder
builder.parameters = orm.Dict({'arguments': ['wannier'], 'variables': {}})
builder.metadata.options.resources = {
    'num_machines': NODES, 'num_mpiprocs_per_machine': MPI, 'num_cores_per_mpiproc': THREADS}
builder.metadata.options.max_wallclock_seconds = WALLTIME
builder.metadata.options.prepend_text = f'export OMP_NUM_THREADS={THREADS}'

node = submit(builder)
print(f'YppCalculation submitted, pk = {node.pk}')
