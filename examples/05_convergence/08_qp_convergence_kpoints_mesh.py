"""Convergence of a GW gap with respect to the k-point **mesh**.

The parameter is a ``KpointMeshConvParam`` with ``var='kpoint_mesh'``: ``start``, ``stop``, ``delta`` and ``max`` are
3-vectors ``[nx, ny, nz]`` (positive integers), and the grid is ``start, start + delta, ... stop``. The mesh is set in
the SCF and in the NSCF (the NSCF one is the one used by Yambo), so every point is a new scf -> nscf -> p2y -> GW.
The fit is done on the number of k-points of the mesh (see "Choose the fit function").

Use it when you want to control the meshes exactly (for example only even meshes, or an anisotropic one for a slab:
``start=[4, 4, 1]``, ``stop=[10, 10, 1]``, ``delta=[2, 2, 0]``). To converge on a density instead, which adapts to the
cell, see 09_qp_convergence_kpoints_density.py.

The settings are small, to test: a real GW k-point convergence is expensive (use ``conv_thr`` and ranges for your
system). The other parameters (``FFTGvecs``, bands, cutoffs) are fixed here: converge them first, or in the same run
(01_qp_convergence.py), and then put their converged values below.

Edit: the code labels, the meshes, ``conv_thr``, ``GROUP_LABEL``.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.utils.parameters import KpointMeshConvParam
from aiida_yambo.workflows.yamboconvergence import YamboConvergence

load_profile()

# ---------------------------------------------------------------------------
# 1. Settings: edit them
# ---------------------------------------------------------------------------
YAMBO_CODE = 'yambo-5.3.0-gcc@corvina-direct'
P2Y_CODE = 'p2y-5.3.0-gcc@corvina-direct'
PW_CODE = 'qe-7.5-gcc-spack@corvina-direct'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'
GROUP_LABEL = 'convergence/silicon/qp_kmesh'  # the calculations are kept here, and reused if you run again
WHAT = ['gap_']  # the quantity to converge: 'gap_' (indirect), 'gap_GG', 'homo', 'lumo'...
CONV_THR = 5  # percent of the converged value (conv_thr_units='%', the default)

structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

# ---------------------------------------------------------------------------
# 2. Inputs
# ---------------------------------------------------------------------------
builder = YamboConvergence.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code(YAMBO_CODE),
    preprocessing_code=orm.load_code(P2Y_CODE),
    pw_code=orm.load_code(PW_CODE),
    pseudo_family=PSEUDO_FAMILY,
    calc_type='qp',
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboConvergence', mode='builder'))
manager.set_nscf_nbnd(value=76)
manager.qp.set_Nb(value=75)
manager.qp.set_Gcut(value=1)
manager.qp.set_variable('FFTGvecs', [20, 'Ry'])

# 3 x 3 x 3, 4 x 4 x 4, 5 x 5 x 5, 6 x 6 x 6: no mesh larger than ``max`` is explored
manager.parameters_space = [
    KpointMeshConvParam(var='kpoint_mesh', start=[3, 3, 3], stop=[6, 6, 6], delta=[1, 1, 1], max=[10, 10, 10],
                        units='mesh', conv_thr=CONV_THR, max_iterations=3, max_concurrent_steps=2),
]
manager.workflow_settings = {'type': 'heavy', 'what': WHAT, 'bands_nscf_update': 'all-at-once', 'skip_pre': False}
manager.additional_parsing = ['gap_', 'homo', 'lumo']
manager.group_label = GROUP_LABEL

# ---------------------------------------------------------------------------
# 3. Submit
# ---------------------------------------------------------------------------
node = submit(YamboConvergence, **manager.generate_AiiDA_inputs(what='YamboConvergence'))
print(f'YamboConvergence submitted, pk = {node.pk}')
print(f'Monitor with:  verdi process status {node.pk}   or   verdi process report {node.pk}')

# ---------------------------------------------------------------------------
# 4. After it finishes (see 07_analysis/ to read the results)
# ---------------------------------------------------------------------------
# node = orm.load_node(<PK>)
# print(node.outputs.convergence_summary.get_dict())   # the converged mesh
