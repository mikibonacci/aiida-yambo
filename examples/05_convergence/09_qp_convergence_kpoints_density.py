"""Convergence of a GW gap with respect to the k-point **density**.

The parameter is a ``KpointMeshConvParam`` with ``var='kpoint_density'``. The values are k-point *distances* in 1/Angstrom,
passed to ``set_kpoints_mesh_from_density`` of AiiDA: the mesh is built from the cell, so the same numbers work for any
structure (and give anisotropic meshes for anisotropic cells). A refinement means a **smaller** distance: ``start`` is the
coarsest (largest distance), ``stop`` the finest (smallest), ``delta`` is negative, and ``max`` is the largest distance
allowed (the coarse cap). The fit is done on ``(1 / distance)**3``, proportional to the number of k-points.

It is the default k-point parameter of the protocol convergence. Use it when the same script must serve different
structures; to fix the meshes exactly, see 08_qp_convergence_kpoints_mesh.py.

The settings are small, to test: a real GW k-point convergence is expensive (use ``conv_thr`` and ranges for your
system). The other parameters (``FFTGvecs``, bands, cutoffs) are fixed here: converge them first, or in the same run
(01_qp_convergence.py), and then put their converged values below.

Edit: the code labels, the distances, ``conv_thr``, ``GROUP_LABEL``.
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
GROUP_LABEL = 'convergence/silicon/qp_kdensity'  # the calculations are kept here, and reused if you run again
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

# distances 0.6, 0.5, 0.4, 0.3 (1/Angstrom): no distance larger than ``max`` is explored
manager.parameters_space = [
    KpointMeshConvParam(var='kpoint_density', start=0.6, stop=0.3, delta=-0.1, max=0.6,
                        units='1/A', conv_thr=CONV_THR, max_iterations=3, max_concurrent_steps=2),
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
# print(node.outputs.convergence_summary.get_dict())   # the converged distance
