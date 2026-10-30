"""Choose the function fitted to the results of each parameter of a convergence.

By default the predictors fit a power law ``a / x**alpha + b`` (1D), or a product of two power laws (coupled
parameters), and keep the exponents with the smallest error. ``fit_functions`` selects another function by name, per
parameter, and its options. The names are stored in the provenance; the functions are looked up when the workflow
runs: built-in (``power_law``, ``exponential`` in 1D), registered, or an entry point of another package. To write and
test your own function without a cluster, see 06_fit_models_offline.py.

Status: tested here (the inputs are valid); not yet run on a cluster.
Edit: the code labels, the functions.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.yamboconvergence import YamboConvergence

load_profile()

YAMBO_CODE = 'yambo-5.3.0-gcc@corvina-direct'
P2Y_CODE = 'p2y-5.3.0-gcc@corvina-direct'
PW_CODE = 'qe-7.5-gcc-spack@corvina-direct'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'

structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

builder = YamboConvergence.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code(YAMBO_CODE),
    preprocessing_code=orm.load_code(P2Y_CODE),
    pw_code=orm.load_code(PW_CODE),
    pseudo_family=PSEUDO_FAMILY,
    calc_type='qp',
    # The protocol grid of the bands is for real calculations (BndsRnXp up to thousands, multiplied by log10 of the cell
    # volume): too large for a test, and the NSCF computes the largest of them. Here a small grid for silicon.
    overrides={'meta_parameters': {'bands': {'start': 20, 'stop': 50, 'delta': 10, 'max': 60}}},
    # the default parameters (FFTGvecs, k-points density, coupled bands and cutoff) with their own fit functions
    fit_functions={
        'FFTGvecs': 'exponential',  # built-in 1D model: a * exp(-x / L) + b
        'BndsRnXp': {'fit_function': 'power_law', 'fit_options': {'alphas': [1, 2], 'betas': [1]}},  # coupled: 2D
    },
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboConvergence', mode='builder'))
manager.scf.set_kmesh([2, 2, 2])
manager.nscf.set_kmesh([2, 2, 2])
manager.set_nscf_nbnd(value=76)
manager.qp.set_Nb(value=75)
manager.qp.set_Gcut(value=1)
manager.additional_parsing = ['gap_', 'homo', 'lumo']
manager.group_label = 'convergence/silicon/qp-fits'

node = submit(YamboConvergence, **manager.generate_AiiDA_inputs(what='YamboConvergence'))
print(f'Convergence with chosen fit functions submitted, pk = {node.pk}')
# After it finishes: for each parameter, parameters_space[i].fit_function / fit_options are in the inputs.
