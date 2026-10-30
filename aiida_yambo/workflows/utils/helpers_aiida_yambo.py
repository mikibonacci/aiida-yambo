# -*- coding: utf-8 -*-
"""Helpers of the YamboConvergence workflow: update the inputs for the next calculation, take the results."""
import copy
import logging
from typing import Any, Dict, Tuple

from aiida.common.exceptions import NotExistent
from aiida.orm import load_node

from aiida_yambo.utils.common_helpers import find_pw_parent, search_in_group, take_calc_from_remote
from aiida_yambo.utils.parallelism_manager import ParallelismManager
from aiida_yambo.utils.resource_rules import apply_resource_rules

LOGGER = logging.getLogger(__name__)

def updater(input_manager, evaluator, explicit_value, workflow_dict: dict, index: int, group, calc_type='qp',
            resource_rules=None, resources_base=None) -> Tuple[Any, Dict[str, Any], Any, Any]:
    """
    Updates the YamboInputManager for a specific iteration step.

    This function dynamically formats parameters based on name conventions ('bnd')
    and metadata provided by the parameter space class instances, completely 
    eliminating hardcoded variables. It mutates the input_manager in place
    (plain python data, no AiiDA nodes), targeting the `qp` or `bse` namespace
    depending on `calc_type`. The AiiDA inputs are materialized later via
    `generate_AiiDA_inputs`.
    
    Args:
        input_manager: The YamboInputManager instance (no AiiDA data, mostly),
        evaluator: The active ConvergenceEval instance tracking runtime state.
        explicit_value: The exact point value/vector to apply (e.g., 20.0 or [20.0, 4.0]).
        workflow_dict: The parallelism instructions dict ({'automatic':..., 'manual':..., ...}).
        index: The loop index tracking finishing concurrent submissions.
        group: AiiDA group where previous wfls are stored (for already-done detection).
        calc_type: 'qp', 'bse' or 'bse@qp' -> selects which namespace to converge.
        resource_rules: the rules of the resources for given values of the parameters (see `utils/resource_rules.py`).
        resources_base: the options of the steps at the start (`snapshot_base`), restored at each point before the rules.
        
    Returns:
        tuple: (input_manager, values_dict, already_done_pk, parent_nscf_node)
    """
    already_done = False
    values_dict = {}
    parallelism_instructions = workflow_dict if isinstance(workflow_dict, dict) else {}
    k_quantity = 0 

    # select the namespace we are converging: qp or bse
    yambo_manager = input_manager.bse if 'bse' in calc_type else input_manager.qp

    # Clean metadata parsing straight from your smart objects
    variables = evaluator.parameters.var
    mirror_config = evaluator.parameters.mirror_values

    # Standardize explicit_value into an iterable list matching variables
    # NOTE: it should also be a list, this first line is just for safety:
    current_values = explicit_value if isinstance(explicit_value, list) else [explicit_value]
    
    is_just_kmesh = len(variables)==1 and 'kpoint_mesh' in variables
    
    # we then take the value [p1,p2] if it's a list, else we take a list [p1] so the for loop below works when getting the `val` value. 
    current_values = current_values[index] if (isinstance(current_values[0], list) and not is_just_kmesh) else current_values[index:index+1]

    # Deepcopy the parameters dict (plain python data) before modifying it
    yambo_params = copy.deepcopy(yambo_manager.parameters)
    variables_block = yambo_params.setdefault('variables', {})
        
    for i,var in enumerate(variables):

        val = current_values[i]
        # 1. K-POINTS HANDLING (kpoint_mesh explicit or kpoint_density scalar)
        if var in ('kpoint_mesh', 'kpoint_density'):
            k_quantity = val

            # Single-point verification wraps the mesh one extra time (e.g.
            # -> [[5,5,5]] vs the flat [5,5,5] used by build_stepped_space).
            # Unwrap nested containers down to the innermost mesh/scalar.
            while isinstance(k_quantity, (tuple, list)) and len(k_quantity) == 1 and isinstance(k_quantity[0], (tuple, list)):
                k_quantity = k_quantity[0]

            # the NSCF has an explicit mesh only if one was set; with a `kpoints_distance` (the protocol) there is none:
            # the shift of the new mesh is then the default one (Gamma centred).
            try:
                k_quantity_shift = input_manager.nscf.get_mesh()[1]
            except (AttributeError, TypeError):
                k_quantity_shift = [0, 0, 0]

            if isinstance(k_quantity, (tuple, list)):
                input_manager.nscf.set_kmesh(k_quantity, k_quantity_shift, store=True)
            else:
                input_manager.nscf.set_k_density(k_quantity, force_parity=False, store=True)
            # `kpoints` and `kpoints_distance` are exclusive inputs of the PW workflow: the new mesh replaces the distance
            (getattr(input_manager.nscf, '__pydantic_extra__', None) or {}).pop('kpoints_distance', None)

            # Safely flush remote parent folder context if swapping grid meshes to prevent errors
            if input_manager.parent_folder is not None:
                try:
                    parent_folder = find_pw_parent(
                        take_calc_from_remote(input_manager.parent_folder), 
                        calc_type=['scf']
                    ).outputs.remote_folder 
                    input_manager.set_parent_folder(parent_folder)
                except Exception as exc: # no SCF parent: the new mesh needs the parents to be recomputed
                    LOGGER.info('no SCF parent for the new k-points mesh (%s), removing the parent folder', exc)
                    input_manager.remove_parent_folder()

            yambo_manager.settings = {**yambo_manager.settings, 'COPY_SAVE': False, 'COPY_DBS': False}
            
            values_dict[var] = k_quantity
            if var == 'kpoint_mesh': 
                k_quantity = 0

        # 2. DECOUPLED GENERIC PARAMETER & MIRROR HANDLING
        else:
            
            # Map out what elements require synchronization
            targets_to_update = [var]
            if isinstance(mirror_config, list):
                targets_to_update.extend(mirror_config)
            elif isinstance(mirror_config, dict) and var in mirror_config:
                targets_to_update.extend(mirror_config[var])

            # Dynamically retrieve unit string from parameter class properties
            declared_unit = evaluator.parameters.units if isinstance(evaluator.parameters.units, str) else evaluator.parameters.units[i]

            for target in targets_to_update:
                if 'bnd' in target.lower():
                    # Format: [[start_band, end_band], unit]
                    # Band counts are unitless...
                    variables_block[target] = [[1, val], '']
                else:
                    # Format: [value, unit] (e.g. [4.0, 'Ry'])
                    variables_block[target] = [val, declared_unit]
            
            values_dict[var] = val    # Extract scalar cutoff value

    yambo_manager.parameters = yambo_params

    # 3. PARALLELISM GENERATION SETUP
    if parallelism_instructions:
        ParallelismManager(input_manager, parallelism_instructions,
                           k_quantity=k_quantity, calc_type=calc_type).apply()
    
    if resource_rules:
        apply_resource_rules(input_manager, resource_rules, calc_type=calc_type,
                             point_values=values_dict, base=resources_base)

    # 4. PARENT CORRELATION DEPENDENCY SEARCH
    already_done, parent_nscf, parent_scf = search_in_group(input_manager, group, calc_type=calc_type)
    
    effective_parent = parent_nscf or parent_scf
    if effective_parent and input_manager.parent_folder is None:
        try:
            input_manager.set_parent_folder(load_node(effective_parent).outputs.remote_folder)
        except (NotExistent, AttributeError, KeyError) as exc:
            LOGGER.debug('calculation %s cannot be the parent: %s', effective_parent, exc)

    return input_manager, values_dict, already_done, parent_nscf

################################## parsers #####################################
def take_quantities(df, what=None):
    """Fetch the `what` results (and the failed/useful flags) from stored workflow nodes.

    The dataframe is already filled row by row in perform_next_step with the
    parameters (and pk/uuid). This function only loads the nodes to retrieve
    the converged quantities, keeping the heavy I/O out of the main loop.
    """

    what = what if what is not None else ['gap_eV']

    df['failed'] = False
    df['useful'] = True

    for i, row in df.iterrows():
        ywf_node = load_node(row['pk'])
        if ywf_node.is_finished_ok:
            quantities = ywf_node.outputs.output_ywfl_parameters.get_dict()
            for q in what:
                df.at[i, q] = quantities.get(q, False)
        else:
            df.at[i, 'failed'] = True
            df.at[i, 'useful'] = False
            for q in what:
                df.at[i, q] = False

    return df
