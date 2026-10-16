# -*- coding: utf-8 -*-
"""Classes for calcs e wfls analysis."""
from __future__ import absolute_import
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt, style
import pandas as pd
import copy

try:
    from aiida.orm import Dict, Str, load_node, KpointsData, RemoteData
    from aiida.plugins import CalculationFactory, DataFactory
    from aiida_yambo.utils.common_helpers import *
except:
    pass
from aiida_yambo.utils.parallelism_finder import *
from aiida_yambo.utils.defaults.create_defaults import *
from aiida_yambo.utils.high_level_API.general_model import OptionsModel, ResourcesModel
from aiida_yambo.utils.parallelism_manager import ParallelismManager
#we try to use netcdf
try:
    from netCDF4 import Dataset
except ImportError:
    _has_netcdf = False
else:
    _has_netcdf = True
################################################################################
################################################################################

def set_parallelism(instructions_, inputs, k_quantity, yambo_manager=None):
    """Backward-compatible wrapper around :class:`ParallelismManager`.

    Returns ``(new_parallelism, new_resources, pop_list)`` as before, so legacy
    callers can keep applying the triple by hand.
    """
    from aiida_yambo.utils.parallelism_manager import ParallelismManager

    calc_type = 'qp'
    if yambo_manager is not None and getattr(inputs, 'bse', None) is yambo_manager:
        calc_type = 'bse'

    manager = ParallelismManager(inputs, instructions_, k_quantity=k_quantity,
                                 calc_type=calc_type)
    bundle = manager.resolve()
    if bundle is None:
        return False, False, False
    return bundle.variables, bundle.resources, bundle.pop_keys


#class calc_manager_aiida_yambo: 
def calc_manager_aiida_yambo(calc_info={}, wfl_settings={}): #tuning of these hyperparameters
    
    calc_dict = {}
    calc_dict.update(wfl_settings)
    calc_dict.update(calc_info)
    calc_dict['iter']  = 0
    calc_dict['G_iter']  = 1
    calc_dict['success'] = False
    calc_dict['skipped'] = 0
    calc_dict['conv_thr'] = calc_dict.pop('conv_thr',0.05)
    calc_dict['conv_thr_units'] = calc_dict.pop('conv_thr_units','eV')
    calc_dict['ratio'] = calc_dict.pop('ratio',1.2)
    calc_dict['max_iterations'] = calc_dict.pop('max_iterations',3)
    calc_dict['steps'] = calc_dict.pop('steps',3)
    calc_dict['conv_window'] = calc_dict.pop('conv_window',calc_dict['steps'])
    calc_dict['functional_form'] = calc_dict.pop('functional_form','power_law')
    
    calc_dict['convergence_algorithm'] = calc_dict.pop('convergence_algorithm','dummy') #1D, multivariate_optimization...

    if len(calc_dict['var']) == 3 and isinstance(calc_dict['var'],list) and calc_dict['convergence_algorithm'] == 'dummy':
        calc_dict['convergence_algorithm'] = 'new_algorithm_2D'
        calc_dict['steps'] = 6
    if len(calc_dict['var']) == 1 and isinstance(calc_dict['var'],list) and 'start' in calc_dict.keys() \
        and calc_dict['convergence_algorithm'] == 'dummy' and calc_dict['var'][0] =='kpoint_mesh' :
        calc_dict['convergence_algorithm'] = 'new_algorithm_1D'
        calc_dict['steps'] = 4
    
    if calc_dict['convergence_algorithm'] == 'new_algorithm_2D':
        calc_dict['thr_fx'] = calc_dict.pop('thr_fx',5e-5)
        calc_dict['thr_fy'] = calc_dict.pop('thr_fy',5e-5)
        calc_dict['thr_fxy'] = calc_dict.pop('thr_fxy',1e-8)
    if calc_dict['convergence_algorithm'] == 'new_algorithm_1D':
        calc_dict['thr_fx'] = calc_dict.pop('thr_fx',5e-5)

    
    return calc_dict

################################## update_parameters - create parameters space #####################################
import copy
from typing import Dict, Any, Tuple
from aiida.orm import Dict as AiiDA_Dict, KpointsData, load_node

def updater(input_manager, evaluator, explicit_value, workflow_dict: dict, index: int, group, calc_type='qp') -> Tuple[Any, Dict[str, Any], Any, Any]:
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

            k_quantity_shift = input_manager.nscf.get_mesh()[1]

            if isinstance(k_quantity, (tuple, list)):
                input_manager.nscf.set_kmesh(k_quantity, k_quantity_shift, store=True)
            else:
                input_manager.nscf.set_k_density(k_quantity, force_parity=False, store=True)

            # Safely flush remote parent folder context if swapping grid meshes to prevent errors
            if input_manager.parent_folder is not None:
                try:
                    parent_folder = find_pw_parent(
                        take_calc_from_remote(input_manager.parent_folder), 
                        calc_type=['scf']
                    ).outputs.remote_folder 
                    input_manager.set_parent_folder(parent_folder)
                except Exception:
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
    
    # 4. PARENT CORRELATION DEPENDENCY SEARCH
    already_done, parent_nscf, parent_scf = search_in_group(input_manager, group, calc_type=calc_type)
    
    effective_parent = parent_nscf or parent_scf
    if effective_parent and input_manager.parent_folder is None:
        try:
            input_manager.set_parent_folder(load_node(effective_parent).outputs.remote_folder)
        except Exception:
            pass

    return input_manager, values_dict, already_done, parent_nscf

################################## parsers #####################################
def take_quantities(df, what=['gap_eV']):
    """Fetch the `what` results (and the failed/useful flags) from stored workflow nodes.

    The dataframe is already filled row by row in perform_next_step with the
    parameters (and pk/uuid). This function only loads the nodes to retrieve
    the converged quantities, keeping the heavy I/O out of the main loop.
    """

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

def start_from_converged(inputs, node_uuid, mesh=False):
    node = load_node(node_uuid)
    #for i in range(1,len(node.called)+1):
    for i in range(len(node.called)):
        try:
            called = node.called[i]
            variables = settings = resources = max_wallclock_seconds = None
            for key in ('yres__yambo__parameters', 'qp__yambo__parameters', 'bse__yambo__parameters'):
                if key in called.inputs:
                    variables = called.inputs[key].get_dict()['variables']
                    settings = called.inputs['settings'].get_dict() if 'settings' in called.inputs else {}
                    break
            if variables is None:
                continue

            options = called.called[0].get_options()
            resources = options['resources']
            max_wallclock_seconds = options['max_wallclock_seconds']

            if hasattr(inputs, 'qp') and hasattr(inputs, 'bse'):
                # new YamboInputManager style
                for yambo_manager in (inputs.qp, inputs.bse):
                    yambo_manager.parameters['variables'].update(variables)
                    if settings:
                        yambo_manager.settings.update(settings)
                    if yambo_manager.metadata.options is None:
                        yambo_manager.metadata.options = OptionsModel()
                    if resources:
                        yambo_manager.metadata.options.resources = ResourcesModel(**resources)
                    if max_wallclock_seconds:
                        yambo_manager.metadata.options.max_wallclock_seconds = max_wallclock_seconds
                if mesh:
                    try:
                        inputs.nscf.set_kmesh(*called.inputs.nscf__kpoints.get_kpoints_mesh(), store=True)
                    except Exception:
                        pass
            else:
                # legacy calc_inputs builder style
                inputs.yres.yambo.parameters = update_dict(
                    inputs.yres.yambo.parameters,
                    list(variables.keys()), list(variables.values()), sublevel='variables'
                )
                if settings:
                    inputs.yres.yambo.settings = update_dict(
                        inputs.yres.yambo.settings,
                        list(settings.keys()), list(settings.values())
                    )
                inputs.yres.yambo.metadata.options.resources = resources
                inputs.yres.yambo.metadata.options.max_wallclock_seconds = max_wallclock_seconds
                if mesh: inputs.nscf.kpoints = called.inputs.nscf__kpoints
            break
        except:
            pass