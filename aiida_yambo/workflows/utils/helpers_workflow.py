# -*- coding: utf-8 -*-
"""Classes for calcs e wfls analysis."""
from __future__ import absolute_import
from aiida_yambo.workflows.utils.predictor_1D import create_grid_1D
from aiida_yambo.workflows.utils.predictor_2D import The_Predictor_2D
import numpy as np
from scipy.optimize import curve_fit, minimize
from matplotlib import pyplot as plt, style
import pandas as pd
import copy
from ase import Atoms
from aiida_yambo.workflows.utils.helpers_aiida_yambo import *
from aiida_yambo.workflows.utils.helpers_aiida_yambo import calc_manager_aiida_yambo as calc_manager
from aiida_yambo.utils.common_helpers import *
from aiida_yambo.workflows.utils.optimization_module import * 
from aiida_yambo.workflows.utils.predictor_2D import * 
from aiida_yambo.workflows.utils.predictor_1D import * 
############################# AiiDA - independent ################################

#class convergence_workflow_manager:
def _normalize_workflow_dict(workflow_dict):
    """Convert persisted workflow history values to runtime numpy / pandas objects.

    During workflow execution, we need to work with numpy arrays and DataFrames
    for efficient computation. This function converts archived dict/list data
    back into their runtime representations.
    """
    workflow_dict = workflow_dict or {}
    # Convert stored array data to numpy arrays for numerical operations
    if 'array_conv' in workflow_dict:
        try:
            workflow_dict['array_conv'] = np.asarray(workflow_dict['array_conv'])
        except Exception:
            pass
    # Convert stored history dict to pandas DataFrame for efficient queries
    if 'workflow_story' in workflow_dict:
        try:
            workflow_dict['workflow_story'] = pd.DataFrame.from_dict(workflow_dict['workflow_story'])
        except Exception:
            pass
    return workflow_dict


def _serialize_workflow_dict(workflow_dict):
    """Convert runtime numpy / pandas objects back to serializable Python structures.

    After computation, convert back to JSON-serializable types (lists, dicts)
    so the workflow dict can be stored in AiiDA and passed between processes.
    """
    # Convert numpy arrays to lists for JSON serialization
    if 'array_conv' in workflow_dict:
        try:
            workflow_dict['array_conv'] = workflow_dict['array_conv'].tolist()
        except Exception:
            pass
    # Convert DataFrame back to dict representation
    if 'workflow_story' in workflow_dict:
        try:
            workflow_dict['workflow_story'] = workflow_dict['workflow_story'].to_dict()
        except Exception:
            pass
    return workflow_dict


def conversion_wrapper(func):
    """Wrap workflow helper functions and normalize serializable structures.

    This decorator ensures that workflow_dict values are converted to their runtime
    types before the wrapped function runs, and converted back afterwards.
    """
    def wrapper(*args, workflow_dict=None, success=False):
        # Convert persisted structure to runtime objects (np array, DataFrame)
        workflow_dict = _normalize_workflow_dict(workflow_dict)
        # Execute the wrapped function with runtime-ready data
        output = func(*args, workflow_dict=workflow_dict, success=success)
        # Convert back to serializable structures for storage
        workflow_dict = _serialize_workflow_dict(workflow_dict)
        return output
    return wrapper


def collect_inputs(inputs, kpoints, ideal_iter):
    """Collect starting input values for convergence variables.

    Parameters:
        inputs: calculation input namespace containing ``variables``.
        kpoints: AiiDA kpoints object to get the mesh from.
        ideal_iter: list of convergence parameter definitions.

    Returns:
        dict: parameter name -> starting value.

    Example:
        starting_inputs = collect_inputs(inputs, kpoints, ideal_iter)
    """
    starting_inputs = {}
    for i in ideal_iter:
        l = i['var']
        if not isinstance(i['var'], list):
            l = [i['var']]
        for var in l:
            if var not in starting_inputs:
                if 'mesh' in var or 'density' in var:
                    starting_inputs[var] = kpoints.get_kpoints_mesh()[0]
                else:
                    starting_inputs[var] = inputs['variables'][var]
                    if var in ['BndsRnXp', 'GbndRnge']:
                        starting_inputs[var] = starting_inputs[var][0][1]
                    elif not isinstance(starting_inputs[var][0], list):
                        starting_inputs[var] = starting_inputs[var][0]
    return starting_inputs


def _listify(value):
    """Return a value as a list if it is not already one."""
    return value if isinstance(value, list) else [value]


def _is_number(value):
    """Return True if the value is a numeric scalar."""
    return isinstance(value, (int, float, np.integer, np.floating))


def _build_dummy_delta_value(base, delta, r, first):
    """Build a dummy-space value using additive delta spacing."""
    if _is_number(delta):
        result = base + delta * (r + first - 1)
        return int(result) if isinstance(base, int) else result
    return [sum(x) for x in zip(base, [d * (r + first - 1) for d in delta])]


def _build_dummy_ratio_value(base, delta, r, var):
    """Build a dummy-space value using multiplicative ratio spacing."""
    if 'mesh' in var:
        factor = delta ** (r - 1)
        return [int(a * factor) if isinstance(a, int) else a * factor for a in base]
    if _is_number(base):
        result = base * delta ** (r - 1)
        return int(result) if isinstance(base, int) else result
    return [int(a * (d ** (r - 1))) if isinstance(a, int) else a * (d ** (r - 1)) for a, d in zip(base, delta)]


def _create_new_algorithm_2D_space(entry, hint, new_grid):
    """Build a parameter space for the new 2D convergence algorithm."""
    space = {}
    small_space = False
    l = _listify(entry['var'])

    if hint and not new_grid:
        for v in l:
            if entry['max'][l.index(v)] < hint[v]:
                small_space = True
            space[v] = [hint[v]]
        return space, small_space

    var = copy.deepcopy(l)
    if 'BndsRnXp' in l and 'GbndRnge' in l:
        start = copy.deepcopy(entry['start'])
        stop = copy.deepcopy(entry['stop'])
        metrics = copy.deepcopy(entry['delta'])
        gb_idx = l.index('GbndRnge')
        start.pop(gb_idx)
        stop.pop(gb_idx)
        metrics.pop(gb_idx)
        var.remove('GbndRnge')
    else:
        start = entry['start']
        stop = entry['stop']
        metrics = entry['delta']

    space.update(create_grid(edges=start + stop, delta=metrics, var=var, shift=[new_grid, new_grid], alpha=1 / 3))

    if 'BndsRnXp' in l and 'GbndRnge' in l:
        space['GbndRnge'] = copy.deepcopy(space['BndsRnXp'])

    return space, small_space


def _create_new_algorithm_1D_space(entry, hint, new_grid):
    """Build a parameter space for the new 1D convergence algorithm."""
    space = {}
    small_space = False
    l = _listify(entry['var'])

    if hint and not new_grid:
        for v in l:
            if v == 'kpoint_mesh':
                if entry['max'][0] < hint[v][0]:
                    small_space = True
            elif entry['max'] < hint[v]:
                small_space = True
            space[v] = [hint[v]]
        return space, small_space

    space.update(create_grid_1D(edges=[entry['start'], entry['stop']], delta=_listify(entry['delta']), var=copy.deepcopy(l), shift=new_grid * 2, alpha=1 / 3))
    return space, small_space


def _create_newton_space(entry, hint):
    """Build a parameter space for Newton-style convergence algorithms."""
    space = {}
    small_space = False
    l = _listify(entry['var'])
    start = _listify(entry['start'])
    stop = _listify(entry['stop'])
    metrics = _listify(entry['delta'])
    logspace = _listify(entry.get('log_spacing', [0]))
    if len(logspace) < len(l):
        logspace = logspace * len(l)

    for idx, v in enumerate(l):
        if logspace[idx]:
            count = int((stop[idx] - start[idx]) / metrics[idx])
            space[v] = list(np.logspace(np.log10(start[idx]), np.log10(stop[idx] + 1), count, dtype=int))
        elif start[idx] == stop[idx]:
            space[v] = [start[idx]] * entry['steps'] * entry['max_iterations']
        else:
            space[v] = list(np.arange(start[idx], stop[idx] + 1, int(metrics[idx])))

    if 'newton_2D_extra' in entry['convergence_algorithm']:
        new_space = {'BndsRnXp': [], 'GbndRnge': [], 'NGsBlkXp': []}
        for b in space['BndsRnXp']:
            for g in space['NGsBlkXp']:
                new_space['BndsRnXp'].append(b)
                new_space['GbndRnge'].append(b)
                new_space['NGsBlkXp'].append(g)

        if hint:
            for k in new_space.keys():
                new_space[k] = new_space[k][entry['steps'] * entry['iter'] :]

        return copy.deepcopy(new_space), small_space, False

    if hint and entry['convergence_algorithm'] == 'newton_1D_ratio':
        for v in l[:-1]:
            current = space[v]
            if v in hint and start[l.index(v)] != stop[l.index(v)]:
                index = abs((np.array(current) - hint[v])).argmin()
                if (len(current) - index - 1) < entry['steps']:
                    index -= 1
                try:
                    if abs((current[index + 1] + current[index]) / 2 - hint[v]) < 1e-1:
                        index += 1
                except Exception:
                    small_space = True
                current = current[index:]
                if len(current) < entry['steps']:
                    small_space = True
            elif (len(current) - 1) < entry['steps']:
                small_space = True
            space[v] = current

        current = space[l[-1]]
        v = l[-1]
        if v in hint and start[l.index(v)] != stop[l.index(v)]:
            index = abs((np.array(current) - hint[v])).argmin()
            if len(current) <= index + 1:
                small_space = True
            else:
                if current[index] < hint[v]:
                    index += 1
            if current[-1] < hint[v]:
                small_space = True
            if hint[v] == current[-1]:
                small_space = False
            current = current[index:]
        space[v] = current
        return space, small_space, True

    if hint and entry['convergence_algorithm'] != 'newton_2D_extra':
        for v in l:
            current = space[v]
            if v in hint and start[l.index(v)] != stop[l.index(v)]:
                index = abs((np.array(current) - hint[v])).argmin()
                if (len(current) - index) < entry['steps']:
                    small_space = True
                current = current[index:]
            elif len(current) < entry['steps']:
                small_space = True
            space[v] = current

        if hint.get('pop'):
            for k in list(space.keys()):
                if k not in l and space[k]:
                    space[k].pop(0)

    return space, small_space, False


def _populate_space_from_entry(space, entry, l, starting_inputs, existing_inputs, calc_dict, hint, first):
    """Populate the parameter space from a single convergence entry."""
    delta = entry.get('delta', entry.get('ratio'))
    if delta is None:
        return space, first

    if not isinstance(delta, list) or 'mesh' in entry['var']:
        delta = [delta]

    for var in l:
        start, stop = 1, entry['steps'] * entry['max_iterations'] + 1
        if hint and 'mesh' not in var:
            length_space = len(starting_inputs[var])
            start = -entry['steps'] * calc_dict['iter'] + 1
            stop = length_space - entry['steps'] * calc_dict['iter'] + 1
            starting_inputs[var] = hint[var]

        if var not in space:
            space[var] = []
            last_inputs = starting_inputs[var]
        else:
            last_inputs = space[var][-1]

        if 'dummy' in entry['convergence_algorithm'] and 'delta' in entry:
            for r in range(start, stop):
                if r <= 0:
                    continue
                new_val = _build_dummy_delta_value(last_inputs, delta[l.index(var)], r, first)
                space[var].append(new_val)

        elif 'dummy' in entry['convergence_algorithm'] and 'ratio' in entry:
            for r in range(start, stop):
                if r <= 0:
                    continue
                new_val = _build_dummy_ratio_value(starting_inputs[var], delta[l.index(var)], r, var)
                space[var].append(new_val)

        else:
            for r in range(len(entry['space'])):
                if 'kpoint_mesh' in entry['var'] and len(entry['var']) == 1:
                    new_val = entry['space'][r]
                else:
                    new_val = entry['space'][r][l.index(var)]
                space[var].append(new_val)

    return space, 1


def _is_explicit_space_request(workflow_dict):
    """Return True when workflow_dict is provided as explicit mode data."""
    if isinstance(workflow_dict, (list, tuple)) and len(workflow_dict) == 2:
        vars_, values = workflow_dict
        if isinstance(vars_, list) and isinstance(values, list) and values and all(isinstance(v, (list, tuple)) for v in values):
            return True
    return False


def _build_explicit_space(workflow_dict):
    """Build an explicit parameter space from explicit variable/value pairs."""
    vars_, values = workflow_dict
    if any(len(row) != len(vars_) for row in values):
        raise ValueError('explicit space rows must match the number of parameters.')
    return {var: [row[idx] for row in values] for idx, var in enumerate(vars_)}


def create_space(starting_inputs=None, workflow_dict=None, calc_dict=None, wfl_type='heavy', hint=None):
    """Create a workflow parameter space from various convergence modes.

    Supports several modes:
      - explicit mode: pass ``[vars, values]`` where ``vars`` is a list of parameter
        names and ``values`` is a list of rows. Example:

            workflow_dict = [['a', 'b', 'c'], [[1, 2, 3], [2, 3, 4]]]

        This returns ``{'a': [1, 2], 'b': [2, 3], 'c': [3, 4]}``.

      - new_algorithm_2D: pass a convergence entry dictionary with keys such as
        ``var``, ``start``, ``stop``, ``delta``, ``max``, ``iter``, and
        ``convergence_algorithm`` set to ``new_algorithm_2D``. Example:

            workflow_dict = [
                {
                    'var': ['BndsRnXp', 'GbndRnge', 'NGsBlkXp'],
                    'start': [10, 10, 1],
                    'stop': [20, 20, 5],
                    'delta': [2, 2, 1],
                    'max': [100, 100, 10],
                    'iter': 1,
                    'convergence_algorithm': 'new_algorithm_2D'
                }
            ]

      - new_algorithm_1D: pass a convergent dictionary with ``var``, ``start``,
        ``stop``, ``delta``, ``max``, ``iter``, and
        ``convergence_algorithm`` set to ``new_algorithm_1D``.
        Example:

            workflow_dict = [
                {
                    'var': 'kpoint_mesh',
                    'start': [4, 4, 4],
                    'stop': [8, 8, 8],
                    'delta': 1,
                    'max': [16, 16, 16],
                    'iter': 1,
                    'convergence_algorithm': 'new_algorithm_1D'
                }
            ]

      - newton: pass a convergent dictionary with ``var``, ``start``,
        ``stop``, ``delta`` and ``convergence_algorithm`` containing ``newton``.
        Example:

            workflow_dict = [
                {
                    'var': ['BndsRnXp', 'NGsBlkXp'],
                    'start': [10, 1],
                    'stop': [20, 5],
                    'delta': [2, 1],
                    'steps': 3,
                    'max_iterations': 3,
                    'convergence_algorithm': 'newton'
                }
            ]

      - dummy: pass a convergence dictionary with explicit ``space`` values or
        with ``delta`` / ``ratio`` defined. Example:

            workflow_dict = [
                {
                    'var': 'BndsRnXp',
                    'space': [[10], [12], [14]],
                    'convergence_algorithm': 'dummy'
                }
            ]

    Parameters:
        starting_inputs (dict): existing parameter values.
        workflow_dict: convergence definitions or explicit mode list.
        calc_dict (dict): current calculation definition for fallback mode.
        wfl_type (str): workflow type, currently kept for compatibility.
        hint (dict): optional hint values from previous analysis.

    Returns:
        tuple: (space, existing_inputs, small_space)
    """
    space = {}
    first = 0
    small_space = False
    new_grid = 0

    starting_inputs = starting_inputs or {}
    existing_inputs = copy.deepcopy(starting_inputs)
    if _is_explicit_space_request(workflow_dict):
        space = _build_explicit_space(workflow_dict)
        return space, existing_inputs, small_space

    if not workflow_dict:
        workflow_dict = [copy.deepcopy(calc_dict or {})]
        space = copy.deepcopy(starting_inputs)

    for entry in workflow_dict:
        wfl_type = entry.pop('convergence_algorithm', 'dummy')
        entry['convergence_algorithm'] = wfl_type
        l = _listify(entry['var'])

        if hint and wfl_type != 'newton_2D_extra' and hint.get('new_grid'):
            new_grid = entry['iter']

        if 'new_algorithm_2D' in wfl_type:
            new_space, small = _create_new_algorithm_2D_space(entry, hint, new_grid)
            space.update(new_space)
            small_space = small_space or small
            continue

        if 'new_algorithm_1D' in wfl_type:
            new_space, small = _create_new_algorithm_1D_space(entry, hint, new_grid)
            space.update(new_space)
            small_space = small_space or small
            continue

        if 'newton' in wfl_type and 'space' not in entry:
            new_space, small, break_loop = _create_newton_space(entry, hint)
            space.update(new_space)
            small_space = small_space or small
            if break_loop:
                break
            continue

        if 'delta' in entry:
            delta = entry['delta']
        elif 'ratio' in entry:
            delta = entry['ratio']
        else:
            delta = None

        if not isinstance(entry['var'], list):
            l = [entry['var']]
        if delta is not None and (not isinstance(delta, list) or 'mesh' in entry['var']):
            delta = [delta]

        space, first = _populate_space_from_entry(space, entry, l, starting_inputs, existing_inputs, calc_dict or {}, hint, first)

    return space, existing_inputs, small_space

def update_space(starting_inputs=None, calc_dict=None, wfl_type='heavy', hint=0,
                 existing_inputs=None, convergence_algorithm='smart'):
    """Update an existing convergence space based on new hint values.

    This helper adjusts ``delta`` or ``ratio`` values from ``calc_dict`` and
    rebuilds an updated parameter space.

    Parameters:
        starting_inputs (dict): current parameter values.
        calc_dict (dict): convergence calculation definition.
        wfl_type (str): workflow type, used for compatibility.
        hint (dict|int): hint values for updating the space.
        existing_inputs (dict): historical inputs to preserve prior values.
        convergence_algorithm (str): update mode, e.g. ``smart`` or ``dummy``.

    Returns:
        tuple: (param_space, existing_inputs)
    """
    starting_inputs = copy.deepcopy(starting_inputs or {})
    existing_inputs = copy.deepcopy(existing_inputs or {})
    space = {}
    first = 1
    if existing_inputs: starting_inputs = copy.deepcopy(existing_inputs)
    if convergence_algorithm == 'smart':
        factor = 1 # (calc_dict['iter']**0.5)
    elif convergence_algorithm == 'aggressive':
        factor = 1
    elif convergence_algorithm == 'dummy' and 'ratio' in calc_dict.keys():
        factor = 1
    elif convergence_algorithm == 'dummy':
        factor = 0   
        
    
    wfl_type = calc_dict['optimization']
    for j in [1]:
        
        l = calc_dict['var']
        i = calc_dict
        if not isinstance(i['var'],list):
            l=[i['var']]
        if 'delta' in i.keys(): 
            delta=i['delta']
            if not isinstance(delta,list) or ('kpoint_mesh' in i['var'] and not isinstance(calc_dict['delta'][calc_dict['var'].index('kpoint_mesh')],list)):
                delta=[delta]
                calc_dict['delta'] = [calc_dict['delta']]
            for var in l: 
                hint_ = (1+hint[var]*factor)
                if 'mesh' in var:
                    hint_= 1

                if isinstance(delta[calc_dict['var'].index(var)],int):
                        calc_dict['delta'][calc_dict['var'].index(var)] = int(calc_dict['delta'][calc_dict['var'].index(var)]*hint_)
                elif isinstance(delta[calc_dict['var'].index(var)],float):
                        calc_dict['delta'][calc_dict['var'].index(var)] = calc_dict['delta'][calc_dict['var'].index(var)]*hint_
                elif isinstance(delta[l.index(var)],list): 
                    if isinstance(delta[l.index(var)][-1],int):
                        calc_dict['delta'][calc_dict['var'].index(var)] = [int(d*hint_) for d in delta[calc_dict['var'].index(var)]]
                    else:
                        calc_dict['delta'][calc_dict['var'].index(var)] = [d*hint_ for d in delta[calc_dict['var'].index(var)]]

            delta = calc_dict['delta']
            
        if 'ratio' in i.keys(): delta=i['ratio']
        # if 'explicit' in ... :
            #for new_val in i['explicit']:
            #    space[var].append(new_val)
            
            #continue
               
        for var in l:  
            if hint==0:
                hint_ = 1 
            else:
                hint_ = hint[var]
        if 'mesh' in var:
            hint_= 1
        if not var in space.keys():
            space[var] = []

        if 'ratio' in i.keys():
            hint_ = hint_**0.5
            if convergence_algorithm == 'dummy':
                hint_ = 1
            starting_inputs[var] =  starting_inputs[var][i['steps']*calc_dict['iter']-1] 
            if isinstance(starting_inputs[var][0],int):
                is_integer = True
                starting_inputs[var][0] = int(starting_inputs[var][0]*hint_)
            elif isinstance(starting_inputs[var][0],list):
                is_integer = isinstance(starting_inputs[var][0][-1],int)
                for j in starting_inputs[var][0]:
                    if i['ratio'][l.index(var)][starting_inputs[var][0].index(j)] == 1:
                        starting_inputs[var][0][starting_inputs[var][0].index(j)] = starting_inputs[var][0][starting_inputs[var][0].index(j)]
                    else:
                        if is_integer:
                            starting_inputs[var][0][starting_inputs[var][0].index(j)] = int(starting_inputs[var][0][starting_inputs[var][0].index(j)]*hint_)
                        else:
                            starting_inputs[var][0][starting_inputs[var][0].index(j)] = starting_inputs[var][0][starting_inputs[var][0].index(j)]*hint_
            else:
                is_integer = False
                starting_inputs[var][0] = starting_inputs[var][0]*hint_
        else: 
            starting_inputs[var] =  starting_inputs[var][i['steps']*calc_dict['iter']-1]
        if 'delta' in i.keys():
            #for r in range(1,i['steps']*i['max_iterations']+1):
            for r in range(-i['steps']*calc_dict['iter']+1,len(existing_inputs[var])-i['steps']*calc_dict['iter']+1):
                if r <= 0: 
                    new_val = existing_inputs[var][i['steps']*calc_dict['iter']+r-1]
                elif isinstance(delta[l.index(var)],int) or isinstance(delta[l.index(var)],float):
                    new_val = starting_inputs[var][0]+delta[l.index(var)]*(r+first-1)
                    if not 'mesh' in var:
                        new_val = [new_val, starting_inputs[var][1]]
                elif isinstance(delta[l.index(var)],list): 
                    if not 'mesh' in var:
                        new_val = [sum(x) for x in zip(starting_inputs[var][0], [d*(r+first-1) for d in delta[l.index(var)]])]
                        new_val = [new_val, starting_inputs[var][1]]
                    else:
                        new_val = [sum(x) for x in zip(starting_inputs[var], [d*(r+first-1) for d in delta[l.index(var)]])]
                space[var].append(new_val)
                #print(new_val)
        elif 'ratio' in i.keys() and wfl_type != 'multivariate_newton':
            for r in range(-i['steps']*calc_dict['iter']+1,len(existing_inputs[var])-i['steps']*calc_dict['iter']+1):
                if r <= 0: 
                    new_val = existing_inputs[var][i['steps']*calc_dict['iter']+r-1]
                elif 'mesh' in var:
                        new_val = [int(a*b) for a,b in zip(starting_inputs[var], [delta[l.index(var)]**((r+first-1)),
                                                                             delta[l.index(var)]**((r+first-1)),
                                                                             delta[l.index(var)]**((r+first-1))])]
                elif isinstance(starting_inputs[var][0],int) or isinstance(starting_inputs[var][0],float):
                    if is_integer:
                        new_val = int(starting_inputs[var][0]*delta[l.index(var)]**((r+first-1)))
                    else:
                        new_val = starting_inputs[var][0]*delta[l.index(var)]**((r+first-1))
                    if not 'mesh' in var:
                        new_val = [new_val, starting_inputs[var][1]]
                elif isinstance(starting_inputs[var][0],list): 
                    if not 'mesh' in var:
                            if is_integer:
                                new_val = [int(a*b) for a,b in zip(starting_inputs[var][0], [d**((r+first-1)) for d in delta[l.index(var)]])]
                            else:
                                new_val = [a*b for a,b in zip(starting_inputs[var][0], [d**((r+first-1)) for d in delta[l.index(var)]])]
                            new_val = [new_val, starting_inputs[var][1]]
                        
                    space[var].append(new_val)
                    #print(new_val)

        elif 'ratio' in i.keys() and wfl_type == 'multivariate_newton':
            for r in range(-i['steps']*calc_dict['iter']+1,len(existing_inputs[var])-i['steps']*calc_dict['iter']+1):
                if r <= 0: 
                    new_val = existing_inputs[var][i['steps']*calc_dict['iter']+r-1]
                elif 'mesh' in var:
                        new_val = [int(a*b) for a,b in zip(starting_inputs[var], [delta[l.index(var)]**((r+first-1)),
                                                                             delta[l.index(var)]**((r+first-1)),
                                                                             delta[l.index(var)]**((r+first-1))])]
                elif isinstance(starting_inputs[var][0],int) or isinstance(starting_inputs[var][0],float):
                    if is_integer:
                        new_val = int(starting_inputs[var][0]*delta[l.index(var)]**((r+first-1)))
                    else:
                        new_val = starting_inputs[var][0]*delta[l.index(var)]**((r+first-1))
                    if not 'mesh' in var:
                        new_val = [new_val, starting_inputs[var][1]]
                elif isinstance(starting_inputs[var][0],list): 
                    if not 'mesh' in var:
                            if is_integer:
                                new_val = [int(a*b) for a,b in zip(starting_inputs[var][0], [d**((r+first-1)) for d in delta[l.index(var)]])]
                            else:
                                new_val = [a*b for a,b in zip(starting_inputs[var][0], [d**((r+first-1)) for d in delta[l.index(var)]])]
                            new_val = [new_val, starting_inputs[var][1]]
                        
                    space[var].append(new_val)

        elif wfl_type == 'multivariate_newton':
            for r in range(-i['steps']*calc_dict['iter']+1,len(existing_inputs[var])-i['steps']*calc_dict['iter']+1):
                if r <= 0: 
                    new_val = existing_inputs[var][i['steps']*calc_dict['iter']+r-1]
                
                else:
                    new_val = hint['space'][r][l.index(var)]
                    if is_integer: new_val = int(hint['space'][r][l.index(var)])
                    if isinstance(existing_inputs[var][0],list) and not 'mesh' in var:
                        new_val = [new_val,existing_inputs[var][0][1]]
                space[var].append(new_val)
            
        else:
            for r in range(len(i['space'])):
                new_val = i['space'][r][l.index(var)]
                
                space[var].append(new_val)
            
            #for ii in range(len(space[var]),len(existing_inputs[var])-len(space[var])):
            #    space[var].append(existing_inputs[var][ii])

        first = 1

    existing_inputs.update(space)
    param_space = copy.deepcopy(existing_inputs)
    for v in l:
        for r in range(calc_dict['steps']*calc_dict['iter']):
            param_space[v].pop(0)
    
    return param_space, existing_inputs


def convergence_workflow_manager(parameters_space, wfl_settings, inputs, kpoints):
    """Build a workflow manager dictionary for convergence analysis.

    The returned dictionary contains the full convergence plan, starting inputs,
    and the first generated parameter space.

    Example:
        workflow_manager = convergence_workflow_manager(parameters_space, wfl_settings, inputs, kpoints)
    """
    workflow_dict = {}
    new_l = []

    #here should be a default list of convergence "parameters_space".
    if isinstance(parameters_space,list):
        parameters_space = List(parameters_space)
    for i in parameters_space.get_list():
        new_conv = copy.deepcopy(i)
        new_conv['max_iterations'] = i.pop('max_iterations', 3)
        #new_conv['delta'] = i.pop('delta', 5)
        #new_conv['ratio'] = i.pop('ratio', 1.5)
        new_conv['steps'] = i.pop('steps', 3)
        new_conv['conv_thr'] = i.pop('conv_thr', 0.05)
        new_l.append(new_conv)
    

    #AiiDA calculation --> this is the only AiiDA dependence of the class...the rest is abstract
    ps = copy.deepcopy(new_l)
    ps.reverse()
    workflow_dict['ideal_iter'] = copy.deepcopy(ps)
    workflow_dict['true_iter'] = copy.deepcopy(ps)
    workflow_dict['type'] = 'AiiDA_calculation'
    
    copy_wfl_sett = copy.deepcopy(wfl_settings)
    workflow_dict['type'] = copy_wfl_sett.pop('type','heavy') #if cheap, it will do the optimization maintaining the other params as input (so: minimal)
    if workflow_dict['type'] == '1D_convergence': workflow_dict['type'] = 'heavy'
    workflow_dict['what'] = copy_wfl_sett.pop('what','gap_')

    workflow_dict['global_step'] = 0
    workflow_dict['fully_success'] = False

    workflow_dict['starting_inputs'] = collect_inputs(inputs, kpoints, new_l)
    workflow_dict['parameter_space'], existing_inputs, small_space = create_space(workflow_dict['starting_inputs'], new_l, workflow_dict['type'])
    workflow_dict['to_be_parsed'] = []
    workflow_dict['wfl_pk'] = []
    return workflow_dict

@conversion_wrapper
def build_story_global(calc_manager, quantities, workflow_dict=None, success=False):
    """Build or extend the global convergence history array."""
    if calc_manager['iter'] == 1:
        try:
            workflow_dict['array_conv']=np.array(workflow_dict['workflow_story']\
                [workflow_dict['workflow_story']['useful'] == True].iloc[-1,])
            workflow_dict['array_conv'] = np.column_stack((workflow_dict['array_conv'],quantities[:,:,1]))
        except:
            workflow_dict['array_conv']=np.array(quantities[:,:,1])
    else:
        workflow_dict['array_conv'] = np.column_stack((workflow_dict['array_conv'],quantities[:,:,1]))

@conversion_wrapper
def update_story_global(calc_manager, quantities, inputs, workflow_dict, success=False):
    """Append the latest calculation results to the global workflow story."""

    errors = False
    final_result = {}
    if workflow_dict['global_step'] == 0 :
        workflow_dict['workflow_story'] = pd.DataFrame(columns = ['global_step']+list(quantities.columns)+['parameters_studied']+\
                        ['useful','failed'])

    for i in range(calc_manager['steps']-calc_manager['skipped']):
            workflow_dict['global_step'] += 1
            if isinstance(calc_manager['var'],list):
                separator = ', '
                var_names = separator.join(calc_manager['var'])
            else:
                var_names = calc_manager['var']

            if  False in quantities.values[i].tolist():
                workflow_story_list = [workflow_dict['global_step']]+quantities.values[i].tolist()+[var_names]+\
                        [False, True]
                errors = True
            else:
                workflow_story_list = [workflow_dict['global_step']]+quantities.values[i].tolist()+[var_names]+\
                        [True, False]

            workflow_df = pd.DataFrame([workflow_story_list], columns = ['global_step']+list(quantities.columns)+['parameters_studied']+\
                    ['useful','failed'])

            #workflow_dict['workflow_story'] = workflow_dict['workflow_story'].append(workflow_df, ignore_index=True)
            workflow_dict['workflow_story'] = pd.concat([workflow_dict['workflow_story'],workflow_df], ignore_index=True)
   
    for i in range(1,len(workflow_dict['workflow_story'])+1):
        try:                
            last_ok_uuid = workflow_dict['workflow_story'].iloc[-i]['uuid']
            #last_ok_wfl = get_caller(last_ok_uuid, depth = 1)
            start_from_converged(inputs, last_ok_uuid)
            if calc_manager['var'] == 'kpoint_mesh' or calc_manager['var'] == 'kpoint_density':
                set_parent(inputs, load_node(last_ok_uuid))
            break
        except:
            last_ok_uuid = workflow_dict['workflow_story'].iloc[-1]['uuid']
            #last_ok_wfl = get_caller(last_ok_uuid, depth = 1)
    
    workflow_dict['workflow_story'] = workflow_dict['workflow_story'].replace({np.nan:None})

    final_result={'uuid': last_ok_uuid,'errors':errors}
        
    return final_result

@conversion_wrapper
def post_analysis_update(inputs, calc_manager, oversteps, none_encountered, workflow_dict=None, success=False):
    """Update the workflow story after convergence analysis completes."""
    final_result = {}
    
    if 'new_algorithm' in calc_manager['convergence_algorithm']:
        workflow_dict['workflow_story'].loc[:,'useful']=False
        print(oversteps)
        if success == 'new_grid': 
            return {}
        elif success:
            workflow_dict['workflow_story'].loc[oversteps[0],'useful']=True
    else:
        for i in oversteps: 
            gs = workflow_dict['workflow_story']['global_step'][workflow_dict['workflow_story']['uuid']==i].index
            workflow_dict['workflow_story'].loc[gs,'useful']=False
    
    if len(oversteps)>0 and calc_manager['convergence_algorithm']=='dummy':
        for i in range(calc_manager['iter']*(calc_manager['steps']-calc_manager['skipped'])-len(oversteps)):
            for j in calc_manager['var']:
                workflow_dict['parameter_space'][j].pop(0)
    for i in none_encountered: 
            gs = workflow_dict['workflow_story']['global_step'][workflow_dict['uuid']==i].index
            workflow_dict['workflow_story'].loc[gs,'failed']=True
            workflow_dict['workflow_story'].loc[gs,'useful']=False

    #try:
    if len(workflow_dict['workflow_story'][(workflow_dict['workflow_story']['useful'] == True) & (workflow_dict['workflow_story']['failed'] == False)]) > 0:
        last_ok_uuid = workflow_dict['workflow_story'][(workflow_dict['workflow_story']['useful'] == True) & (workflow_dict['workflow_story']['failed'] == False)].iloc[-1]['uuid']
        #last_ok_wfl = get_caller(last_ok_uuid, depth = 1)
        mesh = 'kpoint_mesh' in calc_manager['var'] or 'kpoint_density' in calc_manager['var']
        start_from_converged(inputs, last_ok_uuid,mesh=mesh)
        if 'kpoint_density' in calc_manager['var']:
            calc_manager['kdensity'] = workflow_dict['workflow_story'][(workflow_dict['workflow_story']['useful'] == True) & (workflow_dict['workflow_story']['failed'] == False)].iloc[-1]['kpoint_density']

        if 'kpoint_mesh' in calc_manager['var'] or 'kpoint_density' in calc_manager['var']:
            set_parent(inputs, load_node(last_ok_uuid)) 
    else: 
        final_result={}
    #except:
    #    last_ok_uuid = workflow_dict['workflow_story'].iloc[-1]['uuid']
    #    last_ok_wfl = get_caller(last_ok_uuid, depth = 1)

    workflow_dict['workflow_story'] = workflow_dict['workflow_story'].replace({np.nan:None})
    
    try:
        final_result={'uuid': last_ok_uuid,}
    except:
        final_result={}
        
    return final_result

################################################################################
############################## NEW convergence_evaluator #######################

def prepare_for_ce(workflow_dict=None, keys=None, var_=None, var_full=None, bug_newton1d=False, new_algorithm=False):
    """Prepare convergence evaluator inputs from the workflow history.

    This function extracts and reorganizes the workflow history (stored as a DataFrame)
    into the format expected by the predictor classes. It filters to useful calculations,
    extracts parameter arrays and result arrays, and handles special cases like k-mesh
    treatment and duplicate removal.

    Returns:
        tuple: (real, lines, homo, bande)
            real: filtered history DataFrame containing only useful calculations
            lines: dict mapping parameter name -> numpy array of values
            homo: dict mapping analysis key -> numpy array of results
            bande: band structure data from parent PW calculation, if available
    """
    workflow_dict = workflow_dict or pd.DataFrame()
    keys = keys or ['gap_GG']
    var_ = var_ or []
    var_full = var_full or []
    workflow_story = workflow_dict
    # Filter history to only useful, non-failed calculations
    real = workflow_story[(workflow_story.failed == False) & (workflow_story.useful == True)].copy()

    # For new_algorithm modes, further filter by matching all convergence parameters
    if new_algorithm:
        group_label = ', '.join(var_full)  # Create a label like 'BndsRnXp, NGsBlkXp, kpoint_mesh'
        real = real[real.parameters_studied == group_label].copy()

    # Extract parameter arrays from the history
    lines = {}
    for k in var_:
        values = real[k].tolist()  # Get all values for this parameter
        if k == 'kpoint_mesh':
            # For k-mesh, compute the product of the three mesh dimensions
            mesh = np.asarray([v[0] for v in values])  # Extract first row of each mesh tuple
            lines[k] = np.prod(mesh, axis=1)  # Multiply dimensions: nx * ny * nz
            lines['mesh'] = mesh[:, 0]  # Also store the raw mesh for later analysis
        else:
            # For scalar parameters, extract the first (or only) value from each row
            array_vals = np.asarray([list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v] for v in values])
            lines[k] = array_vals[:, 0]

    # Extract result arrays for each analysis quantity (e.g., gap, energy)
    homo = {key: real[key].to_numpy() for key in keys}

    # If dealing with newton_1D and duplicate parameter values exist, keep only first occurrence
    if bug_newton1d:
        for k in var_:
            # Find indices of unique values (keep_index gives first occurrence)
            unique_idx = np.unique(lines[k], return_index=True)[1]
            if len(unique_idx) < len(lines[k]):
                # Create mask for unique indices
                mask = np.zeros(len(lines[k]), dtype=bool)
                mask[unique_idx] = True
                # Apply mask to all parameter and result arrays
                lines[k] = lines[k][mask]
                for key in keys:
                    homo[key] = homo[key][mask]

    # Attempt to load band structure data from parent Quantum ESPRESSO calculation
    bande = 0
    try:
        pw = find_pw_parent(load_node(real.uuid.values[-1]).outputs.remote_folder.creator)
        bande = pw.outputs.output_band.get_bands()  # Shape: (n_kpoints, n_bands)
    except Exception:
        bande = 0  # Not critical if unavailable

    return real, lines, homo, bande


#@conversion_wrapper
def analysis_and_decision(calc_dict, workflow_manager, parameter_space=None, hints=None):
    """Analyze convergence results and decide whether to continue or stop.

    This is the core convergence analysis function. It examines all results from
    the current convergence iteration and uses predictor classes to:
    1. Fit the convergence behavior to a model (power law, exponential, etc.)
    2. Extrapolate to find the parameter value that satisfies convergence criterion
    3. Recommend the next point to run, or declare convergence achieved

    The choice of predictor (dummy evaluator, 1D, 2D) depends on the convergence_algorithm.
    """
    parameter_space = parameter_space or workflow_manager['parameter_space']
    hints = hints or {}  # Hints from previous analysis iterations
    workflow_story = pd.DataFrame(workflow_manager['workflow_story'])

    # Exclude the GbndRnge parameter if BndsRnXp is present (they're usually tied together)
    var = [v for v in calc_dict['var'] if not (v == 'GbndRnge' and 'BndsRnXp' in calc_dict['var'])]
    # Collect UUIDs of failed calculations
    none_encountered = list(workflow_story.uuid[workflow_story.failed == True])
    hint = {}

    # Extract and prepare data from workflow history
    real, lines, homo, bande = prepare_for_ce(
        workflow_dict=workflow_story,
        keys=workflow_manager['what'],  # The quantities we're analyzing (e.g., gap energies)
        var_=var,
        var_full=calc_dict['var'],
        bug_newton1d=calc_dict['convergence_algorithm'] == 'newton_1D',
        new_algorithm='new_algorithm' in calc_dict['convergence_algorithm'],
    )

    is_converged = True
    oversteps = 0

    def _analyse_predictor(predictor, old_hints, thr_kwargs):
        """Helper to run predictor analysis with consistent hint handling.
        
        Args:
            predictor: Predictor instance (1D or 2D variant)
            old_hints: Previous hints from earlier iterations
            thr_kwargs: Threshold parameters to pass to analyse()
        
        Returns:
            tuple: (is_converged, oversteps, hint, extra_value)
        """
        # If we have a new_grid hint, reset old_hints to force fresh analysis
        if old_hints.get('new_grid'):
            predictor.analyse(old_hints={}, **thr_kwargs)
        else:
            # Otherwise, provide previous hints to guide the search
            predictor.analyse(old_hints=old_hints, **thr_kwargs)
        # Check if convergence is reached: fit must be valid AND target point must be found
        return predictor.check_passed and predictor.point_reached, predictor.index, predictor.next_step, predictor.extra

    # Analyze each target quantity (typically just one per convergence step)
    for k in workflow_manager['what']:
        if 'new_algorithm_2D' in calc_dict['convergence_algorithm']:
            # 2D convergence: analyze two parameters simultaneously (e.g., bands and PW cutoff)
            predictor = The_Predictor_2D(
                grid=lines,
                result=real,
                bande=bande,
                r=homo[k],
                calc_dict=calc_dict,
                k=k,
            )
            thr_kwargs = dict(thr_fx=calc_dict['thr_fx'], thr_fy=calc_dict['thr_fy'], thr_fxy=calc_dict['thr_fxy'])
            is_converged, oversteps, hint, extra = _analyse_predictor(predictor, hints, thr_kwargs)
            hint['extrapolation'] = extra
            hint['extrapolation_units'] = 'eV'

        elif 'new_algorithm_1D' in calc_dict['convergence_algorithm']:
            # 1D convergence: analyze one parameter (e.g., k-point mesh or bands)
            predictor = The_Predictor_1D(
                grid=lines,
                result=real,
                bande=bande,
                r=homo[k],
                calc_dict=calc_dict,
                k=k,
            )
            thr_kwargs = dict(thr_fx=calc_dict['thr_fx'])
            is_converged, oversteps, hint, extra = _analyse_predictor(predictor, hints, thr_kwargs)
            hint['extrapolation'] = extra
            hint['extrapolation_units'] = 'eV'

        else:
            # Traditional dummy convergence: simple window-based convergence check
            evaluator = Convergence_evaluator(
                conv_array=homo,
                calc_dict=calc_dict,
                var=var,
                p_val=lines,
                quantities=workflow_manager['what'],
                real=real,
                workflow_dict=workflow_story,
            )
            is_converged, oversteps, hint = evaluator.analysis()

        # If not converged for this quantity, no need to check others
        if not is_converged:
            break

    # If BndsRnXp and GbndRnge are both parameters, keep them synchronized
    if 'BndsRnXp' in calc_dict['var'] and 'GbndRnge' in calc_dict['var'] and hint and 'BndsRnXp' in hint:
        hint['GbndRnge'] = hint['BndsRnXp']

    return is_converged, oversteps, none_encountered, homo, hint
    



###############################  parallelism  ####################################

def build_parallelism_instructions(instructions):
    """Normalize parallelism instruction keys for workflow execution."""

    instructions['automatic'] = instructions.pop('automatic', False)
    instructions['semi-automatic'] = instructions.pop('semi-automatic', False)
    instructions['manual'] = instructions.pop('manual', False)
    instructions['function'] = instructions.pop('function', False)

    return instructions
