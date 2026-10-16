# -*- coding: utf-8 -*-
"""Classes for calcs e wfls analysis. hybrid AiiDA and not_AiiDA...hopefully"""
from __future__ import absolute_import
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt, style
import pandas as pd
import copy
import cmath

try:
    from aiida.orm import Dict, Str, load_node, KpointsData
    from aiida.plugins import CalculationFactory, DataFactory
    from aiida_yambo.utils.common_helpers import *
    from aiida_yambo.utils.parallelism_finder import *
except:
    pass

################################################################################
'''
PAR_def_mode= "balanced"       # [PARALLEL] Default distribution mode ("balanced"/"memory"/"workload")
'''
################################################################################
def fix_parallelism(resources, failed_calc):
    """Reset parallelism to PAR_def_mode='balanced' on a failed calculation.

    The parallelism computation is delegated to :class:`ParallelismManager`
    (single code path with the convergence updater); the legacy triple
    ``(new_parallelism, new_resources, pop_list)`` is returned for the
    yamborestart error handlers.
    """
    bundle = _fix_to_balanced(resources, failed_calc)
    if bundle is None:
        return False, False, False
    return bundle.variables, bundle.resources, bundle.pop_keys

def _fix_to_balanced(resources, failed_calc):
    from aiida_yambo.utils.parallelism_manager import ParallelismManager, RawInputsTarget

    instructions = {'automatic': [{'mode': 'balanced', 'resources': resources}]}
    manager = ParallelismManager(RawInputsTarget(failed_calc.inputs), instructions)
    return manager.resolve()

def fix_memory(resources, failed_calc, exit_status, max_nodes, iteration):
    """Handle a memory error by improving the mpi/openmp balance.

    We increase the number of nodes (respecting ``max_nodes``) when possible and
    swap MPI tasks for extra OpenMP threads when the machine has no GPU. The
    parallelism profile itself is resolved through :class:`ParallelismManager`.
    """
    resources = copy.copy(resources)

    increase_nodes = False
    if max_nodes > resources['num_machines']:  # and iteration < 2:
        increase_nodes = True

    if resources['num_mpiprocs_per_machine'] > 1:
        if increase_nodes:
            resources['num_machines'] = min(max_nodes, int(resources['num_machines'] * 1.5))

        try:
            has_gpu = failed_calc.outputs.output_parameters.get_dict()['has_gpu']
        except Exception:
            has_gpu = False

        if not has_gpu:
            resources['num_cores_per_mpiproc'] = int(resources['num_cores_per_mpiproc'] * 2)
            resources['num_mpiprocs_per_machine'] = int(resources['num_mpiprocs_per_machine'] / 2)

    bundle = _fix_to_balanced(resources, failed_calc)
    if bundle is None:
        return False, False, False
    return bundle.variables, bundle.resources, bundle.pop_keys


def fix_time(options, restart, max_walltime):
    options['max_wallclock_seconds'] = \
                            int(options['max_wallclock_seconds']*1.5*restart)

    if options['max_wallclock_seconds'] > max_walltime:
        options['max_wallclock_seconds'] = int(max_walltime)

    return options
