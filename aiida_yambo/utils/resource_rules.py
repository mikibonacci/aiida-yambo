# -*- coding: utf-8 -*-
"""Resources which depend on the values of the parameters, for the convergence workflow.

A rule says: *when* the parameters have some values, *use* these resources, *for* these steps::

    resource_rules = [
        {'when': {'BndsRnXp': {'min': 200}, 'kpoints': {'max': 8}},
         'for': ['qp'],
         'use': {'num_machines': 2, 'num_mpiprocs_per_machine': 32, 'max_wallclock_seconds': 7200,
                 'parallelism': 'auto'}},
        {'when': {'kpoint_mesh': [4, 4, 4]}, 'use': {'num_machines': 4}},
        {'use': {'num_machines': 1}},                       # no `when`: always matches (the default)
    ]

- ``when``: the names are yambo variables (``BndsRnXp``, ``NGsBlkXp``, ``FFTGvecs``...), or ``bands`` (the largest
  number of bands), ``gcut`` (the largest G-vectors cutoff), ``kpoint_mesh``, ``kpoint_density``, ``kpoints``
  (the number of k-points). A condition is a number or a list (it must be equal) or a dictionary with ``min``
  and/or ``max``. All the conditions must hold. A name which is not available never matches.
- ``for``: the steps of the workflow, among ``scf``, ``nscf``, ``qp``, ``bse`` (default: all of them). The p2y run
  uses the options of the first yambo step (``qp`` or ``bse``).
- ``use``: ``num_machines``, ``num_mpiprocs_per_machine``, ``num_cores_per_mpiproc``, ``max_wallclock_seconds``,
  ``queue_name``, ``account``, ``prepend_text``, ``custom_scheduler_commands``, ``max_walltime`` and ``max_number_of_nodes`` (the limits of the
  restarts of the yambo steps), and, for ``qp`` and ``bse`` only, ``parallelism``: a dictionary of yambo
  parallelism variables, or ``'auto'`` to let the solver of the plugin choose them.

For each step, the first rule which matches is used. At every point of the convergence the options of the steps
are first brought back to the ones of the start, so nothing of a previous point is left.
"""
import copy
import logging

from aiida_yambo.utils.high_level_API.general_model import OptionsModel, ResourcesModel

LOGGER = logging.getLogger(__name__)

STEPS = ('scf', 'nscf', 'qp', 'bse')
YAMBO_STEPS = ('qp', 'bse')
RESOURCES_KEYS = ('num_machines', 'num_mpiprocs_per_machine', 'num_cores_per_mpiproc')
OPTIONS_KEYS = ('max_wallclock_seconds', 'queue_name', 'account', 'prepend_text', 'custom_scheduler_commands')
RESTART_KEYS = ('max_walltime', 'max_number_of_nodes')
USE_KEYS = RESOURCES_KEYS + OPTIONS_KEYS + RESTART_KEYS + ('parallelism',)
SPECIAL_NAMES = ('bands', 'gcut', 'kpoint_mesh', 'kpoint_density', 'kpoints')
_RANGE_KEYS = {'min', 'max'}


################################ validation ###################################

def validate_resource_rules(rules):
    """Raise a ValueError if `rules` is not a valid list of rules (see the module docstring)."""
    if not isinstance(rules, (list, tuple)):
        raise ValueError(f'resource_rules must be a list of rules, got {type(rules).__name__}')
    for index, rule in enumerate(rules):
        where = f'resource_rules[{index}]'
        if not isinstance(rule, dict):
            raise ValueError(f'{where} must be a dictionary, got {rule!r}')
        unknown = sorted(set(rule) - {'when', 'for', 'use'})
        if unknown:
            raise ValueError(f"{where}: unknown keys {unknown}, allowed: 'when', 'for', 'use'")
        if not rule.get('use'):
            raise ValueError(f"{where}: 'use' is required and cannot be empty")

        steps = rule.get('for', STEPS)
        if isinstance(steps, str) or not set(steps) <= set(STEPS) or len(steps) == 0:
            raise ValueError(f"{where}: 'for' must be a list of steps among {STEPS}, got {steps!r}")

        use = rule['use']
        unknown = sorted(set(use) - set(USE_KEYS))
        if unknown:
            raise ValueError(f"{where}: unknown keys in 'use' {unknown}, allowed: {USE_KEYS}")
        for key in RESOURCES_KEYS + ('max_wallclock_seconds',) + RESTART_KEYS:
            if key in use and (isinstance(use[key], bool) or not isinstance(use[key], int) or use[key] < 1):
                raise ValueError(f"{where}: '{key}' must be a positive integer, got {use[key]!r}")
        parallelism = use.get('parallelism')
        if parallelism is not None:
            if parallelism != 'auto' and not isinstance(parallelism, dict):
                raise ValueError(f"{where}: 'parallelism' must be 'auto' or a dictionary of yambo variables")
            if 'for' in rule and not set(steps) <= set(YAMBO_STEPS):
                raise ValueError(f"{where}: 'parallelism' can be used only for the yambo steps {YAMBO_STEPS}")

        when = rule.get('when', {})
        if not isinstance(when, dict):
            raise ValueError(f"{where}: 'when' must be a dictionary, got {when!r}")
        for name, condition in when.items():
            if isinstance(condition, dict):
                if not condition or not set(condition) <= _RANGE_KEYS:
                    raise ValueError(f"{where}: the range of '{name}' needs 'min' and/or 'max', got {condition!r}")
                if None in condition.values():
                    raise ValueError(f"{where}: the range of '{name}' has a None limit, omit it instead")
                if len(condition) == 2 and condition['min'] > condition['max']:
                    raise ValueError(f"{where}: 'min' is larger than 'max' for '{name}'")


################################## values #####################################

def _unwrap(value):
    """The number of a yambo variable: [[1, N], unit] -> N, [v, unit] -> v, v -> v."""
    if isinstance(value, (list, tuple)) and len(value) > 0:
        first = value[0]
        if isinstance(first, (list, tuple)):
            return first[-1]
        return first
    return value


def _find_in_variables(input_manager, name, calc_type):
    order = ('bse', 'qp') if 'bse' in calc_type else ('qp', 'bse')
    for step in order:
        variables = getattr(input_manager, step).parameters.get('variables', {})
        if name in variables:
            return _unwrap(variables[name])
    return None


def current_values(input_manager, calc_type, names, point_values=None):
    """The values of `names` for the point being prepared: `{name: value or None}`."""
    point_values = point_values or {}
    yambo_manager = input_manager.bse if 'bse' in calc_type else input_manager.qp
    values = {}
    mesh = None
    nscf = input_manager.nscf
    if nscf is not None:
        try:
            mesh = [int(m) for m in nscf.get_mesh()[0]]
        except Exception as exc:  # no NSCF mesh yet
            LOGGER.debug('no k-points mesh: %s', exc)
    for name in names:
        if name == 'kpoint_mesh':
            point = point_values.get('kpoint_mesh')
            while isinstance(point, (list, tuple)) and len(point) == 1 and isinstance(point[0], (list, tuple)):
                point = point[0]
            values[name] = list(point) if point is not None else mesh
        elif name == 'kpoint_density':
            values[name] = point_values.get('kpoint_density')
        elif name == 'kpoints':
            values[name] = int(mesh[0] * mesh[1] * mesh[2]) if mesh else None
        elif name == 'bands':
            values[name] = yambo_manager.get_Nb() or None
        elif name == 'gcut':
            values[name] = yambo_manager.get_Gcut() or None
        elif name in point_values:
            values[name] = _unwrap(point_values[name])
        else:
            values[name] = _find_in_variables(input_manager, name, calc_type)
    return values


def _condition_holds(value, condition):
    if value is None:
        return False
    if isinstance(condition, dict):
        if isinstance(value, (list, tuple)):
            return False
        return condition.get('min', value) <= value <= condition.get('max', value)
    if isinstance(condition, (list, tuple)):
        return list(value) == list(condition) if isinstance(value, (list, tuple)) else False
    return value == condition


def rule_matches(rule, values):
    """True if all the conditions of the rule hold for `values` (as given by :func:`current_values`)."""
    return all(_condition_holds(values.get(name), condition) for name, condition in rule.get('when', {}).items())


################################### base ######################################

def _parallelism_keys(variables):
    return [key for key in variables if ('CPU' in key and 'LinAlg' not in key) or 'ROLEs' in key or key == 'PAR_def_mode']


def snapshot_base(input_manager):
    """The options of every step, as plain data: what the steps go back to at each point."""
    base = {}
    for step in STEPS:
        manager = getattr(input_manager, step, None)
        if manager is None:
            continue
        entry = {'options': copy.deepcopy(manager.metadata.options.model_dump(exclude_none=True))}
        variables = manager.parameters.get('variables', {})
        entry['parallelism'] = {key: copy.deepcopy(variables[key]) for key in _parallelism_keys(variables)}
        for key in RESTART_KEYS:
            if getattr(manager, key, None) is not None:
                entry[key] = getattr(manager, key)
        base[step] = entry
    return base


def _reset(manager, entry):
    manager.metadata.options = OptionsModel(**copy.deepcopy(entry['options']))
    variables = manager.parameters.get('variables')
    if variables is not None:
        for key in _parallelism_keys(variables):
            variables.pop(key)
        variables.update(copy.deepcopy(entry['parallelism']))
    for key in RESTART_KEYS:
        if key in entry:
            setattr(manager, key, entry[key])
        elif hasattr(manager, key):
            delattr(manager, key)


################################## apply ######################################

def _strip_omp(text):
    return ''.join(line for line in text.splitlines(keepends=True) if 'OMP_NUM_THREADS' not in line)


def _use(manager, step, use, calc_type, input_manager):
    options = manager.metadata.options
    if any(key in use for key in RESOURCES_KEYS):
        resources = {**options.resources.model_dump(), **{key: use[key] for key in RESOURCES_KEYS if key in use}}
        options.resources = ResourcesModel(**resources)
        options.prepend_text = 'export OMP_NUM_THREADS={}\n'.format(resources['num_cores_per_mpiproc']) \
            + _strip_omp(options.prepend_text)
    for key in ('max_wallclock_seconds', 'queue_name', 'account'):
        if key in use:
            setattr(options, key, use[key])
    for key in ('prepend_text', 'custom_scheduler_commands'):
        if key in use:  # added to the ones set on the manager
            current = getattr(options, key)
            setattr(options, key, current + '\n' + use[key] if current else use[key])
    for key in RESTART_KEYS:
        if key in use:
            setattr(manager, key, use[key])
    parallelism = use.get('parallelism')
    if isinstance(parallelism, dict):
        variables = manager.parameters.setdefault('variables', {})
        for key in _parallelism_keys(variables):
            variables.pop(key)
        variables.update(parallelism)
    elif parallelism == 'auto' and step in YAMBO_STEPS:
        from aiida_yambo.utils.parallelism_manager import ParallelismManager
        ParallelismManager(input_manager, {'semi-automatic': True}, calc_type=step).apply()


def apply_resource_rules(input_manager, rules, calc_type='qp', point_values=None, base=None):
    """Bring the steps back to `base`, then apply to each step the first rule which matches.

    `base` comes from :func:`snapshot_base` (taken when the workflow starts). If it is None the current options are
    kept as they are (so they must not already contain the effect of a previous call).
    Returns {step: index of the rule applied}.
    """
    rules = [dict(rule) for rule in rules or []]
    if not rules:
        return {}
    names = sorted({name for rule in rules for name in rule.get('when', {})})
    applied = {}
    values = current_values(input_manager, calc_type, names, point_values)

    for step in STEPS:
        manager = getattr(input_manager, step, None)
        if manager is None:
            continue
        if base is not None and step in base:
            _reset(manager, base[step])
        for index, rule in enumerate(rules):
            if step not in rule.get('for', STEPS) or not rule_matches(rule, values):
                continue
            use = dict(rule['use'])
            if step not in YAMBO_STEPS:
                use.pop('parallelism', None)
                use.pop('max_walltime', None)
                use.pop('max_number_of_nodes', None)
            _use(manager, step, use, calc_type, input_manager)
            applied[step] = index
            break
    return applied
