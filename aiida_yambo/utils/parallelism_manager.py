# -*- coding: utf-8 -*-
"""Parallelism management for Yambo calculations.

Replaces the monolithic ``set_parallelism`` function with a
``ParallelismManager`` class that:

1. reads the current calculation parameters through the high-level input
   manager getters (``get_Nb``/``get_Gcut``/``nscf.get_mesh``) instead of
   poking raw ``variables`` dicts,
2. resolves the parallelism to apply from a set of instruction profiles
   (``automatic``/``semi-automatic``/``manual``/``function``),
3. applies it directly to a ``YamboInputManager`` (or to a raw AiiDA inputs
   namespace, e.g. in the yamborestart error-handling path).

New-style instruction schema::

    {
      'automatic': [
        {'id': 'std_1', 'bands': [1, 100], 'gcut': [1, 18],
         'mode': 'balanced', 'resources': {'num_machines': 1, ...}},
        {'id': 'std_2', 'bands': [101, 10**4], 'gcut': [1, 18],
         'mode': 'memory', 'resources': {...}},
      ],
      'semi-automatic': {'resources': {...}},
      'manual': [
        {'id': 'low', 'bands': [1, 100], 'parallelism': {'X_CPU': '...', ...},
         'resources': {...}},
      ],
      'function': callable,
    }

Gates are optional; ``None``/``0``/``[0]`` means "always matches". The legacy
schema (a dict keyed by profile name with yambo variable names such as
``BndsRnXp``/``NGsBlkXp``/``GbndRnge``) is still accepted and translated
automatically.
"""
from __future__ import absolute_import

import copy

from aiida_yambo.utils.parallelism_finder import find_parallelism_qp
from aiida_yambo.utils.defaults.create_defaults import periodical
from aiida_yambo.utils.common_helpers import find_gw_info
from aiida_yambo.utils.high_level_API.general_model import OptionsModel, ResourcesModel

#: yambo runlevel arguments which identify a QP-type run (for 'semi-automatic').
_QP_RUNLEVELS = {'gw0', 'gw', 'HF_and_locXC', 'hfx', 'hf', 'sc'}

#: legacy yambo variable names that map to the same semantic quantity.
_BANDS_HINTS = ('BndsRnXp', 'BndsRnXs', 'GbndRnge')
_GCUT_HINTS = ('NGsBlkXp', 'NGsBlkXs', 'BSENGBlk')
_GATE_KEYS = ('bands', 'gcut', 'kpoints', 'kpoint_density')


class ApplySet:
    """The result of resolving a parallelism instruction set."""

    __slots__ = ('variables', 'resources', 'pop_keys')

    def __init__(self, variables=None, resources=None, pop_keys=None):
        self.variables = dict(variables or {})
        self.resources = dict(resources or {})
        self.pop_keys = list(pop_keys or [])


class ParallelismProfile:
    """A single parallelism profile with optional value gates.

    Gates are expressed with semantic keys ('bands', 'gcut', 'kpoints',
    'kpoint_density'); each gate is either ``None`` (always matches) or a
    ``(low, high)`` range (a scalar/list of length 1 means an exact match).
    """

    def __init__(self, id='', gates=None, mode=None, parallelism=None,
                 resources=None, function=None):
        self.id = id or ''
        self.gates = {k: _normalize_gate(v) for k, v in (gates or {}).items()}
        self.mode = mode
        self.parallelism = dict(parallelism or {})
        self.resources = dict(resources or {})
        self.function = function


############################ helpers ##########################################

def _normalize_gate(value):
    """Normalize a gate specification to a ``(low, high)`` tuple or ``None``.

    ``None``, ``[0]`` and ``0`` all mean "always matches" (legacy default).
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = list(value)
        if len(value) == 0:
            return None
        if len(value) == 1:
            if value[0] in (0, None):
                return None
            return (value[0], value[0])
        if len(value) == 2:
            return (min(value), max(value))
        raise ValueError('Invalid gate specification: {!r}'.format(value))
    if isinstance(value, (int, float)):
        if value == 0:
            return None
        return (value, value)
    raise ValueError('Invalid gate specification: {!r}'.format(value))


def _union_gate(hints):
    """Merge several legacy hint ranges into a single gate."""
    normalized = [_normalize_gate(hint) for hint in hints if hint is not None]
    if not normalized or any(gate is None for gate in normalized):
        return None
    return (min(gate[0] for gate in normalized), max(gate[1] for gate in normalized))


def _matches(snapshot, gates):
    for key, gate in gates.items():
        if gate is None:
            continue
        low, high = gate
        value = snapshot.get(key, 0)
        if value is None:
            value = 0
        if not (low <= value <= high):
            return False
    return True


def _bands_gcut_from_variables(variables):
    """Replicate the get_Nb/get_Gcut semantics from a raw variables dict."""
    bands, gcut = 0, 0
    for tag in _BANDS_HINTS:
        value = (variables.get(tag) or [[0, 0], ''])[0]
        bands = max(bands, value[-1])
    for tag in _GCUT_HINTS:
        value = variables.get(tag)
        if isinstance(value, list) and len(value) > 0:
            value = value[0]
        if value is not None:
            gcut = value
    return bands, gcut


def _pop_keys_from_variables(variables):
    """Keys holding a previous parallelism that should be removed/replaced."""
    pop_keys = []
    for key in variables.keys():
        if 'LinAlg' in key:
            continue
        if 'CPU' in key or 'ROLEs' in key:
            pop_keys.append(key)
    return pop_keys


############################ targets ##########################################

class YamboInputManagerTarget:
    """Target adapter operating on the high-level YamboInputManager."""

    def __init__(self, manager, calc_type='qp'):
        self.manager = manager
        self.calc_type = calc_type

    def yambo_manager(self):
        return (self.manager.bse if 'bse' in self.calc_type else self.manager.qp)

    def current_variables(self):
        return self.yambo_manager().parameters.get('variables', {})

    def current_resources(self):
        options = self.yambo_manager().metadata.options
        if options is None or options.resources is None:
            return {}
        return options.resources.model_dump()

    def snapshot(self):
        yambo_manager = self.yambo_manager()
        bands = yambo_manager.get_Nb()
        gcut = yambo_manager.get_Gcut()
        kpoints = 1
        try:
            mesh = self.manager.nscf.get_mesh()[0]
            kpoints = int(mesh[0] * mesh[1] * mesh[2])
        except Exception:
            pass
        occupied, ecut = 1, 0
        try:
            occupied, ecut = periodical(self.manager.scf.structure.get_ase())
        except Exception:
            pass
        bands_solver, qp, last_qp, runlevels = bands, 0, 0, []
        try:
            bands_solver, qp, last_qp, runlevels = find_gw_info(yambo_manager)
        except Exception:
            pass
        return {
            'bands': bands,
            'gcut': gcut,
            'kpoints': kpoints,
            'occupied': occupied,
            'ecut': ecut,
            'bands_solver': bands_solver,
            'qp': qp,
            'last_qp': last_qp,
            'runlevels': runlevels,
        }

    def apply(self, bundle):
        yambo_manager = self.yambo_manager()
        variables_block = yambo_manager.parameters.setdefault('variables', {})
        for key in bundle.pop_keys:
            variables_block.pop(key, None)
        variables_block.update(bundle.variables)
        yambo_manager.parameters['variables'] = variables_block

        options = yambo_manager.metadata.options or OptionsModel()
        if options.resources is None:
            options.resources = ResourcesModel()
        for key, value in bundle.resources.items():
            setattr(options.resources, key, value)
        if 'num_cores_per_mpiproc' in bundle.resources:
            options.prepend_text = 'export OMP_NUM_THREADS={}'.format(
                bundle.resources['num_cores_per_mpiproc'])
        yambo_manager.metadata.options = options
        return True


class RawInputsTarget:
    """Target adapter for legacy flat YamboCalculation inputs namespaces.

    Used by the yamborestart error-handling path, where ``self.ctx.inputs`` is
    the raw AiiDA inputs namespace (``.parameters`` Dict node and
    ``.metadata.options``).
    """

    def __init__(self, inputs):
        self.inputs = inputs

    def current_variables(self):
        parameters = self.inputs.parameters
        if hasattr(parameters, 'get_dict'):
            return parameters.get_dict().get('variables', {})
        return (parameters or {}).get('variables', {})

    def current_resources(self):
        try:
            return dict(self.inputs.metadata.options.resources)
        except Exception:
            return {}

    def snapshot(self):
        variables = self.current_variables()
        bands, gcut = _bands_gcut_from_variables(variables)
        kpoints = 1
        try:
            mesh = self.inputs.kpoints.get_kpoints_mesh()[0]
            kpoints = int(mesh[0] * mesh[1] * mesh[2])
        except Exception:
            pass
        occupied, ecut = 1, 0
        try:
            occupied, ecut = periodical(self.inputs.structure.get_ase())
        except Exception:
            pass
        bands_solver, qp, last_qp, runlevels = bands, 0, 0, []
        try:
            bands_solver, qp, last_qp, runlevels = find_gw_info(self.inputs)
        except Exception:
            pass
        return {
            'bands': bands,
            'gcut': gcut,
            'kpoints': kpoints,
            'occupied': occupied,
            'ecut': ecut,
            'bands_solver': bands_solver,
            'qp': qp,
            'last_qp': last_qp,
            'runlevels': runlevels,
        }

    def apply(self, bundle):
        from aiida_yambo.utils.common_helpers import update_dict
        parameters = self.inputs.parameters
        new_parameters = update_dict(
            parameters,
            list(bundle.variables),
            list(bundle.variables.values()),
            sublevel='variables',
            pop_list=bundle.pop_keys,
        )
        self.inputs.parameters = new_parameters
        try:
            options = self.inputs.metadata.options
            if options is None:
                options = {}
            options['resources'] = dict(bundle.resources)
            self.inputs.metadata.options = options
        except Exception:
            pass
        return True


############################ manager ##########################################

class ParallelismManager:
    """Resolve and apply parallelism options to Yambo inputs.

    Parameters
    ----------
    target:
        A ``YamboInputManager`` or any object exposing the target adapter
        interface (``current_variables``, ``current_resources``,
        ``snapshot``, ``apply``), e.g. a ``RawInputsTarget``.
    instructions:
        Parallelism instructions dict (see module docstring). Legacy schemas
        are translated automatically and the input is never mutated.
    k_quantity:
        The kpoint density (1/Å) of the step being prepared; used for the
        'kpoint_density' gate. 0 means not constrained.
    calc_type:
        'qp', 'bse' or 'bse@qp' -> which sub-manager to modify.
    """

    def __init__(self, target, instructions=None, k_quantity=0.0, calc_type='qp'):
        from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
        if isinstance(target, YamboInputManager):
            target = YamboInputManagerTarget(target, calc_type)
        self.target = target
        self.k_quantity = k_quantity
        self.calc_type = calc_type
        self.instructions = self._normalize(instructions)

    ############################ normalization ################################

    @staticmethod
    def _normalize(instructions):
        instructions = copy.deepcopy(instructions) if instructions else {}
        normalized = {
            'automatic': [],
            'semi-automatic': bool(instructions.pop('semi-automatic', None)),
            'manual': [],
            'function': instructions.pop('function', None) or False,
        }
        normalized['automatic'] = ParallelismManager._to_profiles(
            instructions.pop('automatic', None))
        normalized['manual'] = ParallelismManager._to_profiles(
            instructions.pop('manual', None))
        return normalized

    @staticmethod
    def _to_profiles(specs):
        if not specs or specs is True:
            return []
        if isinstance(specs, dict):
            profiles = []
            for name, spec in specs.items():
                profile = ParallelismProfile(**ParallelismManager._translate_legacy(spec))
                profile.id = profile.id or name
                profiles.append(profile)
            return profiles
        return [ParallelismManager._spec_to_profile(spec) for spec in specs]

    @staticmethod
    def _translate_legacy(spec):
        """Convert a legacy per-profile dict into new-schema keyword arguments."""
        spec = copy.deepcopy(spec) if isinstance(spec, dict) else {}
        gates = {}
        band_hints = [spec.pop(key) for key in _BANDS_HINTS if spec.get(key) is not None]
        if band_hints:
            gates['bands'] = _union_gate(band_hints)
        gcut_hints = [spec.pop(key) for key in _GCUT_HINTS if spec.get(key) is not None]
        if gcut_hints:
            gates['gcut'] = _union_gate(gcut_hints)
        for key, gate_key in (('kpoints', 'kpoints'), ('kpoints_density', 'kpoint_density')):
            if spec.get(key) is not None:
                gates[gate_key] = _normalize_gate(spec.pop(key))
        return {
            'gates': gates,
            'mode': spec.pop('mode', None),
            'parallelism': spec.pop('parallelism', None),
            'resources': spec.pop('resources', None),
        }

    @staticmethod
    def _spec_to_profile(spec):
        if isinstance(spec, ParallelismProfile):
            return spec
        spec = dict(spec or {})
        gates = dict(spec.pop('gates', {}) or {})
        for key in _GATE_KEYS:
            if key in spec:
                gates[key] = spec.pop(key)
        return ParallelismProfile(
            id=spec.pop('id', ''),
            gates=gates,
            mode=spec.pop('mode', None),
            parallelism=spec.pop('parallelism', None),
            resources=spec.pop('resources', None),
            function=spec.pop('function', None),
        )

    ############################ introspection ################################

    def _snapshot(self):
        snapshot = self.target.snapshot()
        snapshot['kpoint_density'] = self.k_quantity if self.k_quantity else 0
        return snapshot

    def _current_resources(self):
        return self.target.current_resources() or {}

    def _pop_keys(self):
        return _pop_keys_from_variables(self.target.current_variables())

    @staticmethod
    def _covers_qp(runlevels):
        return (not runlevels) or any(r in _QP_RUNLEVELS for r in runlevels)

    ############################ resolution ###################################

    def resolve(self):
        if self.instructions.get('automatic'):
            return self._resolve_gated(self.instructions['automatic'], automatic=True)
        if self.instructions.get('semi-automatic'):
            snapshot = self._snapshot()
            if self._covers_qp(snapshot.get('runlevels') or []):
                return self._semi_automatic(snapshot)
            return None
        if self.instructions.get('manual'):
            return self._resolve_gated(self.instructions['manual'], automatic=False)
        function = self.instructions.get('function')
        if function:
            if not callable(function):
                raise TypeError(
                    "'function' parallelism must be a callable, got {!r}".format(function))
            return self._coerce_result(function(self))
        return None

    def _resolve_gated(self, profiles, automatic):
        snapshot = None
        for profile in profiles:
            if profile.gates and snapshot is None:
                snapshot = self._snapshot()
            if profile.gates and not _matches(snapshot, profile.gates):
                continue
            variables = (dict(profile.parallelism) if not automatic
                         else {'PAR_def_mode': profile.mode or 'balanced'})
            resources = dict(profile.resources or self._current_resources())
            return ApplySet(variables, resources, self._pop_keys())
        return None

    def _semi_automatic(self, snapshot):
        resources = self._current_resources()
        names = ('num_machines', 'num_mpiprocs_per_machine', 'num_cores_per_mpiproc')
        solver_resources = {k: resources.get(k, 1) for k in names}
        new_parallelism, new_resources = find_parallelism_qp(
            solver_resources['num_machines'],
            solver_resources['num_mpiprocs_per_machine'],
            solver_resources['num_cores_per_mpiproc'],
            snapshot.get('bands_solver', 1) or 1,
            snapshot.get('occupied', 1) or 1,
            snapshot.get('qp', 0) or 0,
            snapshot.get('kpoints', 1) or 1,
            snapshot.get('last_qp', 0) or 0,
            namelist={},
        )
        return ApplySet(dict(new_parallelism or {}), dict(new_resources or {}),
                        self._pop_keys())

    @staticmethod
    def _coerce_result(result):
        if result is None or result is False:
            return None
        if isinstance(result, ApplySet):
            return result
        if isinstance(result, (tuple, list)) and len(result) == 3:
            return ApplySet(*result)
        if isinstance(result, dict) and result.get('resources') is not None:
            variables = {k: value for k, value in result.items() if k != 'resources'}
            return ApplySet(variables, result['resources'])
        raise ValueError('Invalid result from function parallelism: {!r}'.format(result))

    ############################ application ##################################

    def apply(self):
        """Resolve the parallelism and apply it to the target in place.

        Returns True if a parallelism set was applied, False otherwise.
        """
        bundle = self.resolve()
        if bundle is None:
            return False
        return bool(self.target.apply(bundle))

    __call__ = apply