# -*- coding: utf-8 -*-
"""Unit tests for the ParallelismManager (pure logic, no AiiDA DB needed).

These tests exercise the normalization, gate matching, resolution and
in-place application on the high-level YamboInputManager, whose construction
does not require a database/profile.
"""
import pytest

from aiida_yambo.utils.parallelism_manager import (
    ApplySet,
    ParallelismManager,
    ParallelismProfile,
    YamboInputManagerTarget,
    _matches,
    _normalize_gate,
    _pop_keys_from_variables,
)
from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager


def make_manager(variables, arguments=None, calc_type='qp'):
    im = YamboInputManager()
    im.qp.parameters = {'variables': dict(variables), 'arguments': arguments or []}
    im.bse.parameters = {}
    im.scf.structure = None
    im.nscf = None
    return im


GPU_RESOURCES = {'num_machines': 2, 'num_mpiprocs_per_machine': 16, 'num_cores_per_mpiproc': 1}


class TestGateHelpers:

    def test_normalize_gate(self):
        assert _normalize_gate(None) is None
        assert _normalize_gate([0]) is None
        assert _normalize_gate(0) is None
        assert _normalize_gate(5) == (5, 5)
        assert _normalize_gate([5]) == (5, 5)
        assert _normalize_gate([2, 8]) == (2, 8)
        assert _normalize_gate([8, 2]) == (2, 8)
        with pytest.raises(ValueError):
            _normalize_gate([1, 2, 3])

    def test_matches(self):
        snapshot = {'bands': 50, 'gcut': 4.0, 'kpoints': 24}
        assert _matches(snapshot, {'bands': (0, 100)})
        assert not _matches(snapshot, {'bands': (60, 100)})
        assert _matches(snapshot, {'bands': (0, 100), 'gcut': None})
        assert _matches(snapshot, {'kpoints': (1, 64)})
        assert not _matches(snapshot, {'kpoints': (100, 100)})

    def test_pop_keys(self):
        variables = {'BndsRnXp': [[1, 50], ''], 'X_CPU': '1 2', 'X_ROLEs': 'q k',
                     'SE_CPU': '1', 'LinAlgBreak_CPU': '1', 'PAR_def_mode': 'balanced'}
        pop = _pop_keys_from_variables(variables)
        assert 'X_CPU' in pop and 'X_ROLEs' in pop and 'SE_CPU' in pop
        assert 'BndsRnXp' not in pop
        assert 'LinAlgBreak_CPU' not in pop
        assert 'PAR_def_mode' not in pop


class TestLegacyTranslation:

    def test_automatic_legacy(self):
        im = make_manager({'BndsRnXp': [[1, 50], '']}, arguments=['gw0'])
        instructions = {
            'automatic': {
                'std': {
                    'BndsRnXp': [1, 100],
                    'GbndRnge': [1, 12],
                    'NGsBlkXp': [2, 8],
                    'mode': 'balanced',
                    'resources': GPU_RESOURCES,
                },
            },
        }
        manager = ParallelismManager(im, instructions, calc_type='qp')
        profile = manager.instructions['automatic'][0]
        assert profile.id == 'std'
        # band hints merged into a single bands gate
        assert profile.gates['bands'] == (1, 100)
        assert profile.gates['gcut'] == (2, 8)
        assert profile.mode == 'balanced'
        # input must not be mutated
        assert 'BndsRnXp' in instructions['automatic']['std']

    def test_legacy_match_all_zero(self):
        im = make_manager({'BndsRnXp': [[1, 50], '']}, arguments=['gw0'])
        instructions = {'automatic': {'a': {'NGsBlkXp': [0], 'mode': 'balanced', 'resources': GPU_RESOURCES}}}
        profile = ParallelismManager(im, instructions, calc_type='qp').instructions['automatic'][0]
        assert profile.gates['gcut'] is None

    def test_legacy_kpoints_density_mapped(self):
        im = make_manager({'BndsRnXp': [[1, 50], '']}, arguments=['gw0'])
        instructions = {'automatic': {'a': {'kpoints_density': [0.5, 2.0], 'mode': 'balanced',
                                            'resources': GPU_RESOURCES}}}
        profile = ParallelismManager(im, instructions, calc_type='qp').instructions['automatic'][0]
        assert profile.gates.get('kpoint_density') == (0.5, 2.0)
        assert 'gcut' not in profile.gates


class TestResolution:

    def test_automatic_first_match_wins(self):
        im = make_manager({'BndsRnXp': [[1, 50], ''], 'NGsBlkXp': [12.0, 'Ry']}, arguments=['gw0'])
        instructions = {
            'automatic': [
                {'id': 'a1', 'bands': [1, 100], 'gcut': [0, 8], 'mode': 'balanced', 'resources': GPU_RESOURCES},
                {'id': 'a2', 'bands': [1, 100], 'gcut': [8, 50], 'mode': 'memory',
                 'resources': {'num_machines': 4, 'num_mpiprocs_per_machine': 16, 'num_cores_per_mpiproc': 1}},
            ],
        }
        bundle = ParallelismManager(im, instructions, calc_type='qp').resolve()
        assert bundle.variables == {'PAR_def_mode': 'memory'}

    def test_no_match_returns_none(self):
        im = make_manager({'BndsRnXp': [[1, 50], ''], 'NGsBlkXp': [12.0, 'Ry']}, arguments=['gw0'])
        instructions = {'automatic': [{'bands': [1000, 2000], 'mode': 'balanced', 'resources': GPU_RESOURCES}]}
        assert ParallelismManager(im, instructions, calc_type='qp').resolve() is None

    def test_density_gate_uses_k_quantity(self):
        im = make_manager({'BndsRnXp': [[1, 50], '']}, arguments=['gw0'])
        instructions = {'automatic': [{'kpoint_density': [0.5, 2.0], 'mode': 'balanced', 'resources': GPU_RESOURCES}]}
        assert ParallelismManager(im, instructions, k_quantity=0.0, calc_type='qp').resolve() is None
        assert ParallelismManager(im, instructions, k_quantity=0.8, calc_type='qp').resolve() is not None

    def test_manual_mode(self):
        im = make_manager({'BndsRnXp': [[1, 50], '']}, arguments=['gw0'])
        instructions = {'manual': [
            {'id': 'm1', 'parallelism': {'X_CPU': '1 2 2', 'X_ROLEs': 'q k g c v'},
             'resources': {'num_machines': 8, 'num_mpiprocs_per_machine': 16, 'num_cores_per_mpiproc': 1}},
        ]}
        bundle = ParallelismManager(im, instructions, calc_type='qp').resolve()
        assert bundle.variables['X_CPU'] == '1 2 2'
        assert bundle.resources['num_machines'] == 8

    def test_function_mode(self):
        im = make_manager({'BndsRnXp': [[1, 50], '']}, arguments=['gw0'])

        def profile(manager):
            return {'PAR_def_mode': 'workload', 'resources': GPU_RESOURCES}

        bundle = ParallelismManager(im, {'function': profile}, calc_type='qp').resolve()
        assert bundle.variables == {'PAR_def_mode': 'workload'}

    def test_semi_automatic_solver(self):
        im = make_manager({'BndsRnXp': [[1, 40], ''], 'GbndRnge': [[1, 40], '']},
                          arguments=['dip', 'x', 'gw0'])
        im.qp.metadata.options.resources.num_machines = 2
        im.qp.metadata.options.resources.num_mpiprocs_per_machine = 8
        bundle = ParallelismManager(im, {'semi-automatic': True}, calc_type='qp').resolve()
        assert bundle is not None
        assert 'DIP_CPU' in bundle.variables or 'X_CPU' in bundle.variables

    def test_semi_automatic_skipped_for_non_qp(self):
        im = make_manager({'BndsRnXp': [[1, 40], '']}, arguments=['bse'])
        im.qp.metadata.options.resources.num_machines = 2
        im.qp.metadata.options.resources.num_mpiprocs_per_machine = 8
        assert ParallelismManager(im, {'semi-automatic': True}, calc_type='qp').resolve() is None


class TestApplication:

    def test_apply_modifies_input_manager(self):
        im = make_manager({'BndsRnXp': [[1, 50], ''], 'NGsBlkXp': [4.0, 'Ry'],
                           'X_CPU': '1 2 2', 'X_ROLEs': 'q k g c v'},
                          arguments=['gw0'])
        instructions = {'automatic': [{'bands': [1, 100], 'mode': 'balanced', 'resources': GPU_RESOURCES}]}
        manager = ParallelismManager(im, instructions, calc_type='qp')
        assert manager.apply() is True

        variables = im.qp.parameters['variables']
        assert variables['PAR_def_mode'] == 'balanced'
        assert 'X_CPU' not in variables, 'stale CPU/ROLEs parallelism must be popped'
        assert 'X_ROLEs' not in variables
        resources = im.qp.metadata.options.resources.model_dump()
        assert resources['num_machines'] == 2
        assert im.qp.metadata.options.prepend_text == 'export OMP_NUM_THREADS=1\n'

    def test_apply_false_when_nothing_matches(self):
        im = make_manager({'BndsRnXp': [[1, 50], '']}, arguments=['gw0'])
        instructions = {'automatic': [{'bands': [1000, 2000], 'mode': 'balanced', 'resources': GPU_RESOURCES}]}
        assert ParallelismManager(im, instructions, calc_type='qp').apply() is False

    def test_apply_bse_target(self):
        im = YamboInputManager()
        im.qp.parameters = {}
        im.bse.parameters = {'variables': {'BndsRnXs': [[1, 60], ''], 'NGsBlkXs': [3.0, 'Ry']},
                             'arguments': ['gw0']}
        im.scf.structure = None
        im.nscf = None
        instructions = {'automatic': [{'mode': 'balanced', 'resources': GPU_RESOURCES}]}
        assert ParallelismManager(im, instructions, calc_type='bse').apply() is True
        assert im.bse.parameters['variables']['PAR_def_mode'] == 'balanced'
        assert im.qp.parameters.get('variables', {}) == {}, 'qp must be left untouched'

    def test_manual_with_empty_instructions_returns_none(self):
        im = make_manager({'BndsRnXp': [[1, 50], '']}, arguments=['gw0'])
        assert ParallelismManager(im, {}, calc_type='qp').resolve() is None


class TestApplySetProfile:

    def test_applyset_and_profile(self):
        bundle = ApplySet({'PAR_def_mode': 'balanced'}, GPU_RESOURCES, ['X_CPU'])
        assert bundle.variables == {'PAR_def_mode': 'balanced'}
        profile = ParallelismProfile(id='p', gates={'bands': [1, 10]})
        assert profile.gates['bands'] == (1, 10)

    def test_profile_defaults(self):
        profile = ParallelismProfile()
        assert profile.id == ''
        assert profile.gates == {}
        assert profile.parallelism == {}
        assert profile.resources == {}


class TestYamboInputManagerTarget:

    def test_snapshot_reads_getters(self):
        im = make_manager({'BndsRnXp': [[1, 50], ''], 'GbndRnge': [[1, 40], ''],
                           'NGsBlkXp': [4.0, 'Ry']}, arguments=['dip', 'x', 'gw0'])
        target = YamboInputManagerTarget(im, 'qp')
        snapshot = target.snapshot()
        assert snapshot['bands'] == 50
        assert snapshot['gcut'] == 4.0
        assert snapshot['kpoints'] == 1
        assert 'gw0' in snapshot['runlevels']