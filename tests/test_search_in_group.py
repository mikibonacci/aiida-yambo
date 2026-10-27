"""Reuse of already computed calculations: they must be equivalent, not just similar."""
import io
from uuid import uuid4

import pytest

WORKFLOW = 'aiida.workflows:yambo.yambo.yambowf'


@pytest.fixture
def make(aiida_profile):
    """Factory of the nodes we need: stored workflows with their inputs, and the matching input manager."""
    from aiida import orm
    from aiida.common.links import LinkType
    from aiida.engine import ProcessState
    from aiida_pseudo.data.pseudo import UpfData
    from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager

    class Factory:
        def structure(self, a=5.43):
            structure = orm.StructureData(cell=[[a / 2, a / 2, 0], [a / 2, 0, a / 2], [0, a / 2, a / 2]])
            structure.append_atom(position=(0, 0, 0), symbols='Si', name='Si')
            structure.append_atom(position=(a / 4, a / 4, a / 4), symbols='Si', name='Si')
            return structure.store()

        def pseudo(self, z_valence=4.0):
            content = f'<UPF version="2.0.1"><PP_HEADER\nelement="Si"\nz_valence="{z_valence}"\n/></UPF>\n'
            return UpfData(io.BytesIO(content.encode()), filename='Si.upf').store()

        def kpoints(self, mesh=(2, 2, 2)):
            kpoints = orm.KpointsData()
            kpoints.set_kpoints_mesh(list(mesh))
            return kpoints.store()

        def setup(self, structure=None, pseudo=None, kpoints=None, ecutwfc=30.0, nbnd=20):
            return dict(structure=structure or self.structure(), pseudo=pseudo or self.pseudo(),
                        kpoints=kpoints or self.kpoints(), parameters={'CONTROL': {'calculation': 'nscf'},
                        'SYSTEM': {'ecutwfc': ecutwfc, 'ecutrho': 4 * ecutwfc, 'nbnd': nbnd}})

        def variables(self, **changes):
            variables = {'NGsBlkXp': [2, 'Ry'], 'BndsRnXp': [[1, 20], ''], 'FFTGvecs': [15, 'Ry'],
                         'X_and_IO_CPU': '1 1 1', 'QPkrange': [[[1, 1, 4, 5]], '']}
            variables.update(changes)
            return {k: v for k, v in variables.items() if v is not None}

        def workflow(self, setup, variables=None, additional_parsing=None, finished_ok=True):
            """A finished YamboWorkflow with the inputs of an NSCF + QP calculation."""
            node = orm.WorkChainNode()
            node.set_process_type(WORKFLOW)
            inputs = {
                'nscf__pw__structure': setup['structure'],
                'nscf__pw__pseudos__Si': setup['pseudo'],
                'nscf__pw__parameters': orm.Dict(setup['parameters']).store(),
                'nscf__kpoints': setup['kpoints'],
                'qp__yambo__parameters': orm.Dict({'arguments': ['gw0'], 'variables': variables or self.variables()}).store(),
            }
            if additional_parsing is not None:
                inputs['additional_parsing'] = orm.List(additional_parsing).store()
            for label, data in inputs.items():
                node.base.links.add_incoming(data, link_type=LinkType.INPUT_WORK, link_label=label)
            node.store()
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(0 if finished_ok else 300)
            return node

        def manager(self, setup, variables=None, additional_parsing=()):
            return YamboInputManager(
                nscf={'parameters': setup['parameters'], 'structure': setup['structure'],
                      'pseudos': {'Si': setup['pseudo']}, 'kpoints': setup['kpoints']},
                qp={'parameters': {'arguments': ['gw0'], 'variables': variables or self.variables()}},
                additional_parsing=list(additional_parsing),
            )

        def search(self, manager, *nodes, **kwargs):
            from aiida_yambo.utils.common_helpers import search_in_group

            group = orm.Group(label=f'search-{uuid4().hex}').store()
            group.add_nodes(list(nodes))
            return search_in_group(manager, group, **kwargs)

    return Factory()


def test_equivalent_calculation_is_found(make):
    setup = make.setup()
    old = make.workflow(setup)

    already_done, _, _ = make.search(make.manager(setup), old)

    assert already_done == old.pk


@pytest.mark.parametrize('what, change', [
    ('structure', lambda f: {'structure': f.structure(a=5.5)}),
    ('pseudo', lambda f: {'pseudo': f.pseudo(z_valence=14.0)}),
    ('ecutwfc', lambda f: {'ecutwfc': 40.0}),
    ('k-points', lambda f: {'kpoints': f.kpoints((4, 4, 4))}),
])
def test_different_dft_setup_is_not_reused(make, what, change):
    setup = make.setup()
    old = make.workflow(make.setup(**change(make)))

    already_done, _, _ = make.search(make.manager(setup), old)

    assert already_done is False


def test_the_number_of_nscf_bands_does_not_make_a_calculation_different(make):
    setup = make.setup()
    old = make.workflow(make.setup(nbnd=40))

    assert make.search(make.manager(setup), old)[0] == old.pk


@pytest.mark.parametrize('old_changes', [
    {'FFTGvecs': [20, 'Ry']},  # another value
    {'FFTGvecs': [15, 'RL']},  # same value, another unit
    {'EXXRLvcs': [10, 'Ry']},  # a variable more
    {'FFTGvecs': None},  # a variable less
])
def test_different_yambo_variables_are_not_reused(make, old_changes):
    setup = make.setup()
    old = make.workflow(setup, variables=make.variables(**old_changes))

    assert make.search(make.manager(setup), old)[0] is False


def test_parallelism_does_not_make_a_calculation_different(make):
    setup = make.setup()
    old = make.workflow(setup, variables=make.variables(X_and_IO_CPU='2 2 1', DIP_ROLEs='k c v'))

    assert make.search(make.manager(setup), old)[0] == old.pk


def test_unsuccessful_calculation_is_not_reused(make):
    setup = make.setup()
    old = make.workflow(setup, finished_ok=False)

    assert make.search(make.manager(setup), old)[0] is False


@pytest.mark.parametrize('parsed_before, wanted, found', [
    (['gap_'], ['gap_'], True),
    (['gap_', 'homo'], ['gap_'], True),
    (['homo'], ['gap_'], False),
    (None, ['gap_'], False),
    (None, [], True),
])
def test_the_wanted_quantities_must_have_been_parsed(make, parsed_before, wanted, found):
    setup = make.setup()
    old = make.workflow(setup, additional_parsing=parsed_before)

    already_done, _, _ = make.search(make.manager(setup, additional_parsing=wanted), old)

    assert already_done == (old.pk if found else False)


def test_a_manager_without_dft_inputs_compares_nothing_about_them(make):
    from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager

    setup = make.setup()
    old = make.workflow(setup)
    manager = YamboInputManager(nscf={'kpoints': setup['kpoints']},
                                qp={'parameters': {'arguments': ['gw0'], 'variables': make.variables()}})

    assert make.search(manager, old)[0] == old.pk


@pytest.mark.parametrize('change, same', [({}, True), ({'ecutwfc': 40.0}, False), ({'structure': 'other'}, False)])
def test_pw_calculation_setup(make, change, same):
    from aiida import orm
    from aiida.common.links import LinkType
    from aiida_yambo.utils.common_helpers import same_pw_setup

    setup = make.setup()
    other = make.setup(structure=make.structure(a=5.6) if change.get('structure') else None,
                       ecutwfc=change.get('ecutwfc', 30.0))
    calc = orm.CalcJobNode()
    for label, data in {'structure': other['structure'], 'pseudos__Si': other['pseudo'],
                        'parameters': orm.Dict(other['parameters']).store()}.items():
        calc.base.links.add_incoming(data, link_type=LinkType.INPUT_CALC, link_label=label)
    calc.store()

    assert same_pw_setup(make.manager(setup).nscf, calc) is same
