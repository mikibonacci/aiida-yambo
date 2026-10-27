"""The example scripts must keep working: they are what users copy.

The scripts end by submitting a builder. They are run for real here, with the codes and the parent replaced by test nodes and
`submit` replaced by a function which checks that the inputs are valid for the process (the same validation
AiiDA does at submission), without running anything.
"""
import ast
import importlib
import pathlib
import runpy
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / 'examples'
SCRIPTS = sorted(p.relative_to(EXAMPLES).as_posix() for p in EXAMPLES.rglob('*.py'))

# These examples were written for the `yres` namespace of the old YamboWorkflow (now `qp` and `bse`)
# and do not build valid inputs any more. They have to be rewritten; remove them from this list when done.
LEGACY = {
    'examples_hBN/workflows/yambo_convergence.py',
    'examples_hBN/workflows/yambo_convergence_BSE.py',
    'examples_hBN/workflows/yambo_convergence_BSE_QP.py',
    'examples_hBN/workflows/yambo_workflow_BSE.py',
    'examples_hBN/workflows/yambo_workflow_QP.py',
    'examples_hBN/workflows/yambo_workflow_QP_BSE.py',
}

# command line of the scripts we can run with test nodes: codes by label and the pk of the parent
RUNNABLE = {
    'examples_hBN/yambo/yambo_gw.py': ['--yambocode', '{yambo}', '--yamboprecode', '{p2y}', '--parent', '{parent}'],
    'examples_hBN/yambo/yambo_ip_rpa.py': ['--yambocode', '{yambo}', '--yamboprecode', '{p2y}', '--parent', '{parent}'],
    'examples_hBN/yambo/yambo_p2y.py': ['--yambocode', '{yambo}', '--yamboprecode', '{p2y}', '--parent', '{parent}'],
    'examples_hBN/workflows/yambo_restart.py': ['--yambocode', '{yambo}', '--yamboprecode', '{p2y}', '--parent', '{parent}'],
    'plugin/ypp_wannier.py': ['--yppcode', '{ypp}', '--parent', '{parent}'],
}


@pytest.mark.parametrize('script', SCRIPTS)
def test_example_is_valid_python(script):
    ast.parse((EXAMPLES / script).read_text())


@pytest.mark.parametrize('script', SCRIPTS)
def test_example_imports_exist(aiida_profile, script):
    """Every name the example imports from aiida_yambo exists."""
    for node in ast.walk(ast.parse((EXAMPLES / script).read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('aiida_yambo'):
            module = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(module, alias.name), f'{node.module}.{alias.name}'


@pytest.mark.parametrize('script', [
    pytest.param(s, marks=pytest.mark.xfail(reason='uses the removed `yres` namespace: to be rewritten', strict=True))
    if s in LEGACY else s
    for s in SCRIPTS
])
def test_example_does_not_use_the_removed_yres_namespace(script):
    assert '.yres' not in (EXAMPLES / script).read_text()


@pytest.fixture
def environment(aiida_profile, aiida_localhost, monkeypatch):
    """Codes and a parent calculation with a remote folder, and a `submit` which validates the inputs."""
    from aiida import orm
    from aiida.common.links import LinkType
    from aiida.engine.utils import instantiate_process
    from aiida.manage import get_manager

    def code(name):
        label = f'{name}-{uuid4().hex[:8]}'
        orm.InstalledCode(label=label, computer=aiida_localhost, filepath_executable='/bin/true').store()
        return f'{label}@{aiida_localhost.label}'

    calculation = orm.CalcJobNode(computer=aiida_localhost).store()
    remote = orm.RemoteData(computer=aiida_localhost, remote_path='/tmp/parent')
    remote.base.links.add_incoming(calculation, link_type=LinkType.CREATE, link_label='remote_folder')
    remote.store()

    submitted = []

    def submit(builder, **kwargs):
        process = instantiate_process(get_manager().get_runner(), builder.process_class, **builder._inputs(prune=True))
        process.close()
        submitted.append(builder.process_class.__name__)
        return SimpleNamespace(pk=1)

    monkeypatch.setattr('aiida.engine.submit', submit)
    return SimpleNamespace(yambo=code('yambo'), p2y=code('p2y'), ypp=code('ypp'), parent=calculation.pk, submitted=submitted)


@pytest.mark.parametrize('script', sorted(RUNNABLE))
def test_example_builds_valid_inputs(environment, monkeypatch, script):
    arguments = [a.format(**vars(environment)) for a in RUNNABLE[script]]
    monkeypatch.setattr(sys, 'argv', [script, *arguments])

    runpy.run_path(str(EXAMPLES / script), run_name='__main__')

    assert len(environment.submitted) == 1
