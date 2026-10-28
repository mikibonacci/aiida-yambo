"""The example scripts must keep working: they are what users copy.

Here: every script is valid Python, imports names which exist, and is exercised by a test. The scripts which submit are
run with test nodes in ``test_examples_workflows.py``, which also has the tests of the analysis and offline scripts.
"""
import ast
import importlib
import pathlib

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / 'examples'
SCRIPTS = sorted(p.relative_to(EXAMPLES).as_posix() for p in EXAMPLES.rglob('*.py'))

# scripts which do not submit: they have their own tests (functions on fabricated nodes, a fit on data)
OWN_TESTS = {
    '07_analysis/01_read_workflow_results.py',
    '07_analysis/02_read_convergence_results.py',
    '05_convergence/06_fit_models_offline.py',
}


def test_there_are_examples():
    assert len(SCRIPTS) >= 20


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


@pytest.mark.parametrize('script', SCRIPTS)
def test_example_does_not_use_the_removed_yres_namespace(script):
    assert '.yres' not in (EXAMPLES / script).read_text()


def test_every_example_is_run_by_a_test():
    from test_examples_workflows import RUNS

    run = {script for script, _ in RUNS} | OWN_TESTS
    assert set(SCRIPTS) <= run, sorted(set(SCRIPTS) - run)
