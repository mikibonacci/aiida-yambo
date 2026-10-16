"""Contract tests for the plugin entry points and workflow specifications."""

import pytest

from aiida_yambo.calculations.yambo import YamboCalculation
from aiida_yambo.calculations.ypp import YppCalculation


def test_calcjob_specs_have_required_ports():
    for process, required in ((YamboCalculation, {'code', 'parameters', 'settings', 'parent_folder'}),
                              (YppCalculation, {'code', 'parameters', 'settings', 'parent_folder'})):
        assert required <= set(process.spec().inputs)


def test_workflow_specs_have_required_ports(aiida_profile):
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence
    from aiida_yambo.workflows.yamborestart import YamboRestart
    from aiida_yambo.workflows.yambowf import YamboWorkflow
    from aiida_yambo.workflows.ypprestart import YppRestart
    for process, required in ((YamboRestart, {'yambo', 'parent_folder', 'max_walltime'}),
                              (YppRestart, {'ypp', 'parent_folder'}),
                              (YamboWorkflow, {'qp', 'scf', 'nscf', 'bse'}),
                              (YamboConvergence, {'ywfl', 'parameters_space'})):
        assert required <= set(process.spec().inputs)


@pytest.mark.parametrize('process', [YamboCalculation, YppCalculation])
def test_calcjob_protocol_defaults_are_stable(process):
    options = process.spec().inputs['metadata']['options']
    assert options['input_filename'].default in ('aiida.in', 'ypp.in')
    assert options['output_filename'].default == 'aiida.out'


def test_parser_rejects_wrong_process_type():
    from aiida_yambo.parsers.parsers import YamboParser
    from aiida_yambo.parsers.yppparser import YppParser
    class FakeCalculation:
        process_type = 'aiida.calculations:other'

        def get_last_job_info(self):
            return {}

    from aiida.common.exceptions import OutputParsingError

    with pytest.raises(OutputParsingError):
        YamboParser(FakeCalculation())
    with pytest.raises(OutputParsingError):
        YppParser(FakeCalculation())


@pytest.mark.parametrize('process', [YamboCalculation, YppCalculation])
def test_calcjob_required_inputs_are_declared(process):
    ports = process.spec().inputs
    assert ports['code'].required is True
    assert ports['parameters'].required is True


def test_workflow_protocol_paths_exist():
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence
    from aiida_yambo.workflows.yamborestart import YamboRestart
    from aiida_yambo.workflows.yambowf import YamboWorkflow
    from aiida_yambo.workflows.ypprestart import YppRestart
    for workflow in (YamboWorkflow, YamboConvergence, YamboRestart, YppRestart):
        assert workflow.get_protocol_filepath().exists()
