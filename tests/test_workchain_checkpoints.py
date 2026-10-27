"""Safety net for the workchains: outline steps are called without a daemon, and the
``ctx`` must survive the checkpoint (YAML) round trip used by AiiDA to persist a running process."""
import pytest


def _round_trip(value):
    from aiida.orm.utils.serialize import deserialize_unsafe, serialize

    return deserialize_unsafe(serialize(value))


def test_yamboworkflow_setup_ctx_survives_checkpoint(generate_workchain, ywfl_qp_inputs):
    from aiida_yambo.workflows.yambowf import YamboWorkflow

    process = generate_workchain(YamboWorkflow, ywfl_qp_inputs)
    try:
        process.setup()
        assert process.ctx.all_calcs_to_do == ['scf', 'nscf', 'qp']
        restored = _round_trip(process.ctx)
        assert restored.all_calcs_to_do == ['scf', 'nscf', 'qp']
        assert restored.input_manager.qp.parameters == process.ctx.input_manager.qp.parameters
    finally:
        process.close()


@pytest.fixture
def yc_inputs(ywfl_qp_inputs):
    from aiida import orm
    from aiida_yambo.workflows.utils.parameters import ConvParam, create_parameter_space_inputs

    space = [ConvParam('FFTGvecs', start=10, stop=25, delta=5, max=40, units='Ry', conv_thr=0.5, steps=4)]
    return {
        'ywfl': ywfl_qp_inputs,
        'parameters_space': orm.List(create_parameter_space_inputs(space)),
        'workflow_settings': orm.Dict({'type': 'heavy', 'what': ['gap_']}),
        'group_label': orm.Str('test_checkpoint_group'),
    }


def test_yamboconvergence_setup_ctx_survives_checkpoint(generate_workchain, yc_inputs):
    from aiida_yambo.workflows.yamboconvergence import YamboConvergence

    process = generate_workchain(YamboConvergence, yc_inputs)
    try:
        process.setup()
        restored = _round_trip(process.ctx)
        assert restored.current_convergence.parameters.var == ['FFTGvecs']
        assert list(restored.df.columns) == list(process.ctx.df.columns)
        assert restored.input_manager.qp.parameters == process.ctx.input_manager.qp.parameters
    finally:
        process.close()
