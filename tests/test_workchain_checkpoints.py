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
