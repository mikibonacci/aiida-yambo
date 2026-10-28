# Workflow reference

| Workflow | Use it for |
| --- | --- |
| `YamboWorkflow` | DFT starting point plus GW/BSE/combined flows; recommended entry point |
| `YamboConvergence` | Adaptive convergence studies around `YamboWorkflow` |
| `YamboRestart` | Restart layer with automatic error handling and retries (used inside the others) |
| `YamboCalculation` | Low-level CalcJob for one exact YAMBO calculation |
| `YppRestart` | Restartable YPP post-processing |

Entry points are:

```text
yambo.yambo
yambo.yambo.yamborestart
yambo.yambo.yambowf
yambo.yambo.yamboconvergence
yambo.ypp.ypprestart
```

Use `YamboInputManager` together with `get_builder_from_protocol` whenever possible. It keeps protocol defaults, nested namespaces, and compatibility logic in the plugin rather than duplicating AiiDA data-node construction in scripts. See the [high-level input API tutorial](../tutorials/high-level-api.md).

## `YamboWorkflow`

`from aiida_yambo.workflows.yambowf import YamboWorkflow`. How it works: [Workflow flow](../concepts/workflow-flow.md).

| Input | Meaning |
| --- | --- |
| `scf`, `nscf` | `PwBaseWorkChain` inputs of the DFT steps |
| `qp`, `bse` | `YamboRestart` inputs of the GW and BSE steps (the first one also runs p2y) |
| `parent_folder` | an SCF, NSCF or Yambo `RemoteData` to start from |
| `additional_parsing` | quantities to parse into `output_ywfl_parameters` |
| `QP_subsets_dict` | the QP splitter ([how-to](../how-to/qp-splitter.md)) |
| `already_computed_QP_db`, `clean_failed`, `stop_if_metal` | see [Options and exit codes](options-and-exit-codes.md) |

Outputs: `output_ywfl_parameters`, `nscf_mapping`, `bands_quasiparticle`, `array_chi`, `array_ndb`, `merged_QP`,
`extended_QP`, `splitted_QP_calculations`, `scissor`, and the other outputs of the last Yambo calculation
(`remote_folder`, `retrieved`, `output_parameters`...). See [Outputs](outputs.md).

## `YamboConvergence`

`from aiida_yambo.workflows.yamboconvergence import YamboConvergence`. How it works: [Convergence](../concepts/convergence.md).

| Input | Meaning |
| --- | --- |
| `ywfl` | the `YamboWorkflow` inputs run at every point |
| `parameters_space` | the parameters to converge, in order, or a single scan |
| `workflow_settings` | `type` (`heavy`/`cheap`), `what` (quantities), `bands_nscf_update`, `skip_pre` |
| `resource_rules` | resources for given values of the parameters ([how-to](../how-to/resources.md#in-a-convergence-resource_rules)) |
| `group_label` | group of the calculations, reused if the inputs are the same |
| `parallelism_instructions` | deprecated, use `resource_rules` |

Outputs: `collection` (the story), `convergence_summary`, and the outputs of the last calculation.

## `YamboRestart`

`from aiida_yambo.workflows.yamborestart import YamboRestart`. What it handles: [Restarts](../how-to/restarts-and-troubleshooting.md).

Inputs: `yambo` (the `YamboCalculation` inputs), `parent_folder`, `max_iterations`, `max_walltime`,
`max_number_of_nodes`, `code_version`, `clean_workdir`, `handler_overrides`. Outputs: those of the Yambo calculation.
Example: `examples/06_low_level/04_yambo_restart.py`.

## `YamboCalculation`

`from aiida_yambo.calculations.yambo import YamboCalculation`: one Yambo run (with p2y first when needed), without
restarts. Inputs: `code`, `preprocessing_code`, `parameters` (arguments and variables), `settings`
(`INITIALISE`, `COPY_DBS`, `COPY_SAVE`), `parent_folder`, `precode_parameters`, and the `metadata.options` of the job.
Examples: `examples/06_low_level/`.
