# Options and exit codes

Reference of the less visible inputs of the workflows and of what their exit codes mean. The exit codes are the
ones declared in the process specs (`verdi process show <pk>` prints the status and message; the label is in
`Process.exit_codes`).

## `YamboWorkflow` inputs

| Input | Meaning |
| --- | --- |
| `parent_folder` | `RemoteData` of an SCF, NSCF or YAMBO calculation. The workflow runs only the missing steps; an empty parent folder is ignored (it starts from scratch). |
| `additional_parsing` | `List` of quantities to parse into `output_ywfl_parameters`: `gap_`, `gap_GG`, `homo`, `lumo`, `lowest_exciton`, `brightest_exciton`, or `[k1, k2, b1, b2]` levels. |
| `QP_subsets_dict` | How to split a QP calculation in several runs (see below); needed for `calc_type='bse@qp'`. |
| `already_computed_QP_db` | `List` of pks of `ndb.QP` databases already computed: those quasiparticles are skipped when the subsets are built. |
| `clean_failed` | `Bool`: clean the remote folders of the failed calculations at the end. |
| `stop_if_metal` | `bool`: stop after the NSCF if the DFT bands predict a metal (exit 303). It is a non-stored input: it is not kept in the provenance and `YamboConvergence` does not forward it to the workflows it submits. |

The calculations to be done follow the inputs: `scf`, `nscf` (Quantum ESPRESSO), `qp` (GW) and `bse`
namespaces; a namespace which is not provided is not run. `get_builder_from_protocol(calc_type=...)` fills them:
`'qp'`, `'bse'` (scissor/`KfnQP_E`) or `'bse@qp'` (QP computed explicitly, then BSE). Useful protocol arguments:
`protocol` / `protocol_qe`, `NLCC`, `RIM_v`, `RIM_W` (Coulomb cutoff/RIM options), `electronic_type`, `spin_type`,
`parent_folder`, `pseudo_family`, and `overrides` (`{'scf': {...}, 'nscf': {...}, 'qp': {...}, 'bse': {...}}`;
the old `yres` key still applies to both `qp` and `bse` but is deprecated).

### `QP_subsets_dict`

The QP splitter runs `parallel_runs` YamboRestart at a time, each on `qp_per_subset` quasiparticles, then merges
the `ndb.QP` databases (`yambopy mergeqp`, which must be available where the workflow runs).

| Key | Meaning |
| --- | --- |
| `parallel_runs`, `qp_per_subset` | Required. How many runs at once, how many (k, band) pairs in each. |
| `boundaries` | `{'bi': .., 'bf': .., 'ki': .., 'kf': ..}`: all k-points (or `ki`..`kf`) and bands `bi`..`bf`. |
| `range_QP` | Energy window (eV) around mid-gap: the bands inside are selected automatically; with `full_bands` also the bands in between; `range_spectrum` sets the scissor window. |
| `explicit` | List of `[k1, k2, b1, b2]` ranges. |
| `consider_only` | Keep only these bands. |
| `parallelism` | `'auto'` (the plugin distributes the MPI tasks on the QP of each run) or a dictionary of yambo variables, for the splitted runs. |
| `resources`, `max_wallclock_seconds`, `queue_name`, `prepend_text`, `custom_scheduler_commands`, `clean_workdir` | Resources (the values not given are kept), walltime, queue, scheduler text and cleaning of the splitted runs (after a successful merge; only those folders). |
| `extend_db` | Also build an extended QP database (scissor plus Fermi-Dirac smearing); extra keys `Nb`, `T_smearing`, `consider_only`. |

Give exactly one of `boundaries`, `range_QP`, `explicit` (more than one, or an unknown key, raises an error; with none, all the bands of all the k-points are used). All the modes are explained in [the QP splitter how-to](../how-to/qp-splitter.md). After the merge, outputs `merged_QP` and (if requested)
`extended_QP` are returned, together with `splitted_QP_calculations`.

## `YamboConvergence` inputs

| Input | Meaning |
| --- | --- |
| `ywfl` | The inputs of the `YamboWorkflow` run for every point. |
| `parameters_space` | `List` of parameters (`ConvParam`, `CoupledConvParam`, `KpointMeshConvParam`), converged in order, or a single `ScanParam`. Validated when the process is submitted. |
| `workflow_settings` | `type`: `'heavy'` (the converged values are used by the following parameters) or `'cheap'` (every parameter starts from the initial inputs); `what`: quantities to converge; `bands_nscf_update`: `'all-at-once'`/`'full-step'` (compute the NSCF with the largest number of bands needed), or an integer N > 0 to cap the number of bands at N; `skip_pre`: skip the p2y pre-calculation (debug). |
| `resource_rules` | `List` of rules `{'when': {...}, 'for': [steps], 'use': {...}}` giving the resources (nodes, MPI, threads, walltime, queue, parallelism) for given values of the parameters. Validated at submission; see [Resources](../how-to/resources.md). |
| `parallelism_instructions` | Deprecated, use `resource_rules`. The old way of changing the parallelism with the parameter values. |
| `group_label` | Group where all the calculations are kept and looked for to avoid repeating them (default: `convergence_tests_<formula>`). A calculation is reused only if it has the same structure, pseudopotentials, parameters and yambo variables. |

Outputs: `collection` (always, also when the run stops early), `convergence_summary`, and the outputs of the last
calculation.

## Restart layer (`YamboRestart`) inputs

`max_iterations`, `max_walltime` and `max_number_of_nodes` (0: the number of nodes is not increased) limit the
automatic restarts; `code_version` selects the naming of the parallelism variables; `handler_overrides` and
`clean_workdir` are the usual AiiDA options.

## Exit codes

### `YamboWorkflow`

| Status | Label | What it means / what to do |
| --- | --- | --- |
| 300 | `ERROR_WORKCHAIN_FAILED` | A step (SCF, NSCF, YAMBO, BSE) failed, or its mapping could not be derived: `verdi process report`. |
| 301 | `ERROR_SPLITTED_QP_FAILED` | A splitted QP run failed: look in `splitted_QP_calculations`. |
| 302 | `ERROR_MISSING_BAND_REGION` | The merged QP database lacks valence or conduction states (needed for BSE and `extend_db`): enlarge the QP selection. |
| 303 | `ERROR_METALLIC_SYSTEM` | `stop_if_metal` and the NSCF predicts a metal. |

### `YamboConvergence`

| Status | Label | What it means / what to do |
| --- | --- | --- |
| 300 | `UNDEFINED_STATE` | No results to analyse for the current step. |
| 301 | `PRECALC_FAILED` | The p2y pre-calculation failed. |
| 302 | `CALCS_FAILED` | A calculation failed (in a scan: *all* of them failed; a failed point of a scan is only flagged in `collection`). |
| 303 | `SPACE_TOO_SMALL` | The grid has non-positive or out-of-bound values: change `start`/`stop`/`max`. |
| 400 | `CONVERGENCE_NOT_REACHED` | `max_iterations` reached: raise it, loosen `conv_thr`, or widen the grid. |

In all these cases `collection` and `convergence_summary` are returned.

### `YamboRestart`

| Status | Label | What it means |
| --- | --- | --- |
| 300 | `ERROR_UNRECOVERABLE_FAILURE` | The calculation failed with a status below 400 (scheduler problems such as an invalid account or a node failure): not retried. |
| 301 | `LOW_NUMBER_OF_NSCF_BANDS` | The NSCF has fewer bands than the YAMBO inputs need. |
| 302 | `EMPTY_PARENT` | The parent folder is empty on the remote computer. |
| 303 | `NO_PARENT` | No parent folder. |
| 401 / 402 | `ERROR_MAXIMUM_ITERATIONS_EXCEEDED`, `ERROR_SECOND_CONSECUTIVE_UNHANDLED_FAILURE` | Too many restarts, or two failures in a row that no handler could fix. |

The statuses of the YAMBO calculation and what the restart layer does with them:

| Status | Label | Handled by |
| --- | --- | --- |
| 501 | `WALLTIME_ERROR` | restart with more time (up to `max_walltime`), reusing the databases written |
| 502 | `NO_SUCCESS` | one restart copying the `SAVE` |
| 504 | `PARA_ERROR` | restart with the default (balanced) parallelism |
| 505, 506 | `MEMORY_ERROR`, `X_par_MEMORY_ERROR` | restart with more nodes (up to `max_number_of_nodes`) or less MPI per node, rebalanced |
| 508 | `Variable_NOT_DEFINED` | restart removing the corrupted fragment |
| 500, 503, 507 | `ERROR_NO_RETRIEVED_FOLDER`, `PARSER_ANOMALY`, `NaN_AS_OUTPUT` | not handled: the run stops after two consecutive failures |
| 110-150 | scheduler errors | not retried (300) |

Known issue: in `YamboRestart` the statuses 301 and 302 are each declared twice
(`ERROR_SUB_PROCESS_EXCEPTED`/`LOW_NUMBER_OF_NSCF_BANDS`, `ERROR_SUB_PROCESS_KILLED`/`EMPTY_PARENT`), so the
status alone is ambiguous: check the process report.

### `YppRestart`

The same restart statuses (300, 301, 302, 401, 402) plus 304 (`WANNIER90_PP_PARENT_NOT_PRESENT`, no nnkp file) and
305 (`PW2WANNIER90_PARENT_NOT_PRESENT`, no mmn/amn folder). Its calculation (`YppCalculation`) can also return
504 `NNKP_NOT_PRESENT`, 505 `QP_LIST_NOT_PRESENT`, 506 `MERGE_NOT_COMPLETE` and 507 `PARENT_NOT_VALID` (the parent has no
`QP_DB`).
