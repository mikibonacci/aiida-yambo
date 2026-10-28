# Restarts and troubleshooting

Every Yambo calculation of a workflow runs inside a `YamboRestart`, which restarts it after the usual failures.
This page says what it does by itself, what it does not, and what to check when a workflow stops. The statuses
are in [Options and exit codes](../reference/options-and-exit-codes.md).

## What the restart layer does

| What happened | What it does |
| --- | --- |
| Walltime exhausted | restarts with more time (up to `max_walltime`), reusing the databases already written |
| Parallelization error | restarts with the balanced parallelism (`PAR_def_mode='balanced'`): **your parallelism variables are replaced** |
| Memory error | restarts with more nodes (up to `max_number_of_nodes`, 0 = not increased), or fewer MPI tasks per node and more threads |
| `Variable NOT DEFINED` (corrupted fragment) | restarts removing the corrupted fragment |
| Unknown failure | once, copying the `SAVE` folder |
| Scheduler problem (invalid account, node failure...) | **not retried**: status 300 |

It stops after `max_iterations` restarts (401) or after two failures in a row that no handler could fix (402).
Set the limits on the manager before submitting: `manager.qp.max_walltime = 86400`,
`manager.qp.max_number_of_nodes = 4`, and see [Resources and parallelism](resources.md) to choose the resources.

## When a workflow stops

1. **Read the report.** `verdi process report <pk>` of the workflow, then of the failed child. Most failures are
   scheduler, memory or parent-folder problems, not physics.
2. **Find the exit status** (`verdi process show <pk>`) and look it up in the reference. The first thing to check by status:

| Status | Usually |
| --- | --- |
| `YamboWorkflow` 300 | a step failed: look at the failed child |
| `YamboWorkflow` 301 | a splitted QP run failed ([QP splitter](qp-splitter.md#when-something-goes-wrong)) |
| `YamboWorkflow` 302 | the QP computed lack valence or conduction states: enlarge the selection |
| `YamboWorkflow` 303 | the DFT bands predict a metal (`stop_if_metal`) |
| `YamboRestart` 301 | the NSCF has too few bands for the Yambo inputs (the status is ambiguous with 301 `ERROR_SUB_PROCESS_EXCEPTED`: read the report) |
| `YamboRestart` 302 | the parent folder is empty on the remote machine (cleaned, or the machine was wiped) |
| `YamboConvergence` 302 | calculations failed; the story (`collection`) says which |
| `YamboConvergence` 400 | `max_iterations` reached before the convergence |

3. **Look at the calculation itself**: `verdi calcjob inputcat <pk>` (the input and the submission script) and the
   Yambo report in the retrieved folder ([Recipes](recipes.md) says how to find the files).
4. **Do not edit the generated files.** Fix the cause, then relaunch from the last successful node: give it as the
   parent folder of a new workflow and only the missing steps run.

## Typical causes

- **NSCF with too few bands**: the workflow redoes the NSCF automatically when the Yambo bands exceed it; if the
  restart ends with `LOW_NUMBER_OF_NSCF_BANDS`, raise `manager.set_nscf_nbnd(...)`.
- **Out of memory in the screening**: use more nodes with fewer MPI tasks per node and more threads, or lower
  `BndsRnXp`/`NGsBlkXp`.
- **Walltime too short for every restart**: raise `max_walltime`; the walltime grows by about 1.5 times the
  iteration number, up to the limit.
- **Convergence: one point failed**: in a scan the others continue and the failed one is flagged; in a convergence
  a failed point makes the workflow stop with 302. Fix the resources for that point with
  [`resource_rules`](resources.md#in-a-convergence-resource_rules) and run again: the calculations already done in
  the same group are reused.
