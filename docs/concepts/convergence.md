# How the convergence works

`YamboConvergence` runs `YamboWorkflow` many times with changing parameters, fits the results and picks the
converged point.

- **Parameters.** `parameters_space` is a list, converged in order: a single variable (`ConvParam`), two coupled
  variables increased together (`CoupledConvParam`, e.g. bands and G-vectors), the k-points
  (`KpointMeshConvParam`), or a fixed list of points (`ScanParam`, no convergence and no fit: [scan](../how-to/scan-points.md)).
- **Algorithm.** For each parameter, a few points are run at the same time (`max_concurrent_steps`), a function is
  fitted to the results (a power law by default; [choose the fit function](../how-to/convergence-fit-functions.md))
  and the value where the fit is flat within `conv_thr` is chosen, otherwise the grid is moved and extended, up to
  `max_iterations` and `max`.
- **`heavy` and `cheap`** (`workflow_settings['type']`). With `heavy` the converged value of a parameter is used by
  the next ones, so the final calculation already has them all. With `cheap` every parameter starts from the
  initial inputs, which costs less.
- **Reuse.** The calculations are kept in a group (`group_label`) and are reused if the structure,
  pseudopotentials, parameters and yambo variables are the same.
- **Resources.** The cost grows with the parameters: use `resource_rules` to change nodes, walltime, queue
  and parallelism with their values ([Resources](../how-to/resources.md#in-a-convergence-resource_rules)).
- **Results.** `convergence_summary` (converged values) and `collection` (`story`: one row for each calculation,
  also when the run stops early). See [Outputs](../reference/outputs.md).
