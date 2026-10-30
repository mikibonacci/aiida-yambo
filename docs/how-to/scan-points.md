# Run fixed points: the scan (no convergence)

`YamboConvergence` has two ways of choosing the calculations, and they are exclusive:

| | **Convergence** (`convergence_parameters`, `parameters_space`) | **Scan** (`scan_points`, `ScanParam`) |
| --- | --- | --- |
| What it does | runs a few points, fits a function to the results, finds the converged value, and moves or extends the grid until it is converged | runs exactly the points you give, once |
| Fit function | **yes**: [choose it here](convergence-fit-functions.md) | **no**: `fit_functions` is ignored, and `conv_thr`, `max` and the other convergence options do not apply (`max_concurrent_steps` does) |
| When it ends | converged, or `max_iterations` reached | all the points are done |

Use a scan when you just need several calculations: to run tests, or to collect data for your own post-processing.

## Run a list of parameter sets

Give `scan_points` instead of `convergence_parameters`: one dictionary `{variable: value}` for each calculation.

```python
from aiida_yambo.workflows.yamboconvergence import YamboConvergence

builder = YamboConvergence.get_builder_from_protocol(
    pw_code=pw_code, preprocessing_code=p2y_code, code=yambo_code,
    structure=structure,
    scan_points=[
        {'FFTGvecs': 10, 'BndsRnXp': 100, 'NGsBlkXp': 2},
        {'FFTGvecs': 15, 'BndsRnXp': 200, 'NGsBlkXp': 4},
        {'FFTGvecs': 20, 'BndsRnXp': 300, 'NGsBlkXp': 6},
    ],
)
```

- Every set must define the same variables. Units default to the usual ones (`Ry` for the cutoffs, none for the
  bands, `1/A` for `kpoint_density`) and `GbndRnge` follows `BndsRnXp`; to change them, build the
  `ScanParam` yourself (`ScanParam.from_dicts(points, units=..., mirror_values=...)`) and pass it in `parameters_space`.
- The calculations are submitted `max_concurrent_steps` at a time (default 4). Calculations already done in the
  same group are reused.
- **Nothing is evaluated.** When all the points are done the workflow ends. A calculation which fails does *not*
  stop the others: it is flagged `failed` in the story. The workflow fails (`CALCS_FAILED`) only if *all* of
  them failed.
- A `ScanParam` must be the only entry of `parameters_space`.

The results are in the `collection` output (`story`: one row per calculation, with the values of the variables,
the quantities in `workflow_settings['what']`, and the `failed` flag) and in `convergence_summary['scan']`
(number of points and of failures).

```python
story = node.outputs.collection.get_dict()['story']
```
