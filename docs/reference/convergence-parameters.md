# Convergence parameters

The parameters of `YamboConvergence` are defined in `aiida_yambo.workflows.utils.parameters`. Each one is converged in
turn, in the order of the list `manager.parameters_space`. How the algorithm uses them is in
[Convergence](../concepts/convergence.md); the fit function is in
[Choose the fit function](../how-to/convergence-fit-functions.md).

| Class | Variables | Use |
|---|---|---|
| `ConvParam` | one (`'FFTGvecs'`, `'BndsRnXp'`...) | one-dimensional space |
| `CoupledConvParam` | two (`['BndsRnXp', 'NGsBlkXp']`) | two variables increased together, fitted in 2D |
| `KpointMeshConvParam` | `'kpoint_mesh'` or `'kpoint_density'` | k-points (see the two `05_convergence` examples) |
| `ScanParam` | any | fixed points, no convergence ([Run fixed points](../how-to/scan-points.md)) |

## Options

| Option | Default | Meaning |
|---|---|---|
| `start`, `stop`, `delta` | required | first and last value of the initial grid, and the step. Lists for `CoupledConvParam` (one per variable) and for `kpoint_mesh` (`[nx, ny, nz]`). For `kpoint_density` they are distances in 1/Å: `start >= stop` and `delta < 0` |
| `max` | required | the largest value the algorithm may explore (for `kpoint_density`, the largest distance); a grid beyond it is refused |
| `min` | none | only `kpoint_density`: the smallest distance, i.e. the finest and most expensive mesh allowed. The distance decreases with the refinement, so without `min` only positivity limits it; a grid beyond it stops the workflow (exit code 303). The protocol sets it from the `max` of its `kpoint_density` |
| `units` | required | units of the values, as in yambopy (`'Ry'`, `''`, `'mesh'`, `'1/A'`); a list for a coupled parameter |
| `conv_thr` | `10` | the convergence threshold, in the units of `conv_thr_units` |
| `conv_thr_units` | `'%'` | `'%'`: `conv_thr` is a percentage of the converged value (relative error); `'eV'`: an absolute difference in eV |
| `steps` | 4 (6 for a coupled one) | number of points of the grid built in each iteration |
| `max_iterations` | 4 (8 for a coupled one) | attempts to converge this parameter: if the threshold is not reached, a new grid is built, up to this number |
| `max_concurrent_steps` | 4 | how many calculations of the grid run at the same time |
| `mirror_values` | none | other variables which take the same value; for a coupled parameter a dictionary, `{'BndsRnXp': ['GbndRnge']}` |
| `dtype` | `'int'` | type of the generated values |
| `points` | 6 | (coupled only) points of the 2D grid: 4 (corners), 5 (corners and center), 6 |
| `fit_function`, `fit_options` | `'power_law'` | see [Choose the fit function](../how-to/convergence-fit-functions.md) |

!!! note "The meaning of `conv_thr`"
    With `conv_thr_units='%'` the same number works for any observable. With `'eV'` it is an energy, so it only makes
    sense for the energies (gaps, excitons); the examples use `0.1` eV or a few percent. A coupled parameter usually needs
    a looser threshold than a one-dimensional one.

## Bands of the NSCF

The NSCF must contain at least the bands that `BndsRnXp` and `GbndRnge` ask for. The workflow settings decide how many
it computes (`manager.workflow_settings = {..., 'bands_nscf_update': ..., 'max_nscf_bands': ...}`):

| `bands_nscf_update` | NSCF bands |
|---|---|
| `'all-at-once'` (default) | the largest last value (`stop`) of the grids of **all** the parameters: one NSCF for the whole convergence |
| `'full-step'` | the largest of the grid of the parameter which is converged now |
| an integer `N` | as `'all-at-once'`, but never more than `N`; if a grid needs more the workflow stops, before any calculation |

If a grid moves to larger values (a new grid when the threshold is not reached) and needs more bands than the NSCF
has, the NSCF is computed again. `max` of the parameter is the limit of this growth, not the number of bands of the first
NSCF. Any other value of `bands_nscf_update` is refused when the workflow is submitted.

Before the NSCF the workflow reports the bands it will use. If they are more than `max_nscf_bands` (default 2000) the
workflow stops with exit code 304 and says what to do: reduce the bands, use `'full-step'` or an integer cap, or confirm by
raising `max_nscf_bands`.

!!! warning "The default grid of the bands"
    With the protocol (no `parameters_space` of your own) the bands of the grid are `start`, `stop`, `delta` and `max` of
    `meta_parameters['bands']` of the protocol, multiplied by `log10(volume)` of the cell in Angstrom^3. It is a heuristic
    for real calculations (for silicon `stop` is about 1280 bands), too large for a test: give your own parameters, or
    `overrides={'meta_parameters': {'bands': {'start': 20, 'stop': 50, 'delta': 10, 'max': 60}}}` to
    `get_builder_from_protocol`.
