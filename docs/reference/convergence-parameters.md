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
