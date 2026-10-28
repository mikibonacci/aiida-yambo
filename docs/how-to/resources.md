# Resources and parallelism

Where the resources of a run are set, and how to change them for the QP splitter and for given values of the
parameters of a convergence. For the cluster options you need on top (account, queue, modules), see the
scheduler block of the scripts in `examples/`.

## In a `YamboWorkflow`

Every step has its own options in the `YamboInputManager`: `scf`, `nscf` (Quantum ESPRESSO), `qp` and `bse`
(Yambo). The first yambo step also runs `p2y`, with its options.

```python
manager.scf.set_resources(nodes=1, mpi=16)
manager.scf.set_npools(4)
manager.nscf.set_resources(nodes=1, mpi=16)
manager.qp.set_resources(nodes=2, mpi=8, threads=2, time=14400)   # time in seconds
manager.bse.set_resources(nodes=1, mpi=32)
```

`threads` (`num_cores_per_mpiproc`) goes with `OMP_NUM_THREADS`. The queue and the account are in
`manager.qp.metadata.options` (`queue_name`, `account`, `custom_scheduler_commands`, `prepend_text`).

YAMBO distributes the work over k-points, bands, q-points and G-vectors. Start with
`'PAR_def_mode': 'balanced'` and set the roles by hand only when a production run shows the need:

```python
manager.qp.set_variable('PAR_def_mode', 'balanced')
manager.qp.set_variable('X_and_IO_CPU', '1 1 1 1 1')
manager.qp.set_variable('X_and_IO_ROLEs', 'q k g c v')
```

### Automatic restarts

`YamboRestart` restarts a run that failed for walltime, memory, parallelization or a corrupted database. It
increases the walltime (up to `max_walltime`), and for memory and parallelization errors it changes the MPI/OpenMP
balance and the parallelism (up to `max_number_of_nodes` nodes; 0 means the number of nodes is not increased), so a
parallelism you set can be replaced at a restart. Set the limits on the manager (`manager.qp.max_walltime = 86400`,
`manager.qp.max_number_of_nodes = 4`) before you generate the inputs.

## In the QP splitter

The splitted runs have their own resources, walltime, queue and parallelism: see
[the QP splitter](qp-splitter.md#choose-the-resources-of-the-splitted-runs).

## In a convergence: `resource_rules`

In a convergence the cost of a calculation changes with the values of the parameters: you want few cores for the
small cutoffs and more nodes for the large number of bands. The input `resource_rules` of
`YamboConvergence` is a list of rules, each saying **when** (the values of the parameters), **for** which
steps, and **use** which resources.

```python
manager.resource_rules = [
    # a large number of bands and many k-points: the quasiparticle step needs more nodes and time
    {'when': {'BndsRnXp': {'min': 200}, 'kpoints': {'min': 27}},
     'for': ['qp'],
     'use': {'num_machines': 4, 'num_mpiprocs_per_machine': 32, 'max_wallclock_seconds': 14400,
             'queue_name': 'long', 'parallelism': 'auto'}},
    # a specific mesh: the DFT steps on more nodes
    {'when': {'kpoint_mesh': [6, 6, 6]},
     'for': ['scf', 'nscf'],
     'use': {'num_machines': 2}},
    # no `when`: always matches. The default for everything else.
    {'use': {'num_machines': 1, 'num_mpiprocs_per_machine': 16}},
]
```

- **when** (optional; without it the rule always matches): all the conditions must hold. The names are the
  Yambo variables (`BndsRnXp`, `NGsBlkXp`, `FFTGvecs`...), `bands` (the largest number of bands), `gcut` (the
  largest G-vectors cutoff), `kpoint_mesh`, `kpoint_density` and `kpoints` (the number of k-points). A condition
  is a value (`2`, or a list `[4, 4, 4]`) which must be equal, or a dictionary with `min` and/or `max` (both
  included). The values are the ones of the point being prepared, including the mirror variables
  (`GbndRnge` follows `BndsRnXp`). A name which is not available never matches.
- **for** (optional): a list among `scf`, `nscf`, `qp`, `bse`. Default: all of them. The `p2y` run uses the
  options of the first yambo step.
- **use**: `num_machines`, `num_mpiprocs_per_machine`, `num_cores_per_mpiproc`, `max_wallclock_seconds`,
  `queue_name`, `account`, `prepend_text`, `custom_scheduler_commands` (the last two are added to the ones set on the manager), and, for the yambo steps (`qp`, `bse`), `max_walltime` and
  `max_number_of_nodes` (the limits of the restarts) and `parallelism`: a dictionary of Yambo variables or
  `'auto'`, which distributes the MPI tasks with the solver of the plugin.
- **For each step, the first rule that matches is used.** The values you do not give are the ones you set on the
  manager for that step. Put the specific rules first and the general one last.
- At every point the options of every step are first brought back to the ones you set on the manager, then
  the rule is applied: nothing of a previous point stays (neither the resources nor the parallelism variables).
- The rules are checked when the workflow is submitted: an unknown key, a bad step, a non-positive number, a
  `parallelism` for `scf`/`nscf`, or a range with `min` larger than `max` is an error.

With a scan (`scan_points`) the rules work in the same way, point by point.

!!! note "Replaces `parallelism_instructions`"
    The old input `parallelism_instructions` is deprecated: it could not see the parameter values (only the
    largest number of bands and the cutoff), did not cover the DFT steps, and kept parts of the previous point.
    It still works, but a message is written in the report. Move the profiles to `resource_rules`.

## Check what was used

The options used by each calculation are in its inputs:

```bash
verdi process show <pk>      # the calculations of a workflow
verdi calcjob inputcat <pk>  # the submission script, with the resources and OMP_NUM_THREADS
```
