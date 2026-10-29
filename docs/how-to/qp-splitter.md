# Split the GW calculation: the QP splitter

A GW run computes the quasiparticle (QP) correction of a list of `(k-point, band)` pairs. The **QP splitter**
cuts this list in subsets, runs several smaller Yambo calculations, and merges the results in a single `ndb.QP`.
Use it when the list is too long for one run, when you want to put the resources where they are needed, or when
you want a QP database for a later BSE (`calc_type='bse@qp'`) or an extended one.

## How it works

```text
scf -> nscf -> p2y + GW (first run) -> [ GW on subset 1 ... GW on subset n ] -> merge -> (extend) -> (BSE)
```

1. The first `qp` run is a normal `YamboRestart`. It does `p2y` and the screening, and writes the databases that
   the following runs copy (`COPY_DBS`), so the screening is **not** recomputed in each subset. It is not a run
   of the full QP list.
2. The QP list is built from the option you choose (see the modes below) and cut in subsets of `qp_per_subset`
   `(k, band)` pairs. Each subset becomes one `YamboRestart` with its own `QPkrange`.
3. `parallel_runs` subsets run at the same time. When they finish, the next ones start. **If one of them failed,
   no new one starts** and the workflow stops with the exit code `ERROR_SPLITTED_QP_FAILED`.
4. The databases are merged with `yambopy mergeqp` (it must be installed where the workflow runs). The merged
   database is the output `merged_QP`; the uuids of the splitted runs are in `splitted_QP_calculations`.
5. With `extend_db`, an extended database (scissor + Fermi-Dirac smearing of the corrections, for the bands you
   did not compute) is also built: `extended_QP`.
6. With `calc_type='bse@qp'` the BSE then uses the merged database (`KfnQPdb`). The workflow stops with
   `ERROR_MISSING_BAND_REGION` if the QP computed do not contain both valence and conduction states.

The splitter is set with `manager.set_QP_subsets_dict(...)`: the same keys go in the input `QP_subsets_dict`
(`help` of the port in `YamboWorkflow`). **Unknown keys raise an error, and giving more than one source of QP
(`bi`/`bf`, `range_QP`, `explicit`) raises too.**

```python
--8<-- "examples/02_gw/03_qp_splitter.py:splitter"
```

The complete script is in `examples/02_gw/03_qp_splitter.py` (change `MODE`).

## Which QP: the modes

All the indices (k-points, bands) start from 1.

```python
--8<-- "examples/02_gw/03_qp_splitter.py:modes"
```

| Mode | Keys | What is computed |
| --- | --- | --- |
| boundaries | `bi`, `bf` | All the k-points, the bands `bi..bf` |
| k-range | `ki`, `kf`, `bi`, `bf` | Only the k-points `ki..kf`, the bands `bi..bf` |
| `range_QP` | `range_QP` (eV) | The `(k, band)` pairs whose energy is within `range_QP` of mid-gap. If the window does not contain both valence and conduction states, it is enlarged by 50% until it does |
| `full_bands` | `range_QP`, `full_bands=True` | As `range_QP`, but for every k-point all the bands between the lowest and the highest selected |
| explicit | `explicit=[[k1, k2, b1, b2], ...]` | Exactly the listed ranges |
| extended | any of the above + `extend_db=True`, `consider_only=[v, c]` | Also `extended_QP`: the corrections of the bands you did not compute are built from a scissor and a smeared fit, using the valence and conduction bands `v` and `c` (`Nb` and `T_smearing` set the bands and the smearing in eV) |

With no source at all, the QP of all the bands (`1..Nb`) at all the k-points are computed.

Other keys of the dictionary:

- `consider_only`: only the QP whose band is in the list are computed (default `[-1]`: all). With `extend_db` it
  is `[valence, conduction]`, the two bands used to define the scissor.
- `range_spectrum` (with `range_QP`): the energy window (eV) which defines the band range of the BSE (`BSEBands`),
  if you do not set it yourself. The indices are 1-based, as Yambo wants them.
- Workflow input `already_computed_QP_db` (a list of pks of `ndb.QP`): the QP already present in these databases
  are skipped. This check can be slow on large databases.

## Choose the resources of the splitted runs

Without options, the splitted runs use the resources of the first run. You can set them for the splitted runs only:

```python
manager.set_QP_subsets_dict(
    qp_per_subset=8, parallel_runs=4, range_QP=5.0,
    resources={'num_machines': 1, 'num_mpiprocs_per_machine': 16, 'num_cores_per_mpiproc': 2},
    max_wallclock_seconds=7200,
    queue_name='short',
    parallelism='auto',
    prepend_text='module load yambo',
    custom_scheduler_commands='#SBATCH --constraint=cpu',
)
```

- `resources`: the values not given are kept (giving only `num_machines` does not reset the others to 1).
  `OMP_NUM_THREADS` is exported from `num_cores_per_mpiproc`.
- `max_wallclock_seconds` and `queue_name`: walltime and queue of the splitted runs (they are usually shorter).
- `prepend_text`, `custom_scheduler_commands`: added to the ones of the first run (modules, `#SBATCH` lines).
- `parallelism`: either a dictionary of Yambo variables (`{'SE_CPU': '1 4 2 1', 'SE_ROLEs': 'q qp b g'}`), or
  `'auto'`: for each subset the plugin distributes the MPI tasks (`find_parallelism_qp`) on the QP of the subset,
  and uses fewer tasks than the available ones if the subset is too small to feed them.
- `clean_workdir` (default `True`): the remote folders of the splitted runs, and only those, are cleaned by the workflow once
  the merge is done (the SCF, NSCF and the QP run it starts from are kept). If a splitted run fails, or the run stops
  because the merged database lacks bands, nothing is cleaned, so the runs can be inspected.

Some rules of thumb. They are heuristics, check them on a small case:

- The self-energy is distributed on the QP, the bands of the sum and the G-vectors, so a subset of `qp_per_subset`
  QP can use a limited number of MPI tasks efficiently: more tasks than QP times a few bands are idle. Prefer
  more subsets (`qp_per_subset` smaller) over more tasks per subset when you have many nodes.
- The total you use at the same time is `parallel_runs` times the resources of one run.
- If the memory is the problem, use more threads per task (`num_cores_per_mpiproc`) and fewer tasks per node,
  not more subsets.
- `qp_per_subset` too small gives many short jobs and a longer merge; too large loses the benefit of the split.

## When something goes wrong

- The restart layer (`YamboRestart`) still acts inside every splitted run: it increases the walltime after a
  timeout and it replaces the parallelism after a parallelism error (it uses `PAR_def_mode='balanced'`, so
  your `parallelism` is not kept for that restart). The limits are `max_walltime` and `max_number_of_nodes`.
- `ERROR_SPLITTED_QP_FAILED`: a splitted run failed. Look at it with `verdi process report` on the nodes in
  `splitted_QP_calculations`, and relaunch the workflow with the finished first run as the parent folder (see [GW with `YamboWorkflow`](../tutorials/workflow.md)), so that the DFT part and the first run are not repeated.
- `ERROR_MISSING_BAND_REGION`: the QP computed do not contain both valence and conduction states, which the
  BSE and the extended database need. Enlarge `range_QP` or the band range.
- The merge uses `yambopy`: if it is missing the workflow fails at the merge step, after all the runs.
