# The flow of `YamboWorkflow`

`YamboWorkflow` decides which calculations to run from the inputs it gets: a namespace (`scf`, `nscf`, `qp`,
`bse`) which is not provided is not run, and a parent folder skips the steps already done.

```text
scf -> nscf -> p2y + qp -> [ QP splitter: qp on subsets -> merge QP ] -> bse
```

| `calc_type` | Namespaces | What happens |
| --- | --- | --- |
| `'qp'` | `scf`, `nscf`, `qp` | DFT, then `p2y` and GW |
| `'bse'` | `scf`, `nscf`, `bse` | DFT, then BSE with a scissor (`KfnQP_E`) |
| `'bse@qp'` | `scf`, `nscf`, `qp`, `bse`, `QP_subsets_dict` | DFT, GW on the quasiparticles you choose (QP splitter), merge, then BSE with the merged `ndb.QP` |

- **Parent folder.** The workflow looks at what the parent is: a finished SCF removes the SCF step, an NSCF
  removes SCF and NSCF, a Yambo run is used as it is. An empty parent folder is ignored.
- **The first yambo step.** It runs `p2y`, the initialization and, for `qp`, a first GW run whose databases
  (screening) are copied by the following runs: the QP splitter does not recompute them. With `INITIALISE` only
  this step runs, to produce a `SAVE` folder to reuse.
- **Bands.** The NSCF must have at least as many bands as the GW asks (`manager.set_nscf_nbnd`); the
  workflow checks `manager.has_enough_nbnd` and redoes the NSCF if it has too few.
- **Results.** `output_ywfl_parameters` has the quantities in `additional_parsing`; the other outputs depend on
  the steps (`merged_QP`, `extended_QP`, `bands_quasiparticle`...). See [Outputs](../reference/outputs.md).
- **Restarts.** Every Yambo calculation runs inside a `YamboRestart` that handles the usual failures. See
  [Resources and parallelism](../how-to/resources.md).

Protocols (`protocol='fast' | 'moderate' | 'precise'`, default `moderate`) fill the inputs of every step;
you then change what you need with the `YamboInputManager`.
