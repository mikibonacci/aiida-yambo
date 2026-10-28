# Examples

Linear scripts: the settings are on top, then the structure (silicon, from ASE), the inputs, the scheduler options,
the submission and, commented, how to read the results. Edit the settings, then run

```bash
verdi run examples/02_gw/01_gw_from_scratch.py
```

**Prerequisites**: a configured AiiDA profile and daemon, and the codes `pw`, `p2y`, `yambo` (see the installation
pages of the docs; the scripts use the labels `pw@mycluster`... that you must change); the PseudoDojo
norm-conserving pseudopotentials (`aiida-pseudo install pseudo-dojo`); `yambopy` for the QP splitter.

Status: **cluster** = the same kind of inputs ran on a cluster; **tested** = the inputs are validated against the
process specifications by the tests of this repository (nothing is run), not yet run on a cluster.

| Script | Shows | Status |
| --- | --- | --- |
| `01_dft/01_scf_nscf.py` | SCF and NSCF with `PwBaseWorkChain`, to produce the parents | cluster |
| `02_gw/01_gw_from_scratch.py` | GW from a structure: scf, nscf, p2y, GW | cluster |
| `02_gw/02_gw_from_parent.py` | GW starting from an SCF, an NSCF or a Yambo run | cluster |
| `02_gw/03_qp_splitter.py` | the QP splitter, all the modes (`MODE`), and its resources | cluster (modes), tested (`auto`, walltime) |
| `02_gw/04_qp_already_computed.py` | skip the QP of an existing `ndb.QP` | tested |
| `03_bse/01_bse_scissor.py` | BSE with a scissor | cluster |
| `03_bse/02_bse_at_gw.py` | BSE on explicit GW corrections (QP splitter, merge) | cluster |
| `04_options/01_resources_and_parallelism.py` | resources, queue, parallelism and restart limits of every step | tested |
| `04_options/02_initialise_only.py` | only p2y and the initialization, to reuse the `SAVE` | cluster |
| `04_options/03_cleaning_and_pseudos.py` | `clean_failed`, LDA pseudopotentials, too few NSCF bands | cluster |
| `04_options/04_molecule_and_metals.py` | a molecule, and `stop_if_metal` | experimental (tested) |
| `05_convergence/01_qp_convergence.py` | convergence of a GW gap | cluster |
| `05_convergence/02_bse_convergence.py` | convergence of the lowest exciton of a BSE | cluster |
| `05_convergence/03_bse_at_gw_convergence.py` | convergence of a BSE@GW | **experimental**: the last cluster run failed |
| `05_convergence/04_scan.py` | run a list of parameter sets, no convergence | tested |
| `05_convergence/05_fit_functions.py` | choose the fit function of each parameter | tested |
| `05_convergence/06_fit_models_offline.py` | write and test a fit model on data, no cluster | runs anywhere |
| `05_convergence/07_resource_rules.py` | resources which depend on the values of the parameters | tested |
| `06_low_level/01_p2y.py` ... `04_yambo_restart.py` | one `YamboCalculation` (p2y, GW, IP-RPA) and `YamboRestart` | rewritten from old scripts, tested |
| `07_analysis/01_read_workflow_results.py` | read the outputs of a workflow (parameters, QP, spectrum) | tested |
| `07_analysis/02_read_convergence_results.py` | read the story of a convergence as a table, plot it | tested |
| `08_ypp/01_ypp_wannier.py` | `ypp` for Wannier90 (the plugin support is unfinished) | experimental (tested) |

The scripts are tested by `tests/test_examples_workflows.py` (they run with test nodes and a `submit` which validates
the inputs). The exit codes and the options are in the docs, "Options and exit codes".
