# Outputs and parsed quantities

Common workflow outputs include:

| Output | Meaning |
| --- | --- |
| `output_ywfl_parameters` | Requested gaps, HOMO/LUMO, exciton energies, and workflow metadata |
| `array_ndb` | Arrays parsed from YAMBO databases such as QP corrections |
| `bands_quasiparticle` | Quasiparticle band structure |
| `array_chi` | BSE absorption/loss spectrum arrays |
| `lattice` | `YamboLatticeData` (from `ns.db1`), for yambopy: [Yambopy interface](../how-to/yambopy-interface.md) |
| `exciton_db.Q1`, ... | `YamboExcitonData` (from `ndb.BS_diago_Q*`), one for each Q point of a BSE calculation |
| `collection` | `YamboConvergence`: `story` (one record per calculation: variables, parsed quantities, `failed`, `useful`, `converged`) and the `workflows` pks/uuids |
| `convergence_summary` | `YamboConvergence`: converged values and details for each parameter group (or `scan` details for a scan) |

```python
parameters = run.outputs.output_ywfl_parameters.get_dict()
qp_correction = run.outputs.array_ndb.get_array('E_minus_Eo')
qp_bands = run.outputs.bands_quasiparticle
```

Additional quantities can be requested with strings such as `gap_`, `gap_GG`, `gap_GM`, `homo`, `lumo`, `lowest_exciton`, and `brightest_exciton`. The requested k-point must exist in the NSCF mesh for explicit high-symmetry-point quantities.
