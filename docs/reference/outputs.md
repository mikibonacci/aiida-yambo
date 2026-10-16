# Outputs and parsed quantities

Common workflow outputs include:

| Output | Meaning |
| --- | --- |
| `output_ywfl_parameters` | Requested gaps, HOMO/LUMO, exciton energies, and workflow metadata |
| `array_ndb` | Arrays parsed from YAMBO databases such as QP corrections |
| `bands_quasiparticle` | Quasiparticle band structure |
| `array_chi` | BSE absorption/loss spectrum arrays |
| `history` | Full convergence history from `YamboConvergence` |
| `infos` | Converged parameters and convergence summary |

```python
parameters = run.outputs.output_ywfl_parameters.get_dict()
qp_correction = run.outputs.array_ndb.get_array('E_minus_Eo')
qp_bands = run.outputs.bands_quasiparticle
```

Additional quantities can be requested with strings such as `gap_`, `gap_GG`, `gap_GM`, `homo`, `lumo`, `lowest_exciton`, and `brightest_exciton`. The requested k-point must exist in the NSCF mesh for explicit high-symmetry-point quantities.
