# Wannier90 band interpolation

The Wannier90 examples combine Quantum ESPRESSO, Wannier90, and merged YAMBO QP databases. The workflow is:

1. run a DFT NSCF calculation on a dense mesh;
2. run Wannier90 and validate the interpolated DFT bands;
3. compute QP corrections with `YamboWorkflow` and the QP splitter;
4. merge the QP databases;
5. interpolate the corrected bands along a custom path.

The reason for separating the DFT and QP stages is practical: Wannier90 provides a smooth representation of the DFT bands, while the merged QP database supplies corrections only where they are needed. The splitter avoids one very large YAMBO job and makes the independent QP pieces schedulable in parallel.

The same AiiDA principle applies: pass remote folders and data nodes, not filesystem paths. Set up the codes for your cluster as described in [Configure AiiDA and codes](../setup.md).

## Select the QP window

When band indices are not known in advance, use an energy window around the Fermi level. `range_QP` controls the explicitly corrected window; `range_spectrum` includes additional bands for the final spectrum.

```python
from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager

# manager is a YamboInputManager for a YamboWorkflow with calc_type='qp'.
manager.set_QP_subsets_dict(
    range_QP=5.0,
    range_spectrum=10.0,
    full_bands=True,
    split_bands=True,
    qp_per_subset=8,
    parallel_runs=2,
)
```

Use an explicit list when reproducibility of the selected states matters more than automatic energy selection:

```python
manager.set_QP_subsets_dict(
    explicit=[[1, 1, 1, 2], [1, 1, 2, 2]],
    qp_per_subset=2,
    parallel_runs=1,
)
```

The four integers in each explicit entry are the YAMBO QP range coordinates. Keep them consistent with the k-point and band indexing of the NSCF calculation.

## Validate the interpolation

Always compare the Wannier-interpolated DFT bands with the reference Quantum ESPRESSO bands before applying QP corrections. A mismatch usually means that the disentanglement window, projections, or k-point path needs attention; it is not a YAMBO convergence problem.

The resulting merged database is exposed by the workflow and can be passed to the Wannier90-side workflow. Keep the merged database and the Wannier interpolation in the same AiiDA provenance graph so that the final band structure remains reproducible.
