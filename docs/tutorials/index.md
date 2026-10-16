# Tutorials

These tutorials replace the original notebooks with copyable Markdown examples. They follow the same progression as `tutorials-aiida-yambo/`, but use the current protocol/high-level API patterns exercised in `devbox_aiida/work/final`.

<div class="grid cards" markdown>

-   **Prerequisites**

    Create the Quantum ESPRESSO SCF/NSCF starting point with norm-conserving pseudopotentials.

    [Prepare the DFT starting point](prerequisites.md)

-   **High-level inputs**

    Configure workflows without manually assembling nested AiiDA data nodes.

    [Use the input API](high-level-api.md)

-   **GW and BSE**

    Prepare inputs with the high-level API and run complete QP and excitonic workflows.

    [Input API](high-level-api.md) · [GW workflow](workflow.md) · [BSE](bse.md)

-   **Convergence and interpolation**

    Automate convergence and pass QP corrections to Wannier90.

    [Convergence](convergence.md) · [Wannier90](wannier90.md)

</div>

1. [Prepare the DFT starting point](prerequisites.md)
2. [Prepare inputs with the high-level API](high-level-api.md)
3. [Run GW with YamboWorkflow](workflow.md)
4. [Run BSE and BSE@GW](bse.md)
5. [Automate convergence](convergence.md)
6. [Interpolate GW bands with Wannier90](wannier90.md)

All examples assume that `load_profile()` succeeds and that the code labels and pseudo family have been adapted to your profile.
