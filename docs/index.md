# aiida-yambo

`aiida-yambo` connects [YAMBO](https://www.yambo-code.eu/) to [AiiDA](https://www.aiida.net/). It provides provenance-preserving calculations and workflows for quasiparticle and optical properties, including GW, BSE, IP-RPA, QP splitting/merging, automatic restarts, and convergence studies.

## Choose a path

<div class="grid cards" markdown>

-   :material-rocket-launch: **Install and configure**

    Set up AiiDA, Quantum ESPRESSO, `p2y`, and YAMBO.

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-school: **Run the first workflow**

    Start from a structure, prepare inputs with the high-level API, and let `YamboWorkflow` create the DFT and MBPT steps.

    [:octicons-arrow-right-24: YamboWorkflow tutorial](tutorials/workflow.md)

-   :material-chart-bell-curve-cumulative: **Converge a result**

    Automate coupled convergence of cutoffs, bands, k-points, or FFT grids.

    [:octicons-arrow-right-24: Convergence tutorial](tutorials/convergence.md)

</div>

!!! warning "Norm-conserving pseudopotentials"
    The YAMBO interface requires norm-conserving pseudopotentials for the Quantum ESPRESSO starting point. The examples use the `PseudoDojo/0.4/PBE/SR/standard/upf` family; replace it with a compatible family available in your profile.

## What is stored

Every calculation, input, remote folder, retrieved file, parsed quantity, and workflow relationship is an AiiDA node. You can therefore restart, inspect, reproduce, and query a calculation without relying on notebook state or manually tracked folders.

## Cite

If you use the plugin, cite [Bonacci *et al.*, *npj Computational Materials* 9, 74 (2023)](https://doi.org/10.1038/s41524-023-01027-2), the [YAMBO methodology paper](https://doi.org/10.1088/1361-648X/ab15d0), and [AiiDA 1.0](https://doi.org/10.1038/s41597-020-00638-4).
