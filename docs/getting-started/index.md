# Getting started

The shortest route to a first calculation is:

<div class="grid cards" markdown>

-   **Install the plugin**

    Install `aiida-yambo`, create an AiiDA profile, and start the daemon.

    [Installation](../installation.md)

-   **Configure the codes**

    Register Quantum ESPRESSO, `p2y`, and YAMBO as AiiDA codes.

    [Code setup](../setup.md)

-   **Prepare DFT data**

    Run the SCF and NSCF calculations required by YAMBO.

    [DFT starting point](../tutorials/prerequisites.md)

</div>

!!! warning "Before submitting"
    Confirm that `verdi status` is healthy, the daemon is running, and the YAMBO executable can be launched on the target computer. Use small test inputs before starting a convergence campaign.
