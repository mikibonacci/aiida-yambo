# How-to guides

Concise recipes for common operational tasks. They assume that the installation and code setup are complete.

<div class="grid cards" markdown>

-   **Inspect and recover calculations**

    Locate retrieved files, recover parents, and diagnose failed processes.

    [Recipes](recipes.md)

-   **Split the GW calculation**

    Choose which quasiparticles to compute, run them in several smaller runs, merge them, and size the resources.

    [QP splitter](qp-splitter.md)

-   **Set the resources**

    Nodes, MPI, threads, walltime and parallelism: in a workflow, in the QP splitter, and for given values of
    the parameters in a convergence.

    [Resources and parallelism](resources.md)

-   **Analyse the databases**

    Use the lattice and the excitons of a BSE run in yambopy, from AiiDA nodes: the response function, the
    exciton composition, and the raw files if you keep them.

    [Yambopy interface](yambopy-interface.md)

-   **Understand failures and restarts**

    What the restart layer does by itself, and what to check when a workflow stops.

    [Restarts and troubleshooting](restarts-and-troubleshooting.md)

-   **Choose the fit function of the convergence**

    The built-in fit functions (default: power law), what `alpha` and `beta` are, how to choose them per
    parameter, and how to write your own. Only for the automated convergence.

    [Fit functions](convergence-fit-functions.md)

-   **Run fixed points (scan)**

    Run a list of parameter sets without convergence and without any fit.

    [Scan](scan-points.md)

</div>
