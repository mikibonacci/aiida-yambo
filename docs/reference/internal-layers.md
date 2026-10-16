# Internal calculation layers

Most users should start from `YamboWorkflow` or `YamboConvergence`. The lower-level classes remain part of the plugin because the workflows need them internally and advanced users sometimes need direct control.

## `YamboRestart`

`YamboRestart` wraps a YAMBO calculation with retry logic for walltime exhaustion, memory/parallelization failures, and recoverable database errors. `YamboWorkflow` uses it internally for the QP and BSE stages.

## `YamboCalculation`

`YamboCalculation` is the CalcJob that prepares the working directory, runs `p2y`/YAMBO when necessary, retrieves files, and parses outputs. It is intentionally not part of the getting-started path.

## When to use them

Use these layers only when:

- implementing or debugging a new workflow;
- reproducing a very specific YAMBO input outside the supported workflow logic;
- developing a plugin extension or parser test.

For normal GW, BSE, BSE@GW, restart, and convergence work, use `YamboInputManager` with `YamboWorkflow` or `YamboConvergence`.
