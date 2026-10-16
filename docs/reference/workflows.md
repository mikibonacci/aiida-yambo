# Workflow reference

| Workflow | Use it for |
| --- | --- |
| `YamboWorkflow` | DFT starting point plus GW/BSE/combined flows; recommended entry point |
| `YamboConvergence` | Adaptive convergence studies around `YamboWorkflow` |
| `YamboRestart` | Internal restart layer with automatic error handling and retries |
| `YamboCalculation` | Internal low-level CalcJob for one exact YAMBO calculation |
| `YppRestart` | Restartable YPP post-processing |

Entry points are:

```text
yambo.yambo
yambo.yambo.yamborestart
yambo.yambo.yambowf
yambo.yambo.yamboconvergence
yambo.ypp.ypprestart
```

Use `YamboInputManager` together with `get_builder_from_protocol` whenever possible. It keeps protocol defaults, nested namespaces, and compatibility logic in the plugin rather than duplicating AiiDA data-node construction in scripts. See the [high-level input API tutorial](../tutorials/high-level-api.md).
