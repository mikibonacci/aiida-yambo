# Parallelism and restarts

YAMBO can distribute work over k-points, bands, q-points, and groups. Start with the built-in mode and only add explicit roles when a production calculation demonstrates a need:

```python
builder.parameters = orm.Dict(dict={
    'arguments': ['gw0'],
    'variables': {
        'PAR_def_mode': 'balanced',
        'X_and_IO_CPU': '1 1 1 1 1',
        'X_and_IO_ROLEs': 'q k g c v',
    },
})
```

The internal `YamboRestart` layer handles common failures such as walltime exhaustion, parallelization errors, corrupted databases, and memory imbalance. Users normally access this through `YamboWorkflow`; they do not need to construct the restart builder directly.

Set resources through `YamboInputManager` before generating the `YamboWorkflow` inputs:

```python
manager.qp.set_resources(nodes=2, mpi=8, threads=2, time=14400)
manager.qp.set_npools(4)
manager.additional_parsing = ['gap_', 'homo', 'lumo']
```

Use `max_number_of_nodes` and `max_walltime` to cap automatic resource growth. Always check that the scheduler's `OMP_NUM_THREADS` matches the requested CPU allocation.
