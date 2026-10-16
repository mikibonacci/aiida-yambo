# Recipes and troubleshooting

These are short interventions after a workflow has been created with `YamboInputManager`. They are not replacements for the main tutorials.

## Find retrieved files and remote folders

```python
from aiida import orm

node = orm.load_node(PK)
retrieved_path = node.outputs.retrieved._repository._repo_folder.abspath
remote_path = node.outputs.remote_folder.get_remote_path()
```

## Recover the Quantum ESPRESSO parent

```python
from aiida_yambo.utils.common_helpers import find_pw_parent
scf = find_pw_parent(node, calc_type=['scf'])
nscf = find_pw_parent(node, calc_type=['nscf'])
```

## Continue from a previous YAMBO run

Set the previous node's `remote_folder` through the high-level manager. This keeps the continuation in the normal `YamboWorkflow` path:

```python
previous = orm.load_node(PREVIOUS_PK)
manager.set_parent_folder(previous.outputs.remote_folder)
# Enable COPY_DBS only when the continuation needs previous databases;
# use the workflow/settings escape hatch for this site-specific option.
```

## Inspect a failed process

```bash
verdi process report <pk>
verdi process show <pk>
verdi calcjob inputcat <pk>
```

The most useful first checks are the scheduler exit status, the YAMBO report file, available memory, and whether `p2y` created a valid `SAVE` database. If the parent is wrong, restart from the last successful SCF/NSCF/YAMBO node rather than editing generated remote files.
