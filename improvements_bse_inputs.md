# Fix: BSE inputs always populated from QP inputs

## Problem

`aiida_yambo/utils/high_level_API/yambo_model.py` line 314 always populates the BSE
input with the QP one:

```python
p['bse'] = YamboRestartInputManager.to_AiiDA_inputs(self.qp.model_dump(exclude_none=True))
```

This has two consequences:

1. **Copypaste bug (Critical):** BSE inputs silently mirror the QP inputs instead of
   using `self.bse`.
2. **Both `qp` and `bse` namespaces are always emitted**, even when the corresponding
   sub-manager is empty. Because `get_nested_inputs(..., get_metadata=True)` pulls the
   namespace *default* metadata (`{'options': {...}}`), an unprovided `qp`/`bse`
   sub-manager still dumps to a non-empty `{'metadata': {...}}` dict — a partial
   namespace with **no `code`**.

When `YamboInputManager.populate_builder()` sets that partial namespace on the builder,
AiiDA validation fails:

```
ValueError: invalid attribute value required value was not provided for 'code'
```

This breaks `YamboWorkflow.get_builder_from_protocol(calc_type='bse')` (empty `qp`
namespace is emitted), while `calc_type='qp'` and `'bse@qp'` pass.

A second, related bug blocks **from-scratch** modes: `get_builder_from_protocol`
with `parent_folder=None` raises

```
UnboundLocalError: cannot access local variable 'builder_scf'
```

because `builder_scf` / `builder_nscf` are only assigned inside the
`if parent_folder is not None:` block.

## Root cause chain

```
generate_AiiDA_inputs always emits p['qp'] / p['bse']
        └─ empty sub-manager dump still contains default metadata {'options': {...}}
                └─ populate_builder sets a partial namespace (metadata only, no code)
                        └─ AiiDA validation error: "required value was not provided for 'code'"
```

Verified in the devbox profile:

| `calc_type` | Result today |
|-------------|--------------|
| `qp` | OK |
| `bse` | FAIL (`'code'` required, from empty `qp`) |
| `bse@qp` | OK |

## Proposed changes

### 1. `aiida_yambo/utils/high_level_API/yambo_model.py`

**`generate_AiiDA_inputs`** — fix the copypaste and only emit populated sub-namespaces:

```python
p = YamboRestartInputManager.to_AiiDA_inputs(
    self.model_dump(exclude_none=True, exclude={'scf', 'nscf', 'qp', 'bse'})
)
for key, manager in (('qp', self.qp), ('bse', self.bse),
                     ('scf', self.scf), ('nscf', self.nscf)):
    sub_inputs = YamboRestartInputManager.to_AiiDA_inputs(manager.model_dump(exclude_none=True))
    if sub_inputs and GeneralInputManager._has_provided_inputs(manager):
        p[key] = sub_inputs
```

Notes:

- `exclude={'scf', 'nscf', 'qp', 'bse'}` keeps the (possibly empty) sub-managers out of
  the top-level dump, so they cannot leak in as `orm.Dict`.
- The `self.qp` → `self.bse` copypaste is fixed by iterating the actual managers.
- An empty sub-manager is simply not emitted, so `populate_builder` never sets an
  incomplete namespace on the builder.

**New static helper** on `GeneralInputManager`:

```python
@staticmethod
def _has_provided_inputs(manager) -> bool:
    data = manager.model_dump(exclude_none=True)
    data.pop('metadata', None)          # always-present default metadata
    return any(v not in ({}, None) for v in data.values())
```

**`get_nested_inputs`** (YamboWorkflow and YamboConvergence cases) — guard missing
sub-namespaces, so a workflow launched without `qp` / `bse` / `scf` / `nscf` does not
crash in `setup()`. Stored inputs are an `AttributeDict`, which raises `AttributeError`
on missing keys:

```python
if 'qp' in inputs:
    results['qp'] = YamboRestartInputManager.convert_from_AiiDA_data(inputs.qp)
    results['qp'].update(YamboRestartInputManager.convert_from_AiiDA_data(
        inputs.qp.yambo, get_metadata=True))
# same guard for 'bse', 'scf', 'nscf'
```

For builders the namespaces always exist (`'qp' in builder` is `True` because of
`expose_inputs`), so the guard is a no-op there; emptiness is handled by the emission
check in `generate_AiiDA_inputs`.

### 2. `aiida_yambo/workflows/yambowf.py` — `get_builder_from_protocol`

**Fix the `UnboundLocalError`** for from-scratch modes:

```python
builder = cls.get_builder()
builder_scf, builder_nscf = None, None

if parent_folder is not None:
    from aiida_yambo.utils.common_helpers import get_pw_inputs_builders
    builder_scf, builder_nscf = get_pw_inputs_builders(parent_folder)
    builder.parent_folder = parent_folder
```

**Lazy yambo sub-builders** — only build/populate the namespaces requested by
`calc_type`, so the `bse_*` protocol is not even invoked for `calc_type='qp'`:

```python
qp_builder = YamboRestart.get_builder_from_protocol(
    preprocessing_code=preprocessing_code, code=code, protocol=protocol,
    parent_folder=parent_folder, overrides=overrides_yres, NLCC=NLCC,
    RIM_v=RIM_v, RIM_W=RIM_W, nelectrons=nelectrons, ecutwfc=ecutwfc,
) if calc_type in ('qp', 'bse@qp') else None

bse_builder = YamboRestart.get_builder_from_protocol(
    preprocessing_code=preprocessing_code, code=code, protocol='bse_' + protocol,
    parent_folder=parent_folder, overrides=overrides_yres, NLCC=NLCC,
    RIM_v=RIM_v, RIM_W=RIM_W, nelectrons=nelectrons, ecutwfc=ecutwfc,
) if calc_type in ('bse', 'bse@qp') else None

if qp_builder:
    builder.qp = qp_builder
if bse_builder:
    builder.bse = bse_builder
if calc_type == 'bse@qp':
    builder.QP_subsets_dict = orm.Dict({'parallel_runs': 3, 'qp_per_subset': 10})
```

`populate_builder` itself needs **no** change — once empty namespaces are not emitted,
it simply will not set them.

## Resulting behaviour

| `calc_type` | Builder namespaces after fix |
|-------------|------------------------------|
| `qp` | `scf`, `nscf`, `qp` (no `bse`) |
| `bse` | `scf`, `nscf`, `bse` (no `qp`) |
| `bse@qp` | `scf`, `nscf`, `qp`, `bse` + `QP_subsets_dict` |

And `bse` content now genuinely comes from `self.bse` (the `bse_*` protocol), not from
`self.qp`.

## Verification

1. `get_builder_from_protocol(calc_type='qp' | 'bse' | 'bse@qp')` all succeed, with only
   the requested yambo namespaces populated.
2. `get_builder_from_protocol(parent_folder=None)` succeeds (no `UnboundLocalError`),
   unblocking from-scratch modes (scf → nscf → ...).
3. `get_nested_inputs` on an `AttributeDict` without `bse` no longer raises.
4. Syntax check and import of both modules.
