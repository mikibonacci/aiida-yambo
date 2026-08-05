# Proposed Changes & Test Plan

## Part 1: Issues Found in `high_level_API/`

### `general_model.py`

| Line | Issue | Severity |
|------|-------|----------|
| 33 | `MetadataModel.options: OptionsModel = None` — type annotation says `OptionsModel` but default is `None`; accessing `.options.resources` will raise `AttributeError` if not set | Medium |
| 39, 45 | `GeneralInputManager.parameters: dict = {}` — mutable default argument; **all instances share the same dict** | **High** |
| 76 | `set_prepend_text` uses `+=` — with mutable defaults, text compounds across independent instances | Medium |
| 125 | `except:` bare — should catch `AttributeError` or similar minimally | Low |
| 182-183 | `set_ports` hardcodes `metadata_keys=['pw','yambo']` and `keys_to_extract=['scf','nscf','qp','bse']` — fragile if the model is used elsewhere | Low |

### `qe_model.py`

| Line | Issue | Severity |
|------|-------|----------|
| 29-32 | `set_pseudo_family` accesses `self.scf.pw.structure` — `self.scf` is **not defined** in `PwBaseWorkChainInputManager`. Only works when nested inside `YamboInputManager`. **Bug if used standalone or called on `scf`/`nscf` directly** | **High** |
| 34-38 | `set_kmesh` accesses `self.structure` — not defined in this class | **High** |
| 40-44 | `set_k_inverse_distance` accesses `self.pw.structure` — same issue | **High** |
| 67 | `get_mesh()` does `self.parameters.kpoints` — but `self.parameters` is a `dict`, not an object. Should be `self.kpoints`. **Bug** | **High** |
| 16-17 | `settings` and `parallelization` — mutable default dicts | Medium |

### `yambo_model.py`

| Line | Issue | Severity |
|------|-------|----------|
| 165 | `Nb_bse` property returns `self.qp.get_Nb()` instead of `self.bse.get_Nb()`. **Copypaste bug** | **Critical** |
| 314 | `generate_AiiDA_inputs` — `p['bse'] = ... self.qp.model_dump(...)`. Uses `self.qp` instead of `self.bse`. **Copypaste bug** — BSE inputs silently mirror QP inputs | **Critical** |
| 139-142 | All sub-managers (`scf`, `nscf`, `qp`, `bse`) are evaluated at class-definition time as **a single shared instance**. Every `YamboInputManager()` shares the same mutable sub-managers. **Pydantic `Field(default_factory=...)` is needed** | **Critical** |
| 146 | `QP_subsets_dict: dict = {}` — mutable default | **High** |
| 148 | `additional_parsing: list = []` — mutable default | **High** |
| 324 | `class_ob` maps ALL types to `YamboWorkflow.spec().inputs.ports` — even `'YamboCalculation'` and `'YamboConvergence'` get the wrong ports | Medium |
| 193-194 | `set_parent_folder(update_pw_parameters=True)` uses `YamboRestartInputManager.convert_from_AiiDA_data` — should be `PwBaseWorkChainInputManager` or `GeneralInputManager` | Medium |
| 339-347 | `generate_AiiDA_inputs` manually re-packs results after `set_ports`, duplicating port-matching logic | Low |

### Workflow-level (`yambowf.py`)

| Line | Issue | Severity |
|------|-------|----------|
| 692-693 | `QP = {}` immediately followed by `for k,v in QP.items():` — loop body is dead code. The `QP` dict starts empty, so the check never runs | Low |
| 765-783 | `QP` dict is populated but `return ToContext(QP)` overwrites `self.ctx.calc` mapping `'1'`, `'2'`, etc. with calculation futures. Since it's inside a `while_` loop with `calc_to_do = 'QP splitter'`, the old `self.ctx.calc` is lost after the first batch | Medium |
| 559-566 | Logic for `should_run_bse`/`should_run_QP` is convoluted — multiple overlapping `if/elif` branches with complex conditions on empty dicts | Medium |
| 200-226 | `QP_mapper` recursive fallback increases tolerance by 50% on each failure — can grow unbounded | Low |

---

## Part 2: Test Plan — All YamboWorkflow Modes

The workflow is a state machine (`self.ctx.calc_to_do`):
```
scf → nscf → qp → [QP splitter → post_process →] [bse →] done
```

### Primary modes by input combination

| # | Mode | `parent_folder` | `qp` params | `bse` params | `QP_subsets_dict` | Expected path |
|---|------|----------------|-------------|-------------|-------------------|---------------|
| 1 | **GW from scratch** | None | ✓ | ✗ | ✗ | scf → nscf → qp → done |
| 2 | **GW from SCF** | SCF parent | ✓ | ✗ | ✗ | nscf → qp → done |
| 3 | **GW from NSCF** | NSCF parent | ✓ | ✗ | ✗ | qp → done |
| 4 | **GW from Yambo** | Yambo parent | ✓ | ✗ | ✗ | qp → done |
| 5 | **BSE from scratch** | None | ✗ | ✓ | ✗ | scf → nscf → bse → done |
| 6 | **BSE from SCF** | SCF parent | ✗ | ✓ | ✗ | nscf → bse → done |
| 7 | **BSE from NSCF** | NSCF parent | ✗ | ✓ | ✗ | bse → done |
| 8 | **BSE@GW from scratch** | None | ✓ | ✓ | ✓ | scf → nscf → qp → QP splitter → post_process → bse → done |
| 9 | **BSE@GW from SCF** | SCF parent | ✓ | ✓ | ✓ | nscf → qp → QP splitter → post_process → bse → done |
| 10 | **BSE@GW from NSCF** | NSCF parent | ✓ | ✓ | ✓ | qp → QP splitter → post_process → bse → done |
| 11 | **BSE@GW from Yambo** | Yambo parent | ✓ | ✓ | ✓ | QP splitter → post_process → bse → done |
| 12 | **BSE scissor (no GW)** | any | ✗ | ✓ (scissor) | ✗ | → bse → done |
| 13 | **QP-only with splitter** | any | ✓ | ✗ | ✓ | → qp → QP splitter → post_process → done |
| 14 | **Initialise-only** | any | `INITIALISE=True` | `INITIALISE=True` | any | Skips to report |

### Variations (test on modes 1, 5, 8 as baseline)

| # | Variation | How to trigger |
|---|-----------|---------------|
| 15 | **Additional parsing (gap/homo/lumo)** | `additional_parsing = List(['gap_', 'homo', 'lumo'])` |
| 16 | **Additional parsing (excitons)** | `additional_parsing = List(['lowest_exciton', 'brightest_exciton'])` |
| 17 | **QP with explicit band boundaries** | `QP_subsets_dict = {'boundaries': {'bi':1, 'bf':50}, 'qp_per_subset':10, 'parallel_runs':3}` |
| 18 | **QP with energy range** | `QP_subsets_dict = {'range_QP': 5.0, 'qp_per_subset':10, 'parallel_runs':3}` |
| 19 | **QP with explicit QP list** | `QP_subsets_dict = {'explicit': [[k,k,b,b], ...], 'qp_per_subset':10, 'parallel_runs':3}` |
| 20 | **QP full bands mode** | `QP_subsets_dict = {'range_QP': 5.0, 'full_bands': True, ...}` |
| 21 | **QP with `consider_only`** | `QP_subsets_dict = {'consider_only': [v_min, c_max], 'boundaries': {...}, ...}` |
| 22 | **QP with `extend_db`** | `QP_subsets_dict = {'extend_db': True, ...}` |
| 23 | **Already computed QP** | `already_computed_QP_db = List([pk1, pk2])` + matching boundaries |
| 24 | **Not enough nbnd (auto-redo)** | Provide NSCF with `nbnd < yambo Nb` — workflow should redo NSCF |
| 25 | **Clean on failed** | `clean_failed = Bool(True)` |
| 26 | **Molecule (no PBC, Gamma-only)** | `structure` with no periodic directions |

### Protocol-based construction modes

| # | `calc_type` | What is set |
|---|-------------|-------------|
| 27 | `'qp'` | Only `builder.qp` (from `YamboRestart.get_builder_from_protocol` with GW protocol) |
| 28 | `'bse'` | Only `builder.bse` (from `YamboRestart.get_builder_from_protocol` with `bse_*` protocol) |
| 29 | `'bse@qp'` | Both + `builder.QP_subsets_dict` pre-populated with defaults |

---

## Part 3: Proposed Fixes

### `general_model.py` fixes

**1. Mutable defaults — replace `= {}` / `= []` with `Field(default_factory=...)`**

```python
from pydantic import BaseModel, ConfigDict, Field

class GeneralInputManager(BaseModel):
    parameters: dict = Field(default_factory=dict)
    attached_to: int = None
    metadata: MetadataModel = MetadataModel()

    model_config = ConfigDict(extra='allow')
```

**2. `MetadataModel.options` type mismatch**

Fix: give it a proper default instance:

```python
class MetadataModel(BaseModel):
    options: OptionsModel = OptionsModel()
    model_config = ConfigDict(extra='allow')
```

Or if `None` is intentional, type it as `Optional[OptionsModel]`.

**3. Bare `except:` → narrow to specific exceptions**

```python
try:
    results['metadata'] = inputs.metadata
except AttributeError:
    results['metadata'] = inputs.base.attributes.all.get('metadata_inputs', {})
```

---

### `qe_model.py` fixes

**4. `set_pseudo_family` accessing `self.scf.pw.structure`**

Store a `structure` field directly on `PwBaseWorkChainInputManager`:

```python
class PwBaseWorkChainInputManager(GeneralInputManager):
    settings: dict = Field(default_factory=dict)
    parallelization: dict = Field(default_factory=dict)
    structure: Any = None  # populated externally, or via setter

    def set_pseudo_family(self, value: str):
        from aiida import orm
        family = orm.load_group(value)
        self.pseudos = family.get_pseudos(structure=self.structure)
```

Also, ensure the structure is propagated at construction time in `get_nested_inputs` / `set_parent_folder`.

**5. `set_kmesh` and `set_k_inverse_distance` accessing `self.structure`**

Same fix — use `self.structure` (already stored as above):

```python
def set_kmesh(self, value: list = [1, 1, 1], shift: list = [0, 0, 0]):
    from aiida import orm
    self.kpoints = orm.KpointsData()
    self.kpoints.set_cell_from_structure(self.structure)
    self.kpoints.set_kpoints_mesh(value, shift)
```

**6. `get_mesh()` accessing `self.parameters.kpoints`**

Should read from `self.kpoints` directly:

```python
def get_mesh(self) -> list:
    return self.kpoints.get_kpoints_mesh()
```

---

### `yambo_model.py` fixes

**7. `Nb_bse` copypaste (line 165)**

```python
@property
def Nb_bse(self):
    return self.bse.get_Nb()  # was self.qp.get_Nb()
```

**8. `generate_AiiDA_inputs` BSE copypaste (line 314)**

```python
p['bse'] = YamboRestartInputManager.to_AiiDA_inputs(
    self.bse.model_dump(exclude_none=True)
)  # was self.qp.model_dump(...)
```

**9. Shared mutable sub-managers (lines 139-142)**

Use `Field(default_factory=...)` for all four:

```python
class YamboInputManager(BaseModel):
    scf: PwBaseWorkChainInputManager = Field(default_factory=PwBaseWorkChainInputManager)
    nscf: PwBaseWorkChainInputManager = Field(default_factory=PwBaseWorkChainInputManager)
    qp: YamboRestartInputManager = Field(default_factory=YamboRestartInputManager)
    bse: YamboRestartInputManager = Field(default_factory=YamboRestartInputManager)

    parent_folder: Optional[Any] = None
    QP_subsets_dict: dict = Field(default_factory=dict)
    additional_parsing: list = Field(default_factory=list)
```

**10. `class_ob` port mapping wrong (line 324)**

Map each type to its own class for port lookups:

```python
from aiida_yambo.calculations.yambo import YamboCalculation

class_ob = {
    'YamboCalculation': YamboCalculation,
    'scf': YamboWorkflow,
    'nscf': YamboWorkflow,
    'qp': YamboWorkflow,
    'YamboWorkflow': YamboWorkflow,
    'YamboConvergence': YamboConvergence,
    'bse': YamboWorkflow,
}
```

**11. `set_parent_folder` wrong base class (lines 193-194)**

```python
def set_parent_folder(self, node, update_pw_parameters=False):
    self.parent_folder = node
    if self.parent_folder is not None and update_pw_parameters:
        from aiida_yambo.utils.common_helpers import get_pw_inputs_builders
        builder_scf, builder_nscf = get_pw_inputs_builders(parent_folder)

        if builder_scf:
            dict_scf = GeneralInputManager.convert_from_AiiDA_data(builder_scf)
            dict_scf.update(GeneralInputManager.convert_from_AiiDA_data(
                builder_scf.pw, get_metadata=True))
            self.scf = PwBaseWorkChainInputManager(**dict_scf)

        if builder_nscf:
            dict_nscf = GeneralInputManager.convert_from_AiiDA_data(builder_nscf)
            dict_nscf.update(GeneralInputManager.convert_from_AiiDA_data(
                builder_nscf.pw, get_metadata=True))
            self.nscf = PwBaseWorkChainInputManager(**dict_nscf)
```

---

### `yambowf.py` fixes

**12. Dead code in `QP splitter` (lines 692-693)**

Remove the unreachable block entirely.

**13. `self.ctx.calc` overwritten in splitter**

Use namespaced keys to avoid collision:

```python
# Instead of QP = {}  # empty dict, dead code
# Store QP results with prefixed keys
if self.ctx.qp_splitter == 0:
    # ... setup ...
    pass

qp_futures = {}
for i in range(1, 1 + parallel_runs):
    # ... build inputs ...
    future = self.submit(YamboRestart, **yambo_inputs)
    key = f'qp_splitted_{i + self.ctx.qp_splitter}'
    qp_futures[key] = future

return ToContext(**qp_futures)  # don't overwrite self.ctx.calc
```

**14. Convoluted `should_run_bse`/`should_run_QP` logic (lines 559-566)**

Refactor into a clear decision table:

```python
# Determine workflow mode
has_qp_inputs = len(self.ctx.input_manager.qp.parameters) > 0
has_bse_inputs = len(self.ctx.input_manager.bse.parameters) > 0
has_qp_subsets = len(self.ctx.input_manager.QP_subsets_dict) > 0
just_init = self.ctx.just_initialise

self.ctx.should_run_QP = False
self.ctx.should_run_bse = False

if just_init:
    pass  # neither will run
elif has_qp_inputs and has_bse_inputs and has_qp_subsets:
    self.ctx.should_run_QP = bool(
        self.ctx.input_manager.qp.parameters.get('variables', None))
    self.ctx.should_run_bse = self.ctx.should_run_QP
elif has_qp_inputs and has_qp_subsets:
    self.ctx.should_run_QP = True
elif has_bse_inputs and not has_qp_inputs:
    self.ctx.should_run_bse = True
```
