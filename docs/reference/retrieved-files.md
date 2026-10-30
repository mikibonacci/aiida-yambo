# Retrieved files and settings

What a `YamboCalculation` brings back from the cluster, what is kept in the AiiDA repository, what is only retrieved
temporarily, and the `settings` that change it. The settings are keys of the `settings` input (with the high-level
API: `manager.qp.settings`, `manager.bse.settings`...).

## What is retrieved

| Files | Retrieved | Why |
| --- | --- | --- |
| `r-*` (reports), `l-*` (logs), `o-*` (output tables), `LOG/l*_CPU_1`, `*stderr*` | stored in `retrieved` | parsed (gaps, spectra, errors, timings) and read by the workflows (Fermi level, k-points...); small |
| `ndb.QP` (GW runs) | stored in `retrieved` | it is also the `QP_db` output used by the QP merge, the BSE and `ypp` |
| `ns.db1` (lattice) | **temporary** | the parser creates the `lattice` output ([`YamboLatticeData`](../how-to/yambopy-interface.md)) |
| `ndb.BS_diago_Q*` (excitons) | **temporary** | the parser creates `exciton_db.Q*` ([`YamboExcitonData`](../how-to/yambopy-interface.md)) and `array_chi`, `array_excitonic_states` |
| `ndb.HF_and_locXC` | **temporary** | only read by the parser (`array_ndb`, `array_ndb_HFlocXC`) |
| anything in `ADDITIONAL_RETRIEVE_LIST` | stored in `retrieved` | see [below](#add-files-to-the-retrieve-list) |

*Temporary* means that the file is copied from the cluster to a temporary folder, the parser reads it, and the folder
is deleted: the file does not stay in the AiiDA repository. If the parser is not run (for example the calculation
failed before) nothing is kept. Every `YamboCalculation` retrieves `ns.db1` (`SAVE/ns.db1`).

`ypp` calculations (`YppCalculation`) are not changed: `ndb.QP_merged*`, `*eig` and `*nnkp` are stored.

## Settings

| Key | Default | Effect |
| --- | --- | --- |
| `KEEP_RAW_DBS` | `False` | Retrieve `ns.db1`, `ndb.BS_diago_Q*` and `ndb.HF_and_locXC` into `retrieved` (stored), as well as creating the data nodes. [How to use them](../how-to/yambopy-interface.md#keep-the-raw-files-and-use-them-later). |
| `STORE_EXCITON_EIGENVECTORS` | `False` | Also store the exciton eigenvectors in `YamboExcitonData` (single precision, `complex64`). They exist only if the BSE was run with `WRbsWF`. |
| `N_EXCITON_EIGENVECTORS` | all | Store only the eigenvectors of the first N excitons (a positive integer). Only with `STORE_EXCITON_EIGENVECTORS`. |
| `ADDITIONAL_RETRIEVE_LIST` | `[]` | Extra files to retrieve and store (below). |

The values are checked when the calculation is submitted: `KEEP_RAW_DBS` and `STORE_EXCITON_EIGENVECTORS` must be
booleans, `N_EXCITON_EIGENVECTORS` a positive integer.

```python
settings = {
    'KEEP_RAW_DBS': True,
    'STORE_EXCITON_EIGENVECTORS': True,
    'N_EXCITON_EIGENVECTORS': 20,
    'ADDITIONAL_RETRIEVE_LIST': ['aiida.out/ndb.dipoles'],
}
```

## Add files to the retrieve list

`ADDITIONAL_RETRIEVE_LIST` is a list of paths (patterns are allowed) relative to the working directory of the
calculation on the cluster, where yambo runs with `-J aiida.out`: the databases are in `aiida.out/`, the `SAVE` in
`SAVE/`. Each file is stored in the `retrieved` folder with its file name:

```python
settings = orm.Dict({'ADDITIONAL_RETRIEVE_LIST': ['aiida.out/ndb.dipoles', 'LOG/l-*_CPU_2']})
```

An entry can also be `[remote_path, local_name, depth]` as in AiiDA's `CalcInfo.retrieve_list`, to choose the name
or keep part of the folder structure. These files are always stored: there is no setting to retrieve other files
temporarily. If you need that, ask for the stored file and delete it afterwards, or open an issue.

Large files (for example `ndb.dipoles` or `ndb.BS_PAR_Q*` of a big system) fill the AiiDA repository: add only what
you will read, and `clean_workdir` does not remove files already retrieved.

!!! note "Fix in this version"
    `ADDITIONAL_RETRIEVE_LIST` used to be added to the retrieve list as one nested list, which AiiDA reads as a
    `[remote_path, local_name, depth]` triple: lists of paths did not work. Each entry is now added on its own, and
    an entry which is neither a path nor such a triple raises an error. The same holds for `YppCalculation`.

## Sizes

The size on disk of the stored nodes, which is what the raw files would have taken as well:

- `YamboLatticeData`: a few KB (only the irreducible k-points, symmetries and atoms are kept, not the G-vectors and
  the wavefunction grids of `ns.db1`).
- `YamboExcitonData` without eigenvectors: energies, residuals and (with `WRbsWF`) the transitions table: a few
  numbers per exciton and per transition.
- Eigenvectors: about `number of excitons x number of transitions x 8 bytes` (`complex64`). With 1000 excitons and
  100000 transitions this is about 800 MB: set `N_EXCITON_EIGENVECTORS`. The number of transitions is
  `spin x k-points x valence bands x conduction bands` of the BSE (`BSEBands`, `BSEQptR`...).

The nodes are not smaller than the netCDF files for the same numbers (floating-point data does not compress well):
what saves space is not storing the eigenvectors by default, choosing how many, and not keeping the raw files.
