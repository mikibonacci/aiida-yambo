# Installation

## Python package

```bash
python -m pip install aiida-yambo
```

For development:

```bash
git clone https://github.com/yambo-code/yambo-aiida.git
cd yambo-aiida
python -m pip install -e .
```

The package installs `aiida-quantumespresso` and the YAMBO Python helpers as dependencies. You still need working executables for Quantum ESPRESSO, `p2y`, and `yambo` on a local or remote computer.

## AiiDA profile

```bash
verdi quicksetup --profile yambo
verdi profile setdefault yambo
verdi daemon start
verdi status
```

For an existing profile, only the last two commands are normally needed.

## Documentation development

```bash
python -m pip install mkdocs-material
mkdocs serve
```

The site is generated from `aiida-yambo/docs/` and configured by `aiida-yambo/mkdocs.yml`.

!!! tip
    Keep the profile, computer, code labels, and pseudo-family in a small local configuration file. Do not commit site-specific paths or scheduler accounts.
