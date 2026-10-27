"""Run an example script with test nodes instead of a cluster.

The examples are linear scripts with their settings on top (`YAMBO_CODE = 'yambo@mycluster'`, `PARENT_PK = 1234`...).
Here they are executed for real with:

* the settings which point to the database (code labels, pk of a parent) replaced by test nodes;
* any other setting replaced on request (`overrides`), to test the variations a script offers (e.g. `MODE`);
* `aiida.engine.submit` replaced by a function which does not submit but validates the inputs against the
  process spec (the validation AiiDA does when submitting), so an API change which breaks an example fails a test.
"""
import pathlib
import re
import tempfile
from types import SimpleNamespace
from uuid import uuid4

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / 'examples'

PSEUDO_FAMILIES = ('PseudoDojo/0.4/PBE/SR/standard/upf', 'PseudoDojo/0.4/LDA/SR/standard/upf')


def _pseudo_families():
    """A fake PseudoDojo family (silicon only, a fake pseudopotential) for each label the examples use."""
    from aiida import orm
    from aiida_pseudo.data.pseudo import UpfData
    from aiida_pseudo.groups.family import PseudoDojoFamily

    content = '<UPF version="2.0.1"><PP_HEADER\nelement="Si"\nz_valence="4.0"\n/></UPF>\n'
    for label in PSEUDO_FAMILIES:
        try:
            orm.load_group(label)
            continue
        except Exception:
            pass
        with tempfile.TemporaryDirectory() as directory:
            (pathlib.Path(directory) / 'Si.upf').write_text(content)
            family = PseudoDojoFamily.create_from_folder(pathlib.Path(directory), label, pseudo_type=UpfData)
        family.set_cutoffs({'Si': {'cutoff_wfc': 30.0, 'cutoff_rho': 240.0}}, 'standard', unit='Ry')


def make_environment(computer):
    """Codes, a parent calculation (with a remote folder) and the pseudo families."""
    from aiida import orm
    from aiida.common.links import LinkType

    def code(name):
        label = f'{name}-{uuid4().hex[:8]}'
        orm.InstalledCode(label=label, computer=computer, filepath_executable='/bin/true').store()
        return f'{label}@{computer.label}'

    calculation = orm.CalcJobNode(computer=computer).store()
    remote = orm.RemoteData(computer=computer, remote_path='/tmp/parent')
    remote.base.links.add_incoming(calculation, link_type=LinkType.CREATE, link_label='remote_folder')
    remote.store()
    _pseudo_families()
    return SimpleNamespace(
        settings={'YAMBO_CODE': code('yambo'), 'P2Y_CODE': code('p2y'), 'PW_CODE': code('pw'), 'YPP_CODE': code('ypp'),
                  'PARENT_PK': calculation.pk, 'SCF_PK': calculation.pk},
        submitted=[],
    )


def install_submit(monkeypatch, submitted):
    """Replace `aiida.engine.submit` by a function validating the inputs of the process (nothing is run)."""
    from aiida.engine.utils import instantiate_process
    from aiida.manage import get_manager

    # the aiida_yambo modules must be imported before: they should not get the replacement of `submit`
    import aiida_yambo.workflows.yamboconvergence  # noqa: F401
    import aiida_yambo.workflows.yambowf  # noqa: F401

    def submit(process, **inputs):
        if hasattr(process, 'process_class'):  # a builder
            process_class, inputs = process.process_class, process._inputs(prune=True)
        else:
            process_class = process
        instantiated = instantiate_process(get_manager().get_runner(), process_class, **inputs)
        instantiated.close()
        submitted.append((process_class, inputs))
        return SimpleNamespace(pk=1, uuid='00000000-0000-0000-0000-000000000000')

    monkeypatch.setattr('aiida.engine.submit', submit)


def run_example(script, settings, overrides=None):
    """Execute the script with `settings` (only those the script defines) and `overrides` (all of them must be defined)."""
    path = EXAMPLES / script
    source = path.read_text()

    def replace(source, name, value, required):
        pattern = re.compile(rf'^{name} = .*$', re.M)
        if not pattern.search(source):
            assert not required, f'{script} does not define {name}'
            return source
        return pattern.sub(lambda _: f'{name} = {value!r}', source, count=1)

    for name, value in settings.items():
        source = replace(source, name, value, required=False)
    for name, value in (overrides or {}).items():
        source = replace(source, name, value, required=True)
    exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})
