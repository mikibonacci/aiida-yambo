"""Read the results of a finished ``YamboWorkflow`` (GW, BSE, BSE@GW): nothing is submitted.

Functions to copy into your own analysis: ``summary`` (what the workflow produced and the parsed quantities),
``qp_corrections`` (the GW corrections), ``spectrum`` (the BSE absorption) and ``plot_spectrum``.
Set ``PK`` and run it with ``verdi run``, or import the functions.

Status: tested here on fabricated nodes; the output names are the ones of the workflow spec.
"""
from aiida import load_profile, orm

load_profile()

PK = None  # the pk of a finished YamboWorkflow


def summary(node):
    """The state of the workflow, the outputs it has, and the parsed quantities (gap_, homo, lumo, lowest_exciton...)."""
    result = {
        'finished_ok': node.is_finished_ok,
        'exit_status': node.exit_status,
        'outputs': sorted(node.outputs),
    }
    if 'output_ywfl_parameters' in node.outputs:
        result['parameters'] = node.outputs.output_ywfl_parameters.get_dict()
    if 'nscf_mapping' in node.outputs:  # the DFT gap and what was predicted (insulator or metal)
        result['nscf_mapping'] = node.outputs.nscf_mapping.get_dict()
    return result


def qp_corrections(node):
    """The quasiparticle corrections: the array of the database (``E_minus_Eo``), or None if there are none."""
    if 'array_ndb' not in node.outputs:
        return None
    return node.outputs.array_ndb.get_array('E_minus_Eo')


def spectrum(node):
    """The BSE spectrum: the energies (eV) and the imaginary part of the dielectric function, or None."""
    if 'array_chi' not in node.outputs:
        return None
    chi = node.outputs.array_chi
    return chi.get_array('eV'), chi.get_array('eps_2')


def plot_spectrum(node, filename='spectrum.png'):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    energy, epsilon2 = spectrum(node)
    plt.plot(energy, epsilon2)
    plt.xlabel('energy (eV)')
    plt.ylabel(r'Im $\epsilon$')
    plt.savefig(filename, dpi=120)
    return filename


if PK is not None:
    workflow = orm.load_node(PK)
    print(summary(workflow))
    if spectrum(workflow) is not None:
        print('spectrum plotted in', plot_spectrum(workflow))
    # QP databases of the QP splitter: workflow.outputs.merged_QP / extended_QP (SinglefileData, an ndb.QP)
