"""Read the results of a finished ``YamboConvergence`` (or of a scan): nothing is submitted.

``story`` gives one row for each calculation as a table, ``converged_points`` and ``failed_points`` select rows,
``summary`` the converged values, and ``plot_story`` the quantity against a parameter. The outputs are returned
also when the convergence stops early or fails, so you can see what was computed.
Set ``PK`` and run it with ``verdi run``, or import the functions.

Status: tested here on fabricated nodes; the output names are the ones of the workflow spec.
"""
import pandas as pd
from aiida import load_profile, orm

load_profile()

PK = None  # the pk of a YamboConvergence
WHAT = 'gap_'  # the quantity to plot
VARIABLE = 'FFTGvecs'  # the parameter on the x axis


def story(node):
    """One row per calculation: the values of the variables, the quantities, ``failed``, ``useful``, ``converged``, pk."""
    return pd.DataFrame(node.outputs.collection.get_dict()['story'])


def converged_points(frame):
    return frame[frame['converged'] == True]  # noqa: E712 (the column can contain None)


def failed_points(frame):
    return frame[frame['failed'] == True]  # noqa: E712


def summary(node):
    """The converged value of each parameter group; for a scan, the number of points and of failures."""
    return node.outputs.convergence_summary.get_dict()


def plot_story(frame, what=WHAT, variable=VARIABLE, filename='story.png'):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    good = frame[~frame['failed'].astype(bool)]
    plt.plot(good[variable], good[what], 'o-')
    plt.xlabel(variable)
    plt.ylabel(what)
    plt.savefig(filename, dpi=120)
    return filename


if PK is not None:
    convergence = orm.load_node(PK)
    table = story(convergence)
    print(summary(convergence))
    print(table.drop(columns=['uuid'], errors='ignore'))
    print('failed:', list(failed_points(table)['pk']))
    # the exposed outputs of the last calculation, e.g. convergence.outputs.output_ywfl_parameters
