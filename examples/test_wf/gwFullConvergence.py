from __future__ import absolute_import
from __future__ import print_function
from aiida.backends.utils import load_dbenv, is_dbenv_loaded
if not is_dbenv_loaded():
    load_dbenv()

from aiida_yambo.workflows.gwconvergence import YamboFullConvergenceWorkflow
from aiida.orm.data.base import Float, Str, NumericType, BaseType, List
from aiida.work.run import run, submit
from aiida.orm.utils import DataFactory
ParameterData = DataFactory("parameter")

calculation_set_yambo = {
    'resources': {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 32
    },
    'max_wallclock_seconds': 2 * 60 * 60,
    'max_memory_kb': 1 * 80 * 1000000,
    'custom_scheduler_commands': u"#PBS -A  Pra14_3622\n",
    'environment_variables': {
        "omp_num_threads": "0"
    }
}
calculation_set_pw = {
    'resources': {
        "num_machines": 2,
        "num_mpiprocs_per_machine": 32
    },
    'max_wallclock_seconds': 60 * 45 * 3,
    'max_memory_kb': 1 * 80 * 1000000,
    'custom_scheduler_commands': u"#PBS -A  Pra14_3622\n",
    'environment_variables': {
        "omp_num_threads": "0"
    }
}

if __name__ == "__main__":
    # verdi run  file.py --precode p2h@hyd   --yambocode yamb@hyd  --pwcode qe6.1@hyd --pseudo CHtest  --parent  637  --structure 7
    import argparse
    parser = argparse.ArgumentParser(description='GW QP calculation.')
    parser.add_argument(
        '--precode',
        type=str,
        dest='precode',
        required=True,
        help='The p2y codename to use')
    parser.add_argument(
        '--yambocode',
        type=str,
        dest='yambocode',
        required=True,
        help='The yambo codename to use')

    parser.add_argument(
        '--pwcode',
        type=str,
        dest='pwcode',
        required=True,
        help='The pw codename to use')
    parser.add_argument(
        '--pseudo',
        type=str,
        dest='pseudo',
        required=True,
        help='The pesudo  to use')
    parser.add_argument(
        '--structure',
        type=int,
        dest='structure',
        required=True,
        help='The structure  to use')
    parser.add_argument(
        '--parent',
        type=int,
        dest='parent',
        required=False,
        help='The parent SCF   to use')

    threshold = Float(0.1)
    args = parser.parse_args()
    structure = load_node(int(args.structure))
    extra = {}
    if args.parent:
        parent = load_node(args.parent)
        extra['parent_scf_folder'] = parent.out.remote_folder
    p2y_result = run(
        YamboFullConvergenceWorkflow,
        pwcode=Str(args.pwcode),
        precode=Str(args.precode),
        pseudo=Str(args.pseudo),
        yambocode=Str(args.yambocode),
        structure=structure,
        calculation_set=ParameterData(dict=calculation_set_yambo),
        calculation_set_pw=ParameterData(dict=calculation_set_pw),
        threshold=threshold,
        **extra)
    print(("workflow launched ", p2y_result))
