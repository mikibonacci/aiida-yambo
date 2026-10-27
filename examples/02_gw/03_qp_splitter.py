"""GW with the QP splitter: the quasiparticles are computed by several smaller Yambo runs, then merged.

Flow (from an NSCF parent):  p2y -> [ GW on a subset of the (k, band) pairs ] x N -> merge -> done

The QP splitter cuts the list of quasiparticles (QP) to compute in subsets of ``qp_per_subset`` (k, band) pairs,
runs ``parallel_runs`` of them at the same time (each with its own resources), and merges the databases with
``yambopy mergeqp`` into the output ``merged_QP``. It is useful when the full list is too large for one run,
or to put the resources where they are needed. Choose how the QP are selected with MODE:

    'boundaries'  all the k-points, the bands bi..bf
    'k_range'     the k-points ki..kf, the bands bi..bf (the other k-points are interpolated at the BSE step)
    'range_QP'    the bands within an energy window (eV) around mid-gap, for all the k-points
    'full_bands'  'range_QP', but every band between the lowest and the highest selected
    'explicit'    a list of [k1, k2, b1, b2] ranges
    'extend_db'   'boundaries', then an extended QP database (scissor + Fermi-Dirac smearing) is also built

Status: every MODE ran on a cluster (silicon, PseudoDojo PBE, NSCF parent).
Needs: ``yambopy`` (the ``yambopy mergeqp`` command) where the workflow runs.
Edit: the code labels, MODE, PARENT_PK (a finished NSCF), the resources of the splitted runs.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.yambowf import YamboWorkflow

load_profile()

# ---------------------------------------------------------------------------
# 1. Settings: edit them
# ---------------------------------------------------------------------------
YAMBO_CODE = 'yambo@mycluster'
P2Y_CODE = 'p2y@mycluster'
PW_CODE = 'pw@mycluster'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'
PARENT_PK = 1234  # a finished NSCF (or SCF, or Yambo run)
MODE = 'boundaries'
KMESH = [2, 2, 2]
NSCF_NBND = 76
QP_NB = 75
QP_PER_SUBSET = 2  # (k, band) pairs in each run
PARALLEL_RUNS = 1  # runs at the same time
SPLIT_RESOURCES = None  # e.g. {'num_machines': 1, 'num_mpiprocs_per_machine': 4}: resources of each splitted run

# What each MODE gives to `set_QP_subsets_dict` (bands and k-points are indices, starting from 1; Si has 2 valence bands)
MODES = {
    'boundaries': dict(bi=3, bf=6),
    'k_range': dict(ki=1, kf=1, bi=1, bf=4),
    'range_QP': dict(range_QP=5.0),
    'full_bands': dict(range_QP=5.0, full_bands=True),
    'explicit': dict(explicit=[[1, 1, 1, 2], [1, 1, 2, 2], [2, 2, 5, 5]]),
    'extend_db': dict(bi=2, bf=6, extend_db=True, consider_only=[4, 5]),
}

# ---------------------------------------------------------------------------
# 2. Structure (the one of the parent)
# ---------------------------------------------------------------------------
structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

# ---------------------------------------------------------------------------
# 3. Inputs
# ---------------------------------------------------------------------------
builder = YamboWorkflow.get_builder_from_protocol(
    structure=structure,
    code=orm.load_code(YAMBO_CODE),
    preprocessing_code=orm.load_code(P2Y_CODE),
    pw_code=orm.load_code(PW_CODE),
    pseudo_family=PSEUDO_FAMILY,
    calc_type='qp',
)
manager = YamboInputManager(**YamboInputManager.get_nested_inputs(builder, what='YamboWorkflow', mode='builder'))
manager.scf.set_kmesh(KMESH)
manager.nscf.set_kmesh(KMESH)
manager.set_nscf_nbnd(value=NSCF_NBND)
manager.qp.set_Nb(value=QP_NB)
manager.qp.set_Gcut(value=1)
manager.additional_parsing = ['gap_', 'homo', 'lumo']
manager.set_parent_folder(orm.load_node(PARENT_PK).outputs.remote_folder)

# The QP splitter: how many QP per run, how many runs at once, which QP.
options = dict(MODES[MODE])
if SPLIT_RESOURCES is not None:
    options['resources'] = SPLIT_RESOURCES  # resources of the splitted runs (not of the first, full run)
manager.set_QP_subsets_dict(qp_per_subset=QP_PER_SUBSET, parallel_runs=PARALLEL_RUNS, **options)

# ---------------------------------------------------------------------------
# 4. Scheduler options (see 01_gw_from_scratch.py)
# ---------------------------------------------------------------------------
manager.scf.set_resources(mpi=2)
manager.scf.set_npools(value=2)

# ---------------------------------------------------------------------------
# 5. Submit
# ---------------------------------------------------------------------------
node = submit(YamboWorkflow, **manager.generate_AiiDA_inputs(what='YamboWorkflow'))
print(f"GW with the QP splitter ({MODE}) submitted, pk = {node.pk}")
print(f'Monitor with:  verdi process status {node.pk}')

# ---------------------------------------------------------------------------
# 6. After it finishes
# ---------------------------------------------------------------------------
# node = orm.load_node(<PK>)
# node.outputs.splitted_QP_calculations  # the splitted YamboRestart (uuids)
# node.outputs.merged_QP                 # the merged ndb.QP (SinglefileData)
# node.outputs.extended_QP               # only with MODE = 'extend_db'
# print(node.outputs.output_ywfl_parameters.get_dict())
