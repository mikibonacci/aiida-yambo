# -*- coding: utf-8 -*-
import copy
import logging
import pathlib
import subprocess
import tempfile
import warnings
from typing import Literal

import numpy as np
import xarray
from ase import units

from aiida import orm
from aiida.engine import ToContext, WorkChain, calcfunction, while_
from aiida.orm import load_node
from aiida.plugins import DataFactory
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin, recursive_merge
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_yambo.utils.common_helpers import (
    find_pw_parent, find_pw_type, gap_mapping_from_nscf, store_Dict, store_List, take_calc_from_remote,
)
from aiida_yambo.utils.high_level_API.general_model import ResourcesModel
# the new BaseModel manager for the inputs
from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.workflows.utils.extend_QPDB import FD_and_scissored_db
from aiida_yambo.workflows.utils.helpers_yambowf import (
    QP_analyzer, add_corrections, additional_parsed, check_already_computed,
)
from aiida_yambo.workflows.yamborestart import YamboRestart

LOGGER = logging.getLogger(__name__)

LegacyUpfData = DataFactory('core.upf')
SingleFileData = DataFactory('core.singlefile')



def clean(node):
    cleaned_calcs = []

    for called_descendant in node.called_descendants:
        if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    if not called_descendant.is_finished_ok:
                        called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                        cleaned_calcs.append(called_descendant.pk)
                except Exception as exc: # e.g. the connection to the computer
                    LOGGER.warning('could not clean the remote folder of calculation %s: %s', called_descendant.pk, exc)

    if cleaned_calcs:
        return "cleaned remote folders of calculations: {}".format(', '.join(map(str, cleaned_calcs)))

def sanity_check_QP(v,c,input_db,output_db,create=True,max_correction_eV=5):
    """Replace the QP energies whose correction is larger than `max_correction_eV` with a linear fit
    (valence and conduction separately) of the reliable ones, and align the valence band maximum to zero."""
    d = xarray.open_dataset(input_db,engine='netcdf4')
    d.load() # we modify it in place and close the file at the end
    wrong = np.where(abs(d.QP_E[:,0]-d.QP_Eo[:])*units.Ha>max_correction_eV)
    #v,c = 29,31
    v_cond = np.where((d.QP_table[0] <= v) & (abs(d.QP_E[:,0]-d.QP_Eo[:])*units.Ha<max_correction_eV))
    c_cond = np.where((d.QP_table[0] >= c) & (abs(d.QP_E[:,0]-d.QP_Eo[:])*units.Ha<max_correction_eV))

    #fix with a fit
    # a linear fit requires at least 2 points; if the band region is missing we return None for the fit.
    fit_v = np.polyfit(d.QP_Eo[v_cond[0]],d.QP_E[v_cond[0],0],deg=1) if len(v_cond[0]) >= 2 else None
    fit_c = np.polyfit(d.QP_Eo[c_cond[0]],d.QP_E[c_cond[0],0],deg=1) if len(c_cond[0]) >= 2 else None
    for i in wrong[0]:
        LOGGER.info('QP energy of band %s: DFT %s eV, QP %s eV', int(d.QP_table[0,i]), d.QP_Eo[i].data*units.Ha, d.QP_E[i,0].data*units.Ha)
        if d.QP_table[0,i]>v:
            fit = fit_c
        else:
            fit = fit_v
        if fit is None:
            LOGGER.warning('no %s fit available, leaving QP for band %s unchanged',
                'conduction' if d.QP_table[0,i]>v else 'valence', int(d.QP_table[0,i]))
            continue
        d.QP_E[i,0] = fit[0]*d.QP_Eo[i]+fit[1]
    
    #align to zero wrt to the maximum of valence... fixes the error in Fermi re-evaluation
    #in the BSE RD/ndb.QP. for now.
    if len(v_cond[0]) > 0:
        d.QP_E[:,0] = d.QP_E[:,0] - np.max(d.QP_E[v_cond[0],0])
    
    if create: d.to_netcdf(output_db)
    d.close()

    return output_db,fit_v,fit_c

def valence_conduction_bands(nelectrons, soc):
    """Return the (1-based) indices of the highest occupied and lowest empty band.

    With spin-orbit coupling each (spinor) band holds one electron, otherwise two.
    """
    if soc:
        valence = int(nelectrons)
    else:
        valence = int(nelectrons/2) + int(nelectrons%2)
    return valence, valence + 1

@calcfunction
def merge_QP(filenames_List,output_name,ywfl_pk,qp_settings,already_computed_QP_db=None): #just to have something that works, but it is not correct to proceed this way
        ywfl = load_node(ywfl_pk.value)
        pw = find_pw_parent(ywfl)
        SOC = pw.outputs.output_parameters.get_dict()['spin_orbit_calculation']
        nelectrons = pw.outputs.output_parameters.get_dict()['number_of_electrons']

        valence, conduction = valence_conduction_bands(nelectrons, SOC)
        databases = []
        
        # Create temporary directory
        filename='ndb.QP'
        with tempfile.TemporaryDirectory() as dirpath:
            # Open the output file from the AiiDA storage and copy content to the temporary file
            for i in filenames_List.get_list():
                # Create the file with the desired name
                temp_file = pathlib.Path(dirpath) / str(i)
                with load_node(i).outputs.QP_db.base.repository.open(filename, 'rb') as handle:
                    temp_file.write_bytes(handle.read())

                databases.append(str(temp_file))

            if already_computed_QP_db is not None:
                for i in already_computed_QP_db.get_list():
                    # Create the file with the desired name
                    temp_file = pathlib.Path(dirpath) / str(i)
                    temp_file.write_bytes(load_node(i).get_content("rb"))
                    databases.append(str(temp_file))
            
            merged = str(pathlib.Path(dirpath) / output_name.value)
            # NOTE: we wait for the end of the command, and we fail loudly if it fails.
            subprocess.run(['yambopy', 'mergeqp', *databases, '-o', merged], check=True)
            qp_fixed,fit_v,fit_c = sanity_check_QP(valence,conduction,merged,str(pathlib.Path(dirpath) / output_name.value.replace('merged','fixed')))
            QP_db = SingleFileData(qp_fixed)

            return {
                'QP_db': QP_db,
                'has_valence': orm.Bool(fit_v is not None),
                'has_conduction': orm.Bool(fit_c is not None),
            }

@calcfunction
def extend_QP(filenames_List,output_name,ywfl_pk,qp_settings,QP): #just to have something that works, but it is not correct to proceed this way
    ywfl = load_node(ywfl_pk.value)
    pw = find_pw_parent(ywfl)
    SOC = pw.outputs.output_parameters.get_dict()['spin_orbit_calculation']
    nelectrons = pw.outputs.output_parameters.get_dict()['number_of_electrons']
    nk = pw.outputs.output_parameters.get_dict()['number_of_k_points']
    qp_rules = qp_settings.get_dict()

    valence, conduction = valence_conduction_bands(nelectrons, SOC)
    #output_name = QP._repository._repo_folder.abspath + '/path/ndb.QP_fixed'
    with tempfile.TemporaryDirectory() as dirpath:
        # Open the output file from the AiiDA storage and copy content to the temporary file
        output_name = pathlib.Path(dirpath) / 'ndb.QP_fixed'
        with QP.base.repository.open('ndb.QP_fixed', 'rb') as handle:
            output_name.write_bytes(handle.read())

        qp_fixed,fit_v,fit_c = sanity_check_QP(valence,conduction,output_name,output_name,create=False)
    
        if qp_rules.pop('extend_db', False):
            # In the qp settings dict, I should add:
            #    {
            #        'extend_db':True,
            #        'T_smearing': 1e-2, #smearing for the FD corrections...see the paper MBonacci et al. Towards HT... 
            #        'consider_only':[v_min,c_max]
            #        'Nb': n, #number of bands to be included in the extended QP db (from 1st to nth)
            #        -->'v_min':, #used to evaluate v_max energy and v_min that you want to compute explicitly
            #        --->'c_max':, #used to evaluate c_min energy and c_max that you want to compute explicilty
            #    }
            qp_rules['Nb'] = qp_rules.pop('Nb', [1, conduction + valence])
            db_FD_scissored = FD_and_scissored_db(out_db_path=qp_fixed,pw=pw,Nb=qp_rules['Nb'],Nk=nk,v_max=min(qp_rules['consider_only']),c_min=max(qp_rules['consider_only']),fit_v=fit_v,
                    fit_c=fit_c,conduction=conduction,T=qp_rules.pop('T_smearing',1e-2))
            db_FD_scissored.to_netcdf(str(output_name).replace('fixed','extended'))
            
            QP_db_extended = SingleFileData(str(output_name).replace('fixed','extended'))
            return QP_db_extended

QP_SUBSETS_KEYS = {
    'parallel_runs', 'qp_per_subset', 'boundaries', 'range_QP', 'full_bands', 'range_spectrum', 'explicit',
    'consider_only', 'parallelism', 'resources', 'prepend_text', 'extend_db', 'Nb', 'T_smearing',
    'max_wallclock_seconds', 'queue_name', 'clean_workdir', 'custom_scheduler_commands',
}
QP_SUBSETS_SOURCES = ('boundaries', 'range_QP', 'explicit')

def validate_QP_subsets_dict(QP_subsets_dict):
    """Raise a ValueError if `QP_subsets_dict` has unknown keys or more than one source of subsets."""
    unknown = sorted(set(QP_subsets_dict) - QP_SUBSETS_KEYS)
    if unknown:
        raise ValueError(f'Unknown keys in QP_subsets_dict: {unknown}. Allowed: {sorted(QP_SUBSETS_KEYS)}')
    parallelism = QP_subsets_dict.get('parallelism')
    if parallelism is not None and parallelism != 'auto' and not isinstance(parallelism, dict):
        raise ValueError(f"QP_subsets_dict['parallelism'] must be a dict of yambo variables or 'auto', got {parallelism!r}")
    sources = [key for key in QP_SUBSETS_SOURCES if key in QP_subsets_dict]
    if len(sources) > 1:
        raise ValueError(f'QP_subsets_dict: give only one of {QP_SUBSETS_SOURCES}, got {sources}')

def set_QP_splitter_options(yambo_manager, QP_subsets_dict):
    """Set on the splitted QP runs the options requested in `QP_subsets_dict`.

    `resources` is merged over the resources already set (a partial dictionary does not reset the other values to 1),
    `OMP_NUM_THREADS` follows `num_cores_per_mpiproc`, `max_wallclock_seconds` and `queue_name` are set if given, and
    the remote folders of the splitted runs are cleaned unless `clean_workdir` is False.
    """
    options = yambo_manager.metadata.options
    if 'resources' in QP_subsets_dict:
        merged = {**options.resources.model_dump(), **QP_subsets_dict['resources']}
        options.resources = ResourcesModel(**merged)
        options.prepend_text = 'export OMP_NUM_THREADS={}\n'.format(options.resources.num_cores_per_mpiproc) + options.prepend_text
    for key in ('max_wallclock_seconds', 'queue_name'):
        if key in QP_subsets_dict:
            setattr(options, key, QP_subsets_dict[key])
    if 'prepend_text' in QP_subsets_dict:
        options.prepend_text += QP_subsets_dict['prepend_text']
    if 'custom_scheduler_commands' in QP_subsets_dict:
        options.custom_scheduler_commands += QP_subsets_dict['custom_scheduler_commands']
    # NOTE: no AiiDA node here, generate_AiiDA_inputs creates it
    yambo_manager.clean_workdir = QP_subsets_dict.get('clean_workdir', True)

def set_auto_QP_parallelism(yambo_manager, subset, bands, occupied, kpoints, resources):
    """Set the parallelism of a splitted QP run with `find_parallelism_qp`, for the QP of its `subset`.

    `subset` is a list of [k1, k2, b1, b2] ranges, `bands` the total number of bands, `resources` the ones available (a dict): the solver distributes
    the MPI tasks (it can use fewer of them if the QP are not enough to feed all of them). Starting always from the
    same `resources`, a small subset does not reduce the resources of the following ones.
    """
    from aiida_yambo.utils.parallelism_finder import find_parallelism_qp

    n_qp = sum((k2 - k1 + 1) * (b2 - b1 + 1) for k1, k2, b1, b2 in subset)
    last_band = max(b2 for _, _, _, b2 in subset)
    options = yambo_manager.metadata.options
    parallelism, new_resources = find_parallelism_qp(
        resources['num_machines'], resources['num_mpiprocs_per_machine'], resources['num_cores_per_mpiproc'],
        bands, occupied, n_qp, kpoints, last_band)
    for name, value in parallelism.items():
        yambo_manager.set_variable(name=name, value=value)
    options.resources = ResourcesModel(**{**resources, **new_resources})
    return parallelism

def QP_mapper(ywfl,tol=1,full_bands=False,spectrum_tol=1,max_iterations=30):
    pw = find_pw_parent(ywfl)
    fermi = pw.outputs.output_parameters.get_dict()['fermi_energy']
    SOC = pw.outputs.output_parameters.get_dict()['spin_orbit_calculation']
    nelectrons = pw.outputs.output_parameters.get_dict()['number_of_electrons']
    kpoints = pw.outputs.output_band.get_kpoints()
    bands = pw.outputs.output_band.get_bands()
    
    valence, conduction = valence_conduction_bands(nelectrons, SOC)
    LOGGER.info('valence: %s', valence)
    
    #tol = 10*(-max(bands[:,valence-1])+(min(bands[:,conduction-1])))/2
    mid_gap_energy = max(bands[:,valence-1])+(min(bands[:,conduction-1])-max(bands[:,valence-1]))/2
    
    # each time the selection is not good enough we increase the energy range of 50%.
    for _ in range(max_iterations):
        QP = []
        selected = np.where(abs(bands-mid_gap_energy)<tol)
        LOGGER.debug('tol=%s, selected bands: %s', tol, selected[1])
        if len(selected[1])<2:
            LOGGER.info('#1 redoing analysis incrementing the energy range of 50%%. new tol=%s', tol*1.5)
            tol=tol*1.5
            continue
        
        if not full_bands:
            for i,j in zip(selected[0],selected[1]):
                QP.append([i+1,i+1,j+1,j+1])
        else:
            b_min = selected[1].min()
            b_max = selected[1].max()
            for i in range(len(kpoints)):
                QP.append([i+1,i+1,b_min+1,b_max+1])
                    
        v,c = False,False
        if len(QP)<1:
            LOGGER.info('#2 redoing analysis incrementing the energy range of 50%%. new tol=%s', tol*1.5)
            tol=tol*1.5
            continue
        
        for i in QP:
            if valence >= i[-1]: v = True
            if conduction <= i[-1]: c = True
            if valence >= i[-2]: v = True
            if conduction <= i[-2]: c = True
        
        if not v or not c:
            LOGGER.info('#3 redoing analysis incrementing the energy range of 50%%. new tol=%s', tol*1.5)
            tol=tol*1.5
            continue
            
        LOGGER.info('Found %s QPs', len(QP))
        LOGGER.info('Fermi level=%s eV', fermi)

        # 1-based band indices, as yambo wants them (they are used for BSEBands)
        b_min_scissored = np.where(abs(bands-mid_gap_energy)<spectrum_tol)[1].min() + 1
        b_max_scissored = np.where(abs(bands-mid_gap_energy)<spectrum_tol)[1].max() + 1
        
        return QP, [b_min_scissored,b_max_scissored]

    raise ValueError('QP_mapper: no QP selection containing both valence and conduction states found after {} enlargements of the energy range.'.format(max_iterations))

def QP_subset_groups(nnk_i,nnk_f,bb_i,bb_f,qp_per_subset):
    
    groups, L = [],[]
    
    for k in range(nnk_i,nnk_f+1):
        for b in range(bb_i,bb_f+1):
            L.append([k,k,b,b])
            
            if len(L)==qp_per_subset:
                groups.append(L)
                L=[]
    if len(L)>0:
        if len(L[0])>0:
            groups.append(L)
    return groups

def QP_list_merger(l=None,qp_per_subset=10,consider_only=None,already_computed_QP_db:list=None):
    l = l or []
    consider_only = consider_only or [-1]
    
    subgroup = []
    groups = []
    for qp_set in l:
        for k in list(range(qp_set[0],qp_set[1]+1)):
            for b in list(range(qp_set[2],qp_set[3]+1)):
                
                # skip the QP if it is already in at least one of the databases
                skip = False
                if already_computed_QP_db is not None:
                    skip = any(check_already_computed(QP_db, b, k) for QP_db in already_computed_QP_db)
                
                if not skip:
                    if (b in consider_only) or (consider_only[0]==-1):
                        subgroup.append([k,k,b,b])
                        if len(subgroup)==qp_per_subset:
                            groups.append(subgroup)
                            subgroup=[]

    if len(subgroup)<=qp_per_subset and len(subgroup)>0:
        groups.append(subgroup)
            
    return groups


class YamboWorkflow(ProtocolMixin, WorkChain):

    """This workflow will perform yambo calculation on the top of scf+nscf or from scratch,
        using also the PwBaseWorkChain.
    """
    pw_exclude = ['parent_folder', 'pw.parameters','pw.pseudos','pw.code','pw.structure', 'kpoints']

    @classmethod
    def define(cls, spec):
        """Workfunction definition

        """

        super(YamboWorkflow, cls).define(spec)

        spec.expose_inputs(PwBaseWorkChain, namespace='scf', namespace_options={'required': False,'populate_defaults': False}, 
                            exclude = ['pw.parent_folder'])

        spec.expose_inputs(PwBaseWorkChain, namespace='nscf', namespace_options={'required': False,'populate_defaults': False},
                            exclude = ['pw.parent_folder'])

        #spec.expose_inputs(YamboRestart, namespace='yres', namespace_options={'required': False,'populate_defaults': False}, 
        #                    exclude = ['parent_folder'])
        
        spec.expose_inputs(YamboRestart, namespace='qp', 
                           namespace_options={'required': False,'populate_defaults': False, 
                                              'help': 'inputs for GW and QP if BSE@GW is asked, i.e. we also provide `QP_subsets_dict`'}, 
                            exclude = ['parent_folder'])
        
        spec.expose_inputs(YamboRestart, namespace='bse', 
                           namespace_options={'required': False,'populate_defaults': False, 
                                              'help': 'inputs for BSE.`'}, 
                            exclude = ['parent_folder'])
        
        spec.input("additional_parsing", valid_type=orm.List, required = False,
                    help = 'list of additional quantities to be parsed: gap, homo, lumo, or used defined quantities -with names-[k1,k2,b1,b2], [k1,b1], gap_GG, lowest_exciton')
        
        spec.input("QP_subsets_dict", valid_type=orm.Dict, required = False,
                    help = 'QP_subsets_dict: how to define which quasi-particles (QP) to compute.\n'
                    'The QP splitter runs `parallel_runs` YamboRestart calculations at a time, each\n'
                    'on a subset of the requested QP range (set as QPkrange), then merges all the\n'
                    'ndb.QP databases with yambopy (`merge_QP`). `QP_subsets_dict` defines both the\n'
                    'subsets and the parallelism.\n'
                    '\n'
                    'REQUIRED KEYS (always):\n'
                    '    parallel_runs : int\n'
                    '        how many YamboRestart to submit at once; the splitter iterates until\n'
                    '        all subsets are consumed.\n'
                    '    qp_per_subset : int\n'
                    '        how many QP (k-point, band) pairs per subset.\n'
                    '\n'
                    'SUBSET SOURCES (provide exactly one):\n'
                    '    1. boundaries : dict, explicit band range for ALL k-points\n'
                    '           { \'bi\': int, \'bf\': int, \'ki\': int (opt, default 1),\n'
                    "             'kf': int (opt, default nk) }\n"
                    '       bi/bf are required; ki/kf default to the full k-grid. Equivalent to\n'
                    '       QP_list_merger([[ki, kf, bi, bf]], qp_per_subset, ...).\n'
                    '\n'
                    '    2. range_QP : float (eV), automatic band selection around the gap\n'
                    '       Uses QP_mapper() to pick the bands within `range_QP` of mid-gap for all\n'
                    "       k-points; populates the 'explicit' and 'scissored' keys automatically.\n"
                    '           - full_bands (opt): also run QP on all bands between the selected\n'
                    '             b_min and b_max (QP_mapper full_bands=True).\n'
                    '           - range_spectrum (opt): eV window used to define the scissored\n'
                    '             [b_min, b_max] range used later for BSE BSEBands.\n'
                    '\n'
                    '    3. explicit : list of [k1, k2, b1, b2] ranges\n'
                    '       Directly specify the QP windows; merged into subsets with QP_list_merger.\n'
                    '\n'
                    "    If none of the three is given, the full range of bands (1 to Nb) of all the\n"
                    '    k-points is used. Giving more than one raises an error.\n'
                    '\n'
                    'OPTIONAL KEYS:\n'
                    '    consider_only : list of bands, default [-1] (all)\n'
                    '        only include QP whose band is in the list (or all if [-1]).\n'
                    '    parallelism : dict\n'
                    '        extra yambo parallelism variables set on each splitted run\n'
                    '    resources : dict\n'
                    '        num_machines, num_mpiprocs_per_machine, num_cores_per_mpiproc for the splitted\n'
                    '        runs; the values not given are kept from the previous step. OMP_NUM_THREADS\n'
                    '        is exported accordingly.\n'
                    '    max_wallclock_seconds, queue_name : walltime and queue of the splitted runs.\n'
                    '    prepend_text, custom_scheduler_commands : str\n'
                    '        added to the ones of the splitted runs (modules, #SBATCH lines...).\n'
                    '    clean_workdir : bool, default True\n'
                    '        clean the remote folders of the splitted runs once finished.\n'
                    '    Unknown keys raise an error.\n'
                    '    extend_db : bool, default False\n'
                    '        if True, after the merge, run extend_QP() to build a scissored +\n'
                    '        Fermi-Dirac smeared extended QP database; extra keys:\n'
                    '            Nb          : [1, b_max] band range for the extended db\n'
                    '                          (default [1, valence + conduction])\n'
                    '            T_smearing  : smearing in eV for the FD corrections\n'
                    '                          (default 1e-2)\n'
                    '            consider_only : [v_min, c_max] used to set the scissor window\n'
                    '    already_computed_QP_db (workflow input, not a dict key):\n'
                    '        orm.List of ndb.QP database pks already computed; those QP are skipped\n'
                    '        when building the subsets (see check_already_computed).\n'
                    '\n'
                    'EXAMPLES:\n'
                    "    # all k-points, bands bi..bf, 10 QP per subset, 3 runs at a time:\n"
                    "    {'boundaries': {'bi': 29, 'bf': 32}, 'parallel_runs': 3, 'qp_per_subset': 10}\n"
                    '\n'
                    "    # automatic selection within 1.2 eV of mid-gap:\n"
                    "    {'range_QP': 1.2, 'parallel_runs': 3, 'qp_per_subset': 10}\n"
                    '\n'
                    "    # explicit windows:\n"
                    "    {'explicit': [[1, 1, 29, 29], [1, 1, 30, 30]], 'parallel_runs': 1,\n"
                    "     'qp_per_subset': 1}")

        spec.input("parent_folder", valid_type=orm.RemoteData, required = False,
                    help = 'scf, nscf or yambo remote folder')
        
        spec.input("clean_failed", valid_type=orm.Bool, default=lambda: orm.Bool(False))

        spec.input("already_computed_QP_db", valid_type=orm.List, required=False,
                   help="list of qp db pks already computed, so we don't need to compute all the QP in QP splitter")

        spec.input(
            "stop_if_metal",
            valid_type=bool,
            required=False,
            default=lambda: False,
            non_db=True,
            help="stop after NSCF if the DFT band analysis predicts a metallic system. "
                 "NOTE: being `non_db`, it is not stored in the provenance and it is not restored by "
                 "`get_builder_restart`, so YamboConvergence does not forward it to the YamboWorkflows it submits.",
        )
  

##################################### OUTLINE ####################################

        spec.outline(
            #cls.validate_parameters,
            cls.setup,
            while_(cls.can_continue)(
                    cls.check_nscf_metallicity,
                    cls.perform_next,
            ),
            cls.report_wf,
        )

##################################################################################

        spec.expose_outputs(YamboRestart)
        
        spec.output('output_ywfl_parameters', valid_type = orm.Dict, required = False)
        spec.output('nscf_mapping', valid_type = orm.Dict, required = False)

        spec.output('splitted_QP_calculations', valid_type = orm.List, required = False)
        spec.output('merged_QP', valid_type = SingleFileData, required = False)
        spec.output('extended_QP', valid_type = SingleFileData, required = False)
        
        spec.output('scissor', valid_type = orm.List, required = False)

        spec.exit_code(300, 'ERROR_WORKCHAIN_FAILED',
                             message='The workchain failed with an unrecoverable error.')
        spec.exit_code(301, 'ERROR_SPLITTED_QP_FAILED',
                             message='At least one of the splitted QP calculations failed.')
        spec.exit_code(302, 'ERROR_MISSING_BAND_REGION',
                             message='The merged QP database does not contain both valence and conduction states.')
        spec.exit_code(303, 'ERROR_METALLIC_SYSTEM',
                             message='The NSCF calculation predicts a metallic system.')
    
    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from aiida_yambo.workflows.protocols import yambo as yamboworkflow_protocols
        return files(yamboworkflow_protocols) / 'yamboworkflow.yaml'
    
    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code: orm.Code,
        preprocessing_code: orm.Code,
        code: orm.Code,
        protocol_qe='moderate',
        protocol='moderate',
        calc_type:Literal['qp', 'bse', 'bse@qp']='qp',
        structure=None,
        overrides=None,
        parent_folder=None,
        NLCC=False,
        RIM_v=False,
        RIM_W=False,
        electronic_type=ElectronicType.INSULATOR,
        spin_type=SpinType.NONE,
        initial_magnetic_moments=None,
        pseudo_family = None,
        force_symmorphic = True,
        **_
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        # the codes can be given also by label or pk
        if isinstance(pw_code, (str, int)):
            pw_code = orm.load_code(pw_code)
        if isinstance(preprocessing_code, (str, int)):
            preprocessing_code = orm.load_code(preprocessing_code)
        if isinstance(code, (str, int)):
            code = orm.load_code(code)

        # we never modify the dictionary of the caller
        overrides = copy.deepcopy(overrides) if overrides else {}

        if electronic_type not in [ElectronicType.METAL, ElectronicType.INSULATOR]:
            raise NotImplementedError(f'electronic type `{electronic_type}` is not supported.')

        if spin_type not in [SpinType.NONE, SpinType.COLLINEAR]:
            raise NotImplementedError(f'spin type `{spin_type}` is not supported.')
        
        builder = cls.get_builder()

        overrides_scf = overrides.pop('scf',{})
        overrides_nscf = overrides.pop('nscf',{})
        # `qp` and `bse` are the overrides of the corresponding YamboRestart; the old `yres` key applies to both.
        overrides_yres = overrides.pop('yres',{})
        if overrides_yres:
            warnings.warn("the `yres` overrides are deprecated, use `qp` and/or `bse`.", DeprecationWarning, stacklevel=2)
        overrides_qp = recursive_merge(overrides_yres, overrides.pop('qp',{}))
        overrides_bse = recursive_merge(overrides_yres, overrides.pop('bse',{}))

        for override in [overrides_scf,overrides_nscf]:
            override['clean_workdir'] = override.pop('clean_workdir',False) #required to have a valid parent folder

        if pseudo_family:
            overrides_scf['pseudo_family'] = pseudo_family
            overrides_nscf['pseudo_family'] = pseudo_family
                    
        ######### SCF and NSCF INPUTS/PROTOCOLS #########
        
        # try to get the pw builders from the scf and nscf parents if any
        builder_scf, builder_nscf = None, None
        if parent_folder is not None:
            from aiida_yambo.utils.common_helpers import get_pw_inputs_builders
            builder_scf, builder_nscf = get_pw_inputs_builders(parent_folder)
            builder.parent_folder = parent_folder
            
        if builder_scf:
            builder.scf = builder_scf
        else:
            builder.scf = PwBaseWorkChain.get_builder_from_protocol(
                pw_code,
                structure,
                protocol=protocol_qe,
                overrides=overrides_scf,
                electronic_type=electronic_type,
                spin_type=spin_type,
                initial_magnetic_moments=initial_magnetic_moments,
                pseudo_family=pseudo_family,
            )

        if builder_nscf:
            builder.nscf = builder_nscf
        else:
            builder.nscf = PwBaseWorkChain.get_builder_from_protocol(
                pw_code,
                structure,
                protocol=protocol_qe,
                overrides=overrides_nscf,
                electronic_type=electronic_type,
                spin_type=spin_type,
                initial_magnetic_moments=initial_magnetic_moments,
                pseudo_family=pseudo_family,
            )
        
        nelectrons = 0
        for site in builder.nscf['pw']['structure'].sites:
            nelectrons += builder.nscf['pw']['pseudos'][site.kind_name].z_valence
        ecutwfc = builder.nscf['pw']['parameters'].get_dict()['SYSTEM']['ecutwfc']

        #########  YAMBO PROTOCOL, with or without parent folder #########
        
        #### AAA: for now, the BSE post GW (all done here) should run setting manually the inputs bse.
        
        qp_builder = YamboRestart.get_builder_from_protocol(
                        preprocessing_code=preprocessing_code,
                        code=code,
                        protocol=protocol,
                        parent_folder=parent_folder,
                        overrides=overrides_qp,
                        NLCC=NLCC,
                        RIM_v=RIM_v,
                        RIM_W=RIM_W,
                        nelectrons=nelectrons,
                        ecutwfc=ecutwfc
                    )

        bse_builder = YamboRestart.get_builder_from_protocol(
                preprocessing_code=preprocessing_code,
                code=code,
                protocol='bse_'+protocol,
                parent_folder=parent_folder,
                overrides=overrides_bse,
                NLCC=NLCC,
                RIM_v=RIM_v,
                RIM_W=RIM_W,
                nelectrons=nelectrons,
                ecutwfc=ecutwfc
            )
            
        match calc_type:
            case 'qp':   
                builder.qp = qp_builder  
            case 'bse':   
                builder.bse = bse_builder
                
            case 'bse@qp':
                builder.qp = qp_builder
                builder.bse = bse_builder
                builder.QP_subsets_dict = orm.Dict({
                    'parallel_runs':3,
                    'qp_per_subset': 10,
                })
                

        # initialization of the YamboInputManager object
        nested_inputs = YamboInputManager.get_nested_inputs(builder,what='YamboWorkflow', mode='builder')
        input_manager = YamboInputManager(**nested_inputs)
                
        # setting the ecutrho in case it was not correct. In the end, we always have NC:
        input_manager.scf.set_pw_parameter(namelist='SYSTEM', key='ecutrho', value=4*input_manager.ecutwfc)
        input_manager.nscf.set_pw_parameter(namelist='SYSTEM', key='ecutrho', value=4*input_manager.ecutwfc)
        
        # setting force_symmorphic
        if force_symmorphic:
            input_manager.scf.set_pw_parameter(namelist='SYSTEM', key='force_symmorphic', value=True)
            input_manager.nscf.set_pw_parameter(namelist='SYSTEM', key='force_symmorphic', value=True)
        
        # setting Gamma only if molecule:
        molecule = False
        if protocol == 'molecule' or not any(builder.nscf['pw']['structure'].pbc): molecule=True
        if molecule:
            input_manager.set_gamma_only()
            
        # setting `nscf` and `wf_collect`:
        input_manager.nscf.set_pw_parameter(namelist='CONTROL', key='calculation', value='nscf')
        input_manager.nscf.set_pw_parameter(namelist='CONTROL', key='wf_collect', value=True)
        
        #safety measure, for some system creates chaos in conjunction with smearing
        input_manager.scf.parameters['SYSTEM'].pop('nbnd',0)

        new_builder = input_manager.populate_builder(builder)
        
        return new_builder
        
    def setup(self):
        """Initialize the workflow, set the parent calculation

        This function sets the parent, and its type
        there is no submission done here, only setting up the neccessary inputs the workchain needs in the next
        steps to decide what are the subsequent steps"""
        
        self.ctx.all_calcs_to_do = ['scf', 'nscf', 'qp', 'QP splitter', 'bse']
    
        input_manager = YamboInputManager.get_nested_inputs(
            self.inputs, what='YamboWorkflow', mode='builder'
        )
        self.ctx.input_manager = YamboInputManager(**input_manager)
        
        self.ctx.number_of_subsets = 0
        self.ctx.should_run_qp = False
        self.ctx.should_run_QP_splitter = False
        self.ctx.merge_QP_needed = False
        self.ctx.QP_db = None
        self.ctx.should_run_bse = False
        self.ctx.just_initialise = self.ctx.input_manager.qp.settings.get('INITIALISE', False) or self.ctx.input_manager.bse.settings.get('INITIALISE', False)
                       
        # Determine workflow mode
        has_qp_inputs = len(self.ctx.input_manager.qp.parameters) > 0
        has_bse_inputs = len(self.ctx.input_manager.bse.parameters) > 0
        has_qp_subsets = len(self.ctx.input_manager.QP_subsets_dict) > 0
        just_init = self.ctx.just_initialise

        if has_qp_inputs:
            self.ctx.should_run_qp = True
        if has_qp_subsets:
            self.ctx.should_run_QP_splitter = True
        if has_bse_inputs:
            self.ctx.should_run_bse = True

        if just_init:
            # pre-calculation "up to p2y": run only the FIRST yambo step, which
            # carries INITIALISE=True and thus performs p2y + yambo initialization
            # only. The QP splitter and the trailing qp/bse step must not run: with
            # a yambo parent the INITIALISE flag is bypassed (calculations/yambo.py)
            # and a real qp/bse would be launched without QP corrections.
            self.ctx.should_run_QP_splitter = False
            if has_qp_inputs:
                self.ctx.should_run_qp = True
                self.ctx.should_run_bse = False
            else:
                self.ctx.should_run_qp = False
                self.ctx.should_run_bse = True
        
        if not hasattr(self.ctx.input_manager, 'parent_folder'):
            self.report('No parent folder, we start from scratch.')
        elif self.ctx.input_manager.parent_folder is None:
            self.report('No parent folder, we start from scratch.')
        else:
            parent = take_calc_from_remote(self.ctx.input_manager.parent_folder,level=-1)
            # we get the PwCalculation
            if parent.process_type=='aiida.workflows:quantumespresso.pw.base':
                parent = parent.outputs.remote_folder.creator
            
            if parent.outputs.remote_folder.is_empty:
                # empty parent, we need to recompute it: we keep all the calculations to do.
                self.report('The parent remote folder is empty, we start from scratch.')
                self.ctx.input_manager.remove_parent_folder()

            # pw parent
            elif 'quantumespresso.pw' in parent.process_type:
                parent_params = parent.inputs.parameters.get_dict()
                calc_type = parent_params['CONTROL']['calculation']
                if calc_type in ['scf','relax','vc-relax']:
                    self.ctx.all_calcs_to_do.remove('scf') # scf
                elif calc_type in ['nscf']:
                    self.ctx.all_calcs_to_do.remove('scf')
                    self.ctx.all_calcs_to_do.remove('nscf')
                self.ctx.calc = parent

            # yambo parent
            elif parent.process_type=='aiida.calculations:yambo.yambo':
                self.ctx.all_calcs_to_do.remove('scf')
                self.ctx.all_calcs_to_do.remove('nscf')
                self.ctx.calc = parent
            else:
                self.report('no valid input calculations, so we will start from scratch')
                self.ctx.calc = parent
            
        # check nbnd and overall NSCF validity (beyond nbnd, also kpoints/structure/pseudos/ecut)
        # has_enough_nbnd is the fast nbnd-only gate; is_nscf_valid does the full check
        nscf_valid = True
        try:
            # check against the current parent calc if any
            if hasattr(self.ctx, 'calc') and self.ctx.calc is not None:
                from aiida_yambo.utils.common_helpers import is_nscf_valid
                nscf_valid = is_nscf_valid(parent_node=self.ctx.calc, input_manager=self.ctx.input_manager)
        except Exception as exc:
            self.report(f'Could not check the validity of the NSCF ({type(exc).__name__}: {exc}), checking only its number of bands.')
            nscf_valid = self.ctx.input_manager.has_enough_nbnd

        if not self.ctx.input_manager.has_enough_nbnd or not nscf_valid:
            if not self.ctx.input_manager.has_enough_nbnd:
                self.report(f'Setting nbnd={self.ctx.input_manager.Nb} in the NSCF step, not enough bands: {self.ctx.input_manager.nscf_nbnd}<{self.ctx.input_manager.Nb}')
                self.ctx.input_manager.set_nscf_nbnd(self.ctx.input_manager.Nb)
            else:
                self.report(f'NSCF invalid for current inputs (kpoints/structure/pseudos/ecut mismatch) — recomputing NSCF')
            # we redo nscf if not enough bands (or invalid), we cannot proceed with yambo or QP_splitter:
            if self.ctx.all_calcs_to_do[0] in ['qp', 'QP splitter', 'bse']:
                self.report(f"Recomputing NSCF step with nbnd={self.ctx.input_manager.nscf_nbnd}")
                if 'scf' in self.ctx.all_calcs_to_do: self.ctx.all_calcs_to_do.remove('scf')
                # re-insert nscf at front (after scf if it remains, else at 0)
                if 'nscf' not in self.ctx.all_calcs_to_do:
                    insert_idx = self.ctx.all_calcs_to_do.index('scf') + 1 if 'scf' in self.ctx.all_calcs_to_do else 0
                    self.ctx.all_calcs_to_do.insert(insert_idx, 'nscf')
                # rewire parent to SCF remote (not the stale NSCF/Yambo one)
                try:
                    scf_parent = find_pw_parent(self.ctx.calc, calc_type=['scf'])
                    self.ctx.input_manager.set_parent_folder(scf_parent.outputs.remote_folder)
                    self.ctx.calc = scf_parent
                except Exception as exc:
                    self.report(f'Could not find the SCF parent ({type(exc).__name__}: {exc}), we keep the current parent.')
            
        self.ctx.splitted_QP = []
        self.ctx.qp_splitter = 0
        
        if not self.ctx.should_run_qp:
            self.ctx.all_calcs_to_do.remove('qp')
        if not self.ctx.should_run_QP_splitter:
            self.ctx.all_calcs_to_do.remove('QP splitter')
        if not self.ctx.should_run_bse:
            self.ctx.all_calcs_to_do.remove('bse')
            
        self.ctx.calc_to_do=self.ctx.all_calcs_to_do[0]
        self.report("Workflow initilization step completed.")

    def can_continue(self):

        """This function checks the status of the last calculation and determines what happens next, including a successful exit.
        
        """
        
        if self.ctx.calc_to_do == 'QP splitter' and self.ctx.merge_QP_needed:
            self.report('The workflow continues with a merge QP run')
            self.ctx.all_calcs_to_do = ['merge QP'] + self.ctx.all_calcs_to_do
            return True
        elif len(self.ctx.all_calcs_to_do) != 0:
            self.report('The workflow continues with a {} calculation'.format(self.ctx.all_calcs_to_do[0]))
            return True
        elif len(self.ctx.all_calcs_to_do) == 0:
            self.report('The workflow is finished')
            return False

    @staticmethod
    def _resolve_nscf_calculation(node):
        """Return the NSCF calculation associated with a workflow or Yambo node."""

        try:
            if node.process_type == 'aiida.calculations:quantumespresso.pw':
                return node if find_pw_type(node) == 'nscf' else None
            return find_pw_parent(node, calc_type=['nscf'])
        except Exception as exc:
            LOGGER.debug('no NSCF found for node %s: %s', node.pk, exc)
            return None

    def check_nscf_metallicity(self):
        """Analyse the NSCF bands and optionally stop before any Yambo run."""

        if hasattr(self.ctx, 'mapping'):
            return

        # In a fresh workflow this is the iteration immediately following the
        # NSCF.  With a supplied parent, the first iteration can already be a
        # QP/BSE step, so inspect that parent as well.
        if not hasattr(self.ctx, 'calc'):
            return
        if self.ctx.calc_to_do == 'scf':
            return

        nscf_calculation = self._resolve_nscf_calculation(self.ctx.calc)
        if nscf_calculation is None:
            return

        try:
            mapping = gap_mapping_from_nscf(nscf_calculation.pk)
        except Exception as exception:
            self.report(f'Unable to derive the NSCF mapping: {exception}')
            return self.exit_codes.ERROR_WORKCHAIN_FAILED

        self.ctx.mapping = mapping
        if self.inputs.stop_if_metal and mapping.get('dft_predicted') == 'metal':
            self.report(
                'The NSCF mapping predicts a metallic system '
                f'(gap={mapping.get("nscf_gap_eV")}); stopping before Yambo.'
            )
            self.out('nscf_mapping', store_Dict(mapping))
            return self.exit_codes.ERROR_METALLIC_SYSTEM

    def perform_next(self):
        """This function  will submit the next step, depending on the information provided in the context.

        The next step will be a yambo calculation if the provided inputs are a previous yambo/p2y run
        Will be a PW scf/nscf if the inputs do not provide the NSCF or previous yambo parent calculations
        
        """
        
        self.ctx.calc_to_do = self.ctx.all_calcs_to_do.pop(0)
        # check if the previous run failed.
        if hasattr(self.ctx, 'calc'):
            calc = self.ctx.calc
            if not calc.is_finished_ok:
                # NOTE: failures of the splitted QP runs are checked in run_post_process.
                self.report("last calculation failed, exiting the workflow")
                return self.exit_codes.ERROR_WORKCHAIN_FAILED

        self.report('performing a {} calculation'.format(self.ctx.calc_to_do))

        if self.ctx.calc_to_do == 'scf':
            self.ctx.input_manager.scf.metadata.call_link_label = 'scf'
            scf_inputs = self.ctx.input_manager.generate_AiiDA_inputs(what='scf')
            future = self.submit(PwBaseWorkChain, **scf_inputs)

        elif self.ctx.calc_to_do == 'nscf':
            self.ctx.input_manager.nscf.metadata.call_link_label = 'nscf'
            self.ctx.input_manager.set_parent_folder(self.ctx.calc.outputs.remote_folder)
            nscf_inputs = self.ctx.input_manager.generate_AiiDA_inputs(what='nscf')
            future = self.submit(PwBaseWorkChain, **nscf_inputs)

        elif self.ctx.calc_to_do in ['qp']:
            c_name = self.ctx.calc_to_do
            # only in the QP case, we update the parameters, i.e. we add QP to be computed:
            if len(self.ctx.input_manager.additional_parsing) > 0 and self.ctx.calc_to_do=='qp':
                self.report('updating yambo parameters to parse more results')
                mapping, yambo_parameters = add_corrections(
                    self.ctx.input_manager.qp.parameters, 
                    self.ctx.calc.outputs.remote_folder,
                    self.ctx.input_manager.additional_parsing,
                )
                self.ctx.mapping = mapping
                self.ctx.input_manager.qp.parameters = yambo_parameters
            match c_name:
                case 'qp':
                    self.ctx.input_manager.qp.metadata.call_link_label = self.ctx.calc_to_do
                case 'bse':
                    self.ctx.input_manager.bse.metadata.call_link_label = self.ctx.calc_to_do

            self.ctx.input_manager.set_parent_folder(self.ctx.calc.outputs.remote_folder)
            yambo_inputs = self.ctx.input_manager.generate_AiiDA_inputs(what=self.ctx.calc_to_do)
            future = self.submit(YamboRestart, **yambo_inputs)
        
        elif self.ctx.calc_to_do == 'QP splitter': # This is the multiple QP runs which will then be merged with yambopy

            qp_futures = {}

            # a failed batch stops the following ones: no resources are spent on a merge which cannot be done.
            if self.ctx.qp_splitter > 0 and not all(load_node(uuid).is_finished_ok for uuid in self.ctx.splitted_QP):
                self.report('Some splitted QP is failed, exiting... ')
                return self.exit_codes.ERROR_SPLITTED_QP_FAILED

            # If this is the first QP iteration, we set the parent and all the needed inputs.
            if self.ctx.qp_splitter == 0:
                calc = self.ctx.calc
                if not calc.is_finished_ok:
                    self.report("last calculation failed, exiting the workflow")
                    return self.exit_codes.ERROR_WORKCHAIN_FAILED
                    
                
                # why this below?
                #self.ctx.yambo_inputs.yambo.parameters = take_calc_from_remote(self.ctx.yambo_inputs['parent_folder'],level=-1).inputs.parameters
                
                # we copy the aiida.out so we reuse the screening calculation.
                if self.ctx.should_run_bse:
                    self.ctx.input_manager.qp.settings['COPY_DBS'] = True
                else:
                    self.ctx.input_manager.qp.settings['COPY_DBS'] = True

                # setting some specific configuration:
                if isinstance(self.ctx.input_manager.QP_subsets_dict.get('parallelism'), dict):
                    for k,v in self.ctx.input_manager.QP_subsets_dict['parallelism'].items():
                        self.ctx.input_manager.qp.set_variable(name=k, value=v)
                validate_QP_subsets_dict(self.ctx.input_manager.QP_subsets_dict)
                set_QP_splitter_options(self.ctx.input_manager.qp, self.ctx.input_manager.QP_subsets_dict)
                if hasattr(self.ctx, 'mapping'):
                    mapping = self.ctx.mapping
                else:
                    mapping = gap_mapping_from_nscf(find_pw_parent(take_calc_from_remote(self.ctx.calc.outputs.remote_folder,level=-1)).pk)
                    self.ctx.mapping = mapping

                consider_only = self.ctx.input_manager.QP_subsets_dict.get('consider_only',[-1]) #[1,64], a range.
                self.ctx.input_manager.QP_subsets_dict['consider_only'] = consider_only

                if 'range_QP' in self.ctx.input_manager.QP_subsets_dict: #the name can be changed..
                    Energy_region = self.ctx.input_manager.QP_subsets_dict.get('range_QP', mapping['nscf_gap_eV']*1.2)
                    self.report('range of energy for QP: {} eV'.format(Energy_region))
                    self.ctx.input_manager.QP_subsets_dict['explicit'], self.ctx.input_manager.QP_subsets_dict['scissored'] = QP_mapper(
                                                                    self.ctx.calc,
                                                                    tol = Energy_region,
                                                                    full_bands=self.ctx.input_manager.QP_subsets_dict.get('full_bands',False),
                                                                    spectrum_tol=self.ctx.input_manager.QP_subsets_dict.get('range_spectrum',Energy_region),
                                                                )
                if hasattr(self.ctx.input_manager,'already_computed_QP_db'):
                    already_computed = self.ctx.input_manager.already_computed_QP_db
                else:
                    already_computed = None

                if 'boundaries' in self.ctx.input_manager.QP_subsets_dict:
                    k_i=self.ctx.input_manager.QP_subsets_dict['boundaries'].pop('ki',1)
                    k_f=self.ctx.input_manager.QP_subsets_dict['boundaries'].pop('kf',mapping['number_of_kpoints'])
                    b_i=self.ctx.input_manager.QP_subsets_dict['boundaries']['bi']
                    b_f=self.ctx.input_manager.QP_subsets_dict['boundaries']['bf']

                    self.ctx.input_manager.QP_subsets_dict['subsets'] = QP_list_merger([[k_i,k_f,b_i,b_f]],
                                                                      self.ctx.input_manager.QP_subsets_dict['qp_per_subset'],
                                                                      consider_only=consider_only,
                                                                      already_computed_QP_db = already_computed
                                                                      )

                if not 'subsets' in self.ctx.input_manager.QP_subsets_dict.keys():
                    if 'explicit' in self.ctx.input_manager.QP_subsets_dict.keys():
                        self.ctx.input_manager.QP_subsets_dict['subsets'] = QP_list_merger(self.ctx.input_manager.QP_subsets_dict['explicit'],
                                                                        self.ctx.input_manager.QP_subsets_dict['qp_per_subset'],
                                                                        consider_only=consider_only,
                                                                        already_computed_QP_db = already_computed)
                    else:
                        self.report('No QP subsets defined (no boundaries/explicit/range_QP): building default full-range QP subsets')
                        self.ctx.input_manager.QP_subsets_dict['subsets'] = QP_list_merger(
                                                                        [[1, mapping['number_of_kpoints'], 1, self.ctx.input_manager.Nb]],
                                                                        self.ctx.input_manager.QP_subsets_dict['qp_per_subset'],
                                                                        consider_only=consider_only,
                                                                        already_computed_QP_db = already_computed)

                if self.ctx.input_manager.QP_subsets_dict.get('parallelism') == 'auto':
                    pw = find_pw_parent(self.ctx.calc)
                    pw_output = pw.outputs.output_parameters.get_dict()
                    self.ctx.auto_parallelism = {
                        'occupied': valence_conduction_bands(pw_output['number_of_electrons'], pw_output['spin_orbit_calculation'])[0],
                        'kpoints': mapping['number_of_kpoints'],
                        'bands': self.ctx.input_manager.Nb,
                        'resources': self.ctx.input_manager.qp.metadata.options.resources.model_dump(),
                    }
                
            if self.ctx.number_of_subsets == 0:
                self.ctx.number_of_subsets = len(self.ctx.input_manager.QP_subsets_dict['subsets'])

            for i in range(1,1+self.ctx.input_manager.QP_subsets_dict['parallel_runs']):
                if len(self.ctx.input_manager.QP_subsets_dict['subsets']) > 0:
                    subset = self.ctx.input_manager.QP_subsets_dict['subsets'].pop()
                    self.ctx.input_manager.qp.set_variable(name='QPkrange', value=[subset,''])
                    if self.ctx.input_manager.QP_subsets_dict.get('parallelism') == 'auto':
                        set_auto_QP_parallelism(self.ctx.input_manager.qp, subset, **self.ctx.auto_parallelism)
                    self.ctx.input_manager.qp.metadata.call_link_label = 'yambo_QP_splitted_{}'.format(i+self.ctx.qp_splitter)
                    self.ctx.input_manager.set_parent_folder(self.ctx.calc.outputs.remote_folder)
                    what='qp'
                    yambo_inputs = self.ctx.input_manager.generate_AiiDA_inputs(what=what)
                    future = self.submit(YamboRestart, **yambo_inputs)
                    self.report('Launching YamboRestart <{}> for QP, iteration #{} of {}'.format(future.pk,i+self.ctx.qp_splitter, self.ctx.number_of_subsets))
                    self.ctx.splitted_QP.append(future.uuid)
                    key = f'qp_splitted_{i + self.ctx.qp_splitter}'
                    qp_futures[key] = future
            
            self.ctx.qp_splitter += self.ctx.input_manager.QP_subsets_dict['parallel_runs']

            if len(self.ctx.input_manager.QP_subsets_dict['subsets']) < 1: 
                self.ctx.merge_QP_needed = True
            else:
                self.ctx.all_calcs_to_do = ['QP splitter'] + self.ctx.all_calcs_to_do

            return ToContext(**qp_futures) #wait for all splitted calculations....
        
        elif self.ctx.merge_QP_needed:
            if self.ctx.calc_to_do == 'bse' and len(self.ctx.all_calcs_to_do) == 0:
                self.ctx.all_calcs_to_do = ['bse']
            
            self.ctx.merge_QP_needed = False

            # run_post_process returns an exit code on failure: propagate it.
            exit_code = self.run_post_process()
            if exit_code is not None:
                return exit_code

            return
        
        elif self.ctx.calc_to_do == 'bse':
            bse_params = self.ctx.input_manager.bse.parameters

            if self.ctx.QP_db:
                
                # set QP_corrections input and check the Q index and the bands 
                
                if isinstance(self.ctx.QP_db,tuple): self.ctx.QP_db = self.ctx.QP_db[1]
                
                self.ctx.input_manager.bse.QP_corrections = self.ctx.QP_db
                bse_params['variables']['KfnQPdb'] = "E < ./ndb.QP"

                if not 'BSEBands' in bse_params['variables'].keys():
                    if 'scissored' in self.ctx.input_manager.QP_subsets_dict.keys():
                        bse_params['variables']['BSEBands'] = [[self.ctx.input_manager.QP_subsets_dict['scissored'][0],
                                                                self.ctx.input_manager.QP_subsets_dict['scissored'][1]],
                                                            '']
                    else:
                        bse_params['variables']['BSEBands'] = [[self.ctx.BSE_map['v_min'],
                                                                self.ctx.BSE_map['c_max']],
                                                            '']
                if not 'BSEQptR' in bse_params['variables'].keys():
                    bse_params['variables']['BSEQptR'] = [[self.ctx.BSE_map['q_ind'],
                                                        self.ctx.BSE_map['q_ind']],
                                                        '']


            self.ctx.input_manager.bse.metadata.call_link_label = 'BSE'
            
            self.ctx.input_manager.set_parent_folder(self.ctx.calc.outputs.remote_folder)
            
            yambo_inputs = self.ctx.input_manager.generate_AiiDA_inputs(what='bse')
            future = self.submit(YamboRestart, **yambo_inputs)
            self.ctx.bse = future

        return ToContext(calc = future)
    
    def run_post_process(self):
        #check if all QP splitted calculations were ok:
        for splitted in self.ctx.splitted_QP:
            if not load_node(splitted).is_finished_ok:
                self.report('Some splitted QP is failed, exiting... ')
                return self.exit_codes.ERROR_SPLITTED_QP_FAILED
        #merge
        self.report('Running merge QP')

        splitted = store_List(self.ctx.splitted_QP)
        
        self.out('splitted_QP_calculations', splitted)
        output_name = orm.Str('ndb.QP_merged')
        
        self.ctx.input_manager.QP_subsets_dict['extend_db'] = self.ctx.input_manager.QP_subsets_dict.pop('extend_db',False)

        merge_result = merge_QP(
            splitted,
            output_name,
            orm.Int(self.ctx.calc.pk),
            qp_settings=orm.Dict(dict=self.ctx.input_manager.QP_subsets_dict),
            already_computed_QP_db = self.inputs.get('already_computed_QP_db',orm.List([]))
            )
        self.ctx.QP_db = merge_result['QP_db']
        self.out('merged_QP',self.ctx.QP_db)

        # we need both valence and conduction states to proceed (BSE, extended QP db...):
        has_valence = merge_result['has_valence'].value
        has_conduction = merge_result['has_conduction'].value
        if (not has_valence or not has_conduction) and (self.ctx.should_run_bse or self.ctx.input_manager.QP_subsets_dict['extend_db']):
            missing = 'valence and conduction' if not has_valence and not has_conduction \
                else ('conduction' if not has_conduction else 'valence')
            self.report('The merged QP database does not contain {} states, exiting the workflow'.format(missing))
            return self.exit_codes.ERROR_MISSING_BAND_REGION
        
        # Extend QP is requested:
        if self.ctx.input_manager.QP_subsets_dict['extend_db']:
            self.report('Running extend QP')
            self.ctx.QP_db_extended = extend_QP(
                splitted,output_name,
                orm.Int(self.ctx.calc.pk),
                qp_settings=orm.Dict(dict=self.ctx.input_manager.QP_subsets_dict),
                QP=self.ctx.QP_db)
            self.out('extended_QP',self.ctx.QP_db_extended)
        #else:
            #self.ctx.QP_db = merge_QP(splitted,output_name,Int(self.ctx.calc.pk),qp_settings=Dict(dict=self.ctx.input_manager.QP_subsets_dict))
        
        # We analyse the data for BSE, in case we need to find a new indirect gap for finite-q BSE...
        BSE_map = QP_analyzer(self.ctx.calc.pk, self.ctx.QP_db,self.ctx.mapping)
        self.ctx.BSE_map = BSE_map

        if BSE_map.get('missing_region', None) is not None and self.ctx.should_run_bse:
            self.report('The merged QP database does not contain {} states, exiting the workflow'.format(BSE_map['missing_region']))
            return self.exit_codes.ERROR_MISSING_BAND_REGION

        return

    def report_wf(self):

        self.report('Final step.')

        calc = self.ctx.calc
        if calc.is_finished_ok:

            if hasattr(self.inputs, 'additional_parsing'):
                #self.report('parsing additional quantities')
                mapping, _ = add_corrections(
                    self.ctx.input_manager.qp.parameters, 
                    self.ctx.calc.outputs.remote_folder,
                    self.ctx.input_manager.additional_parsing,
                )
                parsed = additional_parsed(calc, self.ctx.input_manager.additional_parsing, mapping)
                mapping_Dict = store_Dict(mapping)
                self.out('nscf_mapping', mapping_Dict)
                if hasattr(self.ctx, 'BSE_map'):
                    parsed.update(self.ctx.BSE_map)
                if hasattr(self.ctx,'bse'):
                    if self.ctx.bse.is_finished_ok:
                        parsed_bse = additional_parsed(self.ctx.bse, self.ctx.input_manager.additional_parsing, mapping)
                        parsed.update(parsed_bse)
                        if hasattr(self.ctx, 'BSE_map'):
                            parsed.update(self.ctx.BSE_map)
                    else:
                        self.report("workflow NOT completed successfully")
                        return self.exit_codes.ERROR_WORKCHAIN_FAILED
                self.report('PARSED: {}'.format(parsed))
                self.out('output_ywfl_parameters', store_Dict(parsed))
            elif hasattr(self.ctx, 'BSE_map'):
                mapping, _ = add_corrections(
                    self.ctx.input_manager.qp.parameters, 
                    self.ctx.calc.outputs.remote_folder,
                    [])
                mapping_Dict = store_Dict(mapping)
                self.out('nscf_mapping', mapping_Dict)
                self.out('output_ywfl_parameters', store_Dict(self.ctx.BSE_map))
                
            if hasattr(self.ctx,'bse'):
                if self.ctx.bse.is_finished_ok:
                    self.out_many(self.exposed_outputs(self.ctx.bse,YamboRestart))
            else:
                self.out_many(self.exposed_outputs(calc,YamboRestart))
                
            self.report("workflow completed successfully")
            
            if hasattr(self.ctx.input_manager, "clean_failed"):
                if self.ctx.input_manager.clean_failed: 
                    message = clean(calc.caller)
                    if message:
                        self.report(message)
        else:
            self.report("workflow NOT completed successfully")
            return self.exit_codes.ERROR_WORKCHAIN_FAILED
