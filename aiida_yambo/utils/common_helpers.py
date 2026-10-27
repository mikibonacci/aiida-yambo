# -*- coding: utf-8 -*-
"""helpers for many purposes"""
import copy
import logging
import os

import numpy as np

from aiida.common.exceptions import NotExistent
from aiida.engine import calcfunction
from aiida.orm import Dict, KpointsData, List, RemoteData, load_node

from aiida_yambo.utils.k_path_utils import k_path_dealer

LOGGER = logging.getLogger(__name__)

def find_parent(calc):

    if calc.is_finished and 'workflows' in calc.process_type:
        parent_calc = calc.outputs.remote_folder.creator
    else:
        parent_calc = calc.inputs.parent_folder.creator
    if not parent_calc:
        raise Exception('No creator found for parent_folder {}'.format(parent_calc))
    return parent_calc

def find_pw_parent(parent_calc, calc_type = None):

    calc_type = ['scf', 'nscf'] if calc_type is None else calc_type
    has_found_pw = False
    tentatives=0
    while (not has_found_pw):
        tentatives+=1
        #print('tentatives=',tentatives)
        #print(parent_calc)
        if parent_calc.process_type=='aiida.calculations:yambo.yambo' or 'workflows' in parent_calc.process_type:
            has_found_pw = False
            parent_calc = find_parent(parent_calc)

        if parent_calc.process_type=='aiida.calculations:quantumespresso.pw' and \
            find_pw_type(parent_calc) in calc_type:
            has_found_pw = True
            break
        else:
            parent_calc = find_parent(parent_calc)
        if tentatives==10000: 
            LOGGER.debug('too much tentatives to find the parent')
            break

    return parent_calc
    
def old_find_parent(calc):

    try:
        parent_calc = calc.inputs.parent_folder.get_incoming().all_nodes()[-1] #to load the node from a workchain...
    except IndexError:
        try:
            parent_calc = calc.inputs.parent_folder.get_incoming().get_node_by_label('remote_folder')
        except NotExistent:
            parent_calc = calc.called[0] #nscf_workchain...??
    return parent_calc

def old_find_pw_parent(parent_calc, calc_type = None):

    calc_type = ['scf', 'nscf'] if calc_type is None else calc_type
    has_found_pw = False
    while (not has_found_pw):
        if parent_calc.process_type=='aiida.calculations:yambo.yambo' or parent_calc.process_type=='aiida.workflows:yambo.yambo.yambowf':
            has_found_pw = False
            parent_calc = find_parent(parent_calc)
            if parent_calc.process_type=='aiida.workflows:quantumespresso.pw.base':
                for i in range(1,len(parent_calc.called)+1):
                    parent_try = parent_calc.called[-i]
                    if parent_try.process_type=='aiida.calculations:quantumespresso.pw' and \
                find_pw_type(parent_try) in calc_type:
                        parent_calc = parent_try
                        has_found_pw = True
                        break
            if parent_calc.process_type=='aiida.calculations:quantumespresso.pw' and \
                find_pw_type(parent_calc) in calc_type:
                has_found_pw = True
            else:
                parent_calc = find_parent(parent_calc)
        elif parent_calc.process_type=='aiida.calculations:quantumespresso.pw' and \
            find_pw_type(parent_calc) in calc_type:
            has_found_pw = True
        else:
            parent_calc = find_parent(parent_calc)

    return parent_calc

def get_distance_from_kmesh(calc):
    mesh = calc.inputs.kpoints.get_kpoints_mesh()[0]
    LOGGER.debug(mesh)
    k = KpointsData()
    k.set_cell_from_structure(calc.inputs.structure) #these take trace of PBC...if set in the inputs.!!
    for i in range(4,400):
         k.set_kpoints_mesh_from_density(1/(i*0.25))
         LOGGER.debug(k.get_kpoints_mesh()[0])
         if k.get_kpoints_mesh()[0]==mesh:
             LOGGER.debug('ok, {} is the density'.format(i*0.25))
             LOGGER.debug('%s %s', k.get_kpoints_mesh()[0], mesh)
             return i*0.25
         else: #try the same algorithm but using force_parity=True.
            k.set_kpoints_mesh_from_density(1/(i*0.25),force_parity=True)
            LOGGER.debug(k.get_kpoints_mesh()[0])
            if k.get_kpoints_mesh()[0]==mesh:
                LOGGER.debug('ok, {} is the density'.format(i*0.25))
                LOGGER.debug('%s %s', k.get_kpoints_mesh()[0], mesh)
                return i*0.25 

def find_pw_type(calc):
    type = calc.inputs.parameters.get_dict()['CONTROL']['calculation']
    return type

def find_table_ind(kpoint,band,_array_ndb):
    kk = _array_ndb.get_array('qp_table')
    #in the qp table:
    index_b = -1
    index_k = 0
    if np.shape(kk)[0] == 4:
        index_b = -2
        index_k = 0 #spin resolved..

    b = kk[index_b]==band
    c = kk[index_k]==kpoint
    g = (c == True) & (b == True)
    for i in range(len(g)):
        if g[i] == True:
            return(i)


def update_dict(_dict, whats, hows, sublevel=None, pop_list=None):
    pop_list = [] if pop_list is None else pop_list
    if sublevel:
        if not isinstance(whats, list):
            whats = [whats]
        if not isinstance(hows, list):
            hows = [hows] 
        for what,how in zip(whats,hows):    
            new = _dict.get_dict()
            new[sublevel][what] = how
            _dict = Dict(new)
        if pop_list != []:
            for i in pop_list:
                new = _dict.get_dict()
                new[sublevel].pop(i)
                _dict = Dict(new)
    else:
        if not isinstance(whats, list):
            whats = [whats]
        if not isinstance(hows, list):
            hows = [hows] 
        for what,how in zip(whats,hows):    
            new = _dict.get_dict()
            new[what] = how
            _dict = Dict(new)
        if pop_list != []:
            for i in pop_list:
                new = _dict.get_dict()
                new.pop(i)
                _dict = Dict(new)
    
    return _dict

def get_caller(calc_pk, depth = 1):
     calc = load_node(calc_pk)
     for i in range(depth):
         calc = calc.caller.caller
     return calc

def get_called(calc_pk, depth = 2):
     calc = load_node(calc_pk)
     for i in range(depth):
         calc = calc.called[0]
     return calc

def set_parent(inputs, parent):
     if isinstance(parent, RemoteData):
         inputs.parent_folder = parent
     else:
         inputs.parent_folder = parent.outputs.remote_folder

def take_down(node = 0, what = 'CalcJobNode'):

     global calc_node

     if node == 0:
         node = load_node(wfl_pk)
     else:
         node = load_node(node)

     if what not in str(node.get_description):
         take_down(node.called[0])
     else:
         calc_node = node

     return calc_node

def take_super(node = 0, what = 'WorkChainNode'):

     global workchain_node

     if node == 0:
         node = load_node(wfl_pk)
     else:
         node = load_node(node)

     if what not in str(node.get_description):
         take_super(node.caller)
     else:
         workchain_node = node

     return workchain_node

def take_calc_from_remote(parent_folder,level=0):
        try:
            parent_calc = parent_folder.creator
            #parent_calc = parent_folder.get_incoming().all_nodes()[level] #to load the node from a workchain...
        except Exception:
            parent_calc = parent_folder.get_incoming(link_label_filter='remote_folder').one().node
        return parent_calc
    
def get_pw_inputs_builders(parent_folder):
    scf_parent=None
    nscf_parent=None
    builder_scf, builder_nscf=None, None
    parent_calc = take_calc_from_remote(parent_folder)
    # getting the PwCalculations
    ## SCF:
    try:
        scf_parent = find_pw_parent(parent_calc, calc_type=['scf'])
        if scf_parent.caller:
            builder_scf=scf_parent.caller.get_builder_restart()
        else:
            LOGGER.info('SCF parent PwBaseWorkChain not found')
            scf_parent=None
    except Exception as exc:
        scf_parent=None
        LOGGER.info('SCF parent not found: %s', exc)
    ## NSCF:
    try:
        nscf_parent = find_pw_parent(parent_calc, calc_type=['nscf'])
        if nscf_parent.caller:
            builder_nscf=nscf_parent.caller.get_builder_restart()
        else:
            LOGGER.info('SCF parent PwBaseWorkChain not found')
            nscf_parent=None
    except Exception as exc:
        nscf_parent=None
        LOGGER.info('NSCF parent not found: %s', exc)
        
    return builder_scf, builder_nscf

def take_fermi(calc_node_pk):  # calc_node_pk = node_conv_wfl.outputs.last_calculation

    node = load_node(calc_node_pk)
    path_folder = node.outputs.retrieved._repository._repo_folder.abspath+'/path/'
    for i in os.listdir(path_folder):
        if 'r-aiida.out' in i:
            file = open(path_folder+i,'r')
    for line in file:
        if '[X]Fermi Level' in line:
            LOGGER.info('The Fermi level is {}'.format(line.split()[3]))
            ef = float(line.split()[3])

    return ef

def take_filled_states(calc_node_pk):  # calc_node_pk = node_conv_wfl.outputs.last_calculation

    node = load_node(calc_node_pk)
    path_folder = node.outputs.retrieved._repository._repo_folder.abspath+'/path/'
    get_line=False #not so good...
    for i in os.listdir(path_folder):
        if 'r-aiida.out' in i:
            file = open(path_folder+i,'r')
    for line in file:
        if  get_line:
            LOGGER.debug('The VBM {}'.format(line.split()[0]))
            valence = int(line.split()[0].replace('0001-',''))
            return valence
        if '[X]States summary ' in line:
            get_line=True

    

def take_number_kpts(calc_node_pk):  # calc_node_pk = node_conv_wfl.outputs.last_calculation

    node = load_node(calc_node_pk)
    path_folder = node.outputs.retrieved._repository._repo_folder.abspath+'/path/'
    for i in os.listdir(path_folder):
        if 'r-aiida.out' in i:
            file = open(path_folder+i,'r')
    for line in file:
        if 'K-points' in line:
            LOGGER.debug('# of kpts is {}'.format(line.split()[2]))
            kpts = int(line.split()[2])
            return kpts
    
def store_List(a_list):
    the_List = List(a_list)
    the_List.store()
    return the_List

def store_Dict(a_dict):
    the_Dict = Dict(a_dict)
    the_Dict.store()
    return the_Dict

def find_pw_info(calc):

    pw_parent = find_pw_parent(calc, calc_type = ['nscf'])
    info = pw_parent.outputs.output_parameters.get_dict()   
    return info

def find_gw_info(inputs):

    if isinstance(inputs, dict):
        parameters = copy.deepcopy(inputs)
    elif hasattr(inputs, 'parameters') and isinstance(inputs.parameters, dict):
        parameters = copy.deepcopy(inputs.parameters)
    else: #legacy builder or orm.Dict node
        parameters = copy.deepcopy(inputs.parameters.get_dict())
    
    ## bands ##

    BndsRnXp = parameters['variables'].pop('BndsRnXp',[[0,0],''])[0]
    GbndRnge = parameters['variables'].pop('GbndRnge',[[0,0],''])[0]
    X_b = 1 + BndsRnXp[1] - BndsRnXp[0]
    SE_b = 1 + GbndRnge[1] - GbndRnge[0]
    if X_b and SE_b:
        bands = min(X_b,SE_b)
    elif X_b and not SE_b:
        bands = X_b
    elif not X_b and SE_b:
        bands = SE_b
    else: bands = 1

    ## parallelism ##

    ## qp ##

    qp = 0
    last_qp = 0
    try:
        for i in parameters['variables']['QPkrange'][0]:
            qp += (1 + i[1]-i[0])*(1 + i[3]-i[2])
            last_qp = max(i[3],last_qp)
    except (KeyError, IndexError, TypeError): # no (valid) QPkrange
        pass

    ## runlevels ##
    runlevels = []
    for i in parameters['arguments']:
        runlevels.append(i)
    
    return bands, qp, last_qp, runlevels

def understand_valence_metal_wise(bands, fermi, index, valence):
    hp_valence = len(np.where((bands[index-1]-fermi)<=1e-2)[0])
    if hp_valence < valence - 1: hp_valence = valence
    return valence #deactivated.

def build_list_QPkrange(mapping, quantity, nscf_pk, bands, fermi, valence):
    s = load_node(nscf_pk)
    if isinstance(quantity,str):
        if 'gap_' in quantity:
            if quantity[-1] == '_': 
                pass
            else: #high-symmetry
                m,maps = k_path_dealer().check_kpoints_in_qe_grid(s.outputs.output_band.get_kpoints(),
                                       s.inputs.structure.get_ase()) #m stands for missing
                if quantity[-1] in m or quantity[-2] in m: return quantity, 0
                if not quantity[-1] in maps.keys() or not quantity[-2] in maps.keys(): return quantity, 0
                if mapping['dft_predicted']=='metal':
                    valence = understand_valence_metal_wise(bands, fermi, maps[quantity[-2]],valence)
                    LOGGER.info(valence)
                    conduction = understand_valence_metal_wise(bands, fermi, maps[quantity[-1]],valence) + 1 + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation']))
                else:
                    valence = mapping['valence']
                return quantity,[[maps[quantity[-2]],maps[quantity[-2]],
                        valence,valence],
                        [maps[quantity[-1]],maps[quantity[-1]],
                         valence + 1 + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation'])),valence + 1 + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation']))]]
        else: #high-symmetry
                m,maps = k_path_dealer().check_kpoints_in_qe_grid(s.outputs.output_band.get_kpoints(),
                                       s.inputs.structure.get_ase())
                
                if quantity in m : return quantity, 0
                if not quantity[-1] in maps.keys(): return quantity, 0
                valence = understand_valence_metal_wise(bands, fermi, maps[quantity[-1]],valence)
                if '_v' in quantity:
                    return quantity,[[maps[quantity[0]],maps[quantity[0]],
                         valence,valence],]
                elif '_c' in quantity:
                    return quantity,[[maps[quantity[0]],maps[quantity[0]],
                         valence + 1 + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation'])),valence + 1 + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation']))],]
                
                return quantity,[[maps[quantity[-1]],maps[quantity[-1]],
                         valence,valence],
                        [maps[quantity[-1]],maps[quantity[-1]],
                         valence + 1 + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation'])),valence + 1 + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation']))],]
            
    elif isinstance(quantity,list):
        if 'gap_' in quantity[0]:
            m,maps = k_path_dealer().check_kpoints_in_qe_grid(s.outputs.output_band.get_kpoints(),
                                       s.inputs.structure.get_ase(),k_list={quantity[0][-2]:np.array(quantity[1][-2]),
                                                                            quantity[0][-1]:np.array(quantity[1][-1])})

            if quantity[0] in m : return quantity[0], 0    
            valence = understand_valence_metal_wise(bands, fermi, maps[quantity[0][-2]],valence)
            conduction = understand_valence_metal_wise(bands, fermi, maps[quantity[0][-1]],valence) + 1
            return quantity[0],[[maps[quantity[0][-2]],maps[quantity[0][-2]],
                    valence,valence],
                    [maps[quantity[0][-1]],maps[quantity[0][-1]],
                    valence + 1 + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation'])),valence + 1 + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation']))],]
        else:
            m,maps = k_path_dealer().check_kpoints_in_qe_grid(s.outputs.output_band.get_kpoints(),
                                       s.inputs.structure.get_ase(),k_list={quantity[0]:np.array(quantity[1]),
                                                                            quantity[0]:np.array(quantity[1])})
            
            if quantity[0] in m : return quantity[0], 0
            valence = understand_valence_metal_wise(bands, fermi, maps[quantity[0]],valence)
            if '_v' in quantity[0]:
                return quantity[0],[[maps[quantity[0]],maps[quantity[0]],
                     valence,valence],]
            elif '_c' in quantity[0]:
                return quantity[0],[[maps[quantity[0]],maps[quantity[0]],
                     valence + 1 + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation'])),valence + 1 + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation']))],]
            
            
            return quantity[0],[[maps[quantity[0]],maps[quantity[0]],
                     valence,valence],
                    [maps[quantity[0]],maps[quantity[0]],
                     valence + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation'])),valence + 1 + 1*(int(mapping['soc'])-int(mapping['magnetic_calculation']))],]    
    else:
        return 0, 0

def gap_mapping_from_nscf(nscf_pk, additional_parsing_List=None):
    #print('START')
    additional_parsing_List = [] if additional_parsing_List is None else additional_parsing_List
    s_res = False
    nscf = load_node(nscf_pk)
    bands = nscf.outputs.output_band.get_array('bands')
    # if magnetic_calculation, i.e. magnetization is provided, it considers only the first spin (up)
    if len(np.shape(bands))>2:
        s_res = True 
        bands = bands[0,:,:]
    occ = nscf.outputs.output_band.get_array('occupations')
    n_kpoints = nscf.outputs.output_parameters.get_dict()['number_of_k_points']
    k_coords = nscf.outputs.output_band.get_kpoints()
    valence = len(occ[0][occ[0]>0.01]) #band index of the valence. 
    valence = nscf.outputs.output_parameters.get_dict()['number_of_electrons']/2.
    fermi = nscf.outputs.output_parameters.get_dict()['fermi_energy']
    soc = nscf.outputs.output_parameters.get_dict()['spin_orbit_calculation']
    try:
        try:
            nscf.inputs.pw__structure.get.ase()
            #print('HERE0')
        except Exception:
            nscf.inputs.structure.get.ase()
            #print('HERE1')
        cell = structure.get_cell()
        k = cell.bandpath()
        high_symmetry = k.special_points
        #print('HERE2')
    except Exception: # NOTE: this always fails for now ('structure' is not defined above)
        high_symmetry = []
        #print('HERE3')
    if valence%2 != 0:
        valence = int(valence+0.5) #may be a metal
    else:
        valence = int(valence)
    conduction = valence + 1  
    
    if soc and not s_res:
        valence = valence*2 - 1
        conduction = valence + 2

    touch_fermi = bands[:,valence-1]-fermi
    touch_fermi_C = bands[:,valence]-fermi
    above_fermi = np.where(touch_fermi>0)[0]
    below_fermi = np.where(touch_fermi<=0)[0]
    LOGGER.info('%s %s', len(above_fermi), len(below_fermi))

    ind_val = bands[:,valence-1].argmax()
    ind_cond = bands[:,conduction-1].argmin()
    
    L_H = (round((min(bands[:,conduction-1])-max(bands[:,valence-1])),3))
    Crossing = len(np.where((bands[:,valence-1]-fermi)>1e-2)[0])
    if Crossing > 0:
        dft_predicted = 'metal'
    elif L_H<=0.1 and L_H>=-0.2: #and (valence%2 == 0 or (soc and valence%2 == 1)):
        dft_predicted = 'semimetal'
    elif L_H>0.1: # and (valence%2 == 0 or (soc and valence%2 == 1)):
        dft_predicted = 'semiconductor/insulator'
    else:
        dft_predicted = 'metal'

    if ind_val+1 != ind_cond+1:
        gap_type = 'indirect'
    else:
        gap_type = 'direct'

    if s_res: soc = True
    mapping = {
        'dft_predicted': dft_predicted,
        'valence': valence,
        'conduction': conduction,
        'number_of_kpoints':n_kpoints,
        'nscf_gap_eV':round(min(bands[:,conduction-1])-max(bands[:,valence-1]),3),
        'homo_k': ind_val+1,
        'lumo_k': ind_cond+1,
        'gap_type': gap_type,
        'gap_': [[ind_val+1,ind_val+1,valence,valence],
                [ind_cond+1,ind_cond+1,conduction,conduction]], #the qp to be computed
        'soc':soc,
        'magnetic_calculation':s_res
    }

    for i in additional_parsing_List + high_symmetry:
        if i == 'homo' or i == 'lumo' or i == 'gap_':
            pass
        else:
            #print(i)
            name, additional = build_list_QPkrange(mapping, i, nscf_pk, bands, fermi,valence)
            #print(name,additional)
            if additional == 0: 
                pass
            else:
                mapping[name] = additional

    return mapping

def _is_ignored_variable(name):
    """Parallelism variables do not change the results."""
    return 'CPU' in name or 'ROLEs' in name or 'PAR_' in name

def same_yambo_variables(params_to_calc, old_params):
    """True if two sets of yambo variables (`{name: [value, unit]}`) give the same results.

    Same variables, with the same values and units. We do not consider the parallelism,
    and the `QPkrange`, which is set at runtime from the `additional_parsing` quantities.
    """
    new = {k: v for k, v in params_to_calc.items() if not _is_ignored_variable(k) and k != 'QPkrange'}
    old = {k: v for k, v in old_params.items() if not _is_ignored_variable(k) and k != 'QPkrange'}
    return new == old

def _pw_signature(structure, pseudos, parameters):
    """What defines the DFT calculation: structure, pseudopotentials and parameters (but the number of bands)."""
    parameters = copy.deepcopy(parameters)
    parameters.get('SYSTEM', {}).pop('nbnd', None)
    return {
        'structure': structure.base.caching.compute_hash() if structure is not None else None,
        'pseudos': {kind: pseudo.base.caching.compute_hash() for kind, pseudo in (pseudos or {}).items()},
        'parameters': parameters,
    }

def pw_signature_from_manager(pw_manager):
    """Signature of the `scf` or `nscf` PwBaseWorkChainInputManager of a YamboInputManager."""
    return _pw_signature(getattr(pw_manager, 'structure', None), getattr(pw_manager, 'pseudos', None), pw_manager.parameters)

def pw_signature_from_node(node, namespace=()):
    """Signature of a PwCalculation (no namespace) or of a workflow with a `pw` namespace (e.g. namespace=('nscf', 'pw'))."""
    inputs = node.base.links.get_incoming().nested()
    for name in namespace:
        inputs = inputs[name]
    return _pw_signature(inputs['structure'], inputs['pseudos'], inputs['parameters'].get_dict())

def same_pw_setup(pw_manager, node, namespace=()):
    """True if `node` was run with the structure, pseudopotentials and parameters of `pw_manager`.

    We compare only what the manager knows (e.g. nothing but the parameters if we start from a parent folder).
    If the inputs of the node cannot be read, they are not the same.
    """
    try:
        expected = pw_signature_from_manager(pw_manager)
        found = pw_signature_from_node(node, namespace)
    except Exception:
        return False
    for key, known in (('structure', getattr(pw_manager, 'structure', None) is not None),
                       ('pseudos', bool(getattr(pw_manager, 'pseudos', None))),
                       ('parameters', bool(pw_manager.parameters))):
        if not known:
            expected.pop(key), found.pop(key)
    return expected == found

def check_same_yambo(node, params_to_calc, k_mesh_to_calc, up_to_p2y=False, additional=None, bands=None, nscf_manager=None):
    """Return the pk of `node` if it is a successful YamboWorkflow equivalent to the one we want to run, else False.

    :param params_to_calc: the yambo variables of the calculation we want to run.
    :param nscf_manager: the `nscf` PwBaseWorkChainInputManager of the calculation we want to run: if given,
        the structure, the pseudopotentials and the parameters of the NSCF of `node` must be the same.
    :param up_to_p2y: we look for a calculation we can use as parent of the p2y one (the variables do not matter).
    """
    additional = additional or []
    already_done = False
    try:
        if node.is_finished_ok:
            if nscf_manager is not None and not same_pw_setup(nscf_manager, node, namespace=('nscf', 'pw')):
                return False
            same_k = k_mesh_to_calc == node.inputs.nscf__kpoints.get_kpoints_mesh()
            old_params = None
            for key in ('qp__yambo__parameters', 'bse__yambo__parameters', 'yres__yambo__parameters'):
                if key in node.inputs:
                    old_params = node.inputs[key].get_dict()['variables']
                    break
            if old_params is None:
                return already_done
            was_p2y = False
            for key in ('qp__yambo__settings', 'bse__yambo__settings', 'yres__yambo__settings'):
                if key in node.inputs:
                    was_p2y = node.inputs[key].get_dict().pop('INITIALISE', False)
                    break
            if bands : 
                enough_b = node.inputs.nscf__pw__parameters.get_dict()['SYSTEM']['nbnd'] >= bands
            else:
                enough_b = True
            remote_is_used = up_to_p2y and not node.called[0].called[0].outputs.remote_folder.is_empty

            if up_to_p2y:
                if same_k and remote_is_used and enough_b:
                    return node.pk
                return False

            if not (same_k and not was_p2y and enough_b and same_yambo_variables(params_to_calc, old_params)):
                return False
            if 'QPkrange' in params_to_calc:
                # the QP we want must be among the ones parsed by the old calculation
                parsed_before = node.inputs.additional_parsing.get_list() if hasattr(node.inputs, 'additional_parsing') else []
                if not all(i in parsed_before for i in additional):
                    return False
            already_done = node.pk
    
    except Exception:
        already_done = False
    
    return already_done

def is_nscf_valid(parent_node, input_manager, bands=None):
    """Expanded NSCF validity: kpoints, nbnd, structure, pseudos, ecut (+ finished/!empty).

    Returns True if the NSCF parent found from *parent_node* can be reused for
    *input_manager*, False if it must be redone. Conservative: on any
    unresolvable check it returns False (redo).
    """
    try:
        # resolve required bands from Yambo BndsRnXp/GbndRnge if not given
        if bands is None:
            try:
                bands = input_manager.Nb
            except Exception:
                bands = None

        # locate NSCF (and SCF for fallback)
        try:
            nscf_node = find_pw_parent(parent_node, calc_type=['nscf'])
        except Exception:
            # parent_node itself may be the NSCF PwCalculation
            nscf_node = parent_node if 'quantumespresso.pw' in getattr(parent_node, 'process_type', '') else None
            if nscf_node is None:
                return False

        if not getattr(nscf_node, 'is_finished_ok', False):
            return False
        try:
            if nscf_node.outputs.remote_folder.is_empty:
                return False
        except Exception:
            pass

        # kpoints mesh+shift exact equality
        try:
            k_mesh_to_calc = input_manager.nscf.get_mesh()
        except Exception:
            try:
                k_mesh_to_calc = input_manager.nscf.kpoints.get_kpoints_mesh()
            except Exception:
                k_mesh_to_calc = None
        if k_mesh_to_calc is not None:
            try:
                same_k = k_mesh_to_calc == nscf_node.inputs.kpoints.get_kpoints_mesh()
                if not same_k:
                    return False
            except Exception:
                # legacy nscf__kpoints attribute
                try:
                    same_k = k_mesh_to_calc == nscf_node.inputs.nscf__kpoints.get_kpoints_mesh()
                    if not same_k:
                        return False
                except Exception:
                    pass

        # nbnd
        if bands is not None:
            try:
                nscf_nbnd = nscf_node.inputs.parameters.get_dict()['SYSTEM']['nbnd']
                if nscf_nbnd < bands:
                    return False
            except Exception:
                try:
                    nscf_nbnd = nscf_node.inputs.nscf__pw__parameters.get_dict()['SYSTEM']['nbnd']
                    if nscf_nbnd < bands:
                        return False
                except Exception:
                    pass

        # structure
        try:
            input_struct = getattr(input_manager.nscf, 'structure', None) or getattr(input_manager.scf, 'structure', None)
            parent_struct = nscf_node.inputs.structure
            if input_struct is not None and parent_struct is not None:
                # prefer UUID equality, fallback to formula+cell
                if getattr(input_struct, 'uuid', None) and getattr(parent_struct, 'uuid', None):
                    if input_struct.uuid != parent_struct.uuid:
                        return False
                elif input_struct.get_formula() != parent_struct.get_formula():
                    return False
        except Exception:
            pass

        # ecutwfc / ecutrho
        try:
            input_ecutwfc = input_manager.nscf.get_ecutwfc()
            parent_ecutwfc = nscf_node.inputs.parameters.get_dict()['SYSTEM'].get('ecutwfc')
            if input_ecutwfc is not None and parent_ecutwfc is not None and input_ecutwfc != parent_ecutwfc:
                return False
            input_ecutrho = input_manager.nscf.parameters.get('SYSTEM', {}).get('ecutrho')
            parent_ecutrho = nscf_node.inputs.parameters.get_dict()['SYSTEM'].get('ecutrho')
            if input_ecutrho is not None and parent_ecutrho is not None and input_ecutrho != parent_ecutrho:
                return False
        except Exception:
            pass

        # pseudos (kind names + md5)
        try:
            input_pseudos = getattr(input_manager.nscf, 'pseudos', None) or getattr(input_manager.scf, 'pseudos', None)
            parent_pseudos = nscf_node.inputs.pseudos
            if input_pseudos is not None and parent_pseudos is not None:
                # compare kind names
                if set(input_pseudos.keys()) != set(parent_pseudos.keys()):
                    return False
        except Exception:
            pass

        return True
    except Exception:
        return False


def check_same_pw(node, k_mesh_to_calc, already_done, bands = None, input_manager = None):
    """Find in `node` the NSCF/SCF calculations we can use as parents. They must be successful and not empty;
    if `input_manager` (a YamboInputManager) is given, they must have the same structure, pseudopotentials and parameters.
    """
    parent_nscf = False
    parent_scf = False
    enough_b = True
    try:
        if not already_done:
            
            parent_nscf_try = find_pw_parent(node, calc_type=['nscf'])
            parent_scf_try = find_pw_parent(node, calc_type=['scf'])
            if input_manager is not None:
                if not same_pw_setup(input_manager.nscf, parent_nscf_try):
                    parent_nscf_try = None
                if not same_pw_setup(input_manager.scf, parent_scf_try):
                    parent_scf_try = None
            same_k = k_mesh_to_calc == node.inputs.nscf__kpoints.get_kpoints_mesh()
            if bands : enough_b = node.inputs.nscf__pw__parameters.get_dict()['SYSTEM']['nbnd'] >= bands
            if node.is_finished_ok and parent_nscf_try is not None:
                try:
                    y = node.outputs.retrieved._repository._repo_folder.abspath+'/path/'
                    #if 'ns.db1' in  os.listdir(y) and same_k and not node.outputs.remote_folder.is_empty:
                    if same_k and not node.outputs.remote_folder.is_empty:
                        if enough_b:
                            parent_nscf = node.pk      
 
                except (AttributeError, KeyError): # no retrieved/remote folder
                    pass
            if parent_nscf_try is not None and same_k and parent_nscf_try.is_finished_ok and not parent_nscf_try.outputs.remote_folder.is_empty: 
                if enough_b:
                    parent_nscf = parent_nscf_try.pk     
            if parent_scf_try is not None and parent_scf_try.is_finished_ok and not parent_scf_try.outputs.remote_folder.is_empty: 
                if enough_b:
                    parent_scf = parent_scf_try.pk 
    
    except Exception as exc: # no PW calculations in this node
        LOGGER.debug('no usable parent in node %s: %s', getattr(node, 'pk', None), exc)
    
    return parent_nscf, parent_scf   

def search_in_group(YamboWorkflow_inputs, 
                    YamboWorkflow_group,
                    up_to_p2y = False,
                    calc_type='qp'):
    """Look in the group for calculations we can reuse for the inputs of a YamboInputManager.

    :return: `already_done` (pk of an equivalent successful YamboWorkflow, or False), `parent_nscf` and `parent_scf`
        (pks of PW calculations, with the same structure, pseudopotentials and parameters, usable as parents, or False).
    """
    already_done = False
    parent_nscf = False
    parent_scf = False
    # YamboWorkflow_inputs is a YamboInputManager
    k_mesh_to_calc = YamboWorkflow_inputs.nscf.get_mesh()
    yambo_manager = YamboWorkflow_inputs.bse if 'bse' in calc_type else YamboWorkflow_inputs.qp
    params_to_calc = yambo_manager.parameters['variables']
    # `additional_parsing` can be either an orm.List node (legacy builder inputs)
    # or a plain python list (YamboInputManager): support both.
    additional_parsing = getattr(YamboWorkflow_inputs, 'additional_parsing', None)
    if additional_parsing is None:
        additional = []
    elif isinstance(additional_parsing, (list, tuple)):
        additional = list(additional_parsing)
    else:
        try:
            additional = additional_parsing.get_list()
        except Exception:
            additional = []

    def _workflows(group_node):
        """The YamboWorkflows of a node of the group: itself or the ones called by a YamboConvergence."""
        if group_node.process_type == 'aiida.workflows:yambo.yambo.yamboconvergence':
            return list(group_node.called)
        if group_node.process_type == 'aiida.workflows:yambo.yambo.yambowf':
            return [group_node]
        return []

    for old in YamboWorkflow_group.nodes:
        for workflow in _workflows(old):
            already_done = check_same_yambo(workflow, params_to_calc, k_mesh_to_calc, up_to_p2y=up_to_p2y,
                                            additional=additional, nscf_manager=YamboWorkflow_inputs.nscf)
            if already_done:
                break
        if already_done:
            break
            
    for old in YamboWorkflow_group.nodes:
        for workflow in _workflows(old):
            nscf, scf = check_same_pw(workflow, k_mesh_to_calc, already_done, input_manager=YamboWorkflow_inputs)
            parent_scf = parent_scf or scf # we keep the first SCF we found, also if the following nodes have none
            if nscf:
                parent_nscf = nscf
                break
        if parent_nscf:
            break

    return already_done, parent_nscf, parent_scf  

@calcfunction
def store_quantity(quantity):
    #to be changed, but in this way data are created by a calcfunction and not by the WFL.
    quantity.store()
    return

    
