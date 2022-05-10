# -*- coding: utf-8 -*-
from __future__ import absolute_import
from curses import meta
import os
import time

from aiida import orm
from aiida.orm import RemoteData,BandsData
from aiida.orm import Dict,Int,List

from aiida.engine import WorkChain, while_, if_
from aiida.engine import ToContext

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.common.types import ElectronicType, SpinType

from aiida_yambo.utils.common_helpers import *
from aiida_yambo.workflows.yamborestart import YamboRestart

from aiida_yambo.utils.defaults.create_defaults import *

from aiida_yambo.workflows.utils.helpers_yambowf import *
from aiida.plugins import DataFactory
LegacyUpfData = DataFactory('upf')
SingleFileData = DataFactory('singlefile')

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

@calcfunction
def merge_QP(filenames_List,output_name): #just to have something that works, but it is not correct to proceed this way
        string_run = 'yambopy mergeqp'
        for i in filenames_List.get_list():
            j = load_node(i).outputs.QP_db._repository._repo_folder.abspath+'/path/ndb.QP'
            string_run+=' '+j
        string_run+=' -o '+output_name.value
        print(string_run)
        os.system(string_run)
        time.sleep(10)
        QP_db = SingleFileData(output_name.value)
        return QP_db

def QP_mapper(ywfl,tol=1,full_bands=False):
    fermi = find_pw_parent(ywfl).outputs.output_parameters.get_dict()['fermi_energy']
    SOC = find_pw_parent(ywfl).outputs.output_parameters.get_dict()['spin_orbit_calculation']
    nelectrons = find_pw_parent(ywfl).outputs.output_parameters.get_dict()['number_of_electrons']
    kpoints = find_pw_parent(ywfl).outputs.output_band.get_kpoints()
    bands = find_pw_parent(ywfl).outputs.output_band.get_bands()
    
    if SOC:
        valence = int(nelectrons) - 1
        conduction = valence + 2
    else:
        valence = int(nelectrons/2) + int(nelectrons%2)
        conduction = valence + 1
    print('valence: {}'.format(valence))    
    
    #tol = 10*(-max(bands[:,valence-1])+(min(bands[:,conduction-1])))/2
    mid_gap_energy = max(bands[:,valence-1])+(min(bands[:,conduction-1])-max(bands[:,valence-1]))/2
    
    QP = []
    print(tol,QP,np.where(abs(bands-mid_gap_energy)<tol)[1])
    if len(np.where(abs(bands-mid_gap_energy)<tol)[1])<2:
        print('#1 redoing analysis incrementing the energy range of 50%. new tol={}'.format(tol*1.5))
        tol=tol*1.5
    
        return QP_mapper(ywfl,tol=tol,full_bands=full_bands)
    
    print('test passato')
        
    if not full_bands:
        for i,j in zip(np.where(abs(bands-mid_gap_energy)<tol)[0],np.where(abs(bands-mid_gap_energy)<tol)[1]):
            QP.append([i+1,i+1,j+1,j+1])
    else:
        b_min = np.where(abs(bands-mid_gap_energy)<tol)[1].min()
        b_max = np.where(abs(bands-mid_gap_energy)<tol)[1].max()
        for i in range(len(kpoints)):
            QP.append([i+1,i+1,b_min+1,b_max+1])
                
    v,c = False,False
    if len(QP)<1:
        print('#2 redoing analysis incrementing the energy range of 50%. new tol={}'.format(tol*1.5))
        return QP_mapper(ywfl,tol=tol*1.5,full_bands=full_bands)
    
    for i in QP:
        if valence >= i[-1]: v = True
        if conduction <= i[-1]: c = True
        if valence >= i[-2]: v = True
        if conduction <= i[-2]: c = True
    
    if not v or not c:
        print('#3 redoing analysis incrementing the energy range of 50%. new tol={}'.format(tol*1.5))
        return QP_mapper(ywfl,tol=tol*1.5,full_bands=full_bands)
        
    print('Found {} QPs'.format(len(QP
                                   )))
    
    plt.plot(bands-mid_gap_energy,'-o')
    plt.plot(np.where(abs(bands-mid_gap_energy)<tol)[0],bands[np.where(abs(bands-mid_gap_energy)<tol)]-mid_gap_energy,'o',label='to be computed explicitely')
    plt.ylim(-0.25,0.25)
    
    print('Fermi level={} eV'.format(fermi))
    
    return QP

def QP_subset_groups(nnk_i,nnk_f,bb_i,bb_f,qp_for_subset):
    if bb_f-bb_i<nnk_f-nnk_i:
        n = int(min(qp_for_subset,nnk_f-nnk_i+1)/3)+1
        m = bb_f-bb_i+1
    else:
        m = int(min(qp_for_subset,bb_f-bb_i+1)/3)+1
        n = nnk_f-nnk_i+1

    print(n,m)

    #n=58  #length of a set
    #m=3
    groups=[]
    sets_k = int((nnk_f-nnk_i)/n+1)
    sets_b = int((bb_f-bb_i)/m+1)
    print(sets_k,sets_b)
    for i in range(sets_k):
        k_i=1+i*n + (nnk_i-1)
        k_f=k_i+n-1 
        if k_f > nnk_f: k_f = nnk_f
        for j in range(sets_b):
            b_i=1+j*m + (bb_i-1)
            b_f=b_i+m-1
            if b_f > bb_f: b_f = bb_f

            print(k_i,k_f,b_i,b_f)
            groups.append([[k_i,k_f,b_i,b_f]])

    return groups

def QP_list_merger(l=[],qp_per_subset=10):
    ll=[]
    lg = []
    split = False
    First = True
    order=0
    for i in l:
        #print(i,(i[1]-i[0]+1)*(i[3]-i[2]+1),order)
        if First: 
            lg.append(i)
            First = False
            order +=(i[1]-i[0]+1)*(i[3]-i[2]+1)
        elif order + (i[1]-i[0]+1)*(i[3]-i[2]+1) < qp_per_subset:
            lg.append(i)
            order +=(i[1]-i[0]+1)*(i[3]-i[2]+1)
        else:
            ll.append(lg)
            lg = [i]
            order = (i[1]-i[0]+1)*(i[3]-i[2]+1) 
    ll.append(lg)
    return ll


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

        spec.expose_inputs(PwBaseWorkChain, namespace='scf', namespace_options={'required': False}, 
                            exclude = ['parent_folder'])

        spec.expose_inputs(PwBaseWorkChain, namespace='nscf', namespace_options={'required': False}, 
                            exclude = ['parent_folder'])

        spec.expose_inputs(YamboRestart, namespace='yres', namespace_options={'required': False}, 
                            exclude = ['parent_folder'])

        spec.expose_inputs(YamboRestart, namespace='qp', namespace_options={'required': False,'populate_defaults': False}, 
                            exclude = ['parent_folder'])

        spec.input("additional_parsing", valid_type=List, required = False,
                    help = 'list of additional quantities to be parsed: gap, homo, lumo, or used defined quantities -with names-[k1,k2,b1,b2], [k1,b1], excitons: [lowest, brightes]')
        
        spec.input("QP_subset_dict", valid_type=Dict, required = False,
                    help = 'subset of QP that you want to compute, useful if you need to obtain a large number of QP corrections')

        spec.input("parent_folder", valid_type=RemoteData, required = False,
                    help = 'scf, nscf or yambo remote folder')
  

##################################### OUTLINE ####################################

        spec.outline(cls.validate_parameters,
                    cls.start_workflow,
                    while_(cls.can_continue)(
                           cls.perform_next,
                    ),
                    if_(cls.post_processing_needed)(
                        cls.run_post_process,
                    ),
                    if_(cls.should_run_bse)(
                        cls.prepare_and_run_bse,
                    ),
                    cls.report_wf,)

##################################################################################

        spec.expose_outputs(YamboRestart)
        
        spec.output('output_ywfl_parameters', valid_type = Dict, required = False)
        spec.output('nscf_mapping', valid_type = Dict, required = False)

        spec.output('splitted_QP_calculations', valid_type = List, required = False)
        spec.output('merged_QP', valid_type = SingleFileData, required = False)
        
        spec.output('scissor', valid_type = List, required = False)

        spec.exit_code(300, 'ERROR_WORKCHAIN_FAILED',
                             message='The workchain failed with an unrecoverable error.')
    
    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from aiida_yambo.workflows.protocols import yambo as yamboworkflow_protocols
        return files(yamboworkflow_protocols) / 'yamboworkflow.yaml'
    
    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code,
        preprocessing_code,
        code,
        protocol_qe='fast',
        protocol='fast',
        structure=None,
        overrides={},
        parent_folder=None,
        NLCC=False,
        RIM_v=False,
        RIM_W=False,
        electronic_type=ElectronicType.METAL,
        spin_type=SpinType.NONE,
        initial_magnetic_moments=None,
        **_
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        if isinstance(code, str):
            
            preprocessing_code = orm.load_code(preprocessing_code)
            code = orm.load_code(code)

        if electronic_type not in [ElectronicType.METAL, ElectronicType.INSULATOR]:
            raise NotImplementedError(f'electronic type `{electronic_type}` is not supported.')

        if spin_type not in [SpinType.NONE, SpinType.COLLINEAR]:
            raise NotImplementedError(f'spin type `{spin_type}` is not supported.')

        inputs = cls.get_protocol_inputs(protocol, overrides={})

        meta_parameters = inputs.pop('meta_parameters',{})
        
        builder = cls.get_builder()

        overrides_scf = overrides.pop('scf',{})
        overrides_nscf = overrides.pop('nscf',{})
        overrides_yres = overrides.pop('yres',{})

        for override in [overrides_scf,overrides_nscf]:
            override['clean_workdir'] = override.pop('clean_workdir',False) #required to have a valid parent folder
            
            if 'pseudo_family' in override.keys():
                if 'PseudoDojo' in override['pseudo_family']: NLCC = True

        try:
            pw_parent = find_pw_parent(take_calc_from_remote(parent_folder))
            PW_cutoff = pw_parent.inputs.parameters.get_dict()['SYSTEM']['ecutwfc']
            nelectrons = int(pw_parent.outputs.output_parameters.get_dict()['number_of_electrons'])
        except:
            nelectrons, PW_cutoff = periodical(structure.get_ase())
            overrides_yres['nelectrons'] = nelectrons
            overrides_yres['PW_cutoff'] = PW_cutoff

        #########SCF and NSCF PROTOCOLS 
        builder.scf = PwBaseWorkChain.get_builder_from_protocol(
                pw_code,
                structure,
                protocol=protocol_qe,
                overrides=overrides_scf,
                electronic_type=electronic_type,
                spin_type=spin_type,
                initial_magnetic_moments=initial_magnetic_moments,
                )

        builder.nscf = PwBaseWorkChain.get_builder_from_protocol(
                pw_code,
                structure,
                protocol=protocol_qe,
                overrides=overrides_nscf,
                electronic_type=electronic_type,
                spin_type=spin_type,
                initial_magnetic_moments=initial_magnetic_moments,
                )

        builder.nscf['kpoints'] = KpointsData()
        builder.nscf['kpoints'].set_cell_from_structure(builder.scf['pw']['structure'])
        builder.nscf['kpoints'].set_kpoints_mesh_from_density(meta_parameters['k_density'],force_parity=True)

        builder.scf['pw']['parameters']['SYSTEM']['force_symmorphic'] = True #required in yambo
        builder.nscf['pw']['parameters']['SYSTEM']['force_symmorphic'] = True #required in yambo
        
        nelectrons = 0
        for site in builder.nscf['pw']['structure'].sites:
            nelectrons += builder.nscf['pw']['pseudos'][site.kind_name].z_valence
        
        overrides_yres['nelectrons'] = nelectrons
        overrides_yres['PW_cutoff'] = builder.nscf['pw']['parameters'].get_dict()['SYSTEM']['ecutwfc']

        #########YAMBO PROTOCOL, with or without parent folder.
        if not parent_folder: 
            parent_folder = 'YWFL_scratch'
        else:
            builder.parent_folder = parent_folder
            parent_folder = 'YWFL_super_parent'

        yres_builder = YamboRestart.get_builder_from_protocol(
                preprocessing_code=preprocessing_code,
                code=code,
                protocol=protocol,
                parent_folder=parent_folder,
                overrides=overrides_yres,
                NLCC=NLCC,
                RIM_v=RIM_v,
                RIM_W=RIM_W,
            )

        builder.yres = yres_builder

        if 'BndsRnXp' in builder.yres['yambo']['parameters'].get_dict()['variables'].keys():
            yambo_bandsX = builder.yres['yambo']['parameters'].get_dict()['variables']['BndsRnXp'][0][-1]
        else: 
            yambo_bandsX = 0 
        
        if 'GbndRnge' in builder.yres['yambo']['parameters'].get_dict()['variables'].keys():
            yambo_bandsSc = builder.yres['yambo']['parameters'].get_dict()['variables']['GbndRnge'][0][-1]
        else: 
            yambo_bandsSc = 0 
        
        gwbands = max(yambo_bandsX,yambo_bandsSc)

        parameters_scf = builder.scf['pw']['parameters'].get_dict()
        parameters_nscf = builder.nscf['pw']['parameters'].get_dict()
        

        parameters_scf['SYSTEM']['ecutrho'] = max(parameters_scf['SYSTEM'].pop('ecutrho',0),4*parameters_scf['SYSTEM']['ecutwfc'])
        
        parameters_nscf['SYSTEM']['ecutwfc'] = parameters_scf['SYSTEM']['ecutwfc']
        parameters_nscf['SYSTEM']['ecutrho'] = parameters_scf['SYSTEM']['ecutrho']        
        
        parameters_nscf['CONTROL']['calculation'] = 'nscf'
        parameters_scf['SYSTEM'].pop('nbnd',0) #safety measure, for some system creates chaos in conjunction with smearing


        parameters_nscf['SYSTEM']['nbnd'] = max(parameters_nscf['SYSTEM'].pop('nbnd',0),gwbands)
        builder.nscf['pw']['parameters'] = Dict(dict = parameters_nscf)
        builder.scf['pw']['parameters'] = Dict(dict = parameters_scf)

        print('\nkpoint mesh for nscf: {}'.format(builder.nscf['kpoints'].get_kpoints_mesh()[0]))

        return builder


    def validate_parameters(self):

        if hasattr(self.inputs, 'qp'):
            self.ctx.yambo_inputs = self.exposed_inputs(YamboRestart, 'qp')
        else:
            self.ctx.yambo_inputs = self.exposed_inputs(YamboRestart, 'yres')        
        #quantumespresso common inputs
        self.ctx.scf_inputs = self.exposed_inputs(PwBaseWorkChain, 'scf')
        self.ctx.nscf_inputs = self.exposed_inputs(PwBaseWorkChain, 'nscf')
        
        #quantumespresso input parameters check from parents, if any.
        scf_params, nscf_params, redo_nscf, gwbands, messages = quantumespresso_input_validator(self.inputs,)
        if scf_params: self.ctx.scf_inputs.pw.parameters = scf_params
        if nscf_params: self.ctx.nscf_inputs.pw.parameters = nscf_params
        self.ctx.redo_nscf = redo_nscf
        self.ctx.gwbands = gwbands
        #for i in messages:
            #self.report(i)

        if hasattr(self.inputs,'QP_subset_dict'): self.ctx.QP_subsets = self.inputs.QP_subset_dict.get_dict()
        
    def start_workflow(self):
        """Initialize the workflow, set the parent calculation

        This function sets the parent, and its type
        there is no submission done here, only setting up the neccessary inputs the workchain needs in the next
        steps to decide what are the subsequent steps"""
        try:

            parent = take_calc_from_remote(self.inputs.parent_folder,level=-1)
            

            if parent.process_type=='aiida.workflows:quantumespresso.pw.base':
                parent = parent.outputs.remote_folder.creator
            
            if parent.outputs.remote_folder.is_empty: 
                for i in range(2):
                    try:
                        parent = find_pw_parent(parent)
                        if parent.outputs.remote_folder.is_empty and i == 1: continue
                    except:
                        break
                    
            if parent.outputs.remote_folder.is_empty: raise

            if parent.process_type=='aiida.calculations:quantumespresso.pw':

                if parent.inputs.parameters.get_dict()['CONTROL']['calculation'] == 'scf' or parent.inputs.parameters.get_dict()['CONTROL']['calculation'] == 'relax' or \
                parent.inputs.parameters.get_dict()['CONTROL']['calculation'] == 'vc-relax':
                    self.ctx.calc_to_do = 'nscf'
                    self.ctx.redo_nscf = False

                elif parent.inputs.parameters.get_dict()['CONTROL']['calculation'] == 'nscf':
                    if self.ctx.redo_nscf or parent.inputs.parameters.get_dict()['SYSTEM']['nbnd'] < self.ctx.gwbands:
                        parent = find_pw_parent(parent, calc_type = ['scf'])
                        self.report('Recomputing NSCF step, not enough bands. Starting from scf at pk < {} >'.format(parent.pk))
                        self.ctx.calc_to_do = 'nscf'
                        self.ctx.redo_nscf = False
                    else:
                        self.ctx.calc_to_do = 'yambo'

            elif parent.process_type=='aiida.workflows:yambo.yambo.yambowf':
                    parent=parent.outputs.remote_folder.creator
                    self.report('parent is: {}'.format(parent.process_type))
                    nbnd = find_pw_parent(parent, calc_type = ['nscf']).inputs.parameters.get_dict()['SYSTEM']['nbnd']
                    if self.ctx.redo_nscf or nbnd < self.ctx.gwbands:
                        parent = find_pw_parent(parent, calc_type = ['scf'])
                        self.report('Recomputing NSCF step, not enough bands. Starting from scf at pk < {} >'.format(parent.pk))
                        self.ctx.calc_to_do = 'nscf'
                        self.ctx.redo_nscf = False
                    else:
                        self.ctx.calc_to_do = 'yambo'

                    if self.ctx.calc_to_do == 'yambo' and hasattr(self.inputs,'QP_subset_dict'): self.ctx.calc_to_do = 'QP splitter'

            elif parent.process_type=='aiida.calculations:yambo.yambo':
                    nbnd = find_pw_parent(parent, calc_type = ['nscf']).inputs.parameters.get_dict()['SYSTEM']['nbnd']
                    if self.ctx.redo_nscf or nbnd < self.ctx.gwbands:
                        parent = find_pw_parent(parent, calc_type = ['scf'])
                        self.report('Recomputing NSCF step, not enough bands. Starting from scf at pk < {} >'.format(parent.pk))
                        self.ctx.calc_to_do = 'nscf'
                        self.ctx.redo_nscf = False
                    else:
                        self.ctx.calc_to_do = 'yambo'
                    
                    if self.ctx.calc_to_do == 'yambo' and hasattr(self.inputs,'QP_subset_dict'): self.ctx.calc_to_do = 'QP splitter'

            else:
                self.ctx.previous_pw = False
                self.ctx.calc_to_do = 'scf'
                self.report('no valid input calculations, so we will start from scratch')
            
            self.ctx.calc = parent

        except:

            self.report('no previous pw calculation found, we will start from scratch')
            self.ctx.calc_to_do = 'scf'
        
        self.ctx.splitted_QP = []
        self.ctx.qp_splitter = 0
        self.report(" workflow initilization step completed.")

    def can_continue(self):

        """This function checks the status of the last calculation and determines what happens next, including a successful exit"""

        if self.ctx.calc_to_do != 'workflow is finished':
            self.report('the workflow continues with a {} calculation'.format(self.ctx.calc_to_do))
            return True
        else:
            self.report('workflow is finished')
            return False


    def perform_next(self):
        """This function  will submit the next step, depending on the information provided in the context

        The next step will be a yambo calculation if the provided inputs are a previous yambo/p2y run
        Will be a PW scf/nscf if the inputs do not provide the NSCF or previous yambo parent calculations"""
  
        try:
            calc = self.ctx.calc
            if not calc.is_finished_ok:
                if '.yambo' in calc.process_type:
                    if 'COPY_DBS' in self.ctx.yambo_inputs.yres.yambo.settings.get_dict().keys():
                        if self.ctx.yambo_inputs.yres.yambo.settings.get_dict()['COPY_DBS']: pass
                self.report("last calculation failed, exiting the workflow")
                return self.exit_codes.ERROR_WORKCHAIN_FAILED
        except:
            pass

        self.report('performing a {} calculation'.format(self.ctx.calc_to_do))

        if self.ctx.calc_to_do == 'scf':

            self.ctx.scf_inputs.metadata.call_link_label = 'scf'
            future = self.submit(PwBaseWorkChain, **self.ctx.scf_inputs)

            self.ctx.calc_to_do = 'nscf'

        elif self.ctx.calc_to_do == 'nscf':

            self.ctx.nscf_inputs.pw.parent_folder = self.ctx.calc.outputs.remote_folder
            
            self.ctx.nscf_inputs.metadata.call_link_label = 'nscf'  
            future = self.submit(PwBaseWorkChain, **self.ctx.nscf_inputs)

            self.ctx.calc_to_do = 'yambo'

        elif self.ctx.calc_to_do == 'yambo':

            self.ctx.yambo_inputs['parent_folder'] = self.ctx.calc.outputs.remote_folder
            
            if hasattr(self.inputs, 'additional_parsing'):
                self.report('updating yambo parameters to parse more results')
                mapping, yambo_parameters = add_corrections(self.ctx.yambo_inputs, self.inputs.additional_parsing.get_list())
                self.ctx.mapping = mapping
                self.report(mapping)
                self.ctx.yambo_inputs.yambo.parameters = yambo_parameters

            self.ctx.yambo_inputs.metadata.call_link_label = 'yambo'
            future = self.submit(YamboRestart, **self.ctx.yambo_inputs)

            if hasattr(self.inputs,'QP_subset_dict'):
                self.ctx.calc_to_do = 'QP splitter'
                self.ctx.QP_subsets = self.inputs.QP_subset_dict.get_dict()
            else:
                self.ctx.calc_to_do = 'workflow is finished'
        
        elif self.ctx.calc_to_do == 'QP splitter':
            QP = {}
            if self.ctx.qp_splitter == 0:
                calc = self.ctx.calc
                if not calc.is_finished_ok:
                    self.report("last calculation failed, exiting the workflow")
                    return self.exit_codes.ERROR_WORKCHAIN_FAILED
                    
                self.ctx.yambo_inputs.parent_folder= self.ctx.calc.outputs.remote_folder
                
                self.ctx.yambo_inputs.yambo.parameters = take_calc_from_remote(self.ctx.yambo_inputs['parent_folder'],level=-1).inputs.parameters
                self.ctx.yambo_inputs.yambo.settings = update_dict(self.ctx.yambo_inputs.yambo.settings, 'COPY_DBS', True)
                self.ctx.yambo_inputs.clean_workdir = Bool(True)
                mapping = gap_mapping_from_nscf(find_pw_parent(take_calc_from_remote(self.ctx.yambo_inputs['parent_folder'],level=-1)).pk)
                self.ctx.mapping = mapping

                if 'range_QP' in self.ctx.QP_subsets.keys(): #the name can be changed..
                    Energy_region = max(self.ctx.QP_subsets['range_QP'],mapping['nscf_gap_eV']*1.2)
                    self.report('range of energy for QP: {} eV'.format(Energy_region))
                    self.ctx.QP_subsets['explicit'] = QP_mapper(self.ctx.calc,tol = Energy_region,full_bands=self.ctx.QP_subsets.pop('full_bands',False))

                if not 'subsets' in self.ctx.QP_subsets.keys():
                    if 'explicit' in self.ctx.QP_subsets.keys():
                        self.ctx.QP_subsets['subsets'] = QP_list_merger(self.ctx.QP_subsets['explicit'],self.ctx.QP_subsets['qp_per_subset'])
                    elif 'boundaries' in self.ctx.QP_subsets.keys():
                        self.ctx.QP_subsets['subsets'] = QP_subset_groups(1,mapping['number_of_kpoints'],self.ctx.QP_subsets['boundaries']['bi'],self.ctx.QP_subsets['boundaries']['bf'],self.ctx.QP_subsets['qp_per_subset'])
                self.report('subsets: {}'.format(self.ctx.QP_subsets['subsets']))

            for i in range(1,1+self.ctx.QP_subsets['parallel_runs']):
                if len(self.ctx.QP_subsets['subsets']) > 0:
                    self.ctx.yambo_inputs.yambo.parameters = update_dict(self.ctx.yambo_inputs.yambo.parameters,['QPkrange'],[[self.ctx.QP_subsets['subsets'].pop(),'']],sublevel='variables')

                    self.ctx.yambo_inputs.metadata.call_link_label = 'yambo_QP_splitted_{}'.format(i+self.ctx.qp_splitter)
                    future = self.submit(YamboRestart, **self.ctx.yambo_inputs)
                    self.report('launchiing YamboRestart <{}> for QP, iteration#{}'.format(future.pk,i+self.ctx.qp_splitter))
                    self.ctx.splitted_QP.append(future.uuid)
                    QP[str(i+1)] = future
                else:
                    self.ctx.calc_to_do = 'workflow is finished'
            
            self.ctx.qp_splitter += self.ctx.QP_subsets['parallel_runs']

            if len(self.ctx.QP_subsets['subsets']) == 0: self.ctx.calc_to_do = 'workflow is finished'

            return ToContext(QP) #wait for all splitted calculations....

        return ToContext(calc = future)
    
    def post_processing_needed(self):
        #in case of multiple QP calculations, yes
        if len(self.ctx.splitted_QP) > 0 and not self.ctx.yambo_inputs.yambo.settings.get_dict()['INITIALISE']:
            self.report('merge QP needed')
            return True
        self.report('no post processing needed')
        return False

    def run_post_process(self):
        #merge
        self.report('run merge QP')
        splitted = store_List(self.ctx.splitted_QP)
        self.out('splitted_QP_calculations', splitted)
        output_name = Str(self.ctx.calc.outputs.retrieved._repository._repo_folder.abspath+'/path/ndb.QP_merged')
        self.ctx.QP_db = merge_QP(splitted,output_name)
        
        self.out('merged_QP',self.ctx.QP_db)

        return

    def should_run_bse(self):
        #in case of BSE on top of GW just done, yes
        if hasattr(self.inputs, 'qp') and hasattr(self.ctx,'QP_db') and not self.ctx.yambo_inputs.yambo.settings.get_dict()['INITIALISE']:
            self.report('We run BSE@GW')
            return True
        return False

    def prepare_and_run_bse(self):
        
        self.ctx.yambo_inputs = self.exposed_inputs(YamboRestart, 'yres') 
        bse_params = self.ctx.yambo_inputs.yambo.parameters.get_dict()

        self.ctx.yambo_inputs.yambo.QP_corrections = self.ctx.QP_db
        bse_params['variables']['KfnQPdb'] = "E < ./ndb.QP"

        self.ctx.yambo_inputs.parent_folder = self.ctx.calc.outputs.remote_folder
        self.ctx.yambo_inputs.yambo.settings = update_dict(self.ctx.yambo_inputs.yambo.settings, 'COPY_DBS', True)

        BSE_map = QP_analyzer(self.ctx.calc.pk, self.ctx.QP_db,self.ctx.mapping)

        bse_params['variables']['BSEBands'] = [[BSE_map['v_min'],BSE_map['c_max']],'']
        if not 'BSEQptR' in bse_params['variables'].keys():
            bse_params['variables']['BSEQptR'] = [[BSE_map['q_ind'],BSE_map['q_ind']],'']

        self.ctx.yambo_inputs.yambo.parameters = Dict(dict=bse_params)
        
        future = self.submit(YamboRestart, **self.ctx.yambo_inputs)

        return ToContext(bse = future) 

    def report_wf(self):

        #self.report('Final step.')

        calc = self.ctx.calc
        if calc.is_finished_ok:

            if hasattr(self.inputs, 'additional_parsing'):
                #self.report('parsing additional quantities')
                mapping, yambo_parameters = add_corrections(self.ctx.yambo_inputs, self.inputs.additional_parsing.get_list())
                parsed = additional_parsed(calc, self.inputs.additional_parsing.get_list(), mapping)
                self.out('nscf_mapping', store_Dict(mapping))
                if hasattr(self.ctx,'bse'):
                    if self.ctx.bse.is_finished_ok:
                        parsed_bse = additional_parsed(self.ctx.bse, self.inputs.additional_parsing.get_list(), mapping)
                        parsed.update(parsed_bse)
                    else:
                        self.report("workflow NOT completed successfully")
                        return self.exit_codes.ERROR_WORKCHAIN_FAILED
                
                self.out('output_ywfl_parameters', store_Dict(parsed))

                if 'scissor' in self.inputs.additional_parsing.get_list(): #in the future, also needed support for mergeqp, multiple calculations.
                    try:
                        if hasattr(self.ctx,'QP_db'):
                            b = QP_bands_interface(node=Int(self.ctx.calc.pk), QP_merged = self.ctx.QP_db,mapping = Dict(dict = mapping))
                        else:
                            b = QP_bands_interface(node=Int(self.ctx.calc.pk), mapping = Dict(dict = mapping))
                        self.report('electronic band structure computed by interpolation')
                        for k,v in b.items():
                            self.out(k, v)
                    except:
                        self.report('fail in the scissor evaluation')

            if hasattr(self.ctx,'bse'):
                if self.ctx.bse.is_finished_ok:
                    self.out_many(self.exposed_outputs(self.ctx.bse,YamboRestart))
            else:
                self.out_many(self.exposed_outputs(calc,YamboRestart))
                
            self.report("workflow completed successfully")
        else:
            self.report("workflow NOT completed successfully")
            return self.exit_codes.ERROR_WORKCHAIN_FAILED