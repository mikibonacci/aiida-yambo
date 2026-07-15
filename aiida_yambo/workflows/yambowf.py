# -*- coding: utf-8 -*-
from __future__ import absolute_import
#from curses import meta
import os
import time

from aiida import orm
from ase import units

from aiida.engine import WorkChain, while_, if_
from aiida.engine import ToContext

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.common.types import ElectronicType, SpinType

from aiida_yambo.utils.common_helpers import *
from aiida_yambo.workflows.yamborestart import YamboRestart

from aiida_yambo.utils.defaults.create_defaults import *

from aiida_yambo.workflows.utils.helpers_yambowf import *
from aiida_yambo.workflows.utils.extend_QPDB import *

from aiida.plugins import DataFactory

import pathlib
import tempfile

LegacyUpfData = DataFactory('core.upf')
SingleFileData = DataFactory('core.singlefile')

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

#### IMPORT THE NEW BASEMODEL MANAGER FOR INPUTS ####
from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager

def clean(node):
    cleaned_calcs = []

    for called_descendant in node.called_descendants:
        if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    if not called_descendant.is_finished_ok:
                        called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                        cleaned_calcs.append(called_descendant.pk)
                except:
                    pass

    if cleaned_calcs:
        return "cleaned remote folders of calculations: {}".format(join(map(str, cleaned_calcs)))

def sanity_check_QP(v,c,input_db,output_db,create=True):
    d = xarray.open_dataset(input_db,engine='netcdf4')
    wrong = np.where(abs(d.QP_E[:,0]-d.QP_Eo[:])*units.Ha>5)
    #v,c = 29,31
    v_cond = np.where((d.QP_table[0] <= v) & (abs(d.QP_E[:,0]-d.QP_Eo[:])*units.Ha<5))
    c_cond = np.where((d.QP_table[0] >= c) & (abs(d.QP_E[:,0]-d.QP_Eo[:])*units.Ha<5))

    #fix with a fit
    fit_v = np.polyfit(d.QP_Eo[v_cond[0]],d.QP_E[v_cond[0],0],deg=1)
    fit_c = np.polyfit(d.QP_Eo[c_cond[0]],d.QP_E[c_cond[0],0],deg=1)
    for i in wrong[0]:
        print(d.QP_Eo[i].data*units.Ha,d.QP_E[i,0].data*units.Ha)
        if d.QP_table[0,i]>v:
            d.QP_E[i,0] = fit_c[0]*d.QP_Eo[i]+fit_c[1]
        else:
            d.QP_E[i,0] = fit_v[0]*d.QP_Eo[i]+fit_v[1]
    
    #align to zero wrt to the maximum of valence... fixes the error in Fermi re-evaluation
    #in the BSE RD/ndb.QP. for now.
    d.QP_E[:,0] = d.QP_E[:,0] - np.max(d.QP_E[v_cond[0],0])
    
    if create: d.to_netcdf(output_db)

    return output_db,fit_v,fit_c

@calcfunction
def merge_QP(filenames_List,output_name,ywfl_pk,qp_settings,already_computed_QP_db=orm.List([])): #just to have something that works, but it is not correct to proceed this way
        ywfl = load_node(ywfl_pk.value)
        pw = find_pw_parent(ywfl)
        fermi = pw.outputs.output_parameters.get_dict()['fermi_energy']
        SOC = pw.outputs.output_parameters.get_dict()['spin_orbit_calculation']
        nelectrons = pw.outputs.output_parameters.get_dict()['number_of_electrons']
        kpoints = pw.outputs.output_band.get_kpoints()
        bands = pw.outputs.output_band.get_bands()
        nk = pw.outputs.output_parameters.get_dict()['number_of_k_points']
        qp_rules = qp_settings.get_dict()

        if SOC:
            valence = int(nelectrons)
            conduction = valence + 1
        else:
            valence = int(nelectrons/2) + int(nelectrons%2)
            conduction = valence + 1
        string_run = 'yambopy mergeqp'
        
        # Create temporary directory
        filename='ndb.QP'
        with tempfile.TemporaryDirectory() as dirpath:
            # Open the output file from the AiiDA storage and copy content to the temporary file
            for i in filenames_List.get_list():
                # Create the file with the desired name
                temp_file = pathlib.Path(dirpath) / str(i)
                with load_node(i).outputs.QP_db.base.repository.open(filename, 'rb') as handle:
                    temp_file.write_bytes(handle.read())

                string_run+=' '+str(temp_file)

            if len(already_computed_QP_db.get_list())>0:
                for i in already_computed_QP_db.get_list():
                    # Create the file with the desired name
                    temp_file = pathlib.Path(dirpath) / str(i)
                    temp_file.write_bytes(load_node(i).get_content("rb"))
                    string_run+=' '+str(temp_file)
            
            string_run+=' -o '+dirpath+'/'+output_name.value
            print(string_run)
            os.system(string_run)
            time.sleep(10)
            qp_fixed = sanity_check_QP(valence,conduction,dirpath+'/'+output_name.value,dirpath+'/'+output_name.value.replace('merged','fixed'))
            QP_db = SingleFileData(qp_fixed[0])

            return QP_db
        
        return

@calcfunction
def extend_QP(filenames_List,output_name,ywfl_pk,qp_settings,QP): #just to have something that works, but it is not correct to proceed this way
    ywfl = load_node(ywfl_pk.value)
    pw = find_pw_parent(ywfl)
    fermi = pw.outputs.output_parameters.get_dict()['fermi_energy']
    SOC = pw.outputs.output_parameters.get_dict()['spin_orbit_calculation']
    nelectrons = pw.outputs.output_parameters.get_dict()['number_of_electrons']
    kpoints = pw.outputs.output_band.get_kpoints()
    bands = pw.outputs.output_band.get_bands()
    nk = pw.outputs.output_parameters.get_dict()['number_of_k_points']
    qp_rules = qp_settings.get_dict()

    if SOC:
        valence = int(nelectrons)
        conduction = valence + 2
    else:
        valence = int(nelectrons/2) + int(nelectrons%2)
        conduction = valence + 1
    #output_name = QP._repository._repo_folder.abspath + '/path/ndb.QP_fixed'
    with tempfile.TemporaryDirectory() as dirpath:
        # Open the output file from the AiiDA storage and copy content to the temporary file
        output_name = pathlib.Path(dirpath) / 'ndb.QP_fixed'
        with QP.base.repository.open('ndb.QP_fixed', 'rb') as handle:
            output_name.write_bytes(handle.read())

        qp_fixed,fit_v,fit_c = sanity_check_QP(valence,conduction,output_name,output_name,create=False)
    
        if qp_rules.pop('extend_db', False):
            """
            In the qp settings dict, I should add:
                {
                    'extend_db':True,
                    ''T_smearing': 1e-2, #smearing for the FD corrections...see the paper MBonacci et al. Towards HT... 
                    'consider_only':[v_min,c_max]
                    'Nb': n, #number of bands to be included in the extended QP db (from 1st to nth)
                    -->'v_min':, #used to evaluate v_max energy and v_min that you want to compute explicitly
                    --->'c_max':, #used to evaluate c_min energy and c_max that you want to compute explicilty
                }
            """
            qp_rules['Nb'] = qp_rules.pop('Nb', [1, conduction + valence])
            db_FD_scissored = FD_and_scissored_db(out_db_path=qp_fixed,pw=pw,Nb=qp_rules['Nb'],Nk=nk,v_max=min(qp_rules['consider_only']),c_min=max(qp_rules['consider_only']),fit_v=fit_v,
                    fit_c=fit_c,conduction=conduction,T=qp_rules.pop('T_smearing',1e-2))
            db_FD_scissored.to_netcdf(str(output_name).replace('fixed','extended'))
            
            QP_db_extended = SingleFileData(str(output_name).replace('fixed','extended'))
            return QP_db_extended

def QP_mapper(ywfl,tol=1,full_bands=False,spectrum_tol=1):
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

    b_min_scissored = np.where(abs(bands-mid_gap_energy)<spectrum_tol)[1].min()
    b_max_scissored = np.where(abs(bands-mid_gap_energy)<spectrum_tol)[1].max()
    
    return QP, [b_min_scissored,b_max_scissored]

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

def QP_list_merger(l=[],qp_per_subset=10,consider_only=[-1],already_computed_QP_db:list=None):
    
    subgroup = []
    groups = []
    for qp_set in l:
        for k in list(range(qp_set[0],qp_set[1]+1)):
            for b in list(range(qp_set[2],qp_set[3]+1)):
                
                skip = False
                if already_computed_QP_db is not None:
                    for QP_db in already_computed_QP_db:
                        if check_already_computed(QP_db, b, k):
                            skip=True
                        else:
                            skip=False
                
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
        
        spec.expose_inputs(YamboRestart, namespace='qp', namespace_options={'required': False,'populate_defaults': False, 'help': 'QP inputs only if BSE@GW is asked, i.e. we also provide `QP_subsets_dict`'}, 
                            exclude = ['parent_folder'])
        
        spec.expose_inputs(YamboRestart, namespace='bse', namespace_options={'required': False,'populate_defaults': False, 'help': 'QP inputs only if BSE@GW is asked, i.e. we also provide `QP_subsets_dict`'}, 
                            exclude = ['parent_folder'])
        
        spec.input("additional_parsing", valid_type=orm.List, required = False,
                    help = 'list of additional quantities to be parsed: gap, homo, lumo, or used defined quantities -with names-[k1,k2,b1,b2], [k1,b1], gap_GG, lowest_exciton')
        
        spec.input("QP_subsets_dict", valid_type=orm.Dict, required = False,
                    help = 'subset of QP that you want to compute, useful if you need to obtain a large number of QP corrections')

        spec.input("parent_folder", valid_type=orm.RemoteData, required = False,
                    help = 'scf, nscf or yambo remote folder')
        
        spec.input("clean_failed", valid_type=orm.Bool, default=lambda: orm.Bool(False))

        spec.input("already_computed_QP_db", valid_type=orm.List, required=False,
                   help="list of qp db pks already computed, so we don't need to compute all the QP in QP splitter")
  

##################################### OUTLINE ####################################

        spec.outline(
            #cls.validate_parameters,
            cls.setup,
            while_(cls.can_continue)(
                    cls.perform_next,
            ),
            if_(cls.post_processing_needed)(
                cls.run_post_process,
            ),
            if_(cls.should_run_bse)(
                cls.prepare_and_run_bse,
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
        pw_code: orm.Code,
        preprocessing_code: orm.Code,
        code: orm.Code,
        protocol_qe='moderate',
        protocol='moderate',
        calc_type='gw',
        structure=None,
        overrides={},
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
        if isinstance(code, str):
            
            preprocessing_code = orm.load_code(preprocessing_code)
            code = orm.load_code(code)

        if electronic_type not in [ElectronicType.METAL, ElectronicType.INSULATOR]:
            raise NotImplementedError(f'electronic type `{electronic_type}` is not supported.')

        if spin_type not in [SpinType.NONE, SpinType.COLLINEAR]:
            raise NotImplementedError(f'spin type `{spin_type}` is not supported.')
        
        builder = cls.get_builder()

        overrides_scf = overrides.pop('scf',{})
        overrides_nscf = overrides.pop('nscf',{})
        overrides_yres = overrides.pop('yres',{})

        for override in [overrides_scf,overrides_nscf]:
            override['clean_workdir'] = override.pop('clean_workdir',False) #required to have a valid parent folder

        if pseudo_family:
            overrides_scf['pseudo_family'] = pseudo_family
            overrides_nscf['pseudo_family'] = pseudo_family
                    
        ######### SCF and NSCF INPUTS/PROTOCOLS #########
        
        # try to get the pw builders from the scf and nscf parents if any
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
        
        if calc_type=='bse':
            protocol_ = 'bse_'+protocol
        else:
            protocol_ = protocol
            
        yres_builder = YamboRestart.get_builder_from_protocol(
                preprocessing_code=preprocessing_code,
                code=code,
                protocol=protocol_,
                parent_folder=parent_folder,
                overrides=overrides_yres,
                NLCC=NLCC,
                RIM_v=RIM_v,
                RIM_W=RIM_W,
                nelectrons=nelectrons,
                ecutwfc=ecutwfc
            )

        builder.yres = yres_builder

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
        if protocol == 'molecule' or structure.pbc.count(True)==0: molecule=True
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
        
        self.ctx.all_calcs_to_do = ['scf', 'nscf', 'yambo', 'QP splitter', 'bse']
    
        input_manager = YamboInputManager.get_nested_inputs(
            self.inputs, what='YamboWorkflow', mode='builder'
        )
        self.ctx.input_manager = YamboInputManager(**input_manager)
        
        self.ctx.should_run_QP = False
        self.ctx.should_run_bse = False
                       
        # we switch the yres <---> qp inputs so first we do QP, and then yres (which is BSE in this specific case).
        if len(self.ctx.input_manager.qp.parameters) > 0 and len(self.ctx.input_manager.QP_subsets_dict) > 0 and not self.ctx.input_manager.yres.settings.get('INITIALISE', False):
            self.ctx.should_run_bse = True if self.ctx.input_manager.qp.parameters.get('variables', None) else False
            self.ctx.should_run_QP = self.ctx.should_run_bse
        elif len(self.ctx.input_manager.qp.parameters) == 0 and len(self.ctx.input_manager.QP_subsets_dict) > 0 and not self.ctx.input_manager.yres.settings.get('INITIALISE', False): 
            self.ctx.should_run_QP = True
        
        if not hasattr(self.ctx.input_manager, 'parent_folder'):
            self.report('No parent folder, we start from scratch.')
            self.ctx.calc_to_do = 'scf'
        elif self.ctx.input_manager.parent_folder is None:
            self.report('No parent folder, we start from scratch.')
            self.ctx.calc_to_do = 'scf'
        else:
            parent = take_calc_from_remote(self.ctx.input_manager.parent_folder,level=-1)
            # we get the PwCalculation
            if parent.process_type=='aiida.workflows:quantumespresso.pw.base':
                parent = parent.outputs.remote_folder.creator
            
            # empty parent, we need to recompute it.
            if parent.outputs.remote_folder.is_empty:
                self.report('The parent remote folder is empty, we start from scratch.')
                self.ctx.calc_to_do = 'scf'

            # pw parent
            if 'quantumespresso.pw' in parent.process_type:
                parent_params = parent.inputs.parameters.get_dict()
                calc_type = parent_params['CONTROL']['calculation']
                if calc_type in ['scf','relax','vc-relax']:
                    self.ctx.calc_to_do = 'nscf'
                elif calc_type in ['nscf']:
                    self.ctx.calc_to_do = 'yambo'

            # yambo parent
            elif parent.process_type=='aiida.calculations:yambo.yambo':
                self.ctx.calc_to_do = 'yambo'
                if len(self.ctx.input_manager.QP_subsets_dict)>0: self.ctx.calc_to_do = 'QP splitter'
            else:
                self.ctx.calc_to_do = 'scf'
                self.report('no valid input calculations, so we will start from scratch')
            
            self.ctx.calc = parent
            
        # check nbnd is fine:
        if not self.ctx.input_manager.has_enough_nbnd:
            self.report(f'Setting nbnd={self.ctx.input_manager.Nb} in the NSCF step, not enough bands: {self.ctx.input_manager.nscf_nbnd}<{self.ctx.input_manager.Nb}')
            self.ctx.input_manager.set_nscf_nbnd(self.ctx.input_manager.Nb)
            # we redo nscf if not enough bands, we cannot proceed with yambo or QP_splitter:
            if self.ctx.calc_to_do in ['yambo', 'QP_splitter']:
                self.report(f"Recomputing NSCF step with nbnd={self.ctx.input_manager.nscf_nbnd}")
                self.ctx.calc_to_do = 'nscf' 
            
        self.ctx.splitted_QP = []
        self.ctx.qp_splitter = 0
        self.report("Workflow initilization step completed.")

    def can_continue(self):

        """This function checks the status of the last calculation and determines what happens next, including a successful exit.
        
        """
        if self.ctx.calc_to_do != 'The workflow is finished':
            self.report('The workflow continues with a {} calculation'.format(self.ctx.calc_to_do))
            return True
        else:
            self.report('The workflow is finished')
            return False

    def perform_next(self):
        """This function  will submit the next step, depending on the information provided in the context.

        The next step will be a yambo calculation if the provided inputs are a previous yambo/p2y run
        Will be a PW scf/nscf if the inputs do not provide the NSCF or previous yambo parent calculations
        
        """
        # check if the previous run failed.
        if hasattr(self.ctx, 'calc'):
            calc = self.ctx.calc
            if not calc.is_finished_ok:
                # we allow some QP calc to fail.
                if self.ctx.calc_to_do=='QP_splitter': pass
                self.report("last calculation failed, exiting the workflow")
                return self.exit_codes.ERROR_WORKCHAIN_FAILED

        self.report('performing a {} calculation'.format(self.ctx.calc_to_do))

        if self.ctx.calc_to_do == 'scf':
            self.ctx.input_manager.scf.metadata.call_link_label = 'scf'
            scf_inputs = self.ctx.input_manager.generate_AiiDA_inputs(what='scf')
            future = self.submit(PwBaseWorkChain, **scf_inputs)
            self.ctx.calc_to_do = 'nscf'

        elif self.ctx.calc_to_do == 'nscf':
            self.ctx.input_manager.nscf.metadata.call_link_label = 'nscf'
            self.ctx.input_manager.set_parent_folder(self.ctx.calc.outputs.remote_folder)
            nscf_inputs = self.ctx.input_manager.generate_AiiDA_inputs(what='nscf')
            future = self.submit(PwBaseWorkChain, **nscf_inputs)
            self.ctx.calc_to_do = 'yambo'

        elif self.ctx.calc_to_do == 'yambo':
            if len(self.ctx.input_manager.additional_parsing) > 0:
                self.report('updating yambo parameters to parse more results')
                mapping, yambo_parameters = add_corrections(
                    self.ctx.input_manager.parameters, 
                    self.ctx.calc.outputs.remote_folder,
                    self.ctx.input_manager.additional_parsing,
                )
                self.ctx.mapping = mapping
                self.ctx.input_manager.parameters = yambo_parameters
            if self.ctx.should_run_QP and self.ctx.should_run_bse:
                call_link_label = 'chi_for_qp'
                what='qp'
                self.ctx.input_manager.qp.metadata.call_link_label = call_link_label
            else:
                call_link_label = 'yambo'
                what='YamboRestart'
                self.ctx.input_manager.yres.metadata.call_link_label = call_link_label
            self.ctx.input_manager.set_parent_folder(self.ctx.calc.outputs.remote_folder)
            yambo_inputs = self.ctx.input_manager.generate_AiiDA_inputs(what=what)
            future = self.submit(YamboRestart, **yambo_inputs)

            if len(self.ctx.input_manager.QP_subsets_dict)>0:
                # if we pass the instructions for the QP, it means we want to compute them.
                self.ctx.calc_to_do = 'QP splitter'
            else:
                self.ctx.calc_to_do = 'The workflow is finished'
        
        elif self.ctx.calc_to_do == 'QP splitter': # This is the multiple QP runs which will then be merged with yambopy
            
            QP = {}
            
            for k,v in QP.items():
                if not QP[k].is_finished_ok:
                    self.report('some calculation failed')
                    return self.exit_codes.ERROR_WORKCHAIN_FAILED

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
                    self.ctx.input_manager.yres.settings['COPY_DBS'] = True

                # setting some specific configuration:
                if 'parallelism' in self.ctx.input_manager.QP_subsets_dict:
                    for k,v in self.ctx.input_manager.QP_subsets_dict['parallelism'].items():
                        self.ctx.input_manager.set_variable(name=k, value=v)
                if 'resources' in self.ctx.input_manager.QP_subsets_dict:
                    self.ctx.input_manager.metadata.options.resources = self.ctx.input_manager.QP_subsets_dict['resources']
                if 'prepend_text' in self.ctx.input_manager.QP_subsets_dict:
                    self.ctx.input_manager.metadata.options.prepend_text = self.ctx.input_manager.QP_subsets_dict['prepend_text']
                

                self.ctx.input_manager.clean_workdir = orm.Bool(True)
                mapping = gap_mapping_from_nscf(find_pw_parent(take_calc_from_remote(self.ctx.calc.outputs.remote_folder,level=-1)).pk)
                self.ctx.mapping = mapping

                consider_only = self.ctx.input_manager.QP_subsets_dict.get('consider_only',[-1]) #[1,64], a range.
                self.ctx.input_manager.QP_subsets_dict['consider_only'] = consider_only

                if 'range_QP' in self.ctx.input_manager.QP_subsets_dict: #the name can be changed..
                    Energy_region = max(self.ctx.input_manager.QP_subsets_dict['range_QP'], mapping['nscf_gap_eV']*1.2)
                    self.report('range of energy for QP: {} eV'.format(Energy_region))
                    self.ctx.input_manager.QP_subsets_dict['explicit'], self.ctx.input_manager.QP_subsets_dict['scissored'] = QP_mapper(
                                                                    self.ctx.calc,
                                                                    tol = Energy_region,
                                                                    full_bands=self.ctx.input_manager.QP_subsets_dict.get('full_bands',False),
                                                                    spectrum_tol=self.ctx.input_manager.QP_subsets_dict.get('range_spectrum',Energy_region),
                                                                )
                if 'boundaries' in self.ctx.input_manager.QP_subsets_dict:
                    k_i=self.ctx.input_manager.QP_subsets_dict['boundaries'].pop('ki',1)
                    k_f=self.ctx.input_manager.QP_subsets_dict['boundaries'].pop('kf',mapping['number_of_kpoints'])
                    b_i=self.ctx.input_manager.QP_subsets_dict['boundaries']['bi']
                    b_f=self.ctx.input_manager.QP_subsets_dict['boundaries']['bf']

                    if hasattr(self.ctx.input_manager,'already_computed_QP_db'):
                        already_computed = self.ctx.input_manager.already_computed_QP_db
                    else:
                        already_computed = None

                    self.ctx.input_manager.QP_subsets_dict['subsets'] = QP_list_merger([[k_i,k_f,b_i,b_f]],
                                                                      self.ctx.input_manager.QP_subsets_dict['qp_per_subset'],
                                                                      consider_only=consider_only,
                                                                      already_computed_QP_db = already_computed
                                                                      )

                if not 'subsets' in self.ctx.input_manager.QP_subsets_dict.keys():
                    if 'explicit' in self.ctx.input_manager.QP_subsets_dict.keys():
                        self.ctx.input_manager.QP_subsets_dict['subsets'] = QP_list_merger(self.ctx.input_manager.QP_subsets_dict['explicit'],
                                                                        self.ctx.input_manager.QP_subsets_dict['qp_per_subset'],
                                                                        consider_only=consider_only)

                #self.report('subsets: {}'.format(self.ctx.input_manager.QP_subsets_dict['subsets']))

            for i in range(1,1+self.ctx.input_manager.QP_subsets_dict['parallel_runs']):
                if len(self.ctx.input_manager.QP_subsets_dict['subsets']) > 0:
                    self.ctx.input_manager.parameters.set_variable(name='QPkrange', value=[[self.ctx.input_manager.QP_subsets_dict['subsets'].pop(),'']])
                    self.ctx.input_manager.metadata.call_link_label = 'yambo_QP_splitted_{}'.format(i+self.ctx.qp_splitter)
                    self.ctx.input_manager.set_parent_folder = self.ctx.calc.outputs.remote_folder
                    if self.ctx.should_run_QP and self.ctx.should_run_bse:
                        # if run bse@QP, it means that we need to run the qp using the qp inputs.
                        what='qp'
                    else:
                        what='YamboRestart'
                    yambo_inputs = self.ctx.input_manager.generate_AiiDA_inputs(what=what)
                    future = self.submit(YamboRestart, **yambo_inputs)
                    self.report('Launching YamboRestart <{}> for QP, iteration#{} of {}'.format(future.pk,i+self.ctx.qp_splitter, len(self.ctx.input_manager.QP_subsets_dict['subsets'])))
                    self.ctx.splitted_QP.append(future.uuid)
                    QP[str(i+1)] = future
                else:
                    self.ctx.calc_to_do = 'The workflow is finished'
            
            self.ctx.qp_splitter += self.ctx.input_manager.QP_subsets_dict['parallel_runs']

            if len(self.ctx.input_manager.QP_subsets_dict['subsets']) == 0: self.ctx.calc_to_do = 'The workflow is finished'

            return ToContext(QP) #wait for all splitted calculations....

        return ToContext(calc = future)
    
    def post_processing_needed(self):
        #in case of multiple QP calculations, yes
        if len(self.ctx.splitted_QP) > 0:
            self.report('We need to merge the computed QP')
            return True
        self.report('No post processing needed')
        return False

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

        self.ctx.QP_db = merge_QP(
            splitted,
            output_name,
            orm.Int(self.ctx.calc.pk),
            qp_settings=orm.Dict(dict=self.ctx.input_manager.QP_subsets_dict),
            already_computed_QP_db = self.inputs.get('already_computed_QP_db',orm.List([]))
            )
        self.out('merged_QP',self.ctx.QP_db)
        
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

        return

    def should_run_bse(self):
        #in case of BSE on top of GW just done, yes
        return self.ctx.should_run_bse

    def prepare_and_run_bse(self):
        
        self.ctx.calc_to_do='bse'
        
        # here we don't care of using the YamboInputManager, we do not need it for now.
        self.ctx.yambo_inputs = self.exposed_inputs(YamboRestart, 'yres') 
        bse_params = self.ctx.yambo_inputs.yambo.parameters.get_dict()

        if isinstance(self.ctx.QP_db,tuple): self.ctx.QP_db = self.ctx.QP_db[1]
            
        self.ctx.yambo_inputs.yambo.QP_corrections = self.ctx.QP_db
        bse_params['variables']['KfnQPdb'] = "E < ./ndb.QP"

        self.ctx.yambo_inputs.parent_folder = self.ctx.calc.outputs.remote_folder

        if not 'BSEBands' in bse_params['variables'].keys():
            if 'scissored' in self.ctx.input_manager.QP_subsets_dict.keys():
                bse_params['variables']['BSEBands'] = [[self.ctx.input_manager.QP_subsets_dict['scissored'][0],self.ctx.input_manager.QP_subsets_dict['scissored'][1]],'']
            else:
                bse_params['variables']['BSEBands'] = [[self.ctx.BSE_map['v_min'],self.ctx.BSE_map['c_max']],'']
        if not 'BSEQptR' in bse_params['variables'].keys():
            bse_params['variables']['BSEQptR'] = [[self.ctx.BSE_map['q_ind'],self.ctx.BSE_map['q_ind']],'']

        self.ctx.yambo_inputs.yambo.parameters = orm.Dict(dict=bse_params)

        self.ctx.yambo_inputs.metadata.call_link_label = 'BSE'
        future = self.submit(YamboRestart, **self.ctx.yambo_inputs)

        return ToContext(bse = future) 

    def report_wf(self):

        self.report('Final step.')

        calc = self.ctx.calc
        if calc.is_finished_ok:

            if hasattr(self.inputs, 'additional_parsing'):
                #self.report('parsing additional quantities')
                mapping, yambo_parameters = add_corrections(
                    self.ctx.input_manager.parameters, 
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
                mapping, yambo_parameters = add_corrections(
                    self.ctx.input_manager.parameters, 
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
                    self.report(message)
        else:
            self.report("workflow NOT completed successfully")
            return self.exit_codes.ERROR_WORKCHAIN_FAILED