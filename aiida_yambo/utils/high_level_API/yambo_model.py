###### methods for easy interaction with AiiDA and yambopy data structures
import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import Literal, Union, Optional, Any
from aiida_yambo.workflows.utils.parameters import ConvParam, CoupledConvParam
from aiida_yambo.utils.high_level_API.general_model import GeneralInputManager
from aiida_yambo.utils.high_level_API.qe_model import PwBaseWorkChainInputManager

class YamboRestartInputManager(GeneralInputManager):
    
    settings:dict={}
    
    ############ START setter methods ############
    
    @property
    def Nb(self,):
        return self.get_Nb()
    
    @property
    def Gcut(self,):
        return self.get_Gcut()
    
    @property
    def BSEGcut(self,):
        return self.get_BSEGcut()
    
    def remove_arguments(self, remove_tags:list[str]=[]):
        for tag in remove_tags:
            self.parameters['arguments'].remove(tag, None)
            
    def set_arguments(self, add_tags:list[str]=[]):
        for tag in add_tags:
            if tag not in self.parameters['arguments']:
                self.parameters['arguments'] += [tag]
    
    def remove_variables(self, remove_tags:list[str]=[]):
        for tag in remove_tags:
            self.parameters['variables'].pop(tag, None)
    
    def set_variable(self, name:str='', value:Union[list, int, float]=None):
        """General method to set or add a parameter in the self.parameters['variables'] dictionary.
        
        """
        if value is not None:
            self.parameters['variables'][name] = value
            
    def set_Nb(self, 
               setter_tags:list[Literal['BndsRnXs','BndsRnXp','BndsRnXm','GbndRnge',]]=['BndsRnXp','GbndRnge',], 
               value:Union[list, int, float]=None, add_if_not_present:bool=True):
        for variant in setter_tags:
            if variant in self.parameters['variables'] or add_if_not_present:
                if not value:
                    raise ValueError('value parameter cannot be None.')
                if isinstance(value, int) or isinstance(value, float):
                    value=[[1,value],'']
                elif isinstance(value,list):
                    if len(value) == 2 and value[-1]!='':
                        value=[value,'']
                self.set_variable(variant, value)
                
    def set_Gcut(self, 
                 setter_tags:list[Literal['NGsBlkXs','NGsBlkXp','BSENGBlk']]=['NGsBlkXp'], 
                 value:Union[list, int, float]=None, add_if_not_present:bool=True, units:Literal['Ry', 'RL', 'mRy']='Ry'):
        for variant in setter_tags:
            if variant in self.parameters['variables'] or add_if_not_present:
                if not value:
                    raise ValueError('value parameter cannot be None.')
                if isinstance(value, int) or isinstance(value, float):
                    value=[value, units]
                self.set_variable(variant, value)
    
    def set_BSEGcut(self,
                    setter_tags:list[Literal['NGsBlkXs','NGsBlkXp','BSENGBlk']]=['NGsBlkXs','BSENGBlk']):
        self.set_Gcut(setter_tags=setter_tags)
                
    def set_BSEbands(self,value:list[int]):
        #value = [v,c]
        if len(value) == 2 and value[-1]!='': value=[value,'']
        return self.set_variable('BSEbands', [value,''])
                    
    ############ END setter methods ############
    
    ############ START getter methods ############
    
    def get_variable_value(self, name:str=''):
        """General method to get a parameter from the self.parameters['variables'] dictionary.
        
        """
        
        return self.parameters.get('variables',{}).get(name, None)
    
    def get_Nb(self, search_tags:list[str]=['BndsRnXs','BndsRnXp','GbndRnge']):
        """Get the number of bands used for the calculation
        
        If different values are used e.g. for X and Sc, we take the maximum by default.
        
        We get the last number from the yambopy format [[1,Nb],''], as in the end Nb is what we need 
        to understand the nbnd value for QE runs.
        """
        Nb = 0
        for variant in search_tags:
            if variant not in self.parameters.get('variables',{}): continue
            Nb_variant = self.get_variable_value(name=variant)[0][-1]
            Nb = Nb if Nb_variant < Nb else Nb_variant
        return Nb
    
    def get_Gcut(self, search_tags:list[str]=['NGsBlkXs','NGsBlkXp','BSENGBlk']):
        Gcut = 0
        for variant in search_tags:
            if variant not in self.parameters.get('variables',{}): continue
            Gcut = self.get_variable_value(name=variant)
            if isinstance(Gcut, list):
                Gcut = Gcut[0]
        return Gcut
    
    def get_BSEGcut(self, search_tags:list[str]=['BSENGBlk']):
        return self.get_Gcut(search_tags=search_tags)
    
    def get_BSEbands(self,):
        return self.parameters.get('variables',{}).get('BSEbands', None)
    
    ############ END getter methods ############

class YamboInputManager(BaseModel): # TODO: maybe you can do the same for PwCalculations, so it's very easy to deal with them? maybe just utilities... mmm
    """Class to easily manage the interactions with AiiDA data belogining to inputs of Calculations and Workflows.
    
    This should be a layer between the inputs nodes, the builders, to reduce the direct interaction with AiiDA data.
    It should be for YamboWorkflow nodes/builders. It contains all the inputs we need to run, also the QE ones.
    
    We put fields to represent the WorkChains which directly manages Calcjobs, it's the only necessary bit for now. 
    It is needed to avoid overwiting of `parameters` between different kind of CalcJobs.
    all the other yambo workchain inputs are directly exposed to this model.
    
    so, we expect to find yres.parameters, scf.parameters... then only when getting AiiDA inputs, we will put them under the correct place.
    
    NOTE: we always have all the fields below... but if they are empty or None, they should be counted as not being provided as inputs.
          This is the logic adopted in the workflows.
    """    
    scf: PwBaseWorkChainInputManager = PwBaseWorkChainInputManager()
    nscf: PwBaseWorkChainInputManager = PwBaseWorkChainInputManager()
    qp: YamboRestartInputManager = YamboRestartInputManager() # only used for QP to be used in a following BSE@QP.
    bse: YamboRestartInputManager = YamboRestartInputManager() # the standard yambo inputs holder.
    
    parent_folder: Optional[Any] = None
    
    QP_subsets_dict: dict = {} # required keys: parallel_runs, qp_per_subset. Then: range_QP or explicit or boundaries

    additional_parsing: list = []
    
    model_config = ConfigDict(
        extra='allow', # e.g. parent_folder.
    )
    
    #### PUT ALSO SETTERS? YES
    @property
    def Nb(self,):
        return max(self.qp.get_Nb(),self.bse.get_Nb())
    
    @property
    def Nb_qp(self,):
        return self.qp.get_Nb()
    
    @property
    def Nb_bse(self,):
        return self.bse.get_Nb()
    
    @property
    def Gcut(self,):
        return max(self.qp.get_Gcut(),self.bse.get_Gcut())
    
    @property
    def Gcut_qp(self,):
        return self.qp.get_Gcut()
    
    @property
    def Gcut_bse(self,):
        return self.bse.get_BSEGcut()
    
    @property
    def nscf_nbnd(self,):
        return self.nscf.get_nbnd()
    
    @property
    def ecutwfc(self,):
        return self.nscf.get_ecutwfc()
    
    @property
    def has_enough_nbnd(self,):
        return bool(self.nscf_nbnd >= self.Nb)
    
    def set_parent_folder(self,node, update_pw_parameters=False):
        self.parent_folder = node
        if self.parent_folder is not None and update_pw_parameters:
            from aiida_yambo.utils.common_helpers import get_pw_inputs_builders
            builder_scf, builder_nscf = get_pw_inputs_builders
                
            if builder_scf:
                dict_scf = YamboRestartInputManager.convert_from_AiiDA_data(builder_scf)
                dict_scf.update(YamboRestartInputManager.convert_from_AiiDA_data(builder_scf.pw, get_metadata=True))
                self.scf = PwBaseWorkChainInputManager(**dict_scf)
            
            if builder_nscf:
                dict_nscf = YamboRestartInputManager.convert_from_AiiDA_data(builder_nscf)
                dict_nscf.update(YamboRestartInputManager.convert_from_AiiDA_data(builder_nscf.pw, get_metadata=True))
                self.nscf = PwBaseWorkChainInputManager(**dict_nscf)
                
    
    def remove_parent_folder(self,):
        self.parent_folder = None
        
    def set_QP_subsets_dict(self, bi, bf, qp_per_subset=1, parallel_runs=1):
        # this sets to compute given range of bands for all kpoints.
        self.QP_subsets_dict = {
            'boundaries':{
                'bi':bi,'bf':bf,
            },
            'parallel_runs':parallel_runs,
            'qp_per_subset':qp_per_subset,
        }
    
    ############ START Redundant SETTER methods for yres easy access ############
    def remove_arguments(self, remove_tags:list[str]=[]):
        self.qp.remove_arguments(remove_tags)
        self.bse.remove_arguments(remove_tags)
            
    def set_arguments(self, add_tags:list[str]=[]):
        self.qp.set_arguments(add_tags)
        self.bse.set_arguments(add_tags)
    
    def remove_variables(self, remove_tags:list[str]=[]):
        self.qp.remove_variables(remove_tags)
        self.bse.remove_variables(remove_tags)
    
    def set_variable(self, name:str='', value:Union[list, int, float]=None):
        self.qp.set_variable(name, value)
        self.bse.set_variable(name, value)
            
    def set_Nb(self, setter_tags:list[str]=['BndsRnXs','BndsRnXp','GbndRnge',], value:Union[list, int, float]=None, add_if_not_present:bool=False):
        self.qp.set_Nb(setter_tags, value, add_if_not_present)
        self.bse.set_Nb(setter_tags, value, add_if_not_present)
                
    def set_Gcut(self, setter_tags:list[str]=['NGsBlkXs','NGsBlkXp'], value:Union[list, int, float]=None, add_if_not_present:bool=False,units:Literal['Ry', 'RL', 'mRy']='Ry'):
        self.qp.set_Gcut(setter_tags, value, add_if_not_present,units)
        self.bse.set_BSEGcut(setter_tags, value, add_if_not_present,units)
    ############ END Redundant SETTER methods for yres easy access ############

    ############ START QE setter methods ############
    
    # redundant methods with respect to QuantumEspressoInputManager, so that it's easier to use them within Yambo Workflows
    # mainly, methods which should update both scf and nscf at the same time (SOC, PSEUDO, GAMMA ONLY...)        
        
    def set_nscf_nbnd(self,value:int):
        self.nscf.set_nbnd(value)
        
    def set_SOC(self,value=True):
        self.scf.set_SOC()
        self.nscf.set_SOC()
        
    def set_pseudo_family(self, value:str):
        self.scf.set_pseudo_family(value)
        self.nscf.set_pseudo_family(value)
        
    def set_gamma_only(self,calculation:Literal['scf', 'nscf']='nscf', both=True):
        pw_dict = {'scf':self.scf, 'nscf':self.nscf}
        if both:
            for calc, object in pw_dict.items():
                object.set_gamma_only()
        else:
            pw_dict[calculation].set_gamma_only()
            
        self.scf=pw_dict['scf']
        self.nscf=pw_dict['nscf']
        
    ############ END QE setter methods ############            
        
        
    ############ START Redundant GETTER methods for yres easy access ############
    
    def get_variable_value(self, name:str=''):
        return self.qp.get_variable_value(name)
    
    def get_Nb(self, search_tags:list[str]=['BndsRnXs','BndsRnXp','GbndRnge']):
        return self.qp.get_Nb(search_tags)
    
    def get_Gcut(self, search_tags:list[str]=['NGsBlkXs','NGsBlkXp']):
        return self.qp.get_Gcut(search_tags)
    
    def get_BSEGcut(self, search_tags:list[str]=['BSENGBlk']):
        return self.bse.get_Gcut(search_tags=search_tags)

    def get_BSEbands(self,):
        return self.bse.get_BSEbands()

    ############ END Redundant GETTER methods for yres easy access ############

    ######### START AiiDA getter STUFF #########
        
    def generate_AiiDA_inputs(self,
        what:Literal['YamboCalculation', 'YamboRestart','YamboWorkflow','YamboConvergence','scf','nscf','qp']='YamboWorkflow',
    ):
        """This will return a dictionary with several stuff.
        
        For now, we mainly need `parameters` and `metadata`, which are the ones we need to update more often, 
        so that we can just do modification and then, case by case
        
        builder.yambo.parameters = p['parameters']
        builder.yambo.metadata = p['metadata']
        builder.yambo.settings = p['settings']
        
        TODO: generalize this and move it to the base class.
        """
        p = YamboRestartInputManager.to_AiiDA_inputs(
            self.model_dump(exclude_none=True, exclude={'scf', 'nscf', 'qp', 'bse'})
        )
        for key, manager in (('qp', self.qp), ('bse', self.bse),
                             ('scf', self.scf), ('nscf', self.nscf)):
            sub_inputs = YamboRestartInputManager.to_AiiDA_inputs(manager.model_dump(exclude_none=True))
            if sub_inputs and GeneralInputManager._has_provided_inputs(manager):
                p[key] = sub_inputs
        
        
        from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
        from aiida_yambo.workflows.yamborestart import YamboRestart
        from aiida_yambo.workflows.yambowf import YamboWorkflow
        from aiida_yambo.workflows.yamboconvergence import YamboConvergence
        
        class_ob = { #TOTEST
            'YamboCalculation':YamboWorkflow,
            'scf': YamboWorkflow,
            'nscf': YamboWorkflow,
            'qp': YamboWorkflow,
            'YamboWorkflow': YamboWorkflow,
            'YamboConvergence': YamboConvergence,
            'bse': YamboWorkflow,
        }
        results = YamboRestartInputManager.set_ports(processed_inputs=p, class_instance_ports=class_ob[what].spec().inputs.ports)
        
        # here we effectively return the correct set of inputs. 
        # we notice that we need to manually set the parent_folder in all cases except YamboWorkflow and YamboConvergence, as the others does not use the specific 
        # corresponding Workchain.
        # not clear why the parent_folder doesn't go automatically in the right place, but ok.
        if what in ['scf','nscf']:
            results = results[what]
            if what == 'nscf': results['pw']['parent_folder'] = p.get('parent_folder',None)
        elif what in ['qp']:
            results = results['qp']
            results['parent_folder'] = p.get('parent_folder',None)
        elif what in ['bse']:
            results = results['bse']
            results['parent_folder'] = p.get('parent_folder',None)
        elif what in ['YamboCalculation']:
            results = results['qp']['yambo']
            results['parent_folder'] = p.get('parent_folder',None)
            
        return results
    
    def populate_builder(self, builder, what:Literal['YamboCalculation', 'YamboRestart','YamboWorkflow','YamboConvergence']='YamboWorkflow', skip:list[str]=[]):
        from aiida.common import AttributeDict
        p = self.generate_AiiDA_inputs(what=what)
        pp = AttributeDict(p)
        for k,v in pp.items():
            if k in skip:
                continue
            setattr(builder,k,v)
        return builder
    
    ######### END AiiDA getter STUFF #########
    
    @staticmethod
    def get_nested_inputs(
        node_or_builder, 
        what:Literal['YamboCalculation', 'YamboRestart','YamboWorkflow','YamboConvergence']='YamboWorkflow',
        mode:Literal['node','builder']='node'
    ):
        """Fetches the nested inputs to get parameters and metadata.
        
        Nesting is: YamboConvergence().inputs.ywfl.yres.yambo
        
        A bit hardcoded but it's fine.
        """
        # we always grab the highest level inputs (not hidden under exposed inputs, but directly declared in the wfl)
        # we do this so we can expose directly the yambo inputs, and basically dump to every kind of yambo calcjob/workchain inputs
        # the price to pay is this hardcoding, which is not so dramatic, and that `scf` and `nscf` are explicitly prompted like you see below.
        # for YamboWannier90WorkChain, something similar should happen.
        
        if mode=='node':
            # in the lines below, we implement only the logic for the builder.
            inputs = node_or_builder.get_builder_restart()
        else:
            inputs = node_or_builder
        
        # this extract the highest level inputs
        results = YamboRestartInputManager.convert_from_AiiDA_data(inputs, get_metadata=what=='YamboCalculation')
        
        match what:
            
            case "YamboRestart":
                results.update(YamboRestartInputManager.convert_from_AiiDA_data(inputs.yambo, get_metadata=True))

            case "YamboWorkflow": 
                if 'qp' in inputs:
                    results['qp'] = YamboRestartInputManager.convert_from_AiiDA_data(inputs.qp)
                    results['qp'].update(YamboRestartInputManager.convert_from_AiiDA_data(inputs.qp.yambo, get_metadata=True))
                if 'scf' in inputs:
                    results['scf'] = YamboRestartInputManager.convert_from_AiiDA_data(inputs.scf)
                    results['scf'].update(YamboRestartInputManager.convert_from_AiiDA_data(inputs.scf.pw, get_metadata=True))
                if 'nscf' in inputs:
                    results['nscf'] = YamboRestartInputManager.convert_from_AiiDA_data(inputs.nscf)
                    results['nscf'].update(YamboRestartInputManager.convert_from_AiiDA_data(inputs.nscf.pw, get_metadata=True))
                if 'bse' in inputs:
                    results['bse'] = YamboRestartInputManager.convert_from_AiiDA_data(inputs.bse)
                    results['bse'].update(YamboRestartInputManager.convert_from_AiiDA_data(inputs.bse.yambo, get_metadata=True))

            case "YamboConvergence":
                if 'qp' in inputs.ywfl:
                    results['qp'] = YamboRestartInputManager.convert_from_AiiDA_data(inputs.ywfl.qp)
                    results['qp'].update(YamboRestartInputManager.convert_from_AiiDA_data(inputs.ywfl.qp.yambo, get_metadata=True))
                if 'scf' in inputs.ywfl:
                    results['scf'] = YamboRestartInputManager.convert_from_AiiDA_data(inputs.ywfl.scf)
                    results['scf'].update(YamboRestartInputManager.convert_from_AiiDA_data(inputs.ywfl.scf.pw, get_metadata=True))
                if 'nscf' in inputs.ywfl:
                    results['nscf'] = YamboRestartInputManager.convert_from_AiiDA_data(inputs.ywfl.nscf)
                    results['nscf'].update(YamboRestartInputManager.convert_from_AiiDA_data(inputs.ywfl.nscf.pw, get_metadata=True))
                if 'bse' in inputs.ywfl:
                    results['bse'] = YamboRestartInputManager.convert_from_AiiDA_data(inputs.ywfl.bse)
                    results['bse'].update(YamboRestartInputManager.convert_from_AiiDA_data(inputs.ywfl.bse.yambo, get_metadata=True))
        return results