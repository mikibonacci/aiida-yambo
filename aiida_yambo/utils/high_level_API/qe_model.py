###### methods for easy interaction with AiiDA and yambopy data structures

from typing import Literal, Union
from aiida_yambo.utils.high_level_API.general_model import GeneralInputManager


class PwBaseWorkChainInputManager(GeneralInputManager):
    
    """For now, it will work with PwBaseWorkChain inputs.
    
    The idea is to move all the relevant inputs under self, not self.pw (.metadata, .parameters)... 
    And move them when it's needed (i.e. after modification and before AiiDA submission).
    
    """
    
    settings: dict = {}
    parallelization: dict = {}
    
    def set_pw_parameter(self, namelist:Literal['CONTROL', 'SYSTEM', 'ELECTRONS'], key:str, value):
        self.parameters[namelist][key] = value
                
    def set_nbnd(self,value:int):
        self.set_pw_parameter(namelist='SYSTEM', key='nbnd', value=value)
        
    def set_SOC(self,value=True):
        for p in ['noncolin', 'lspinorb']:
            self.set_pw_parameter(namelist='SYSTEM', key=p, value=value)
        
    def set_pseudo_family(self, value:str):
        from aiida import orm
        family = orm.load_group(value)
        self.pseudos = family.get_pseudos(structure=self.structure)
    
    def set_kmesh(self, value:list=[1,1,1], shift:list=[0,0,0]):
        from aiida import orm
        self.kpoints = orm.KpointsData()
        self.kpoints.set_cell_from_structure(self.structure)
        self.kpoints.set_kpoints_mesh(value, shift) 
    
    def set_k_inverse_distance(self, value:float=1.0, force_parity=True):
        from aiida import orm
        self.kpoints = orm.KpointsData()
        self.kpoints.set_cell_from_structure(self.structure)
        self.kpoints.set_kpoints_mesh_from_density(1.0 / value, force_parity=force_parity)
        
    def set_gamma_only(self):
        self.set_kmesh() # the default is Gamma.
        self.settings['GAMMA_ONLY'] = True
        
    def set_npools(self, value:int=1):
        self.parallelization['npool'] = value
        
    def set_nbands(self, value:int=1):
        self.parallelization['nbands'] = value
            
    ############ END QE getter methods ############
       
    ############ START QE getter methods ############
    
    def get_nbnd(self,):
        return self.parameters.get('SYSTEM',{}).get('nbnd',-1)
    
    def get_ecutwfc(self,):
        return self.parameters.get('SYSTEM',{}).get('ecutwfc',-1)
        
    def get_mesh(self,) -> list[list]:
        return self.kpoints.get_kpoints_mesh()
    
    def get_kpoints_distance(self,) -> float:
        return self.kpoints_distance.value
    
    ############ END QE getter methods ############
    
    ############ START AiiDA stuff ############
    
    def generate_AiiDA_inputs(self):
        raise NotImplementedError
    
    ############ END AiiDA stuff ############