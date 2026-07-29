###### methods for easy interaction with AiiDA and yambopy data structures

from pydantic import BaseModel, ConfigDict
from typing import Literal, Union

class ResourcesModel(BaseModel):
    
    num_machines: int = 1
    num_mpiprocs_per_machine: int = 1
    num_cores_per_mpiproc: int = 1
    
    model_config = ConfigDict(
        extra='allow',
    )
       
class OptionsModel(BaseModel):
    
    resources: ResourcesModel = ResourcesModel()
    max_wallclock_seconds: int = 86400
    
    custom_scheduler_commands: str = ""
    prepend_text: str = ""

    account: str = None
    queue_name: str = None

    model_config = ConfigDict(
        extra='allow',
    )
    
class MetadataModel(BaseModel):
    
    options: OptionsModel = None
    
    model_config = ConfigDict(
        extra='allow',
    )
    
class GeneralInputManager(BaseModel):
    
    """General input manager, can be used for different objects: QuantumESPRESSO, Yambo...
    
    """
    
    parameters: dict = {}
    attached_to: int = None # a node pk
    metadata: MetadataModel = MetadataModel()
    
    model_config = ConfigDict(
        extra='allow',
    )
    
    def set_time(self, time:int=86400):
        self.metadata.resources.max_wallclock_seconds=time
        
    def set_nodes(self, nodes:int=1):
        self.metadata.options.resources.num_machines=nodes
        
    def set_mpi(self, mpi:int=1):
        self.metadata.options.resources.num_mpiprocs_per_machine=mpi
        
    def set_threads(self, threads:int=1):
        self.metadata.options.resources.num_cores_per_mpiproc=threads
        
    def set_resources(self, nodes:int=None, mpi:int=None, threads:int=None, time:int=None):
        if nodes: self.set_nodes(nodes)
        if mpi: self.set_mpi(mpi)
        if threads: self.set_threads(threads)
        if time: self.set_time(time)
        return
    
    def set_scheduler_commands(self, commands:str):
        self.metadata.options.custom_scheduler_commands+=commands
        
    def set_prepend_text(self, commands:str):
        self.metadata.options.prepend_text+=commands
    
    def set_parent_folder(self,node):
        self.parent_folder = node
    
    def remove_parent_folder(self,):
        self.parent_folder = None
        
    ############ static methods ############
    
    ## These methods are very hard-coded, but honestly I think it is fine.
    
    @staticmethod
    def convert_from_AiiDA_data(inputs, get_metadata=False, verbose=False):
        from aiida import orm
        
        results = {}

        for k in inputs:
            v = getattr(inputs, k, None)
            if v is None: # skipping excluded inputs in the exposed ones, if any
                continue
            if verbose:
                print(f"getting {k}, value {v}")
            if isinstance(v, orm.InstalledCode):
                results[k] = v.full_label
            elif isinstance(v, orm.Dict):
                results[k] = v.get_dict()
            elif isinstance(v, orm.List):
                results[k] = v.get_list()
            elif isinstance(v, orm.Float):
                results[k] = v.value
            elif isinstance(v, orm.Bool):
                results[k] = v.value
            elif isinstance(v, orm.Int):
                results[k] = v.value
            elif isinstance(v, orm.Str):
                results[k] = v.value
            elif isinstance(v, orm.Node):
                results[k] = v
            elif k == 'pseudos':
                results[k] = v
            #else:
            #    results[k] = GeneralInputManager.convert_from_AiiDA_data(v)
                
        if get_metadata:
            try:
                #builder mode
                results['metadata'] = inputs.metadata 
            except:
                #node mode
                inputs.base.attributes.all['metadata_inputs']
            
        return results
    
    @staticmethod
    def to_AiiDA_inputs(
        processed_inputs,
        ):
        
        # to be called within the specific generate_AiiDA_inputs
        from aiida import orm
        
        results = {}
        for k in processed_inputs:
            v = processed_inputs.get(k, None)
            if v is None: continue
            if isinstance(v, (list,dict)):
                if len(v)==0: continue
            if k in ['metadata','pseudos']:
                results[k] = v
            elif isinstance(v, dict):
                results[k] = orm.Dict(v)
            elif isinstance(v, list):
                results[k] = orm.List(v)
            elif isinstance(v, float):
                results[k] = orm.Float(v)
            elif isinstance(v, bool):
                results[k] = orm.Bool(v)
            elif isinstance(v, int):
                results[k] = orm.Int(v)
            elif isinstance(v, str):
                if 'code' in k and not 'version' in k:
                    results[k] = orm.load_code(v)
                else:
                    results[k] = orm.Str(v)
            elif isinstance(v, orm.Node):
                results[k] = v
            
        return results

    @staticmethod
    def set_ports(
        processed_inputs,
        class_instance_ports,
        last_key: str='',
        metadata_keys: list[str]= ['pw','yambo'],
        keys_to_extract: list[str] = ['scf','nscf','qp','bse'],
    ):
        results = {}
        for k,v in class_instance_ports.items():
            skip_ports = False
            if hasattr(v,'ports'):
                if len(v.ports) == 0:
                    skip_ports = True
            if k == 'metadata' and last_key in metadata_keys:
                results[k] = processed_inputs[k] 
                last_key=''
            elif k == 'metadata':
                continue
            elif hasattr(v, 'ports') and not skip_ports:
                if k in keys_to_extract: 
                    post_processed_inputs = processed_inputs.get(k, None)
                    if post_processed_inputs is None: continue
                else:
                    post_processed_inputs = processed_inputs
                results[k] = GeneralInputManager.set_ports(post_processed_inputs,class_instance_ports[k],last_key=k, metadata_keys=metadata_keys)
            elif k not in processed_inputs: 
                continue
            else:      
                results[k] = processed_inputs[k] 
               
        return results