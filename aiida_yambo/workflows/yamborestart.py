# -*- coding: utf-8 -*-
import logging
from contextlib import contextmanager
from copy import deepcopy

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import while_
from aiida.engine.processes.workchains.restart import BaseRestartWorkChain
from aiida.engine.processes.workchains.utils import ProcessHandlerReport, process_handler
from aiida.orm import Bool, Dict, Int, RemoteData, Str
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_yambo.calculations.yambo import YamboCalculation
from aiida_yambo.utils.common_helpers import find_pw_parent, take_calc_from_remote
from aiida_yambo.utils.high_level_API.general_model import ResourcesModel
from aiida_yambo.utils.high_level_API.yambo_model import YamboRestartInputManager
from aiida_yambo.utils.parallel_namelists import check_para_namelists, check_variables
from aiida_yambo.workflows.utils.helpers_yamborestart import fix_memory, fix_parallelism, fix_time

LOGGER = logging.getLogger(__name__)

class YamboRestart(ProtocolMixin, BaseRestartWorkChain):

    """This module interacts directly with the yambo plugin to submit calculations

    This module submits calculations using the yambo plugin, and manages them, including
    restarting the calculation in case of:
    1. Memory problems (will reduce MPI parallelism before resubmitting) -- to be fixed
    2. Queue time exhaustions (will increase time by a fraction before resubmitting)
    3. Parallelism errors (will reduce the MPI the parallelism before resubmitting)  -- to be fixed
    4. Errors originating from a few select unphysical input parameters like too low bands.  -- to be fixed
    """

    _process_class = YamboCalculation
    _error_handler_entry_point = 'aiida_yambo.workflow_error_handlers.yamborestart'

    @classmethod
    def define(cls, spec):

        super(YamboRestart, cls).define(spec)
        spec.expose_inputs(YamboCalculation, namespace='yambo', namespace_options={'required': True}, \
                            exclude = ['parent_folder'])
        spec.input("parent_folder", valid_type=RemoteData, required=False) #false to build the ywfl protocols
        spec.input("max_walltime", valid_type=Int, default=lambda: Int(86400))
        spec.input("max_number_of_nodes", valid_type=Int, default=lambda: Int(0),
                    help = 'max number of nodes for restarts; if 0, it does not increase the number of nodes')
        spec.input("code_version", valid_type=Str, default=lambda: Str('5.x'))


##################################### OUTLINE ####################################

        spec.outline(
            cls.setup,
            cls.validate_parameters,
            cls.validate_resources,
            cls.validate_parent,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )


###################################################################################

        spec.expose_outputs(YamboCalculation)

        spec.exit_code(300, 'ERROR_UNRECOVERABLE_FAILURE',
            message='The calculation failed with an unrecoverable error.')
        spec.exit_code(301, 'LOW_NUMBER_OF_NSCF_BANDS',
            message='not enough bands in the Nscf dft step - nbnd - .')
        spec.exit_code(302, 'EMPTY_PARENT',
            message='parent is empty in the remote computer.')
        spec.exit_code(303, 'NO_PARENT',
            message='parent is not provided.')
    
    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from aiida_yambo.workflows.protocols import yambo as yamborestart_protocols
        return files(yamborestart_protocols) / 'yamborestart.yaml'
    
    @classmethod
    def get_builder_from_protocol(
        cls,
        preprocessing_code,
        code,
        protocol='fast',
        overrides=None,
        parent_folder=None,
        NLCC=False,
        RIM_v=False,
        RIM_W=False,
        ecutwfc=-1,
        nelectrons=0,
        **_
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        overrides = {} if overrides is None else overrides
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        if isinstance(code, str):
            
            preprocessing_code = orm.load_code(preprocessing_code)
            code = orm.load_code(code)

        inputs = cls.get_protocol_inputs(protocol, overrides={})

        meta_parameters = inputs.pop('meta_parameters')

        try:
            pw_parent = find_pw_parent(take_calc_from_remote(parent_folder,level=-1))
            ecutwfc = pw_parent.inputs.parameters.get_dict()['SYSTEM']['ecutwfc']
            nelectrons = int(pw_parent.outputs.output_parameters.get_dict()['number_of_electrons'])
        except Exception as exc: # no (valid) parent: we keep the values given as arguments
            LOGGER.debug('cannot take ecutwfc and nelectrons from the parent: %s', exc)

        # Update the parameters based on the protocol inputs
        parameters = inputs['yambo']['parameters']
        metadata = inputs['yambo']['metadata']

        # NLCC check:
        if NLCC and not 'NLCC' in parameters['arguments']:
            parameters['arguments'].append('NLCC')
        
        # RIM_v check:
        if RIM_v and not 'rim_cut' in parameters['arguments']:
            parameters['arguments'].append('rim_cut')
            parameters['variables']['RandQpts'] = [5000000, '']
            parameters['variables']['RandGvec'] = [100, 'RL']

        # RIM_W check:
        if RIM_W and not 'RIM_W' in parameters['arguments']:
            parameters['arguments'].append('RIM_W')
            parameters['variables']['CUTGeo'] = 'slab Z'
            parameters['variables']['RandGvecW'] = [13, 'RL']

        #if protocols GW
        screening_PW_cutoff = int(ecutwfc*meta_parameters['ratio_PW_cutoff'])
        screening_PW_cutoff -= screening_PW_cutoff%2 
        parameters['variables']['NGsBlkXp'] = [max(1,screening_PW_cutoff),'Ry']

        parameters['variables']['FFTGvecs'] = [int(ecutwfc*meta_parameters['ratio_FFTGvecs']),'Ry']
        
        bands = int(max(6,nelectrons/2) * meta_parameters['ratio_bands_electrons']) #want something also Volume dependent.

        parameters['variables']['BndsRnXp'] = [[1, bands], '']

        if 'bse' in protocol:
            parameters['variables']['BndsRnXs'] = parameters['variables'].pop('BndsRnXp')
            parameters['variables']['NGsBlkXs'] = parameters['variables'].pop('NGsBlkXp')
            parameters['variables']['BSENGBlk'] = parameters['variables']['NGsBlkXs']
        else:
            parameters['variables']['GbndRnge'] = parameters['variables']['BndsRnXp']

        # If overrides are provided, they are considered absolute
        if overrides:
            parameter_arguments_overrides = overrides.get('yambo', {}).get('parameters', {}).get('arguments', [])
            parameters['arguments'] += parameter_arguments_overrides
            parameters['arguments'] = list(set(parameters['arguments']))
            
            parameter_variables_overrides = overrides.get('yambo', {}).get('parameters', {}).get('variables', {})
            parameters['variables'] = recursive_merge(parameters['variables'], parameter_variables_overrides)

            metadata_overrides = overrides.get('yambo', {}).get('metadata', {})
            metadata = recursive_merge(metadata, metadata_overrides)

        if not 'QPkrange' in parameters['variables'].keys() and 'gw0' in parameters['arguments']:
            parameters['variables']['QPkrange'] = [[[1, 1, 32, 32,]], ''] #fictitious

        if not 'bse' in protocol:
            LOGGER.info('Summary of the main inputs:\nBndsRnXp = {}\nGbndRnge = {}\nNGsBlkXp = {} {}\nFFTGvecs = {} {}\n'\
                .format(parameters['variables']['BndsRnXp'][0][1],parameters['variables']['GbndRnge'][0][1],parameters['variables']['NGsBlkXp'][0],parameters['variables']['NGsBlkXp'][1],\
                parameters['variables']['FFTGvecs'][0],parameters['variables']['FFTGvecs'][1]))
        else:
            LOGGER.info('Summary of the main inputs:\nBndsRnXs = {}\nNGsBlkXs = {} {}\nBSENGBlk = {} {}\nFFTGvecs = {} {}\n'\
                .format(parameters['variables']['BndsRnXs'][0][1],parameters['variables']['NGsBlkXs'][0],parameters['variables']['NGsBlkXs'][1],\
                parameters['variables']['BSENGBlk'][0],parameters['variables']['BSENGBlk'][1],\
                parameters['variables']['FFTGvecs'][0],parameters['variables']['FFTGvecs'][1]))

        builder = cls.get_builder()
        builder.yambo['preprocessing_code'] = preprocessing_code
        builder.yambo['code'] = code
        builder.yambo['parameters'] = Dict(parameters)
        builder.yambo['metadata'] = metadata
        if 'settings' in inputs['yambo']:
            builder.yambo['settings'] = Dict(inputs['yambo']['settings'])
        builder.clean_workdir = Bool(inputs['clean_workdir'])

        if parent_folder:
            builder.parent_folder = parent_folder
        # pylint: enable=no-member

        return builder
    
    #############################################################################
    def setup(self):
        """setup of the calculation and run
        """
        super(YamboRestart, self).setup()
        # setup #
        self.ctx.inputs = self.exposed_inputs(YamboCalculation, 'yambo')

    @contextmanager
    def _edit_inputs(self):
        """Edit Python data locally, then commit changes to the restart inputs.

        ``ctx.inputs`` remains the sole persistent input state. Codes, file nodes
        and unrelated ports retain their identity; unchanged Dict nodes are reused.
        Do not inject the manager's resource defaults into scheduler metadata.
        """
        inputs = self.ctx.inputs
        parameters = inputs.parameters.get_dict()
        settings = inputs.settings.get_dict() if 'settings' in inputs else {}
        metadata = deepcopy(inputs.get('metadata', {}))
        parent = inputs.get('parent_folder')
        manager = YamboRestartInputManager(
            parameters=deepcopy(parameters), settings=deepcopy(settings),
            metadata=metadata, parent_folder=parent,
        )
        yield manager
        if manager.parameters != parameters:
            inputs.parameters = Dict(manager.parameters)
        if manager.settings != settings:
            inputs.settings = Dict(manager.settings)
        updated_metadata = manager.metadata.model_dump(exclude_unset=True)
        if updated_metadata != metadata:
            inputs.metadata = AttributeDict(updated_metadata)
        if manager.parent_folder is not parent:
            if manager.parent_folder is None:
                inputs.pop('parent_folder', None)
            else:
                inputs.parent_folder = manager.parent_folder

    def validate_parameters(self):
        """Normalize Yambo variables and check the available NSCF bands."""
        with self._edit_inputs() as manager:
            new_para = check_para_namelists(manager.parameters['variables'], self.inputs.code_version.value)
            if new_para:
                manager.parameters['variables'].update(new_para)
                self.report('adjusting parallelism namelist... please check yambo documentation')
            try:
                new_para = check_variables(manager.parameters)
                if new_para:
                    manager.parameters['variables'].update(new_para)
                    self.report('adjusting bse namelist... please check yambo documentation')
            except Exception as exc: # e.g. no BSE variables
                LOGGER.debug('no bse namelist to check: %s', exc)
            required_bands = manager.get_Nb()

        try:
            nscf_parent = find_pw_parent(take_calc_from_remote(self.inputs.parent_folder, level=-1))
        except Exception: # the parent is not a PW calculation: it is already the yambo one
            nscf_parent = take_calc_from_remote(self.inputs.parent_folder, level=-1)
        if nscf_parent.inputs.parameters.get_dict()['SYSTEM']['nbnd'] < required_bands:
            self.report('You must run an nscf with nbnd at least =  {}'.format(required_bands))
            return self.exit_codes.LOW_NUMBER_OF_NSCF_BANDS

    def validate_resources(self):
        """validation of machines... completeness and with respect para options
        """
        pass

    def validate_parent(self):
        """validation of the parent calculation --> should be at least nscf/p2y
        """
        if not hasattr(self.inputs,'parent_folder'):
            return self.exit_codes.NO_PARENT
        else:
            self.ctx.inputs['parent_folder'] = self.inputs.parent_folder
        try: #sometimes "Authentication timeout".
            if self.ctx.inputs['parent_folder'].is_empty: 
                return self.exit_codes.EMPTY_PARENT
        except Exception as exc:
            self.report(f'Could not check if the parent folder is empty ({exc}), assuming it is not.')
            self.ctx.inputs['parent_folder'] = self.inputs.parent_folder

    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.
        This should be called in a registered error handler if its condition is met and an action was taken.
        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        arguments = [calculation.process_label, calculation.pk, calculation.exit_status, calculation.exit_message]
        with self._edit_inputs() as manager:
            manager.settings['ITERATION'] = self.ctx.iteration
        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
        self.report('Action taken: {}'.format(action))

    @process_handler(priority = 600)
    def _handle_unrecoverable_failure(self, calculation):
        """
        Handle calculations with an exit status below 400 which are unrecoverable, 
        so abort the work chain.
        """        
        if calculation.exit_status < 400 and not calculation.is_finished_ok:
            self.report_error_handled(calculation, 'unrecoverable error, aborting...')
            return ProcessHandlerReport(True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE)
   
    @process_handler(priority = 590, exit_codes = [YamboCalculation.exit_codes.NO_SUCCESS])
    def _handle_unknown_error(self, calculation):
        """
        Handle calculations for an unknown reason; 
        we copy the SAVE already created, if any.
        """
        
        with self._edit_inputs() as manager:
            manager.settings['COPY_SAVE'] = True

        self.report_error_handled(calculation, 'Trying to copy the SAVE folder and restart')
        return ProcessHandlerReport(True)
    
    
    @process_handler(priority = 559, exit_codes = [YamboCalculation.exit_codes.WALLTIME_ERROR])
    def _handle_walltime_error(self, calculation):
        """
        Handle calculations for a walltime error; 
        we increase the simulation time and copy the database already created.
        """
        
        with self._edit_inputs() as manager:
            options = manager.metadata.options.model_dump(exclude_unset=True)
            options = fix_time(options, self.ctx.iteration, self.inputs.max_walltime.value)
            manager.set_time(options['max_wallclock_seconds'])
            manager.set_parent_folder(calculation.outputs.remote_folder)
            if calculation.outputs.output_parameters.get_dict()['yambo_wrote_dbs']:
                manager.settings['COPY_DBS'] = True

        self.report_error_handled(calculation, 'walltime error detected, so we increase time: {} \
                                                seconds and link outputs'\
                                                .format(int(self.ctx.inputs.metadata.options['max_wallclock_seconds'])))
        return ProcessHandlerReport(True)

    @process_handler(priority =  560, exit_codes = [YamboCalculation.exit_codes.PARA_ERROR])
    def _handle_parallelism_error(self, calculation):
        """
        Handle calculations for a parallelism error; 
        we try to change the parallelism options.
        """
        with self._edit_inputs() as manager:
            new_para, new_resources, pop_list = fix_parallelism(
                manager.metadata.options.resources.model_dump(exclude_unset=True), calculation)
            if new_para is False or new_resources is False:
                return ProcessHandlerReport(True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE)
            self._apply_parallelism_fix(manager, new_para, new_resources, pop_list)
            if calculation.outputs.output_parameters.get_dict()['yambo_wrote_dbs']:
                manager.set_parent_folder(calculation.outputs.remote_folder)
                manager.settings['COPY_DBS'] = True

        self.report_error_handled(calculation, 'parallelism error detected, so we try to fix it setting PAR_def_mode= "balanced"')
        return ProcessHandlerReport(True)

    @process_handler(priority =  561, exit_codes = [YamboCalculation.exit_codes.MEMORY_ERROR, \
                                                    YamboCalculation.exit_codes.X_par_MEMORY_ERROR])
    def _handle_memory_error(self, calculation):
        """
        Handle calculations for a memory error; 
        we try to change the parallelism options, in particular the mpi-openmp balance.
        if cpu_per_task(mpi/node) is already set to 1, we can increase the number of nodes,
        accordingly to the inputs permissions.
        """
        with self._edit_inputs() as manager:
            new_para, new_resources, pop_list = fix_memory(
                manager.metadata.options.resources.model_dump(exclude_unset=True),
                calculation, calculation.exit_status, self.inputs.max_number_of_nodes.value,
                self.ctx.iteration)
            if new_para is False or new_resources is False:
                return ProcessHandlerReport(True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE)
            self._apply_parallelism_fix(manager, new_para, new_resources, pop_list)
            if calculation.outputs.output_parameters.get_dict()['yambo_wrote_dbs']:
                manager.set_parent_folder(calculation.outputs.remote_folder)
                manager.settings['COPY_DBS'] = True

        self.report_error_handled(calculation, 'memory error detected, so we change mpi-openmpi balance and set PAR_def_mode= "balanced"')
        return ProcessHandlerReport(True)

    @process_handler(priority =  562, exit_codes = [YamboCalculation.exit_codes.Variable_NOT_DEFINED])
    def _handle_variable_NOT_DEFINED(self, calculation):
        """
        Handle calculations Variable NOT DEFINED error, happens with ndb.pp_fragments.
        redo the calculation, trying to delete the wrong fragment and recompute it.
        """

        with self._edit_inputs() as manager:
            output_parameters = calculation.outputs.output_parameters.get_dict()
            if output_parameters['yambo_wrote_dbs']:
                corrupted_fragment = output_parameters['corrupted_fragment']
                manager.set_parent_folder(calculation.outputs.remote_folder)
                manager.set_prepend_text("\nrm aiida.out/" + str(corrupted_fragment))
                manager.settings['COPY_DBS'] = True

        self.report_error_handled(calculation, 'Variable NOT DEFINED error detected, so we restart and recompute the corrupted fragment')
        return ProcessHandlerReport(True)

    @staticmethod
    def _apply_parallelism_fix(manager, variables, resources, pop_list):
        """Apply a resolved parallelism change to the local input manager."""
        manager.metadata.options.resources = ResourcesModel(**resources)
        manager.set_prepend_text("\nexport OMP_NUM_THREADS=" + str(resources['num_cores_per_mpiproc']))
        manager.parameters.setdefault('variables', {}).update(variables)
        manager.remove_variables(pop_list)
