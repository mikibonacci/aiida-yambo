# -*- coding: utf-8 -*-
from __future__ import absolute_import

from aiida import orm
from aiida.orm import Dict, Str, Bool, KpointsData, List, load_node, Group, load_group
from aiida.engine import WorkChain, while_ , if_
from aiida.engine import ToContext
from aiida.plugins import WorkflowFactory

from aiida_quantumespresso.common.types import ElectronicType, SpinType

from aiida_yambo.workflows.utils.helpers_aiida_yambo import *
from aiida_yambo.workflows.utils.helpers_workflow import *
from aiida_yambo.utils.common_helpers import *
from aiida_yambo.workflows.utils.helpers_yambowf import *

from aiida_yambo.workflows.utils.parameters import (
    ConvParam,
    load_parameter_space
)

from aiida_yambo.workflows.utils.convergence import (
    ConvergenceEvaluator
)



from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

YamboWorkflow = WorkflowFactory('yambo.yambo.yambowf')

class YamboConvergence(ProtocolMixin, WorkChain):

    """This workflow will perform yambo convergences with respect to some parameter. It can be used also to run multi-parameter
       calculations.
    """

    @classmethod
    def define(cls, spec):
        """WorkChain definition

        """
        super(YamboConvergence, cls).define(spec)

        spec.expose_inputs(
            YamboWorkflow, 
            namespace="ywfl", 
            namespace_options={
                "required": True,
                "populate_defaults": False
            }
        )
        spec.input(
            "parameters_space", 
            valid_type=orm.List, 
            required=True,
            help = 'variables to converge, range, steps, and max iterations'
        )
        spec.input(
            "workflow_settings", 
            valid_type=orm.Dict, 
            required=True,
            help = 'settings for the workflow: type, quantities to be examinated...'
        )
        spec.input(
            "parallelism_instructions", 
            valid_type=orm.Dict, 
            required=False,
            help = 'indications for the parallelism to be used wrt values of the parameters.'
        )
        spec.input(
            "group_label", 
            valid_type=orm.Str, 
            required=False, 
            help = 'group of calculations already done for this system.'
        )

##################################### OUTLINE ####################################

        spec.outline(
            cls.start_workflow,
            while_(cls.has_to_continue)(
                if_(cls.p2y_needed)(
                    cls.do_p2y,
                    cls.prepare_calculations
                ),
                cls.update_next_step,
                cls.perform_next_step,
                cls.data_analysis),
            cls.report_wf,
        )

##################################################################################

        spec.expose_outputs(YamboWorkflow) #the last calculation
        spec.output(
            'collection', 
            valid_type=orm.Dict, 
            help='all calculations'
        )
        spec.output(
            'convergence_summary', 
            valid_type=orm.Dict, 
            required=False,
            help='information on the convergence' 
        )

        spec.exit_code(300, 'UNDEFINED_STATE',
                             message='The workchain is in an undefined state.') 
        spec.exit_code(301, 'PRECALC_FAILED',
                             message='The workchain failed the precalc step.')    
        spec.exit_code(302, 'CALCS_FAILED',
                             message='The workchain failed some calculations.')       
        spec.exit_code(303, 'SPACE_TOO_SMALL',
                             message='The workchain failed because the space is too small.')                           
        spec.exit_code(400, 'CONVERGENCE_NOT_REACHED',
                             message='The workchain failed to reach convergence.')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from aiida_yambo.workflows.protocols import yambo as yamboconvergence_protocols
        return files(yamboconvergence_protocols) / 'yamboconvergence.yaml'
    
    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code:orm.Code,
        preprocessing_code:orm.Code,
        code:orm.Code,
        structure:orm.StructureData=None,
        protocol_qe:str='moderate',
        protocol:str='moderate',
        calc_type:str='gw',
        workflow_settings:dict=None,
        convergence_parameters:list=['FFTGvecs',['BndsRnXp','GbndRnge','NGsBlkXp'],'kpoint_density'],
        overrides:dict={},
        NLCC:bool=False,
        RIM_v:bool=False,
        RIM_W:bool=False,
        parent_folder:orm.RemoteData=None,
        pseudo_family:str=None,
        electronic_type=ElectronicType.INSULATOR,
        spin_type=SpinType.NONE,
        initial_magnetic_moments=None,
        **_
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        :return: a process builder instance with all inputs defined ready for launch.
        
        By default, we will do convergence on: 'FFTGvecs','BndsRnXp','GbndRnge','NGsBlkXp','kpoint_density'
        using the automated workflow as described in Bonacci et al. npj Comput Mater 9, 74 (2023).
        """

        # Setting the inputs: merging protocols and overrides
        inputs = cls.get_protocol_inputs(protocol, overrides=overrides)

        # Getting the protocols to determine the parameters_space input
        meta_parameters = inputs.pop('meta_parameters',{})
        
        # Initialization of the builder
        builder = cls.get_builder()

        # Getting the YamboWorkflow overrides, to pass in its corresponding constructor below
        overrides_ywfl = overrides.pop('ywfl',{})
        overrides_ywfl['clean_workdir'] = overrides_ywfl.pop('clean_workdir',False)            

        ######### YamboWorkflow initialization #########
        ywfl_builder = YamboWorkflow.get_builder_from_protocol(
                pw_code,
                preprocessing_code,
                code,
                structure=structure,
                protocol_qe=protocol_qe,
                protocol=protocol,
                overrides=overrides_ywfl,
                calc_type=calc_type,
                NLCC=NLCC,
                RIM_v=RIM_v,
                RIM_W=RIM_W,
                electronic_type=electronic_type,
                spin_type=spin_type,
                initial_magnetic_moments=initial_magnetic_moments,
                parent_folder=parent_folder,
                pseudo_family=pseudo_family,
                **_ # here we can pass more parameters specific to YamboWorkflow
        )

        builder.ywfl = ywfl_builder._inputs(prune=True)
        
        
        if workflow_settings:
            builder.workflow_settings = orm.Dict(workflow_settings)
        else:
            builder.workflow_settings = orm.Dict(overrides.get('workflow_settings', None))
        
        if calc_type=='bse':
            builder.workflow_settings['what'] = ['lowest_exciton','brightest_exciton']
            builder.workflow_settings = orm.Dict(inputs['workflow_settings'])

        ################ K-points density ################
        builder.ywfl['nscf']['kpoints'] = orm.KpointsData()
        builder.ywfl['nscf']['kpoints'].set_cell_from_structure(builder.ywfl['nscf']['pw']['structure'])

        builder.ywfl['nscf']['kpoints'].set_kpoints_mesh_from_density(meta_parameters['kpoint_density']['max'],force_parity=True)
        k_end = builder.ywfl['nscf']['kpoints'].get_kpoints_mesh()[0]
        
        builder.ywfl['nscf']['kpoints'].set_kpoints_mesh_from_density(meta_parameters['kpoint_density']['stop'],force_parity=True)
        k_stop = builder.ywfl['nscf']['kpoints'].get_kpoints_mesh()[0]

        builder.ywfl['nscf']['kpoints'].set_kpoints_mesh_from_density(meta_parameters['kpoint_density']['start'],force_parity=True)
        k_start = builder.ywfl['nscf']['kpoints'].get_kpoints_mesh()[0]
        
        k_delta = np.zeros(3,dtype='int64')
        k_delta[np.where(builder.ywfl['nscf']['pw']['structure'].pbc)] = int(meta_parameters['kpoint_density']['delta'])
        k_delta = list(k_delta)

        # Getting information on the nelectrons, ecutwfc and volume
        nelectrons, ecutwfc = periodical(structure.get_ase()) #nelectrons just as hint for later, not guaranteed to be correct (different pseudos can have different nelectrons)
        ecutwfc = builder.ywfl['nscf']['pw']['parameters'].get_dict()['SYSTEM']['ecutwfc']
        volume = structure.get_cell_volume()
        
        ################ FFTGVecs ################
        FFT_start=int(meta_parameters['FFTGvecs']['start_ratio']*ecutwfc)
        FFT_stop=int(meta_parameters['FFTGvecs']['stop_ratio']*ecutwfc)
        FFT_max=int(meta_parameters['FFTGvecs']['max_ratio']*ecutwfc)
        FFT_delta=int(meta_parameters['FFTGvecs']['delta_ratio']*ecutwfc)
        
        ################ Nb: bands for the X and Sc ################
        # here the logic considers a ratio which depends on the dimensionality, or better on the volume:
        # if the volume is lower then 
        
        b_start=meta_parameters['bands']['start']
        b_stop=meta_parameters['bands']['stop']
        b_max=meta_parameters['bands']['max']
        b_delta=meta_parameters['bands']['delta']

        volume_factor = np.log10(volume) # TODO: put the correct factor and describe it!!!
        print('WARNING: still to fix the factor!!!!')
        b_start=meta_parameters['bands']['start']*volume_factor
        b_stop=meta_parameters['bands']['stop']*volume_factor
        b_max=meta_parameters['bands']['max']*volume_factor
        
        ################ G_cut for X, Sc, BSE kernel ################
        G_start=meta_parameters['G_vectors']['start']
        G_stop=meta_parameters['G_vectors']['stop']
        G_max=meta_parameters['G_vectors']['max']
        G_delta=meta_parameters['G_vectors']['delta']

        ######################### PARAMETERS SPACE CREATION #########################
        builder.parameters_space =  orm.List([
            {
                           'var':['FFTGvecs'],
                           'start': FFT_start,
                           'stop':FFT_stop ,
                           'delta':FFT_delta,
                           'max':FFT_max,
                           'steps': 4, 
                           'max_iterations': 4, \
                           'conv_thr': meta_parameters['convergence_thesholds']['FFTgvecs'],
                           'conv_thr_units':'%',
                           'convergence_algorithm':'new_algorithm_1D',
                           },
            {
                           'var':['kpoint_mesh'],
                           'start': k_start,
                           'stop': k_stop ,
                           'delta': k_delta,
                           'max':k_end,
                           'steps': 4, 
                           'max_iterations': 4, \
                           'conv_thr': meta_parameters['convergence_thesholds']['kpoints'],
                           'conv_thr_units':'%',
                           'convergence_algorithm':'new_algorithm_1D',
                           },  
                           
            {
                           'var':['BndsRnXp','GbndRnge','NGsBlkXp'],
                           'start': [b_start,b_start,G_start],
                           'stop':[b_stop,b_stop,G_stop] ,
                           'delta':[b_delta,b_delta,G_delta],
                           'max':[b_max,b_max,G_max],
                           'steps': 6, 
                           'max_iterations': 8, \
                           'conv_thr': meta_parameters['convergence_thesholds']['Nb_Gcut'],
                           'conv_thr_units':'%',
                           'convergence_algorithm':'new_algorithm_2D',
                           },
                        
                        ])

        if protocol == 'molecule' or structure.pbc.count(True)==0:
            builder.parameters_space = orm.List(builder.parameters_space.get_list()[::2]) #no k-points, so we skip the second element in the parameters_space.

        return builder
    
    def from_yamboworkflow(self,):
        raise NotImplementedError
    
    def from_yambopy(self,):
        raise NotImplementedError
        
    def start_workflow(self):
        """Initialize the workflow"""
        self.ctx.limit_reached = False # this checks if we cannot increase the parameter more than the `max` value.
        self.ctx.convergence_summary = {}   #converged parameters
        self.ctx.calc_inputs = self.exposed_inputs(YamboWorkflow, 'ywfl')        
        self.ctx.hint = {}
        self.ctx.global_step = 0
        self.ctx.wfl_pk = []
        self.ctx.workflow_settings = self.inputs.workflow_settings.get_dict()
        self.ctx.bands_mode = self.ctx.workflow_settings.pop('bands_nscf_update', 0)
        self.ctx.parameters_space = copy.deepcopy(load_parameter_space(self.inputs.parameters_space))
        parameters_space = copy.deepcopy(load_parameter_space(self.inputs.parameters_space))

        # we add the quantity to be converged to the additional_parsing list.
        if hasattr(self.ctx.calc_inputs,'additional_parsing'):
            l = self.ctx.workflow_settings['what']+self.ctx.calc_inputs.additional_parsing.get_list()
            self.ctx.calc_inputs.additional_parsing = orm.List(list(dict.fromkeys(l)))
        else:
            self.ctx.calc_inputs.additional_parsing = orm.List(list(self.ctx.workflow_settings['what']))

        # here we load or create the group where to put all the calculations.
        if hasattr(self.inputs, "group_label"):
            self.ctx.group = orm.load_group(self.inputs.group_label.value)
            self.report('group: {}'.format(self.inputs.group_label.value))
        else:
            try:
                self.ctx.group = orm.load_group("convergence_tests_{}".format(self.ctx.calc_inputs.scf.pw.structure.get_formula()))
            except:
                self.ctx.group = Group(label="convergence_tests_{}".format(self.ctx.calc_inputs.scf.pw.structure.get_formula()))
                self.report('creating group: {}'.format(self.ctx.calc_inputs.scf.pw.structure.get_formula()))
            if not self.ctx.group.is_stored: self.ctx.group.store()
        
        # here we add to the ctx the parallelism instructions, i.e. the parameter-dependent set of resources to be used.
        if hasattr(self.inputs, "parallelism_instructions"):
            self.ctx.parallelism_instructions = build_parallelism_instructions(self.inputs.parallelism_instructions.get_dict(),)
        else:
            self.ctx.parallelism_instructions = {}  
        
        # this calc_manager can be removed? we substitue it with self.ctx.current_convergence
        self.ctx.current_convergence = ConvergenceEvaluator(
            parameters=self.ctx.parameters_space.pop(),
            what=self.ctx.workflow_settings['what']
        ) 

        self.ctx.final_result = {}     

        self.report(
            "Workflow type: {}; looking for convergence of {}"
            .format(
                self.ctx.workflow_settings['type'], 
                self.ctx.current_convergence.what,
            )
        )
        self.report(
            "Workflow initilization step completed, the parameters will be: {}."
            .format(
                [params.var for params in parameters_space]
            )
        )

    def has_to_continue(self):
        
        """This function checks the status of the last calculation and determines what happens next, 

        """
        
        if self.ctx.current_convergence.cannot_stop:
            self.report('We should finish the set of calculations before the analysis.')
            return True
        
        #failed cases
        elif self.ctx.current_convergence.success and len(self.ctx.parameters_space)==0: 
            self.report('Workflow finished')
            return False
        elif not self.ctx.current_convergence.success and self.ctx.current_convergence.iterations_exceeded:
            self.report('Workflow failed due to max restarts exceeded for variable {}'.format(self.ctx.calc_manager['var']))
            return False
        elif self.ctx.current_convergence.edges_exceeded:
            self.report('space not large enough to complete convergence')
            return False
        elif not self.ctx.current_convergence.success and not self.ctx.current_convergence.iterations_exceeded:
            self.report('Still iteration on {}'.format(self.ctx.current_convergence.parameters.var))
            self.ctx.current_convergence.advance_iteration()
            return True
        
        # TODO: change the following we add some tolerance for failed calculations, to try to anyway converge
        elif not self.ctx.calc_manager['success'] and \
                   self.ctx.calc_manager['iter'] > self.ctx.calc_manager['max_iterations']:
            self.report('Iterations: {} - Max iterations: {}'.format(self.ctx.calc_manager['iter'],self.ctx.calc_manager['max_iterations']))
            self.report('Workflow failed due to some failed calculation in the investigation of {}'.format(self.ctx.calc_manager['var']))

            return False
        
        # successful case:
        elif self.ctx.current_convergence.success:                
            if len(self.ctx.parameters_space) == 0:
                self.report("No more parameters to be converged, the workflow will not continue.")
                return False
            self.ctx.hint = {}
            return True
        else:
            self.report('Undefined state on {}, so we exit'.format(self.ctx.calc_manager['var']))
            self.ctx.calc_manager['success'] = 'undefined'
            return False
        
    def update_next_step(self):
        
        if self.ctx.current_convergence.success:
            self.ctx.current_convergence = ConvergenceEvaluator(
                self.parameters_space.pop(),
                what=self.ctx.workflow_settings['what']
            ) 
            
            if self.ctx.workflow_settings.get('type') == 'cheap':
                self.report('Mode is "cheap", so we reset the other parameters to the initial ones.')
                self.ctx.calc_inputs = self.exposed_inputs(YamboWorkflow, 'ywfl')
                if hasattr(self.ctx.calc_inputs,'additional_parsing'):
                    l = self.ctx.current_convergence.what+self.ctx.calc_inputs.additional_parsing.get_list()
                    self.ctx.calc_inputs.additional_parsing = List(list(dict.fromkeys(l)))
                else:
                    self.ctx.calc_inputs.additional_parsing = List(list(self.ctx.current_convergence.what))
                self.ctx.convergence_summary.update(self.ctx.hint)
            else:
                self.report('Mode is "heavy", so we keep the previous parameters to be the converged ones, if any.')
        
        self.report(f'Convergence on: {self.ctx.current_convergence.parameters.var}')
        self.report(f'With them, we mirror the following parameters {self.ctx.current_convergence.parameters.mirror_values}')
        

    def perform_next_step(self):
        """This function will submit the next step"""
        
        # 1. Grab our smart evaluator tracker from context
        evaluator = self.ctx.current_convergence

        calc = {}
                
        if evaluator.new_grid: # shift the grid
            # if iteration is not the first, we shift
            if evaluator.current_iteration > 1:
                self.ctx.params_space = self.ctx.params_space.with_shift(
                        start_shift = self.ctx.delta, 
                        stop_shift = self.ctx.delta
                    )
                evaluator.concurrent_steps_count=0
        
        # Dynamically picks build_linear_space() or custom grid based on the class type
        if isinstance(evaluator.parameters, ConvParam):
            self.ctx.params_space = evaluator.parameters.build_stepped_space()
        else:
            # Returns your clean 2D corner/cross grid
            self.ctx.params_space = evaluator.parameters.build_diagonal_corners()
        
        if evaluator.check_single_point:
            self.report(f'Now we verify the accuracy of the prediction by computing a single predicted point: {evaluator.inquired_point}')
            self.ctx.params_space = [evaluator.inquired_point]
            evaluator.concurrent_steps_count = evaluator.parameters.steps-1
        
        self.report(f"Running a max of {evaluator.parameters.max_concurrent_steps} concurrent steps of {evaluator.parameters.steps}")
        for i in range(evaluator.parameters.max_concurrent_steps):
            
            if evaluator.concurrent_steps_count >= evaluator.parameters.steps:
                break
            
            # Your updater stays the same, but passes explicit objects/grids cleanly
            self.ctx.calc_inputs, values, already_done, parent_nscf = updater(
                calc_inputs=self.ctx.calc_inputs,
                evaluator=evaluator,
                explicit_value=self.ctx.params_space, 
                workflow_dict=self.ctx.parallelism_instructions,
                index=evaluator.concurrent_steps_count if not evaluator.inquired_point else 0, # so if iteration = 1 and not finished, should start from evaluator.max_concurrent_steps+1
                group=self.ctx.group,
            )
                                    
            self.report(f"New parameters are: {values}")
            
            if not already_done:
                label = f"iteration_{self.ctx.global_step}"
                self.ctx.calc_inputs.metadata.call_link_label = label
                
                future = self.submit(YamboWorkflow, **self.ctx.calc_inputs)
                self.ctx.group.add_nodes(future.caller)
            else:
                self.report(f"Calculation already done: {already_done}")
                future = load_node(already_done)

            calc[str(i+1)] = future
            self.ctx.wfl_pk.insert(0, future.pk)  # Cleaner way to prepend to list
            self.ctx.group.add_nodes(future)
            
            evaluator.concurrent_steps_count += 1
            self.ctx.global_step += 1
            
            if len(self.ctx.params_space) == 0:
                break

        self.ctx.current_convergence=evaluator
        
        return ToContext(calc)

    def data_analysis(self):
        """Analyze completed calculations using the ConvergenceEvaluator object.
        
        This replaces the old, dictionary-heavy analysis_and_decision loops.
        """
        
        if self.ctx.current_convergence.cannot_stop:
            self.report("Continuing the submission to complete the step.")
            return
                    
        self.report('Starting unified data analysis step...')
        evaluator = self.ctx.current_convergence

        # 1. Dynamically retrieve outputs from completed calculations in this context iteration
        # Assumes your workflow outputs are mapped via ToContext(calc) dynamically as strings '1', '2', etc.
        completed_calcs = []
        for key in list(self.ctx):
            if key.isdigit() and hasattr(self.ctx, key):
                completed_calcs.append(getattr(self.ctx, key))

        if not completed_calcs:
            self.report("Error: No calculation nodes found to analyze.")
            return self.exit_codes.UNDEFINED_STATE

        # 2. Let the Evaluator object process the finished calculations directly
        # It handles parsing internal outputs ('what'), tracking historical states, and counting iterations
        if 1: #try:
            evaluator.evaluate_nodes(completed_calcs)
        #except Exception as e:
         #   self.report(f"Failed during node evaluation/parsing: {str(e)}")
          #  self.ctx.none_encountered = True  # Signal calculation errors
           # return self.exit_codes.CALCS_FAILED

        # 3. Request next-step execution strategy from your Convergence Evaluator
        decision = evaluator.get_decision()
        self.ctx.hint = decision.get('hints', {})
        
        self.report(f"Evaluator decision: status={decision['status']} | Hints: {self.ctx.hint}")

        # 4. Handle state machine responses based on clean evaluator decisions
        if decision['status'] == 'CONVERGED':
            self.report(f"Success! Parameter group {evaluator.parameters.var} has converged.")
            
            # Store hints/converged data into the global tracker
            self.ctx.convergence_summary.update(evaluator.get_summary())
            
            # Clean up current dynamic context trackers for the next iteration group
            for key in list(self.ctx):
                if key.isdigit():
                    delattr(self.ctx, key)
                    
        elif decision['status'] == 'CONTINUE':
            self.report(f"Convergence on {evaluator.parameters.var} not met yet. Refining parameter values.")
            # Evaluator keeps its internal updated space, perform_next_step will pick up the updated step/shift.
            
        elif decision['status'] == 'FAILED_MAX_ITERATIONS':
            self.report(f"Workflow reached max limit thresholds for {evaluator.parameters.var}.")
            evaluator.success = False
            
        elif decision['status'] == 'SPACE_EXCEEDED':
            self.report("The boundary configuration space is too small to fulfill convergence constraints.")
            self.ctx.limit_reached = True

        # Clean the context loop's temporary counter
        evaluator.concurrent_steps_count = 0

    def report_wf(self):
        """Finalize outputs and set workflow exit status after convergence ends."""

        evaluator = self.ctx.current_convergence
        self.report('Final step. Workflow success: {}'.format(evaluator.success))

        summary = self.ctx.convergence_summary.copy() if hasattr(self.ctx, 'convergence_summary') else {}
        summary.pop('new_grid', None)
        summary.pop('already_computed', None)
        summary.pop('extrapolation_units', None)
        self.out('collection', store_Dict(summary))
        self.out('convergence_summary', store_Dict(summary))

        try:
            if self.ctx.final_result.get('uuid'):
                calc = load_node(self.ctx.final_result['uuid'])
                if evaluator.success:
                    calc.set_extra('converged', True)
                self.out_many(self.exposed_outputs(calc, YamboWorkflow))
        except Exception:
            self.report('no YamboWorkflows available to expose outputs')

        if self.ctx.none_encountered:
            self.report('Some calculation failed, so we stopped the workflow')
            return self.exit_codes.CALCS_FAILED
        elif self.ctx.limit_reached:
            self.report('Space too small to complete convergence.')
            return self.exit_codes.SPACE_TOO_SMALL
        elif not evaluator.success:
            self.report('Convergence not reached')
            return self.exit_codes.CONVERGENCE_NOT_REACHED
        elif self.ctx.calc_manager['success'] == 'undefined':
            self.report('Undefined state')
            return self.exit_codes.UNDEFINED_STATE    

############################### preliminary calculation #####################
    def p2y_needed(self):
        """Determine whether a preliminary calculation is required before convergence."""

        if self.ctx.workflow_settings.get('skip_pre'):
            self.report('skipping pre (debug mode)')
            return False

        current_parameters = self.ctx.current_convergence.parameters
        if not hasattr(self.ctx, 'params_space'):
            self.ctx.params_space = copy.deepcopy(current_parameters)

        self.report(f'Current convergence variables: {current_parameters.var}')

        if any(v in ('kpoint_mesh', 'kpoint_density') for v in current_parameters.var):
            return False

        def _param_stop_value(identifier='nbnd'):
            name = None
            for n in current_parameters.var:
                if identifier in n.lower():
                    name = n
            if not name:
                return 0
            index = current_parameters.var.index(name)
            stop = current_parameters.stop[index]
            if isinstance(stop, list):
                return stop[index]
            return stop

        self.ctx.gwbands = _param_stop_value(identifier='nbnd')
        if isinstance(self.ctx.bands_mode, int) and self.ctx.gwbands > 0:
            self.ctx.gwbands = min(self.ctx.gwbands, self.ctx.bands_mode)

        try:
            already_done, parent_nscf, parent_scf = search_in_group(self.ctx.calc_inputs, self.ctx.group, up_to_p2y=True)

            if already_done:
                try:
                    self.ctx.calc_inputs.parent_folder = load_node(already_done).outputs.remote_folder
                except Exception:
                    pass
                return False

            if parent_nscf:
                try:
                    self.ctx.calc_inputs.parent_folder = load_node(parent_nscf).outputs.remote_folder
                except Exception:
                    pass
            elif parent_scf:
                try:
                    self.ctx.calc_inputs.parent_folder = load_node(parent_scf).outputs.remote_folder
                except Exception:
                    pass

            scf_params, nscf_params, redo_nscf, self.ctx.bands, messages = quantumespresso_input_validator(self.ctx.calc_inputs)
            self.report(messages)
            self.ctx.gwbands = max(self.ctx.gwbands, self.ctx.bands)
            parent_calc = take_calc_from_remote(self.ctx.calc_inputs.parent_folder)
            nbnd = nscf_params.get_dict()['SYSTEM']['nbnd']

            if nbnd < self.ctx.gwbands:
                set_parent(self.ctx.calc_inputs, find_pw_parent(parent_calc, calc_type=['scf']))
                return True

            if parent_calc.process_type == 'aiida.calculations:yambo.yambo' and not hasattr(self.inputs, 'precalc_inputs'):
                return False

            return hasattr(self.inputs, 'precalc_inputs') or True
        except Exception:
            try:
                already_done, parent_nscf, parent_scf = search_in_group(self.ctx.calc_inputs, self.ctx.group, up_to_p2y=True)

                if already_done:
                    set_parent(self.ctx.calc_inputs, load_node(already_done))
                    return False
                elif parent_nscf:
                    set_parent(self.ctx.calc_inputs, load_node(parent_nscf))
                    return True
                else:
                    parent_calc = take_calc_from_remote(self.ctx.calc_inputs.parent_folder)
                    set_parent(self.ctx.calc_inputs, find_pw_parent(parent_calc, calc_type=['scf']))
                    return True
            except Exception:
                return True

    def do_p2y(self):
        """Submit the pre-calculation or parent preparation step."""
        self.ctx.pre_inputs = self.exposed_inputs(YamboWorkflow, 'ywfl')
        self.ctx.pre_inputs.yres.clean_workdir = Bool(False)

        if hasattr(self.ctx.calc_inputs, 'parent_folder'):
            set_parent(self.ctx.pre_inputs, self.ctx.calc_inputs.parent_folder)
        
        if hasattr(self.ctx.calc_inputs.nscf,'kpoints'):
            self.report('mesh check')
            self.ctx.pre_inputs.nscf.kpoints = self.ctx.calc_inputs.nscf.kpoints

        self.ctx.calculation_type='p2y'
        self.ctx.pre_inputs.yres.yambo.settings = update_dict(self.ctx.pre_inputs.yres.yambo.settings, 'INITIALISE', True)
            
        if hasattr(self.ctx.pre_inputs, 'additional_parsing'):
            delattr(self.ctx.pre_inputs, 'additional_parsing')

        self.report('doing the calculation: {}'.format(self.ctx.calculation_type))
        calc = {}
        self.ctx.pre_inputs.metadata.call_link_label = self.ctx.calculation_type
        calc[self.ctx.calculation_type] = self.submit(YamboWorkflow, **self.ctx.pre_inputs) #################run
        self.ctx.PRE = calc[self.ctx.calculation_type]
        self.report('Submitted YamboWorkflow up to {}, pk = {}'.format(self.ctx.calculation_type,calc[self.ctx.calculation_type].pk))
        self.ctx.group.add_nodes(calc[self.ctx.calculation_type]) #when added the whole YC, remove that

        orm.load_node(calc[self.ctx.calculation_type].pk).label = self.ctx.calculation_type

        self.ctx.calc_inputs.yres.yambo.settings = update_dict(self.ctx.calc_inputs.yres.yambo.settings, 'INITIALISE', False)

        return ToContext(calc)

    def prepare_calculations(self):
        """Attach the pre-calculation remote folder as parent before launching convergence calcs."""
        if not self.ctx.PRE.is_finished_ok:
            self.report('the pre calc was not succesful, exiting...')
            return self.exit_codes.PRECALC_FAILED

        self.report('setting the pre calc remote folder {} as parent'.format(self.ctx.PRE.outputs.remote_folder.pk))
        set_parent(self.ctx.calc_inputs, self.ctx.PRE.outputs.remote_folder)


if __name__ == "__main__":
    pass
