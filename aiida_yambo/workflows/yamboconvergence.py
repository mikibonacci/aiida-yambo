# -*- coding: utf-8 -*-
import copy
import math
from typing import Literal

import numpy as np
import pandas as pd

from aiida import orm
from aiida.common.exceptions import NotExistent
from aiida.engine import ToContext, WorkChain, if_, while_
from aiida.orm import Group, load_node
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_yambo.utils.common_helpers import (
    find_pw_parent, is_nscf_valid, search_in_group, store_Dict, take_calc_from_remote,
)
# the new BaseModel manager for the inputs
from aiida_yambo.utils.high_level_API.yambo_model import YamboInputManager
from aiida_yambo.utils.resource_rules import snapshot_base, validate_resource_rules
from aiida_yambo.workflows.utils.convergence import ConvergenceEvaluator
from aiida_yambo.workflows.utils.helpers_aiida_yambo import take_quantities, updater
from aiida_yambo.workflows.utils.helpers_yambowf import quantumespresso_input_validator
from aiida_yambo.workflows.utils.parameters import ConvParam, CoupledConvParam, KpointMeshConvParam, ScanParam, load_parameter_space
from aiida_yambo.workflows.yambowf import YamboWorkflow

# Columns shared by every convergence record (mirrors update_story_global + pk-based).
# The per-parameter value columns (parameters.var) and the `what` result columns
# are appended at setup time, since they depend on the parameter group under study.
CONVERGENCE_DF_COLUMNS = ['pk', 'global_step', 'uuid', 'failed', 'useful', 'parameters_studied', 'converged']

def _to_jsonable(obj):
    """Recursively convert numpy scalars (and containers) to plain python types.

    The predictors' hints carry e.g. np.int64/np.float64 values, which are not
    JSON-serializable by aiida's Dict node. NaN/Inf are mapped to None because
    aiida's clean_builtin rejects them. All floats are rounded to 3 decimals
    at the store boundary (does not affect internal thr/MAE math).
    """
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, float):
        return round(obj, 3)
    if isinstance(obj, np.generic):
        v = obj.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, float):
            return round(v, 3)
        return v
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    return obj

def _build_params_space(parameters):
    """The values to be computed in an iteration: the explicit points of a scan, the steps of a 1D
    parameter, the corners of the box of a coupled (2D) parameter."""
    if isinstance(parameters, ScanParam):
        return parameters.build_points()
    if isinstance(parameters, ConvParam):
        return parameters.build_stepped_space()
    return parameters.build_diagonal_corners()

def _max_bands(parameters_list, reach='max'):
    """The largest number of bands the convergence parameters (`ConvParam` or `CoupledConvParam`) ask for.

    `reach='max'`: the `max` of each parameter, the limit the algorithm may reach if the convergence needs it;
    `reach='stop'`: the last value of the grid which is built (the `max` of the points for a scan).
    """
    band_max = 0
    for parameters in parameters_list:
        limit = getattr(parameters, 'stop', None) if reach == 'stop' else None
        if limit is None:
            limit = parameters.max
        for idx, var in enumerate(parameters.var):
            if 'bnd' in var.lower():
                value = limit[idx] if isinstance(limit, list) else limit
                if value is not None:
                    band_max = max(band_max, int(value))
    return band_max

#: how the NSCF number of bands follows the bands of the parameters: see `YamboConvergence`, `bands_nscf_update`
BANDS_NSCF_MODES = ('all-at-once', 'full-step')
#: the NSCF is not run above this number of bands, unless `workflow_settings['max_nscf_bands']` says otherwise
DEFAULT_MAX_NSCF_BANDS = 2000

def check_bands_settings(settings):
    """The error message if `bands_nscf_update` or `max_nscf_bands` of the `workflow_settings` are not valid, else None."""
    mode = settings.get('bands_nscf_update', 'all-at-once')
    is_integer = isinstance(mode, int) and not isinstance(mode, bool)
    if not (mode in BANDS_NSCF_MODES or (is_integer and mode > 0)):
        return (f"bands_nscf_update must be one of {BANDS_NSCF_MODES} or a positive integer (a cap on the NSCF bands), "
                f"not {mode!r}.")
    limit = settings.get('max_nscf_bands', DEFAULT_MAX_NSCF_BANDS)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        return f'max_nscf_bands must be a positive integer, not {limit!r}.'
    return None

def _get_convergence_thresholds(meta_parameters):
    """Convergence thresholds (in %) of the protocol.

    The old (misspelled) `convergence_thesholds` key can only come from the overrides of the user: it is honoured.
    """
    thresholds = dict(meta_parameters.get('convergence_thresholds') or {})
    for key, value in (meta_parameters.get('convergence_thesholds') or {}).items():
        thresholds['FFTGvecs' if key == 'FFTgvecs' else key] = value
    return {name: thresholds[name] for name in ('FFTGvecs', 'kpoints', 'Nb_Gcut')}

def _parameter_names(parameters):
    """All the variable names handled by a convergence parameter, mirrors included."""
    names = set(parameters.var)
    mirrors = parameters.mirror_values
    if isinstance(mirrors, dict):
        for mirrored in mirrors.values():
            names.update(mirrored)
    else:
        names.update(mirrors)
    return names

def apply_fit_functions(parameters_list, fit_functions):
    """Set the fit function of the convergence parameters.

    `fit_functions` maps a variable name of a parameter (e.g. 'FFTGvecs', 'BndsRnXp', 'kpoint_density') to the name of
    the fit function, or to a dictionary {'fit_function': name, 'fit_options': {...}} (see fit_models.py).
    The parameters are validated again, so an unknown function fails here.
    """
    parameters_list = list(parameters_list)
    for variable, setting in (fit_functions or {}).items():
        setting = {'fit_function': setting} if isinstance(setting, str) else dict(setting)
        for index, parameters in enumerate(parameters_list):
            if variable in _parameter_names(parameters):
                parameters_list[index] = type(parameters)(**{**parameters.model_dump(), **setting})
                break
        else:
            raise ValueError(f'No convergence parameter has the variable {variable!r}: '
                             f'{[sorted(_parameter_names(p)) for p in parameters_list]}')
    return parameters_list

def select_convergence_parameters(available, requested):
    """Select, in the requested order, the convergence parameters of `available`.

    Each entry of `requested` is a variable name or a list of names (coupled
    parameters, mirrored ones included, e.g. ['BndsRnXp','GbndRnge','NGsBlkXp']).
    """
    selected = []
    for entry in requested:
        names = {entry} if isinstance(entry, str) else set(entry)
        for parameters in available:
            if set(parameters.var) <= names <= _parameter_names(parameters):
                if parameters not in selected:
                    selected.append(parameters)
                break
        else:
            raise ValueError(
                f'Unknown convergence parameter {entry!r}. Available: '
                f'{[sorted(_parameter_names(p)) for p in available]}'
            )
    return selected

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
            validator=cls._validate_parameters_space,
            help = 'variables to converge, range, steps, and max iterations'
        )
        spec.input(
            "workflow_settings", 
            valid_type=orm.Dict, 
            required=True,
            validator=cls._validate_workflow_settings,
            help = 'settings for the workflow: type, quantities to be examinated, bands_nscf_update, max_nscf_bands...'
        )
        spec.input(
            "parallelism_instructions", 
            valid_type=orm.Dict, 
            required=False,
            help = 'DEPRECATED, use `resource_rules`. Indications for the parallelism to be used wrt values of the parameters.'
        )
        spec.input(
            "resource_rules",
            valid_type=orm.List,
            required=False,
            validator=cls._validate_resource_rules,
            help = 'resources (nodes, mpi, threads, walltime, queue, parallelism) to be used for given values of the parameters: '
                   'a list of {"when": {...}, "for": [steps], "use": {...}}. See aiida_yambo.utils.resource_rules.'
        )
        spec.input(
            "group_label", 
            valid_type=orm.Str, 
            required=False, 
            help = 'group of calculations already done for this system.'
        )

##################################### OUTLINE ####################################

        spec.outline(
            cls.setup,
            while_(cls.has_to_continue)(
                cls.update_next_step,
                cls.prepare_grid,
                if_(cls.p2y_needed)(
                    cls.do_p2y,
                ),
                cls.prepare_calculations,
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
        spec.exit_code(304, 'TOO_MANY_BANDS',
                             message='The NSCF would need more bands than allowed (`bands_nscf_update` as an integer, or `max_nscf_bands`).')
        spec.exit_code(400, 'CONVERGENCE_NOT_REACHED',
                             message='The workchain failed to reach convergence.')

    @staticmethod
    def _validate_workflow_settings(value, _):
        return check_bands_settings(value.get_dict())

    @staticmethod
    def _validate_resource_rules(value, _):
        try:
            validate_resource_rules(value.get_list())
        except ValueError as exc:
            return str(exc)

    @staticmethod
    def _validate_parameters_space(value, _):
        """The parameters must be valid, and a scan (an explicit list of sets) cannot be mixed with convergences."""
        try:
            space = load_parameter_space(value.get_list())
        except Exception as exc: # pydantic validation errors and wrong shapes
            return f'invalid parameters space: {exc}'
        if any(isinstance(parameters, ScanParam) for parameters in space) and len(space) > 1:
            return 'a ScanParam must be the only entry of the parameters space.'
        return None

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
        calc_type:Literal['qp', 'bse', 'bse@qp']='qp',
        workflow_settings:dict=None,
        convergence_parameters:list=None,
        scan_points:list=None,
        fit_functions:dict=None,
        overrides:dict=None,
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
        
        By default, we will do convergence on: 'FFTGvecs','kpoint_density' and then the coupled
        'BndsRnXp','GbndRnge','NGsBlkXp' (k-points are skipped for molecules)
        using the automated workflow as described in Bonacci et al. npj Comput Mater 9, 74 (2023).

        `convergence_parameters` selects which of them to converge, and in which order, e.g.
        ['FFTGvecs', ['BndsRnXp','GbndRnge','NGsBlkXp']]: coupled parameters go in a nested list.

        `fit_functions` sets the function fitted to the results of a parameter, by name (see `fit_models.py`):
        {'FFTGvecs': 'exponential', 'BndsRnXp': {'fit_function': 'power_law', 'fit_options': {'alphas': [1, 2]}}}.

        `scan_points` is an alternative: a list of dictionaries {variable: value}, one for each calculation to be
        run, e.g. [{'FFTGvecs': 10, 'BndsRnXp': 100}, {'FFTGvecs': 15, 'BndsRnXp': 200}]. No convergence is
        evaluated: the workflow just runs them (see `ScanParam`) and collects the results.
        """
        if scan_points is not None and convergence_parameters is not None:
            raise ValueError('`scan_points` and `convergence_parameters` cannot be used together.')

        # we never modify the dictionary of the caller
        overrides = copy.deepcopy(overrides) if overrides else {}

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
        nested_inputs = YamboInputManager.get_nested_inputs(builder,what='YamboConvergence', mode='builder')
        input_manager = YamboInputManager(**nested_inputs)

        if structure is None:
            structure = input_manager.scf.structure # e.g. when starting from a parent folder
        if structure is None:
            raise ValueError('the `structure` is required to build the convergence parameters space.')
                
        # the settings given by the user update the ones of the protocol, which are the defaults.
        settings = dict(inputs.get('workflow_settings') or {})
        settings.update(workflow_settings or {})
        if calc_type=='bse' and 'what' not in (workflow_settings or {}):
            settings['what'] = ['lowest_exciton','brightest_exciton']
        input_manager.workflow_settings = settings

        if scan_points is not None:
            input_manager.parameters_space = [ScanParam.from_dicts(scan_points)]
            return input_manager.populate_builder(builder, what='YamboConvergence')

        ################ if K-points density ################
        # kpoint_density semantics = AiiDA distance (1/Å) passed straight to
        # set_kpoints_mesh_from_density; refinement means SMALLER distances, so
        # the grid runs from start (coarse, largest distance) down to stop
        # (fine, smallest). The yaml still holds increasing "pseudo-densities";
        # invert them to physical distances here.
        kd = meta_parameters['kpoint_density']
        k_steps = 4  # steps of the KpointMeshConvParam grid below
        k_pseudo_step = (kd['stop'] - kd['start']) / (k_steps - 1)
        k_start = 1.0 / kd['start']                   # coarse distance (largest)
        k_stop = 1.0 / kd['stop']                     # fine distance (smallest)
        k_max = float(max(k_start, k_stop))           # coarse cap
        k_min = min(1.0 / kd['max'], k_stop)          # fine cap: the most expensive mesh allowed
        k_delta = -(1.0 / kd['stop'] - 1.0 / (kd['stop'] + k_pseudo_step))

        # Getting information on the nelectrons, ecutwfc and volume
        ecutwfc = input_manager.scf.parameters['SYSTEM']['ecutwfc']
        volume = structure.get_ase().get_volume()
        
        ################ FFTGVecs ################
        thresholds = _get_convergence_thresholds(meta_parameters)
        FFT_start=int(meta_parameters['FFTGvecs']['start_ratio']*ecutwfc)
        FFT_stop=int(meta_parameters['FFTGvecs']['stop_ratio']*ecutwfc)
        FFT_max=int(meta_parameters['FFTGvecs']['max_ratio']*ecutwfc)
        FFT_delta=int(meta_parameters['FFTGvecs']['delta_ratio']*ecutwfc)
        
        ################ Nb: bands for the X and Sc ################
        # here the logic considers a ratio which depends on the dimensionality, or better on the volume:
        # if the volume is lower then 
        
        # The bands of the protocol are multiplied by log10 of the cell volume (Angstrom^3): the larger the cell, the
        # more bands for the same energy range. It is a heuristic, not a fit: to use other numbers, override
        # `meta_parameters['bands']` (start, stop, delta, max), which are multiplied by it.
        volume_factor = np.log10(volume)
        bands = meta_parameters['bands'] if 'bands' in meta_parameters else meta_parameters['Nb'] # `Nb` is the old name
        b_start=bands['start']*volume_factor
        b_stop=bands['stop']*volume_factor
        b_delta=bands['delta']*volume_factor
        b_max=bands['max']*volume_factor
        
        ################ G_cut for X, Sc, BSE kernel ################
        G_start=meta_parameters['G_vectors']['start']
        G_stop=meta_parameters['G_vectors']['stop']
        G_max=meta_parameters['G_vectors']['max']
        G_delta=meta_parameters['G_vectors']['delta']

        ######################### PARAMETERS SPACE CREATION #########################
        all_parameters = [
            ConvParam('FFTGvecs',                start=FFT_start,stop=FFT_stop,delta=FFT_delta,max=FFT_max,
                steps=4,max_iterations=4,conv_thr=thresholds['FFTGvecs'],
                conv_thr_units='%',convergence_algorithm='new_algorithm_1D', units='Ry'),
            KpointMeshConvParam(var='kpoint_density', start=k_start, stop=k_stop, delta=k_delta, max=k_max, min=k_min,
                steps=k_steps,max_iterations=4,conv_thr=thresholds['kpoints'],
                conv_thr_units='%',convergence_algorithm='new_algorithm_1D', units='1/A'),
            CoupledConvParam(var=['BndsRnXp','NGsBlkXp'], start=[b_start,G_start],
                stop=[b_stop,G_stop],delta=[b_delta,G_delta],max=[b_max,G_max],
                steps=6,max_iterations=8,conv_thr=thresholds['Nb_Gcut'],
                conv_thr_units='%',convergence_algorithm='new_algorithm_2D', units=['', 'Ry'],
                mirror_values={'BndsRnXp':['GbndRnge']},),
        ]

        if convergence_parameters is None:
            # default selection and order: FFTGvecs, kpoint_density, (BndsRnXp, NGsBlkXp)
            parameters_space = all_parameters
        else:
            parameters_space = select_convergence_parameters(all_parameters, convergence_parameters)

        if protocol == 'molecule' or structure.pbc.count(True)==0:
            #no k-points, so we skip them in the parameters_space.
            parameters_space = [p for p in parameters_space if not isinstance(p, KpointMeshConvParam)]

        input_manager.parameters_space = apply_fit_functions(parameters_space, fit_functions)

        new_builder = input_manager.populate_builder(builder, what='YamboConvergence')
        
        return new_builder
            
    def from_yamboworkflow(self,):
        raise NotImplementedError
    
    def from_yambopy(self,):
        raise NotImplementedError
        
    def setup(self):
        """Initialize the workflow"""
        
        # NOTE: mode='node' on self.node rebuilds the full builder via
        # get_builder_restart(), so also metadata/options (resources, walltime...)
        # are restored; `self.inputs` would carry an empty `metadata` namespace.
        input_manager = YamboInputManager.get_nested_inputs(
            self.node, what='YamboConvergence', mode='node'
        )
        self.ctx.input_manager = YamboInputManager(**input_manager)

        self.ctx.limit_reached = False # this checks if we cannot increase the parameter more than the `max` value.
        self.ctx.convergence_summary = {}   #converged parameters
        #self.ctx.calc_inputs = self.exposed_inputs(YamboWorkflow, 'ywfl')        
        self.ctx.hint = {}
        self.ctx.global_step = 0
        self.ctx.wfl_pk = []
        self.ctx.bands_mode = self.ctx.input_manager.workflow_settings.get('bands_nscf_update', 'all-at-once')
        self.ctx.none_encountered = False

        self.ctx.calc_type = 'bse' if len(self.ctx.input_manager.bse.parameters) > 0 else 'qp'
        
        # we add the quantity to be converged to the additional_parsing list,
        # without duplicating entries already provided through the workflow inputs.
        additional_parsing = list(self.ctx.input_manager.additional_parsing or [])
        for quantity in self.ctx.input_manager.workflow_settings['what']:
            if quantity not in additional_parsing:
                additional_parsing.append(quantity)
        self.ctx.input_manager.additional_parsing = additional_parsing

        
        # here we load or create the group where to put all the calculations.
        if hasattr(self.inputs, "group_label"):
            group_label = self.ctx.input_manager.group_label
        else:
            group_label = "convergence_tests_{}".format(self.ctx.input_manager.scf.structure.get_formula())
        try:
            # by label: a label which looks like a pk or the start of a uuid is not taken for one
            self.ctx.group = orm.load_group(label=group_label)
            self.report('using the existing group: {} (pk {}, {} nodes)'.format(
                group_label, self.ctx.group.pk, self.ctx.group.count()))
        except NotExistent:
            self.ctx.group = Group(label=group_label).store()
            self.report('the group {} did not exist: created it (pk {})'.format(group_label, self.ctx.group.pk))
        
        # the options of the steps at the start: every point goes back to them before the resource rules are applied.
        if self.ctx.input_manager.resource_rules:
            self.ctx.resources_base = snapshot_base(self.ctx.input_manager)
        if self.ctx.input_manager.parallelism_instructions:
            self.report('WARNING: `parallelism_instructions` is deprecated, use `resource_rules`.')

        self.report(
            "Workflow initilization step completed, the parameters will be: {}."
            .format(
                [params.var for params in self.ctx.input_manager.parameters_space]
            )
        )
        
        # this calc_manager can be removed? we substitue it with self.ctx.current_convergence
        # NOTE: parameters are consumed in INPUT order (FIFO): the first entry
        # of `parameters_space` is the first one to be converged.
        self.ctx.current_convergence = ConvergenceEvaluator(
            parameters=self.ctx.input_manager.parameters_space.pop(0),
            what=self.ctx.input_manager.workflow_settings['what']
        )

        # initialize the dataframe holding all the calculation records:
        # Use the UNION of all convergence variables (current + remaining) so that
        # every row has a column for every var (FFTGvecs, BndsRnXp, NGsBlkXp, ...).
        # Previously only the current var was used, causing NaN for the fixed
        # heavy values (e.g. FFTGvecs NaN on Bnds rows). Mirrors (GbndRnge) are
        # handled via the same value as BndsRnXp.
        all_vars = set(self.ctx.current_convergence.parameters.var)
        for p in self.ctx.input_manager.parameters_space:
            all_vars.update(p.var)
            # include mirrors so they also get a column if present
            mv = getattr(p, 'mirror_values', {})
            if isinstance(mv, dict):
                for v in mv.values():
                    all_vars.update(v if isinstance(v, list) else [v])
            elif isinstance(mv, list):
                all_vars.update(mv)
        # also include mirrors of current
        mv0 = getattr(self.ctx.current_convergence.parameters, 'mirror_values', {})
        if isinstance(mv0, dict):
            for v in mv0.values():
                all_vars.update(v if isinstance(v, list) else [v])
        elif isinstance(mv0, list):
            all_vars.update(mv0)
        # keep a stable order: FFTGvecs first, then as in input order
        ordered_vars = []
        for p in [self.ctx.current_convergence.parameters] + self.ctx.input_manager.parameters_space:
            for v in p.var:
                if v not in ordered_vars:
                    ordered_vars.append(v)
                mv = getattr(p, 'mirror_values', {})
                if isinstance(mv, dict) and v in mv:
                    for m in (mv[v] if isinstance(mv[v], list) else [mv[v]]):
                        if m not in ordered_vars:
                            ordered_vars.append(m)
                elif isinstance(mv, list):
                    for m in mv:
                        if m not in ordered_vars:
                            ordered_vars.append(m)
        # fallback to sorted if ordering fails
        if not ordered_vars:
            ordered_vars = sorted(all_vars)
        df_columns = CONVERGENCE_DF_COLUMNS + ordered_vars + self.ctx.current_convergence.what_list
        self.ctx.df = pd.DataFrame(columns=df_columns)
        # stash for perform_next_step to know all vars
        self.ctx._all_var_columns = ordered_vars

        self.report(
            "Workflow type: {}; looking for convergence of {}"
            .format(
                self.ctx.input_manager.workflow_settings['type'], 
                self.ctx.current_convergence.what,
            )
        )

        self.ctx.final_result = {}
        self.ctx.current_batch_pks = []

    def has_to_continue(self):
        
        """This function checks the status of the last calculation and determines what happens next, 

        """
        
        # termination check first: a decided convergence with nothing left to
        # converge must exit the loop even if the batch counter would still
        # report "cannot_stop" (it is reset to 0 at the end of data_analysis).
        # NOTE: this must be checked BEFORE cannot_stop, otherwise a converged
        # single-parameter workflow takes one extra lap and crashes in
        # update_next_step popping from an empty parameters_space.
        if self.ctx.current_convergence.success and len(self.ctx.input_manager.parameters_space) == 0:
            self.report('Workflow finished')
            return False

        if self.ctx.none_encountered:
            self.report('Some calculation failed, so we stop the workflow.')
            return False

        if self.ctx.current_convergence.cannot_stop:
            self.report('We should finish the set of calculations before the analysis.')
            return True
        
        # NOTE: this is a predicate: it must not modify the state (it can be evaluated again when
        # the workchain is reloaded from a checkpoint). The iteration counter is advanced in update_next_step.

        #failed cases
        if not self.ctx.current_convergence.success and self.ctx.current_convergence.iterations_exceeded:
            self.report('Workflow failed due to max restarts exceeded for variable {}'.format(self.ctx.current_convergence.parameters.var))
            return False
        if self.ctx.current_convergence.edges_exceeded:
            self.report('space not large enough to complete convergence')
            return False

        if not self.ctx.current_convergence.success:
            # Single-point verification should not consume an iteration (it's a check, not a new grid)
            if self.ctx.current_convergence.check_single_point:
                self.report('Verifying predicted point for {}'.format(self.ctx.current_convergence.parameters.var))
            else:
                self.report('Still iteration on {}'.format(self.ctx.current_convergence.parameters.var))
            return True

        # successful case, with other parameters to be converged:
        # do not clear hint here — update_next_step needs it to persist the
        # just-converged values (FFTGvecs) and to apply them in heavy mode
        return True
        
    def update_next_step(self):
        
        if self.ctx.current_convergence.success:
            remaining = self.ctx.input_manager.parameters_space
            
            if len(remaining) == 0:
                # nothing left to converge: do not swap the evaluator; the loop
                # will exit at the next has_to_continue check. NOTE: this also
                # keeps `verdi process play` recovery of old checkpoints alive,
                # as plumpy resumes ONLY this step, without re-running
                # has_to_continue (which would otherwise be the guard).
                self.report('No more parameters to be converged.')
                return
            
            if self.ctx.input_manager.workflow_settings.get('type') == 'cheap':
                self.report('Mode is "cheap", so we reset the other parameters to be the initial ones.')
                input_manager = YamboInputManager.get_nested_inputs(
                    self.node, what='YamboConvergence', mode='node'
                )
                current_parent_folder = self.ctx.input_manager.parent_folder
                self.ctx.input_manager = YamboInputManager(**input_manager)
                # keep the parameters not yet converged, otherwise the reset would
                # re-converge the very first variable.
                self.ctx.input_manager.parameters_space = remaining
                # the reset must not discard the parent folder reached so far
                # (e.g. the pre-calculation remote folder): restore it.
                if current_parent_folder is not None:
                    self.ctx.input_manager.set_parent_folder(current_parent_folder)
                # re-add the converged quantity, without duplicating inputs entries
                additional_parsing = list(self.ctx.input_manager.additional_parsing or [])
                for quantity in self.ctx.input_manager.workflow_settings['what']:
                    if quantity not in additional_parsing:
                        additional_parsing.append(quantity)
                self.ctx.input_manager.additional_parsing = additional_parsing
                # accumulate per-parameter (flat + namespaced) instead of shallow overwrite
                # keep FFTGvecs even after BndsRnXp converges; use distinct
                # group keys to avoid colliding with flat var names (FFTGvecs)
                for var in self.ctx.current_convergence.parameters.var:
                    if var in self.ctx.hint:
                        self.ctx.convergence_summary[var] = self.ctx.hint[var]
                        self.ctx.convergence_summary.setdefault('hints', {})[var] = self.ctx.hint[var]
                        # mirror e.g. BndsRnXp -> GbndRnge
                        mirror_cfg = getattr(self.ctx.current_convergence.parameters, 'mirror_values', {})
                        if isinstance(mirror_cfg, dict) and var in mirror_cfg:
                            for m in mirror_cfg[var]:
                                self.ctx.convergence_summary[m] = self.ctx.hint[var]
                                self.ctx.convergence_summary['hints'][m] = self.ctx.hint[var]
                group_key = '_'.join(self.ctx.current_convergence.parameters.var)
                summary_key = group_key + '_summary'
                self.ctx.convergence_summary[summary_key] = self.ctx.current_convergence.get_summary()
                self.ctx.convergence_summary[group_key + '_hints'] = dict(self.ctx.hint)
            else:
                self.report('Mode is "heavy", so we keep the previous parameters to be the converged ones, if any.')
                # accumulate per-parameter (flat + namespaced) instead of shallow overwrite
                for var in self.ctx.current_convergence.parameters.var:
                    if var in self.ctx.hint:
                        self.ctx.convergence_summary[var] = self.ctx.hint[var]
                        self.ctx.convergence_summary.setdefault('hints', {})[var] = self.ctx.hint[var]
                        mirror_cfg = getattr(self.ctx.current_convergence.parameters, 'mirror_values', {})
                        if isinstance(mirror_cfg, dict) and var in mirror_cfg:
                            for m in mirror_cfg[var]:
                                self.ctx.convergence_summary[m] = self.ctx.hint[var]
                                self.ctx.convergence_summary['hints'][m] = self.ctx.hint[var]
                group_key = '_'.join(self.ctx.current_convergence.parameters.var)
                summary_key = group_key + '_summary'
                self.ctx.convergence_summary[summary_key] = self.ctx.current_convergence.get_summary()
                self.ctx.convergence_summary[group_key + '_hints'] = dict(self.ctx.hint)
                # Heavy: apply the converged values (from hint) to the live manager
                # so the next parameter's YamboWorkflows run with the converged
                # previous parameter, not the last grid point (e.g. FFTGvecs 10 not 30).
                prev_params = self.ctx.current_convergence.parameters
                for var in prev_params.var:
                    if var in self.ctx.hint:
                        val = self.ctx.hint[var]
                        # normalize numpy scalars
                        if isinstance(val, np.generic):
                            val = val.item()
                        # units from the just-converged parameter definition
                        try:
                            if isinstance(prev_params.units, str):
                                unit = prev_params.units
                            else:
                                unit = prev_params.units[prev_params.var.index(var)]
                        except (ValueError, IndexError, TypeError):
                            unit = 'Ry' if var.lower() == 'fftgvecs' else ''
                        # mirror handling (e.g. BndsRnXp -> GbndRnge)
                        mirror_cfg = getattr(prev_params, 'mirror_values', None)
                        targets = [var]
                        if isinstance(mirror_cfg, dict) and var in mirror_cfg:
                            targets.extend(mirror_cfg[var])
                        elif isinstance(mirror_cfg, list):
                            targets.extend(mirror_cfg)
                        for target in targets:
                            # choose manager(s) that actually have yambo inputs
                            for mgr_name in ('qp', 'bse'):
                                mgr = getattr(self.ctx.input_manager, mgr_name, None)
                                if mgr is None or len(mgr.parameters) == 0:
                                    continue
                                if 'bnd' in target.lower():
                                    mgr.set_variable(target, [[1, int(val)], ''])
                                else:
                                    # Gcut / FFTGvecs etc: [value, unit]
                                    mgr.set_variable(target, [val, unit])
                                self.report(f"Heavy: setting {mgr_name}.variables['{target}'] = {mgr.parameters['variables'].get(target)} (converged {var}={val})")
            
            self.ctx.current_convergence = ConvergenceEvaluator(
                # NOTE: must be keyword: ConvergenceEvaluator is a pydantic
                # BaseModel and does not accept positional arguments.
                parameters=remaining.pop(0), #FIFO: keep the input order of parameters_space
                what=self.ctx.input_manager.workflow_settings['what']
            )
            self.ctx.hint = {}
        elif not self.ctx.current_convergence.cannot_stop and not self.ctx.current_convergence.check_single_point:
            # the set of calculations is complete and the convergence not reached: a new grid (iteration) starts.
            # A single-point verification does not consume an iteration (it's a check, not a new grid).
            self.ctx.current_convergence.advance_iteration()
        
        self.report(f'Convergence on: {self.ctx.current_convergence.parameters.var}')
        if len(self.ctx.current_convergence.parameters.mirror_values) >0: self.report(f'With them, we mirror the following parameters {self.ctx.current_convergence.parameters.mirror_values}')
        
    def prepare_grid(self):
        """Build the parameter grid for the upcoming submission.

        Split out from perform_next_step so that p2y_needed/do_p2y can see the
        actual max bands (e.g. shifted window) before deciding whether the NSCF
        needs more bands. Mirrors the first half of the old perform_next_step.
        """
        evaluator = self.ctx.current_convergence

        # Shift window if predictor requested a new grid (first retry is iter 1)
        if evaluator.new_grid:
            if evaluator.current_iteration >= 1:
                if isinstance(evaluator.parameters, CoupledConvParam):
                    evaluator.parameters = evaluator.parameters.with_shift(
                        start_shifts=evaluator.parameters.delta,
                        stop_shifts=evaluator.parameters.delta,
                    )
                elif isinstance(evaluator.parameters, KpointMeshConvParam):
                    evaluator.parameters = evaluator.parameters.with_shift(
                        start_shift=evaluator.parameters.delta,
                        stop_shift=evaluator.parameters.delta,
                    )
                else:
                    evaluator.parameters = evaluator.parameters.with_shift(
                        start_shift=evaluator.parameters.delta,
                        stop_shift=evaluator.parameters.delta,
                    )
                evaluator.concurrent_steps_count = 0
            # Consume the flag so we don't shift again on next point of same grid
            evaluator.new_grid = False

        # Build grid
        self.ctx.params_space = _build_params_space(evaluator.parameters)

        if self.ctx.params_space == ['space_exceeds_max']:
            self.report(
                f'Grid for {", ".join(evaluator.parameters.var)} contains only invalid '
                '(non-positive/out-of-bound) values; stopping convergence.'
            )
            self._emit_story_outputs()
            return self.exit_codes.SPACE_TOO_SMALL

        if evaluator.check_single_point:
            self.report(f'Now we verify the accuracy of the prediction by computing a single predicted point: {evaluator.inquired_point}')
            self.ctx.params_space = [evaluator.inquired_point]
            evaluator.concurrent_steps_count = evaluator.parameters.steps - 1

        # Anticipate max bands for p2y decision (ceiling over remaining + current)
        self.ctx.max_bands_needed = self._capped_nscf_bands(self._nscf_bands_for(evaluator))

        self.report(f'Prepared grid with {len(self.ctx.params_space)} points, max_bands_needed={self.ctx.max_bands_needed}')
        exit_code = self._check_nscf_bands(evaluator)
        if exit_code:
            self._emit_story_outputs()
            return exit_code

        # do we need a preliminary calculation? We decide here, so the `p2y_needed` predicate stays pure.
        self.ctx.run_p2y = self._evaluate_p2y_needed()

    def perform_next_step(self):
        """This function will submit the next step"""
        
        # 1. Grab our smart evaluator tracker from context
        evaluator = self.ctx.current_convergence

        calc = {}

        # Grid already prepared in prepare_grid (including new_grid shift and
        # single-point handling). Reuse it here; do not rebuild.
        if not hasattr(self.ctx, 'params_space') or not self.ctx.params_space:
            # Fallback for direct calls / old checkpoints
            self.ctx.params_space = _build_params_space(evaluator.parameters)
            if evaluator.check_single_point:
                self.ctx.params_space = [evaluator.inquired_point]
                evaluator.concurrent_steps_count = evaluator.parameters.steps - 1

        # For 2D coupled grids the number of points is `points` (e.g. 6), which may differ
        # from `steps`; use the actual grid length for the completion check, except for
        # single-point verification which keeps the `steps`-based sentinel.
        grid_len = len(self.ctx.params_space)
        limit = evaluator.parameters.steps if evaluator.check_single_point else grid_len
        self.report(f"Running a max of {evaluator.parameters.max_concurrent_steps} concurrent steps of {limit}")
        for i in range(evaluator.parameters.max_concurrent_steps):
            
            if evaluator.concurrent_steps_count >= limit:
                break
            
            # Your updater stays the same, but passes explicit objects/grids cleanly
            self.ctx.input_manager, values, already_done, _ = updater(
                input_manager=self.ctx.input_manager,
                evaluator=evaluator,
                explicit_value=self.ctx.params_space, 
                workflow_dict=self.ctx.input_manager.parallelism_instructions,
                resource_rules=self.ctx.input_manager.resource_rules,
                resources_base=self.ctx.get('resources_base'),
                index=evaluator.concurrent_steps_count if evaluator.inquired_point is None else 0, # so if iteration = 1 and not finished, should start from evaluator.max_concurrent_steps+1
                group=self.ctx.group,
                calc_type=self.ctx.calc_type, 
            )
                                    
            self.report(f"New parameters are: {values}")
            
            if not already_done:
                label = f"iteration_{self.ctx.global_step}"
                calc_inputs = self.ctx.input_manager.generate_AiiDA_inputs(what='YamboWorkflow')
                metadata = dict(calc_inputs.get('metadata', {}) or {})
                metadata['call_link_label'] = label
                calc_inputs['metadata'] = metadata
                
                future = self.submit(YamboWorkflow, **calc_inputs)
                # register this workchain in the group: search_in_group scans the
                # `called` children of YamboConvergence nodes to find done calcs.
                self.ctx.group.add_nodes(self.node)
            else:
                self.report(f"Calculation already done: {already_done}")
                future = load_node(already_done)

            if not already_done:
                # only pending processes can be awaited in a ToContext; nodes
                # loaded from `already_done` are already finished.
                calc[str(i+1)] = future
            self.ctx.wfl_pk.insert(0, future.pk)  # Cleaner way to prepend to list
            self.ctx.group.add_nodes(future)
            self.ctx.current_batch_pks.append(future.pk)

            # Avoid duplicate pk in story (single-point verification reusing a grid point, e.g. 30,6)
            # The verification point is already in the 6-point grid; adding it again would create
            # a duplicate row (pk 19116 appears twice). Skip row creation but keep batch tracking.
            if already_done and future.pk in self.ctx.df['pk'].values:
                self.report(f"Skipping duplicate story row for already_computed pk {future.pk} (already in DF)")
                evaluator.concurrent_steps_count += 1
                self.ctx.global_step += 1
                if len(self.ctx.params_space) == 0:
                    break
                continue
            
            # record the row: active var(s) from `values`, fixed vars from the live
            # input_manager (so heavy's converged FFTGvecs appears on Bnds rows
            # instead of NaN), plus the new `converged` flag (False until
            # data_analysis marks the whole group as converged).
            # Use the union built in setup (fallback to active var if old checkpoint).
            all_vars = getattr(self.ctx, '_all_var_columns', evaluator.parameters.var)
            row = {}
            for var in all_vars:
                if var in values:
                    v = values[var]
                    # keep kpoint_mesh as a proper 3-vector list so the analysis
                    # grid/result stay homogeneous (a stray scalar/None would
                    # make the predictor's np.asarray raise).
                    row[var] = list(v) if var == 'kpoint_mesh' and isinstance(v, (list, tuple)) else v
                else:
                    # fixed value currently in use for this calculation
                    fixed_val = None
                    for mgr_name in ('qp', 'bse'):
                        mgr = getattr(self.ctx.input_manager, mgr_name, None)
                        if mgr is None or not getattr(mgr, 'parameters', None):
                            continue
                        vars_dict = mgr.parameters.get('variables', {})
                        if var in vars_dict:
                            raw = vars_dict[var]
                            try:
                                if isinstance(raw, list) and len(raw) == 2 and isinstance(raw[0], list):
                                    # [[1,N],''] -> N
                                    fixed_val = raw[0][1] if len(raw[0]) > 1 else raw[0][0]
                                elif isinstance(raw, list):
                                    fixed_val = raw[0]
                                else:
                                    fixed_val = raw
                            except (IndexError, TypeError):
                                fixed_val = None
                            break
                    row[var] = fixed_val
            row.update({
                'global_step': self.ctx.global_step,
                'pk': future.pk,
                'uuid': future.uuid,
                'parameters_studied': ', '.join(evaluator.parameters.var),
                'useful': False,
                'failed': False,
                'converged': False,
            })
            row.update({q: False for q in evaluator.what_list})
            self.ctx.df = pd.concat([self.ctx.df, pd.DataFrame([row])], ignore_index=True)
            
            evaluator.concurrent_steps_count += 1
            self.ctx.global_step += 1
            
            if len(self.ctx.params_space) == 0:
                break

        self.ctx.current_convergence=evaluator
        
        # if every calculation was already done, there is nothing to wait for.
        if calc:
            return ToContext(calc)
        return

    def data_analysis(self):
        """Analyze completed calculations using the ConvergenceEvaluator object.
        
        The current batch is sliced from the cumulative dataframe, the `what`
        results (and the failed/useful flags) are fetched from the calculation
        nodes and written back into the dataframe, and the evaluator is then fed
        with the batch to decide the next step.
        """
        
        evaluator = self.ctx.current_convergence

        if evaluator.cannot_stop:
            self.report("Continuing the submission to complete the step.")
            return
                    
        self.report('Starting unified data analysis step...')

        # 1. Slice the current batch (the rows submitted since the last analysis)
        #    from the cumulative dataframe and fetch the quantities to converge
        #    plus the failed/useful flags from the calculation nodes.
        df_batch = self.ctx.df[self.ctx.df['pk'].isin(self.ctx.current_batch_pks)].copy()

        # Restrict to the current parameter group only: the same already-computed
        # node can be reused across groups (e.g. pk 20077 for Bnds/NGs and again
        # for a kpoint_mesh grid point), which would otherwise drag in stale rows
        # whose other parameter columns are None (kpoint_mesh col shows None).
        group_label = ', '.join(evaluator.parameters.var)
        if 'parameters_studied' in df_batch.columns:
            df_batch = df_batch[df_batch['parameters_studied'] == group_label].copy()

        # Front-line guarantee: every active parameter of this group must be non-None
        # in the batch (defense-in-depth against any cross-group leftover).
        for _var in evaluator.parameters.var:
            if _var in df_batch.columns:
                df_batch = df_batch[df_batch[_var].notna()].copy()

        if df_batch.empty:
            self.report("Error: No dataframe rows found to analyze.")
            self._emit_story_outputs()
            return self.exit_codes.UNDEFINED_STATE

        self.report(
            f"df_batch rows: {len(df_batch)} | "
            f"{evaluator.parameters.var[0]} col: {df_batch.get(evaluator.parameters.var[0], pd.Series(dtype=object)).tolist()[:8]}"
        )
        self.report(
            f"df_batch failed: {df_batch.get('failed', pd.Series(dtype=bool)).tolist()} | "
            f"useful: {df_batch.get('useful', pd.Series(dtype=bool)).tolist()}"
        )

        take_quantities(df_batch, evaluator.what_list)

        # 2. Write the updated columns back into the cumulative dataframe and
        #    consume the batch tracker.
        updated_cols = evaluator.what_list + ['failed', 'useful']
        self.ctx.df.loc[df_batch.index, updated_cols] = df_batch[updated_cols].values
        self.ctx.current_batch_pks = []

        # A scan has nothing to evaluate: all the points were computed, we collect the results.
        if isinstance(evaluator.parameters, ScanParam):
            return self._finish_scan(evaluator, df_batch)

        # 3. Let the Evaluator object process the current batch directly from the dataframe
        # For single-point verification, do a direct gap comparison (no refit on 1-2 points)
        # to avoid curve_fit with insufficient data. The predicted gap is in
        # evaluator.hints['gap_'] (or old_hints), actual gap is in df_batch.
        if getattr(evaluator, 'check_single_point', False):
            try:
                what = evaluator.what_list[0] if evaluator.what_list else 'gap_'
                # predicted gap from previous grid's fit (stored in hints)
                pred_gap = None
                # hints may contain gap_ from previous CONTINUE
                if what in evaluator.hints:
                    pred_gap = evaluator.hints[what]
                elif what in self.ctx.hint:
                    pred_gap = self.ctx.hint[what]
                # actual gap at verification point (df_batch has 1 row)
                actual_gap = None
                if what in df_batch.columns:
                    # df_batch already has gap_ filled via take_quantities
                    # take the first (and only) row's gap
                    try:
                        actual_gap = float(df_batch.iloc[0][what])
                    except (TypeError, ValueError):
                        actual_gap = df_batch.iloc[0][what]
                # threshold
                conv_thr = getattr(evaluator.parameters, 'conv_thr', 0.05)
                conv_units = getattr(evaluator.parameters, 'conv_thr_units', 'eV')
                # determine threshold in eV
                if conv_units == '%':
                    # need reference value for %: use E_ref or predicted gap
                    ref = evaluator.hints.get('E_ref', pred_gap)
                    try:
                        thr = abs(float(ref)) * float(conv_thr) / 100.0 if ref is not None else float(conv_thr)
                    except (TypeError, ValueError):
                        thr = float(conv_thr)
                else:
                    thr = float(conv_thr)
                # direct verification
                if pred_gap is not None and actual_gap is not None and not (isinstance(actual_gap, bool) or isinstance(pred_gap, bool)):
                    try:
                        # handle numpy types and possible False placeholders
                        if isinstance(actual_gap, bool) or isinstance(pred_gap, bool):
                            raise ValueError("placeholder gap")
                        disc = abs(float(actual_gap) - float(pred_gap))
                        self.report(f"Single-point verification: predicted {pred_gap:.4f} vs actual {actual_gap:.4f}, disc {disc:.4f} thr {thr:.4f}")
                        if disc <= thr:
                            # converged
                            decision = {'status': 'CONVERGED', 'hints': dict(evaluator.hints), 'new_grid': False}
                            # ensure hints contain the verified point's gap and already_computed flag
                            decision['hints'][what] = actual_gap
                            decision['hints']['already_computed'] = bool(df_batch.iloc[0]['pk'] in self.ctx.df['pk'].values) if len(self.ctx.df) > 0 else False
                        else:
                            # need new grid
                            decision = {'status': 'CONTINUE', 'hints': {'new_grid': True}, 'new_grid': True}
                            # preserve E_ref etc. for next iteration
                            decision['hints'].update({k: v for k, v in evaluator.hints.items() if k in ['E_ref', 'gap_']})
                    except Exception as e:
                        self.report(f"Single-point direct check failed ({e}), falling back to full evaluation")
                        # fallback to full history evaluation with enough points
                        n_grid = getattr(evaluator.parameters, 'points', None) or evaluator.parameters.steps
                        useful_mask = self.ctx.df['useful'].astype(bool) if 'useful' in self.ctx.df.columns else pd.Series([True]*len(self.ctx.df), index=self.ctx.df.index)
                        param_mask = self.ctx.df['parameters_studied'] == ', '.join(evaluator.parameters.var)
                        useful_df = self.ctx.df[useful_mask & param_mask]
                        if len(useful_df) >= n_grid + 1:
                            df_for_eval = useful_df.tail(n_grid + 1).copy()
                        else:
                            df_for_eval = pd.concat([useful_df.tail(n_grid), df_batch], ignore_index=False).tail(n_grid + 1)
                        decision = evaluator.evaluate_dataframe(df_for_eval)
                else:
                    # fallback to full evaluation with history
                    n_grid = getattr(evaluator.parameters, 'points', None) or evaluator.parameters.steps
                    useful_mask = self.ctx.df['useful'].astype(bool) if 'useful' in self.ctx.df.columns else pd.Series([True]*len(self.ctx.df), index=self.ctx.df.index)
                    param_mask = self.ctx.df['parameters_studied'] == ', '.join(evaluator.parameters.var)
                    useful_df = self.ctx.df[useful_mask & param_mask]
                    if len(useful_df) >= n_grid + 1:
                        df_for_eval = useful_df.tail(n_grid + 1).copy()
                    else:
                        df_for_eval = pd.concat([useful_df.tail(n_grid), df_batch], ignore_index=False).tail(n_grid + 1)
                    decision = evaluator.evaluate_dataframe(df_for_eval)
            except Exception as exc:
                self.report(f"Single-point direct check failed ({exc}), using batch only")
                decision = evaluator.evaluate_dataframe(df_batch)
        else:
            decision = evaluator.evaluate_dataframe(df_batch)

        # 4. Request next-step execution strategy from your Convergence Evaluator
        self.ctx.hint = decision.get('hints', {})
        
        self.report(f"Evaluator decision: status={decision['status']} | Hints: {self.ctx.hint}")

        # 5. Handle state machine responses based on clean evaluator decisions
        if decision['status'] == 'CONVERGED':
            self.report(f"Success! Parameter group {evaluator.parameters.var} has converged.")
            evaluator.success = True
            
            # Accumulate per-parameter: keep FFTGvecs even after BndsRnXp converges
            # (shallow update would overwrite 'hints' and 'parameters')
            summary = evaluator.get_summary()
            # Inject uuid/pk into hints when already_computed and expose directly at top level
            # (user requested uuid directly exposed as key-value)
            if 'last_converged_uuid' in summary:
                # top-level direct exposure (always, for the converged point)
                self.ctx.convergence_summary['uuid'] = summary['last_converged_uuid']
                try:
                    self.ctx.convergence_summary['pk'] = load_node(summary['last_converged_uuid']).pk
                except NotExistent:
                    self.report(f"The converged calculation {summary['last_converged_uuid']} is not in the database.")
                if summary.get('hints', {}).get('already_computed'):
                    summary['hints']['uuid'] = summary['last_converged_uuid']
                    try:
                        summary['hints']['pk'] = load_node(summary['last_converged_uuid']).pk
                    except NotExistent:
                        pass # already reported above
                    self.ctx.hint = dict(summary['hints'])
            for var in evaluator.parameters.var:
                if var in summary.get('hints', {}):
                    self.ctx.convergence_summary[var] = summary['hints'][var]
                    self.ctx.convergence_summary.setdefault('hints', {})[var] = summary['hints'][var]
                    # mirror e.g. BndsRnXp -> GbndRnge
                    mirror_cfg = getattr(evaluator.parameters, 'mirror_values', {})
                    if isinstance(mirror_cfg, dict) and var in mirror_cfg:
                        for m in mirror_cfg[var]:
                            self.ctx.convergence_summary[m] = summary['hints'][var]
                            self.ctx.convergence_summary['hints'][m] = summary['hints'][var]
            # namespaced bucket for full envelope (use distinct suffix to avoid colliding with flat var names like FFTGvecs)
            group_key = '_'.join(evaluator.parameters.var)
            summary_key = group_key + '_summary'
            self.ctx.convergence_summary[summary_key] = summary
            self.ctx.convergence_summary[group_key + '_hints'] = dict(summary.get('hints', {}))
            # keep flat hints merged for backwards compat (now includes uuid/pk when already_computed)
            self.ctx.convergence_summary.setdefault('hints', {}).update(summary.get('hints', {}))
            # keep last success/uuid at top level as well
            self.ctx.convergence_summary['success'] = summary.get('success', True)
            if 'last_converged_uuid' in summary:
                self.ctx.convergence_summary['last_converged_uuid'] = summary['last_converged_uuid']
            # mark only the verified converged point(s) as converged (not the whole grid)
            # e.g. FFTGvecs 10 and BndsRnXp 30 / NGsBlkXp 6 → 2 rows total
            try:
                # inject uuid/pk into hints when already_computed (provenance)
                if summary.get('hints', {}).get('already_computed') and 'last_converged_uuid' in summary:
                    summary['hints']['uuid'] = summary['last_converged_uuid']
                    try:
                        summary['hints']['pk'] = load_node(summary['last_converged_uuid']).pk
                    except NotExistent:
                        pass # already reported above
                    # also reflect in the flat hints for report_wf
                    self.ctx.hint = dict(summary['hints'])
                # value-match ONLY within the evaluated batch (so only the
                # converged point, not the whole historical grid, is marked)
                # e.g. FFTGvecs 4-point batch → 1 row with 10, Bnds single-point
                # batch → 1 row with 30,6 → total 2 rows, not 17
                hint_vals = summary.get('hints', {})
                # eval_idx is the batch just evaluated (1 row for single-point, 4/6 for normal)
                eval_idx = df_batch.index
                eval_df = self.ctx.df.loc[eval_idx]
                # build mask only over the evaluated batch
                mask_match = pd.Series([True] * len(eval_df), index=eval_df.index)
                for var in evaluator.parameters.var:
                    if var in hint_vals:
                        try:
                            hv = float(hint_vals[var].item() if isinstance(hint_vals[var], np.generic) else hint_vals[var])
                            mask_match = mask_match & (eval_df[var].astype(float).round(6) == round(hv, 6))
                        except (TypeError, ValueError):
                            try:
                                mask_match = mask_match & (eval_df[var] == hint_vals[var])
                            except (TypeError, ValueError, KeyError):
                                self.report(f'Could not compare the values of {var} with the converged one, not using it to mark the converged row.')
                # restrict to useful
                if 'useful' in eval_df.columns:
                    mask_match = mask_match & eval_df['useful'].astype(bool)
                if mask_match.any():
                    self.ctx.df.loc[mask_match[mask_match].index, 'converged'] = True
                else:
                    # fallback: mark by uuid (covers already_computed duplicates, pick last occurrence)
                    uuid = summary.get('last_converged_uuid')
                    if uuid:
                        uuid_mask = self.ctx.df['uuid'] == uuid
                        if uuid_mask.any():
                            # for single-point, the verification row is the last occurrence (max global_step)
                            last_idx = self.ctx.df[uuid_mask]['global_step'].idxmax()
                            self.ctx.df.loc[last_idx, 'converged'] = True
                        else:
                            self.ctx.df.loc[eval_idx, 'converged'] = True
                    else:
                        self.ctx.df.loc[eval_idx, 'converged'] = True
            except Exception as exc:
                self.report(f"Could not mark converged rows: {exc}")
                    
        elif decision['status'] == 'CONTINUE':
            self.report(f"Convergence on {evaluator.parameters.var} not met yet. Refining parameter values.")
            # A CONTINUE that requests a new (shifted) grid must leave single-point
            # verification mode; otherwise prepare_grid re-verifies the very same
            # point forever (check_single_point stays True, iteration never advances).
            if decision.get('new_grid'):
                evaluator.new_grid = True
                evaluator.check_single_point = False
                evaluator.inquired_point = None
            
        elif decision['status'] == 'FAILED_CALCULATION':
            self.report(f"Some calculations failed while converging {evaluator.parameters.var}.")
            self.ctx.none_encountered = True
            
        elif decision['status'] == 'SPACE_EXCEEDED':
            self.report("The boundary configuration space is too small to fulfill convergence constraints.")
            self.ctx.limit_reached = True

        # For CONTINUE (new_grid loop) keep the counter at `steps` so that
        # has_to_continue can advance the iteration; perform_next_step will
        # reset it to 0 after actually shifting the window. For terminal
        # decisions the counter is cleared.
        if decision['status'] != 'CONTINUE':
            evaluator.concurrent_steps_count = 0

    def _emit_story_outputs(self):
        """Set the `convergence_summary` and `collection` outputs: what has been found and computed so far.

        It is used also by the steps which stop the workflow with an exit code, so that the story of
        the calculations is not lost when the convergence cannot be completed.
        """
        summary = self.ctx.convergence_summary.copy() if hasattr(self.ctx, 'convergence_summary') else {}
        summary.pop('new_grid', None)
        summary.pop('already_computed', None)
        summary.pop('extrapolation_units', None)
        self.out('convergence_summary', store_Dict(_to_jsonable(summary)))

        # the `collection` output holds the full story of all the calculations
        # performed (the cumulative dataframe), not only the converged values.
        # NOTE: `workflows` is ordered from the most recent calculation to the first one.
        collection = {
            'workflows': [int(pk) for pk in self.ctx.wfl_pk],
            'workflows_uuids': [load_node(int(pk)).uuid for pk in self.ctx.wfl_pk],
        }
        if hasattr(self.ctx, 'df'):
            df_records = self.ctx.df.where(pd.notnull(self.ctx.df), None).to_dict(orient='records')
            collection['story'] = df_records
        self.out('collection', store_Dict(_to_jsonable(collection)))

    def _finish_scan(self, evaluator, df_batch):
        """The scan is complete: the failed calculations are recorded (not errors), unless all of them failed."""
        failed = df_batch[df_batch['failed'].astype(bool)]
        succeeded = df_batch[~df_batch['failed'].astype(bool)]
        self.report(f'Scan finished: {len(succeeded)} calculations succeeded, {len(failed)} failed.')

        evaluator.success = True
        evaluator.concurrent_steps_count = 0
        self.ctx.hint = {}
        if not succeeded.empty:
            last = succeeded.sort_values('global_step').iloc[-1]
            evaluator.last_succeeded_pk = int(last['pk'])
            evaluator.last_succeeded_uuid = str(last['uuid'])
        self.ctx.convergence_summary['scan'] = {
            'variables': list(evaluator.parameters.var),
            'points': int(len(df_batch)),
            'failed': int(len(failed)),
            'failed_pks': [int(pk) for pk in failed['pk']],
        }
        if succeeded.empty:
            self.report('All the calculations of the scan failed.')
            self.ctx.none_encountered = True

    def report_wf(self):
        """Finalize outputs and set workflow exit status after convergence ends."""

        evaluator = self.ctx.current_convergence
        self.report('Final step. Workflow success: {}'.format(evaluator.success))

        self._emit_story_outputs()

        try:
            last_uuid = evaluator.last_succeeded_uuid
            if last_uuid:
                calc = load_node(last_uuid)
                self.out_many(self.exposed_outputs(calc, YamboWorkflow))
        except Exception as exc:
            self.report(f'no YamboWorkflows available to expose outputs: {exc}')

        if self.ctx.none_encountered:
            self.report('Some calculation failed, so we stopped the workflow')
            return self.exit_codes.CALCS_FAILED
        elif self.ctx.limit_reached:
            self.report('Space too small to complete convergence.')
            return self.exit_codes.SPACE_TOO_SMALL
        elif not evaluator.success:
            self.report('Convergence not reached')
            return self.exit_codes.CONVERGENCE_NOT_REACHED

############################### preliminary calculation #####################
    def p2y_needed(self):
        """Predicate for the outline: the decision is taken (and stored) in `prepare_grid`.

        It must not modify the state, as it can be evaluated again when the workchain is reloaded from a checkpoint.
        """
        if getattr(self.ctx, 'run_p2y', None) is None: # old checkpoints
            self.ctx.run_p2y = self._evaluate_p2y_needed()
        return self.ctx.run_p2y

    def _set_parent_from(self, pk):
        """Set the remote folder of the calculation `pk` as parent. If it has none, report it and keep the current parent."""
        try:
            self.ctx.input_manager.set_parent_folder(load_node(pk).outputs.remote_folder)
        except (NotExistent, AttributeError, KeyError) as exc:
            self.report(f'Could not use the calculation {pk} as parent ({type(exc).__name__}: {exc}).')

    def _capped_nscf_bands(self, required_bands):
        """Apply `bands_nscf_update` to the number of bands to be computed in the NSCF.

        An integer N > 0 caps the number of bands to N; the string modes do not cap.
        """
        mode = self.ctx.bands_mode
        if isinstance(mode, int) and not isinstance(mode, bool) and mode > 0 and required_bands > 0:
            return min(required_bands, mode)
        return required_bands

    def _nscf_bands_for(self, evaluator):
        """The bands the NSCF is computed with, following `bands_nscf_update`:

        - `'all-at-once'`: the largest of the grids of all the parameters (their `stop`), so that the NSCF is done once;
        - `'full-step'`: the largest of the grid of the parameter which is being converged now;
        - an integer N: as `'all-at-once'`, but never more than N.

        If a grid shifts to larger values during the convergence, the NSCF is computed again with more bands.
        """
        if self.ctx.bands_mode == 'full-step':
            return _max_bands([evaluator.parameters], reach='stop')
        return _max_bands(list(self.ctx.input_manager.parameters_space) + [evaluator.parameters], reach='stop')

    def _check_nscf_bands(self, evaluator):
        """Report the NSCF bands and stop if they are not allowed: exit code, or None.

        Not allowed: more than the integer of `bands_nscf_update` (Yambo would ask for bands which are not computed, after
        the NSCF is done), or more than `max_nscf_bands` (the default is `DEFAULT_MAX_NSCF_BANDS`): raise it to confirm.
        """
        grid_bands = _max_bands([evaluator.parameters], reach='stop')
        settings = self.ctx.input_manager.workflow_settings
        limit = settings.get('max_nscf_bands', DEFAULT_MAX_NSCF_BANDS)
        cap = self.ctx.bands_mode if isinstance(self.ctx.bands_mode, int) else None
        nscf_bands = self.ctx.max_bands_needed
        if nscf_bands:
            self.report(f'The NSCF will be computed with {nscf_bands} bands (bands_nscf_update={self.ctx.bands_mode!r}, '
                        f'the largest bands of the grid are {grid_bands}).')
        if cap is not None and grid_bands > cap:
            self.report(f'The grid of {", ".join(evaluator.parameters.var)} needs {grid_bands} bands, more than the {cap} '
                        'of bands_nscf_update: raise it, or reduce the grid.')
            return self.exit_codes.TOO_MANY_BANDS
        if nscf_bands > limit:
            self.report(f'The NSCF would need {nscf_bands} bands, more than max_nscf_bands={limit}. Reduce the bands of '
                        'the parameters, use bands_nscf_update="full-step" or an integer cap, or confirm by setting '
                        f'workflow_settings["max_nscf_bands"] to at least {nscf_bands}.')
            return self.exit_codes.TOO_MANY_BANDS
        return None

    def _evaluate_p2y_needed(self):
        """Determine whether a preliminary calculation is required before convergence."""

        if self.ctx.input_manager.workflow_settings.get('skip_pre'):
            self.report('skipping pre (debug mode)')
            return False

        current_parameters = self.ctx.current_convergence.parameters
        if not hasattr(self.ctx, 'params_space'):
            # build the value grid (not the parameter definition itself)
            self.ctx.params_space = _build_params_space(current_parameters)

        #self.report(f'Current convergence variables: {current_parameters.var}')

        if any(v in ('kpoint_mesh', 'kpoint_density') for v in current_parameters.var):
            return False

        # the grid prepared in prepare_grid knows the max number of bands (also of the shifted window)
        self.ctx.gwbands = self._capped_nscf_bands(int(getattr(self.ctx, 'max_bands_needed', 0) or 0))

        try:
            already_done, parent_nscf, parent_scf = search_in_group(self.ctx.input_manager, self.ctx.group, calc_type=self.ctx.calc_type, up_to_p2y=True)

            if already_done:
                self._set_parent_from(already_done)
                return False

            if parent_nscf:
                # full NSCF validity (kpoints, nbnd, structure, pseudos, ecut) — not just kpoints/nbnd
                if not is_nscf_valid(load_node(parent_nscf), self.ctx.input_manager):
                    parent_nscf = False
                else:
                    self._set_parent_from(parent_nscf)
                if not parent_nscf and parent_scf:
                    self._set_parent_from(parent_scf)
            elif parent_scf:
                self._set_parent_from(parent_scf)

            _, nscf_params, _, self.ctx.bands, messages = quantumespresso_input_validator(self.ctx.input_manager, calc_type=self.ctx.calc_type)
            self.report(messages)
            self.ctx.gwbands = max(self.ctx.gwbands, self.ctx.bands)
            parent_calc = take_calc_from_remote(self.ctx.input_manager.parent_folder)
            nbnd = nscf_params['SYSTEM']['nbnd'] if isinstance(nscf_params, dict) else nscf_params.get_dict()['SYSTEM']['nbnd']

            if nbnd < self.ctx.gwbands:
                self.ctx.input_manager.set_parent_folder(find_pw_parent(parent_calc, calc_type=['scf']).outputs.remote_folder)
                return True

            # NOTE: `precalc_inputs` is not a declared input of this workchain:
            # the legacy checks on it were always False and are removed.
            if parent_calc.process_type == 'aiida.calculations:yambo.yambo':
                return False

            return True
        except Exception as exc:
            self.report(f'Could not inspect the parent calculations ({type(exc).__name__}: {exc}), trying a simpler search.')
            try:
                already_done, parent_nscf, parent_scf = search_in_group(self.ctx.input_manager, self.ctx.group, calc_type=self.ctx.calc_type, up_to_p2y=True)

                if already_done:
                    self.ctx.input_manager.set_parent_folder(load_node(already_done).outputs.remote_folder)
                    return False
                elif parent_nscf:
                    self.ctx.input_manager.set_parent_folder(load_node(parent_nscf).outputs.remote_folder)
                    return True
                else:
                    parent_calc = take_calc_from_remote(self.ctx.input_manager.parent_folder)
                    self.ctx.input_manager.set_parent_folder(find_pw_parent(parent_calc, calc_type=['scf']).outputs.remote_folder)
                    return True
            except Exception as exc:
                self.report(f'Could not find any valid parent ({type(exc).__name__}: {exc}), so the pre-calculation will be done.')
                return True

    def do_p2y(self):
        """Submit the pre-calculation or parent preparation step."""
        # rebuild a fresh pristine manager, so the pre-calculation is not polluted
        # by the (possibly shifted/converged) live inputs.
        # NOTE: actually, it should be. If we increase the bands, we need to know it to do an NSCF with more bands...
        # mode='node' restores also the metadata/options (see setup()).
        pre_manager = YamboInputManager.get_nested_inputs(
            self.node, what='YamboConvergence', mode='node'
        )
        pre_manager = YamboInputManager(**pre_manager)

        if self.ctx.input_manager.parent_folder is not None:
            pre_manager.set_parent_folder(self.ctx.input_manager.parent_folder)

        try:
            pre_manager.nscf.set_kmesh(*self.ctx.input_manager.nscf.get_mesh(), store=True)
        except AttributeError: # no k-points mesh in the NSCF (yet)
            self.report('The NSCF inputs have no k-points mesh, the pre-calculation will use the original one.')

        # Use the max bands already computed in prepare_grid (which knows the shifted
        # window's ceiling). Fallback to scanning the space for old checkpoints.
        try:
            band_max = getattr(self.ctx, 'max_bands_needed', 0)
            if not band_max:
                band_max = self._nscf_bands_for(self.ctx.current_convergence)
            # `bands_nscf_update`: the integer caps (see `_nscf_bands_for`)
            required_nbnd = self._capped_nscf_bands(band_max)
            if required_nbnd and pre_manager.nscf.get_nbnd() < required_nbnd:
                self.report(f"Raising NSCF nbnd to {required_nbnd} for bands convergence (was {pre_manager.nscf.get_nbnd()})")
                pre_manager.set_nscf_nbnd(required_nbnd)
        except Exception as exc:
            self.report(f"Band anticipation in do_p2y skipped: {exc}")

        self.ctx.calculation_type = 'p2y'
        # only initialise the namespaces that actually carry inputs: setting
        # INITIALISE on an empty manager would make it look like "provided",
        # generating a partial namespace which fails validation at submission
        # (e.g. required 'bse.yambo.code' not provided).
        for name in ('qp', 'bse'):
            yres = getattr(pre_manager, name)
            if len(yres.parameters) > 0:
                yres.settings['INITIALISE'] = True
        pre_manager.additional_parsing = []

        self.report('doing the calculation: {}'.format(self.ctx.calculation_type))

        pre_inputs = pre_manager.generate_AiiDA_inputs(what='YamboWorkflow')
        metadata = dict(pre_inputs.get('metadata', {}) or {})
        metadata['call_link_label'] = self.ctx.calculation_type
        pre_inputs['metadata'] = metadata

        calc = {}
        calc[self.ctx.calculation_type] = self.submit(YamboWorkflow, **pre_inputs) #################run
        self.ctx.PRE = calc[self.ctx.calculation_type]
        self.report('Submitted YamboWorkflow up to {}, pk = {}'.format(self.ctx.calculation_type, calc[self.ctx.calculation_type].pk))
        self.ctx.group.add_nodes(calc[self.ctx.calculation_type]) #when added the whole YC, remove that

        orm.load_node(calc[self.ctx.calculation_type].pk).label = self.ctx.calculation_type

        # remove (rather than inject) INITIALISE: injecting False into an empty
        # manager would equally make it look like "provided" (flag-only), and a
        # missing key is already treated as False by YamboWorkflow.setup().
        for name in ('qp', 'bse'):
            getattr(self.ctx.input_manager, name).settings.pop('INITIALISE', None)

        return ToContext(calc)

    def prepare_calculations(self):
        """Attach the pre-calculation remote folder as parent before launching convergence calcs."""
        if not hasattr(self.ctx, 'PRE') or self.ctx.PRE is None:
            return
        if not self.ctx.PRE.is_finished_ok:
            self.report('the pre calc was not succesful, exiting...')
            self._emit_story_outputs()
            return self.exit_codes.PRECALC_FAILED

        self.report('setting the pre calc remote folder {} as parent'.format(self.ctx.PRE.outputs.remote_folder.pk))
        self.ctx.input_manager.set_parent_folder(self.ctx.PRE.outputs.remote_folder)
        # it is consumed: in the next laps a stale pre-calculation must not override the parent we find.
        self.ctx.PRE = None


if __name__ == "__main__":
    pass
