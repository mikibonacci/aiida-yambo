# -*- coding: utf-8 -*-
"""
Plugin to create a Yambo input file and run a calculation with the yambo executable.
"""
import os
import six

from aiida.engine import CalcJob


from aiida.common.datastructures import CalcInfo
from aiida.common.exceptions import UniquenessError, InputValidationError, ValidationError

from aiida.orm import Code
from aiida.orm import Dict
from aiida.orm import RemoteData, BandsData, ArrayData

from aiida.plugins import DataFactory, CalculationFactory

from aiida_yambo.data import YamboExcitonData, YamboLatticeData
from aiida_yambo.utils.common_helpers import take_calc_from_remote

from yambopy.io.inputfile import YamboIn

PwCalculation = CalculationFactory('quantumespresso.pw')
SingleFileData = DataFactory('core.singlefile')

__authors__ = " Miki Bonacci (miki.bonacci@unimore.it)," \
              " Gianluca Prandini (gianluca.prandini@epfl.ch)," \
              " Antimo Marrazzo (antimo.marrazzo@epfl.ch)," \
              " Michael Atambo (michaelontita.atambo@unimore.it)."


# the databases which are retrieved only temporarily (to be parsed), see the settings KEEP_RAW_DBS
_TEMPORARY_DBS = ('ns.db1', 'ndb.BS_diago_Q*', 'ndb.HF_and_locXC')


def additional_retrieve_entries(additional):
    """The entries of the setting `ADDITIONAL_RETRIEVE_LIST`: a list of paths (or patterns) in the working directory of
    the remote computer, or `[remote_path, local_name, depth]` (as in `CalcInfo.retrieve_list`). They are stored."""
    if not isinstance(additional, (list, tuple)):
        raise InputValidationError("ADDITIONAL_RETRIEVE_LIST must be a list of paths")
    for entry in additional:
        valid_string = isinstance(entry, str)
        valid_triple = (isinstance(entry, (list, tuple)) and len(entry) == 3 and isinstance(entry[0], str)
                        and isinstance(entry[1], str) and isinstance(entry[2], int))
        if not (valid_string or valid_triple):
            raise InputValidationError(
                f"ADDITIONAL_RETRIEVE_LIST: {entry!r} is not a path or a [remote_path, local_name, depth] list")
    return list(additional)


def split_retrieve_list(paths, keep_raw_dbs=False):
    """Split the files to retrieve in the ones stored in the `retrieved` folder and the temporary ones.

    The raw databases `ns.db1`, `ndb.BS_diago_Q*` and `ndb.HF_and_locXC` are only used by the parser (which keeps what
    is needed as `YamboLatticeData`, `YamboExcitonData` and arrays): they are retrieved temporarily unless
    `keep_raw_dbs` (setting `KEEP_RAW_DBS`). The other entries (also the ones which are not strings) are stored.
    """
    permanent, temporary = [], []
    for path in paths:
        if isinstance(path, str) and not keep_raw_dbs and os.path.basename(path) in _TEMPORARY_DBS:
            temporary.append(path)
        else:
            permanent.append(path)
    return permanent, temporary


class YamboCalculation(CalcJob):
    """
    AiiDA plugin for the Yambo code.
    For more information, refer to http://www.yambo-code.org/
    https://github.com/yambo-code/yambo-aiida and http://aiida-yambo.readthedocs.io/en/latest/
    """

    # Default input and output files
    _DEFAULT_INPUT_FILE = 'aiida.in'
    _DEFAULT_OUTPUT_FILE = 'aiida.out'

    @classmethod
    def define(cls,spec):
        super(YamboCalculation, cls).define(spec)
        spec.input('metadata.options.input_filename', valid_type=six.string_types, default=cls._DEFAULT_INPUT_FILE)
        spec.input('metadata.options.output_filename', valid_type=six.string_types, default=cls._DEFAULT_OUTPUT_FILE)

       # Default output parser provided by AiiDA
        spec.input('metadata.options.parser_name', valid_type=six.string_types, default='yambo.yambo')

       # self._SCRATCH_FOLDER = 'SAVE'
        spec.input('metadata.options.scratch_folder', valid_type=six.string_types, default='SAVE')
        spec.input('metadata.options.withmpi', valid_type=bool, default=True)

        spec.input('metadata.options.logostring', valid_type=six.string_types, default="""
#
# Y88b    /   e           e    e      888~~\    ,88~-_
#  Y88b  /   d8b         d8b  d8b     888   |  d888   \\
#   Y88b/   /Y88b       d888bdY88b    888 _/  88888    |
#    Y8Y   /  Y88b     / Y88Y Y888b   888  \  88888    |
#     Y   /____Y88b   /   YY   Y888b  888   |  Y888   /
#    /   /      Y88b /          Y888b 888__/    `88_-~
#
#             AIIDA input plugin.  YAMBO >4.0 compatible
#               http://www.yambo-code.org
#
"""
)

        spec.input('settings',valid_type=Dict,
                default=lambda: Dict(dict={'INITIALISE':False}))
        spec.input('parameters',valid_type=Dict,
                help='Use a node that specifies the input parameters')
        spec.input('parent_folder',valid_type=RemoteData,
                help='Use a remote folder as parent folder (for "restarts and similar"')
        spec.input('preprocessing_code',valid_type=Code,
                help='Use a preprocessing code for starting yambo',required=False)
        spec.input('precode_parameters',valid_type=Dict,
                default=lambda: Dict(dict={}),
                help='Use a node that specifies the input parameters for the yambo precode')
        spec.input('code',valid_type=Code,
                help='Use a main code for yambo calculation')
        spec.input('QP_corrections', valid_type=SingleFileData,
                required=False, help='ndb.QP for BSE calculations.')

        spec.exit_code(500, 'ERROR_NO_RETRIEVED_FOLDER',
                message='The retrieved folder data node could not be accessed.')
        spec.exit_code(501, 'WALLTIME_ERROR',
                message='time exceeded the max walltime')
        spec.exit_code(502, 'NO_SUCCESS',
                message='failed calculation for unknown reason')
        spec.exit_code(503, 'PARSER_ANOMALY',
                message='Unexpected behavior of YamboFolder')
        spec.exit_code(504, 'PARA_ERROR',
                message='parallelization error')
        spec.exit_code(505, 'MEMORY_ERROR',
                message='general memory error')
        spec.exit_code(506, 'X_par_MEMORY_ERROR',
                message='x_par allocation memory error')
        spec.exit_code(507, 'NaN_AS_OUTPUT',
                message='NaN error')
        spec.exit_code(508, 'Variable_NOT_DEFINED',
                message='Variable NOT DEFINED error, happens with ndb.pp_fragments.')


        #outputs definition:

        spec.output('output_parameters', valid_type=Dict,
                required=True, help='returns the output parameters')
        spec.output('array_alpha', valid_type=ArrayData,
                required=False, help='returns the alpha array')
        spec.output('array_alpha_bands', valid_type=ArrayData,
                required=False, help='returns the alpha array bands')
        spec.output('array_alpha_array', valid_type=ArrayData,
                required=False, help='returns the alpha array')
        spec.output('bands_quasiparticle', valid_type=BandsData,
                required=False, help='returns the quasiparticle band structure')
        spec.output('array_qp', valid_type=ArrayData,
                required=False, help='returns the quasiparticle array band structure')
        spec.output('array_eels', valid_type=ArrayData,
                required=False, help='returns the eels array')
        spec.output('array_eps', valid_type=ArrayData,
                required=False, help='returns the eps array')
        spec.output('array_ndb', valid_type=ArrayData,
                required=False, help='returns the array for ndb')
        spec.output('array_ndb_QP', valid_type=ArrayData,
                required=False, help='returns the array for ndbQP')
        spec.output('array_ndb_HFlocXC', valid_type=ArrayData,
                required=False, help='returns the array ndb for HFlocXC')
        spec.output('array_chi', valid_type=ArrayData,
                required=False, help='returns the array for chi')
        spec.output('array_excitonic_states', valid_type=ArrayData,
                required=False, help='returns the array for excitonic states')
        spec.output('system_info', valid_type=Dict,
                required=False, help='returns some system information after a p2y')
        spec.output('QP_db', valid_type=SingleFileData,
                required=False, help='returns the singlefiledata for ndbQP')
        spec.output('lattice', valid_type=YamboLatticeData,
                required=False, help='the lattice (ns.db1) of the calculation, for yambopy: see YamboLatticeData')
        spec.output_namespace('exciton_db', valid_type=YamboExcitonData, required=False, dynamic=True,
                help='the excitons of a BSE calculation, one node per Q point (Q1, Q2...): see YamboExcitonData')


    def prepare_for_submission(self, tempfolder):

        _dbs_accepted = {'gw0': 'ndb.QP', 'HF_and_locXC': 'ndb.HF_and_locXC','p2y':'ns.db1','bse':'ndb.BS_diago_Q*'}

        local_copy_list = []
        remote_copy_list = []
        remote_symlink_list = []

        # Settings can be undefined, and defaults to an empty dictionary.
        # They will be used for any input that doen't fit elsewhere.

        if hasattr(self.inputs,'QP_corrections'): local_copy_list.append((self.inputs.QP_corrections.uuid, self.inputs.QP_corrections.filename, 'ndb.QP'))
        
        settings = self.inputs.settings.get_dict()

        initialise = settings.pop('INITIALISE', None)
        if initialise is not None:
            if not isinstance(initialise, bool):
                raise InputValidationError("INITIALISE must be " " a boolean")

        copy_save = settings.pop('COPY_SAVE', None)
        if copy_save is not None:
            if not isinstance(copy_save, bool):
                raise InputValidationError("COPY_SAVE must be " " a boolean")

        copy_dbs = settings.pop('COPY_DBS', None)
        if copy_dbs is not None:
            if not isinstance(copy_dbs, bool):
                raise InputValidationError("COPY_DBS must be " " a boolean")
        
        restart_yambo = settings.pop('RESTART_YAMBO', None)
        if restart_yambo is not None:
            if not isinstance(restart_yambo, bool):
                raise InputValidationError("RESTART_YAMBO must be " " a boolean")
        
        verbose_timing = settings.pop('T_VERBOSE', None)
        if verbose_timing is not None:
            if not isinstance(verbose_timing, bool):
                raise InputValidationError("T_VERBOSE must be " " a boolean")
        
        iteration = settings.pop('ITERATION', None)

        # the databases yambopy reads are parsed from a temporary folder and kept as YamboLatticeData and
        # YamboExcitonData; these settings are used here (the raw files) and by the parser (the eigenvectors)
        keep_raw_dbs = settings.pop('KEEP_RAW_DBS', False)
        if not isinstance(keep_raw_dbs, bool):
            raise InputValidationError("KEEP_RAW_DBS must be a boolean")
        store_eigenvectors = settings.pop('STORE_EXCITON_EIGENVECTORS', False)
        if not isinstance(store_eigenvectors, bool):
            raise InputValidationError("STORE_EXCITON_EIGENVECTORS must be a boolean")
        number_of_eigenvectors = settings.pop('N_EXCITON_EIGENVECTORS', None)
        if number_of_eigenvectors is not None and (
                not isinstance(number_of_eigenvectors, int) or isinstance(number_of_eigenvectors, bool)
                or number_of_eigenvectors < 1):
            raise InputValidationError("N_EXCITON_EIGENVECTORS must be a positive integer")

        parameters = self.inputs.parameters

        if not initialise:
            if not isinstance(parameters, Dict):
                raise InputValidationError("parameters is not of type Dict")

        parent_calc_folder = self.inputs.parent_folder

        main_code = self.inputs.code

        preproc_code = self.inputs.preprocessing_code

        parent_calc = take_calc_from_remote(parent_calc_folder,level=-1)

        if parent_calc.process_type in [
            'aiida.calculations:yambo.yambo',
            'aiida.workflows:yambo.yambo.yamborestart',
            'aiida.workflows:yambo.yambo.yambowf',
        ]:
            yambo_parent=True
        else:
            yambo_parent=False

        # flags for yambo interfaces
        try:
            precode_param_dict = self.inputs.precode_parameters
        except AttributeError: # not provided
            precode_param_dict = Dict({})
        # check the precode parameters given in input
        input_cmdline = settings.pop('CMDLINE', None)
        import re
        precode_params_list = [] #['cd aiida.save'] ##.format(parent_calc_folder._PREFIX)
        pattern = re.compile(r"(^\-)([a-zA-Z])")
        for key, value in six.iteritems(precode_param_dict.get_dict()):
            if re.search(pattern, key) is not None:
                if key == '-O' or key == '-H' or key == '-h' or key == '-F':
                    raise InputValidationError(
                        "Precode flag {} is not allowed".format(str(key)))
                else:
                    if precode_param_dict[key] is True:
                        precode_params_list.append(str(key))
                    elif precode_param_dict[key] is False:
                        pass
                    else:
                        precode_params_list.append('{}'.format(str(key)))
                        precode_params_list.append('{}'.format(str(value)))
            else:
                raise InputValidationError(
                    "Wrong format of precode_parameters")
        # Adding manual cmdline input (e.g. for DB fragmentation)
        if input_cmdline is not None:
            precode_params_list = precode_params_list + input_cmdline

        # TODO: check that remote data must be on the same computer
        
        ##############################
        # END OF INITIAL INPUT CHECK #
        ##############################

        if not initialise:
            ###################################################
            # Prepare yambo input file
            ###################################################

            params_dict = parameters.get_dict() ##needed loop for retrocompatibility

            y = YamboIn().from_dictionary(params_dict)

            input_filename = tempfolder.get_abs_path(self.metadata.options.input_filename)

            y.write(input_filename, prefix=self.metadata.options.logostring)

        ############################################
        # set copy of the parent calculation
        ############################################

        try:
            parent_calc = parent_calc_folder.get_incoming().all_nodes()[-1] #to load the node from a workchain...
        except IndexError:
            parent_calc = parent_calc_folder.get_incoming().get_node_by_label('remote_folder')

        if yambo_parent:
            if copy_save:
                try:
                    remote_copy_list.append((parent_calc_folder.computer.uuid,parent_calc_folder.get_remote_path()+"/SAVE/",'./SAVE/'))
                except (AttributeError, TypeError):
                    remote_copy_list.append((parent_calc_folder.computer.uuid,parent_calc_folder.get_remote_path()+"out/aiida.save/SAVE/",'./SAVE/'))
            else:
                try:
                    remote_symlink_list.append((parent_calc_folder.computer.uuid,parent_calc_folder.get_remote_path()+"/SAVE/",'./SAVE/'))
                except (AttributeError, TypeError):
                    remote_symlink_list.append((parent_calc_folder.computer.uuid,parent_calc_folder.get_remote_path()+"out/aiida.save/SAVE/",'./SAVE/'))

            if copy_dbs:
                    remote_copy_list.append((parent_calc_folder.computer.uuid,parent_calc_folder.get_remote_path()+"/aiida.out/",'./aiida.out/'))
            if restart_yambo:
                    remote_symlink_list.append((parent_calc_folder.computer.uuid,parent_calc_folder.get_remote_path()+"/aiida.out/",'./aiida.out/'))
        else:
            remote_copy_list.append(
                (parent_calc_folder.computer.uuid,
                 os.path.join(parent_calc_folder.get_remote_path(),
                              PwCalculation._OUTPUT_SUBFOLDER,
                              "aiida.save","*" ),  ##.format(parent_calc_folder._PREFIX)
                                     "."
                                     )
                                    )
        ############################################
        # set Calcinfo
        ############################################

        calcinfo = CalcInfo()

        calcinfo.uuid = self.uuid

        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.remote_symlink_list = remote_symlink_list

        # Retrieve by default the output file and the xml file
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list.append('r*')
        calcinfo.retrieve_list.append('l*')
        calcinfo.retrieve_list.append('o*')
        calcinfo.retrieve_list.append('LOG/l*_CPU_1')
        #calcinfo.retrieve_list.append('LOG/l*_CPU_2')
        calcinfo.retrieve_list.append('*stderr*') #standard errors
        extra_retrieved = []

        if initialise:
            extra_retrieved.append('SAVE/'+_dbs_accepted['p2y'])
        else:
            for dbs in _dbs_accepted.keys():
                if dbs in params_dict['arguments']:
                    extra_retrieved.append('aiida.out/'+_dbs_accepted[dbs])

        extra_retrieved.extend(additional_retrieve_entries(settings.pop('ADDITIONAL_RETRIEVE_LIST', [])))

        from aiida.common.datastructures import CodeRunMode, CodeInfo

        # c1 = interface dft codes and yambo (ex. p2y or a2y)
        c1 = CodeInfo()
        c1.withmpi = True
        c1.cmdline_params = precode_params_list

        # c2 = yambo initialization
        c2 = CodeInfo()
        c2.withmpi = True
        c2.cmdline_params = []
        c2.code_uuid = main_code.uuid

        # if the parent calculation is a yambo calculation skip the interface (c1) and the initialization (c2)
        '''if yambo_parent:
            try:
                parent_settings = _uppercase_dict(
                    parent_calc.inputs.settings.get_dict(),
                    dict_name='parent settings')
                parent_initialise = parent_settings['INITIALISE']
            except KeyError:
                parent_initialise = False
            #c1 = None
            #if not parent_initialise:
            #    c2 = None
        else:'''
        c1.cmdline_params = precode_params_list
        c1.code_uuid = preproc_code.uuid

        # c3 = yambo calculation
        c3 = CodeInfo()
        c3.withmpi = True
        #c3.withmpi = self.get_withmpi()
        c3.cmdline_params = [
            "-F", self.metadata.options.input_filename, \
            '-J', self.metadata.options.output_filename, \
        ]
        c3.code_uuid = main_code.uuid

        if initialise:
            #c2 = None
            c3 = None

        #logic of the execution
        #calcinfo.codes_info = [c1, c2, c3] if not yambo_parent else [c3]   
        try:
            parent_save_path = parent_calc.outputs.output_parameters.get_dict().pop('ns_db1_path', None)
        except AttributeError: # the parent has no output parameters
            parent_save_path = None

        if yambo_parent:
            calcinfo.codes_info = [c3]
            #if not parent_initialise:
            #    calcinfo.codes_info = [c3]
            #else:
            #    calcinfo.codes_info = [c2, c3]
        elif initialise:
            calcinfo.codes_info = [c1, c2]
            
        else:
            calcinfo.codes_info = [c1, c2, c3]
        
        #if not parent_save_path: extra_retrieved.append('SAVE/ns.db1')
        extra_retrieved.append('SAVE/ns.db1')  # the lattice, for the parser (retrieved temporarily, see split_retrieve_list)

        calcinfo.codes_run_mode = CodeRunMode.SERIAL
        
        if settings:
            raise InputValidationError(
                "The following keys have been found in "
                "the settings input node, but were not understood: {}".format(
                    ",".join(list(settings.keys()))))

        permanent, temporary = split_retrieve_list(extra_retrieved, keep_raw_dbs)
        calcinfo.retrieve_list.extend(permanent)
        calcinfo.retrieve_temporary_list = temporary

        return calcinfo

################################################################################
    #the following functions are not used
    def _check_valid_parent(self, calc):
        """
        Check that calc is a valid parent for a YamboCalculation.
        It can be a PwCalculation or a YamboCalculation.
        """

        try:
            if ((not isinstance(calc, PwCalculation))
                    and (not isinstance(calc, YamboCalculation))):
                raise ValueError(
                    "Parent calculation must be a PwCalculation or a YamboCalculation"
                )

        except ImportError:
            if ((not isinstance(calc, PwCalculation))
                    and (not isinstance(calc, YamboCalculation))):
                raise ValueError(
                    "Parent calculation must be a PwCalculation or a YamboCalculation"
                )

    def use_parent_calculation(self, calc):
        """
        Set the parent calculation of Yambo,
        from which it will inherit the outputsubfolder.
        The link will be created from parent RemoteData to YamboCalculation
        """
        from aiida.common.exceptions import NotExistent

        self._check_valid_parent(calc)

        remotedatas = calc.get_outputs(node_type=RemoteData)
        if not remotedatas:
            raise NotExistent("No output remotedata found in " "the parent")
        if len(remotedatas) != 1:
            raise UniquenessError("More than one output remotedata found in "
                                  "the parent")
        remotedata = remotedatas[0]

        self._set_parent_remotedata(remotedata)

    def _set_parent_remotedata(self, remotedata):
        """
        Used to set a parent remotefolder in the start of Yambo.
        """
        if not isinstance(remotedata, RemoteData):
            raise ValueError('remotedata must be a RemoteData')

        # complain if another remotedata is already found
        input_remote = self.get_inputs(node_type=RemoteData)
        if input_remote:
            raise ValidationError("Cannot set several parent calculation to a "
                                  "Yambo calculation")

        self.use_parent_folder(remotedata)