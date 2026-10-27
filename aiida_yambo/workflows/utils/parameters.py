import itertools
from math import isfinite
from typing import List, Dict, Any, Union, Tuple, Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

class ConvParamBase(BaseModel):
    """Base configuration shared across all convergence parameter types."""
    steps: int = 4 # steps to build the space during a given iteration
    max_concurrent_steps: int = 4 # max number of concurrent steps to run in a single iteration of the workflow.
    max_iterations: int = 4 # max attempts in the workflow to converge this params
    conv_thr_units: Literal['eV', '%'] = '%'
    convergence_algorithm: Literal['new_algorithm_1D','new_algorithm_2D', 'dummy'] = 'new_algorithm_1D'
    
    model_config = ConfigDict(populate_by_name=True)

    @field_validator('steps', 'max_concurrent_steps', 'max_iterations')
    @classmethod
    def _positive_counts(cls, value):
        if value <= 0:
            raise ValueError('Step, concurrency and iteration counts must be positive.')
        return value

    @staticmethod
    def _validate_axis(start, stop, delta, maximum, *, decreasing=False, fixed=False):
        if not all(isfinite(value) for value in (start, stop, delta, maximum)):
            raise ValueError('Parameter bounds and delta must be finite.')
        if fixed and start == stop and delta == 0:
            return
        if decreasing:
            if start < stop or delta >= 0:
                raise ValueError('Density requires start >= stop and a negative delta.')
        elif start > stop or delta <= 0:
            raise ValueError('Parameter bounds require start <= stop and a positive delta.')

    def to_dict(self) -> Dict[str, Any]:
        """Dumps data using the exact dictionary keys your workflow expects."""
        return self.model_dump(by_alias=True)


class ConvParam(ConvParamBase):
    """
    Single-variable (1D) parameter space.
    
    Example:
        fft = ConvParam('FFTGvecs', start=10, stop=30, delta=5, max=50, units='Ry', conv_thr=0.5, mirror_values:['BndsRnXp'])
    """
    var: List[str]
    start: Union[float, int]
    stop: Union[float, int]
    delta: Union[float, int]
    max: Union[float, int]
    units: str # units, as in yambopy
    conv_thr: Union[float, int] = 10 # 10%
    dtype: str = 'int'
    convergence_algorithm: str = 'new_algorithm_1D'
    mirror_values: List[str] = Field(default_factory=list) # description='list of other parameters we want to change simultaneously with var, putting the same value')

    def __init__(self, var: Union[str, List[str]], **data):
        super().__init__(var=var, **data)

    @field_validator('var', mode='before')
    @classmethod
    def _wrap_var(cls, v):
        wrapped = v if isinstance(v, list) else [v]
        if len(wrapped) != 1 or not wrapped[0]:
            raise ValueError('ConvParam requires exactly one non-empty variable name.')
        return wrapped

    @model_validator(mode='after')
    def _validate_bounds(self):
        # K-point subclasses validate their scalar/vector bounds separately.
        if self.var[0] not in ('kpoint_mesh', 'kpoint_density'):
            self._validate_axis(self.start, self.stop, self.delta, self.max)
        elif type(self) is ConvParam:
            raise ValueError('Use KpointMeshConvParam for k-point convergence.')
        return self
        
    @property
    def space_length(self) -> int:
        """Returns the number of elements in the generated grid slice."""
        # For 2D, this checks the length of the grid generated on the first axis
        return len(self.build_stepped_space())
        
    def check_bounds(self) -> Tuple[bool, str]:
        """
        Safely checks constraints without throwing an exception.
        Returns: (is_valid, error_message)
        """
        if self.stop > self.max:
            return False, f"Variable {self.var[0]}: Stop value ({self.stop}) exceeds max ({self.max})."
        return True, "Valid"

    def with_shift(self, start_shift: float = 0, stop_shift: float = 0) -> 'ConvParam':
        """Returns a cloned instance with shifted start or stop boundaries."""
        updated_data = self.model_dump(by_alias=True)
        updated_data['start'] += start_shift
        updated_data['stop'] += stop_shift
        return ConvParam(**updated_data)

    def build_linear_space(self) -> List[float]:
        """Generates explicit points from start to stop spaced by delta."""
        space = np.arange(self.start, self.stop + (self.delta / 2), self.delta, dtype=self.dtype).tolist()
        if max(space) > self.max:
            return ['space_exceeds_max']
        else:
            return space

    def build_stepped_space(self) -> List[float]:
        """Generates an explicit list of points containing exactly N steps."""
        space = np.linspace(self.start, self.stop, self.steps, dtype=self.dtype).tolist()
        if max(space) > self.max:
            return ['space_exceeds_max']
        else:
            return space


class KpointMeshConvParam(ConvParam):
    """
    Single-variable (1D) parameter space for k-point meshes or densities.

    For ``kpoint_mesh`` the bounds are 3-vectors ``[nx, ny, nz]``.
    For ``kpoint_density`` the bounds are scalar floats equal to the AiiDA
    distance (1/Å) passed straight to ``set_kpoints_mesh_from_density``: the
    grid runs from ``start`` (coarsest, largest distance) down to ``stop``
    (finest, smallest distance), with ``delta`` negative to shift the window
    finer, and ``max`` the coarse cap (no distance larger than ``max``).

    Example:
        mesh = KpointMeshConvParam(start=[4, 4, 4], stop=[12, 12, 12],
                                   delta=[1, 1, 1], max=[22, 22, 22],
                                   units='mesh', conv_thr=10)
        dens = KpointMeshConvParam(var='kpoint_density', start=0.80, stop=0.20,
                                   delta=-0.04, max=0.80, units='1/A', conv_thr=10)
    """
    start: Union[List[Union[float, int]], float]
    stop: Union[List[Union[float, int]], float]
    delta: Union[List[Union[float, int]], float]
    max: Union[List[Union[float, int]], float]

    def __init__(self, var: Union[str, List[str]] = 'kpoint_mesh', **data):
        super().__init__(var=var, **data)

    @field_validator('var', mode='before')
    @classmethod
    def _ensure_single_kpoint(cls, v):
        wrapped = v if isinstance(v, list) else [v]
        if len(wrapped) != 1 or wrapped[0] not in ('kpoint_mesh', 'kpoint_density'):
            raise ValueError(f"KpointMeshConvParam must have exactly one var in "
                             f"('kpoint_mesh', 'kpoint_density'). Received: {wrapped}.")
        return wrapped

    @model_validator(mode='after')
    def _validate_kpoint_shapes(self):
        is_mesh = self.var[0] == 'kpoint_mesh'
        for name in ('start', 'stop', 'delta', 'max'):
            v = getattr(self, name)
            if is_mesh:
                if not isinstance(v, list) or len(v) != 3:
                    raise ValueError(f"Kpoint mesh '{name}' must be 3-vector [nx,ny,nz]. Got: {v}")
            else:  # kpoint_density
                if isinstance(v, list):
                    raise ValueError(f"Kpoint density '{name}' must be scalar float, not list. Got: {v}")
        if is_mesh:
            for start, stop, delta, maximum in zip(self.start, self.stop, self.delta, self.max):
                self._validate_axis(start, stop, delta, maximum, fixed=True)
                if any(value <= 0 or not float(value).is_integer() for value in (start, stop, maximum)):
                    raise ValueError('K-point mesh bounds must be positive integers.')
                if not float(delta).is_integer():
                    raise ValueError('K-point mesh delta must be an integer.')
        else:
            self._validate_axis(self.start, self.stop, self.delta, self.max, decreasing=True)
            if min(self.start, self.stop, self.max) <= 0:
                raise ValueError('K-point distances must be positive.')
        return self

    def check_bounds(self) -> Tuple[bool, str]:
        if self.var[0] == 'kpoint_density':
            # decreasing density: distances must be positive and the coarsest
            # value (start) must stay within the coarse cap (max).
            if min(self.start, self.stop) <= 0:
                return False, f"Variable {self.var[0]}: distances must be positive (start={self.start}, stop={self.stop})."
            if self.start > self.max:
                return False, f"Variable {self.var[0]}: Start value ({self.start}) exceeds max ({self.max})."
            return True, "Valid"
        if any(s > m for s, m in zip(self.stop, self.max)):
            return False, f"Variable {self.var[0]}: Stop value ({self.stop}) exceeds max ({self.max})."
        return True, "Valid"

    def build_stepped_space(self) -> Union[List[List[float]], List[float]]:
        """Generates explicit list of mesh points or densities."""
        if self.var[0] == 'kpoint_density':
            space = np.linspace(self.start, self.stop, self.steps).tolist()
            if max(space) > self.max or min(space) <= 0:
                return ['space_exceeds_max']
            return space
        space = np.linspace(self.start, self.stop, self.steps, dtype=self.dtype).tolist()
        if any(np.any(np.asarray(p) > np.asarray(self.max)) for p in space):
            return ['space_exceeds_max']
        return space

    def build_linear_space(self) -> Union[List[List[float]], List[float]]:
        """Generates explicit mesh points or densities spaced by delta."""
        if self.var[0] == 'kpoint_density':
            # decreasing density: walk down from start; drop any non-positive
            # distance that would arise from a coarse delta overshooting 0.
            space = np.arange(self.start, self.stop + (self.delta / 2), self.delta).tolist()
            return [x for x in space if x > 0]
        space = [
            [start] if delta == 0 else np.arange(start, stop + (delta / 2), delta, dtype=self.dtype).tolist()
            for start, stop, delta in zip(self.start, self.stop, self.delta)
        ]
        return space

    def with_shift(self, start_shift: Union[float, List[float]] = 0, stop_shift: Union[float, List[float]] = 0) -> 'KpointMeshConvParam':
        """Returns a cloned instance with shifted start or stop boundaries."""
        updated_data = self.model_dump(by_alias=True)
        if self.var[0] == 'kpoint_density':
            # scalar density, decreasing with refinement. delta is expected to
            # be negative (window moves finer); clamp degenerate shifts so the
            # window stays positive and ordered (start > stop > 0).
            new_start = self.start + (start_shift if not isinstance(start_shift, list) else start_shift[0])
            new_stop = self.stop + (stop_shift if not isinstance(stop_shift, list) else stop_shift[0])
            if new_stop <= 0:
                new_stop = self.stop / 2
            if new_start <= 0 or new_start <= new_stop:
                new_start = max(self.start / 2, new_stop)
            updated_data['start'] = new_start
            updated_data['stop'] = new_stop
            return KpointMeshConvParam(**updated_data)
        if isinstance(start_shift, list):
            updated_data['start'] = [s + sh for s, sh in zip(self.start, start_shift)]
        else:
            updated_data['start'] = [s + start_shift for s in self.start]
        if isinstance(stop_shift, list):
            updated_data['stop'] = [s + sh for s, sh in zip(self.stop, stop_shift)]
        else:
            updated_data['stop'] = [s + stop_shift for s in self.stop]
        return KpointMeshConvParam(**updated_data)


class CoupledConvParam(ConvParamBase):
    """
    Strictly Two-Variable (2D) coupled parameter space.
    
    Example:
        coupled = CoupledConvParam(
            var=['BndsRnXp', 'NGsBlkXp'], 
            start=[10, 2], stop=[30, 6], delta=[10, 2], max=[50, 10],
            units=['', 'Ry'] 
            conv_thr=1.0,
            mirror_values = {'BndsRnXp': ['GbndRnge']}
        )
    """
    var: List[str]
    start: List[Union[float, int]]
    stop: List[Union[float, int]]
    delta: List[Union[float, int]]
    max: List[Union[float, int]]
    units: List[str] #'units, as in yambopy')
    conv_thr: Union[float, int] = 10 # 10%
    dtype: str = 'int'
    steps: int = 6
    max_iterations: int = 8
    points: Literal[4, 5, 6] = 6
    convergence_algorithm: str = 'new_algorithm_2D'
    # Dictionary mapping a parent variable to its specific mirrors: e.g., {'NGsBlkXp': ['NGsBlkXs', 'BSENGBlk']}
    mirror_values: Dict[str, List[str]] = Field(default_factory=dict)

    def __init__(self, var: Union[str, List[str]], **data):
        super().__init__(var=var, **data)

    @field_validator('var', mode='before')
    @classmethod
    def _wrap_var(cls, v):
        return v if isinstance(v, list) else [v]

    @model_validator(mode='after')
    def validate_strict_2d(self) -> 'CoupledConvParam':
        """Validates that exactly 2 variables are provided and all vector dimensions match."""
        lists_to_check = [self.var, self.start, self.stop, self.delta, self.max, self.units]
        lengths = {len(lst) for lst in lists_to_check}
        if len(lengths) > 1:
            raise ValueError("Dimensions mismatch! 'var', bounds, 'delta', and 'units' must have identical lengths.")
        if len(self.var) != 2:
            raise ValueError(f"CoupledConvParam is strictly for 2 dimensions. Received {len(self.var)} variables.")
        if any(not var for var in self.var) or len(set(self.var)) != 2:
            raise ValueError('Coupled variables must have distinct, non-empty names.')
        for bounds in zip(self.start, self.stop, self.delta, self.max):
            self._validate_axis(*bounds)
        return self
    
    @property
    def space_length(self) -> int:
        """Returns the number of elements in the generated grid slice."""
        # For 2D, this checks the length of the grid generated on the first axis
        return len(self.build_diagonal_corners())
    
    def check_bounds(self) -> Tuple[bool, str]:
        """
        Safely checks constraints for both dimensions.
        Returns: (is_valid, error_message)
        """
        # Internal dimension validation 
        if len(self.var) != 2 or len(self.start) != 2 or len(self.stop) != 2 or len(self.max) != 2:
            return False, "Dimension mismatch: This parameters space must have exactly 2 dimensions configured."

        # Bounds checking
        for i, (v, sp, mx) in enumerate(zip(self.var, self.stop, self.max)):
            if sp > mx:
                return False, f"Variable '{v}': Stop value ({sp}) exceeds max limit ({mx}) at axis index {i}."
        
        return True, "Valid"

    def with_shift(self, start_shifts: List[float], stop_shifts: List[float]) -> 'CoupledConvParam':
        """Returns a cloned instance with element-wise shifted boundaries."""
        if len(start_shifts) != 2 or len(stop_shifts) != 2:
            raise ValueError("Shift vectors must be exactly of length 2.")
        updated_data = self.model_dump(by_alias=True)
        updated_data['start'] = [s + shift for s, shift in zip(self.start, start_shifts)]
        updated_data['stop'] = [s + shift for s, shift in zip(self.stop, stop_shifts)]
        return CoupledConvParam(**updated_data)

    def build_linear_space(self) -> List[List[float]]:
        """Generates explicit line points for both axes using their respective deltas."""
        space = [
            np.arange(start, stop + (delta / 2), delta, dtype=self.dtype).tolist()
            for start, stop, delta in zip(self.start, self.stop, self.delta)
        ]
        
        return space

    def build_stepped_space(self) -> List[List[float]]:
        """Generates explicit points for both axes matching the required step count."""
        space = [
            np.linspace(start, stop, self.steps, dtype=self.dtype).tolist()
            for start, stop in zip(self.start, self.stop)
        ]
        
        return space

    def build_diagonal_corners(self) -> List[List[float]]:
        """
        Maps out geometric points on the 2D bounding rectangle box.
        
        Args:
            self.points: 4 -> Returns the 4 corners of the rectangle.
                    5 -> Returns the 4 corners + the exact center midpoint.
                    6 -> Returns the 4 corners + 25% and 75% points on the
                         UL->LR diagonal (upper-left [start[0], stop[1]] to
                         lower-right [stop[0], start[1]]). This is the default.
        """
        boundary_choices = [[st, sp] for st, sp in zip(self.start, self.stop)]
        grid_points = [list(p) for p in itertools.product(*boundary_choices)]
        
        if self.points == 4:
            return grid_points
        elif self.points == 5:
            center_point = [st + (sp - st) / 2.0 for st, sp in zip(self.start, self.stop)]
            grid_points.append([int(c) for c in center_point] if isinstance(self.start[0],int) else center_point)
            return grid_points
        elif self.points == 6:
            # diagonal from upper-left (start[0], stop[1]) to lower-right (stop[0], start[1])
            ul = [self.start[0], self.stop[1]]
            lr = [self.stop[0], self.start[1]]
            p25 = [ul[0] + 0.25 * (lr[0] - ul[0]), ul[1] + 0.25 * (lr[1] - ul[1])]
            p75 = [ul[0] + 0.75 * (lr[0] - ul[0]), ul[1] + 0.75 * (lr[1] - ul[1])]
            # keep int semantics when start is int (common for bands)
            is_int = isinstance(self.start[0], int) and isinstance(self.start[1], int)
            if is_int:
                p25 = [int(round(c)) for c in p25]
                p75 = [int(round(c)) for c in p75]
            grid_points.extend([p25, p75])
            return grid_points
        else:
            raise ValueError("For 2D, requested points can only be 4 (corners), 5 (corners + center) or 6 (corners + 25%/75% UL->LR diagonal).")


def create_parameter_space_inputs(params: List[Union[ConvParam, CoupledConvParam]]) -> list:
    """Helper utility to compile a list of parameter configurations into raw dictionaries."""
    return [p.to_dict() for p in params]


from typing import List, Dict, Any, Union

def load_parameter_space(raw_dicts: List[Dict[str, Any]]) -> List[Union[ConvParam, CoupledConvParam]]:
    """
    Parses a list of raw workflow dictionaries and reconstructs them into 
    their corresponding ConvParam or CoupledConvParam objects.
    
    Args:
        raw_dicts: A list of dictionaries (e.g., from builder.parameters_space).
        
    Returns:
        A list of initialized Pydantic parameter objects.
    """
    instantiated_params = []
    
    for d in raw_dicts:
        # Extract the list of variables to determine dimensionality
        variables = d.get('var', [])
        if isinstance(variables, str):
            variables = [variables]
        
        if not variables:
            raise ValueError("Invalid parameter dictionary: Missing or empty 'var' key.")
            
        if len(variables) == 1:
            # Reconstruct 1D space
            # Extract the string out of the list since ConvParam takes a string as its first arg
            var_name = variables[0]
            
            # Create a copy of the dict without 'var' to pass as kwargs
            kwargs = {k: v for k, v in d.items() if k != 'var'}
            
            # Map workflow dictionary keys back to internal fields if necessary
            # (Pydantic's populate_by_name allows using either aliases or field names)
            if var_name in ('kpoint_mesh', 'kpoint_density'):
                param_obj = KpointMeshConvParam(var=var_name, **kwargs)
            else:
                param_obj = ConvParam(var=var_name, **kwargs)
            instantiated_params.append(param_obj)
            
        elif len(variables) == 2:
            # Reconstruct 2D space
            # CoupledConvParam expects a list of strings mapped to the 'var' alias
            kwargs = {k: v for k, v in d.items() if k != 'var'}
            kwargs['var'] = variables
            
            param_obj = CoupledConvParam(**kwargs)
            instantiated_params.append(param_obj)
            
        else:
            raise ValueError(f"Unsupported dimensionality. Found {len(variables)} variables in space: {variables}")
            
    return instantiated_params
