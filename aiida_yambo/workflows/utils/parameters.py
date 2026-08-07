import itertools
from typing import List, Dict, Any, Union, Tuple, Literal
import numpy as np
from pydantic import BaseModel, Field, model_validator, field_validator

class ConvParamBase(BaseModel):
    """Base configuration shared across all convergence parameter types."""
    steps: int = 4 # steps to build the space during a given iteration
    max_concurrent_steps: int = 4 # max number of concurrent steps to run in a single iteration of the workflow.
    max_iterations: int = 4 # max attempts in the workflow to converge this params
    conv_thr_units: Literal['eV', '%'] = '%'
    convergence_algorithm: Literal['new_algorithm_1D','new_algorithm_2D', 'dummy'] = 'new_algorithm_1D'
    
    class Config:
        populate_by_name = True

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
        return v if isinstance(v, list) else [v]
        
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
    steps: int = 5
    max_iterations: int = 8
    points: int = 5
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
        lists_to_check = [self.var, self.start, self.stop, self.delta, self.max]
        lengths = {len(lst) for lst in lists_to_check}
        if len(lengths) > 1:
            raise ValueError("Dimensions mismatch! 'var', 'start', 'stop', 'delta', and 'max' must have identical lengths.")
        if len(self.var) != 2:
            raise ValueError(f"CoupledConvParam is strictly for 2 dimensions. Received {len(self.var)} variables.")
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
            self.points: 4 -> Returns the 4 corners of the rectangle (main & cross diagonals).
                    5 -> Returns the 4 corners + the exact center midpoint.
        """
        boundary_choices = [[st, sp] for st, sp in zip(self.start, self.stop)]
        grid_points = [list(p) for p in itertools.product(*boundary_choices)]
        
        if self.points == 4:
            return grid_points
        elif self.points == 5:
            center_point = [st + (sp - st) / 2.0 for st, sp in zip(self.start, self.stop)]
            grid_points.append([int(c) for c in center_point] if isinstance(self.start[0],int) else center_point)
            return grid_points
        else:
            raise ValueError("For 2D, requested points can only be 4 (corners) or 5 (corners + center).")


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


# =====================================================================
# Verification / Example Usage block
# =====================================================================
if __name__ == "__main__":
    print("--- Testing 1D Space Configuration ---")
    fft_space = ConvParam('FFTGvecs', start=10, stop=30, delta=5, max=50, units='Ry', conv_thr=0.5, steps=5)
    print("Workflow Dict Format:", fft_space.to_dict())
    print("Linear Space Array:  ", fft_space.build_linear_space())
    print("Stepped Space Array: ", fft_space.build_stepped_space())
    
    print('checking bounds... ', fft_space.check_bounds())

    print("\n--- Testing 2D Coupled Space Configuration ---")
    coupled_space = CoupledConvParam(
        var=['BndsRnXp', 'NGsBlkXp'],
        start=[10, 2],
        stop=[30, 6],
        delta=[10, 2],
        max=[50, 10],
        units=['','Ry'],
        conv_thr=1.0,
        mirror_values={'BndsRnXp':['GbndRnge']}
    )
    print("Workflow Dict Format:", coupled_space.to_dict())
    print("Linear Arrays per axis: ", coupled_space.build_linear_space())
    
    print("\nEvaluating build_diagonal_corners(points=4):")
    coupled_space.points=4
    for pt in coupled_space.build_diagonal_corners():
        print("  Corner:", pt)
        
    print("\nEvaluating build_diagonal_corners(points=5):")
    coupled_space.points=5
    for pt in coupled_space.build_diagonal_corners():
        print("  Point: ", pt)
        
    print('checking bounds... ', coupled_space.check_bounds())
    
    print('TESTING round trip')
    
    # 1. Initialize objects
    param_1d = ConvParam('FFTGvecs', start=10, stop=30, delta=5, max=50, units='Ry', conv_thr=0.5)
    param_2d = CoupledConvParam(var=['BndsRnXp', 'NGsBlkXp'], start=[10, 2], stop=[30, 6], 
                                units=['', 'Ry'],
                                delta=[10, 2], max=[50, 10], 
                                conv_thr=1.0,mirror_values={'BndsRnXp':['GbndRnge']})

    # 2. Convert to raw dicts (simulate saving to database / workflow input)
    raw_dictionaries = create_parameter_space_inputs([param_1d, param_2d])
    
    print(raw_dictionaries)

    # 3. Reload back into smart objects
    reconstructed_space = load_parameter_space(raw_dictionaries)

    # 4. Verify functionality is perfectly restored
    for obj in reconstructed_space:
        print(f"\nVariables: {obj.var} | Safe: {obj.check_bounds()[0]}")
        
        if isinstance(obj, ConvParam):
            print("Linear Space: ", obj.build_linear_space())
        else:
            print("5-Point Grid:", obj.build_diagonal_corners())
        