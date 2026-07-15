from typing import Union, List, Optional, Any, Dict
import pandas as pd
from pydantic import BaseModel, Field

from aiida_yambo.workflows.utils.parameters import ConvParam, CoupledConvParam
from aiida_yambo.workflows.utils.predictor_1D import The_Predictor_1D
from aiida_yambo.workflows.utils.predictor_2D import The_Predictor_2D


class ConvergenceAnalyzer:
    """Unified convergence analysis engine.

    This class normalizes evaluation inputs for 1D and 2D convergence,
    builds a consistent history from completed workflow nodes, and
    dispatches the correct predictor routine.
    """

    def __init__(self, parameters: Union[ConvParam, CoupledConvParam], what: Union[str, List[str]], hints: Optional[Dict[str, Any]] = None):
        self.parameters = parameters
        self.what = what if isinstance(what, list) else [what]
        self.hints: Dict[str, Any] = hints.copy() if hints else {}
        self.history: Optional[pd.DataFrame] = None
        self.last_succeeded_node = None
        self.next_step: Dict[str, Any] = {}
        self.new_grid: bool = True
        self.check_single_point: bool = False
        self.inquired_point: Optional[List[Any]] = None
        self.none_encountered = False

    def _normalize_value(self, raw_value: Any, var_name: str) -> Any:
        if raw_value is None:
            return None
        if isinstance(raw_value, list) and len(raw_value) > 0:
            first = raw_value[0]
            if isinstance(first, list) and len(first) > 1:
                return first[1]
            if var_name == 'kpoint_mesh' and isinstance(first, (list, tuple)):
                return first
            return first
        return raw_value

    def _build_history(self, nodes: List[Any]) -> pd.DataFrame:
        records = []

        for node in nodes:
            row = {
                'uuid': node.uuid,
                'failed': not getattr(node, 'is_finished_ok', False)
            }

            for var in self.parameters.var:
                if 'mesh' in var:
                    try:
                        row[var] = node.inputs.nscf__kpoints.get_kpoints_mesh()[0]
                    except Exception:
                        row[var] = None
                    continue

                try:
                    variables = node.inputs.yres__yambo__parameters.get_dict()['variables']
                    raw_value = variables.get(var)
                    row[var] = self._normalize_value(raw_value, var)
                except Exception:
                    row[var] = None

            for quantity in self.what:
                if getattr(node, 'is_finished_ok', False):
                    try:
                        row[quantity] = node.outputs.output_ywfl_parameters.get_dict().get(quantity)
                    except Exception:
                        row[quantity] = None
                else:
                    row[quantity] = None

            records.append(row)

        if not records:
            raise ValueError('No calculation nodes were provided to ConvergenceAnalyzer.')

        return pd.DataFrame(records)

    def _build_grid(self) -> Dict[str, Any]:
        if self.history is None or self.history.empty:
            raise ValueError('Cannot build grid without history.')

        grid: Dict[str, Any] = {}
        for var in self.parameters.var:
            grid[var] = self.history[var].tolist()

        if 'kpoint_mesh' in self.parameters.var:
            grid['mesh'] = [list(value) for value in grid['kpoint_mesh'] if value is not None]

        return grid

    def _calculate_predictor(self, quantity: str) -> Dict[str, Any]:
        if self.history is None or self.history.empty:
            raise ValueError('No history available to calculate convergence.')

        grid = self._build_grid()
        target = self.history[quantity].to_numpy()
        result = self.history

        calc_dict = self.parameters.model_dump()
        calc_dict['var'] = self.parameters.var

        predictor_kwargs = {
            'grid': grid,
            'result': result,
            'bande': 0,
            'r': target,
            'calc_dict': calc_dict,
            'k': quantity,
        }

        if 'new_algorithm_2D' in self.parameters.convergence_algorithm:
            predictor = The_Predictor_2D(**predictor_kwargs)
        else:
            predictor = The_Predictor_1D(**predictor_kwargs)

        predictor.analyse(old_hints=self.hints)

        return {
            'converged': bool(getattr(predictor, 'check_passed', False) and getattr(predictor, 'point_reached', False)),
            'next_step': getattr(predictor, 'next_step', {}),
            'new_grid': bool(getattr(predictor, 'next_step', {}).get('new_grid', False)),
            'already_computed': bool(getattr(predictor, 'next_step', {}).get('already_computed', False)),
            'hints': getattr(predictor, 'next_step', {}),
        }

    def analyze(self, nodes: List[Any]) -> Dict[str, Any]:
        self.history = self._build_history(nodes)
        self.none_encountered = bool(self.history['failed'].any())
        self.last_succeeded_node = next((node for node in reversed(nodes) if getattr(node, 'is_finished_ok', False)), None)

        if self.none_encountered:
            return {'status': 'FAILED_CALCULATION', 'hints': self.hints}

        if not self.parameters.check_bounds()[0]:
            return {'status': 'SPACE_EXCEEDED', 'hints': self.hints}

        for quantity in self.what:
            result = self._calculate_predictor(quantity)
            self.hints.update(result.get('hints', {}) or {})
            self.next_step = result.get('next_step', {})
            self.new_grid = result.get('new_grid', True)
            self.check_single_point = not self.new_grid
            if self.check_single_point and self.next_step:
                self.inquired_point = [self.next_step[var] for var in self.parameters.var if var in self.next_step]

            if not result['converged']:
                return {
                    'status': 'CONTINUE',
                    'hints': self.hints,
                    'next_step': self.next_step,
                    'new_grid': self.new_grid,
                }

        return {
            'status': 'CONVERGED',
            'hints': self.hints,
            'next_step': self.next_step,
            'new_grid': self.new_grid,
        }


class ConvergenceEvaluator(BaseModel):
    """
    Tracks runtime execution metadata, histories, and strategy shifts
    for an ongoing convergence workflow loop.
    """
    parameters: Union[ConvParam, CoupledConvParam]
    what: Union[str, List[str]]

    success: bool = False
    current_iteration: int = 0
    requires_shift: bool = False
    concurrent_steps_count: int = 0 # description="Active number of calculation nodes currently submitted before the analysis.")

    hints: Dict[str, Any] = Field(default_factory=dict)
    error_log: Optional[str] = None

    history: Optional[pd.DataFrame] = None
    none_encountered: bool = False
    last_succeeded_node: Optional[Any] = None
    next_step: Dict[str, Any] = Field(default_factory=dict)
    new_grid: bool = False
    check_single_point: bool = False
    inquired_point: Optional[List[Any]] = None
    analyzer: Optional[ConvergenceAnalyzer] = None
    decision: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def steps(self) -> int:
        return self.parameters.steps

    @property
    def algo(self) -> str:
        return self.parameters.convergence_algorithm

    @property
    def what_list(self) -> List[str]:
        if isinstance(self.what, list):
            return self.what
        return [self.what]

    def advance_iteration(self, hints: Optional[Dict[str, Any]] = None):
        self.current_iteration += 1
        if hints:
            self.hints.update(hints)

    def trigger_boundary_shift(self, reasons: Dict[str, Any]):
        self.requires_shift = True
        self.hints.update(reasons)

    @property
    def edges_exceeded(self) -> bool:
        return not self.parameters.check_bounds()[0]

    @property
    def iterations_exceeded(self) -> bool:
        return self.current_iteration >= self.parameters.max_iterations

    def reset_concurrent_counter(self):
        self.concurrent_steps_count = 0

    @property
    def cannot_stop(self) -> bool:
        return self.concurrent_steps_count < self.parameters.steps
    
    def evaluate_nodes(self, nodes: List[Any]) -> Dict[str, Any]:
        self.analyzer = ConvergenceAnalyzer(self.parameters, self.what, hints=self.hints)
        self.decision = self.analyzer.analyze(nodes)
        self.history = self.analyzer.history
        self.none_encountered = self.analyzer.none_encountered
        self.last_succeeded_node = self.analyzer.last_succeeded_node
        self.next_step = self.analyzer.next_step
        self.new_grid = self.analyzer.new_grid
        self.check_single_point = self.analyzer.check_single_point
        self.inquired_point = self.analyzer.inquired_point
        self.hints = self.analyzer.hints
        return self.decision

    def get_decision(self) -> Dict[str, Any]:
        return self.decision or {'status': 'CONTINUE', 'hints': self.hints}

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            'parameters': self.parameters.var,
            'current_iteration': self.current_iteration,
            'success': self.success,
            'hints': self.hints,
        }
        if self.last_succeeded_node is not None:
            summary['last_converged_uuid'] = self.last_succeeded_node.uuid
        return summary
