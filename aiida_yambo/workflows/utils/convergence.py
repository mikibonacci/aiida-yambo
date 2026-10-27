from typing import Union, List, Optional, Any, Dict
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

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
        self.last_succeeded_pk = None
        self.last_succeeded_uuid = None

    def _build_grid(self) -> Dict[str, Any]:
        if self.history is None or self.history.empty:
            raise ValueError('Cannot build grid without history.')

        grid: Dict[str, Any] = {}
        for var in self.parameters.var:
            grid[var] = self.history[var].tolist()

        if 'kpoint_mesh' in self.parameters.var:
            # Sanitize ragged/None entries so the predictor never chokes on a
            # mix of 3-vectors and scalars/NaN (a None row becomes NaN in pandas).
            grid['kpoint_mesh'] = [
                list(value) for value in grid['kpoint_mesh']
                if value is not None and isinstance(value, (list, tuple)) and len(value) == 3
            ]
            grid['mesh'] = grid['kpoint_mesh']

        return grid

    def _calculate_predictor(self, quantity: str) -> Dict[str, Any]:
        if self.history is None or self.history.empty:
            raise ValueError('No history available to calculate convergence.')

        grid = self._build_grid()
        target = self.history[quantity].to_numpy()
        result = self.history

        calc_dict = self.parameters.model_dump()
        calc_dict['var'] = self.parameters.var
        if not isinstance(calc_dict.get('delta'), list):
            calc_dict['delta'] = [calc_dict['delta']]

        bande = 0
        try:
            from aiida.orm import load_node
            from aiida_yambo.utils.common_helpers import find_pw_parent
            last_pk = self.last_succeeded_pk or int(self.history['pk'].iloc[-1])
            pw = find_pw_parent(load_node(last_pk))
            bande = pw.outputs.output_band.get_bands()
        except Exception:
            bande = 0

        predictor_kwargs = {
            'grid': grid,
            'result': result,
            'bande': bande,
            'r': target,
            'calc_dict': calc_dict,
            'k': quantity,
            'what': quantity,
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

    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze convergence from a pre-filled dataframe.

        The dataframe is expected to already contain the parameter columns
        (one per ``parameters.var``), the ``what`` result columns (filled by
        ``take_quantities``), plus the ``pk``, ``uuid`` and ``failed`` columns.
        No AiiDA node is loaded during the analysis.
        """
        if df is None or df.empty:
            raise ValueError('No dataframe rows were provided to ConvergenceAnalyzer.')

        self.history = df.copy()
        if 'failed' not in self.history.columns:
            self.history['failed'] = False
        self.history['failed'] = self.history['failed'].astype(bool)

        # Drop rows carrying a missing value for any active convergence parameter
        # (e.g. kpoint_mesh=None rows from another parameter group); they would
        # otherwise reach _build_grid/predictor and poison the grid.
        for _var in self.parameters.var:
            if _var in self.history.columns:
                self.history = self.history[self.history[_var].notna()].copy()
        if self.history.empty:
            raise ValueError('No valid dataframe rows were provided to ConvergenceAnalyzer.')

        self.none_encountered = bool(self.history['failed'].any())
        succeeded = self.history[~self.history['failed']]
        if not succeeded.empty:
            self.last_succeeded_pk = int(succeeded.iloc[-1]['pk'])
            self.last_succeeded_uuid = str(succeeded.iloc[-1]['uuid'])
        else:
            self.last_succeeded_pk = None
            self.last_succeeded_uuid = None

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
                pt = [self.next_step[var] for var in self.parameters.var if var in self.next_step]
                # 1-var groups use bare values as grid points (like
                # build_stepped_space returns [5,5,5], not [[5,5,5]]);
                # only multi-var groups wrap the values in a list.
                self.inquired_point = pt[0] if len(self.parameters.var) == 1 else pt

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

    history: Optional[pd.DataFrame] = None # not used anymore, kept to load old checkpoints
    none_encountered: bool = False
    last_succeeded_node: Optional[Any] = None
    last_succeeded_pk: Optional[int] = None
    last_succeeded_uuid: Optional[str] = None
    next_step: Dict[str, Any] = Field(default_factory=dict)
    new_grid: bool = False
    check_single_point: bool = False
    inquired_point: Optional[List[Any]] = None
    analyzer: Optional[ConvergenceAnalyzer] = None # not used anymore, kept to load old checkpoints
    decision: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
        # mid-batch submissions still pending AND convergence not yet decided:
        # once converged, the counter reset (concurrent_steps_count = 0) must
        # not force the workflow into an extra loop lap.
        return (not self.success) and self.concurrent_steps_count < self.parameters.steps
    
    def evaluate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate convergence from a pre-filled dataframe (no node loading).

        The dataframe must already contain the parameter columns, the ``what``
        result columns, and the ``pk``, ``uuid``, ``failed``, ``useful`` columns
        (see ``take_quantities`` and ``YamboConvergence.perform_next_step``).
        """
        # NOTE: the analyzer and its history (a DataFrame) are not kept in the evaluator, which lives in the
        # `ctx` of the workchain, so it is written in each checkpoint.
        analyzer = ConvergenceAnalyzer(self.parameters, self.what, hints=self.hints)
        self.decision = analyzer.analyze_dataframe(df)
        self.none_encountered = analyzer.none_encountered
        self.last_succeeded_pk = analyzer.last_succeeded_pk
        self.last_succeeded_uuid = analyzer.last_succeeded_uuid
        self.next_step = analyzer.next_step
        self.new_grid = analyzer.new_grid
        self.check_single_point = analyzer.check_single_point
        self.inquired_point = analyzer.inquired_point
        self.hints = analyzer.hints
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
        if self.last_succeeded_uuid is not None:
            summary['last_converged_uuid'] = self.last_succeeded_uuid
        elif self.last_succeeded_node is not None:
            summary['last_converged_uuid'] = self.last_succeeded_node.uuid
        return summary
