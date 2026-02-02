"""
Moon Logistics & Task Network Optimization Model.

This module implements the hierarchical MILP model described in model.md:
- Level 1 (Upper): AON/RCPSP task network with capability transitions
- Level 2 (Lower): Time-expanded multi-commodity network flow

The model supports three scenarios:
- E-only: Space Elevator only
- R-only: Rocket only
- Mix: Combined Elevator + Rocket

Author: [Your Name]
Based on: outline/model.md
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyomo.environ as pyo
    from pyomo.opt import TerminationCondition
except (
    ModuleNotFoundError
):  # pragma: no cover - exercised only when pyomo isn't installed
    pyo = None
    TerminationCondition = None

from entities import ModelData, StateVariables, DecisionVariables
import utils
from config.settings import ModelSettings


# =============================================================================
# Core Model Class
# =============================================================================


class MoonLogisticsModel:
    """
    Hierarchical MILP model for Earth-Moon logistics and task scheduling.

    Implements the two-level structure from model.md:
    - Level 1: AON/RCPSP task network with capability transitions
    - Level 2: Time-expanded multi-commodity network flow
    """

    def __init__(
        self,
        settings: ModelSettings,
        constants_path: Path | str,
    ):
        """
        Initialize the model.

        Args:
            settings: Runtime settings (REQUIRED - no defaults)
            constants_path: Path to constants YAML file (REQUIRED)

        Raises:
            ValueError: If settings is None
            FileNotFoundError: If constants file not found
            KeyError: If required keys missing from constants
        """
        # Strict validation - no defaults allowed
        if settings is None:
            raise ValueError("settings is REQUIRED - no defaults allowed")
        if constants_path is None:
            raise ValueError("constants_path is REQUIRED - no defaults allowed")

        self.settings = settings
        self.constants = utils.load_constants_from_file(constants_path)
        utils.validate_constants(self.constants)

        self.data: ModelData | None = None
        self.state: StateVariables | None = None
        self.decisions: DecisionVariables | None = None
        self._solver = None
        self._model: Any | None = None

    def check_unused_parameters(self) -> None:
        """Report unused parameters."""
        if hasattr(self.constants, "report_unused"):
            self.constants.report_unused()

    def load_data(self) -> None:
        """Load and process all model data from constants and settings."""
        self.data, self.state, self.decisions = utils.load_model_data(
            self.settings, self.constants
        )

    # -------------------------------------------------------------------------
    # Level 1: Capability Transition Equations (Eq. 3.1.A)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Level 1: Capability Transition Equations (REMOVED - Continuous Growth)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Section 5: Parameter Summary Helpers
    # -------------------------------------------------------------------------

    def _year_for_t(self, t: int) -> float:
        """Convert discrete time index t to calendar year."""
        return utils.get_year_for_t(t, self.constants)

    def _tier_definitions(self) -> list[dict[str, Any]]:
        """Return BOM tier definitions from parameter_summary."""
        return utils.get_tier_definitions(self.constants)

    def _rocket_cost_usd_per_kg(self, t: int) -> float:
        """Compute rocket transport cost per kg."""
        return utils.get_rocket_cost_usd_per_kg(t, self.constants)

    # -------------------------------------------------------------------------
    # Solver Interface
    # -------------------------------------------------------------------------

    def _require_pyomo(self) -> None:
        """Raise a helpful error if Pyomo is unavailable."""
        if pyo is None:
            raise ModuleNotFoundError(
                "Pyomo is not installed. Install with `pip install pyomo` "
                "and ensure a MILP solver (gurobi/cplex/highs) is available."
            )

    def _build_pyomo_model(self) -> Any:
        """Build the Pyomo MILP model."""
        self._require_pyomo()
        if self.data is None:
            raise RuntimeError("Model data not loaded.")

        # Delay import to avoid hard dependency at module level
        # (and circular dependency if optimization imports model.py)
        try:
            from optimization import PyomoBuilder
        except ImportError as e:
            # If optimization.py has issues, report
            print(f"Failed to import PyomoBuilder: {e}")
            raise

        builder = PyomoBuilder(self.data, self.settings, self.constants)
        return builder.build()

    def build_model(self) -> None:
        """Build the complete MILP model."""
        if self.data is None:
            self.load_data()
        self._model = self._build_pyomo_model()
        print("Pyomo model built.")

    def solve(self) -> dict[str, Any]:
        """
        Solve the optimization model.

        Returns:
            Solution dictionary with optimal values and status
        """
        self._require_pyomo()

        solver_candidates = ["highs", "gurobi", "cplex"]
        solver = None
        for name in solver_candidates:
            try:
                candidate = pyo.SolverFactory(name)
                if candidate is not None and candidate.available(False):
                    solver = candidate
                    break
            except Exception:
                continue
        if solver is None:
            raise RuntimeError(
                "No MILP solver available. Install Gurobi/CPLEX or HiGHS and ensure it is on PATH."
            )

        solver_name = solver.name.lower() if solver.name else ""
        solver_cfg = self.constants.get("solver", {})
        time_limit = solver_cfg.get("time_limit", self.settings.solver_timeout)
        gap_tol = solver_cfg.get("gap_tolerance", self.settings.mip_gap)
        threads = solver_cfg.get("threads")
        if solver_name == "gurobi":
            solver.options["TimeLimit"] = time_limit
            solver.options["MIPGap"] = gap_tol
            if threads is not None:
                solver.options["Threads"] = threads
        elif solver_name == "cplex":
            solver.options["timelimit"] = time_limit
            solver.options["mipgap"] = gap_tol
            if threads is not None:
                solver.options["threads"] = threads
        elif solver_name == "highs":
            solver.options["time_limit"] = time_limit
            solver.options["mip_rel_gap"] = gap_tol
            if threads is not None:
                solver.options["threads"] = threads

        def _solve_single_objective() -> tuple[dict[str, Any], Any]:
            if self._model is None:
                self.build_model()
            model = self._model

            result = solver.solve(model, tee=True, load_solutions=False)
            
            term = result.solver.termination_condition
            status_str = str(result.solver.status)
            
            is_ok = False
            if term == TerminationCondition.optimal:
                is_ok = True
            elif term == TerminationCondition.feasible:
                is_ok = True
            
            if not is_ok:
                return (
                    {
                        "status": str(term).upper(),
                        "objective_value": None,
                        "T_end": None,
                        "total_cost": None,
                        "solution_time": None,
                        "solver_status": status_str,
                    },
                    term,
                )

            model.solutions.load_from(result)

            try:
                total_obj = float(pyo.value(model.obj_total))
                total_cost = float(pyo.value(model.cost_total))
                T_end_val = None
                if hasattr(model, "Cumulative_City"):
                    target_mass = (
                        self.constants["parameter_summary"]["materials"]["bom"][
                            "total_demand_tons"
                        ]
                        * self.constants["units"]["ton_to_kg"]
                    )
                    for t in model.T:
                        val = pyo.value(model.Cumulative_City[t])
                        if val is None:
                            continue
                        if val >= target_mass:
                            T_end_val = int(t) + 1
                            break
            except (ValueError, TypeError):
                T_end_val = None
                total_obj = 0.0
                total_cost = 0.0

            return (
                {
                    "status": str(term).upper(),
                    "objective_value": total_obj,
                    "T_end": T_end_val,
                    "total_cost": total_cost,
                    "solution_time": None,
                    "solver_status": status_str,
                },
                term,
            )

        # Simple Horizon Loop (Removed complex task-based adaptation)
        horizon_step = int(solver_cfg.get("horizon_step_months", 12))
        horizon_attempts = int(solver_cfg.get("horizon_max_attempts", 0))
        T_max = int(self.constants["time"]["T_max"])
        original_T = self.settings.T_horizon

        def _rebuild_for_horizon(new_horizon: int) -> None:
            self.settings.T_horizon = new_horizon
            self.data = None
            self.state = None
            self.decisions = None
            self._model = None
            self.load_data()
            self.build_model()

        current_horizon = original_T
        max_attempts = horizon_attempts
        if max_attempts <= 0:
            max_attempts = max(0, (T_max - original_T) // horizon_step)

        for attempt in range(max_attempts + 1):
            if self._model is None or self.settings.T_horizon != current_horizon:
                _rebuild_for_horizon(current_horizon)
            result, term = _solve_single_objective()
            if term not in (
                TerminationCondition.infeasible,
                TerminationCondition.infeasibleOrUnbounded,
            ):
                return result
            if current_horizon + horizon_step > T_max:
                break
            current_horizon += horizon_step

        return result

    def sanity_check(self, tol: float = 1e-6, top_n: int = 8) -> dict[str, Any]:
        """
        Run a post-solve feasibility sanity check and report max violations.

        Returns:
            Dictionary with aggregate metrics and top constraint violations.
        """
        self._require_pyomo()
        if self._model is None:
            raise RuntimeError("No model available. Call build_model/solve first.")

        m = self._model

        def _iter_constraint_data(con):
            if con.is_indexed():
                return con.values()
            return [con]

        def _max_violation(con):
            max_v = 0.0
            count = 0
            worst_idx = None
            for c in _iter_constraint_data(con):
                if not c.active:
                    continue
                val = pyo.value(c.body)
                v = 0.0
                if c.equality:
                    target = pyo.value(c.lower)
                    v = abs(val - target)
                else:
                    lb = c.lower
                    ub = c.upper
                    if lb is not None:
                        lb_val = pyo.value(lb)
                        if val < lb_val:
                            v = max(v, lb_val - val)
                    if ub is not None:
                        ub_val = pyo.value(ub)
                        if val > ub_val:
                            v = max(v, val - ub_val)
                if v > max_v:
                    max_v = v
                    worst_idx = c.index()
                if v > tol:
                    count += 1
            return max_v, count, worst_idx

        violations = []
        max_overall = 0.0
        for con in m.component_objects(pyo.Constraint, active=True):
            max_v, count, worst_idx = _max_violation(con)
            max_overall = max(max_overall, max_v)
            violations.append(
                {
                    "name": con.name,
                    "max_violation": max_v,
                    "count": count,
                    "worst_index": worst_idx,
                }
            )

        violations_sorted = sorted(
            violations, key=lambda item: item["max_violation"], reverse=True
        )
        top_violations = violations_sorted[:top_n]

        total_demand = (
            self.constants["parameter_summary"]["materials"]["bom"]["total_demand_tons"]
            * self.constants["units"]["ton_to_kg"]
        )
        total_mass = sum(
            pyo.value(m.A_E[r, t] + m.Q[r, t]) for r in m.R for t in m.T
        )
        target_pop = self.constants["parameter_summary"]["colony"]["target"]["population"]
        
        t_end = len(m.T) - 1
        population_at_end = pyo.value(m.Pop[t_end])
        
        # New Metric: Cumulative City Mass
        if hasattr(m, "Cumulative_City"):
             cum_city_at_end = pyo.value(m.Cumulative_City[t_end])
        else:
             cum_city_at_end = 0.0

        rocket_trips = 0.0
        if hasattr(m, "A_Rock") and len(m.A_Rock) > 0:
            rocket_trips = sum(pyo.value(m.y[a, t]) for a in m.A_Rock for t in m.T)

        return {
            "total_mass": total_mass,
            "total_demand": total_demand,
            "population_at_end": population_at_end,
            "target_pop": target_pop,
            "cum_city_mass": cum_city_at_end,
            "max_violation": max_overall,
            "top_violations": top_violations,
            "rocket_trips": rocket_trips,
        }

    # -------------------------------------------------------------------------
    # Result Extraction
    # -------------------------------------------------------------------------

    def get_schedule(self) -> dict[str, list[tuple[int, int]]]:
        """
        DEPRECATED: Extract task schedule. (Continuous Growth uses flow rate).
        Returns empty dict.
        """
        return {}

    def get_flow_plan(self) -> dict[str, np.ndarray]:
        """
        Extract logistics flow plan from solution.

        Returns:
            Dictionary with flow arrays indexed by arc/resource/time
        """
        if self._model is None and self.decisions is None:
            raise RuntimeError("No solution available.")

        if self._model is not None:
            m = self._model
            x_vals = np.zeros((len(m.A), len(m.R), len(m.T)))
            y_vals = np.zeros((len(m.A), len(m.T)))
            arc_list = list(m.A)
            res_list = list(m.R)
            time_list = list(m.T)
            for ai, a in enumerate(arc_list):
                for ti, t in enumerate(time_list):
                    y_vals[ai, ti] = pyo.value(m.y[a, t])
                    for ri, r in enumerate(res_list):
                        x_vals[ai, ri, ti] = pyo.value(m.x[a, r, t])
            return {"x_E": x_vals, "y": y_vals}

        return {"x_E": self.decisions.x_E.copy(), "y": self.decisions.y.copy()}

    def export_results(self, output_path: Path | str) -> None:
        """Export solution results to file."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            from reporting import export_solution_report
        except ModuleNotFoundError:
            export_solution_report = None

        if export_solution_report is not None:
            export_solution_report(self, output_path)

        print(f"Results exported to {output_path}")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_model(
    settings: ModelSettings,
    constants_path: Path | str,
) -> MoonLogisticsModel:
    """
    Create and initialize a model with given settings.

    ALL PARAMETERS ARE REQUIRED - no defaults allowed.

    Args:
        settings: Validated ModelSettings instance (REQUIRED)
        constants_path: Path to constants YAML file (REQUIRED)

    Returns:
        Initialized MoonLogisticsModel

    Raises:
        ValueError: If any required parameter is missing
    """
    if settings is None:
        raise ValueError("settings is REQUIRED")
    if constants_path is None:
        raise ValueError("constants_path is REQUIRED")

    model = MoonLogisticsModel(settings=settings, constants_path=constants_path)
    model.load_data()
    return model
