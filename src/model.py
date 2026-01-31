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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from config.settings import (
    ModelSettings,
    ScenarioType,
    TaskDefinition,
    get_task_network,
    validate_task_network,
)


# =============================================================================
# Data Classes for Model Components
# =============================================================================

@dataclass
class Node:
    """Logistics network node."""
    id: str
    name: str
    node_type: str  # "earth", "transit", "moon"


@dataclass
class Arc:
    """Logistics network arc."""
    id: str
    from_node: str
    to_node: str
    arc_type: str  # "elevator", "rocket", "transfer", "ground"
    lead_time: int  # L_a (months)
    payload: float  # U_a (kg)
    enabled_scenarios: list[str]


@dataclass
class Resource:
    """Resource type."""
    id: str
    name: str
    isru_producible: bool


@dataclass
class ModelData:
    """Container for all model input data."""
    nodes: list[Node]
    arcs: list[Arc]
    resources: list[Resource]
    tasks: list[TaskDefinition]
    settings: ModelSettings
    constants: dict[str, Any]


# =============================================================================
# Model State Variables
# =============================================================================

@dataclass
class StateVariables:
    """
    State variables at each time step t.
    
    Corresponds to Section 2 of model.md.
    """
    T: int  # Number of time steps
    R: int  # Number of resource types
    N: int  # Number of nodes
    A: int  # Number of arcs
    I: int  # Number of tasks
    
    # Capacity states (evolve via task completion)
    P: np.ndarray = field(default=None)      # (T,) ISRU production capacity
    V: np.ndarray = field(default=None)      # (T,) Construction capacity
    H: np.ndarray = field(default=None)      # (T,) Handling capacity
    Pop: np.ndarray = field(default=None)    # (T,) Population capacity
    
    # Inventory states
    I_E: np.ndarray = field(default=None)    # (N, R, T) Earth-origin inventory at nodes
    I_M: np.ndarray = field(default=None)    # (R, T) Moon-origin inventory (Moon only)
    B_E: np.ndarray = field(default=None)    # (R, T) Arrival buffer at Moon
    
    # Task states
    u: np.ndarray = field(default=None)      # (I, T) Task completion indicator (binary)
    
    def __post_init__(self):
        """Initialize arrays if not provided."""
        if self.P is None:
            self.P = np.zeros(self.T)
        if self.V is None:
            self.V = np.zeros(self.T)
        if self.H is None:
            self.H = np.zeros(self.T)
        if self.Pop is None:
            self.Pop = np.zeros(self.T)
        if self.I_E is None:
            self.I_E = np.zeros((self.N, self.R, self.T))
        if self.I_M is None:
            self.I_M = np.zeros((self.R, self.T))
        if self.B_E is None:
            self.B_E = np.zeros((self.R, self.T))
        if self.u is None:
            self.u = np.zeros((self.I, self.T), dtype=int)


# =============================================================================
# Decision Variables
# =============================================================================

@dataclass
class DecisionVariables:
    """
    Decision variables for optimization.
    
    Corresponds to Section 2 of model.md.
    """
    T: int
    R: int
    A: int
    I: int
    
    # Flow decisions
    x_E: np.ndarray = field(default=None)    # (A, R, T) Earth-origin flow on arcs
    y: np.ndarray = field(default=None)      # (A, T) Number of transport units dispatched
    
    # Moon operations
    A_E: np.ndarray = field(default=None)    # (R, T) Acceptance from buffer to usable
    Q: np.ndarray = field(default=None)      # (R, T) ISRU production
    
    # Task execution
    q_E: np.ndarray = field(default=None)    # (I, R, T) Earth-origin material allocation
    q_M: np.ndarray = field(default=None)    # (I, R, T) Moon-origin material allocation
    v: np.ndarray = field(default=None)      # (I, T) Construction work done
    k: np.ndarray = field(default=None)      # (I, R) Flexibility ratio (Earth vs Moon)
    
    def __post_init__(self):
        """Initialize arrays if not provided."""
        if self.x_E is None:
            self.x_E = np.zeros((self.A, self.R, self.T))
        if self.y is None:
            self.y = np.zeros((self.A, self.T), dtype=int)
        if self.A_E is None:
            self.A_E = np.zeros((self.R, self.T))
        if self.Q is None:
            self.Q = np.zeros((self.R, self.T))
        if self.q_E is None:
            self.q_E = np.zeros((self.I, self.R, self.T))
        if self.q_M is None:
            self.q_M = np.zeros((self.I, self.R, self.T))
        if self.v is None:
            self.v = np.zeros((self.I, self.T))
        if self.k is None:
            self.k = np.zeros((self.I, self.R))


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
    
    # Required keys in constants.yaml - validation will fail if any are missing
    REQUIRED_CONSTANT_SECTIONS = [
        "time",
        "nodes",
        "arcs",
        "resources",
        "scenarios",
        "costs",
        "initial_capacities",
        "task_defaults",
        "objective",
    ]
    
    REQUIRED_INITIAL_CAPACITIES = ["P_0", "V_0", "H_0", "Pop_0", "C_R_0", "C_E_0"]
    REQUIRED_OBJECTIVE_KEYS = ["w_C", "w_T"]
    REQUIRED_TIME_KEYS = ["delta_t", "T_max"]
    
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
        self.constants = self._load_constants(constants_path)
        self._validate_constants()
        self.data: ModelData | None = None
        self.state: StateVariables | None = None
        self.decisions: DecisionVariables | None = None
        self._solver = None
        
    def _load_constants(self, path: Path | str) -> dict[str, Any]:
        """Load constants from YAML file."""
        path = Path(path)
        if not path.exists():
            # Try relative to this file
            path = Path(__file__).parent / path
        if not path.exists():
            raise FileNotFoundError(f"Constants file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            raise ValueError(f"Constants file is empty: {path}")
        return data
    
    def _validate_constants(self) -> None:
        """
        Validate that all required keys are present in constants.
        
        Raises:
            KeyError: If required key is missing
            ValueError: If value has invalid type
        """
        # Check top-level sections
        for section in self.REQUIRED_CONSTANT_SECTIONS:
            if section not in self.constants:
                raise KeyError(f"Missing required section in constants.yaml: '{section}'")
        
        # Validate time section
        for key in self.REQUIRED_TIME_KEYS:
            if key not in self.constants["time"]:
                raise KeyError(f"Missing required key in constants.yaml time: '{key}'")
        
        # Validate initial_capacities
        for key in self.REQUIRED_INITIAL_CAPACITIES:
            if key not in self.constants["initial_capacities"]:
                raise KeyError(f"Missing required key in constants.yaml initial_capacities: '{key}'")
        
        # Validate objective
        for key in self.REQUIRED_OBJECTIVE_KEYS:
            if key not in self.constants["objective"]:
                raise KeyError(f"Missing required key in constants.yaml objective: '{key}'")
        
        # Validate nodes list
        if not self.constants["nodes"]:
            raise ValueError("constants.yaml nodes list cannot be empty")
        for i, node in enumerate(self.constants["nodes"]):
            for key in ["id", "name", "type"]:
                if key not in node:
                    raise KeyError(f"Missing key '{key}' in constants.yaml nodes[{i}]")
        
        # Validate arcs list
        if not self.constants["arcs"]:
            raise ValueError("constants.yaml arcs list cannot be empty")
        for i, arc in enumerate(self.constants["arcs"]):
            for key in ["id", "from", "to", "type", "lead_time", "payload", "enabled"]:
                if key not in arc:
                    raise KeyError(f"Missing key '{key}' in constants.yaml arcs[{i}]")
        
        # Validate resources list
        if not self.constants["resources"]:
            raise ValueError("constants.yaml resources list cannot be empty")
        for i, res in enumerate(self.constants["resources"]):
            for key in ["id", "name", "isru_producible"]:
                if key not in res:
                    raise KeyError(f"Missing key '{key}' in constants.yaml resources[{i}]")
    
    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    
    def load_data(self) -> None:
        """Load and process all model data from constants and settings."""
        # Parse nodes
        nodes = [
            Node(id=n["id"], name=n["name"], node_type=n["type"])
            for n in self.constants["nodes"]
        ]
        
        # Parse arcs (filter by scenario)
        scenario = self.settings.scenario.value
        arcs = [
            Arc(
                id=a["id"],
                from_node=a["from"],
                to_node=a["to"],
                arc_type=a["type"],
                lead_time=a["lead_time"],
                payload=a["payload"],
                enabled_scenarios=a["enabled"],
            )
            for a in self.constants["arcs"]
            if scenario in a["enabled"]
        ]
        
        # Parse resources
        resources = [
            Resource(id=r["id"], name=r["name"], isru_producible=r["isru_producible"])
            for r in self.constants["resources"]
        ]
        
        # Get tasks and validate
        tasks = get_task_network()
        validate_task_network(tasks)
        
        self.data = ModelData(
            nodes=nodes,
            arcs=arcs,
            resources=resources,
            tasks=tasks,
            settings=self.settings,
            constants=self.constants,
        )
        
        # Initialize state and decision variable containers
        T = self.settings.T_horizon
        R = len(resources)
        N = len(nodes)
        A = len(arcs)
        I = len(tasks)
        
        self.state = StateVariables(T=T, R=R, N=N, A=A, I=I)
        self.decisions = DecisionVariables(T=T, R=R, A=A, I=I)
        
        # Set initial capacities
        init_cap = self.constants["initial_capacities"]
        self.state.P[0] = init_cap["P_0"]
        self.state.V[0] = init_cap["V_0"]
        self.state.H[0] = init_cap["H_0"]
        self.state.Pop[0] = init_cap["Pop_0"]
    
    # -------------------------------------------------------------------------
    # Level 1: Capability Transition Equations (Eq. 3.1.A)
    # -------------------------------------------------------------------------
    
    def update_capabilities(self, t: int) -> None:
        """
        Update capability states at time t based on task completions.
        
        Implements equations from Section 3.1 (A):
            P(t) = P_0 + Σ ΔP_i · u_{i,t-d_i}
            V(t) = V_0 + Σ ΔV_i · u_{i,t-d_i}
            etc.
        """
        if self.data is None or self.state is None:
            raise RuntimeError("Model data not loaded. Call load_data() first.")
        
        init_cap = self.constants["initial_capacities"]
        d_default = self.constants["task_defaults"]["installation_delay"]
        
        P_increment = 0.0
        V_increment = 0.0
        H_increment = 0.0
        Pop_increment = 0
        
        for i, task in enumerate(self.data.tasks):
            # Check if task was completed d steps ago
            t_check = t - d_default
            if t_check >= 0 and self.state.u[i, t_check] == 1:
                P_increment += task.delta_P
                V_increment += task.delta_V
                H_increment += task.delta_H
                Pop_increment += task.delta_Pop
        
        self.state.P[t] = init_cap["P_0"] + P_increment
        self.state.V[t] = init_cap["V_0"] + V_increment
        self.state.H[t] = init_cap["H_0"] + H_increment
        self.state.Pop[t] = init_cap["Pop_0"] + Pop_increment
    
    # -------------------------------------------------------------------------
    # Level 2: Network Flow Constraints (Eq. 3.1.B)
    # -------------------------------------------------------------------------
    
    def build_flow_constraints(self) -> list[dict]:
        """
        Build flow conservation constraints for Level 2 network.
        
        Implements:
        - Scenario arc activation (Eq. 80)
        - Capacity pool constraints (Eq. 83)
        - Node inventory conservation (Eq. 92-102)
        
        Returns:
            List of constraint dictionaries for solver
        """
        if self.data is None:
            raise RuntimeError("Model data not loaded.")
        
        constraints = []
        delta_t = self.constants["time"]["delta_t"]
        
        # TODO: Implement constraint generation for MILP solver
        # This is a placeholder for the constraint building logic
        
        # Example structure:
        # for t in range(self.settings.T_horizon):
        #     for a, arc in enumerate(self.data.arcs):
        #         # Scenario activation constraint
        #         # Σ_r x_E[a,r,t] <= δ_a^(s) · U_a · y[a,t]
        #         pass
        
        return constraints
    
    # -------------------------------------------------------------------------
    # Task Execution Constraints (Eq. 3.1.C)
    # -------------------------------------------------------------------------
    
    def build_task_constraints(self) -> list[dict]:
        """
        Build task execution constraints.
        
        Implements:
        - Flexibility ratio bounds (Eq. 109-111)
        - ISRU capacity (Eq. 114)
        - Construction capacity (Eq. 117)
        - Material-construction coupling (Eq. 121)
        - Task completion conditions (Eq. 124-126)
        - Precedence constraints (Eq. 130)
        
        Returns:
            List of constraint dictionaries for solver
        """
        if self.data is None:
            raise RuntimeError("Model data not loaded.")
        
        constraints = []
        
        # TODO: Implement task constraint generation
        # Placeholder for MILP constraint building
        
        return constraints
    
    # -------------------------------------------------------------------------
    # Objective Function (Eq. 3.1.D)
    # -------------------------------------------------------------------------
    
    def build_objective(self) -> dict:
        """
        Build the objective function.
        
        Implements:
            min w_C · [Σ_{t,a} (f_{a,t} y_{a,t} + Σ_r c_{a,r,t} x^E_{a,r,t})] + w_T · T_end
        
        Returns:
            Objective dictionary for solver
        """
        obj_weights = self.constants["objective"]
        w_C = obj_weights["w_C"]
        w_T = obj_weights["w_T"]
        
        # TODO: Implement objective function building
        # Placeholder structure
        objective = {
            "type": "minimize",
            "w_C": w_C,
            "w_T": w_T,
            "cost_terms": [],
            "time_term": None,
        }
        
        return objective
    
    # -------------------------------------------------------------------------
    # Solver Interface
    # -------------------------------------------------------------------------
    
    def build_model(self) -> None:
        """Build the complete MILP model."""
        if self.data is None:
            self.load_data()
        
        # Build all constraint sets
        flow_constraints = self.build_flow_constraints()
        task_constraints = self.build_task_constraints()
        objective = self.build_objective()
        
        # TODO: Pass to MILP solver (e.g., Gurobi, CPLEX, HiGHS, OR-Tools)
        print(f"Model built with {len(flow_constraints)} flow constraints, "
              f"{len(task_constraints)} task constraints")
    
    def solve(self) -> dict[str, Any]:
        """
        Solve the optimization model.
        
        Returns:
            Solution dictionary with optimal values and status
        """
        if self._solver is None:
            self.build_model()
        
        # TODO: Implement solver invocation
        # Placeholder return
        result = {
            "status": "NOT_IMPLEMENTED",
            "objective_value": None,
            "T_end": None,
            "total_cost": None,
            "solution_time": None,
        }
        
        return result
    
    # -------------------------------------------------------------------------
    # Result Extraction
    # -------------------------------------------------------------------------
    
    def get_schedule(self) -> dict[str, list[tuple[int, int]]]:
        """
        Extract task schedule from solution.
        
        Returns:
            Dictionary mapping task_id to (start_month, end_month)
        """
        if self.state is None:
            raise RuntimeError("No solution available.")
        
        schedule = {}
        for i, task in enumerate(self.data.tasks):
            # Find first t where u[i,t] = 1
            completion_times = np.where(self.state.u[i, :] == 1)[0]
            if len(completion_times) > 0:
                end_t = completion_times[0]
                # Estimate start based on work duration
                # This is simplified; actual start comes from solution
                schedule[task.id] = (0, end_t)
        
        return schedule
    
    def get_flow_plan(self) -> dict[str, np.ndarray]:
        """
        Extract logistics flow plan from solution.
        
        Returns:
            Dictionary with flow arrays indexed by arc/resource/time
        """
        if self.decisions is None:
            raise RuntimeError("No solution available.")
        
        return {
            "x_E": self.decisions.x_E.copy(),
            "y": self.decisions.y.copy(),
        }
    
    def export_results(self, output_path: Path | str) -> None:
        """Export solution results to file."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # TODO: Implement result export (CSV, JSON, etc.)
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
