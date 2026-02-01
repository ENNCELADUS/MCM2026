from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from utils import ConfigTracker
    from config.settings import ModelSettings, TaskDefinition


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
    lead_time: int  # L_a (time steps)
    payload: float  # U_a (mass units)
    cost_per_kg_2050: float  # Cost per mass unit
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
    constants: ConfigTracker | dict[str, Any]


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
    P: np.ndarray = field(default=None)  # (T,) ISRU production capacity
    V: np.ndarray = field(default=None)  # (T,) Construction capacity
    H: np.ndarray = field(default=None)  # (T,) Handling capacity
    Pop: np.ndarray = field(default=None)  # (T,) Population capacity
    Power: np.ndarray = field(default=None)  # (T,) Power capacity (MW)

    # Inventory states
    I_E: np.ndarray = field(default=None)  # (N, R, T) Earth-origin inventory at nodes
    I_M: np.ndarray = field(default=None)  # (R, T) Moon-origin inventory (Moon only)
    B_E: np.ndarray = field(default=None)  # (R, T) Arrival buffer at Moon

    # Task states
    u: np.ndarray = field(default=None)  # (I, T) Task completion indicator (binary)

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
        if self.Power is None:
            self.Power = np.zeros(self.T)
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
    x_E: np.ndarray = field(default=None)  # (A, R, T) Earth-origin flow on arcs
    y: np.ndarray = field(default=None)  # (A, T) Number of transport units dispatched

    # Moon operations
    A_E: np.ndarray = field(default=None)  # (R, T) Acceptance from buffer to usable
    Q: np.ndarray = field(default=None)  # (R, T) ISRU production

    # Task execution
    q_E: np.ndarray = field(default=None)  # (I, R, T) Earth-origin material allocation
    q_M: np.ndarray = field(default=None)  # (I, R, T) Moon-origin material allocation
    v: np.ndarray = field(default=None)  # (I, T) Construction work done
    k: np.ndarray = field(default=None)  # (I, R) Flexibility ratio (Earth vs Moon)

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
