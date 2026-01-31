"""
Variable Settings for Moon Logistics & Task Network Model.

This module contains runtime-configurable settings that may change between runs,
including task definitions, scenario selections, and solver options.

All physical constants are in config/constants.yaml.

STRICT VALIDATION: No hardcoded defaults. All required fields must be explicitly provided.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Sentinel for detecting missing required fields
_MISSING = object()


class ScenarioType(Enum):
    """Transport scenario types."""
    E_ONLY = "E-only"    # Elevator only
    R_ONLY = "R-only"    # Rocket only
    MIX = "Mix"          # Combined


def _validate_required(obj: Any, field_name: str, value: Any, expected_type: type) -> None:
    """Validate that a required field is provided and has correct type."""
    if value is _MISSING or value is None:
        raise ValueError(f"{obj.__class__.__name__}.{field_name} is REQUIRED and was not provided")
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{obj.__class__.__name__}.{field_name} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def _validate_positive(obj: Any, field_name: str, value: float | int) -> None:
    """Validate that a numeric field is positive."""
    if value <= 0:
        raise ValueError(f"{obj.__class__.__name__}.{field_name} must be positive, got {value}")


def _validate_non_negative(obj: Any, field_name: str, value: float | int) -> None:
    """Validate that a numeric field is non-negative."""
    if value < 0:
        raise ValueError(f"{obj.__class__.__name__}.{field_name} must be non-negative, got {value}")


def _validate_range(obj: Any, field_name: str, value: float, min_val: float, max_val: float) -> None:
    """Validate that a numeric field is within range."""
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"{obj.__class__.__name__}.{field_name} must be in [{min_val}, {max_val}], got {value}"
        )


@dataclass
class TaskDefinition:
    """
    AON/RCPSP Task Node Definition.
    
    ALL FIELDS ARE VALIDATED - no implicit defaults.
    
    Attributes:
        id: Unique task identifier (REQUIRED, non-empty string)
        name: Human-readable task name (REQUIRED, non-empty string)
        predecessors: List of predecessor task IDs (REQUIRED, can be empty list)
        M_earth: Material requirements from Earth (REQUIRED, can be empty dict)
        M_moon: Material requirements from Moon (REQUIRED, can be empty dict)
        M_flex: Flexible material requirements (REQUIRED, can be empty dict)
        W: Construction workload in kg (REQUIRED, >= 0)
        delta_P: ISRU capacity increment (REQUIRED, >= 0)
        delta_V: Construction capacity increment (REQUIRED, >= 0)
        delta_H: Handling capacity increment (REQUIRED, >= 0)
        delta_Pop: Population capacity increment (REQUIRED, >= 0)
    """
    id: str = field(default=_MISSING)
    name: str = field(default=_MISSING)
    predecessors: list[str] = field(default=_MISSING)
    M_earth: dict[str, float] = field(default=_MISSING)
    M_moon: dict[str, float] = field(default=_MISSING)
    M_flex: dict[str, float] = field(default=_MISSING)
    W: float = field(default=_MISSING)
    delta_P: float = field(default=_MISSING)
    delta_V: float = field(default=_MISSING)
    delta_H: float = field(default=_MISSING)
    delta_Pop: int = field(default=_MISSING)
    
    def __post_init__(self) -> None:
        """Validate all fields after initialization."""
        # Required string fields
        _validate_required(self, "id", self.id, str)
        _validate_required(self, "name", self.name, str)
        if not self.id.strip():
            raise ValueError("TaskDefinition.id cannot be empty")
        if not self.name.strip():
            raise ValueError("TaskDefinition.name cannot be empty")
        
        # Required list/dict fields (must be provided, but can be empty)
        _validate_required(self, "predecessors", self.predecessors, list)
        _validate_required(self, "M_earth", self.M_earth, dict)
        _validate_required(self, "M_moon", self.M_moon, dict)
        _validate_required(self, "M_flex", self.M_flex, dict)
        
        # Required numeric fields (must be provided and non-negative)
        _validate_required(self, "W", self.W, (int, float))
        _validate_required(self, "delta_P", self.delta_P, (int, float))
        _validate_required(self, "delta_V", self.delta_V, (int, float))
        _validate_required(self, "delta_H", self.delta_H, (int, float))
        _validate_required(self, "delta_Pop", self.delta_Pop, (int, float))
        
        _validate_non_negative(self, "W", self.W)
        _validate_non_negative(self, "delta_P", self.delta_P)
        _validate_non_negative(self, "delta_V", self.delta_V)
        _validate_non_negative(self, "delta_H", self.delta_H)
        _validate_non_negative(self, "delta_Pop", self.delta_Pop)
        
        # Validate dict value types
        for key, val in self.M_earth.items():
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError(f"TaskDefinition.M_earth['{key}'] must be non-negative number")
        for key, val in self.M_moon.items():
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError(f"TaskDefinition.M_moon['{key}'] must be non-negative number")
        for key, val in self.M_flex.items():
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError(f"TaskDefinition.M_flex['{key}'] must be non-negative number")


@dataclass
class ModelSettings:
    """
    Runtime model settings.
    
    ALL FIELDS ARE REQUIRED - no hardcoded defaults allowed.
    
    Attributes:
        scenario: Active transport scenario (REQUIRED)
        T_horizon: Planning horizon in months (REQUIRED, > 0)
        enable_learning_curve: Whether to apply learning curves (REQUIRED)
        enable_preposition: Allow material pre-positioning (REQUIRED)
        solver_timeout: Maximum solver time in seconds (REQUIRED, > 0)
        mip_gap: Acceptable MIP optimality gap (REQUIRED, 0 < gap < 1)
        output_dir: Directory for results output (REQUIRED)
    """
    scenario: ScenarioType = field(default=_MISSING)
    T_horizon: int = field(default=_MISSING)
    enable_learning_curve: bool = field(default=_MISSING)
    enable_preposition: bool = field(default=_MISSING)
    solver_timeout: int = field(default=_MISSING)
    mip_gap: float = field(default=_MISSING)
    output_dir: Path = field(default=_MISSING)
    
    def __post_init__(self) -> None:
        """Validate all fields after initialization."""
        # Scenario validation
        _validate_required(self, "scenario", self.scenario, ScenarioType)
        
        # T_horizon validation
        _validate_required(self, "T_horizon", self.T_horizon, int)
        _validate_positive(self, "T_horizon", self.T_horizon)
        
        # Boolean validations
        _validate_required(self, "enable_learning_curve", self.enable_learning_curve, bool)
        _validate_required(self, "enable_preposition", self.enable_preposition, bool)
        
        # Solver timeout validation
        _validate_required(self, "solver_timeout", self.solver_timeout, int)
        _validate_positive(self, "solver_timeout", self.solver_timeout)
        
        # MIP gap validation
        _validate_required(self, "mip_gap", self.mip_gap, float)
        _validate_range(self, "mip_gap", self.mip_gap, 0.0, 1.0)
        if self.mip_gap == 0.0:
            raise ValueError("ModelSettings.mip_gap cannot be exactly 0 (use small value like 1e-6)")
        
        # Output directory validation
        _validate_required(self, "output_dir", self.output_dir, Path)
        # Convert string to Path if needed
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))


# =============================================================================
# Task Network Definitions (AON/RCPSP)
# ALL FIELDS MUST BE EXPLICITLY PROVIDED - no defaults
# =============================================================================

TASK_NETWORK: list[TaskDefinition] = [
    # Phase 0: Initial Landing & Setup
    TaskDefinition(
        id="T0_landing",
        name="Initial Landing Site Preparation",
        predecessors=[],
        M_earth={"equipment": 5000.0, "electronics": 1000.0},
        M_moon={},
        M_flex={},
        W=6000.0,
        delta_P=0.0,
        delta_V=0.0,
        delta_H=0.001,  # Basic handling capability
        delta_Pop=0,
    ),
    
    # Phase 1: Resource Extraction
    TaskDefinition(
        id="T1_isru_pilot",
        name="ISRU Pilot Plant",
        predecessors=["T0_landing"],
        M_earth={"equipment": 20000.0, "electronics": 5000.0},
        M_moon={},
        M_flex={"structure": 10000.0},
        W=35000.0,
        delta_P=0.01,  # Initial ISRU capability
        delta_V=0.0,
        delta_H=0.0,
        delta_Pop=0,
    ),
    
    # Phase 2: Habitat Construction
    TaskDefinition(
        id="T2_habitat_core",
        name="Core Habitat Module",
        predecessors=["T1_isru_pilot"],
        M_earth={"life_support": 30000.0, "electronics": 10000.0},
        M_moon={"structure": 50000.0},
        M_flex={"water": 20000.0},
        W=110000.0,
        delta_P=0.0,
        delta_V=0.005,
        delta_H=0.0,
        delta_Pop=10,
    ),
    
    # TODO: Add remaining tasks from problem definition
    # - Power generation
    # - Transportation infrastructure
    # - Agricultural systems
    # - Population phases (100, 1000, 10000, ...)
]


# =============================================================================
# Capacity Growth Profiles
# =============================================================================

@dataclass
class CapacityProfile:
    """
    Time-varying capacity profile.
    
    ALL FIELDS ARE REQUIRED - must explicitly pass empty dict if not used.
    
    Attributes:
        C_R: Rocket capacity schedule {month: kg/s} (REQUIRED)
        C_E: Elevator capacity schedule {month: kg/s} (REQUIRED)
    """
    C_R: dict[int, float] = field(default=_MISSING)
    C_E: dict[int, float] = field(default=_MISSING)
    
    def __post_init__(self) -> None:
        """Validate all fields."""
        _validate_required(self, "C_R", self.C_R, dict)
        _validate_required(self, "C_E", self.C_E, dict)
        
        # Validate dict contents
        for month, val in self.C_R.items():
            if not isinstance(month, int) or month < 0:
                raise ValueError(f"CapacityProfile.C_R key must be non-negative int, got {month}")
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError(f"CapacityProfile.C_R[{month}] must be non-negative number")
        for month, val in self.C_E.items():
            if not isinstance(month, int) or month < 0:
                raise ValueError(f"CapacityProfile.C_E key must be non-negative int, got {month}")
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError(f"CapacityProfile.C_E[{month}] must be non-negative number")


# Example capacity profile - must be explicitly passed to model
EXAMPLE_CAPACITY_PROFILE = CapacityProfile(
    C_R={
        0: 0.5,
        24: 1.0,   # Year 2: doubled rocket capacity (Starship)
        48: 2.0,   # Year 4: further expansion
    },
    C_E={
        12: 0.2,   # Year 1: elevator operational
        36: 1.0,   # Year 3: expanded capacity
        60: 3.0,   # Year 5: multi-tether system
    },
)


# =============================================================================
# Task Network Loader
# =============================================================================

def get_task_network() -> list[TaskDefinition]:
    """
    Return the task network definitions.
    
    Returns a copy to prevent accidental mutation.
    """
    return TASK_NETWORK.copy()


def validate_task_network(tasks: list[TaskDefinition]) -> None:
    """
    Validate task network integrity.
    
    Raises:
        ValueError: If task network has invalid structure
    """
    if not tasks:
        raise ValueError("Task network cannot be empty")
    
    task_ids = {t.id for t in tasks}
    
    # Check for duplicate IDs
    if len(task_ids) != len(tasks):
        raise ValueError("Task network contains duplicate task IDs")
    
    # Check predecessor references
    for task in tasks:
        for pred_id in task.predecessors:
            if pred_id not in task_ids:
                raise ValueError(
                    f"Task '{task.id}' references unknown predecessor '{pred_id}'"
                )
    
    # Check for at least one root task (no predecessors)
    root_tasks = [t for t in tasks if not t.predecessors]
    if not root_tasks:
        raise ValueError("Task network must have at least one root task (no predecessors)")
