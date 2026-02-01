"""
Variable Settings for Moon Logistics & Task Network Model.

This module contains runtime-configurable settings that may change between runs,
including task definitions, scenario selections, and solver options.

All physical constants are in config/constants.yaml.

STRICT VALIDATION: No hardcoded defaults. All required fields must be explicitly provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Sentinel for detecting missing required fields
_MISSING = object()


class ScenarioType(Enum):
    """Transport scenario types."""

    E_ONLY = "E-only"  # Elevator only
    R_ONLY = "R-only"  # Rocket only
    MIX = "Mix"  # Combined


def _validate_required(
    obj: Any, field_name: str, value: Any, expected_type: type
) -> None:
    """Validate that a required field is provided and has correct type."""
    if value is _MISSING or value is None:
        raise ValueError(
            f"{obj.__class__.__name__}.{field_name} is REQUIRED and was not provided"
        )
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{obj.__class__.__name__}.{field_name} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def _validate_positive(obj: Any, field_name: str, value: float | int) -> None:
    """Validate that a numeric field is positive."""
    if value <= 0:
        raise ValueError(
            f"{obj.__class__.__name__}.{field_name} must be positive, got {value}"
        )


def _validate_non_negative(obj: Any, field_name: str, value: float | int) -> None:
    """Validate that a numeric field is non-negative."""
    if value < 0:
        raise ValueError(
            f"{obj.__class__.__name__}.{field_name} must be non-negative, got {value}"
        )


def _validate_range(
    obj: Any, field_name: str, value: float, min_val: float, max_val: float
) -> None:
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
        duration_months: Task setup duration in model time steps (derived from months)
        delta_P: ISRU capacity increment (REQUIRED, >= 0)
        delta_V: Construction capacity increment (REQUIRED, >= 0)
        delta_H: Handling capacity increment (REQUIRED, >= 0)
        delta_Pop: Population capacity increment (REQUIRED, >= 0)
        delta_power_mw: Power capacity increment (REQUIRED, >= 0)
        task_type: Task type flag (REQUIRED, "capability" or "construction")
    """

    id: str = field(default=_MISSING)
    name: str = field(default=_MISSING)
    predecessors: list[str] = field(default=_MISSING)
    M_earth: dict[str, float] = field(default=_MISSING)
    M_moon: dict[str, float] = field(default=_MISSING)
    M_flex: dict[str, float] = field(default=_MISSING)
    W: float = field(default=_MISSING)
    duration_months: float = field(default=_MISSING)
    delta_P: float = field(default=_MISSING)
    delta_V: float = field(default=_MISSING)
    delta_H: float = field(default=_MISSING)
    delta_Pop: int = field(default=_MISSING)
    delta_power_mw: float = field(default=_MISSING)
    task_type: str = field(default=_MISSING)

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
        _validate_required(self, "duration_months", self.duration_months, (int, float))
        _validate_required(self, "delta_P", self.delta_P, (int, float))
        _validate_required(self, "delta_V", self.delta_V, (int, float))
        _validate_required(self, "delta_H", self.delta_H, (int, float))
        _validate_required(self, "delta_Pop", self.delta_Pop, (int, float))
        _validate_required(self, "delta_power_mw", self.delta_power_mw, (int, float))
        _validate_required(self, "task_type", self.task_type, str)

        _validate_non_negative(self, "W", self.W)
        _validate_non_negative(self, "duration_months", self.duration_months)
        _validate_non_negative(self, "delta_P", self.delta_P)
        _validate_non_negative(self, "delta_V", self.delta_V)
        _validate_non_negative(self, "delta_H", self.delta_H)
        _validate_non_negative(self, "delta_Pop", self.delta_Pop)
        _validate_non_negative(self, "delta_power_mw", self.delta_power_mw)

        allowed_task_types = {"capability", "construction"}
        if self.task_type not in allowed_task_types:
            raise ValueError(
                f"TaskDefinition.task_type must be one of {sorted(allowed_task_types)}, got {self.task_type}"
            )

        # Validate dict value types
        for key, val in self.M_earth.items():
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError(
                    f"TaskDefinition.M_earth['{key}'] must be non-negative number"
                )
        for key, val in self.M_moon.items():
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError(
                    f"TaskDefinition.M_moon['{key}'] must be non-negative number"
                )
        for key, val in self.M_flex.items():
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError(
                    f"TaskDefinition.M_flex['{key}'] must be non-negative number"
                )


@dataclass
class ModelSettings:
    """
    Runtime model settings.

    ALL FIELDS ARE REQUIRED - no hardcoded defaults allowed.

    Attributes:
        scenario: Active transport scenario (REQUIRED)
        T_horizon: Planning horizon in years/time steps (REQUIRED, > 0)
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
        _validate_required(
            self, "enable_learning_curve", self.enable_learning_curve, bool
        )
        _validate_required(self, "enable_preposition", self.enable_preposition, bool)

        # Solver timeout validation
        _validate_required(self, "solver_timeout", self.solver_timeout, int)
        _validate_positive(self, "solver_timeout", self.solver_timeout)

        # MIP gap validation
        _validate_required(self, "mip_gap", self.mip_gap, float)
        _validate_range(self, "mip_gap", self.mip_gap, 0.0, 1.0)
        if self.mip_gap == 0.0:
            raise ValueError(
                "ModelSettings.mip_gap cannot be exactly 0 (use small value like 1e-6)"
            )

        # Output directory validation
        _validate_required(self, "output_dir", self.output_dir, Path)
        # Convert string to Path if needed
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))


# =============================================================================
# Task Network Definitions (AON/RCPSP)
# Populated from constants.yaml implementation_details.tasks.wbs_tasks
# =============================================================================

TASK_NETWORK: list[TaskDefinition] = []


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
                raise ValueError(
                    f"CapacityProfile.C_R key must be non-negative int, got {month}"
                )
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError(
                    f"CapacityProfile.C_R[{month}] must be non-negative number"
                )
        for month, val in self.C_E.items():
            if not isinstance(month, int) or month < 0:
                raise ValueError(
                    f"CapacityProfile.C_E key must be non-negative int, got {month}"
                )
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError(
                    f"CapacityProfile.C_E[{month}] must be non-negative number"
                )


# Example capacity profile - must be explicitly passed to model
EXAMPLE_CAPACITY_PROFILE = CapacityProfile(
    C_R={
        0: 0.5,
        24: 1.0,  # Year 2: doubled rocket capacity (Starship)
        48: 2.0,  # Year 4: further expansion
    },
    C_E={
        12: 0.2,  # Year 1: elevator operational
        36: 1.0,  # Year 3: expanded capacity
        60: 3.0,  # Year 5: multi-tether system
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
        raise ValueError(
            "Task network must have at least one root task (no predecessors)"
        )


def build_task_network_from_wbs(
    wbs_tasks: list[dict[str, Any]],
    parameter_summary: dict[str, Any],
    units: dict[str, Any],
    time: dict[str, Any],
) -> list[TaskDefinition]:
    """
    Build TaskDefinition list from WBS task dictionaries in constants.yaml.
    """
    if not wbs_tasks:
        raise ValueError("wbs_tasks cannot be empty")

    ton_to_kg = units["ton_to_kg"]
    delta_t = time["delta_t"]
    steps_per_year = time["steps_per_year"]
    seconds_per_year = steps_per_year * delta_t
    seconds_per_month = seconds_per_year / 12.0
    steps_per_month = steps_per_year / 12.0

    tiers = parameter_summary["materials"]["bom"]["tiers"]
    tier1 = next((t for t in tiers if t["class"] == 1), None)
    tier2 = next((t for t in tiers if t["class"] == 2), None)
    tier3 = next((t for t in tiers if t["class"] == 3), None)
    if tier1 is None or tier2 is None:
        raise ValueError(
            "parameter_summary.materials.bom.tiers must include class 1 and class 2"
        )
    if tier3 is None:
        raise ValueError("parameter_summary.materials.bom.tiers must include class 3")

    tier1_share = tier1.get("share_initial", 0.0)
    tier2_share = tier2.get("share_initial", 0.0)
    tier12_total = tier1_share + tier2_share
    if tier12_total <= 0:
        raise ValueError("Tier 1+2 share_initial must be positive")

    tier1_resources = tier1.get("resources", [])
    tier2_resources = tier2.get("resources", [])
    tier3_resources = tier3.get("resources", [])
    if not tier1_resources or not tier2_resources:
        raise ValueError("Tier 1 and Tier 2 resources cannot be empty")
    if not tier3_resources:
        raise ValueError("Tier 3 resources cannot be empty")

    tasks: list[TaskDefinition] = []
    for task in wbs_tasks:
        earth_mass_kg = task["earth_mass_t"] * ton_to_kg
        regolith_mass_kg = task["regolith_mass_t"] * ton_to_kg

        tier1_mass = earth_mass_kg * (tier1_share / tier12_total)
        tier2_mass = earth_mass_kg * (tier2_share / tier12_total)

        M_earth: dict[str, float] = {}
        tier1_each = tier1_mass / len(tier1_resources)
        tier2_each = tier2_mass / len(tier2_resources)
        for res in tier1_resources:
            M_earth[res] = M_earth.get(res, 0.0) + tier1_each
        for res in tier2_resources:
            M_earth[res] = M_earth.get(res, 0.0) + tier2_each

        M_moon: dict[str, float] = {}
        tier3_each = regolith_mass_kg / len(tier3_resources)
        for res in tier3_resources:
            M_moon[res] = M_moon.get(res, 0.0) + tier3_each

        unlocks = task["unlocks"]
        delta_P = (unlocks["production_t_per_year"] * ton_to_kg) / seconds_per_year
        delta_H = (unlocks["handling_t_per_month"] * ton_to_kg) / seconds_per_month
        delta_V = (
            unlocks.get("construction_t_per_month", 0.0) * ton_to_kg
        ) / seconds_per_month
        delta_Pop = unlocks["population"]
        delta_power_mw = unlocks["power_mw"]
        duration_steps = task["duration_months"] * steps_per_month

        tasks.append(
            TaskDefinition(
                id=task["id"],
                name=task["name"],
                predecessors=task["predecessors"],
                M_earth=M_earth,
                M_moon=M_moon,
                M_flex={},
                W=regolith_mass_kg,
                duration_months=duration_steps,
                delta_P=delta_P,
                delta_V=delta_V,
                delta_H=delta_H,
                delta_Pop=delta_Pop,
                delta_power_mw=delta_power_mw,
                task_type=task["task_type"],
            )
        )

    return tasks


# =============================================================================
# Parameter Summary Validation (Section 5, model.md)
# =============================================================================


def validate_parameter_summary(summary: dict[str, Any], resource_ids: set[str]) -> None:
    """
    Validate parameter_summary structure and references.

    Ensures BOM tiers map to known resources and all required fields exist.
    """
    if not summary:
        raise ValueError("parameter_summary cannot be empty")

    # Materials / BOM validation
    materials = summary.get("materials")
    if materials is None:
        raise KeyError("parameter_summary.materials is required")
    bom = materials.get("bom")
    if bom is None:
        raise KeyError("parameter_summary.materials.bom is required")
    if "total_demand_tons" not in bom:
        raise KeyError("parameter_summary.materials.bom.total_demand_tons is required")
    tiers = bom.get("tiers")
    if not tiers:
        raise ValueError("parameter_summary.materials.bom.tiers cannot be empty")

    valid_source_policies = {"earth_only", "mixed", "isru_only"}
    for i, tier in enumerate(tiers):
        for key in ["id", "name", "class", "source_policy", "resources"]:
            if key not in tier:
                raise KeyError(
                    f"parameter_summary.materials.bom.tiers[{i}].{key} is required"
                )
        tier_id = tier["id"]
        tier_name = tier["name"]
        tier_class = tier["class"]
        _ = tier_id, tier_name, tier_class
        if tier["source_policy"] not in valid_source_policies:
            raise ValueError(
                f"parameter_summary.materials.bom.tiers[{i}].source_policy must be one of "
                f"{sorted(valid_source_policies)}"
            )
        if "examples" in tier:
            examples = tier["examples"]
            if not isinstance(examples, list):
                raise ValueError(
                    f"parameter_summary.materials.bom.tiers[{i}].examples must be a list"
                )
        if "chi" in tier:
            chi = tier["chi"]
            if "min" not in chi or "max" not in chi:
                raise KeyError(
                    f"parameter_summary.materials.bom.tiers[{i}].chi requires min/max"
                )
            _ = chi["min"]
            _ = chi["max"]
        resources = tier.get("resources", [])
        if not resources:
            raise ValueError(
                f"parameter_summary.materials.bom.tiers[{i}].resources cannot be empty"
            )
        unknown = [r for r in resources if r not in resource_ids]
        if unknown:
            raise ValueError(
                f"parameter_summary.materials.bom.tiers[{i}].resources contains unknown ids: {unknown}"
            )
        for share_key in [
            "share_initial",
            "share_final",
            "share_final_min",
            "share_final_max",
        ]:
            if share_key in tier:
                share_val = tier[share_key]
                if not isinstance(share_val, (int, float)) or not (0 <= share_val <= 1):
                    raise ValueError(
                        f"parameter_summary.materials.bom.tiers[{i}].{share_key} must be in [0,1]"
                    )

    covered_resources = set()
    for tier in tiers:
        covered_resources.update(tier.get("resources", []))
    missing_resources = resource_ids - covered_resources
    if missing_resources:
        raise ValueError(
            "parameter_summary.materials.bom.tiers must cover all resources. Missing: "
            f"{sorted(missing_resources)}"
        )

    # Transport validation
    transport = summary.get("transport")
    if transport is None:
        raise KeyError("parameter_summary.transport is required")
    capacities = transport.get("capacities")
    if capacities is None:
        raise KeyError("parameter_summary.transport.capacities is required")
    rocket_capacity = capacities.get("rocket_capacity")
    if rocket_capacity is None:
        raise KeyError(
            "parameter_summary.transport.capacities.rocket_capacity is required"
        )
    for key in [
        "growth_model",
        "year_2050_single_launch_capacity_mt",
        "max_capacity_mt",
    ]:
        if key not in rocket_capacity:
            raise KeyError(
                f"parameter_summary.transport.capacities.rocket_capacity.{key} is required"
            )
    _ = rocket_capacity["growth_model"]
    _ = rocket_capacity["max_capacity_mt"]
    year_2050_cap = rocket_capacity["year_2050_single_launch_capacity_mt"]
    _ = year_2050_cap["min"]
    _ = year_2050_cap["max"]
    cost_model = transport.get("cost_model")
    if cost_model is None:
        raise KeyError("parameter_summary.transport.cost_model is required")
    cost_2050 = cost_model.get("year_2050_cost_usd_per_kg")
    if cost_2050 is None:
        raise KeyError(
            "parameter_summary.transport.cost_model.year_2050_cost_usd_per_kg is required"
        )
    _ = cost_2050["min"]
    _ = cost_2050["max"]
    if "elevator_capacity_upper_tpy" not in capacities:
        raise KeyError(
            "parameter_summary.transport.capacities.elevator_capacity_upper_tpy is required"
        )
    _ = cost_model.get("launch_cost_model")

    # ISRU bootstrapping validation
    isru = summary.get("isru")
    if isru is None:
        raise KeyError("parameter_summary.isru is required")
    isru_boot = isru.get("bootstrapping")
    if isru_boot is None:
        raise KeyError("parameter_summary.isru.bootstrapping is required")
    for key in [
        "eta",
        "phi",
        "alpha_per_year",
        "beta",
        "gamma",
        "seed_input_source",
        "mode",
    ]:
        if key not in isru_boot:
            raise KeyError(f"parameter_summary.isru.bootstrapping.{key} is required")
    eta = isru_boot["eta"]
    _ = eta.get("min")
    _ = eta.get("max")
    phi = isru_boot["phi"]
    phi_initial = phi.get("initial", {})
    _ = phi_initial.get("min")
    _ = phi_initial.get("max")
    _ = phi.get("mature")
    alpha = isru_boot["alpha_per_year"]
    _ = alpha.get("min")
    _ = alpha.get("max")
    beta = isru_boot["beta"]
    _ = beta.get("min")
    _ = beta.get("max")
    gamma = isru_boot["gamma"]
    _ = gamma.get("note")
    _ = gamma.get("magnitude")
    _ = isru_boot.get("seed_input_source")
    _ = isru_boot.get("mode")

    # ISRU yield validation
    yields = isru.get("yields")
    if yields is None:
        raise KeyError("parameter_summary.isru.yields is required")
    _ = yields["regolith_to_oxygen"]
    _ = yields["regolith_to_silicon"]
    _ = yields["regolith_to_aluminum"]
    _ = yields["regolith_to_slag"]
    _ = yields["tier3_combined_yield"]
    tier12 = yields["tier1_2_late_stage_yield"]
    _ = tier12.get("min")
    _ = tier12.get("max")

    # Colony target & phase timeline validation
    colony = summary.get("colony")
    if colony is None:
        raise KeyError("parameter_summary.colony is required")
    target = colony.get("target")
    if target is None or "population" not in target:
        raise KeyError("parameter_summary.colony.target.population is required")
    phases = colony.get("phases")
    if phases is None:
        raise KeyError("parameter_summary.colony.phases is required")
    _ = phases.get("pivot_condition")
    if "timeline" not in phases:
        raise KeyError("parameter_summary.colony.phases.timeline is required")
    timeline = phases.get("timeline", [])
    if not timeline:
        raise ValueError("parameter_summary.colony.phases.timeline cannot be empty")
    for i, phase in enumerate(timeline):
        if "phase" not in phase or "years" not in phase:
            raise KeyError(
                f"parameter_summary.colony.phases.timeline[{i}] requires phase/years"
            )
        _ = phase.get("description")
        _ = phase["phase"]
        years = phase["years"]
        if "start" not in years:
            raise KeyError(
                f"parameter_summary.colony.phases.timeline[{i}].years.start is required"
            )
        _ = years["start"]
        _ = years.get("end")

    # Environment validation
    environment = summary.get("environment")
    if environment is None:
        raise KeyError("parameter_summary.environment is required")
    transport_constraint = environment.get("transport_constraint")
    if transport_constraint is None:
        raise KeyError("parameter_summary.environment.transport_constraint is required")
    _ = transport_constraint.get("index")
    _ = transport_constraint.get("black_carbon_weight")
    energy_pivot = environment.get("energy_transport_pivot")
    if energy_pivot is None:
        raise KeyError("parameter_summary.environment.energy_transport_pivot is required")
    _ = energy_pivot.get("local_energy_cost_usd_per_kwh")
    _ = energy_pivot.get("energy_intensity_kwh_per_kg")
    _ = energy_pivot.get("pivot_condition")
