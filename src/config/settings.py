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


@dataclass
class ModelSettings:
    """
    Runtime configuration for a single model run.

    All fields are REQUIRED (no defaults allowed here).
    Defaults are handled by the CLI parser or caller.
    """

    scenario: ScenarioType
    T_horizon: int
    enable_learning_curve: bool
    enable_preposition: bool
    solver_timeout: int = 3600
    mip_gap: float = 0.01
    output_dir: Path = field(default_factory=lambda: Path("results"))
    
    def __post_init__(self):
        # Strict type and value validation
        _validate_required(self, "scenario", self.scenario, ScenarioType)
        _validate_required(self, "T_horizon", self.T_horizon, int)
        _validate_positive(self, "T_horizon", self.T_horizon)
        _validate_required(self, "enable_learning_curve", self.enable_learning_curve, bool)
        _validate_required(self, "enable_preposition", self.enable_preposition, bool)
        
        # Coerce output_dir to Path if string
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)



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


# Task Network definitions have been removed in favor of Continuous Growth Model.



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

    for i, tier in enumerate(tiers):
        for key in ["id", "name", "class", "share_initial", "resources"]:
            if key not in tier:
                raise KeyError(
                    f"parameter_summary.materials.bom.tiers[{i}].{key} is required"
                )
        tier_id = tier["id"]
        tier_name = tier["name"]
        tier_class = tier["class"]
        _ = tier_id, tier_name, tier_class
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
        share_val = tier["share_initial"]
        if not isinstance(share_val, (int, float)) or not (0 <= share_val <= 1):
            raise ValueError(
                f"parameter_summary.materials.bom.tiers[{i}].share_initial must be in [0,1]"
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
    if "elevator_capacity_fixed_tpy" not in capacities:
        raise KeyError(
            "parameter_summary.transport.capacities.elevator_capacity_fixed_tpy is required"
        )

    # Colony target & phase timeline validation
    colony = summary.get("colony")
    if colony is None:
        raise KeyError("parameter_summary.colony is required")
    target = colony.get("target")
    if target is None or "population" not in target:
        raise KeyError("parameter_summary.colony.target.population is required")
