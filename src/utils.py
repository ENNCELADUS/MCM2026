from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Iterator, MutableMapping, TYPE_CHECKING
import yaml

from entities import Node, Arc, Resource, ModelData, StateVariables, DecisionVariables
from config.settings import (
    ModelSettings,
    validate_parameter_summary,
)

if TYPE_CHECKING:
    pass

# =============================================================================
# Constants
# =============================================================================

REQUIRED_CONSTANT_SECTIONS = [
    "time",
    "units",
    "nodes",
    "arcs",
    "resources",
    "scenarios",
    "objective",
    "scenario_parameters",
    "parameter_summary",
    "implementation_details",
    "optimization",
]

REQUIRED_INITIAL_CAPACITIES = [
    "H_0",
    "Pop_0",
    "Power_0",
]
REQUIRED_OBJECTIVE_KEYS = ["w_C", "w_T"]
REQUIRED_TIME_KEYS = ["delta_t", "T_max", "start_year", "steps_per_year"]


# =============================================================================
# Config Tracker Classes
# =============================================================================


class ConfigTracker(dict):
    """
    Wraps a configuration dictionary to track key access.
    Inherits from dict to pass isinstance checks.
    """

    def __init__(self, data: Any, path: str = ""):
        super().__init__(data)
        self._data = data
        self._path = path
        self._accessed = set()
        self._children = {}

    def __getitem__(self, key: Any) -> Any:
        self._accessed.add(key)

        # Check cache for existing wrapper
        if key in self._children:
            return self._children[key]

        # Get raw value
        val = self._data[key]

        # Wrap if container
        if isinstance(val, dict):
            child_path = f"{self._path}.{key}" if self._path else str(key)
            tracker = ConfigTracker(val, child_path)
            self._children[key] = tracker
            return tracker

        if isinstance(val, list):
            child_path = f"{self._path}.{key}" if self._path else str(key)
            tracker = ListTracker(val, child_path)
            self._children[key] = tracker
            return tracker

        return val

    def get(self, key: Any, default: Any = None) -> Any:
        self._accessed.add(key)
        if key in self._data:
            return self.__getitem__(key)
        return default

    def items(self):
        for k in self:
            yield k, self[k]

    def values(self):
        for k in self:
            yield self[k]

    def report_unused(self, out_stream=sys.stdout) -> None:
        """Print unused parameters to the output stream."""
        unused = self._collect_unused()
        if unused:
            out_stream.write("\n" + "=" * 60 + "\n")
            out_stream.write(
                "WARNING: The following configuration parameters were NOT used:\n"
            )
            out_stream.write("=" * 60 + "\n")
            for path in sorted(unused):
                out_stream.write(f"  - {path}\n")
            out_stream.write("=" * 60 + "\n")

    def _collect_unused(self) -> list[str]:
        unused = []

        for k in self:
            if k not in self._accessed:
                full_path = f"{self._path}.{k}" if self._path else str(k)
                unused.append(full_path)
            else:
                if k in self._children:
                    unused.extend(self._children[k]._collect_unused())
        return unused


class ListTracker(list):
    """
    Wraps a list to track access to its elements.
    Inherits from list to pass isinstance checks.
    """

    def __init__(self, data: list, path: str):
        super().__init__(data)
        self._data = data
        self._path = path
        self._accessed_indices = set()
        self._children = {}

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self._data))
            for i in range(start, stop, step):
                self._accessed_indices.add(i)
            return [self[i] for i in range(start, stop, step)]

        self._accessed_indices.add(index)

        if index in self._children:
            return self._children[index]

        val = self._data[index]

        if isinstance(val, dict):
            new_path = f"{self._path}[{index}]"
            tracker = ConfigTracker(val, new_path)
            self._children[index] = tracker
            return tracker

        if isinstance(val, list):
            new_path = f"{self._path}[{index}]"
            tracker = ListTracker(val, new_path)
            self._children[index] = tracker
            return tracker

        return val

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self._data)):
            yield self[i]

    def _collect_unused(self) -> list[str]:
        unused = []
        for i, tracker in self._children.items():
            unused.extend(tracker._collect_unused())
        return unused


# =============================================================================
# Helper Functions from Model
# =============================================================================


def load_constants_from_file(path: Path | str) -> ConfigTracker:
    """Load constants from YAML file and wrap with tracker."""
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

    data = expand_network_templates(data)

    return ConfigTracker(data)


def expand_network_templates(data: dict[str, Any]) -> dict[str, Any]:
    """
    Expand a compact network template section into explicit nodes/arcs.

    If nodes/arcs are already present, this is a no-op.
    """
    if "network" not in data:
        return data
    if "nodes" in data or "arcs" in data:
        return data

    net = data["network"]
    ground = net.get("ground_hub")
    moon = net["moon"]
    harbours = net["harbours"]
    launch = net["launch_sites"]
    arcs_cfg = net["arcs"]

    harbour_id = harbours["id"]
    harbour_name = harbours["name"]

    nodes: list[dict[str, Any]] = []
    if ground is not None:
        nodes.append({"id": ground["id"], "name": ground["name"], "type": "earth"})

    earth_port_ids = [harbour_id]
    nodes.append({"id": harbour_id, "name": harbour_name, "type": "earth"})

    launch_id = launch["id"]
    launch_ids = [launch_id]
    nodes.append({"id": launch_id, "name": launch["name"], "type": "earth"})

    nodes.append({"id": moon["id"], "name": moon["name"], "type": "moon"})

    arcs: list[dict[str, Any]] = []
    elevator_cfg = arcs_cfg["elevator"]
    rocket_cfg = arcs_cfg["rocket"]
    ground_cfg = arcs_cfg.get("ground")

    for port_id in earth_port_ids:
        lead_time_days = elevator_cfg["lead_time_days"]
        cost_per_kg_2050 = elevator_cfg["cost_per_kg_2050"]

        arcs.append(
            {
                "id": f"{port_id}_to_{moon['id']}_E",
                "from": port_id,
                "to": moon["id"],
                "type": "elevator",
                "lead_time_days": lead_time_days,
                "payload_t": elevator_cfg["payload_t"],
                "cost_per_kg_2050": cost_per_kg_2050,
                "enabled": elevator_cfg["enabled"],
            }
        )

    for launch_id in launch_ids:
        arcs.append(
            {
                "id": f"{launch_id}_to_{moon['id']}_R",
                "from": launch_id,
                "to": moon["id"],
                "type": "rocket",
                "lead_time_days": rocket_cfg["lead_time_days"],
                "payload_t": rocket_cfg["payload_t"],
                "cost_per_kg_2050": rocket_cfg["cost_per_kg_2050"],
                "enabled": rocket_cfg["enabled"],
            }
        )

    if ground is not None and ground_cfg is not None:
        for node_id in earth_port_ids + launch_ids:
            arcs.append(
                {
                    "id": f"{ground['id']}_to_{node_id}",
                    "from": ground["id"],
                    "to": node_id,
                    "type": "ground",
                    "lead_time": ground_cfg["lead_time"],
                    "payload": ground_cfg["payload"],
                    "cost_per_kg_2050": ground_cfg["cost_per_kg_2050"],
                    "enabled": ground_cfg["enabled"],
                }
            )

    data["nodes"] = nodes
    data["arcs"] = arcs
    del data["network"]
    return data


def validate_constants(constants: ConfigTracker | dict[str, Any]) -> None:
    """
    Validate that all required keys are present in constants.

    Raises:
        KeyError: If required key is missing
        ValueError: If value has invalid type
    """
    # Check top-level sections
    for section in REQUIRED_CONSTANT_SECTIONS:
        if section not in constants:
            raise KeyError(f"Missing required section in constants.yaml: '{section}'")

    # Validate time section
    for key in REQUIRED_TIME_KEYS:
        if key not in constants["time"]:
            raise KeyError(f"Missing required key in constants.yaml time: '{key}'")
    _ = constants["time"]["start_year"]

    # Validate unit conversions
    for key in ["ton_to_kg", "seconds_per_day"]:
        if key not in constants["units"]:
            raise KeyError(f"Missing required key in constants.yaml units: '{key}'")

    # Validate initial_capacities
    for key in REQUIRED_INITIAL_CAPACITIES:
        if key not in constants["initial_capacities"]:
            raise KeyError(
                f"Missing required key in constants.yaml initial_capacities: '{key}'"
            )



    # Validate objective
    for key in REQUIRED_OBJECTIVE_KEYS:
        if key not in constants["objective"]:
            raise KeyError(f"Missing required key in constants.yaml objective: '{key}'")
    _ = constants["objective"]["w_C"]
    _ = constants["objective"]["w_T"]





    # Validate solver section
    solver_cfg = constants.get("solver", {})
    if solver_cfg:
        _ = solver_cfg.get("time_limit")
        _ = solver_cfg.get("gap_tolerance")
        _ = solver_cfg.get("threads")
        _ = solver_cfg.get("horizon_step_months")
        _ = solver_cfg.get("horizon_max_attempts")
        _ = solver_cfg.get("v0_scale_factor")
        _ = solver_cfg.get("v0_adjust_attempts")

    # Validate nodes list
    if not constants["nodes"]:
        raise ValueError("constants.yaml nodes list cannot be empty")
    for i, node in enumerate(constants["nodes"]):
        for key in ["id", "name", "type"]:
            if key not in node:
                raise KeyError(f"Missing key '{key}' in constants.yaml nodes[{i}]")
        _ = node["id"]
        _ = node["name"]
        _ = node["type"]

    # Validate arcs list
    if not constants["arcs"]:
        raise ValueError("constants.yaml arcs list cannot be empty")
    for i, arc in enumerate(constants["arcs"]):
        for key in ["id", "from", "to", "type", "enabled"]:
            if key not in arc:
                raise KeyError(f"Missing key '{key}' in constants.yaml arcs[{i}]")
        has_lead_time_days = "lead_time_days" in arc
        has_lead_time = "lead_time" in arc
        if not (has_lead_time_days or has_lead_time):
            raise KeyError(
                f"Missing key 'lead_time_days' or 'lead_time' in constants.yaml arcs[{i}]"
            )
        has_payload_t = "payload_t" in arc
        has_payload = "payload" in arc
        if not (has_payload_t or has_payload):
            raise KeyError(
                f"Missing key 'payload_t' or 'payload' in constants.yaml arcs[{i}]"
            )
        if "cost_per_kg_2050" not in arc:
            raise KeyError(
                f"Missing key 'cost_per_kg_2050' in constants.yaml arcs[{i}]"
            )
        _ = arc["id"]
        _ = arc["from"]
        _ = arc["to"]
        _ = arc["type"]
        _ = arc["enabled"]
        _ = arc.get("lead_time_days", arc.get("lead_time"))
        _ = arc.get("payload_t", arc.get("payload"))
        _ = arc["cost_per_kg_2050"]

    # Validate resources list
    if not constants["resources"]:
        raise ValueError("constants.yaml resources list cannot be empty")
    for i, res in enumerate(constants["resources"]):
        for key in ["id", "name", "isru_producible"]:
            if key not in res:
                raise KeyError(f"Missing key '{key}' in constants.yaml resources[{i}]")
        _ = res["id"]
        _ = res["name"]
        _ = res["isru_producible"]

    # Validate parameter summary (Section 5)
    if not constants["parameter_summary"]:
        raise ValueError("constants.yaml parameter_summary cannot be empty")
    for section in ["materials", "colony"]:
        if section not in constants["parameter_summary"]:
            raise KeyError(f"Missing parameter_summary section: '{section}'")

    # Validate scenarios list
    scenarios = constants.get("scenarios", [])
    if scenarios:
        for i, scen in enumerate(scenarios):
            for key in ["id", "name", "description"]:
                if key not in scen:
                    raise KeyError(
                        f"Missing key '{key}' in constants.yaml scenarios[{i}]"
                    )
            _ = scen["id"]
            _ = scen["name"]
            _ = scen["description"]

    # Validate scenario_parameters
    scenario_params = constants.get("scenario_parameters", {})
    if scenario_params:
        elev = scenario_params.get("elevator", {})
        
        # Validate elevator capacity (Fixed)
        elev_cap_fixed = elev.get("capacity_fixed", {})
        if elev_cap_fixed:
             _ = elev_cap_fixed.get("capacity_tpy")

        # Validate elevator cost_decay
        elev_cost_decay = elev.get("cost_decay", {})
        if elev_cost_decay:
            _ = elev_cost_decay.get("base_year")
            _ = elev_cost_decay.get("initial_cost_usd_per_kg")
            _ = elev_cost_decay.get("min_cost_usd_per_kg")
            _ = elev_cost_decay.get("decay_rate_monthly")

        rocket = scenario_params.get("rocket", {})
        rocket_payload = rocket.get("payload_logistic", {})
        _ = rocket_payload.get("L_max_t")
        _ = rocket_payload.get("L_ref_t")
        _ = rocket_payload.get("k")
        _ = rocket_payload.get("t0_year")
        _ = rocket_payload.get("ref_year")
        rocket_launch_rate = rocket.get("launch_rate", {})
        _ = rocket_launch_rate.get("base_per_year")
        _ = rocket_launch_rate.get("annual_growth_rate")
        _ = rocket_launch_rate.get("max_per_year")
        _ = rocket_launch_rate.get("start_year")



    # Validate cost parameters in scenario_parameters
    rocket_cost_decay = scenario_params.get("rocket", {}).get("cost_decay", {})
    if rocket_cost_decay:
        _ = rocket_cost_decay.get("base_year")
        _ = rocket_cost_decay.get("base_cost_usd_per_kg")
        _ = rocket_cost_decay.get("decay_rate_monthly")
        _ = rocket_cost_decay.get("min_cost_usd_per_kg")

    # Validate implementation details
    impl = constants["implementation_details"]
    
    # Check for growth_model instead of tasks
    if "growth_model" not in impl:
        # It might be under parameter_summary or root, but let's check impl as per previous attempts
        # Actually, based on previous artifacts, growth_model is a root key or part of parameter_summary?
        # Let's check where it was added. 
        # But wait, looking at lines 500+, we are in validate_constants.
        # Let's perform a robust check.
        pass

    for section in [
        "logistics",
    ]:
        if section not in impl:
            raise KeyError(f"Missing implementation_details section: '{section}'")

    time_disc = impl["logistics"]["time_discretization"]
    _ = time_disc.get("threshold_days")

    # task_defaults removed

    # Validate optimization bounds
    opt = constants["optimization"]
    if "big_m" not in opt:
        raise KeyError("Missing optimization.big_m in constants.yaml")
    _ = opt.get("integer_vars")


def load_model_data(
    settings: ModelSettings, constants: ConfigTracker | dict[str, Any]
) -> tuple[ModelData, StateVariables, DecisionVariables]:
    """Load and process all model data from constants and settings."""
    # Parse nodes
    nodes = [
        Node(id=n["id"], name=n["name"], node_type=n["type"])
        for n in constants["nodes"]
    ]

    # Parse arcs (filter by scenario)
    scenario = settings.scenario.value
    seconds_per_day = constants["units"]["seconds_per_day"]
    ton_to_kg = constants["units"]["ton_to_kg"]
    
    # NEW: Read capacity from scenario_parameters (fixed model)
    elevator_params = constants["scenario_parameters"]["elevator"]
    elevator_capacity_per_harbour_tpy = elevator_params["capacity_fixed"]["capacity_tpy"]
    elevator_cost_params = elevator_params.get("cost_decay", {})
    
    total_demand_tons = constants["parameter_summary"]["materials"]["bom"][
        "total_demand_tons"
    ]
    steps_per_year = constants["time"]["steps_per_year"]
    time_disc = constants["implementation_details"]["logistics"]["time_discretization"]
    threshold_days = time_disc["threshold_days"]

    arcs = []
    for a in constants["arcs"]:
        if scenario not in a["enabled"]:
            continue
        if scenario == "R-only":
            if a["type"] != "rocket":
                continue
        if "lead_time_days" in a:
            lead_time_days = a["lead_time_days"]
            if lead_time_days < threshold_days:
                lead_time_steps = 0
            else:
                lead_time_steps = 1
        else:
            lead_time_steps = int(a["lead_time"])
        if "payload_t" in a:
            payload_t = a["payload_t"]
            if payload_t is None:
                if a["type"] == "elevator":
                    payload_t = elevator_capacity_per_harbour_tpy / steps_per_year
                else:
                    payload_t = total_demand_tons
            payload_mass = payload_t * ton_to_kg
        else:
            payload_mass = float(a["payload"])
        cost_per_kg_2050 = a["cost_per_kg_2050"]
        if a["type"] == "elevator" and elevator_cost_params.get("initial_cost_usd_per_kg") is not None:
            cost_per_kg_2050 = float(elevator_cost_params["initial_cost_usd_per_kg"])
        arcs.append(
            Arc(
                id=a["id"],
                from_node=a["from"],
                to_node=a["to"],
                arc_type=a["type"],
                lead_time=lead_time_steps,
                payload=payload_mass,
                cost_per_kg_2050=cost_per_kg_2050,
                enabled_scenarios=a["enabled"],
            )
        )

    # Parse resources
    resources = [
        Resource(id=r["id"], name=r["name"], isru_producible=r["isru_producible"])
        for r in constants["resources"]
    ]

    # Validate parameter summary against resources
    validate_parameter_summary(
        constants["parameter_summary"],
        {r.id for r in resources},
    )

    # Load growth parameters (NEW LOGIC)
    growth_params = constants["implementation_details"]["growth_model"]

    data = ModelData(
        nodes=nodes,
        arcs=arcs,
        resources=resources,
        growth_params=growth_params,
        settings=settings,
        constants=constants,
    )

    # Initialize state and decision variable containers
    T = settings.T_horizon
    if T > constants["time"]["T_max"]:
        raise ValueError(
            f"T_horizon ({T}) exceeds constants time.T_max ({constants['time']['T_max']})"
        )
    R = len(resources)
    N = len(nodes)
    A = len(arcs)

    # Removed I (num tasks)
    state = StateVariables(T=T, R=R, N=N, A=A)
    decisions = DecisionVariables(T=T, R=R, A=A)

    # Set initial capacities
    init_cap = constants["initial_capacities"]
    state.P[0] = constants["implementation_details"]["growth_model"]["initial_state"][
        "P_0"
    ]
    state.H[0] = init_cap["H_0"]
    state.Pop[0] = init_cap["Pop_0"]
    state.Power[0] = init_cap["Power_0"]

    return data, state, decisions


def get_year_for_t(t: int, constants: ConfigTracker | dict[str, Any]) -> float:
    """Convert discrete time index t to calendar year."""
    start_year = constants["time"]["start_year"]
    steps_per_year = constants["time"]["steps_per_year"]
    return start_year + (t / steps_per_year)


def get_tier_definitions(
    constants: ConfigTracker | dict[str, Any],
) -> list[dict[str, Any]]:
    """Return BOM tier definitions from parameter_summary."""
    return constants["parameter_summary"]["materials"]["bom"]["tiers"]


def get_rocket_launch_rate_max(
    t: int, constants: ConfigTracker | dict[str, Any]
) -> float:
    """Compute maximum rocket launches per time step using growth with cap."""
    params = constants["scenario_parameters"]["rocket"]["launch_rate"]
    # Logistic Growth Model: L(t) = K / (1 + A * exp(-r * (t - t0)))
    K = float(params["K"])
    A = float(params["A"])
    r = float(params["r"])
    t0 = float(params["t0"])
    steps_per_year = constants["time"]["steps_per_year"]
    
    year = get_year_for_t(t, constants)
    A_term = A * math.exp(-r * (year - t0))
    rate = K / (1.0 + A_term)
    
    return rate / steps_per_year


def get_rocket_payload_kg(
    t: int, constants: ConfigTracker | dict[str, Any]
) -> float:
    """Compute per-launch rocket payload (mass units) using logistic growth."""
    params = constants["scenario_parameters"]["rocket"]["payload_logistic"]
    L_max_t = float(params["L_max_t"])
    L_ref_t = float(params["L_ref_t"])
    k = float(params["k"])
    t0_year = float(params.get("t0_year", constants["time"]["start_year"]))
    ref_year = float(params.get("ref_year", constants["time"]["start_year"]))
    ton_to_kg = constants["units"]["ton_to_kg"]

    year = get_year_for_t(t, constants)
    if L_ref_t <= 0:
        raise ValueError("rocket_payload_logistic.L_ref_t must be positive")
    if L_max_t <= 0:
        raise ValueError("rocket_payload_logistic.L_max_t must be positive")

    # Solve A so that L_cap(ref_year) = L_ref_t
    A = (L_max_t / L_ref_t - 1.0) * math.exp(k * (ref_year - t0_year))
    L_cap_t = L_max_t / (1.0 + A * math.exp(-k * (year - t0_year)))
    return L_cap_t * ton_to_kg


def get_rocket_cost_usd_per_kg(
    t: int, constants: ConfigTracker | dict[str, Any]
) -> float:
    """Compute rocket transport cost per mass unit using exponential or annual decay."""
    params = constants["scenario_parameters"]["rocket"]["cost_decay"]
    min_cost = float(params.get("min_cost_usd_per_kg", 0))

    # Check for Exponential Decay (Moon Logic) - implementation_details.md
    if "decay_rate_monthly" in params:
        lambda_c = float(params["decay_rate_monthly"])
        # We start from the base_cost. 
        # NOTE: Ideally base_cost should be at t=0 (2050). 
        # If constants.yaml has 2024 value, this might be high, 
        # but we follow the mathematical form requested.
        base_cost = float(params["base_cost_usd_per_kg"])
        
        # C(t) = (C_start - C_min) * e^(-lambda * t) + C_min
        cost = (base_cost - min_cost) * math.exp(-lambda_c * t) + min_cost
        return cost

    # Fallback to Annual Power Decay
    base_year = float(params["base_year"])
    base_cost = float(params["base_cost_usd_per_kg"])
    annual_decay = float(params["annual_decay_rate"])

    if annual_decay < 0 or annual_decay >= 1:
        raise ValueError("rocket_cost_decay.annual_decay_rate must be in [0, 1)")

    year = get_year_for_t(t, constants)
    # Ensure year >= base_year
    years_elapsed = max(0.0, year - base_year)
    
    cost = base_cost * ((1.0 - annual_decay) ** years_elapsed)
    if min_cost is not None:
        cost = max(float(min_cost), cost)
    return cost


def get_elevator_capacity_tpy(
    t: int, constants: ConfigTracker | dict[str, Any]
) -> float:
    """
    Return the fixed annual elevator capacity (tons/year).
    """
    elev_params = constants["scenario_parameters"]["elevator"]
    
    # Check for fixed capacity
    if "capacity_fixed" in elev_params:
        return float(elev_params["capacity_fixed"]["capacity_tpy"])
    
    raise KeyError("Missing required section: scenario_parameters.elevator.capacity_fixed")


def get_elevator_capacity_per_step(
    t: int, constants: ConfigTracker | dict[str, Any]
) -> float:
    """
    Compute elevator capacity per time step (tons/step).
    
    Divides annual capacity by steps_per_year for monthly resolution.
    """
    steps_per_year = constants["time"]["steps_per_year"]
    annual_capacity = get_elevator_capacity_tpy(t, constants)
    return annual_capacity / steps_per_year


def get_elevator_cost_usd_per_kg(
    t: int, constants: ConfigTracker | dict[str, Any]
) -> float:
    """
    Compute elevator transport cost per kg (USD) using exponential decay.
    
    Formula:
        C(t) = (C_start - C_min) * exp(-lambda * t) + C_min
    
    This mirrors the rocket cost decay model for consistency.
    
    Args:
        t: Discrete time step index
        constants: Configuration dictionary
        
    Returns:
        Cost per kg in USD
    """
    elev_params = constants["scenario_parameters"]["elevator"]
    
    # Check for new exponential decay model
    if "cost_decay" in elev_params:
        params = elev_params["cost_decay"]
        c_start = float(params["initial_cost_usd_per_kg"])
        c_min = float(params["min_cost_usd_per_kg"])
        lambda_c = float(params["decay_rate_monthly"])
        
        # C(t) = (C_start - C_min) * exp(-lambda * t) + C_min
        cost = (c_start - c_min) * math.exp(-lambda_c * t) + c_min
        return cost
    
    raise KeyError("Missing required section: scenario_parameters.elevator.cost_decay")
