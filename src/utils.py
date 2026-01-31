from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Iterator, MutableMapping, TYPE_CHECKING
import yaml

from entities import Node, Arc, Resource, ModelData, StateVariables, DecisionVariables
from config.settings import (
    ModelSettings,
    TaskDefinition,
    build_task_network_from_wbs,
    validate_task_network,
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
    "costs",
    "initial_capacities",
    "task_defaults",
    "objective",
    "scenario_parameters",
    "parameter_summary",
    "implementation_details",
    "optimization",
]

REQUIRED_INITIAL_CAPACITIES = [
    "P_0",
    "V_0",
    "H_0",
    "Pop_0",
    "Power_0",
    "C_R_0",
    "C_E_0",
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

    return ConfigTracker(data)


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
    _ = constants["initial_capacities"]["C_R_0"]
    _ = constants["initial_capacities"]["C_E_0"]

    # Validate objective
    for key in REQUIRED_OBJECTIVE_KEYS:
        if key not in constants["objective"]:
            raise KeyError(f"Missing required key in constants.yaml objective: '{key}'")
    _ = constants["objective"]["w_C"]
    _ = constants["objective"]["w_T"]

    # Validate costs section (touch keys for audit)
    costs = constants["costs"]
    _ = costs["elevator"]["initial"]
    _ = costs["elevator"]["mature"]
    _ = costs["elevator"]["fixed_per_trip"]
    _ = costs["rocket"]["falcon_heavy"]["per_kg"]
    _ = costs["rocket"]["falcon_heavy"]["fixed_per_launch"]
    _ = costs["rocket"]["starship"]["per_kg_initial"]
    _ = costs["rocket"]["starship"]["per_kg_target"]
    _ = costs["rocket"]["starship"]["fixed_per_launch"]

    # Validate learning section
    learning = constants["learning"]
    _ = learning["elevator_rate"]
    _ = learning["rocket_rate"]
    _ = learning["isru_rate"]

    # Validate ISRU section
    isru = constants["isru"]
    _ = isru["water_extraction_rate"]
    _ = isru["oxygen_from_ilmenite"]
    _ = isru["metal_from_regolith"]
    _ = isru["water_energy"]
    _ = isru["oxygen_energy"]
    _ = isru["metal_energy"]

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
    for section in ["bom", "logistics", "isru_bootstrapping", "scenarios"]:
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
        _ = elev.get("climber_payload_t")
        throughput_tpy = elev.get("annual_throughput_tpy", {})
        _ = throughput_tpy.get("initial")
        _ = throughput_tpy.get("mature")
        _ = throughput_tpy.get("six_tethers")
        throughput_kg = elev.get("annual_throughput_kg_per_s", {})
        _ = throughput_kg.get("initial")
        _ = throughput_kg.get("mature")
        _ = throughput_kg.get("six_tethers")
        _ = elev.get("transit_time_days")
        _ = elev.get("climber_velocity_kmh")
        req_vel = elev.get("required_avg_velocity", {})
        _ = req_vel.get("mps")
        _ = req_vel.get("kmh")
        _ = elev.get("power_limit_mw")
        strength = elev.get("material_specific_strength_gpa_cc_g", {})
        _ = strength.get("min")
        _ = strength.get("max")
        _ = elev.get("construction_cost_usd")
        cost_per_kg = elev.get("cost_per_kg_usd", {})
        _ = cost_per_kg.get("initial")
        _ = cost_per_kg.get("mature")

        rocket = scenario_params.get("rocket", {})
        fh_payload = rocket.get("falcon_heavy_payload_tli_t", {})
        _ = fh_payload.get("min")
        _ = fh_payload.get("max")
        _ = rocket.get("falcon_heavy_cost_usd")
        starship_cost = rocket.get("starship_cost_per_kg_usd", {})
        _ = starship_cost.get("upper")
        _ = starship_cost.get("target")
        _ = rocket.get("delta_v_leo_to_moon_kms")

    # Validate implementation details
    impl = constants["implementation_details"]
    if "wbs_tasks" not in impl or not impl["wbs_tasks"]:
        raise ValueError("implementation_details.wbs_tasks cannot be empty")
    for section in [
        "handling_capacity",
        "isru_yields",
        "bom_mapping",
        "linearization_guidance",
        "additional_parameters",
    ]:
        if section not in impl:
            raise KeyError(f"Missing implementation_details section: '{section}'")

    # Touch implementation detail parameters for audit/validation
    handling = impl["handling_capacity"]
    _ = handling["excavator_efficiency_t_per_hour_per_ton"]
    _ = handling["handling_t_per_month_per_ton_excavator"]

    yields = impl["isru_yields"]
    _ = yields["regolith_to_oxygen"]
    _ = yields["regolith_to_silicon"]
    _ = yields["regolith_to_aluminum"]
    _ = yields["regolith_to_slag"]
    _ = yields["tier3_combined_yield"]
    tier12 = yields["tier1_2_late_stage_yield"]
    _ = tier12.get("min")
    _ = tier12.get("max")

    bom_map = impl["bom_mapping"]
    _ = bom_map["total_demand_tons"]
    for category in bom_map.get("categories", []):
        _ = category.get("id")
        _ = category.get("share")
        _ = category.get("source_policy")
        _ = category.get("earth_share_initial")
        _ = category.get("isru_share_final")
        _ = category.get("earth_share")
        _ = category.get("isru_share")
        _ = category.get("notes")

    lin = impl["linearization_guidance"]
    _ = lin["total_mass_equation"]
    _ = lin["bottleneck_note"]

    additional = impl["additional_parameters"]
    arc_sel = additional["arc_selection"]
    _ = arc_sel.get("simplified_single_arc")
    _ = arc_sel.get("arc_rocket_id")
    _ = arc_sel.get("arc_elevator_id")

    elev_stream = additional["elevator_stream_model"]
    _ = elev_stream.get("enabled")
    _ = elev_stream.get("constraint")

    rocket_growth = additional["rocket_capacity_growth"]
    _ = rocket_growth.get("baseline_tpy")
    _ = rocket_growth.get("annual_growth_rate")
    _ = rocket_growth.get("formula")

    wrights = additional["wrights_law_cost"]
    _ = wrights.get("c_start_usd_per_kg")
    _ = wrights.get("c_min_usd_per_kg")
    _ = wrights.get("lambda_per_month")
    _ = wrights.get("formula")

    isru_boot = additional["isru_bootstrapping"]
    _ = isru_boot.get("seed_input_source")
    _ = isru_boot.get("mode")

    task_duration = additional["task_duration"]
    _ = task_duration.get("fixed_setup_time")
    _ = task_duration.get("note")

    handle_cfg = additional["handling_capacity"]
    _ = handle_cfg.get("tied_to_tasks")
    _ = handle_cfg.get("note")

    inventory = additional["inventory_policy"]
    _ = inventory.get("unlimited_prepositioning")
    _ = inventory.get("note")

    pivot = additional["energy_transport_pivot"]
    _ = pivot.get("local_energy_cost_usd_per_kwh")
    _ = pivot.get("energy_intensity_kwh_per_kg")
    _ = pivot.get("local_cost_per_kg_formula")
    _ = pivot.get("pivot_condition")

    time_disc = additional["time_discretization"]
    _ = time_disc.get("steps_per_month")
    _ = time_disc.get("threshold_days")
    _ = time_disc.get("arrival_rule")

    # Touch task_defaults
    task_defaults = constants["task_defaults"]
    _ = task_defaults["min_duration"]
    _ = task_defaults["max_concurrent_tasks"]
    _ = task_defaults["installation_delay"]

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
    elevator_capacity_upper_tpy = constants["parameter_summary"]["logistics"][
        "elevator_capacity_upper_tpy"
    ]
    total_demand_tons = constants["parameter_summary"]["bom"]["total_demand_tons"]
    steps_per_year = constants["time"]["steps_per_year"]
    time_disc = constants["implementation_details"]["additional_parameters"][
        "time_discretization"
    ]
    threshold_days = time_disc["threshold_days"]
    _ = time_disc.get("arrival_rule")
    _ = time_disc.get("steps_per_month")

    arcs = []
    for a in constants["arcs"]:
        if scenario not in a["enabled"]:
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
                    payload_t = elevator_capacity_upper_tpy / steps_per_year
                else:
                    payload_t = total_demand_tons
            payload_kg = payload_t * ton_to_kg
        else:
            payload_kg = float(a["payload"])
        arcs.append(
            Arc(
                id=a["id"],
                from_node=a["from"],
                to_node=a["to"],
                arc_type=a["type"],
                lead_time=lead_time_steps,
                payload=payload_kg,
                cost_per_kg_2050=a["cost_per_kg_2050"],
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

    # Get tasks from implementation details and validate
    tasks = build_task_network_from_wbs(
        constants["implementation_details"]["wbs_tasks"],
        constants["parameter_summary"],
        constants["units"],
        constants["time"],
    )
    validate_task_network(tasks)

    data = ModelData(
        nodes=nodes,
        arcs=arcs,
        resources=resources,
        tasks=tasks,
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
    I = len(tasks)

    state = StateVariables(T=T, R=R, N=N, A=A, I=I)
    decisions = DecisionVariables(T=T, R=R, A=A, I=I)

    # Set initial capacities
    init_cap = constants["initial_capacities"]
    state.P[0] = init_cap["P_0"]
    state.V[0] = init_cap["V_0"]
    state.H[0] = init_cap["H_0"]
    state.Pop[0] = init_cap["Pop_0"]
    state.Power[0] = init_cap["Power_0"]

    return data, state, decisions


def get_year_for_t(t: int, constants: ConfigTracker | dict[str, Any]) -> float:
    """Convert discrete time index t to calendar year."""
    start_year = constants["time"]["start_year"]
    steps_per_year = constants["time"]["steps_per_year"]
    return start_year + (t / steps_per_year)


def get_phase_time_ranges(
    settings: ModelSettings, constants: ConfigTracker | dict[str, Any]
) -> list[tuple[str, int, int]]:
    """
    Map scenario phases to time index ranges [start, end] (inclusive).

    Uses parameter_summary.scenarios.timeline and time.start_year.
    """
    start_year = constants["time"]["start_year"]
    steps_per_year = constants["time"]["steps_per_year"]
    timeline = constants["parameter_summary"]["scenarios"]["timeline"]

    ranges: list[tuple[str, int, int]] = []
    for phase in timeline:
        years = phase["years"]
        phase_start = years["start"]
        phase_end = years.get("end")
        t_start = max(0, int((phase_start - start_year) * steps_per_year))
        if phase_end is None:
            t_end = settings.T_horizon - 1
        else:
            t_end = min(
                settings.T_horizon - 1,
                int((phase_end - start_year) * steps_per_year) - 1,
            )
        if t_start <= t_end:
            ranges.append((phase["phase"], t_start, t_end))
    return ranges


def get_tier_definitions(
    constants: ConfigTracker | dict[str, Any],
) -> list[dict[str, Any]]:
    """Return BOM tier definitions from parameter_summary."""
    return constants["parameter_summary"]["bom"]["tiers"]


def get_rocket_capacity_kg_s(
    t: int, constants: ConfigTracker | dict[str, Any]
) -> float:
    """Compute rocket capacity C_R(t) in kg/s using additional parameters."""
    params = constants["implementation_details"]["additional_parameters"][
        "rocket_capacity_growth"
    ]
    baseline_tpy = params["baseline_tpy"]
    annual_growth_rate = params["annual_growth_rate"]
    _ = params.get("formula")
    steps_per_year = constants["time"]["steps_per_year"]
    ton_to_kg = constants["units"]["ton_to_kg"]
    delta_t = constants["time"]["delta_t"]
    capacity_tpy = baseline_tpy * ((1 + annual_growth_rate) ** (t / steps_per_year))
    capacity_kg_s = (capacity_tpy * ton_to_kg) / (steps_per_year * delta_t)
    C_R_0 = constants["initial_capacities"]["C_R_0"]
    return max(C_R_0, capacity_kg_s)


def get_rocket_cost_usd_per_kg(
    t: int, constants: ConfigTracker | dict[str, Any]
) -> float:
    """Compute rocket transport cost per kg using Wright's law curve."""
    params = constants["implementation_details"]["additional_parameters"][
        "wrights_law_cost"
    ]
    c_start = params["c_start_usd_per_kg"]
    c_min = params["c_min_usd_per_kg"]
    lam = params["lambda_per_month"]
    _ = params.get("formula")
    return (c_start - c_min) * math.exp(-lam * t) + c_min
