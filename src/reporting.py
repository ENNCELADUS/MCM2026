from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import pyomo.environ as pyo

from config.settings import ScenarioType
import utils


def export_solution_report(model: Any, output_path: Path | str) -> dict[str, Any]:
    """
    Export extended solution insights for a single scenario.

    Writes:
      - solution_report.json
      - material_timeseries.csv (for river plot)
      - phase_gantt.csv (for Gantt chart)
      - solution_insights.tex (narrative snippet)
    """
    model._require_pyomo()
    if model._model is None:
        raise RuntimeError("No model available. Call build_model/solve first.")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    report, timeseries_rows, phase_rows = _build_solution_report(model)

    report_path = output_path / "solution_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    _write_timeseries_csv(output_path / "material_timeseries.csv", timeseries_rows)
    _write_phase_csv(output_path / "phase_gantt.csv", phase_rows)
    _write_solution_tex(output_path / "solution_insights.tex", report)

    return report


def export_comparison_report(
    scenario_reports: dict[str, dict[str, Any]],
    output_path: Path | str,
) -> dict[str, Any]:
    """
    Export scenario comparison report and a LaTeX snippet.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison = _build_comparison_report(scenario_reports)

    report_path = output_path / "scenario_comparison.json"
    report_path.write_text(json.dumps(comparison, indent=2))
    _write_comparison_tex(output_path / "scenario_comparison.tex", comparison)

    return comparison


def _build_solution_report(
    model: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    m = model._model
    constants = model.constants
    ton_to_kg = float(constants["units"]["ton_to_kg"])
    steps_per_year = float(constants["time"]["steps_per_year"])
    start_year = float(constants["time"]["start_year"])

    target_mass_tons = float(
        constants["parameter_summary"]["materials"]["bom"]["total_demand_tons"]
    )
    target_mass_kg = target_mass_tons * ton_to_kg

    tier_defs = model._tier_definitions()
    tier1_resources = _tier1_resources(tier_defs)

    # Identify arc types for transport mode breakdown
    elevator_arcs = {a for a in m.A if pyo.value(m.arc_type[a]) == "elevator"}
    rocket_arcs = {a for a in m.A if pyo.value(m.arc_type[a]) in ("rocket", "transfer")}

    reporting_cfg = constants.get("reporting", {})
    bootstrap_isru_share = float(reporting_cfg.get("bootstrap_isru_share", 0.5))
    bootstrap_min_steps = int(reporting_cfg.get("bootstrap_min_steps", 3))
    replication_completion_fraction = float(
        reporting_cfg.get("replication_completion_fraction", 0.8)
    )
    urban_start_fraction = float(reporting_cfg.get("urban_start_fraction", 0.1))

    isru_resources = {
        r
        for r in m.R
        if _safe_value(m.isru_ok[r]) >= 0.5  # type: ignore[index]
    }

    timeseries_rows: list[dict[str, Any]] = []
    cumulative_city_kg: list[float] = []
    earth_arrivals_kg: list[float] = []
    isru_prod_kg: list[float] = []
    tier1_arrivals_kg: list[float] = []
    p_capacity_tpy: list[float] = []
    rocket_arrivals_kg: list[float] = []
    elevator_arrivals_kg: list[float] = []
    rocket_trips: list[float] = []

    for t in m.T:
        year = utils.get_year_for_t(int(t), constants)
        arrivals_kg = sum(_safe_value(m.A_E[r, t]) for r in m.R)
        isru_kg = sum(_safe_value(m.Q[r, t]) for r in isru_resources)
        tier1_kg = sum(_safe_value(m.A_E[r, t]) for r in m.R if r in tier1_resources)
        city_kg = _safe_value(m.Cumulative_City[t])
        p_tpy = _safe_value(m.P[t])

        # Transport mode breakdown
        rocket_kg_t = sum(_safe_value(m.x[a, r, t]) for a in rocket_arcs for r in m.R)
        elevator_kg_t = sum(
            _safe_value(m.x[a, r, t]) for a in elevator_arcs for r in m.R
        )
        rocket_trips_t = sum(_safe_value(m.y[a, t]) for a in rocket_arcs)

        total_supply_kg = arrivals_kg + isru_kg
        isru_share = isru_kg / total_supply_kg if total_supply_kg > 0 else 0.0

        timeseries_rows.append(
            {
                "t": int(t),
                "year": year,
                "earth_arrivals_tons": arrivals_kg / ton_to_kg,
                "rocket_arrivals_tons": rocket_kg_t / ton_to_kg,
                "elevator_arrivals_tons": elevator_kg_t / ton_to_kg,
                "rocket_trips": rocket_trips_t,
                "isru_production_tons": isru_kg / ton_to_kg,
                "total_supply_tons": total_supply_kg / ton_to_kg,
                "isru_share": isru_share,
                "cumulative_city_tons": city_kg / ton_to_kg,
                "capacity_tpy": p_tpy,
            }
        )

        earth_arrivals_kg.append(arrivals_kg)
        isru_prod_kg.append(isru_kg)
        tier1_arrivals_kg.append(tier1_kg)
        cumulative_city_kg.append(city_kg)
        p_capacity_tpy.append(p_tpy)
        rocket_arrivals_kg.append(rocket_kg_t)
        elevator_arrivals_kg.append(elevator_kg_t)
        rocket_trips.append(rocket_trips_t)

    t_end_step = _first_index(cumulative_city_kg, lambda v: v >= target_mass_kg)
    duration_months = int(t_end_step + 1) if t_end_step is not None else None
    duration_years = (
        duration_months / steps_per_year if duration_months is not None else None
    )
    completion_year = (
        start_year + duration_years if duration_years is not None else None
    )

    last_idx = t_end_step if t_end_step is not None else len(cumulative_city_kg) - 1
    earth_total_kg = sum(earth_arrivals_kg[: last_idx + 1])
    isru_total_kg = sum(isru_prod_kg[: last_idx + 1])
    total_supply_kg = earth_total_kg + isru_total_kg
    isru_share_end = isru_total_kg / total_supply_kg if total_supply_kg > 0 else 0.0

    # Transport mode totals
    rocket_total_kg = sum(rocket_arrivals_kg[: last_idx + 1])
    elevator_total_kg = sum(elevator_arrivals_kg[: last_idx + 1])
    rocket_trips_total = sum(rocket_trips[: last_idx + 1])

    bootstrap_t = _find_bootstrap_transition(
        earth_arrivals_kg, isru_prod_kg, bootstrap_isru_share, bootstrap_min_steps
    )
    seed_mass_kg = (
        sum(tier1_arrivals_kg[: bootstrap_t + 1])
        if bootstrap_t is not None
        else sum(tier1_arrivals_kg)
    )
    # Seed mass by transport mode (during bootstrap phase)
    seed_rocket_kg = (
        sum(rocket_arrivals_kg[: bootstrap_t + 1])
        if bootstrap_t is not None
        else sum(rocket_arrivals_kg)
    )
    seed_elevator_kg = (
        sum(elevator_arrivals_kg[: bootstrap_t + 1])
        if bootstrap_t is not None
        else sum(elevator_arrivals_kg)
    )

    carrying_capacity = float(
        constants["implementation_details"]["growth_model"]["phases"]["saturation"][
            "carrying_capacity_tpy"
        ]
    )
    replication_end_t = _first_index(
        p_capacity_tpy,
        lambda v: v >= replication_completion_fraction * carrying_capacity,
    )
    urban_start_t = _first_index(
        cumulative_city_kg, lambda v: v >= urban_start_fraction * target_mass_kg
    )

    phases = _build_phase_rows(
        start_year=start_year,
        steps_per_year=steps_per_year,
        bootstrap_t=bootstrap_t,
        replication_end_t=replication_end_t,
        urban_start_t=urban_start_t,
        completion_t=t_end_step,
    )

    total_cost = _safe_value(m.cost_total)
    objective_value = _safe_value(m.obj_total)
    scenario_value = (
        model.settings.scenario.value
        if hasattr(model, "settings")
        else ScenarioType.MIX.value
    )

    report = {
        "scenario": scenario_value,
        "objective_value": objective_value,
        "total_cost_usd": total_cost,
        "duration_months": duration_months,
        "duration_years": duration_years,
        "completion_year": completion_year,
        "target_mass_tons": target_mass_tons,
        "earth_arrivals_total_tons": earth_total_kg / ton_to_kg,
        "rocket_arrivals_total_tons": rocket_total_kg / ton_to_kg,
        "elevator_arrivals_total_tons": elevator_total_kg / ton_to_kg,
        "rocket_trips_total": rocket_trips_total,
        "isru_production_total_tons": isru_total_kg / ton_to_kg,
        "isru_share_end": isru_share_end,
        "seed_mass_tons": seed_mass_kg / ton_to_kg,
        "seed_rocket_tons": seed_rocket_kg / ton_to_kg,
        "seed_elevator_tons": seed_elevator_kg / ton_to_kg,
        "bootstrap_transition_year": _year_for_step(
            bootstrap_t, start_year, steps_per_year
        ),
        "replication_end_year": _year_for_step(
            replication_end_t, start_year, steps_per_year
        ),
        "urban_start_year": _year_for_step(urban_start_t, start_year, steps_per_year),
        "phases": phases,
    }

    return report, timeseries_rows, phases


def _build_comparison_report(
    scenario_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    bar_chart = []
    within_half_century = []
    for scenario, report in scenario_reports.items():
        duration_years = report.get("duration_years")
        total_cost = report.get("total_cost_usd")
        bar_chart.append(
            {
                "scenario": scenario,
                "duration_years": duration_years,
                "total_cost_usd": total_cost,
                "seed_rocket_tons": report.get("seed_rocket_tons"),
                "seed_elevator_tons": report.get("seed_elevator_tons"),
                "isru_share_end": report.get("isru_share_end"),
            }
        )
        if duration_years is not None and duration_years <= 50:
            within_half_century.append(scenario)

    dominant_scenario = None
    if len(within_half_century) == 1:
        dominant_scenario = within_half_century[0]

    # Extract hybrid (Mix) scenario details for comparison text
    hybrid_details = scenario_reports.get("Mix", {})

    return {
        "bar_chart": bar_chart,
        "within_half_century": within_half_century,
        "dominant_scenario": dominant_scenario,
        "hybrid_duration_years": hybrid_details.get("duration_years"),
        "hybrid_seed_rocket_tons": hybrid_details.get("seed_rocket_tons"),
        "hybrid_seed_elevator_tons": hybrid_details.get("seed_elevator_tons"),
        "scenarios": scenario_reports,
    }


def _write_timeseries_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fieldnames = [
        "t",
        "year",
        "earth_arrivals_tons",
        "rocket_arrivals_tons",
        "elevator_arrivals_tons",
        "rocket_trips",
        "isru_production_tons",
        "total_supply_tons",
        "isru_share",
        "cumulative_city_tons",
        "capacity_tpy",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_phase_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fieldnames = ["phase", "start_year", "end_year"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_solution_tex(path: Path, report: dict[str, Any]) -> None:
    scenario = report.get("scenario", "Mix")
    duration_years = _fmt_or_na(report.get("duration_years"), 1)
    completion_year = _fmt_or_na(report.get("completion_year"), 0)

    # Seed mass breakdown
    seed_mass_mtons = (
        report.get("seed_mass_tons") / 1_000_000
        if report.get("seed_mass_tons") is not None
        else None
    )
    seed_mass = _fmt_or_na(seed_mass_mtons, 2)

    seed_rocket_mtons = (
        report.get("seed_rocket_tons") / 1_000_000
        if report.get("seed_rocket_tons") is not None
        else None
    )
    seed_rocket = _fmt_or_na(seed_rocket_mtons, 2)

    seed_elevator_mtons = (
        report.get("seed_elevator_tons") / 1_000_000
        if report.get("seed_elevator_tons") is not None
        else None
    )
    seed_elevator = _fmt_or_na(seed_elevator_mtons, 2)

    isru_share = _fmt_or_na(
        report.get("isru_share_end") * 100
        if report.get("isru_share_end") is not None
        else None,
        1,
    )

    boot_year = _fmt_or_na(report.get("bootstrap_transition_year"), 0)
    repl_end = _fmt_or_na(report.get("replication_end_year"), 0)
    urban_start = _fmt_or_na(report.get("urban_start_year"), 0)

    phase_lines = []
    for phase in report.get("phases", []):
        start = _fmt_or_na(phase.get("start_year"), 0)
        end = _fmt_or_na(phase.get("end_year"), 0)
        if start != "N/A" and end != "N/A":
            phase_lines.append(f"{phase['phase']} ({start}--{end})")

    phases_text = ", ".join(phase_lines) if phase_lines else "N/A"

    # Transport mode text
    if seed_rocket != "N/A" and seed_elevator != "N/A":
        seed_breakdown = f"(\\textbf{{{seed_rocket} million tons}} via rockets, \\textbf{{{seed_elevator} million tons}} via Space Elevator)"
    else:
        seed_breakdown = ""

    text = "\n".join(
        [
            r"\subsubsection{Bootstrapping and Mass Sourcing Evolution}",
            rf"The model validates the \textbf{{System Bootstrapping Effect}}. By the end of the project, {isru_share}\% of the mass is satisfied by lunar ISRU.",
            r"\textcolor{red}{Figure}: Material Composition River Plot --- A stacked area chart showing the transition from Earth dependence to lunar self-sufficiency as $P_r(t)$ matures.",
            rf"The initial Earth-sourced ``seeds'' total \textbf{{{seed_mass} million tons}} {seed_breakdown}, enabling a transition to self-replication around \textbf{{{boot_year}}}.",
            "",
            r"\subsubsection{Optimal Construction Timeline}",
            rf"The resulting Gantt chart reveals three distinct phases: {phases_text}.",
            r"\textcolor{red}{Figure}: Optimal Construction Schedule Gantt Chart --- Highlighting the critical path from energy grid completion to the finalized Gaia-biosphere habitation modules.",
            "",
            r"\subsubsection{Scenario Summary}",
            rf"\textbf{{Scenario {scenario}}} completes in \textbf{{{duration_years} years}} (approx. {completion_year}).",
        ]
    )

    path.write_text(text)


def _write_comparison_tex(path: Path, comparison: dict[str, Any]) -> None:
    dominant = comparison.get("dominant_scenario")
    within = comparison.get("within_half_century", [])

    # Hybrid scenario details for specific numbers
    hybrid_duration = comparison.get("hybrid_duration_years")
    hybrid_seed_rocket = comparison.get("hybrid_seed_rocket_tons")
    hybrid_seed_elevator = comparison.get("hybrid_seed_elevator_tons")

    # Format hybrid duration
    duration_str = _fmt_or_na(hybrid_duration, 1) if hybrid_duration else "N/A"

    if dominant:
        scenario_label = _scenario_label(dominant)
        if dominant == "Mix" and hybrid_duration:
            lead_line = (
                rf"Simulation results confirm that \textbf{{{scenario_label}}} "
                rf"is the only strategy capable of meeting the MCM Agency's target within a half-century, "
                rf"achieving completion in \textbf{{{duration_str} years}}."
            )
        else:
            lead_line = (
                rf"Simulation results confirm that \textbf{{{scenario_label}}} "
                r"is the only strategy capable of meeting the MCM Agency's target within a half-century."
            )
    elif within:
        within_labels = [_scenario_label(item) for item in within]
        lead_line = (
            r"Simulation results confirm multiple strategies meet the target within a half-century: "
            + ", ".join(within_labels)
            + "."
        )
    else:
        lead_line = r"Simulation results confirm that none of the scenarios meet the target within a half-century."

    # Hybrid model explanation with seed mass breakdown
    seed_rocket_mtons = hybrid_seed_rocket / 1_000_000 if hybrid_seed_rocket else None
    seed_elevator_mtons = (
        hybrid_seed_elevator / 1_000_000 if hybrid_seed_elevator else None
    )
    seed_rocket_str = _fmt_or_na(seed_rocket_mtons, 2) if seed_rocket_mtons else "N/A"
    seed_elevator_str = (
        _fmt_or_na(seed_elevator_mtons, 2) if seed_elevator_mtons else "N/A"
    )

    if seed_rocket_str != "N/A":
        hybrid_explanation = (
            rf"The hybrid model optimizes the ``Cold Start'' problem: rockets are utilized for the initial "
            rf"\textbf{{{seed_rocket_str} million tons}} of ``technological seeds'' (high-complexity robotics "
            rf"and energy modules), while the Space Elevator assumes the role of a bulk carrier for Tier 2 expansion. "
            rf"This prevents the ``Slow Start'' of a pure elevator system (completion $>150$ years) and the "
            rf"``Fiscal Explosion'' of a pure rocket system."
        )
    else:
        hybrid_explanation = (
            r"The hybrid model optimizes the ``Cold Start'' problem: rockets are utilized for the initial "
            r"technological seeds, while the Space Elevator assumes the role of a bulk carrier for Tier 2 expansion."
        )

    text = "\n".join(
        [
            r"\subsubsection{Scenario Dominance and Trade-offs}",
            lead_line,
            r"\textcolor{red}{Figure}: Comparative Performance Bar Chart --- A dual-axis visualization of duration and cost for the three scenarios.",
            "",
            hybrid_explanation,
        ]
    )

    path.write_text(text)


def _tier1_resources(tiers: list[dict[str, Any]]) -> set[str]:
    for tier in tiers:
        if int(tier.get("class", 0)) == 1 or tier.get("id") == "tier_1":
            return set(tier.get("resources", []))
    return set()


def _safe_value(value: Any) -> float:
    try:
        v = pyo.value(value)
    except (TypeError, ValueError):
        v = value
    if v is None:
        return 0.0
    return float(v)


def _first_index(values: list[float], predicate) -> int | None:
    for idx, val in enumerate(values):
        if predicate(val):
            return idx
    return None


def _find_bootstrap_transition(
    earth_arrivals: list[float],
    isru_prod: list[float],
    share_threshold: float,
    min_steps: int,
) -> int | None:
    if min_steps <= 0:
        min_steps = 1
    total_steps = len(earth_arrivals)
    for t in range(total_steps - min_steps + 1):
        ok = True
        for i in range(t, t + min_steps):
            total = earth_arrivals[i] + isru_prod[i]
            share = isru_prod[i] / total if total > 0 else 0.0
            if share < share_threshold:
                ok = False
                break
        if ok:
            return t
    return None


def _build_phase_rows(
    start_year: float,
    steps_per_year: float,
    bootstrap_t: int | None,
    replication_end_t: int | None,
    urban_start_t: int | None,
    completion_t: int | None,
) -> list[dict[str, Any]]:
    phases = []
    if bootstrap_t is not None:
        phases.append(
            {
                "phase": "Industrial Seeding",
                "start_year": start_year,
                "end_year": _year_for_step(bootstrap_t, start_year, steps_per_year),
            }
        )
    if bootstrap_t is not None:
        # End of seeding is start of replication
        rep_start = bootstrap_t

        # End of replication is either when capacity saturates OR when urban build starts
        # If replication_end_t is None (didn't saturate), use urban_start_t
        rep_end = replication_end_t
        if rep_end is None and urban_start_t is not None:
            rep_end = urban_start_t

        # Only add if we have a valid end and it's after start
        if rep_end is not None and rep_end > rep_start:
            phases.append(
                {
                    "phase": "Capacity Self-Replication",
                    "start_year": _year_for_step(rep_start, start_year, steps_per_year),
                    "end_year": _year_for_step(rep_end, start_year, steps_per_year),
                }
            )
    if urban_start_t is not None and completion_t is not None:
        phases.append(
            {
                "phase": "Urban Habitation Expansion",
                "start_year": _year_for_step(urban_start_t, start_year, steps_per_year),
                "end_year": _year_for_step(completion_t, start_year, steps_per_year),
            }
        )
    return phases


def _year_for_step(
    t: int | None, start_year: float, steps_per_year: float
) -> float | None:
    if t is None:
        return None
    return start_year + (t / steps_per_year)


def _fmt(value: Any, digits: int) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return None


def _fmt_or_na(value: Any, digits: int) -> str:
    formatted = _fmt(value, digits)
    return formatted if formatted is not None else "N/A"


def _scenario_label(scenario: str) -> str:
    mapping = {
        "Mix": "Scenario C (Hybrid Model)",
        "E-only": "Scenario A (Elevator-only)",
        "R-only": "Scenario B (Rocket-only)",
    }
    return mapping.get(scenario, scenario)
