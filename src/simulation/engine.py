"""Forward simulation engine for risk assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from .stochastics import RiskSampler, RiskParams


@dataclass
class SimulationResult:
    """Result of a single simulation run."""

    completion_step: int
    completion_year: float
    total_elevator_delivered: float
    total_rocket_delivered: float
    total_rocket_lost: float
    total_isru_produced: float
    # Cost tracking
    total_elevator_cost_usd: float = 0.0
    total_rocket_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    # Detailed tracking
    elevator_deferrals: list[float] = field(default_factory=list)
    rocket_failures_by_step: list[int] = field(default_factory=list)


class SimulationEngine:
    """Replays baseline plan under stochastic conditions."""

    def __init__(
        self,
        baseline_df: pd.DataFrame,
        constants: dict[str, Any],
        target_mass: float = 100_000_000.0,
    ):
        """
        Args:
            baseline_df: DataFrame from material_timeseries.csv
            constants: Loaded constants.yaml
            target_mass: Target cumulative mass (tons)
        """
        self.baseline = baseline_df.copy()
        self.constants = constants
        self.target_mass = target_mass

        # Extract elevator capacity from constants
        elev_cfg = constants.get("scenario_parameters", {}).get("elevator", {})
        capacity_fixed = elev_cfg.get("capacity_fixed", {})
        self.elevator_capacity_tpy = float(capacity_fixed.get("capacity_tpy", 358000))

        # Convert to per-step capacity
        steps_per_year = float(constants.get("time", {}).get("steps_per_year", 12))
        self.elevator_capacity_per_step = self.elevator_capacity_tpy / steps_per_year

        self.risk_params = RiskParams.from_config(constants)

    def run_once(
        self, seed: int | None = None, inject_failure_at_step: int | None = None
    ) -> SimulationResult:
        """Execute one simulation run.

        The simulation tracks how delays accumulate due to:
        1. Elevator capacity constraints (deferrals add up)
        2. Rocket failures (payload lost)

        Key insight: The baseline plan has a target delivery schedule.
        Under stochastic conditions, actual deliveries may fall short,
        causing the target to be reached later.

        Args:
            seed: Random seed for reproducibility.
            inject_failure_at_step: If set, inject a 100-ton loss at this step (for impulse response).

        Returns:
            SimulationResult with completion metrics.
        """
        sampler = RiskSampler(self.risk_params, seed=seed)

        # Track deviation from baseline cumulative
        cumulative_shortfall = 0.0  # How much we're behind baseline
        deferred_elev = 0.0  # Material waiting to be delivered
        total_elev = 0.0
        total_rock = 0.0
        total_rock_lost = 0.0
        total_isru = 0.0
        total_elev_cost = 0.0
        total_rock_cost = 0.0
        deferrals = []
        failures = []

        # Extract cost parameters
        elev_cost_cfg = (
            self.constants.get("scenario_parameters", {})
            .get("elevator", {})
            .get("cost_decay", {})
        )
        rock_cost_cfg = (
            self.constants.get("scenario_parameters", {})
            .get("rocket", {})
            .get("cost_decay", {})
        )

        elev_base_year = float(elev_cost_cfg.get("base_year", 2050))
        elev_init_cost = float(elev_cost_cfg.get("initial_cost_usd_per_kg", 50))
        elev_min_cost = float(elev_cost_cfg.get("min_cost_usd_per_kg", 5))
        elev_decay = float(elev_cost_cfg.get("decay_rate_monthly", 0.005))

        rock_base_year = float(rock_cost_cfg.get("base_year", 2050))
        rock_init_cost = float(rock_cost_cfg.get("base_cost_usd_per_kg", 110))
        rock_min_cost = float(rock_cost_cfg.get("min_cost_usd_per_kg", 100))
        rock_decay = float(rock_cost_cfg.get("decay_rate_monthly", 0.0145))

        start_year = float(self.constants.get("time", {}).get("start_year", 2050))
        steps_per_year = float(self.constants.get("time", {}).get("steps_per_year", 12))

        # Find when baseline reaches target
        baseline_target_step = len(self.baseline) - 1
        for idx, row in self.baseline.iterrows():
            if row.get("cumulative_city_tons", 0) >= self.target_mass:
                baseline_target_step = int(idx)
                break

        # Simulate each step up to baseline completion
        for t in range(baseline_target_step + 1):
            row = self.baseline.iloc[t]
            planned_elev = float(row.get("elevator_arrivals_tons", 0))
            planned_rock = float(row.get("rocket_arrivals_tons", 0))
            n_launches = int(row.get("rocket_trips", 0))
            isru = float(row.get("isru_production_tons", 0))

            # Inject failure if requested (for impulse response analysis)
            # This simulates a month's worth of lost deliveries (catastrophic event)
            if inject_failure_at_step is not None and t == inject_failure_at_step:
                # Inject substantial loss: ~1 month of combined deliveries
                inject_loss = 50000.0  # 50k tons = significant setback
                if planned_elev >= inject_loss / 2:
                    # Split loss between elevator and rocket
                    elev_loss = min(planned_elev, inject_loss * 0.6)
                    rock_loss = min(planned_rock, inject_loss * 0.4)
                    planned_elev -= elev_loss
                    planned_rock -= rock_loss
                    cumulative_shortfall += elev_loss + rock_loss
                elif planned_rock >= inject_loss:
                    planned_rock -= inject_loss
                    cumulative_shortfall += inject_loss

            # --- Elevator: efficiency constraint ---
            eta = sampler.sample_elevator_efficiency()
            effective_capacity = self.elevator_capacity_per_step * eta

            # Attempt to deliver planned + deferred
            available_elev = planned_elev + deferred_elev
            actual_elev = min(available_elev, effective_capacity)
            deferred_elev = available_elev - actual_elev

            deferrals.append(deferred_elev)
            total_elev += actual_elev

            # Cost for elevator: exponential decay from base_year
            current_year = start_year + (t / steps_per_year)
            months_since_base = (current_year - elev_base_year) * 12
            elev_cost_per_kg = (elev_init_cost - elev_min_cost) * np.exp(
                -elev_decay * months_since_base
            ) + elev_min_cost
            total_elev_cost += actual_elev * 1000 * elev_cost_per_kg  # tons -> kg

            # --- Rocket: Bernoulli failures ---
            if n_launches > 0 and planned_rock > 0:
                successes = sampler.sample_rocket_success(n_launches)
                success_ratio = successes / n_launches
                actual_rock = planned_rock * success_ratio
                lost_rock = planned_rock - actual_rock
            else:
                actual_rock = planned_rock  # 0 launches = 0 planned
                lost_rock = 0.0
                successes = n_launches

            failures.append(n_launches - successes if n_launches > 0 else 0)
            total_rock += actual_rock
            total_rock_lost += lost_rock

            # Cost for rocket (including failed launches - still paid for)
            months_since_rock_base = (current_year - rock_base_year) * 12
            rock_cost_per_kg = (rock_init_cost - rock_min_cost) * np.exp(
                -rock_decay * months_since_rock_base
            ) + rock_min_cost
            total_rock_cost += (
                planned_rock * 1000 * rock_cost_per_kg
            )  # Pay for all attempted

            # --- ISRU: assume deterministic ---
            total_isru += isru

            # --- Track cumulative shortfall ---
            # Rocket loss is permanent, elevator deferral will be caught up
            cumulative_shortfall += lost_rock

        # After baseline timeline, we need to "catch up" for:
        # 1. Cumulative rocket losses (must be replaced)
        # 2. Remaining deferred elevator cargo

        # Estimate additional steps needed to make up shortfall
        # Assuming we continue at average delivery rate
        if baseline_target_step > 0:
            avg_delivery_per_step = (total_elev + total_rock + total_isru) / (
                baseline_target_step + 1
            )
        else:
            avg_delivery_per_step = 1.0  # Avoid division by zero

        # Total shortfall = rocket losses + deferred (still pending)
        total_shortfall = cumulative_shortfall + deferred_elev

        # Additional steps to make up shortfall (with some efficiency loss)
        if avg_delivery_per_step > 0:
            additional_steps = int(
                np.ceil(total_shortfall / (avg_delivery_per_step * 0.85))
            )
        else:
            additional_steps = 0

        completion_step = baseline_target_step + additional_steps

        # Compute completion year
        completion_year = start_year + (completion_step / steps_per_year)

        # Add cost for additional steps (at end-of-project rates)
        if additional_steps > 0:
            # Use end-of-project cost rates
            final_months = (completion_year - elev_base_year) * 12
            end_elev_cost = (elev_init_cost - elev_min_cost) * np.exp(
                -elev_decay * final_months
            ) + elev_min_cost
            end_rock_cost = (rock_init_cost - rock_min_cost) * np.exp(
                -rock_decay * final_months
            ) + rock_min_cost
            # Approximate cost of shortfall makeup
            makeup_cost = total_shortfall * 1000 * (end_elev_cost + end_rock_cost) / 2
            total_elev_cost += makeup_cost * 0.5
            total_rock_cost += makeup_cost * 0.5

        return SimulationResult(
            completion_step=completion_step,
            completion_year=completion_year,
            total_elevator_delivered=total_elev,
            total_rocket_delivered=total_rock,
            total_rocket_lost=total_rock_lost,
            total_isru_produced=total_isru,
            total_elevator_cost_usd=total_elev_cost,
            total_rocket_cost_usd=total_rock_cost,
            total_cost_usd=total_elev_cost + total_rock_cost,
            elevator_deferrals=deferrals,
            rocket_failures_by_step=failures,
        )

    def run_monte_carlo(
        self, n_runs: int | None = None, base_seed: int = 42
    ) -> list[SimulationResult]:
        """Run multiple simulations.

        Args:
            n_runs: Number of runs (defaults to config value).
            base_seed: Base seed for reproducibility.

        Returns:
            List of SimulationResult objects.
        """
        if n_runs is None:
            n_runs = self.risk_params.n_runs

        results = []
        for i in range(n_runs):
            result = self.run_once(seed=base_seed + i)
            results.append(result)
        return results

    def run_impulse_response(
        self, n_points: int = 30, runs_per_point: int = 50, base_seed: int = 42
    ) -> dict[str, list]:
        """Run impulse response analysis.

        Injects a 100-ton loss at different points in the timeline and measures
        the resulting delay. This quantifies the "Butterfly Effect" - how early
        failures have disproportionate impact.

        Args:
            n_points: Number of failure points to test across timeline.
            runs_per_point: Monte Carlo runs per failure point.
            base_seed: Random seed.

        Returns:
            Dictionary with failure_years, mean_delays, std_delays.
        """
        start_year = float(self.constants.get("time", {}).get("start_year", 2050))
        steps_per_year = float(self.constants.get("time", {}).get("steps_per_year", 12))

        # Find baseline completion step
        baseline_target_step = len(self.baseline) - 1
        for idx, row in self.baseline.iterrows():
            if row.get("cumulative_city_tons", 0) >= self.target_mass:
                baseline_target_step = int(idx)
                break

        baseline_year = start_year + (baseline_target_step / steps_per_year)

        # Get baseline completion (no failure injection) for reference
        baseline_results = [
            self.run_once(seed=base_seed + i) for i in range(runs_per_point)
        ]
        baseline_mean = np.mean([r.completion_year for r in baseline_results])

        # Test failure injection at different points
        failure_steps = np.linspace(
            0, int(baseline_target_step * 0.8), n_points, dtype=int
        )
        failure_years = []
        mean_delays = []
        std_delays = []
        mean_costs = []

        for step in failure_steps:
            year = start_year + (step / steps_per_year)
            failure_years.append(year)

            # Run MC with failure at this step
            completion_years = []
            costs = []
            for i in range(runs_per_point):
                result = self.run_once(
                    seed=base_seed + i, inject_failure_at_step=int(step)
                )
                completion_years.append(result.completion_year)
                costs.append(result.total_cost_usd)

            mean_delay = np.mean(completion_years) - baseline_mean
            mean_delays.append(mean_delay)
            std_delays.append(np.std(completion_years))
            mean_costs.append(np.mean(costs))

        return {
            "failure_years": failure_years,
            "mean_delays": mean_delays,
            "std_delays": std_delays,
            "mean_costs": mean_costs,
            "baseline_mean_year": baseline_mean,
        }


def load_baseline(baseline_dir: str | Path) -> pd.DataFrame:
    """Load baseline material timeseries from results directory.

    Args:
        baseline_dir: Path to results directory (e.g., results/20260202_161902)

    Returns:
        DataFrame with baseline plan.
    """
    baseline_path = Path(baseline_dir) / "Mix" / "material_timeseries.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    return pd.read_csv(baseline_path)
