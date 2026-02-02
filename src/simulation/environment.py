"""
Environmental Sustainability Simulation Module (Model IV)

Computes environmental debt metrics from optimized solution data:
- D_E(t): Earth atmospheric debt (stratospheric emissions from rockets)
- D_M(t): Moon surface debt (regolith disturbance from ISRU)

Simulates a "conservative baseline" counterfactual for comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml


@dataclass
class EnvironmentParams:
    """Environmental impact parameters loaded from constants.yaml."""

    # Rocket emissions (kg pollutant per ton payload)
    e_CO2: float
    e_BC: float
    e_Al2O3: float
    e_NOx: float

    # Stratospheric weights ($/kg)
    w_CO2: float
    w_BC: float
    w_Al2O3: float
    w_NOx: float

    # Elevator emissions
    CI_grid: float  # kg CO2 / kWh
    energy_per_ton: float  # kWh / ton

    # Moon disturbance
    epsilon_reg: float  # $/ton baseline
    epsilon_sens: float  # $/ton sensitive zone
    sens_fraction: float  # fraction in sensitive zones

    # Conservative baseline
    bootstrap_delay_years: float
    capacity_growth_penalty: float

    @classmethod
    def from_config(cls, config: dict) -> "EnvironmentParams":
        """Load from constants.yaml environment section."""
        env = config["environment"]
        rocket = env["rocket_emissions"]
        weights = env["stratospheric_weights"]
        elevator = env["elevator_emissions"]
        moon = env["moon_disturbance"]
        baseline = env["conservative_baseline"]

        return cls(
            e_CO2=rocket["e_CO2"],
            e_BC=rocket["e_BC"],
            e_Al2O3=rocket["e_Al2O3"],
            e_NOx=rocket["e_NOx"],
            w_CO2=weights["w_CO2"],
            w_BC=weights["w_BC"],
            w_Al2O3=weights["w_Al2O3"],
            w_NOx=weights["w_NOx"],
            CI_grid=elevator["CI_grid_kg_CO2_per_kWh"],
            energy_per_ton=elevator["energy_per_ton_kWh"],
            epsilon_reg=moon["epsilon_reg"],
            epsilon_sens=moon["epsilon_sens"],
            sens_fraction=moon["sens_fraction"],
            bootstrap_delay_years=baseline["bootstrap_delay_years"],
            capacity_growth_penalty=baseline["capacity_growth_penalty"],
        )

    @property
    def rocket_cost_per_ton(self) -> float:
        """
        Compute weighted emission cost per ton of rocket payload.
        D_E contribution = sum_m (w_m * e_m) for each pollutant m.
        """
        return (
            self.w_CO2 * self.e_CO2
            + self.w_BC * self.e_BC
            + self.w_Al2O3 * self.e_Al2O3
            + self.w_NOx * self.e_NOx
        )

    @property
    def elevator_cost_per_ton(self) -> float:
        """
        Compute elevator emission cost per ton.
        Based on grid carbon intensity and energy consumption.
        """
        co2_per_ton = self.CI_grid * self.energy_per_ton  # kg CO2
        return self.w_CO2 * co2_per_ton

    @property
    def moon_cost_per_ton(self) -> float:
        """Effective lunar disturbance cost per ton ISRU."""
        return self.epsilon_reg + self.sens_fraction * self.epsilon_sens


def load_solution(result_path: Path) -> pd.DataFrame:
    """
    Load material_timeseries.csv from result directory.

    Expected columns:
    - t, year
    - rocket_arrivals_tons, elevator_arrivals_tons
    - isru_production_tons
    - cumulative_city_tons
    """
    csv_path = result_path / "material_timeseries.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Solution file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def compute_earth_debt(df: pd.DataFrame, params: EnvironmentParams) -> pd.Series:
    """
    Compute cumulative Earth atmospheric debt D_E(t).

    D_E(t) = integral of [rocket_cost * rocket_arrivals + elevator_cost * elevator_arrivals]
    """
    rocket_cost = params.rocket_cost_per_ton
    elevator_cost = params.elevator_cost_per_ton

    # Per-step debt
    step_debt = (
        rocket_cost * df["rocket_arrivals_tons"]
        + elevator_cost * df["elevator_arrivals_tons"]
    )

    # Cumulative debt
    cumulative_debt = step_debt.cumsum()
    return cumulative_debt


def compute_moon_debt(df: pd.DataFrame, params: EnvironmentParams) -> pd.Series:
    """
    Compute cumulative Moon surface debt D_M(t).

    D_M(t) = integral of [moon_cost * isru_production]
    """
    moon_cost = params.moon_cost_per_ton

    step_debt = moon_cost * df["isru_production_tons"]
    cumulative_debt = step_debt.cumsum()
    return cumulative_debt


def simulate_conservative_baseline(
    df: pd.DataFrame, params: EnvironmentParams
) -> pd.DataFrame:
    """
    Simulate conservative baseline: "Struggling Plan" (Elevator-Only).

    Instead of simply shifting the aggressive curve, we model a fundamentally
    different growth trajectory defined by:
    - No initial rocket pulse -> severely delayed bootstrapping.
    - Reliance on trickling elevator supply -> very slow growth rate (r).
    - Long-term dependency on Earth imports -> high cumulative elevator debt.

    Models P_cons(t) as a logistic function:
       P(t) = K / (1 + A * exp(-r * (t - t0)))
    Where r is significantly smaller than the aggressive case.
    """
    # Copy original data
    df_cons = df.copy()

    # 1. Model Conservative Production Curve (Synthetic Logistic)
    # Get simulation time (years)
    years = df["year"].values
    t_start = years[0]

    # Logistic Parameters
    K = df["isru_production_tons"].max()  # Same max capacity
    if K <= 0: K = 8e7 # Fallback if max is 0

    # Conservative Growth Rate (r)
    # Aggressive r ~ 0.35 (alpha). Conservative is heavily penalized.
    r_cons = 0.35 * params.capacity_growth_penalty

    # Conservative Delay (t0_shift)
    # Moves the inflection point later
    delay_years = params.bootstrap_delay_years
    t_inflection = t_start + delay_years + 10.0 # +10y for natural slow startup

    # Calculate A for logistic
    # P(t_start) ~ small epsilon
    # K / (1 + A) = epsilon -> A ~ K/epsilon
    A = 1000.0

    # Generate P_cons(t) - Monthly Production Rate
    # Note: Logistic gives Total Annual Capacity P(t). Monthly prod = P(t)/12.
    # We adjust the formula to act as cumulative adoption or just rate?
    # Let's model the Production Rate directly.

    # Conservative ISRU profile
    # Calculate vector of production (tons/month assumed similar scale to input)
    # Use simple scaling relative to max aggressive output to ensure comparability
    max_agg_prod = df["isru_production_tons"].max()

    # Create mask for delay period
    delay_mask = (years - t_start) < delay_years
    
    # Normalized logistic curve [0, 1] starting AFTER delay
    # We want sigmoid(0) ~ small number, sigmoid(15) ~ 0.5
    # norm_time should start at 0 after delay
    valid_time = (years - t_start - delay_years)
    
    # Sigmoid centered at 15 years post-delay
    # t=0 -> exp(-r*-15) = exp(large pos) -> 1/(1+large) ~ 0
    sigmoid = 1 / (1 + np.exp(-r_cons * (valid_time - 15)))

    # Scale to demand
    isru_cons = max_agg_prod * sigmoid
    
    # Zero out production during delay 
    isru_cons[delay_mask] = 0.0

    # Apply to dataframe
    df_cons["isru_production_tons"] = isru_cons

    # 2. Calculate Supply Deficit
    # Total Demand = Aggressive Supply (assuming demand is fixed by project goals)
    total_supply_agg = df["isru_production_tons"] + df["earth_arrivals_tons"]

    # Deficit = Target Supply - Conservative ISRU
    supply_deficit = total_supply_agg - isru_cons
    supply_deficit = supply_deficit.clip(lower=0)

    # 3. Fill Deficit with Elevator Transport
    # Conservative scenario uses NO rockets.
    df_cons["rocket_arrivals_tons"] = 0.0

    # Everything else comes from Elevator
    df_cons["elevator_arrivals_tons"] = supply_deficit

    # Update total earth arrivals
    df_cons["earth_arrivals_tons"] = df_cons["elevator_arrivals_tons"]

    return df_cons


def find_crossover_year(
    years: pd.Series,
    aggressive_debt: pd.Series,
    conservative_debt: pd.Series,
) -> float | None:
    """
    Find the year where aggressive cumulative debt becomes less than conservative.

    Returns the crossover year or None if no crossover occurs.
    """
    diff = aggressive_debt - conservative_debt

    # Find where diff goes from positive to negative
    crossover_indices = np.where((diff.values[:-1] > 0) & (diff.values[1:] <= 0))[0]

    if len(crossover_indices) > 0:
        idx = crossover_indices[0]
        # Linear interpolation for more precise year
        if diff.iloc[idx] != diff.iloc[idx + 1]:
            frac = diff.iloc[idx] / (diff.iloc[idx] - diff.iloc[idx + 1])
            crossover_year = years.iloc[idx] + frac * (
                years.iloc[idx + 1] - years.iloc[idx]
            )
        else:
            crossover_year = years.iloc[idx]
        return float(crossover_year)

    return None


def run_environmental_analysis(
    result_path: Path,
    output_path: Path,
    config: dict,
) -> dict:
    """
    Main entry point for environmental analysis.

    1. Load solution data
    2. Compute D_E and D_M for aggressive (actual) scenario
    3. Simulate conservative baseline
    4. Compute debts for conservative scenario
    5. Find crossover year
    6. Save results

    Returns metrics dictionary.
    """
    params = EnvironmentParams.from_config(config)

    # Load solution
    df = load_solution(result_path)

    # Compute debts for aggressive scenario (actual solution)
    df["D_E_aggressive"] = compute_earth_debt(df, params)
    df["D_M"] = compute_moon_debt(df, params)
    df["D_total_aggressive"] = df["D_E_aggressive"] + df["D_M"]

    # Conservative baseline
    df_cons = simulate_conservative_baseline(df, params)
    df["D_E_conservative"] = compute_earth_debt(df_cons, params)
    df["D_M_conservative"] = compute_moon_debt(df_cons, params)
    df["D_total_conservative"] = df["D_E_conservative"] + df["D_M_conservative"]

    # Find crossover
    crossover_year = find_crossover_year(
        df["year"], df["D_total_aggressive"], df["D_total_conservative"]
    )

    # Compute final metrics
    final_idx = df["cumulative_city_tons"].idxmax()
    completion_year = df.loc[final_idx, "year"] if final_idx is not None else None

    # Total debts at completion
    total_D_E = df["D_E_aggressive"].iloc[final_idx] if final_idx else 0
    total_D_M = df["D_M"].iloc[final_idx] if final_idx else 0
    total_debt = total_D_E + total_D_M

    # Conservative comparison
    total_D_E_cons = df["D_E_conservative"].iloc[final_idx] if final_idx else 0
    total_D_M_cons = df["D_M_conservative"].iloc[final_idx] if final_idx else 0
    total_debt_cons = total_D_E_cons + total_D_M_cons

    # Delta metrics
    delta_debt = total_debt_cons - total_debt  # Savings from aggressive strategy

    # Per-unit costs
    rocket_cost_per_ton = params.rocket_cost_per_ton
    elevator_cost_per_ton = params.elevator_cost_per_ton
    moon_cost_per_ton = params.moon_cost_per_ton

    # Build metrics
    metrics = {
        "scenario": result_path.name,
        "crossover_year": crossover_year,
        "completion_year": completion_year,
        # Aggressive (actual) scenario
        "total_D_E_aggressive": total_D_E,
        "total_D_M": total_D_M,
        "total_debt_aggressive": total_debt,
        # Conservative scenario
        "total_D_E_conservative": total_D_E_cons,
        "total_D_M_conservative": total_D_M_cons,
        "total_debt_conservative": total_debt_cons,
        # Delta
        "debt_savings": delta_debt,
        # Unit costs
        "rocket_cost_per_ton_usd": rocket_cost_per_ton,
        "elevator_cost_per_ton_usd": elevator_cost_per_ton,
        "moon_cost_per_ton_usd": moon_cost_per_ton,
    }

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save timeseries
    env_cols = [
        "t",
        "year",
        "rocket_arrivals_tons",
        "elevator_arrivals_tons",
        "isru_production_tons",
        "cumulative_city_tons",
        "D_E_aggressive",
        "D_M",
        "D_total_aggressive",
        "D_E_conservative",
        "D_M_conservative",
        "D_total_conservative",
    ]
    df_out = df[[c for c in env_cols if c in df.columns]]
    df_out.to_csv(output_path / "env_timeseries.csv", index=False)

    # Save summary
    with open(output_path / "env_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Environmental analysis complete. Results saved to {output_path}")
    print(f"  Crossover year: {crossover_year}")
    print(f"  Total debt (aggressive): ${total_debt / 1e9:.2f}B")
    print(f"  Total debt (conservative): ${total_debt_cons / 1e9:.2f}B")
    print(f"  Debt savings: ${delta_debt / 1e9:.2f}B")

    return metrics


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run environmental sustainability analysis (Model IV)"
    )
    parser.add_argument(
        "--result_path",
        type=Path,
        required=True,
        help="Path to scenario result directory (e.g., results/20260202_161902/Mix)",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output directory for env results (default: result_path/env)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "constants.yaml",
        help="Path to constants.yaml",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set default output path
    output_path = args.output_path or args.result_path / "env"

    # Run analysis
    run_environmental_analysis(args.result_path, output_path, config)


if __name__ == "__main__":
    main()
