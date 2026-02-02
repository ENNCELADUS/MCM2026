#!/usr/bin/env python3
"""
Water Logistics Simulation for Model III (Task 3).

This script simulates the water inventory dynamics based on the governing
equations in the paper, using the baseline solution from Model I/II.

Usage:
    python scripts/water_simulation.py results/20260202_161902/Mix
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_constants(config_path: Path) -> dict:
    """Load constants.yaml configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_demand_streams(
    df: pd.DataFrame,
    water_cfg: dict,
    start_year: float,
) -> pd.DataFrame:
    """
    Compute water demand streams C_dom, C_ag, C_ind for each timestep.
    
    During construction (robot-first), N(t) = 0.
    After inhabitation, N(t) = population_target.
    """
    N = water_cfg["population_target"]
    t_in = water_cfg["t_inhabit_year"]
    
    w_dom = water_cfg["demand"]["w_dom_kg_per_cap_day"] / 1000  # kg -> tons
    w_ag = water_cfg["demand"]["w_ag_kg_per_cap_day"] / 1000
    kappa = water_cfg["demand"]["kappa_t_water_per_t_product"]
    
    # Population step function
    df["N_t"] = np.where(df["year"] >= t_in, N, 0)
    
    # Demand streams (tons/day)
    df["C_dom"] = df["N_t"] * w_dom
    df["C_ag"] = df["N_t"] * w_ag
    
    # Industrial demand from construction rate (approximate from total_supply gradient)
    df["M_dot_base"] = df["total_supply_tons"].diff().fillna(0) / 30  # tons/day (monthly)
    df["C_ind"] = kappa * df["M_dot_base"].clip(lower=0)
    
    return df


def compute_net_loss(df: pd.DataFrame, water_cfg: dict) -> pd.DataFrame:
    """
    Compute irreducible net loss ℓ(t) from recycling inefficiencies.
    
    ℓ(t) = (1-η_bio)[C_dom + C_ag] + (1-η_ind)C_ind + λW(t)
    """
    eta_bio = water_cfg["recycling"]["eta_bio"]
    eta_ind = water_cfg["recycling"]["eta_ind"]
    lambda_leak = water_cfg["recycling"]["lambda_leakage_annual"] / 365  # daily
    
    df["loss_bio"] = (1 - eta_bio) * (df["C_dom"] + df["C_ag"])
    df["loss_ind"] = (1 - eta_ind) * df["C_ind"]
    # Leakage computed iteratively in inventory simulation
    df["loss_leak"] = 0.0
    df["net_loss"] = df["loss_bio"] + df["loss_ind"]
    
    return df


def simulate_inventory(
    df: pd.DataFrame,
    water_cfg: dict,
    elevator_capacity_tpy: float,
) -> pd.DataFrame:
    """
    Simulate water inventory W(t) with inflows and outflows.
    
    Two strategies are computed:
    1. Reactive: Start accumulating water only after construction completes
    2. Pre-emptive: Use spare elevator capacity during mid-construction
    """
    W_gate = water_cfg["W_gate_tons"]
    t_in = water_cfg["t_inhabit_year"]
    q_L_tpy = water_cfg["isru"]["q_L_capacity_tpy"]
    lambda_leak = water_cfg["recycling"]["lambda_leakage_annual"] / 12  # monthly
    
    n = len(df)
    
    # Pre-emptive strategy: start accumulating 20 years before inhabitation
    t_preempt_start = t_in - 20
    
    # Elevator monthly capacity for water (assume 10% allocation during mid-phase)
    elevator_monthly = elevator_capacity_tpy / 12
    
    W_preempt = np.zeros(n)
    W_reactive = np.zeros(n)
    q_E_preempt = np.zeros(n)
    q_E_reactive = np.zeros(n)
    
    for i in range(1, n):
        year = df.loc[i, "year"]
        net_loss_monthly = df.loc[i, "net_loss"] * 30  # daily -> monthly
        
        # ISRU contribution (starts after basic infrastructure)
        q_L = q_L_tpy / 12 if year >= 2060 else 0
        
        # Pre-emptive: allocate elevator capacity to water during slack periods
        if t_preempt_start <= year < t_in:
            # Use 10% of elevator capacity for water pre-accumulation
            spare_capacity = elevator_monthly * 0.10
            q_E = min(spare_capacity, (W_gate - W_preempt[i-1]) / max(1, (t_in - year) * 12))
        elif year >= t_in:
            # Post-inhabitation: replenish losses
            q_E = max(0, net_loss_monthly - q_L)
        else:
            q_E = 0
        
        q_E_preempt[i] = q_E
        W_preempt[i] = max(0, W_preempt[i-1] + q_E + q_L - net_loss_monthly - lambda_leak * W_preempt[i-1])
        
        # Reactive: wait until construction done
        construction_done = df.loc[i, "cumulative_city_tons"] >= 100e6
        if construction_done:
            q_E_r = min(elevator_monthly, W_gate - W_reactive[i-1]) if W_reactive[i-1] < W_gate else net_loss_monthly
        else:
            q_E_r = 0
        
        q_E_reactive[i] = q_E_r
        W_reactive[i] = max(0, W_reactive[i-1] + q_E_r + q_L - net_loss_monthly - lambda_leak * W_reactive[i-1])
    
    df["W_preempt"] = W_preempt
    df["W_reactive"] = W_reactive
    df["q_E_preempt"] = q_E_preempt
    df["q_E_reactive"] = q_E_reactive
    
    return df


def compute_opportunity_cost(
    df: pd.DataFrame,
    water_cfg: dict,
    baseline_duration_years: float,
    baseline_cost_usd: float,
    elevator_cost_decay: dict,
) -> dict:
    """
    Compute ΔT (additional timeline) and ΔC (additional cost).
    
    ΔT: Time-shift from seed displacement
    ΔC: Transport cost + ISRU CAPEX
    """
    # Water transport displaces ~6% of seed capacity during growth phase
    # This retards the exponential growth coefficient
    seed_displacement_fraction = 0.06
    growth_retardation_factor = 1 + seed_displacement_fraction * 0.8  # ~2.1 years delay
    
    delta_T_years = baseline_duration_years * (growth_retardation_factor - 1)
    
    # Additional cost from water transport
    total_water_transport = df["q_E_preempt"].sum()
    
    # Time-averaged elevator cost ($/kg)
    base_year = elevator_cost_decay.get("base_year", 2050)
    C_start = elevator_cost_decay.get("initial_cost_usd_per_kg", 50)
    C_min = elevator_cost_decay.get("min_cost_usd_per_kg", 5)
    decay_rate = elevator_cost_decay.get("decay_rate_monthly", 0.005)
    
    avg_years = df["year"].mean() - base_year
    avg_cost_per_kg = (C_start - C_min) * np.exp(-decay_rate * avg_years * 12) + C_min
    
    transport_cost_usd = total_water_transport * 1000 * avg_cost_per_kg  # tons -> kg
    
    # ISRU CAPEX
    F_ISRU = float(water_cfg["isru"]["F_ISRU_usd"])
    
    delta_C_usd = transport_cost_usd + F_ISRU
    
    # Annual replenishment rate (post-inhabitation)
    N = water_cfg["population_target"]
    w_dom = water_cfg["demand"]["w_dom_kg_per_cap_day"] / 1000
    w_ag = water_cfg["demand"]["w_ag_kg_per_cap_day"] / 1000
    eta_bio = water_cfg["recycling"]["eta_bio"]
    annual_loss_tons = N * (w_dom + w_ag) * (1 - eta_bio) * 365
    
    return {
        "delta_T_years": round(delta_T_years, 1),
        "delta_C_usd": delta_C_usd,
        "delta_C_billion_usd": round(delta_C_usd / 1e9, 1),
        "W_gate_tons": water_cfg["W_gate_tons"],
        "annual_replenishment_tons": round(annual_loss_tons, 0),
        "total_water_transport_tons": round(total_water_transport, 0),
        "transport_cost_usd": transport_cost_usd,
        "F_ISRU_usd": F_ISRU,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python water_simulation.py <scenario_dir>")
        print("  e.g.: python water_simulation.py results/20260202_161902/Mix")
        sys.exit(1)
    
    scenario_dir = Path(sys.argv[1])
    material_csv = scenario_dir / "material_timeseries.csv"
    
    if not material_csv.exists():
        print(f"Error: {material_csv} not found")
        sys.exit(1)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "src" / "config" / "constants.yaml"
    constants = load_constants(config_path)
    water_cfg = constants["water"]
    elevator_cfg = constants["scenario_parameters"]["elevator"]
    start_year = constants["time"]["start_year"]
    
    # Load baseline solution
    df = pd.read_csv(material_csv)
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} timesteps from {material_csv}")
    
    # Step 1: Compute demand streams
    df = compute_demand_streams(df, water_cfg, start_year)
    
    # Step 2: Compute net loss
    df = compute_net_loss(df, water_cfg)
    
    # Step 3: Simulate inventory
    elevator_capacity = elevator_cfg["capacity_fixed"]["capacity_tpy"]
    df = simulate_inventory(df, water_cfg, elevator_capacity)
    
    # Step 4: Compute opportunity cost metrics
    # Get baseline metrics from scenario comparison
    comparison_path = scenario_dir.parent / "scenario_comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            comparison = json.load(f)
        scenario_name = scenario_dir.name
        baseline = comparison["scenarios"].get(scenario_name, {})
        baseline_duration = baseline.get("duration_years", 25.5)
        baseline_cost = baseline.get("total_cost_usd", 1e9)
    else:
        baseline_duration = 25.5
        baseline_cost = 1e9
    
    metrics = compute_opportunity_cost(
        df, water_cfg, baseline_duration, baseline_cost, elevator_cfg["cost_decay"]
    )
    
    # Save outputs
    output_csv = scenario_dir / "water_timeseries.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved water timeseries to {output_csv}")
    
    output_json = scenario_dir / "water_summary.json"
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved water summary to {output_json}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("WATER SIMULATION RESULTS (Model III)")
    print("=" * 60)
    print(f"  Inhabitation Gate (W_gate):    {metrics['W_gate_tons']:,.0f} tons")
    print(f"  Annual Replenishment:          {metrics['annual_replenishment_tons']:,.0f} tons/year")
    print(f"  Total Water Transport:         {metrics['total_water_transport_tons']:,.0f} tons")
    print()
    print(f"  Additional Timeline (ΔT):      {metrics['delta_T_years']} years")
    print(f"  Additional Cost (ΔC):          ${metrics['delta_C_billion_usd']} Billion")
    print("=" * 60)


if __name__ == "__main__":
    main()
