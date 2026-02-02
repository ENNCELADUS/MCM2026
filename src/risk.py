"""Risk analysis entry point for Task 2."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simulation.engine import SimulationEngine, SimulationResult, load_baseline
from simulation.stochastics import RiskParams


def compute_statistics(results: list[SimulationResult]) -> dict[str, Any]:
    """Compute summary statistics from Monte Carlo results.

    Args:
        results: List of simulation results.

    Returns:
        Dictionary with statistical summaries.
    """
    completion_years = [r.completion_year for r in results]
    rocket_losses = [r.total_rocket_lost for r in results]
    total_costs = [r.total_cost_usd for r in results]

    years_arr = np.array(completion_years)
    costs_arr = np.array(total_costs)

    return {
        "n_runs": len(results),
        "completion_year": {
            "mean": float(np.mean(years_arr)),
            "std": float(np.std(years_arr)),
            "min": float(np.min(years_arr)),
            "max": float(np.max(years_arr)),
            "median": float(np.median(years_arr)),
            "p5": float(np.percentile(years_arr, 5)),
            "p95": float(np.percentile(years_arr, 95)),
        },
        "total_cost_usd": {
            "mean": float(np.mean(costs_arr)),
            "std": float(np.std(costs_arr)),
            "min": float(np.min(costs_arr)),
            "max": float(np.max(costs_arr)),
            "median": float(np.median(costs_arr)),
            "p5": float(np.percentile(costs_arr, 5)),
            "p95": float(np.percentile(costs_arr, 95)),
        },
        "rocket_losses_tons": {
            "mean": float(np.mean(rocket_losses)),
            "std": float(np.std(rocket_losses)),
            "total_max": float(np.max(rocket_losses)),
        },
    }


def run_risk_analysis(
    baseline_dir: str | Path,
    constants: dict[str, Any],
    output_dir: str | Path | None = None,
    n_runs: int | None = None,
    seed: int = 42,
    verbose: bool = True,
    run_impulse: bool = True,
) -> dict[str, Any]:
    """Run full risk analysis pipeline.

    Args:
        baseline_dir: Path to baseline results directory.
        constants: Loaded constants configuration.
        output_dir: Optional output directory for results.
        n_runs: Number of Monte Carlo runs (overrides config).
        seed: Random seed.
        verbose: Print progress.
        run_impulse: Whether to run impulse response analysis.

    Returns:
        Dictionary with statistics and raw results.
    """
    if verbose:
        print(f"[Risk] Loading baseline from: {baseline_dir}")

    baseline_df = load_baseline(baseline_dir)

    # Get baseline deterministic completion
    baseline_completion = baseline_df[
        baseline_df["cumulative_city_tons"] >= 100_000_000
    ]
    if len(baseline_completion) > 0:
        det_step = int(baseline_completion.iloc[0]["t"])
        det_year = float(baseline_completion.iloc[0]["year"])
    else:
        det_step = len(baseline_df) - 1
        det_year = float(baseline_df.iloc[-1]["year"])

    if verbose:
        print(f"[Risk] Deterministic baseline: step {det_step}, year {det_year:.2f}")

    engine = SimulationEngine(baseline_df, constants)

    if n_runs is not None:
        engine.risk_params.n_runs = n_runs

    if verbose:
        print(f"[Risk] Running {engine.risk_params.n_runs} Monte Carlo simulations...")

    results = engine.run_monte_carlo(base_seed=seed)

    stats = compute_statistics(results)
    stats["deterministic_baseline"] = {
        "completion_step": det_step,
        "completion_year": det_year,
    }

    if verbose:
        print(
            f"[Risk] Completed. Mean completion: {stats['completion_year']['mean']:.2f} years"
        )
        print(f"[Risk] Mean cost: ${stats['total_cost_usd']['mean']:.2e}")
        print(
            f"[Risk] 95% CI (years): [{stats['completion_year']['p5']:.2f}, {stats['completion_year']['p95']:.2f}]"
        )

    # Run impulse response analysis
    impulse_data = None
    if run_impulse:
        if verbose:
            print("[Risk] Running impulse response analysis...")
        impulse_data = engine.run_impulse_response(
            n_points=25, runs_per_point=30, base_seed=seed
        )
        if verbose:
            print(
                f"[Risk] Impulse analysis complete: {len(impulse_data['failure_years'])} points tested"
            )

    # Save results if output_dir provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save statistics
        stats_file = output_path / "risk_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        if verbose:
            print(f"[Risk] Statistics saved to: {stats_file}")

        # Save raw completion years and costs for plotting
        years_df = pd.DataFrame(
            {
                "run": range(len(results)),
                "completion_year": [r.completion_year for r in results],
                "completion_step": [r.completion_step for r in results],
                "rocket_lost_tons": [r.total_rocket_lost for r in results],
                "total_cost_usd": [r.total_cost_usd for r in results],
                "elevator_cost_usd": [r.total_elevator_cost_usd for r in results],
                "rocket_cost_usd": [r.total_rocket_cost_usd for r in results],
            }
        )
        years_file = output_path / "risk_completion_years.csv"
        years_df.to_csv(years_file, index=False)
        if verbose:
            print(f"[Risk] Raw results saved to: {years_file}")

        # Save impulse response data
        if impulse_data is not None:
            impulse_df = pd.DataFrame(
                {
                    "failure_year": impulse_data["failure_years"],
                    "mean_delay": impulse_data["mean_delays"],
                    "std_delay": impulse_data["std_delays"],
                    "mean_cost": impulse_data["mean_costs"],
                }
            )
            impulse_file = output_path / "impulse_response.csv"
            impulse_df.to_csv(impulse_file, index=False)
            if verbose:
                print(f"[Risk] Impulse response saved to: {impulse_file}")

    return {
        "statistics": stats,
        "results": results,
        "impulse_response": impulse_data,
    }


def main():
    """CLI entry point for standalone risk analysis."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Risk Analysis (Task 2)")
    parser.add_argument(
        "--baseline", required=True, help="Path to baseline results directory"
    )
    parser.add_argument(
        "--config", default="src/config/constants.yaml", help="Path to constants.yaml"
    )
    parser.add_argument("--output", default=None, help="Output directory for results")
    parser.add_argument("--runs", type=int, default=None, help="Number of MC runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load constants
    with open(args.config) as f:
        constants = yaml.safe_load(f)

    # Determine output directory
    output_dir = args.output
    if output_dir is None:
        output_dir = Path(args.baseline) / "risk"

    run_risk_analysis(
        baseline_dir=args.baseline,
        constants=constants,
        output_dir=output_dir,
        n_runs=args.runs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
