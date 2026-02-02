#!/usr/bin/env python3
"""Generate Probability Distribution visualization (Figure 2).

Dual-axis histogram showing completion years AND total cost distributions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate completion time and cost distribution plot"
    )
    parser.add_argument(
        "--results", required=True, help="Path to risk results directory"
    )
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    results_dir = Path(args.results)
    csv_file = results_dir / "risk_completion_years.csv"
    stats_file = results_dir / "risk_statistics.json"

    if not csv_file.exists():
        print(f"Error: Results file not found: {csv_file}")
        return 1

    df = pd.read_csv(csv_file)
    with open(stats_file) as f:
        stats = json.load(f)

    completion_years = df["completion_year"].values
    total_costs = df["total_cost_usd"].values / 1e12  # Convert to trillions

    baseline_year = stats["deterministic_baseline"]["completion_year"]
    mean_year = stats["completion_year"]["mean"]
    p5_year = stats["completion_year"]["p5"]
    p95_year = stats["completion_year"]["p95"]

    mean_cost = stats["total_cost_usd"]["mean"] / 1e12
    p5_cost = stats["total_cost_usd"]["p5"] / 1e12
    p95_cost = stats["total_cost_usd"]["p95"] / 1e12

    # Create figure with two subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Left: Completion Year Distribution =====
    n_bins = min(50, len(np.unique(completion_years)))
    ax1.hist(
        completion_years,
        bins=n_bins,
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
        label="Simulated Outcomes",
    )

    # Add KDE
    if len(np.unique(completion_years)) > 1:
        kde = sp_stats.gaussian_kde(completion_years)
        x_range = np.linspace(min(completion_years), max(completion_years), 200)
        ax1.plot(x_range, kde(x_range), "darkblue", linewidth=2, label="Kernel Density")

    # Reference lines
    ax1.axvline(
        x=baseline_year,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Baseline ({baseline_year:.1f})",
    )
    ax1.axvline(
        x=mean_year,
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f"Mean ({mean_year:.1f})",
    )

    # Confidence interval
    ax1.axvspan(p5_year, p95_year, alpha=0.15, color="yellow", label=f"90% CI")

    ax1.set_xlabel("Completion Year", fontsize=12)
    ax1.set_ylabel("Probability Density", fontsize=12)
    ax1.set_title("Distribution of Project Completion Time", fontsize=14)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Statistics text box
    delay = mean_year - baseline_year
    textstr = f"Mean Delay: {delay:.2f} yrs"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.7)
    ax1.text(
        0.05,
        0.95,
        textstr,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    # ===== Right: Cost Distribution =====
    n_bins_cost = min(50, len(np.unique(total_costs)))
    ax2.hist(
        total_costs,
        bins=n_bins_cost,
        density=True,
        alpha=0.7,
        color="#A23B72",
        edgecolor="white",
        label="Simulated Outcomes",
    )

    # Add KDE for cost
    if len(np.unique(total_costs)) > 1:
        kde_cost = sp_stats.gaussian_kde(total_costs)
        x_range_cost = np.linspace(min(total_costs), max(total_costs), 200)
        ax2.plot(
            x_range_cost,
            kde_cost(x_range_cost),
            "darkred",
            linewidth=2,
            label="Kernel Density",
        )

    # Reference lines
    ax2.axvline(
        x=mean_cost,
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f"Mean (${mean_cost:.2f}T)",
    )

    # Confidence interval
    ax2.axvspan(p5_cost, p95_cost, alpha=0.15, color="yellow", label=f"90% CI")

    ax2.set_xlabel("Total Mission Cost (Trillion USD)", fontsize=12)
    ax2.set_ylabel("Probability Density", fontsize=12)
    ax2.set_title("Distribution of Total Mission Cost", fontsize=14)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Statistics text box
    std_cost = stats["total_cost_usd"]["std"] / 1e12
    textstr_cost = f"Std Dev: ${std_cost:.2f}T"
    ax2.text(
        0.05,
        0.95,
        textstr_cost,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    # Output
    output_path = args.output
    if output_path is None:
        figures_dir = results_dir.parent / "figures"
        figures_dir.mkdir(exist_ok=True)
        output_path = figures_dir / "probability_distribution.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
