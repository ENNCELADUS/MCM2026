#!/usr/bin/env python3
"""Generate Bootstrap Latency Impulse Response visualization (Figure 1).

Uses actual impulse response data from the simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generate impulse response plot")
    parser.add_argument(
        "--results", required=True, help="Path to risk results directory"
    )
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    results_dir = Path(args.results)
    impulse_file = results_dir / "impulse_response.csv"

    if not impulse_file.exists():
        print(f"Error: Impulse response data not found: {impulse_file}")
        print(
            "Run risk analysis first with: python src/main.py --risk --baseline <dir>"
        )
        return 1

    df = pd.read_csv(impulse_file)

    failure_years = df["failure_year"].values
    mean_delays = df["mean_delay"].values
    std_delays = df["std_delay"].values

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot impulse response curve with confidence band
    ax.plot(
        failure_years, mean_delays, "b-", linewidth=2.5, label="Mean Project Delay (ΔT)"
    )
    ax.fill_between(
        failure_years,
        mean_delays - std_delays,
        mean_delays + std_delays,
        alpha=0.3,
        color="blue",
        label="±1 Std Dev",
    )

    # Mark critical phases (based on typical bootstrapping timeline)
    seeding_end = 2052.5  # End of seeding phase
    replication_end = 2058.0  # End of replication phase

    ax.axvline(
        x=seeding_end,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label="End of Seeding Phase",
    )
    ax.axvline(
        x=replication_end,
        color="orange",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label="End of Replication Phase",
    )

    # Shade critical regions
    ax.axvspan(
        failure_years[0], seeding_end, alpha=0.1, color="red", label="_nolegend_"
    )
    ax.axvspan(
        seeding_end, replication_end, alpha=0.08, color="orange", label="_nolegend_"
    )

    ax.set_xlabel("Year of 100-ton Payload Loss", fontsize=12)
    ax.set_ylabel("Total Project Delay ΔT (years)", fontsize=12)
    ax.set_title(
        "The Butterfly Effect of Bootstrap Latency\n(Impulse Response from Monte Carlo Simulation)",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(failure_years[0], failure_years[-1])

    # Set y-axis to start from 0
    y_max = max(mean_delays + std_delays) * 1.2
    ax.set_ylim(0, max(y_max, 0.5))

    # Annotations for interpretation
    if len(mean_delays) > 5 and mean_delays[0] > mean_delays[-1]:
        # Early failure has high impact
        ax.annotate(
            "High Sensitivity\n(Critical Seeds)",
            xy=(failure_years[2], mean_delays[2]),
            xytext=(failure_years[5], mean_delays[0] * 0.8),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=10,
            color="red",
            fontweight="bold",
        )

        ax.annotate(
            "Low Sensitivity\n(Mature Colony)",
            xy=(failure_years[-3], mean_delays[-3]),
            xytext=(failure_years[-8], mean_delays[-1] + 0.1),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
            fontsize=10,
            color="green",
            fontweight="bold",
        )

    # Output
    output_path = args.output
    if output_path is None:
        figures_dir = results_dir.parent / "figures"
        figures_dir.mkdir(exist_ok=True)
        output_path = figures_dir / "impulse_response.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
