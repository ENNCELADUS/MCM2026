#!/usr/bin/env python3
"""Generate Transport Elasticity visualization (Figure 3) - Stacked Area Plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generate transport elasticity plot")
    parser.add_argument(
        "--baseline", required=True, help="Path to baseline results directory"
    )
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument(
        "--simulate-event",
        action="store_true",
        help="Simulate a swaying event showing elasticity",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    csv_file = baseline_dir / "Mix" / "material_timeseries.csv"

    if not csv_file.exists():
        print(f"Error: Baseline not found: {csv_file}")
        return 1

    df = pd.read_csv(csv_file)

    # Filter to active period only
    df = df[df["total_supply_tons"] > 0].copy()

    years = df["year"].values
    elevator = df["elevator_arrivals_tons"].values
    rocket = df["rocket_arrivals_tons"].values
    isru = df["isru_production_tons"].values

    # If simulating swaying event, create alternate scenario
    if args.simulate_event:
        # Simulate elevator efficiency drop from year 2055-2060
        event_start = 2055
        event_end = 2060
        efficiency_drop = 0.6  # 60% capacity

        mask = (years >= event_start) & (years <= event_end)

        # Reduced elevator during event
        elevator_event = elevator.copy()
        elevator_event[mask] = elevator[mask] * efficiency_drop

        # Rocket compensates (if capacity allows)
        rocket_event = rocket.copy()
        compensation = elevator[mask] - elevator_event[mask]
        rocket_event[mask] = (
            rocket[mask] + compensation * 0.7
        )  # 70% compensation possible

        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Top: Normal operation
        ax1.stackplot(
            years,
            elevator,
            rocket,
            isru,
            labels=["Space Elevator", "Rockets", "ISRU Production"],
            colors=["#2E86AB", "#A23B72", "#F18F01"],
            alpha=0.8,
        )
        ax1.set_ylabel("Material Flow (tons/month)", fontsize=12)
        ax1.set_title("Normal Operation (Deterministic Baseline)", fontsize=14)
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(2050, years.max())

        # Bottom: During swaying event
        ax2.stackplot(
            years,
            elevator_event,
            rocket_event,
            isru,
            labels=["Space Elevator (Degraded)", "Rockets (Surge)", "ISRU Production"],
            colors=["#2E86AB", "#A23B72", "#F18F01"],
            alpha=0.8,
        )
        ax2.axvspan(
            event_start,
            event_end,
            alpha=0.3,
            color="red",
            label="Elevator Swaying Event",
        )
        ax2.set_xlabel("Year", fontsize=12)
        ax2.set_ylabel("Material Flow (tons/month)", fontsize=12)
        ax2.set_title(
            f"During Swaying Event ({event_start}-{event_end}): Rocket Surge Compensates",
            fontsize=14,
        )
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(2050, years.max())

    else:
        # Simple stacked area plot
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.stackplot(
            years,
            elevator,
            rocket,
            isru,
            labels=["Space Elevator", "Rockets", "ISRU Production"],
            colors=["#2E86AB", "#A23B72", "#F18F01"],
            alpha=0.8,
        )

        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Material Flow (tons/month)", fontsize=12)
        ax.set_title(
            "Transport Mode Composition Over Time\n(Hybrid Model - Scenario C)",
            fontsize=14,
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2050, years.max())

    # Output
    output_path = args.output
    if output_path is None:
        figures_dir = baseline_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        suffix = "_elasticity" if args.simulate_event else ""
        output_path = figures_dir / f"transport_composition{suffix}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
