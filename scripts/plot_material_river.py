#!/usr/bin/env python3
"""
Material Composition River Plot.

Creates a stacked area chart showing the transition from 100% Earth-dependence
to 90%+ lunar self-sufficiency as ISRU capacity matures.

Usage:
    python plot_material_river.py results/20260202_144204 [--scenario Mix]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    }
)


def load_timeseries(results_dir: Path, scenario: str) -> list[dict]:
    """Load material timeseries CSV for a scenario."""
    path = results_dir / scenario / "material_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Timeseries file not found: {path}")

    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "year": float(row["year"]),
                    "earth_arrivals_tons": float(row["earth_arrivals_tons"]),
                    "isru_production_tons": float(row["isru_production_tons"]),
                    "total_supply_tons": float(row["total_supply_tons"]),
                    "isru_share": float(row["isru_share"]),
                    "cumulative_city_tons": float(row["cumulative_city_tons"]),
                }
            )
    return rows


def plot_river(rows: list[dict], output_path: Path, scenario: str) -> None:
    """Create stacked area (river) plot."""
    years = np.array([r["year"] for r in rows])
    earth = (
        np.array([r["earth_arrivals_tons"] for r in rows]) / 1e6
    )  # Convert to million tons
    isru = np.array([r["isru_production_tons"] for r in rows]) / 1e6

    # Calculate cumulative sums for stacking
    cumulative_earth = np.cumsum(earth)
    cumulative_isru = np.cumsum(isru)
    cumulative_total = cumulative_earth + cumulative_isru

    # For the river plot, show percentage composition over time
    earth_share = np.where(
        cumulative_total > 0, cumulative_earth / cumulative_total * 100, 100
    )
    isru_share = np.where(
        cumulative_total > 0, cumulative_isru / cumulative_total * 100, 0
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.1},
    )

    # Colors
    earth_color = "#4A90D9"  # Blue for Earth
    isru_color = "#7CB342"  # Green for ISRU/Lunar

    # --- Panel 1: Stacked Area (Cumulative Mass) ---
    ax1.fill_between(
        years, 0, cumulative_earth, label="Earth Supply", color=earth_color, alpha=0.8
    )
    ax1.fill_between(
        years,
        cumulative_earth,
        cumulative_earth + cumulative_isru,
        label="Lunar ISRU",
        color=isru_color,
        alpha=0.8,
    )

    ax1.set_ylabel("Cumulative Mass Delivered\n(Million Tons)", fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)

    # Format y-axis
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))

    # Add 100M ton target line
    if cumulative_total.max() >= 100:
        ax1.axhline(y=100, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax1.annotate(
            "100M Ton Target",
            xy=(years[len(years) // 4], 102),
            fontsize=9,
            color="red",
            style="italic",
        )

    # --- Panel 2: ISRU Share Percentage ---
    ax2.fill_between(years, 0, isru_share, color=isru_color, alpha=0.6)
    ax2.plot(years, isru_share, color=isru_color, linewidth=2)

    ax2.set_xlabel("Year", fontweight="bold")
    ax2.set_ylabel("ISRU Share (%)", fontweight="bold")
    ax2.set_ylim(0, 100)

    # 80% and 90% reference lines
    ax2.axhline(y=80, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.axhline(y=90, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.annotate("80%", xy=(years[-1] + 0.5, 80), fontsize=8, color="gray", va="center")
    ax2.annotate("90%", xy=(years[-1] + 0.5, 90), fontsize=8, color="gray", va="center")

    # Mark the final ISRU share
    final_isru = isru_share[-1]
    ax2.annotate(
        f"Final: {final_isru:.1f}%",
        xy=(years[-1], final_isru),
        xytext=(-60, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color=isru_color,
        arrowprops=dict(arrowstyle="->", color=isru_color, lw=1.5),
    )

    # Title
    scenario_labels = {
        "E-only": "Elevator-only",
        "R-only": "Rocket-only",
        "Mix": "Hybrid (Scenario C)",
    }
    title = (
        f"Material Composition River Plot — {scenario_labels.get(scenario, scenario)}"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()

    # Save
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Material Composition River Plot"
    )
    parser.add_argument("results_dir", type=Path, help="Path to results directory")
    parser.add_argument(
        "--scenario", default="Mix", help="Scenario to plot (default: Mix)"
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1

    rows = load_timeseries(args.results_dir, args.scenario)

    # Output to peper/figures
    output_dir = Path(__file__).parent.parent / "peper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "material_river.png"

    plot_river(rows, output_path, args.scenario)
    print(
        f"\nTo include in LaTeX:\n  \\includegraphics[width=1\\linewidth]{{figures/material_river.png}}"
    )
    return 0


if __name__ == "__main__":
    exit(main())
