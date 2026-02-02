#!/usr/bin/env python3
"""
Comparative Performance Bar Chart for Scenario Comparison.

Creates a dual-axis visualization of duration (years) and cost (USD)
for the three transport scenarios (E-only, R-only, Mix).

Usage:
    python plot_scenario_comparison.py results/20260202_144204
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def load_comparison_data(results_dir: Path) -> dict:
    """Load scenario comparison JSON."""
    path = results_dir / "scenario_comparison.json"
    if not path.exists():
        raise FileNotFoundError(f"Comparison file not found: {path}")
    return json.loads(path.read_text())


def plot_scenario_comparison(data: dict, output_path: Path) -> None:
    """Create dual-axis bar chart comparing scenarios."""
    bar_chart = data["bar_chart"]

    # Extract data - order: E-only, R-only, Mix (Hybrid)
    scenario_order = ["E-only", "R-only", "Mix"]
    scenario_labels = {
        "E-only": "Scenario A\n(Elevator-only)",
        "R-only": "Scenario B\n(Rocket-only)",
        "Mix": "Scenario C\n(Hybrid)",
    }

    # Reorder data
    ordered_data = []
    for scenario in scenario_order:
        for item in bar_chart:
            if item["scenario"] == scenario:
                ordered_data.append(item)
                break

    labels = [scenario_labels[d["scenario"]] for d in ordered_data]
    durations = [d["duration_years"] for d in ordered_data]
    costs = [d["total_cost_usd"] / 1e9 for d in ordered_data]  # Convert to billions

    x = np.arange(len(labels))
    width = 0.35

    # Create figure with dual axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Colors
    duration_color = "#2E86AB"  # Steel blue
    cost_color = "#E94F37"  # Vermillion

    # Bar plots
    bars1 = ax1.bar(
        x - width / 2,
        durations,
        width,
        label="Duration (years)",
        color=duration_color,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )
    bars2 = ax2.bar(
        x + width / 2,
        costs,
        width,
        label="Cost ($ Billions)",
        color=cost_color,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )

    # Labels and formatting
    ax1.set_xlabel("Transport Scenario", fontweight="bold")
    ax1.set_ylabel("Project Duration (Years)", color=duration_color, fontweight="bold")
    ax2.set_ylabel("Total Cost ($ Billions)", color=cost_color, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax1.tick_params(axis="y", labelcolor=duration_color)
    ax2.tick_params(axis="y", labelcolor=cost_color)

    # Add value labels on bars
    for bar, val in zip(bars1, durations):
        ax1.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=duration_color,
        )

    for bar, val in zip(bars2, costs):
        ax2.annotate(
            f"${val:.2f}B",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=cost_color,
        )

    # 50-year deadline line
    ax1.axhline(y=50, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.annotate(
        "50-year Target", xy=(0.02, 51), fontsize=9, color="gray", style="italic"
    )

    # Highlight the winning scenario (Mix)
    mix_idx = scenario_order.index("Mix")
    bars1[mix_idx].set_edgecolor("#FFD700")  # Gold edge
    bars1[mix_idx].set_linewidth(3)
    bars2[mix_idx].set_edgecolor("#FFD700")
    bars2[mix_idx].set_linewidth(3)

    # Title
    fig.suptitle(
        "Comparative Performance: Duration vs Cost by Scenario",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    # Save
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_scenario_comparison.py <results_dir>")
        print("Example: python plot_scenario_comparison.py results/20260202_144204")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    data = load_comparison_data(results_dir)

    # Output to peper/figures
    output_dir = Path(__file__).parent / "peper"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "scenario_comparison.png"

    plot_scenario_comparison(data, output_path)
    print(
        f"\nTo include in LaTeX:\n  \\includegraphics[width=1\\linewidth]{{scenario_comparison.png}}"
    )


if __name__ == "__main__":
    main()
