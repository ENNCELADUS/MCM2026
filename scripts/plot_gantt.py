#!/usr/bin/env python3
"""
Optimal Construction Schedule Gantt Chart.

Creates a horizontal Gantt chart showing the three distinct project phases:
- Industrial Seeding
- Capacity Self-Replication  
- Urban Habitation Expansion

Usage:
    python plot_gantt.py results/20260202_144204 [--scenario Mix]
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def load_phases(results_dir: Path, scenario: str) -> list[dict]:
    """Load phase data from phase_gantt.csv or solution_report.json."""
    # Try CSV first
    csv_path = results_dir / scenario / "phase_gantt.csv"
    if csv_path.exists():
        phases = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                phases.append({
                    "phase": row["phase"],
                    "start_year": float(row["start_year"]),
                    "end_year": float(row["end_year"]),
                })
        return phases
    
    # Fall back to JSON
    json_path = results_dir / scenario / "solution_report.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        return data.get("phases", [])
    
    raise FileNotFoundError(f"No phase data found in {results_dir / scenario}")


def plot_gantt(phases: list[dict], output_path: Path, scenario: str) -> None:
    """Create Gantt chart for project phases."""
    if not phases:
        print("No phase data available")
        return
    
    # Filter out phases with missing data
    valid_phases = [p for p in phases if p.get("start_year") and p.get("end_year")]
    if not valid_phases:
        print("No valid phase data")
        return
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Phase colors - industrial to urban gradient
    phase_colors = {
        "Industrial Seeding": "#E94F37",           # Red-orange (bootstrap)
        "Capacity Self-Replication": "#F39C12",    # Orange-gold (growth)
        "Urban Habitation Expansion": "#27AE60",   # Green (completion)
    }
    default_color = "#3498DB"
    
    # Chart settings
    bar_height = 0.6
    y_positions = list(range(len(valid_phases)))
    
    # Draw bars
    for i, phase in enumerate(valid_phases):
        start = phase["start_year"]
        end = phase["end_year"]
        duration = end - start
        
        color = phase_colors.get(phase["phase"], default_color)
        
        # Main bar
        bar = ax.barh(i, duration, left=start, height=bar_height,
                      color=color, alpha=0.85, edgecolor="white", linewidth=2)
        
        # Duration label inside bar
        mid_x = start + duration / 2
        if duration > 3:  # Only if bar is wide enough
            ax.text(mid_x, i, f"{duration:.1f} yrs", 
                    ha="center", va="center", fontsize=10, 
                    color="white", fontweight="bold")
        
        # Year labels at edges
        ax.text(start - 0.3, i, f"{start:.0f}", 
                ha="right", va="center", fontsize=9, color="gray")
        ax.text(end + 0.3, i, f"{end:.0f}", 
                ha="left", va="center", fontsize=9, color="gray")
    
    # Y-axis - phase names
    ax.set_yticks(y_positions)
    ax.set_yticklabels([p["phase"] for p in valid_phases], fontsize=11)
    ax.invert_yaxis()  # Top to bottom
    
    # X-axis
    min_year = min(p["start_year"] for p in valid_phases) - 2
    max_year = max(p["end_year"] for p in valid_phases) + 2
    ax.set_xlim(min_year, max_year)
    ax.set_xlabel("Year", fontweight="bold")
    
    # Grid lines for decades
    for decade in range(2050, 2110, 10):
        if min_year < decade < max_year:
            ax.axvline(x=decade, color="lightgray", linestyle="--", linewidth=0.8, alpha=0.7)
    
    # Project completion marker
    completion_year = max(p["end_year"] for p in valid_phases)
    ax.axvline(x=completion_year, color="red", linestyle="-", linewidth=2, alpha=0.8)
    ax.annotate(f"Completion\n{completion_year:.0f}", 
                xy=(completion_year, len(valid_phases) - 0.5),
                xytext=(5, 0), textcoords="offset points",
                fontsize=9, color="red", fontweight="bold",
                va="center")
    
    # 50-year deadline marker (2100)
    if min_year < 2100 < max_year:
        ax.axvline(x=2100, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
        ax.annotate("50-Year\nDeadline", xy=(2100, -0.3),
                    fontsize=8, color="gray", style="italic", ha="center")
    
    # Legend
    handles = [mpatches.Patch(color=c, label=l, alpha=0.85) 
               for l, c in phase_colors.items() if any(p["phase"] == l for p in valid_phases)]
    if handles:
        ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=9)
    
    # Title
    scenario_labels = {
        "E-only": "Elevator-only",
        "R-only": "Rocket-only",
        "Mix": "Hybrid (Scenario C)",
    }
    title = f"Optimal Construction Schedule — {scenario_labels.get(scenario, scenario)}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    
    # Clean up
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Gantt Chart for Construction Phases")
    parser.add_argument("results_dir", type=Path, help="Path to results directory")
    parser.add_argument("--scenario", default="Mix", help="Scenario to plot (default: Mix)")
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1
    
    phases = load_phases(args.results_dir, args.scenario)
    
    # Output to peper/figures
    output_dir = Path(__file__).parent.parent / "peper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "construction_gantt.png"
    
    plot_gantt(phases, output_path, args.scenario)
    print(f"\nTo include in LaTeX:\n  \\includegraphics[width=1\\linewidth]{{figures/construction_gantt.png}}")
    return 0


if __name__ == "__main__":
    exit(main())
