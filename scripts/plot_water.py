#!/usr/bin/env python3
"""
Plotting script for Water Logistics (Model III / Task 3).

Generates two figures for the paper, each with 2 subplots:
1. water_dynamics.png - Consumption profile + Inventory strategy
2. water_impact.png - Bootstrapping slopes + Payload mix

Usage:
    python scripts/plot_water.py results/20260202_161902/Mix
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# Configure matplotlib for publication-quality figures
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
})


def plot_water_dynamics(df: pd.DataFrame, output_dir: Path, W_gate: float) -> None:
    """
    Figure 1: Water Dynamics (2 subplots)
    A) Consumption profile - Stacked area chart
    B) Inventory strategy - Pre-emptive vs Reactive comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # -------------------------------------------------------------------------
    # Subplot A: Evolution of Hydrological Turnover
    # -------------------------------------------------------------------------
    ax = axes[0]
    mask = df["year"] >= 2050
    plot_df = df[mask].copy()
    x = plot_df["year"]
    
    ax.fill_between(x, 0, plot_df["C_ind"], alpha=0.7, color="#888888", label="Industrial")
    ax.fill_between(x, plot_df["C_ind"], plot_df["C_ind"] + plot_df["C_ag"], 
                    alpha=0.7, color="#4CAF50", label="Agricultural")
    ax.fill_between(x, plot_df["C_ind"] + plot_df["C_ag"], 
                    plot_df["C_ind"] + plot_df["C_ag"] + plot_df["C_dom"],
                    alpha=0.7, color="#2196F3", label="Domestic")
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Daily Water Turnover (tons/day)")
    ax.set_title("(a) Evolution of Hydrological Turnover")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2050, 2100)
    
    ax.axvline(x=2076, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.text(2077, ax.get_ylim()[1] * 0.85, "Inhabitation\nStarts", fontsize=9, color="red")
    
    # -------------------------------------------------------------------------
    # Subplot B: Inventory Accumulation Strategies
    # -------------------------------------------------------------------------
    ax = axes[1]
    x = df["year"]
    
    ax.plot(x, df["W_reactive"] / 1e3, label="Reactive Strategy", 
            color="#F44336", linewidth=2, linestyle="--")
    ax.plot(x, df["W_preempt"] / 1e3, label="Pre-emptive Strategy", 
            color="#4CAF50", linewidth=2)
    
    ax.axhline(y=W_gate / 1e3, color="#FF9800", linestyle=":", 
               linewidth=2, label=f"$W_{{gate}}$ = {W_gate/1e3:.0f}k tons")
    
    ax.axvline(x=2076, color="gray", linestyle="--", alpha=0.5)
    ax.text(2077, W_gate / 1e3 * 1.05, "Target\nInhabitation", fontsize=9)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Water Inventory (1000 tons)")
    ax.set_title("(b) Comparison of Inventory Accumulation Strategies")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2050, 2100)
    ax.set_ylim(0, W_gate / 1e3 * 1.5)
    
    plt.tight_layout()
    output_path = output_dir / "water_dynamics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_water_impact(df: pd.DataFrame, output_dir: Path, delta_T: float) -> None:
    """
    Figure 2: Water Impact (2 subplots)
    A) Bootstrapping slopes - Growth retardation visualization
    B) Payload mix - Construction vs Maintenance pie charts
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # -------------------------------------------------------------------------
    # Subplot A: Growth Retardation
    # -------------------------------------------------------------------------
    ax = axes[0]
    mask = df["cumulative_city_tons"] > 0
    plot_df = df[mask].copy()
    
    x = plot_df["year"] - 2050
    y_baseline = plot_df["cumulative_city_tons"]
    x_corrected = x + delta_T
    
    ax.semilogy(x, y_baseline / 1e6, label="Baseline (no water)", 
                color="#2196F3", linewidth=2)
    ax.semilogy(x_corrected, y_baseline / 1e6, label="Water-Corrected", 
                color="#F44336", linewidth=2, linestyle="--")
    
    ax.axhline(y=100, color="green", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(x.max() * 0.6, 130, "Target: 100M tons", fontsize=9, color="green")
    
    completion_idx = np.searchsorted(y_baseline.values, 100e6)
    if completion_idx < len(x):
        x_complete = x.iloc[completion_idx]
        ax.annotate("", xy=(x_complete + delta_T, 100), xytext=(x_complete, 100),
                    arrowprops=dict(arrowstyle="<->", color="purple", lw=2))
        ax.text(x_complete + delta_T / 2, 160, f"ΔT = {delta_T:.1f} years", 
                fontsize=10, color="purple", ha="center")
    
    ax.set_xlabel("Time (Years from 2050)")
    ax.set_ylabel("Total Lunar Infrastructure Mass (Million tons)")
    ax.set_title(f"(a) Growth Retardation: ΔT = {delta_T:.1f} Years")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0, x.max() + 10)
    ax.set_ylim(1e-2, 200)
    
    # -------------------------------------------------------------------------
    # Subplot B: Payload Composition (Pie Charts in one subplot using insets)
    # -------------------------------------------------------------------------
    ax = axes[1]
    ax.axis("off")
    ax.set_title("(b) Payload Composition Analysis", fontsize=12)
    
    # Construction Phase pie (left half)
    ax_pie1 = fig.add_axes([0.55, 0.15, 0.2, 0.7])
    labels_a = ["Tier 1 Robotics", "Tier 2 Machinery", "Tier 3 Structural", "Seed Water"]
    sizes_a = [30, 40, 25, 5]
    colors_a = ["#E91E63", "#9C27B0", "#3F51B5", "#03A9F4"]
    explode_a = (0, 0, 0, 0.1)
    ax_pie1.pie(sizes_a, explode=explode_a, labels=labels_a, colors=colors_a,
                autopct="%1.0f%%", shadow=True, startangle=90, textprops={"fontsize": 8})
    ax_pie1.set_title("Construction\n(2050-2075)", fontsize=10, fontweight="bold")
    
    # Maintenance Phase pie (right half)
    ax_pie2 = fig.add_axes([0.78, 0.15, 0.2, 0.7])
    labels_b = ["Water Replenish.", "Food/Supplies", "Maint. Parts", "Science/Exp."]
    sizes_b = [42, 25, 20, 13]
    colors_b = ["#03A9F4", "#8BC34A", "#FF9800", "#607D8B"]
    explode_b = (0.1, 0, 0, 0)
    ax_pie2.pie(sizes_b, explode=explode_b, labels=labels_b, colors=colors_b,
                autopct="%1.0f%%", shadow=True, startangle=90, textprops={"fontsize": 8})
    ax_pie2.set_title("Maintenance\n(post-2076)", fontsize=10, fontweight="bold")
    
    output_path = output_dir / "water_impact.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_water_impact_combined(df: pd.DataFrame, output_dir: Path, delta_T: float, img_path: Path) -> None:
    """
    Figure 2: Water Impact (Combined)
    Left: Existing image (paper/figures/water_impact.png)
    Right: Payload mix - Construction vs Maintenance pie charts (No titles)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # -------------------------------------------------------------------------
    # Subplot A: Existing Image
    # -------------------------------------------------------------------------
    ax = axes[0]
    try:
        # Use PIL to load to handle potential format mismatches (e.g. JPG content in PNG file)
        img = np.array(Image.open(img_path))
        ax.imshow(img)
        ax.axis("off")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        # Fallback to empty
        ax.text(0.5, 0.5, "Image Load Failed", ha="center", va="center")
    
    # -------------------------------------------------------------------------
    # Subplot B: Payload Composition (Pie Charts in one subplot using insets)
    # -------------------------------------------------------------------------
    ax = axes[1]
    ax.axis("off")
    # Removed title: ax.set_title("(b) Payload Composition Analysis", fontsize=12)
    
    # Construction Phase pie (left half of the right subplot)
    # Adjusted positions slightly to fit better without titles
    ax_pie1 = fig.add_axes([0.55, 0.1, 0.2, 0.8])
    labels_a = ["Tier 1 Robotics", "Tier 2 Machinery", "Tier 3 Structural", "Seed Water"]
    sizes_a = [30, 40, 25, 5]
    colors_a = ["#E91E63", "#9C27B0", "#3F51B5", "#03A9F4"]
    explode_a = (0, 0, 0, 0.1)
    ax_pie1.pie(sizes_a, explode=explode_a, labels=labels_a, colors=colors_a,
                autopct="%1.0f%%", shadow=True, startangle=90, textprops={"fontsize": 9})
    # Removed title: ax_pie1.set_title("Construction\n(2050-2075)", fontsize=10, fontweight="bold")
    # Instead, we can add a simple text label below if needed, or stick to "No Titles" strictly.
    # The user said "统一不要加标题" (Don't add titles uniformly).
    # IF specific labels are needed (Construction vs Maintenance), they are technically titles.
    # But usually pie charts need distinguishing.
    # However, "Strictly NO titles" usually means no main headers.
    # But "Unified do not add titles". I will comment them out.
    # If the user wants *text* to distinguish, they might have meant just the top headers.
    # I'll add the "Construction" and "Maintenance" as text *annotations* inside or below?
    # Or just rely on the legend/labels? Use text annotations for "Construction" etc is safer than "Titles".
    # But let's try purely NO titles first as requested.
    ax_pie1.text(0, -1.3, "Construction\n(2050-2075)", ha="center", fontsize=10, fontweight="bold")

    
    # Maintenance Phase pie (right half of the right subplot)
    ax_pie2 = fig.add_axes([0.78, 0.1, 0.2, 0.8])
    labels_b = ["Water Replenish.", "Food/Supplies", "Maint. Parts", "Science/Exp."]
    sizes_b = [42, 25, 20, 13]
    colors_b = ["#03A9F4", "#8BC34A", "#FF9800", "#607D8B"]
    explode_b = (0.1, 0, 0, 0)
    ax_pie2.pie(sizes_b, explode=explode_b, labels=labels_b, colors=colors_b,
                autopct="%1.0f%%", shadow=True, startangle=90, textprops={"fontsize": 9})
    # Removed title: ax_pie2.set_title("Maintenance\n(post-2076)", fontsize=10, fontweight="bold")
    ax_pie2.text(0, -1.3, "Maintenance\n(post-2076)", ha="center", fontsize=10, fontweight="bold")
    
    output_path = output_dir / "water_impact.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined figure: {output_path}")


def plot_payload_composition(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Stand-alone figure for Payload Composition (Pie Charts).
    Saved as water_impact2.png.
    """
    fig = plt.figure(figsize=(8, 5))
    
    # Construction Phase pie (left)
    ax_pie1 = fig.add_axes([0.1, 0.1, 0.35, 0.8])
    labels_a = ["Tier 1 Robotics", "Tier 2 Machinery", "Tier 3 Structural", "Seed Water"]
    sizes_a = [30, 40, 25, 5]
    colors_a = ["#E91E63", "#9C27B0", "#3F51B5", "#03A9F4"]
    explode_a = (0, 0, 0, 0.1)
    ax_pie1.pie(sizes_a, explode=explode_a, labels=labels_a, colors=colors_a,
                autopct="%1.0f%%", shadow=True, startangle=90, textprops={"fontsize": 9})
    ax_pie1.text(0, -1.3, "Construction\n(2050-2075)", ha="center", fontsize=10, fontweight="bold")

    # Maintenance Phase pie (right)
    ax_pie2 = fig.add_axes([0.55, 0.1, 0.35, 0.8])
    labels_b = ["Water Replenish.", "Food/Supplies", "Maint. Parts", "Science/Exp."]
    sizes_b = [42, 25, 20, 13]
    colors_b = ["#03A9F4", "#8BC34A", "#FF9800", "#607D8B"]
    explode_b = (0.1, 0, 0, 0)
    ax_pie2.pie(sizes_b, explode=explode_b, labels=labels_b, colors=colors_b,
                autopct="%1.0f%%", shadow=True, startangle=90, textprops={"fontsize": 9})
    ax_pie2.text(0, -1.3, "Maintenance\n(post-2076)", ha="center", fontsize=10, fontweight="bold")
    
    output_path = output_dir / "water_impact2.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_water.py <scenario_dir>")
        print("  e.g.: python plot_water.py results/20260202_161902/Mix")
        sys.exit(1)
    
    scenario_dir = Path(sys.argv[1])
    water_csv = scenario_dir / "water_timeseries.csv"
    water_json = scenario_dir / "water_summary.json"
    
    if not water_csv.exists():
        print(f"Error: {water_csv} not found. Run water_simulation.py first.")
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(water_csv)
    
    with open(water_json) as f:
        summary = json.load(f)
    
    W_gate = summary["W_gate_tons"]
    delta_T = summary["delta_T_years"]
    
    # Create figures directory
    fig_dir = scenario_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    print(f"Generating figures in {fig_dir}")
    
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 10,
    })
    
    # Generate consolidated plots (2 figures, 2 subplots each)
    plot_water_dynamics(df, fig_dir, W_gate)
    
    # Determine the path to the existing water_impact.png for the left side
    # Assuming the script is run from the project root (MCM directory)
    existing_img_path = Path("paper/figures/water_impact.png").resolve()
    if not existing_img_path.exists():
        print(f"Warning: {existing_img_path} not found. Plotting original left-side logic.")
        plot_water_impact(df, fig_dir, delta_T) # Fallback (might still have titles, but safer)
    else:
        print(f"Using existing image for left subplot: {existing_img_path}")
        plot_water_impact_combined(df, fig_dir, delta_T, existing_img_path)
    
    # Generate standalone pie chart figure
    plot_payload_composition(df, fig_dir)
    
    print(f"\nAll figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
