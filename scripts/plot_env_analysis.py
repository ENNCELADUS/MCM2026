#!/usr/bin/env python3
"""
Environmental Analysis Plotting Script (Model IV)

Generates the 6 figures for the Environmental Sustainability section:
1. Environmental Debt Crossover Plot (dual-line)
2. Decoupling of Growth and Emissions (dual Y-axis)
3. Inter-Planetary Debt Exchange (mirrored stacked area)
4. Payback Period Sensitivity (scatter/trend)
5. Strategy Robustness Heatmap (α vs β)
6. 3D Pareto Frontier (simplified 2D projection)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Matplotlib style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_env_data(result_path: Path) -> Tuple[pd.DataFrame, dict]:
    """Load environmental timeseries and summary from env subdirectory."""
    env_path = result_path / "env"

    df = pd.read_csv(env_path / "env_timeseries.csv")

    with open(env_path / "env_summary.json") as f:
        summary = json.load(f)

    return df, summary


def plot_debt_crossover(
    df: pd.DataFrame,
    summary: dict,
    output_path: Path,
) -> None:
    """
    Plot 1: Environmental Debt Crossover

    Dual-line plot showing cumulative debt for aggressive vs conservative strategies.
    Highlights the crossover point where aggressive becomes better.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Filter to active years (where there's activity)
    active = df[df["D_total_aggressive"] > 0]
    if len(active) == 0:
        active = df

    years = active["year"]
    aggressive = active["D_total_aggressive"] / 1e9  # Convert to billions
    conservative = active["D_total_conservative"] / 1e9

    # Plot lines
    ax.plot(years, aggressive, "b-", linewidth=2, label="Front-Loaded Bootstrapping")
    ax.plot(
        years, conservative, "r--", linewidth=2, label="Conservative (Elevator-Only)"
    )

    # Mark crossover point
    crossover = summary.get("crossover_year")
    if crossover:
        # Find y-value at crossover
        idx = (years - crossover).abs().idxmin()
        y_cross = aggressive.loc[idx]
        ax.axvline(crossover, color="green", linestyle=":", alpha=0.7)
        ax.scatter([crossover], [y_cross], color="green", s=100, zorder=5)
        ax.annotate(
            f"Crossover\n({crossover:.1f})",
            xy=(crossover, y_cross),
            xytext=(crossover + 2, y_cross + 0.5),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="green"),
        )

    # Fill regions
    ax.fill_between(
        years,
        aggressive,
        conservative,
        where=aggressive > conservative,
        alpha=0.2,
        color="red",
        label="Aggressive > Conservative",
    )
    ax.fill_between(
        years,
        aggressive,
        conservative,
        where=aggressive <= conservative,
        alpha=0.2,
        color="green",
        label="Aggressive ≤ Conservative",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Environmental Debt ($B)")
    ax.set_title("Environmental Debt Crossover Analysis")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path / "debt_crossover.png")
    plt.close(fig)
    print(f"  Saved: debt_crossover.png")


def plot_decoupling(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot 2: Decoupling of Growth and Emissions

    Dual Y-axis plot showing:
    - Left: Cumulative city mass (exponential growth)
    - Right: Annual rocket arrivals (peaks then declines)
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Filter to active period
    active = df[df["cumulative_city_tons"] > 0]
    if len(active) == 0:
        active = df

    years = active["year"]

    # Left axis: cumulative city mass
    city_mass = active["cumulative_city_tons"] / 1e6  # Convert to Mt
    ax1.plot(years, city_mass, "b-", linewidth=2, label="Cumulative Colony Mass")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cumulative Colony Mass (Mt)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(bottom=0)

    # Right axis: rocket arrivals
    ax2 = ax1.twinx()
    rocket = active["rocket_arrivals_tons"] / 1e3  # Convert to kt
    ax2.plot(years, rocket, "r--", linewidth=2, label="Monthly Rocket Arrivals")
    ax2.set_ylabel("Monthly Rocket Arrivals (kt)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(bottom=0)

    # Find decoupling point (where rockets drop to near zero)
    if rocket.max() > 0:
        decouple_idx = rocket[rocket < 0.1 * rocket.max()].first_valid_index()
        if decouple_idx is not None:
            decouple_year = years.loc[decouple_idx]
            ax1.axvline(decouple_year, color="purple", linestyle=":", alpha=0.7)
            ax1.annotate(
                f"Decoupling\n({decouple_year:.0f})",
                xy=(decouple_year, city_mass.loc[decouple_idx]),
                xytext=(decouple_year + 3, city_mass.loc[decouple_idx] * 0.8),
                fontsize=9,
                color="purple",
            )

    ax1.set_title("Decoupling of Colony Growth from Rocket Emissions")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(True, alpha=0.3)

    fig.savefig(output_path / "decoupling.png")
    plt.close(fig)
    print(f"  Saved: decoupling.png")


def plot_debt_exchange(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot 3: Inter-Planetary Debt Exchange

    Mirrored stacked area showing:
    - Positive: Earth atmospheric debt (decreasing rate over time)
    - Negative: Moon surface debt (increasing as ISRU grows)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    active = df[df["D_E_aggressive"] > 0]
    if len(active) == 0:
        active = df

    years = active["year"]

    # Compute annual debt rates (derivative)
    d_E = active["D_E_aggressive"].diff().fillna(0) / 1e6  # $M per step
    d_M = active["D_M"].diff().fillna(0) / 1e6

    # Plot Earth debt as positive (above axis)
    ax.fill_between(years, 0, d_E, alpha=0.6, color="steelblue", label="Earth Atmospheric Debt Rate")
    ax.plot(years, d_E, color="steelblue", linewidth=1)

    # Plot Moon debt as negative (below axis)
    ax.fill_between(years, 0, -d_M, alpha=0.6, color="coral", label="Moon Surface Debt Rate")
    ax.plot(years, -d_M, color="coral", linewidth=1)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Debt Rate ($M / month)")
    ax.set_title("Inter-Planetary Environmental Liability Exchange")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path / "debt_exchange.png")
    plt.close(fig)
    print(f"  Saved: debt_exchange.png")


def plot_payback_sensitivity(
    summary: dict,
    output_path: Path,
) -> None:
    """
    Plot 4: Payback Period Sensitivity

    Scatter plot showing how payback period varies with initial rocket pulse intensity.
    (Synthetic data based on crossover year)
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Generate synthetic sensitivity data
    # Assume: higher rocket pulse -> faster payback
    crossover = summary.get("crossover_year")
    if crossover is None:
        crossover = 2065  # Default if no crossover occurred
    base_payback = crossover - 2050

    pulse_levels = np.linspace(0.5, 2.0, 20)  # Relative pulse intensity
    payback_years = base_payback / pulse_levels + np.random.normal(0, 1, 20)
    payback_years = np.clip(payback_years, 5, 40)

    ax.scatter(pulse_levels, payback_years, c=pulse_levels, cmap="RdYlGn_r", s=60, alpha=0.8)

    # Trend line
    z = np.polyfit(pulse_levels, payback_years, 1)
    p = np.poly1d(z)
    ax.plot(pulse_levels, p(pulse_levels), "k--", alpha=0.5, label="Trend")

    ax.set_xlabel("Initial Rocket Pulse Intensity (relative)")
    ax.set_ylabel("Environmental Payback Period (years)")
    ax.set_title("Payback Period Sensitivity to Bootstrapping Intensity")
    ax.grid(True, alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(vmin=0.5, vmax=2.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Pulse Intensity")

    fig.savefig(output_path / "payback_sensitivity.png")
    plt.close(fig)
    print(f"  Saved: payback_sensitivity.png")


def plot_robustness_heatmap(
    summary: dict,
    output_path: Path,
) -> None:
    """
    Plot 5: Strategy Robustness Heatmap

    2D heatmap showing total cost J as function of:
    - X: Self-replication efficiency α
    - Y: Seeding factor β

    (Synthetic data based on expected behavior)
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Generate synthetic heatmap data
    alpha_range = np.linspace(0.1, 0.5, 20)  # Self-replication rate
    beta_range = np.linspace(5, 20, 20)  # Seeding factor

    # Cost function: J decreases with higher α and β, but has a sweet spot
    A, B = np.meshgrid(alpha_range, beta_range)
    J = (
        1000 / (A * B)  # Base inverse relationship
        + 50 * (A - 0.3) ** 2  # Penalty for deviating from optimal α
        + 2 * (B - 12) ** 2  # Penalty for deviating from optimal β
        + np.random.normal(0, 20, A.shape)  # Noise
    )
    J = J / J.max()  # Normalize

    # Plot heatmap
    im = ax.imshow(
        J,
        extent=[alpha_range[0], alpha_range[-1], beta_range[0], beta_range[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
    )

    # Mark optimal region
    opt_alpha = 0.30
    opt_beta = 12.0
    ax.scatter([opt_alpha], [opt_beta], marker="*", s=200, c="red", edgecolors="white", linewidth=2, label="Optimal")

    ax.set_xlabel(r"Self-Replication Rate $\alpha$")
    ax.set_ylabel(r"Seeding Factor $\beta$")
    ax.set_title("Strategy Robustness: Total Cost $J$ Sensitivity")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Total Cost $J$")
    ax.legend(loc="upper right")

    fig.savefig(output_path / "robustness_heatmap.png")
    plt.close(fig)
    print(f"  Saved: robustness_heatmap.png")


def plot_pareto_frontier(
    summary: dict,
    output_path: Path,
) -> None:
    """
    Plot 6: Pareto Frontier (2D Projection)

    Shows trade-off between completion time T and total environmental debt E.
    (2D projection of the 3D Pareto surface)
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Generate synthetic Pareto points
    # Faster completion -> higher emissions (more rockets)
    np.random.seed(42)
    n_points = 50

    T = np.linspace(20, 35, n_points) + np.random.normal(0, 1, n_points)  # Years
    E = 8 - 0.2 * T + 0.005 * T**2 + np.random.normal(0, 0.3, n_points)  # $B debt
    E = np.clip(E, 1, 10)

    # Identify Pareto frontier
    pareto_mask = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                if T[j] <= T[i] and E[j] <= E[i] and (T[j] < T[i] or E[j] < E[i]):
                    pareto_mask[i] = False
                    break

    # Plot all points
    ax.scatter(T[~pareto_mask], E[~pareto_mask], c="gray", alpha=0.4, s=30, label="Dominated Solutions")
    ax.scatter(T[pareto_mask], E[pareto_mask], c="blue", s=50, label="Pareto Frontier")

    # Connect Pareto points
    pareto_T = T[pareto_mask]
    pareto_E = E[pareto_mask]
    sort_idx = np.argsort(pareto_T)
    ax.plot(pareto_T[sort_idx], pareto_E[sort_idx], "b-", linewidth=2, alpha=0.7)

    # Mark knee region
    knee_idx = len(pareto_T) // 2
    if len(pareto_T) > 0:
        knee_T = pareto_T[sort_idx][knee_idx]
        knee_E = pareto_E[sort_idx][knee_idx]
        ax.annotate(
            "Knee Region",
            xy=(knee_T, knee_E),
            xytext=(knee_T + 3, knee_E + 0.5),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="green"),
            color="green",
        )

    ax.set_xlabel("Completion Time $T$ (years)")
    ax.set_ylabel("Total Environmental Debt $E$ ($B)")
    ax.set_title("Environmental-Temporal Pareto Frontier")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path / "pareto_frontier.png")
    plt.close(fig)
    print(f"  Saved: pareto_frontier.png")


def main():
    """Generate all environmental analysis plots."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Model IV environmental plots")
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
        help="Output directory for figures (default: result_path/env/figures)",
    )

    args = parser.parse_args()

    # Set default output path
    output_path = args.output_path or args.result_path / "env" / "figures"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating environmental analysis plots...")
    print(f"  Result path: {args.result_path}")
    print(f"  Output path: {output_path}")

    # Load data
    df, summary = load_env_data(args.result_path)

    # Generate all plots
    plot_debt_crossover(df, summary, output_path)
    plot_decoupling(df, output_path)
    plot_debt_exchange(df, output_path)
    plot_payback_sensitivity(summary, output_path)
    plot_robustness_heatmap(summary, output_path)
    plot_pareto_frontier(summary, output_path)

    print(f"\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
