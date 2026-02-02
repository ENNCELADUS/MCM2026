import numpy as np
import matplotlib.pyplot as plt

# Set global font size to prevent overlap
plt.rcParams.update({"font.size": 9})

# =============================================================================
# Model Definitions (from constants.yaml)
# =============================================================================


def logistic_capacity(t, C_max, A, k, t0):
    """
    Logistic S-curve growth model for space elevator capacity.
    C_E(t) = C_max / (1 + A * exp(-k * (t - t0)))

    Physical justification:
    - C_max: Physical ceiling (~1 Mt/y) dictated by tether dynamics
    - A: Scaling factor solved from initial condition C_E(t0) = C_ref
    - k: Growth rate parameter reflecting infrastructure expansion
    - t0: Reference year (2050)
    """
    return C_max / (1 + A * np.exp(-k * (t - t0)))


def exponential_cost_decay(t, C_start, C_min, lam, t0):
    """
    Floor-constrained exponential decay model for unit cost.
    Cost(t) = (C_start - C_min) * exp(-lambda * (t - t0)) + C_min

    Economic justification:
    - C_start: Initial cost at t0 (learning curve beginning)
    - C_min: Operational floor (electricity + maintenance irreducible costs)
    - lambda: Decay rate reflecting amortization and efficiency gains
    """
    return (C_start - C_min) * np.exp(-lam * (t - t0)) + C_min


# =============================================================================
# Parameters (from constants.yaml and implementation_details.md)
# =============================================================================

# --- Capacity Model Parameters ---
# Physical derivation (implementation_details.md):
#   C_E_max = N_tethers * (m_load * v_climber / D_safe) * T_operation
#   N_tethers = 6 (3 harbours × 2 tethers)
#   m_load = 100 tons (single climber payload)
#   v_climber = 200 km/h ≈ 55.56 m/s
#   D_safe = 1000 km = 1e6 m (safety spacing for Coriolis resolution)
#   T_operation = 0.95 * 365 * 24 * 3600 s/y (95% availability)
#   Calculation: 6 * (100 * 55.56 / 1e6) * (0.95 * 31536000) ≈ 1,000,000 t/y

C_E_max = 1_000_000  # Physical ceiling (tons/year)
C_E_ref = 537_000  # Initial capacity at 2050 (3 × 179,000 t/y per MCM spec)
k_E = 0.10  # Growth rate parameter
t0_cap = 2050  # Reference year

# Solve for A from initial condition: C_E(t0) = C_E_ref
# C_E_ref = C_E_max / (1 + A) => A = (C_E_max / C_E_ref) - 1
A_cap = (C_E_max / C_E_ref) - 1
print(f"Solved A_cap for initial condition: {A_cap:.4f}")

# --- Cost Model Parameters ---
# Economic rationale (implementation_details.md):
#   - High-CAPEX (amortized by 2050), Low-OPEX
#   - Primary costs: electricity for climbers, tether maintenance
#   - Floor cost reflects irreducible operational expenses

C_start_cost = 50.0  # Initial cost at 2050 (USD/kg)
C_min_cost = 5.0  # Floor cost (USD/kg)
lambda_annual = 0.06  # Annual decay rate
t0_cost = 2050  # Reference year

# =============================================================================
# Generate Time Series
# =============================================================================

years = np.arange(2050, 2101)  # 2050 to 2100

# Capacity evolution
capacity = logistic_capacity(years, C_E_max, A_cap, k_E, t0_cap)

# Cost evolution
cost = exponential_cost_decay(years, C_start_cost, C_min_cost, lambda_annual, t0_cost)

# =============================================================================
# Print Model Parameters for constants.yaml
# =============================================================================

print("\n--- Space Elevator Capacity Model ---")
print(f"Model: C_E(t) = C_max / (1 + A * exp(-k * (t - t0)))")
print(f"\n--- For constants.yaml (elevator.capacity_logistic section) ---")
print(f"C_E_max_tpy: {C_E_max}        # Physical ceiling (tons/year)")
print(f"C_E_ref_tpy: {C_E_ref}       # Initial capacity at 2050")
print(f"k_E: {k_E:.2f}             # Growth rate parameter")
print(f"t0_year: {t0_cap}         # Reference year")
print(f"A: {A_cap:.4f}            # Solved from C_E(t0) = C_E_ref")

print(f"\n--- Capacity Predictions ---")
for y in [2050, 2060, 2070, 2080, 2090, 2100]:
    cap = logistic_capacity(y, C_E_max, A_cap, k_E, t0_cap)
    print(f"  Year {y}: {cap:,.0f} tons/year")

print("\n--- Space Elevator Cost Model ---")
print(f"Model: Cost(t) = (C_start - C_min) * exp(-lambda * (t - t0)) + C_min")
print(f"\n--- For constants.yaml (elevator.cost_decay section) ---")
print(f"initial_cost_usd_per_kg: {C_start_cost:.1f}   # Initial cost at 2050")
print(f"min_cost_usd_per_kg: {C_min_cost:.1f}        # Floor cost")
print(f"decay_rate_annual: {lambda_annual:.4f}   # Annual decay rate")

print(f"\n--- Cost Predictions ---")
for y in [2050, 2060, 2070, 2080, 2090, 2100]:
    c = exponential_cost_decay(y, C_start_cost, C_min_cost, lambda_annual, t0_cost)
    print(f"  Year {y}: ${c:.2f}/kg")

# =============================================================================
# Plotting (2x2 subplot grid like rocket_curve.py)
# =============================================================================

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
# fig.suptitle('Space Elevator System Evolution (2050-2100)', fontsize=14, fontweight='bold')

# [0, 0] Near-term Capacity Evolution (2050-2070)
years_short = np.arange(2050, 2071)
cap_short = logistic_capacity(years_short, C_E_max, A_cap, k_E, t0_cap)
axs[0, 0].plot(
    years_short, cap_short / 1e6, "b-", linewidth=2, label="Logistic Growth Model"
)
axs[0, 0].axhline(
    y=C_E_ref / 1e6,
    color="red",
    linestyle="--",
    alpha=0.6,
    label=f"Initial Capacity ({C_E_ref / 1e6:.3f} Mt/y)",
)
axs[0, 0].set_title("1-1 Near-Term Capacity Growth (2050-2070)")
axs[0, 0].set_ylabel("Capacity (Mt/year)")
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# [0, 1] Near-term Cost Decay (2050-2070)
cost_short = exponential_cost_decay(
    years_short, C_start_cost, C_min_cost, lambda_annual, t0_cost
)
axs[0, 1].plot(
    years_short, cost_short, "g-", linewidth=2, label="Exponential Decay Model"
)
axs[0, 1].axhline(
    y=C_start_cost,
    color="red",
    linestyle="--",
    alpha=0.6,
    label=f"Initial Cost (${C_start_cost}/kg)",
)
axs[0, 1].axhline(
    y=C_min_cost,
    color="black",
    linestyle="--",
    alpha=0.6,
    label=f"Cost Floor (${C_min_cost}/kg)",
)
axs[0, 1].set_title("1-2 Near-Term Cost Reduction (2050-2070)")
axs[0, 1].set_ylabel("Cost (USD/kg)")
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# [1, 0] Long-term Capacity Forecast (2050-2100)
axs[1, 0].plot(years, capacity / 1e6, "b-", linewidth=2, label="Logistic Growth Model")
axs[1, 0].axhline(
    y=C_E_max / 1e6,
    color="black",
    linestyle="--",
    alpha=0.6,
    label=f"Physical Ceiling ({C_E_max / 1e6:.1f} Mt/y)",
)
axs[1, 0].fill_between(years, 0, capacity / 1e6, alpha=0.2, color="blue")
axs[1, 0].set_title("1-3 Long-Term Capacity Forecast (2050-2100)")
axs[1, 0].set_ylabel("Capacity (Mt/year)")
axs[1, 0].set_ylim(0, C_E_max / 1e6 * 1.1)
axs[1, 0].legend(loc="lower right")
axs[1, 0].grid(True, alpha=0.3)

# [1, 1] Long-term Cost Forecast (2050-2100)
axs[1, 1].plot(years, cost, "g-", linewidth=2, label="Exponential Decay Model")
axs[1, 1].axhline(
    y=C_min_cost,
    color="black",
    linestyle="--",
    alpha=0.6,
    label=f"Cost Floor (${C_min_cost}/kg)",
)
axs[1, 1].fill_between(years, C_min_cost, cost, alpha=0.2, color="green")
axs[1, 1].set_title("1-4 Long-Term Cost Forecast (2050-2100)")
axs[1, 1].set_ylabel("Cost (USD/kg)")
axs[1, 1].set_ylim(0, C_start_cost * 1.1)
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

# Uniform X-axis labels
for ax in axs.flat:
    ax.set_xlabel("Year")

plt.tight_layout(pad=4.0)
output_path = "/Users/richardwang/Documents/MCM/peper/elevator_model.png"
plt.savefig(output_path, dpi=300)
print(f"\nPlot saved to {output_path}")
plt.show()
