#!/usr/bin/env python3
"""
Quick estimate of minimum feasible deadline based on transport capacity.
Does NOT consider ISRU growth, just raw Earth-to-Moon throughput.
"""
import math

# Target
TARGET_MASS_TONS = 100_000_000  # 100 Mt

# Elevator Capacity (Logistic S-curve)
C_E_max = 1_000_000  # t/y
C_E_ref = 537_000    # t/y at 2050
k_E = 0.15
t0_E = 2050

# Solve for A such that C_E(t0) = C_E_ref
# C_E_ref = C_E_max / (1 + A * exp(0)) => A = C_E_max / C_E_ref - 1
A_E = (C_E_max / C_E_ref) - 1

def elevator_capacity_tpy(year: float) -> float:
    """Elevator capacity in tons/year."""
    return C_E_max / (1 + A_E * math.exp(-k_E * (year - t0_E)))

# Rocket Capacity (Payload * Launch Rate)
L_max = 250  # t per launch
L_ref = 150  # t at 2050
k_L = 0.12
t0_L = 2050

# Solve A for payload
A_L = (L_max / L_ref - 1) * math.exp(k_L * (t0_L - t0_L))  # = L_max/L_ref - 1

def rocket_payload_t(year: float) -> float:
    """Rocket payload per launch in tons."""
    A = (L_max / L_ref - 1)
    return L_max / (1 + A * math.exp(-k_L * (year - t0_L)))

# Launch Rate (Logistic)
K_rate = 35274  # max launches/year
A_rate = 1000
r_rate = 0.0991
t0_rate = 2005

def launch_rate_per_year(year: float) -> float:
    """Max launches per year."""
    return K_rate / (1 + A_rate * math.exp(-r_rate * (year - t0_rate)))

def rocket_capacity_tpy(year: float) -> float:
    """Total rocket capacity in tons/year."""
    return rocket_payload_t(year) * launch_rate_per_year(year)

def total_capacity_tpy(year: float) -> float:
    """Combined elevator + rocket capacity."""
    return elevator_capacity_tpy(year) + rocket_capacity_tpy(year)

# Simulate cumulative mass delivered
def find_deadline(target: float, start_year: int = 2050, max_years: int = 200) -> tuple[int, float]:
    cumulative = 0.0
    for y in range(start_year, start_year + max_years):
        cap = total_capacity_tpy(y)
        cumulative += cap
        if cumulative >= target:
            return y, cumulative
    return start_year + max_years, cumulative

if __name__ == "__main__":
    print("=== Transport Capacity Estimation ===")
    print(f"Target Mass: {TARGET_MASS_TONS:,.0f} tons ({TARGET_MASS_TONS/1e6:.0f} Mt)")
    print()
    
    # Sample capacities
    for year in [2050, 2060, 2070, 2080, 2100]:
        e_cap = elevator_capacity_tpy(year)
        r_cap = rocket_capacity_tpy(year)
        print(f"{year}: Elevator {e_cap:,.0f} t/y | Rocket {r_cap:,.0f} t/y | Total {e_cap+r_cap:,.0f} t/y")
    
    print()
    
    # Find deadline for different scenarios
    print("=== Minimum Deadline (Pure Transport, No ISRU) ===")
    
    # Elevator Only
    cumulative = 0.0
    for y in range(2050, 2250):
        cumulative += elevator_capacity_tpy(y)
        if cumulative >= TARGET_MASS_TONS:
            print(f"Elevator Only: {y} ({y-2050} years)")
            break
    
    # Rocket Only  
    cumulative = 0.0
    for y in range(2050, 2250):
        cumulative += rocket_capacity_tpy(y)
        if cumulative >= TARGET_MASS_TONS:
            print(f"Rocket Only: {y} ({y-2050} years)")
            break
    
    # Combined
    deadline, final_mass = find_deadline(TARGET_MASS_TONS)
    print(f"Combined (E+R): {deadline} ({deadline-2050} years)")
    
    print()
    print("=== With ISRU (Rough Estimate) ===")
    print("If ISRU provides 70% of mass locally after bootstrapping (~10 years),")
    print("then only 30% needs to be shipped (30 Mt).")
    deadline_isru, _ = find_deadline(30_000_000)
    print(f"Combined (30 Mt shipped): {deadline_isru} ({deadline_isru-2050} years)")
