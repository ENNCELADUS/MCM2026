from __future__ import annotations

import math
from typing import Any
import pyomo.environ as pyo

from entities import ModelData
import utils
from config.settings import ModelSettings


class PyomoBuilder:
    """Builder class for the Pyomo MILP model."""

    def __init__(self, data: ModelData, settings: ModelSettings, constants: Any):
        self.data = data
        self.settings = settings
        self.constants = constants
        self.m = pyo.ConcreteModel(name="MoonLogisticsMILP")

        # Precompute constants and lookups
        self._precompute()

    def build(self) -> pyo.ConcreteModel:
        """Construct and return the Pyomo model."""
        self._create_sets()
        self._create_params()
        self._create_variables()
        self._create_constraints()
        self._create_objective()
        return self.m

    def _precompute(self):
        data = self.data
        constants = self.constants

        self.node_ids = [n.id for n in data.nodes]
        self.node_types = {n.id: n.node_type for n in data.nodes}
        self.arc_ids = [a.id for a in data.arcs]
        self.res_ids = [r.id for r in data.resources]

        # Growth Model Parameters
        growth = data.growth_params
        self.bootstrapping = growth["phases"]["bootstrapping"]
        self.replication = growth["phases"]["replication"]
        self.saturation = growth["phases"]["saturation"]
        self.initial_state = growth["initial_state"]

        self.arcs_by_id = {a.id: a for a in data.arcs}
        self.res_by_id = {r.id: r for r in data.resources}

        self.arcs_from = {n: [] for n in self.node_ids}
        self.arcs_to = {n: [] for n in self.node_ids}
        for a in data.arcs:
            self.arcs_from[a.from_node].append(a.id)
            self.arcs_to[a.to_node].append(a.id)

        self.elevator_arcs = [a.id for a in data.arcs if a.arc_type == "elevator"]
        self.rocket_arcs = [
            a.id for a in data.arcs if a.arc_type in ("rocket", "transfer")
        ]
        self.launch_arcs = [a.id for a in data.arcs if a.arc_type == "rocket"]
        self.ground_arcs = [a.id for a in data.arcs if a.arc_type == "ground"]
        self.arcs_to_moon = self.arcs_to["Moon"]

        self.moon_node = "Moon"
        if self.moon_node not in self.node_ids:
            raise ValueError("Expected Moon node id 'Moon' in constants.yaml nodes")

        self.big_m = constants["optimization"]["big_m"]
        self.delta_t = constants["time"]["delta_t"]
        self.steps_per_year = constants["time"]["steps_per_year"]
        self.ton_to_kg = constants["units"]["ton_to_kg"]

        self.ton_to_kg = constants["units"]["ton_to_kg"]

        # self.C_E_0 = constants["initial_capacities"]["C_E_0"] # Removed

        # Get fixed elevator capacity
        elevator_capacity_tpy = utils.get_elevator_capacity_tpy(0, constants)

        # Convert to per-step capacity in mass/second
        self.elevator_capacity_fixed_mass_s = (
            elevator_capacity_tpy
            * self.ton_to_kg
            / (self.steps_per_year * self.delta_t)
        )
        # Enforce consistency: C_E_0 matches calculated capacity
        self.C_E_0 = self.elevator_capacity_fixed_mass_s

        self.enable_learning_curve = self.settings.enable_learning_curve
        # Stream model strictly enforced by default now
        self.stream_model = True

        self.total_demand_kg = (
            constants["parameter_summary"]["materials"]["bom"]["total_demand_tons"]
            * self.ton_to_kg
        )
        self.target_pop = constants["parameter_summary"]["colony"]["target"][
            "population"
        ]
        self.deadline_year = constants["parameter_summary"]["colony"]["target"].get(
            "deadline_year"
        )

        # BOM Mappings
        self.tiers = utils.get_tier_definitions(constants)
        self.res_tier_map = {}
        for tier in self.tiers:
            for rid in tier["resources"]:
                self.res_tier_map[rid] = tier["id"]

        # Identify Tier 1 (High-Tech/Equipment) for Growth
        self.tier1_res = [
            r for r in self.res_ids if self.res_tier_map.get(r) == "tier_1"
        ]
        # Fallback
        if not self.tier1_res:
            self.tier1_res = [r for r in self.res_ids if "electronics" in r]

        # Water Resource for Population
        self.water_res = next(
            (r for r in self.res_ids if "water" in r or "volatile" in r), None
        )
        self.water_per_capita = float(
            constants["parameter_summary"]["colony"]
            .get("requirements", {})
            .get("water_per_capita_per_month", 0.5)
        )

        # Growth BOM (resource mix for delta_Growth consumption)
        growth_bom_cfg = (
            constants.get("implementation_details", {})
            .get("growth_model", {})
            .get("growth_bom", {})
        )
        self.growth_bom: dict[str, float] = {}
        for rid, share in growth_bom_cfg.items():
            if rid in self.res_ids:
                self.growth_bom[rid] = float(share)
        if not self.growth_bom and "structure" in self.res_ids:
            self.growth_bom = {"structure": 1.0}

    def _create_sets(self):
        m = self.m
        settings = self.settings

        m.T = pyo.RangeSet(0, settings.T_horizon - 1)
        m.R = pyo.Set(initialize=self.res_ids, ordered=True)
        m.N = pyo.Set(initialize=self.node_ids, ordered=True)
        m.A = pyo.Set(initialize=self.arc_ids, ordered=True)

        m.A_Elev = pyo.Set(initialize=self.elevator_arcs)
        m.A_Rock = pyo.Set(initialize=self.rocket_arcs)
        m.A_Launch = pyo.Set(initialize=self.launch_arcs)
        m.A_Ground = pyo.Set(initialize=self.ground_arcs)
        m.A_to_Moon = pyo.Set(initialize=self.arcs_to_moon)

    def _create_params(self):
        m = self.m
        data = self.data
        settings = self.settings

        # Arc Params
        arc_from = {a.id: a.from_node for a in data.arcs}
        arc_to = {a.id: a.to_node for a in data.arcs}
        arc_lead = {a.id: int(a.lead_time) for a in data.arcs}
        arc_payload = {a.id: float(a.payload) for a in data.arcs}
        arc_type = {a.id: a.arc_type for a in data.arcs}

        arc_cost = {}
        for a_id in self.arc_ids:
            a = self.arcs_by_id[a_id]
            for t_idx in range(settings.T_horizon):
                if a.arc_type in ("rocket", "transfer") and self.enable_learning_curve:
                    cost = utils.get_rocket_cost_usd_per_kg(t_idx, self.constants)
                    arc_cost[(a_id, t_idx)] = float(cost)
                elif a.arc_type == "elevator" and self.enable_learning_curve:
                    cost = utils.get_elevator_cost_usd_per_kg(t_idx, self.constants)
                    arc_cost[(a_id, t_idx)] = float(cost)
                else:
                    arc_cost[(a_id, t_idx)] = float(a.cost_per_kg_2050)

        rocket_payload = {}
        rocket_launch_rate = {}
        for t_idx in range(settings.T_horizon):
            rocket_payload[t_idx] = utils.get_rocket_payload_kg(t_idx, self.constants)
            rocket_launch_rate[t_idx] = utils.get_rocket_launch_rate_max(
                t_idx, self.constants
            )

        m.arc_from = pyo.Param(m.A, initialize=arc_from, within=pyo.Any)
        m.arc_to = pyo.Param(m.A, initialize=arc_to, within=pyo.Any)
        m.arc_lead = pyo.Param(m.A, initialize=arc_lead, within=pyo.NonNegativeIntegers)
        m.arc_payload = pyo.Param(
            m.A, initialize=arc_payload, within=pyo.NonNegativeReals
        )
        m.arc_type = pyo.Param(m.A, initialize=arc_type, within=pyo.Any)
        m.arc_cost = pyo.Param(
            m.A, m.T, initialize=arc_cost, within=pyo.NonNegativeReals
        )
        m.rocket_payload = pyo.Param(
            m.T, initialize=rocket_payload, within=pyo.NonNegativeReals
        )
        m.rocket_launch_rate = pyo.Param(
            m.T, initialize=rocket_launch_rate, within=pyo.NonNegativeReals
        )

        m.isru_ok = pyo.Param(
            m.R,
            initialize={
                r: 1 if self.res_by_id[r].isru_producible else 0 for r in self.res_ids
            },
            within=pyo.Binary,
        )

    def _create_variables(self):
        m = self.m
        settings = self.settings

        m.x = pyo.Var(m.A, m.R, m.T, domain=pyo.NonNegativeReals)
        m.y = pyo.Var(
            m.A, m.T, domain=pyo.NonNegativeIntegers
        )  # Changed to Integer explicitly
        m.I_E = pyo.Var(m.N, m.R, m.T, domain=pyo.NonNegativeReals)
        m.B_E = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals)
        m.I_M = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals)
        m.A_E = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals)
        m.Q = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals)

        # Growth Variables
        m.delta_Growth = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # Reinvestment
        m.delta_City = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # Consumption

        # State Variables
        m.P = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.H = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.Pop = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.Power = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.Cumulative_City = pyo.Var(
            m.T, domain=pyo.NonNegativeReals
        )  # Accumulated structure mass

        # Phase II Binary (0 = Bootstrapping, 1 = Self-Replication)
        # We might want to allow switching ONCE. P_t >= P_{t-1} ensures monotonic growth mostly?
        # But this binary allows changing the growth equation.
        m.z_PhaseII = pyo.Var(m.T, domain=pyo.Binary)

    # -------------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------------

    def _create_constraints(self):
        m = self.m

        # 1. Logistics Flow (Source -> Moon)
        self._add_logistics_constraints()

        # 2. Capacity Growth (Phase I/II/III Dynamics)
        self._add_capacity_growth_constraints()

        # 3. Material Balance (Inventory, Production, Consumption)
        self._add_material_balance_constraints()

        # 4. Goal / Objective Limits
        self._add_goal_constraints()

    def _add_logistics_constraints(self):
        m = self.m

        # Arc Capacity Constraints
        def _arc_capacity_rule(mdl, a, t):
            # Dynamic Payload Support (e.g. Rocket growing payload)
            payload = mdl.arc_payload[a]
            if mdl.arc_type[a] in ("rocket", "transfer"):
                payload = mdl.rocket_payload[t]

            # Total mass on arc <= Payload * Units
            return sum(mdl.x[a, r, t] for r in mdl.R) <= payload * mdl.y[a, t]

        m.arc_capacity = pyo.Constraint(m.A, m.T, rule=_arc_capacity_rule)

        # Arc Horizon (cannot ship if arrival is after T_horizon)
        m.arc_horizon = pyo.Constraint(
            m.A,
            m.T,
            rule=lambda mdl, a, t: mdl.y[a, t] == 0
            if t + mdl.arc_lead[a] >= self.settings.T_horizon
            else pyo.Constraint.Skip,
        )

        # Rocket Launch Limits (Global Pool)
        def _rocket_launch_rate_rule(mdl, t):
            if not self.launch_arcs:
                return pyo.Constraint.Skip
            return sum(mdl.y[a, t] for a in mdl.A_Launch) <= mdl.rocket_launch_rate[t]

        m.rocket_launch_limit = pyo.Constraint(m.T, rule=_rocket_launch_rate_rule)

        # Elevator Capacity Limits (Mass/Time or Stream)
        if self.elevator_arcs:
            cap_per_step = self.C_E_0 * self.delta_t

            # Pool Constraint (Total Mass on all elevator arcs)
            def _elevator_pool_rule(mdl, t):
                return (
                    sum(mdl.y[a, t] * mdl.arc_payload[a] for a in mdl.A_Elev)
                    <= cap_per_step
                )

            m.elevator_pool = pyo.Constraint(m.T, rule=_elevator_pool_rule)

            # Steam Model (Flow Constraint)
            if self.stream_model:

                def _elevator_stream_rule(mdl, t):
                    return (
                        sum(mdl.x[a, r, t] for a in mdl.A_Elev for r in mdl.R)
                        <= cap_per_step
                    )

                m.elevator_stream = pyo.Constraint(m.T, rule=_elevator_stream_rule)

    def _add_capacity_growth_constraints(self):
        m = self.m

        # Parameters for Growth
        beta = float(
            self.bootstrapping["beta_equipment_to_capacity"]
        )  # Phase I Multiplier
        # Alpha is Annual Growth Rate (1/yr). We need to verify if constant has it.
        # If not, default to 0.35.
        alpha_val = float(self.replication.get("alpha_replication_rate", 0.35))

        limit_K = float(self.saturation["carrying_capacity_tpy"])
        decay = float(self.saturation["decay_rate_phi"])

        def _p_evolution_rule(mdl, t):
            if t == 0:
                return mdl.P[t] == self.initial_state["P_0"]

            prev_P = mdl.P[t - 1]

            # 1. Earth Imports (Bootstrapping)
            earth_inflow = sum(
                mdl.x[a, r, t - mdl.arc_lead[a]]
                for a in mdl.A_to_Moon
                for r in self.tier1_res
                if t - mdl.arc_lead[a] >= 0
            )

            # 2. Local Growth (Replication)
            # Switch to Alpha-based logic.
            # Local Capacity Addition = (Alpha/12) * P_prev matches exponential growth.
            # But we want to link this to Material Investment (delta_Growth).
            # If we assume efficiency allows: Cap_Add <= beta_local * delta_Growth
            # And we set beta_local such that at full reinvestment, we hit Alpha growth.
            # beta_local = Alpha (annual) ?
            # If P=100. Growth=35/yr. Monthly=3.
            # If delta_Growth = P (unrealistic, P is product, delta_G is mass).
            # If P means "Tons/Year Production". Max reinvestment = P/12 (tons/month).
            # Cap_Add = beta_local * (P/12).
            # We want Cap_Add = (Alpha/12) * P.
            # Implying: beta_local * (P/12) = (Alpha/12) * P  => beta_local = Alpha.
            # So multiplier for delta_Growth should be equal to 'Alpha'.
            # Beta(50) was way too high.

            local_growth = 0
            if t > 0:
                # Use alpha as the mass-to-capacity multiplier
                local_growth = alpha_val * mdl.delta_Growth[t - 1]

            # 3. Decay
            decay_amount = (decay / self.steps_per_year) * prev_P

            # Constraint
            potential_P = prev_P + (beta * earth_inflow) + local_growth - decay_amount

            return mdl.P[t] <= potential_P

        m.p_evolution = pyo.Constraint(m.T, rule=_p_evolution_rule)

        m.p_limit = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.P[t] <= limit_K)

        # H (Handling) and Pop (Population) Evolution
        # Assume they scale with P for now, or have their own logic?
        # Model description says P is the main driver. H and Pop are auxiliary.
        # Let's assume H = ratio * P and Pop = ratio * P for simplicity in this refactor,
        # Or add independent growth flows.
        # Plan says: "H_ratio: 1.5".
        h_ratio = float(self.initial_state.get("H_ratio", 1.5))
        m.h_link = pyo.Constraint(
            m.T, rule=lambda mdl, t: mdl.H[t] == h_ratio * mdl.P[t]
        )

        # Population: 1 person per X tons of P? Or independent?
        # Let's leave Pop free but constrained by Life Support (which we might not have modeled yet).
        # For now, let's make Pop grow with City Mass?
        # Or just unconstrained (decision variable) but capped by Habitat Mass?
        # We have Cumulative_City. Let's say Pop <= Cumulative_City / MassPerPerson.
        # MassPerPerson approx 10 tons (structure).
        # Population: 1 person per X tons of City Structure?
        # Let's link Pop to Cumulative City mass (Habitat).
        # Assumption: 10 tons structure per person.
        m.pop_constraint = pyo.Constraint(
            m.T, rule=lambda mdl, t: mdl.Pop[t] <= mdl.Cumulative_City[t] / 10.0
        )

        # NEW: Water Demand Constraint
        # Pop[t] * WaterPerCapita <= Stock_Water[t] + Production_Water[t]
        # (Assuming we consume flow or stock. Using stock for safety).
        if self.water_res:

            def _water_pop_rule(mdl, t):
                # Total available water at step t
                available = (
                    mdl.I_M[self.water_res, t]
                    + mdl.I_E[self.moon_node, self.water_res, t]
                )
                demand = mdl.Pop[t] * self.water_per_capita
                return demand <= available

            m.water_demand = pyo.Constraint(m.T, rule=_water_pop_rule)

    def _add_material_balance_constraints(self):
        m = self.m

        # 1. Earth Inventory (Standard Node Balance)
        def _inv_earth_rule(mdl, n, r, t):
            if n == self.moon_node or self.node_types.get(n) in ("source", "earth"):
                return pyo.Constraint.Skip

            prev = mdl.I_E[n, r, t - 1] if t > 0 else 0
            inflow = sum(
                mdl.x[a, r, t - mdl.arc_lead[a]]
                for a in self.arcs_to[n]
                if t - mdl.arc_lead[a] >= 0
            )
            outflow = sum(mdl.x[a, r, t] for a in self.arcs_from[n])
            return mdl.I_E[n, r, t] == prev + inflow - outflow

        m.inv_balance_earth = pyo.Constraint(m.N, m.R, m.T, rule=_inv_earth_rule)

        # 2. Moon Material Balance (I_M tracks refined/available materials on Moon)
        # We assume Arrivals go to I_E[Moon], then move to I_M (or used directly).
        # Let's simplify: All On-Moon Inventory is I_M.
        # Arrivals (A_E) add to I_M.
        # Production (Q) add to I_M.
        # Consumption (Growth + City) subtracts from I_M.

        # I_M[r, t] = I_M[r, t-1] + Arrivals[r, t] + Production[r, t] - Consumption[r, t]

        # Define Consumption per Resource based on BOM
        # Growth: Consumes 'structure' (ISRU) mostly.
        # City: Consumes weighted mix.

        def _consumption_rule(mdl, r, t):
            # 1. Growth consumption
            growth_share = self.growth_bom.get(r, 0.0)

            # 2. City consumption (BOM)
            # Simplified:
            # - 'structure': 0.7
            # - 'electronics': 0.1
            # - 'life_support': 0.1
            # - 'water': 0.1
            city_share = 0.0
            if "structure" in r:
                city_share = 0.7
            elif "electronics" in r:
                city_share = 0.1
            elif "life_support" in r:
                city_share = 0.1
            elif "water" in r:
                city_share = 0.1

            consumed = (growth_share * mdl.delta_Growth[t]) + (
                city_share * mdl.delta_City[t]
            )

            # 3. Population Consumption (Water)
            # If r is water, add Pop * rate
            if self.water_res == r:
                consumed += mdl.Pop[t] * self.water_per_capita

            return consumed

        def _moon_balance_rule(mdl, r, t):
            prev = mdl.I_M[r, t - 1] if t > 0 else 0

            # Arrivals from Earth (via I_E or direct A_E)
            # In previous code, A_E was arrivals.
            arrivals = mdl.A_E[r, t]

            # Production
            produced = mdl.Q[r, t]

            # Consumption
            consumed = _consumption_rule(mdl, r, t)

            return mdl.I_M[r, t] == prev + arrivals + produced - consumed

        m.inv_balance_moon = pyo.Constraint(m.R, m.T, rule=_moon_balance_rule)

        # Link A_E (Arrivals) to Inflows
        def _arrival_link_rule(mdl, r, t):
            inflow = sum(
                mdl.x[a, r, t - mdl.arc_lead[a]]
                for a in mdl.A_to_Moon
                if t - mdl.arc_lead[a] >= 0
            )
            return mdl.A_E[r, t] == inflow

        m.arrival_link = pyo.Constraint(m.R, m.T, rule=_arrival_link_rule)

        # 3. Production Capacity Limit
        # Total Q <= P / steps
        m.production_limit = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: sum(mdl.Q[r, t] for r in mdl.R)
            <= mdl.P[t] / self.steps_per_year,
        )

        # 3b. ISRU Feasibility: Non-ISRU resources cannot be produced locally
        def _isru_prod_rule(mdl, r, t):
            if pyo.value(mdl.isru_ok[r]) < 0.5:
                return mdl.Q[r, t] == 0
            return pyo.Constraint.Skip

        m.isru_production = pyo.Constraint(m.R, m.T, rule=_isru_prod_rule)

        # Note: I_E at Moon is now redundant if we assume everything dumps into I_M.
        # But we need to handle "Transiting" Inventory?
        # I_E[Moon] is zeroed out or forced to 0?
        # Let's just not constrain I_E[Moon] and let it be 0.

    def _add_goal_constraints(self):
        m = self.m

        # Cumulative City Mass
        def _cum_city_rule(mdl, t):
            if t == 0:
                return mdl.Cumulative_City[t] == mdl.delta_City[t]
            return (
                mdl.Cumulative_City[t] == mdl.Cumulative_City[t - 1] + mdl.delta_City[t]
            )

        m.cum_city_state = pyo.Constraint(m.T, rule=_cum_city_rule)

        # Goal: Hard deadline on cumulative city mass
        target_mass_kg = self.total_demand_kg
        if self.deadline_year is not None:
            start_year = float(self.constants["time"]["start_year"])
            steps_per_year = float(self.constants["time"]["steps_per_year"])
            deadline_step = int(
                math.ceil((float(self.deadline_year) - start_year) * steps_per_year)
            )
            if deadline_step >= self.settings.T_horizon:
                raise ValueError(
                    "deadline_year exceeds or equals current horizon; "
                    "increase T_horizon or move the deadline earlier."
                )
            m.city_deadline = pyo.Constraint(
                expr=m.Cumulative_City[deadline_step] >= target_mass_kg
            )
        else:
            # Default: enforce at final horizon
            last_t = self.settings.T_horizon - 1
            m.city_deadline = pyo.Constraint(
                expr=m.Cumulative_City[last_t] >= target_mass_kg
            )

    def _create_objective(self):
        m = self.m
        w_C = float(self.constants["objective"]["w_C"])
        w_T = float(
            self.constants["objective"]["w_T"]
        )  # Use w_T as weight for "Early Growth"

        m.cost_total = pyo.Expression(
            expr=sum(
                m.arc_cost[a, t] * m.x[a, r, t] for a in m.A for r in m.R for t in m.T
            )
        )

        # New Objective: Minimize Cost - Reward for City Mass accumulation (Integral)
        # Maximizing Sum(Cumulative_City[t]) pushes for early growth.
        # Scale: Cost is in USD (~1e9 to 1e12). City Mass is 1e5 to 1e8 tons.
        # We need to balance weights.
        # Assuming w_T in constants is appropriately scaled or we might need to adjust.
        # For now, simplistic implementation.

        # Capped Reward Logic
        # We want to reward growth only up to the target.
        # Define auxiliary variable: Rewardable_City[t]
        # Rewardable_City[t] <= Cumulative_City[t]
        # Rewardable_City[t] <= Target_Mass

        m.Rewardable_City = pyo.Var(m.T, domain=pyo.NonNegativeReals)

        target_mass_kg = self.total_demand_kg

        def _reward_cap_rule1(mdl, t):
            return mdl.Rewardable_City[t] <= mdl.Cumulative_City[t]

        m.reward_cap1 = pyo.Constraint(m.T, rule=_reward_cap_rule1)

        def _reward_cap_rule2(mdl, t):
            return mdl.Rewardable_City[t] <= target_mass_kg

        m.reward_cap2 = pyo.Constraint(m.T, rule=_reward_cap_rule2)

        m.city_integral = pyo.Expression(expr=sum(m.Rewardable_City[t] for t in m.T))

        # ---------------------------------------------------------------------
        # Environmental Shadow Pricing (Model IV) - Added when --env flag is set
        # ---------------------------------------------------------------------
        env_penalty_expr = 0
        if self.settings.enable_env:
            env_cfg = self.constants.get("environment", {})

            # Rocket emission cost per ton payload ($ / ton)
            rocket_emissions = env_cfg.get("rocket_emissions", {})
            strat_weights = env_cfg.get("stratospheric_weights", {})
            rocket_cost_per_ton = (
                strat_weights.get("w_CO2", 0.05) * rocket_emissions.get("e_CO2", 3100)
                + strat_weights.get("w_BC", 500) * rocket_emissions.get("e_BC", 0.6)
                + strat_weights.get("w_Al2O3", 100) * rocket_emissions.get("e_Al2O3", 15)
                + strat_weights.get("w_NOx", 50) * rocket_emissions.get("e_NOx", 2.5)
            )

            # Elevator emission cost per ton (much lower)
            elev_cfg = env_cfg.get("elevator_emissions", {})
            ci_grid = elev_cfg.get("CI_grid_kg_CO2_per_kWh", 0.05)
            energy_per_ton = elev_cfg.get("energy_per_ton_kWh", 500)
            elevator_cost_per_ton = strat_weights.get("w_CO2", 0.05) * ci_grid * energy_per_ton

            # Lunar disturbance cost per ton ISRU
            moon_cfg = env_cfg.get("moon_disturbance", {})
            epsilon_reg = moon_cfg.get("epsilon_reg", 0.5)
            epsilon_sens = moon_cfg.get("epsilon_sens", 10.0)
            sens_fraction = moon_cfg.get("sens_fraction", 0.05)
            moon_cost_per_ton = epsilon_reg + sens_fraction * epsilon_sens

            # Get shadow price weight (lambda_env)
            # Default to 1.0 so that the values above are used directly
            w_env = float(self.constants.get("objective", {}).get("w_env", 1.0))

            # D_E: Earth Atmospheric Debt (rockets + elevators)
            # Sum over rocket arcs: rocket_cost_per_ton * sum_r(x[a,r,t])
            d_E_expr = sum(
                rocket_cost_per_ton * m.x[a, r, t]
                for a in self.rocket_arcs
                for r in m.R
                for t in m.T
            ) + sum(
                elevator_cost_per_ton * m.x[a, r, t]
                for a in self.elevator_arcs
                for r in m.R
                for t in m.T
            )

            # D_M: Moon Surface Debt (ISRU production)
            # Sum over all ISRU-producible resources: moon_cost_per_ton * Q[r,t]
            d_M_expr = sum(
                moon_cost_per_ton * m.Q[r, t]
                for r in m.R
                for t in m.T
                if self.res_by_id[r].isru_producible
            )

            # Store expressions for reporting
            m.D_E = pyo.Expression(expr=d_E_expr)
            m.D_M = pyo.Expression(expr=d_M_expr)

            env_penalty_expr = w_env * (d_E_expr + d_M_expr)

        # We minimize: w_C * Cost - w_T * Integral + env_penalty
        m.obj_total = pyo.Objective(
            expr=w_C * m.cost_total - w_T * m.city_integral + env_penalty_expr,
            sense=pyo.minimize,
        )

