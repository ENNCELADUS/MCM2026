from __future__ import annotations

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
        self.arc_ids = [a.id for a in data.arcs]
        self.res_ids = [r.id for r in data.resources]
        self.task_ids = [t.id for t in data.tasks]

        self.arcs_by_id = {a.id: a for a in data.arcs}
        self.tasks_by_id = {t.id: t for t in data.tasks}
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
        self.d_install = constants["task_defaults"]["installation_delay"]
        self.min_task_duration = constants["task_defaults"]["min_duration"]
        self.max_concurrent_tasks = constants["task_defaults"]["max_concurrent_tasks"]

        self.C_E_0 = constants["initial_capacities"]["C_E_0"]
        elevator_capacity_upper_tpy = constants["parameter_summary"]["logistics"][
            "elevator_capacity_upper_tpy"
        ]
        self.elevator_capacity_upper_kg_s = (
            elevator_capacity_upper_tpy
            * self.ton_to_kg
            / (self.steps_per_year * self.delta_t)
        )

        self.enable_learning_curve = self.settings.enable_learning_curve
        self.stream_model = constants["implementation_details"][
            "additional_parameters"
        ]["elevator_stream_model"]["enabled"]

        # Task type flags: capability tasks use handling capacity for installation
        self.tasks_with_V = [
            i for i in self.task_ids if self.tasks_by_id[i].task_type == "capability"
        ]
        self.tasks_without_V = [i for i in self.task_ids if i not in self.tasks_with_V]

        self.total_demand_kg = (
            constants["parameter_summary"]["bom"]["total_demand_tons"] * self.ton_to_kg
        )
        self.target_pop = constants["parameter_summary"]["colony_target"]["population"]

    def _create_sets(self):
        m = self.m
        settings = self.settings

        m.T = pyo.RangeSet(0, settings.T_horizon - 1)
        m.R = pyo.Set(initialize=self.res_ids, ordered=True)
        m.N = pyo.Set(initialize=self.node_ids, ordered=True)
        m.A = pyo.Set(initialize=self.arc_ids, ordered=True)
        m.I = pyo.Set(initialize=self.task_ids, ordered=True)

        m.I_V = pyo.Set(initialize=self.tasks_with_V)
        m.I_nonV = pyo.Set(initialize=self.tasks_without_V)

        m.A_Elev = pyo.Set(initialize=self.elevator_arcs)
        m.A_Rock = pyo.Set(initialize=self.rocket_arcs)
        m.A_Launch = pyo.Set(initialize=self.launch_arcs)
        m.A_Ground = pyo.Set(initialize=self.ground_arcs)
        m.A_to_Moon = pyo.Set(initialize=self.arcs_to_moon)

        # Predecessor Pairs
        pred_pairs = []
        for task in self.data.tasks:
            for pred in task.predecessors:
                pred_pairs.append((pred, task.id))

        if pred_pairs:
            m.PRED = pyo.Set(initialize=pred_pairs, dimen=2)

    def _create_params(self):
        m = self.m
        data = self.data
        settings = self.settings

        # Task Params
        task_work = {i: self.tasks_by_id[i].W for i in self.task_ids}
        task_delta_P = {i: self.tasks_by_id[i].delta_P for i in self.task_ids}
        task_delta_V = {i: self.tasks_by_id[i].delta_V for i in self.task_ids}
        task_delta_H = {i: self.tasks_by_id[i].delta_H for i in self.task_ids}
        task_delta_Pop = {i: self.tasks_by_id[i].delta_Pop for i in self.task_ids}
        task_delta_Power = {
            i: self.tasks_by_id[i].delta_power_mw for i in self.task_ids
        }
        task_duration = {}
        task_max_rate = {}

        task_M_earth = {}
        task_M_moon = {}
        task_M_flex = {}
        self.task_M_total = {}

        for i in self.task_ids:
            t = self.tasks_by_id[i]
            duration = max(float(t.duration_months), float(self.min_task_duration))
            task_duration[i] = duration
            if duration > 0:
                task_max_rate[i] = float(t.W) / duration
            else:
                task_max_rate[i] = 0.0
            total_req = 0.0
            for r in self.res_ids:
                me = float(t.M_earth.get(r, 0.0))
                mm = float(t.M_moon.get(r, 0.0))
                mf = float(t.M_flex.get(r, 0.0))
                task_M_earth[(i, r)] = me
                task_M_moon[(i, r)] = mm
                task_M_flex[(i, r)] = mf
                total_req += me + mm + mf
            self.task_M_total[i] = total_req

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

        m.M_earth = pyo.Param(
            m.I, m.R, initialize=task_M_earth, within=pyo.NonNegativeReals
        )
        m.M_moon = pyo.Param(
            m.I, m.R, initialize=task_M_moon, within=pyo.NonNegativeReals
        )
        m.M_flex = pyo.Param(
            m.I, m.R, initialize=task_M_flex, within=pyo.NonNegativeReals
        )
        m.W = pyo.Param(m.I, initialize=task_work, within=pyo.NonNegativeReals)
        m.delta_P = pyo.Param(m.I, initialize=task_delta_P, within=pyo.NonNegativeReals)
        m.delta_V = pyo.Param(m.I, initialize=task_delta_V, within=pyo.NonNegativeReals)
        m.delta_H = pyo.Param(m.I, initialize=task_delta_H, within=pyo.NonNegativeReals)
        m.delta_Pop = pyo.Param(
            m.I, initialize=task_delta_Pop, within=pyo.NonNegativeReals
        )
        m.delta_Power = pyo.Param(
            m.I, initialize=task_delta_Power, within=pyo.NonNegativeReals
        )
        m.duration = pyo.Param(m.I, initialize=task_duration, within=pyo.NonNegativeReals)
        m.max_rate = pyo.Param(
            m.I, initialize=task_max_rate, within=pyo.NonNegativeReals
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
        m.y = pyo.Var(m.A, m.T, domain=pyo.NonNegativeIntegers)
        m.I_E = pyo.Var(m.N, m.R, m.T, domain=pyo.NonNegativeReals)
        m.B_E = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals)
        m.I_M = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals)
        m.A_E = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals)
        m.Q = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals)
        m.q_E = pyo.Var(m.I, m.R, m.T, domain=pyo.NonNegativeReals)
        m.q_M = pyo.Var(m.I, m.R, m.T, domain=pyo.NonNegativeReals)
        m.v = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
        m.v_inst = pyo.Var(m.I_V, m.T, domain=pyo.NonNegativeReals)
        m.u = pyo.Var(m.I, m.T, domain=pyo.Binary)
        m.u_done = pyo.Var(m.I, m.T, domain=pyo.Binary)
        m.S_E = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals)
        m.z = pyo.Var(m.I, m.T, domain=pyo.Binary)
        m.N_rate = pyo.Var(
            m.T,
            domain=pyo.NonNegativeIntegers,
            bounds=lambda mdl, t: (0, mdl.rocket_launch_rate[t]),
        )

        m.P = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.V = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.H = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.Pop = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.Power = pyo.Var(m.T, domain=pyo.NonNegativeReals)

        m.T_end = pyo.Var(
            domain=pyo.NonNegativeIntegers, bounds=(0, settings.T_horizon - 1)
        )
        m.z_end = pyo.Var(m.T, domain=pyo.Binary)

    # -------------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------------

    def _completed_by(self, mdl: Any, i: str, t: int) -> Any:
        # u is now cumulative: u[i,t]=1 means 'finished by t'
        return mdl.u[i, t]

    def _create_constraints(self):
        m = self.m

        # 1. Task Logic
        self._add_task_logic_constraints()
        self._add_duration_concurrency_constraints()

        # 2. Material Requirements
        self._add_material_req_constraints()

        # 3. Precedence
        if hasattr(m, "PRED"):
            m.precedence = pyo.Constraint(m.PRED, m.T, rule=self._precedence_rule)

        # 4. Capacity Evolution
        self._add_capacity_evolution_constraints()

        # 5. Logistics Flow
        self._add_logistics_constraints()

        # 6. Prepositioning / JIT
        self._add_prepositioning_constraints()

    def _add_task_logic_constraints(self):
        m = self.m

        def _task_complete_once_rule(mdl, i):
            return sum(mdl.u_done[i, t] for t in mdl.T) == 1

        m.task_complete_once = pyo.Constraint(m.I, rule=_task_complete_once_rule)

        def _task_cumulative_rule(mdl, i, t):
            return mdl.u[i, t] == sum(mdl.u_done[i, tau] for tau in mdl.T if tau <= t)

        m.task_cumulative = pyo.Constraint(m.I, m.T, rule=_task_cumulative_rule)

        def _task_initial_rule(mdl, i):
            return mdl.u[i, 0] == 0

        m.task_initial = pyo.Constraint(m.I, rule=_task_initial_rule)

        def _all_tasks_done_at_end_rule(mdl, i, t):
            # If z_end[t] == 1, then u[i, t] must be 1 (task i completed by t)
            return mdl.u[i, t] >= mdl.z_end[t]

        m.all_tasks_done_at_end = pyo.Constraint(m.I, m.T, rule=_all_tasks_done_at_end_rule)

        # Removed _makespan_rule (old T_end >= t * u pulse constraint)

        m.end_time_select = pyo.Constraint(
            rule=lambda mdl: sum(mdl.z_end[t] for t in mdl.T) == 1
        )
        m.end_time_link = pyo.Constraint(
            rule=lambda mdl: mdl.T_end
            == sum(t * mdl.z_end[t] for t in mdl.T)
        )
        m.population_target = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: mdl.Pop[t] >= self.target_pop * mdl.z_end[t],
        )

        def _work_completion_rule(mdl, i, t):
            comp = self._completed_by(mdl, i, t)
            if i in mdl.I_V:
                return (
                    sum(mdl.v_inst[i, tau] for tau in mdl.T if tau <= t)
                    >= mdl.W[i] * comp
                )
            return sum(mdl.v[i, tau] for tau in mdl.T if tau <= t) >= mdl.W[i] * comp

        m.work_completion = pyo.Constraint(m.I, m.T, rule=_work_completion_rule)

        def _material_work_rule(mdl, i, t):
            if i in mdl.I_V:
                lhs = sum(mdl.v_inst[i, tau] for tau in mdl.T if tau <= t)
                rhs = sum(
                    mdl.q_E[i, r, tau] + mdl.q_M[i, r, tau]
                    for r in mdl.R
                    for tau in mdl.T
                    if tau <= t
                )
                return lhs <= rhs
            lhs = sum(mdl.v[i, tau] for tau in mdl.T if tau <= t)
            rhs = sum(
                mdl.q_E[i, r, tau] + mdl.q_M[i, r, tau]
                for r in mdl.R
                for tau in mdl.T
                if tau <= t
            )
            return lhs <= rhs

        m.material_work_coupling = pyo.Constraint(m.I, m.T, rule=_material_work_rule)

    def _add_duration_concurrency_constraints(self):
        m = self.m
        
        # 1. Max Concurrent Tasks
        max_concurrent = self.constants["task_defaults"]["max_concurrent_tasks"]
        m.concurrency_limit = pyo.Constraint(
            m.T, 
            rule=lambda mdl, t: sum(mdl.z[i, t] for i in mdl.I) <= max_concurrent
        )
        
        # 2. Duration / Rate Limits
        # Link z[i,t] with v[i,t] and enforce max rate
        min_duration = self.constants["task_defaults"]["min_duration"]
        
        def _rate_limit_rule(mdl, i, t):
            duration = max(self.tasks_by_id[i].duration_months, min_duration)
            max_rate = mdl.W[i] / duration
            
            # v <= max_rate * z
            # Force z=1 if v > 0
            if i in mdl.I_V:
                return mdl.v_inst[i, t] <= max_rate * mdl.z[i, t]
            return mdl.v[i, t] <= max_rate * mdl.z[i, t]
            
        m.rate_limit = pyo.Constraint(m.I, m.T, rule=_rate_limit_rule)
        
        # Ensure z is 0 if task not started or already finished?
        # Actually, the optimizer minimizes concurrency cost (shadow) if bounded.
        # But we need to ensure z isn't 0 if working. (Handled by v <= R*z).
        # Do we need to force z=0 if NOT working? 
        # No, allowing z=1 when v=0 just wastes a concurrency slot, which is suboptimal but valid.
        pass

        def _task_rate_rule(mdl, i, t):
            if i in mdl.I_V:
                return mdl.v_inst[i, t] <= mdl.max_rate[i] * mdl.z[i, t]
            return mdl.v[i, t] <= mdl.max_rate[i] * mdl.z[i, t]

        m.task_rate_limit = pyo.Constraint(m.I, m.T, rule=_task_rate_rule)

        m.concurrent_limit = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: sum(mdl.z[i, t] for i in mdl.I)
            <= self.max_concurrent_tasks,
        )

    def _add_material_req_constraints(self):
        m = self.m
        m.material_req_earth = pyo.Constraint(
            m.I,
            m.R,
            rule=lambda mdl, i, r: sum(mdl.q_E[i, r, t] for t in mdl.T)
            >= mdl.M_earth[i, r],
        )
        m.material_req_moon = pyo.Constraint(
            m.I,
            m.R,
            rule=lambda mdl, i, r: sum(mdl.q_M[i, r, t] for t in mdl.T)
            >= mdl.M_moon[i, r],
        )
        m.material_req_total = pyo.Constraint(
            m.I,
            m.R,
            rule=lambda mdl, i, r: sum(
                mdl.q_E[i, r, t] + mdl.q_M[i, r, t] for t in mdl.T
            )
            == mdl.M_earth[i, r] + mdl.M_moon[i, r] + mdl.M_flex[i, r],
        )

    def _precedence_rule(self, mdl, pred, succ, t):
        # If prepositioning is ENABLED:
        # - Materials (q) are NOT constrained by predecessor.
        # - But Work (v) or Start must be constrained.
        # - Here we enforce: Cannot START succ until pred is DONE.
        # - Using z[succ, t] <= completed_by(pred, t) ? No, z is active.
        # - Using completed_by(succ, t) <= completed_by(pred, t)? No, can run in parallel? No, precedence.
        # - Standard AON: Start(succ) >= Finish(pred).
        # - We simulate this by strictly blocking z (active) if pred not done?
        # - Actually, just blocking 'v' is enough.
        
        if self.settings.enable_preposition:
            # Relax material constraint. Enforce work precedence.
            # Work done on succ <= Total Work * Pred Completed
            # v[succ] can only be non-zero if pred is done? 
            # Or cumulative work?
            # sum(v_succ) <= W_succ * u_pred
            comp_pred = self._completed_by(mdl, pred, t)
            
            # If pred is defined as "must be done before start":
            if succ in mdl.I_V:
                 work_done = sum(mdl.v_inst[succ, tau] for tau in mdl.T if tau <= t)
            else:
                 work_done = sum(mdl.v[succ, tau] for tau in mdl.T if tau <= t)
                 
            return work_done <= mdl.W[succ] * comp_pred

        # If prepositioning is DISABLED (Strict):
        # - Materials cannot arrive until pred is done.
        # - This implicitly blocks work (since v <= q).
        if self.task_M_total[succ] <= 0:
            # Fallback for tasks with no materials: constrain work directly
            comp_pred = self._completed_by(mdl, pred, t)
            if succ in mdl.I_V:
                 work_done = sum(mdl.v_inst[succ, tau] for tau in mdl.T if tau <= t)
            else:
                 work_done = sum(mdl.v[succ, tau] for tau in mdl.T if tau <= t)
            return work_done <= mdl.W[succ] * comp_pred

        consumed = sum(
            mdl.q_E[succ, r, tau] + mdl.q_M[succ, r, tau]
            for r in mdl.R
            for tau in mdl.T
            if tau <= t
        )
        return consumed <= self.task_M_total[succ] * self._completed_by(mdl, pred, t)

    def _add_capacity_evolution_constraints(self):
        m = self.m
        init_cap = self.constants["initial_capacities"]

        def _cap_update_rule(mdl, t, cap_name):
            # Calculate completed tasks considering d_install
            t_install = t - self.d_install
            completed_mask = {}
            # Optimizing: only compute if needed inside sum, but pyomo logic needs expression
            # Actually, to make expression, we iterate.

            def _completed_sum(task_set):
                if t_install < 0:
                    return 0
                return sum(
                    mdl.u[i, tau] for tau in mdl.T if tau <= t_install for i in task_set
                )

            # Wait, the original code did:
            # sum(mdl.delta_P[i] * _task_completed_by(i) for i in mdl.I)
            # where _task_completed_by(i) is sum(u[i,tau]...)

            if t_install < 0:
                inc = 0
            else:
                # u is cumulative, so u[i, t_install] == 1 means "completed by t_install"
                # This persists, so the capacity increment persists.
                if cap_name == "P":
                    inc = sum(mdl.delta_P[i] * mdl.u[i, t_install] for i in mdl.I)
                elif cap_name == "V":
                    inc = sum(mdl.delta_V[i] * mdl.u[i, t_install] for i in mdl.I_V)
                elif cap_name == "H":
                    inc = sum(mdl.delta_H[i] * mdl.u[i, t_install] for i in mdl.I)
                elif cap_name == "Pop":
                    inc = sum(mdl.delta_Pop[i] * mdl.u[i, t_install] for i in mdl.I)
                else:  # Power
                    inc = sum(mdl.delta_Power[i] * mdl.u[i, t_install] for i in mdl.I)

            base = init_cap[f"{cap_name}_0"]
            return getattr(mdl, cap_name)[t] == base + inc

        m.cap_P = pyo.Constraint(m.T, rule=lambda mdl, t: _cap_update_rule(mdl, t, "P"))
        m.cap_V = pyo.Constraint(m.T, rule=lambda mdl, t: _cap_update_rule(mdl, t, "V"))
        m.cap_H = pyo.Constraint(m.T, rule=lambda mdl, t: _cap_update_rule(mdl, t, "H"))
        m.cap_Pop = pyo.Constraint(
            m.T, rule=lambda mdl, t: _cap_update_rule(mdl, t, "Pop")
        )
        m.cap_Power = pyo.Constraint(
            m.T, rule=lambda mdl, t: _cap_update_rule(mdl, t, "Power")
        )

    def _add_logistics_constraints(self):
        m = self.m

        def _arc_capacity_rule(mdl, a, t):
            payload = mdl.arc_payload[a]
            if mdl.arc_type[a] in ("rocket", "transfer"):
                payload = mdl.rocket_payload[t]
            return sum(mdl.x[a, r, t] for r in mdl.R) <= payload * mdl.y[a, t]

        m.arc_capacity = pyo.Constraint(m.A, m.T, rule=_arc_capacity_rule)

        m.arc_horizon = pyo.Constraint(
            m.A,
            m.T,
            rule=lambda mdl, a, t: mdl.y[a, t] == 0
            if t + mdl.arc_lead[a] >= self.settings.T_horizon
            else pyo.Constraint.Skip,
        )

        # Pools
        self._add_pool_constraints()

        # Inventory
        self._add_inventory_constraints()

        # Capacity Checks
        self._add_capacity_check_constraints()

    def _add_pool_constraints(self):
        m = self.m

        def _rocket_launch_rate_rule(mdl, t):
            if not self.launch_arcs:
                return pyo.Constraint.Skip
            return sum(mdl.y[a, t] for a in mdl.A_Launch) <= mdl.N_rate[t]

        m.rocket_launch_limit = pyo.Constraint(m.T, rule=_rocket_launch_rate_rule)

        def _elevator_pool_rule(mdl, t):
            if not self.elevator_arcs:
                return pyo.Constraint.Skip
            return (
                sum(mdl.y[a, t] * mdl.arc_payload[a] for a in mdl.A_Elev)
                <= self.C_E_0 * self.delta_t
            )

        m.elevator_pool = pyo.Constraint(m.T, rule=_elevator_pool_rule)

        def _elevator_upper_rule(mdl, t):
            if not self.elevator_arcs:
                return pyo.Constraint.Skip
            # Use precomputed float capacity
            cap = self.elevator_capacity_upper_kg_s
            return (
                sum(mdl.y[a, t] * mdl.arc_payload[a] for a in mdl.A_Elev)
                <= cap * self.delta_t
            )

        m.elevator_upper = pyo.Constraint(m.T, rule=_elevator_upper_rule)

        if self.stream_model:
            def _elevator_stream_rule(mdl, t):
                if not self.elevator_arcs:
                    return pyo.Constraint.Skip
                return sum(mdl.x[a, r, t] for a in mdl.A_Elev for r in mdl.R) <= (
                    self.C_E_0 * self.delta_t
                )

            m.elevator_stream = pyo.Constraint(m.T, rule=_elevator_stream_rule)

    def _add_inventory_constraints(self):
        m = self.m

        def _inv_balance_rule(mdl, n, r, t):
            if n == self.moon_node:
                return pyo.Constraint.Skip
            prev = mdl.I_E[n, r, t - 1] if t > 0 else 0

            inflow = sum(
                mdl.x[a, r, t - mdl.arc_lead[a]]
                for a in self.arcs_to[n]
                if t - mdl.arc_lead[a] >= 0
            )
            outflow = sum(mdl.x[a, r, t] for a in self.arcs_from[n])
            supply = mdl.S_E[r, t] if n == "E_agg" else 0
            return mdl.I_E[n, r, t] == prev + inflow - outflow + supply

        m.inv_balance = pyo.Constraint(m.N, m.R, m.T, rule=_inv_balance_rule)

        def _moon_buffer_rule(mdl, r, t):
            prev = mdl.B_E[r, t - 1] if t > 0 else 0
            arrivals = sum(
                mdl.x[a, r, t - mdl.arc_lead[a]]
                for a in mdl.A_to_Moon
                if t - mdl.arc_lead[a] >= 0
            )
            return mdl.B_E[r, t] == prev + arrivals - mdl.A_E[r, t]

        m.moon_buffer = pyo.Constraint(m.R, m.T, rule=_moon_buffer_rule)

        def _moon_inventory_rule(mdl, r, t):
            prev = mdl.I_E[self.moon_node, r, t - 1] if t > 0 else 0
            consumed = sum(mdl.q_E[i, r, t] for i in mdl.I)
            return mdl.I_E[self.moon_node, r, t] == prev + mdl.A_E[r, t] - consumed

        m.moon_inventory = pyo.Constraint(m.R, m.T, rule=_moon_inventory_rule)

        def _moon_isru_rule(mdl, r, t):
            prev = mdl.I_M[r, t - 1] if t > 0 else 0
            consumed = sum(mdl.q_M[i, r, t] for i in mdl.I)
            return mdl.I_M[r, t] == prev + mdl.Q[r, t] - consumed

        m.moon_isru_inventory = pyo.Constraint(m.R, m.T, rule=_moon_isru_rule)

    def _add_capacity_check_constraints(self):
        m = self.m

        m.handling_capacity = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: (
                sum(mdl.A_E[r, t] for r in mdl.R)
                + sum(mdl.v_inst[i, t] for i in mdl.I_V)
            )
            <= mdl.H[t] * self.delta_t,
        )
        m.isru_capacity = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: sum(mdl.Q[r, t] for r in mdl.R)
            <= mdl.P[t] * self.delta_t,
        )
        m.isru_resource = pyo.Constraint(
            m.R,
            m.T,
            rule=lambda mdl, r, t: mdl.Q[r, t] == 0
            if mdl.isru_ok[r] == 0
            else pyo.Constraint.Skip,
        )
        m.construction_capacity = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: sum(mdl.v[i, t] for i in mdl.I_nonV)
            <= mdl.V[t] * self.delta_t
            if self.tasks_without_V
            else pyo.Constraint.Skip,
        )
        m.installation_total = pyo.Constraint(
            m.I_V, rule=lambda mdl, i: sum(mdl.v_inst[i, t] for t in mdl.T) == mdl.W[i]
        )
        m.construction_total = pyo.Constraint(
            m.I_nonV, rule=lambda mdl, i: sum(mdl.v[i, t] for t in mdl.T) == mdl.W[i]
        )
        m.standard_work_zero = pyo.Constraint(
            m.I_V, m.T, rule=lambda mdl, i, t: mdl.v[i, t] == 0
        )

        # Power Constraint
        # Map resource to energy (kWh/kg)
        r_energy = {
             "water": self.constants["isru"]["water_energy"],
             "fuel": self.constants["isru"]["oxygen_energy"], 
             "structure": self.constants["isru"]["metal_energy"]
        }
        
        def _power_rule(mdl, t):
             # Total Energy kWh used in month t
             total_energy_kwh = sum(
                 mdl.Q[r, t] * r_energy.get(r, 0.0) 
                 for r in mdl.R
             )
             
             # Convert to Average Power (MW) required
             # Power(kW) = Energy(kWh) / Hours
             hours = self.delta_t / 3600.0
             avg_power_kw = total_energy_kwh / hours
             avg_power_mw = avg_power_kw / 1000.0
             
             return avg_power_mw <= mdl.Power[t]

        m.power_balance = pyo.Constraint(m.T, rule=_power_rule)

        m.total_mass_sanity = pyo.Constraint(
            rule=lambda mdl: sum(
                mdl.A_E[r, t] + mdl.Q[r, t] for r in mdl.R for t in mdl.T
            )
            >= self.total_demand_kg
        )

    def _add_prepositioning_constraints(self):
        m = self.m
        if not self.settings.enable_preposition:
            # Enforce Zero Inventory Policy on Moon (Strict JIT)
            # This prevents materials from arriving early and waiting.
            
            def _no_inventory_rule(mdl, r, t):
                return mdl.I_E[self.moon_node, r, t] == 0
                
            m.no_moon_inventory = pyo.Constraint(m.R, m.T, rule=_no_inventory_rule)
            
            def _no_buffer_rule(mdl, r, t):
                return mdl.B_E[r, t] == 0
                
            m.no_moon_buffer = pyo.Constraint(m.R, m.T, rule=_no_buffer_rule)

    def _create_objective(self):
        m = self.m
        w_C = float(self.constants["objective"]["w_C"])
        w_T = float(self.constants["objective"]["w_T"])

        m.cost_total = pyo.Expression(
            expr=sum(
                m.arc_cost[a, t] * m.x[a, r, t] for a in m.A for r in m.R for t in m.T
            )
        )
        m.obj_total = pyo.Objective(
            expr=w_T * m.T_end + w_C * m.cost_total, sense=pyo.minimize
        )

        # Validate integer variables
        self._validate_integer_vars()

    def _validate_integer_vars(self):
        m = self.m
        integer_vars = self.constants["optimization"].get("integer_vars", [])
        integer_map = {"u": m.u, "u_done": m.u_done, "y": m.y}
        for var_name in integer_vars:
            if var_name in integer_map:
                var_obj = integer_map[var_name]
                if var_obj.is_indexed():
                    first_key = next(iter(var_obj))
                    domain = var_obj[first_key].domain
                else:
                    domain = var_obj.domain
                if domain not in (pyo.Binary, pyo.NonNegativeIntegers, pyo.Integers):
                    raise ValueError(
                        f"Variable '{var_name}' must be integer/binary per optimization.integer_vars"
                    )
