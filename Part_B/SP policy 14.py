"""
SP policy 14.py
Multi-stage Stochastic Programming Policy for Restaurant Energy Management.

Group 14 - DTU 02435 Decision-Making Under Uncertainty, Spring 2026.

Structure (configurable):
  - Lookahead horizon : HORIZON timesteps
  - Branching factor  : BRANCHES scenarios per stage
  - Scenario tree     : 1 + B + B^2 + ... + B^H decision nodes
  - Scenario reduction: Monte Carlo sampling (N_INIT) + k-means clustering
  - Solver            : Gurobi via Pyomo (MILP)

Usage:
    from "SP policy 14" import policy
    action = policy.select_action(state)

    # Custom horizon / branching:
    from "SP policy 14" import SPRestaurantPolicy
    policy = SPRestaurantPolicy(horizon=2, branches=4)
"""

import numpy as np
from scipy.cluster.vq import kmeans2
import pyomo.environ as pyo
from v2_SystemCharacteristics import get_fixed_data
from PriceProcessRestaurant import price_model
from OccupancyProcessRestaurant import next_occupancy_levels

# ============================================================
# System parameters  (v2_SystemCharacteristics.py)
# ============================================================
SYS_AUX = get_fixed_data()
_P = {
    'P_max'    : SYS_AUX['heating_max_power'],
    'P_vent'   : SYS_AUX['ventilation_power'],
    'T_low'    : SYS_AUX['temp_min_comfort_threshold'],
    'T_OK'     : SYS_AUX['temp_OK_threshold'],
    'T_high'   : SYS_AUX['temp_max_comfort_threshold'],
    'H_high'   : SYS_AUX['humidity_threshold'],
    'zeta_exch': SYS_AUX['heat_exchange_coeff'],
    'zeta_loss': SYS_AUX['thermal_loss_coeff'],
    'zeta_conv': SYS_AUX['heating_efficiency_coeff'],
    'zeta_cool': SYS_AUX['heat_vent_coeff'],
    'zeta_occ' : SYS_AUX['heat_occupancy_coeff'],
    'eta_occ'  : SYS_AUX['humidity_occupancy_coeff'],
    'eta_vent' : SYS_AUX['humidity_vent_coeff'],
    'PENALTY'  : 1000.0,
}

# Default tree parameters (can be overridden via SPRestaurantPolicy constructor)
HORIZON  = 3    # lookahead stages
BRANCHES = 3    # scenario branches per stage
N_INIT   = 150  # Monte Carlo draws before clustering

VENT_UPTIME = 3  # minimum ventilation on-time (hours) — fixed by system spec


# ============================================================
# Deterministic outdoor temperature (from v2_SystemCharacteristics)
# ============================================================
def _T_out(t: int) -> float:
    """Get outdoor temperature at time t from system characteristics."""
    t_capped = max(0, min(t, len(SYS_AUX['outdoor_temperature']) - 1))
    return float(SYS_AUX['outdoor_temperature'][t_capped])


# ============================================================
# Stochastic process samplers (using system process functions)
# ============================================================
def _sample_price(current: float, previous: float, n: int) -> np.ndarray:
    """Generate n price samples using price_model from PriceProcessRestaurant."""
    samples = np.array([price_model(current, previous) for _ in range(n)])
    return samples


def _sample_occ(r1: float, r2: float, n: int):
    """Generate n occupancy samples using next_occupancy_levels from OccupancyProcessRestaurant."""
    r1_samples = np.zeros(n)
    r2_samples = np.zeros(n)
    for i in range(n):
        r1_samples[i], r2_samples[i] = next_occupancy_levels(r1, r2)
    return r1_samples, r2_samples


# ============================================================
# K-means scenario reduction helper
# ============================================================
def _cluster(features: np.ndarray, k: int):
    """Return (centers [k x d], probabilities [k]) via k-means."""
    n = len(features)
    if n < k:
        reps = int(np.ceil(k / n))
        features = np.tile(features, (reps, 1))[:k]
        n = k
    std = features.std(axis=0)
    std[std == 0] = 1.0
    feat_norm = features / std
    centers_norm, labels = kmeans2(feat_norm, k, iter=10, minit='points', seed=42)
    centers = centers_norm * std
    counts = np.bincount(labels, minlength=k).astype(float)
    counts = np.maximum(counts, 1.0)
    return centers, counts / counts.sum()


# ============================================================
# Scenario tree construction  (general: any horizon and branches)
# ============================================================
def build_scenario_tree(state: dict,
                        horizon:  int = HORIZON,
                        branches: int = BRANCHES,
                        n_init:   int = N_INIT) -> list:
    """
    Build a multi-stage scenario tree via Monte Carlo + k-means.

    Returns a list of node dicts.  Node 0 is always the root (stage 0).
    Each node contains:
        id         : int   — unique index
        stage      : int   — 0 = root, 1 = first branching, …
        parent_id  : int|None
        children   : list[int]
        price      : float — representative price at this node
        price_prev : float — previous price (for AR(1) sampling of children)
        occ1, occ2 : float — representative occupancies
        prob       : float — joint probability (root = 1.0)
    """
    price0    = float(state['price'])
    price_prv = float(state.get('price_previous', price0))
    occ1_0    = float(state['occ1'])
    occ2_0    = float(state['occ2'])

    nodes = [{
        'id': 0, 'stage': 0, 'parent_id': None,
        'price': price0, 'price_prev': price_prv,
        'occ1': occ1_0, 'occ2': occ2_0,
        'prob': 1.0, 'children': [],
    }]

    m_cond   = max(n_init // branches, 30)
    leaf_ids = [0]

    for stage in range(horizon):
        n_smp = n_init if stage == 0 else m_cond
        new_leaf_ids = []
        for parent_id in leaf_ids:
            parent = nodes[parent_id]
            p_smp          = _sample_price(parent['price'], parent['price_prev'], n_smp)
            o1_smp, o2_smp = _sample_occ(parent['occ1'], parent['occ2'], n_smp)
            feat           = np.column_stack([p_smp, o1_smp, o2_smp])
            centers, probs = _cluster(feat, branches)

            for b in range(branches):
                child_id = len(nodes)
                nodes.append({
                    'id':         child_id,
                    'stage':      stage + 1,
                    'parent_id':  parent_id,
                    'price':      float(centers[b, 0]),
                    'price_prev': parent['price'],
                    'occ1':       float(centers[b, 1]),
                    'occ2':       float(centers[b, 2]),
                    'prob':       parent['prob'] * float(probs[b]),
                    'children':   [],
                })
                parent['children'].append(child_id)
                new_leaf_ids.append(child_id)
        leaf_ids = new_leaf_ids

    return nodes


# ============================================================
# Multi-stage stochastic MILP  (general: any scenario tree)
# ============================================================
def solve_multistage_sp(state: dict, nodes: list) -> dict:
    """
    Formulate and solve the multi-stage stochastic MILP over the given
    scenario tree (list of node dicts from build_scenario_tree).

    All variables are indexed by node ID.

    Dynamics at each node link to its parent's state variables
    (root node links to the observed current state).

    Returns here-and-now action dict.
    """
    p       = _P
    T1_obs  = float(state['T1'])
    T2_obs  = float(state['T2'])
    H_obs   = float(state['H'])
    occ1_0  = float(state['occ1'])
    occ2_0  = float(state['occ2'])
    c       = int(state.get('c', 0))
    y_lo1   = int(state.get('y_low_1',  0))
    y_lo2   = int(state.get('y_low_2',  0))
    y_hi1   = int(state.get('y_high_1', 0))
    y_hi2   = int(state.get('y_high_2', 0))
    t_now   = int(state.get('current_time', 0))

    node_ids     = [n['id'] for n in nodes]
    non_root_ids = [n['id'] for n in nodes if n['parent_id'] is not None]
    node_map     = {n['id']: n for n in nodes}
    BIG          = 100.0

    m = pyo.ConcreteModel()

    # ── Variables (indexed by node id) ────────────────────────────────────
    m.p1      = pyo.Var(node_ids, bounds=(0.0, p['P_max']))
    m.p2      = pyo.Var(node_ids, bounds=(0.0, p['P_max']))
    m.v       = pyo.Var(node_ids, domain=pyo.Binary)
    m.T1      = pyo.Var(node_ids)
    m.T2      = pyo.Var(node_ids)
    m.H       = pyo.Var(node_ids, bounds=(0.0, 200.0))
    m.sk_T1lo = pyo.Var(node_ids, bounds=(0.0, BIG))
    m.sk_T2lo = pyo.Var(node_ids, bounds=(0.0, BIG))
    m.sk_T1hi = pyo.Var(node_ids, bounds=(0.0, BIG))
    m.sk_T2hi = pyo.Var(node_ids, bounds=(0.0, BIG))
    m.sk_Hhi  = pyo.Var(node_ids, bounds=(0.0, BIG))

    # ── Root (stage 0): dynamics from observed current state ──────────────
    To0 = _T_out(t_now)
    m.dyn_T1_root = pyo.Constraint(expr=(
        m.T1[0] == T1_obs
        + p['zeta_exch'] * (T2_obs - T1_obs)
        + p['zeta_loss'] * (To0   - T1_obs)
        + p['zeta_conv'] * m.p1[0]
        - p['zeta_cool'] * m.v[0]
        + p['zeta_occ']  * occ1_0
    ))
    m.dyn_T2_root = pyo.Constraint(expr=(
        m.T2[0] == T2_obs
        + p['zeta_exch'] * (T1_obs - T2_obs)
        + p['zeta_loss'] * (To0   - T2_obs)
        + p['zeta_conv'] * m.p2[0]
        - p['zeta_cool'] * m.v[0]
        + p['zeta_occ']  * occ2_0
    ))
    m.dyn_H_root = pyo.Constraint(expr=(
        m.H[0] == H_obs
        + p['eta_occ']  * (occ1_0 + occ2_0)
        - p['eta_vent'] * m.v[0]
    ))

    # ── Overrule controllers at root (hard constraints) ───────────────────
    if c > 0:              m.ov_vc  = pyo.Constraint(expr=m.v[0]  == 1)
    if H_obs >= p['H_high']: m.ov_vh = pyo.Constraint(expr=m.v[0]  == 1)
    if y_lo1:              m.ov_lo1 = pyo.Constraint(expr=m.p1[0] == p['P_max'])
    if y_lo2:              m.ov_lo2 = pyo.Constraint(expr=m.p2[0] == p['P_max'])
    if y_hi1:              m.ov_hi1 = pyo.Constraint(expr=m.p1[0] == 0.0)
    if y_hi2:              m.ov_hi2 = pyo.Constraint(expr=m.p2[0] == 0.0)

    # ── Non-root dynamics: link to parent variables ───────────────────────
    def _dyn_T1(md, nid):
        nd  = node_map[nid]
        pid = nd['parent_id']
        To  = _T_out(t_now + nd['stage'])
        return (md.T1[nid] == md.T1[pid]
                + p['zeta_exch'] * (md.T2[pid] - md.T1[pid])
                + p['zeta_loss'] * (To          - md.T1[pid])
                + p['zeta_conv'] * md.p1[nid]
                - p['zeta_cool'] * md.v[nid]
                + p['zeta_occ']  * nd['occ1'])

    def _dyn_T2(md, nid):
        nd  = node_map[nid]
        pid = nd['parent_id']
        To  = _T_out(t_now + nd['stage'])
        return (md.T2[nid] == md.T2[pid]
                + p['zeta_exch'] * (md.T1[pid] - md.T2[pid])
                + p['zeta_loss'] * (To          - md.T2[pid])
                + p['zeta_conv'] * md.p2[nid]
                - p['zeta_cool'] * md.v[nid]
                + p['zeta_occ']  * nd['occ2'])

    def _dyn_H(md, nid):
        nd  = node_map[nid]
        pid = nd['parent_id']
        return (md.H[nid] == md.H[pid]
                + p['eta_occ']  * (nd['occ1'] + nd['occ2'])
                - p['eta_vent'] * md.v[nid])

    if non_root_ids:
        m.dyn_T1 = pyo.Constraint(non_root_ids, rule=_dyn_T1)
        m.dyn_T2 = pyo.Constraint(non_root_ids, rule=_dyn_T2)
        m.dyn_H  = pyo.Constraint(non_root_ids, rule=_dyn_H)

    # ── Comfort slacks (all nodes) ────────────────────────────────────────
    m.c_T1lo = pyo.Constraint(node_ids, rule=lambda md, nid: md.sk_T1lo[nid] >= p['T_low']  - md.T1[nid])
    m.c_T2lo = pyo.Constraint(node_ids, rule=lambda md, nid: md.sk_T2lo[nid] >= p['T_low']  - md.T2[nid])
    m.c_T1hi = pyo.Constraint(node_ids, rule=lambda md, nid: md.sk_T1hi[nid] >= md.T1[nid] - p['T_high'])
    m.c_T2hi = pyo.Constraint(node_ids, rule=lambda md, nid: md.sk_T2hi[nid] >= md.T2[nid] - p['T_high'])
    m.c_Hhi  = pyo.Constraint(node_ids, rule=lambda md, nid: md.sk_Hhi[nid]  >= md.H[nid]  - p['H_high'])

    # ── Ventilation inertia ───────────────────────────────────────────────
    # Pre-existing forced-on: if c > node.stage the counter hasn't expired yet.
    # New-cycle propagation:  for stages 1 .. VENT_UPTIME-1, if the pre-existing
    #   counter doesn't cover this stage (c <= stage-1), a new cycle started
    #   at the parent propagates upward (v[node] >= v[parent]).
    vin_forced = [n['id'] for n in nodes
                  if n['parent_id'] is not None and c > n['stage']]
    vin_prop   = [n['id'] for n in nodes
                  if n['parent_id'] is not None
                  and 1 <= n['stage'] <= VENT_UPTIME - 1
                  and c <= n['stage'] - 1]

    if vin_forced:
        m.vin_forced = pyo.Constraint(vin_forced,
                                      rule=lambda md, nid: md.v[nid] == 1)
    if vin_prop:
        m.vin_prop = pyo.Constraint(
            vin_prop,
            rule=lambda md, nid: md.v[nid] >= md.v[node_map[nid]['parent_id']])

    # ── Objective: E[energy cost] + penalty × E[comfort violations] ───────
    PEN = p['PENALTY']

    def _obj(md):
        return sum(
            nd['prob'] * (
                nd['price'] * (md.p1[nd['id']] + md.p2[nd['id']] + p['P_vent'] * md.v[nd['id']])
                + PEN * (md.sk_T1lo[nd['id']] + md.sk_T2lo[nd['id']]
                         + md.sk_T1hi[nd['id']] + md.sk_T2hi[nd['id']] + md.sk_Hhi[nd['id']])
            )
            for nd in nodes
        )

    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    # ── Solve ─────────────────────────────────────────────────────────────
    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    solver.options['TimeLimit']  = 10
    result = solver.solve(m)

    ok = (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible)
    if result.solver.termination_condition in ok:
        return {
            'HeatPowerRoom1': float(pyo.value(m.p1[0])),
            'HeatPowerRoom2': float(pyo.value(m.p2[0])),
            'VentilationON' : int(round(float(pyo.value(m.v[0])))),
            '_model': m,
            '_nodes': nodes,
        }

    # Fallback: greedy
    fallback_p1 = p['P_max'] if T1_obs < p['T_low'] + 1 else 0.0
    fallback_p2 = p['P_max'] if T2_obs < p['T_low'] + 1 else 0.0
    fallback_v  = 1 if (H_obs >= p['H_high'] or c > 0) else 0
    return {
        'HeatPowerRoom1': fallback_p1,
        'HeatPowerRoom2': fallback_p2,
        'VentilationON' : fallback_v,
    }


# ============================================================
# Policy class (interface expected by v2_Checks.py)
# ============================================================
class SPRestaurantPolicy:
    """
    Multi-stage stochastic programming policy.

    Parameters
    ----------
    horizon  : int  — stages to look ahead (default 3)
    branches : int  — k-means clusters per stage (default 3)
    n_init   : int  — Monte Carlo draws before clustering (default 150)
    """

    def __init__(self,
                 horizon:  int = HORIZON,
                 branches: int = BRANCHES,
                 n_init:   int = N_INIT):
        self.horizon  = int(horizon)
        self.branches = int(branches)
        self.n_init   = int(n_init)

    def select_action(self, state: dict) -> dict:
        try:
            nodes  = build_scenario_tree(state, self.horizon, self.branches, self.n_init)
            result = solve_multistage_sp(state, nodes)
            return {
                'HeatPowerRoom1': result['HeatPowerRoom1'],
                'HeatPowerRoom2': result['HeatPowerRoom2'],
                'VentilationON' : result['VentilationON'],
            }
        except Exception:
            c    = int(state.get('c', 0))
            H    = float(state.get('H', 0.0))
            y_l1 = int(state.get('y_low_1',  0))
            y_l2 = int(state.get('y_low_2',  0))
            y_h1 = int(state.get('y_high_1', 0))
            y_h2 = int(state.get('y_high_2', 0))
            return {
                'HeatPowerRoom1': _P['P_max'] if y_l1 else (0.0 if y_h1 else 0.0),
                'HeatPowerRoom2': _P['P_max'] if y_l2 else (0.0 if y_h2 else 0.0),
                'VentilationON' : 1 if (c > 0 or H >= _P['H_high']) else 0,
            }


# Module-level instance (uses default HORIZON=3, BRANCHES=3)
policy = SPRestaurantPolicy()
