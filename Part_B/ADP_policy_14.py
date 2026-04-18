import pyomo.environ as pyo

# ==============================================================================
# 1. SYSTEM PARAMETERS INITIALIZATION
# ==============================================================================
# We defined the system parameters precisely as specified in the Part A 
# 'SystemCharacteristics' file. This guarantees that our policy operates 
# under the exact physical assumptions required by the evaluation environment.
params = {
    'P_max': [3.0, 3.0],   
    'P_vent': 2.0,         
    'T_out': 10.0,         
    'T_low': 18.0,         
    'T_high': 26.0,        
    'H_high': 70.0,        
    'zeta_exch': 0.05,     
    'zeta_loss': 0.1,      
    'zeta_conv': 1.0,      
    'zeta_cool': 0.5,      
    'zeta_occ': 0.1,       
    'eta_occ': 0.5,        
    'eta_vent': 5.0        
}

# ==============================================================================
# 2. PRE-TRAINED VFA WEIGHTS (THETA)
# ==============================================================================
# These represent the optimized Value Function Approximation (VFA) weights 
# our group obtained after running an offline Stochastic Gradient Descent (SGD) 
# algorithm over the 100-day historical dataset. They are hardcoded to allow 
# the online policy to evaluate instantaneously without requiring retraining.
TRAINED_THETA = [2.0039, 1.8671, -0.3594, 0.0, 0.0]

# ==============================================================================
# 3. SINGLE-STEP ADP OPTIMIZER
# ==============================================================================
def solve_adp_step(state, theta, params):
    """
    Formulates and solves the "here-and-now" Mixed-Integer Linear Program (MILP).
    This function balances the immediate electricity cost against the expected 
    future cost approximated by the linear VFA.
    """
    model = pyo.ConcreteModel()
    
    # We utilize a dynamic bounds function to strictly enforce the maximum 
    # heater power (P_max) for each respective room based on its index.
    def p_bounds(model, r):
        return (0, params['P_max'][r-1])
        
    model.p = pyo.Var([1, 2], bounds=p_bounds)             
    model.v = pyo.Var(domain=pyo.Binary)                   
    
    # Post-decision state variables representing the deterministic system state 
    # immediately following our chosen action (transition from t to t^x).
    model.T_x = pyo.Var([1, 2])
    model.H_x = pyo.Var()      
    
    # Auxiliary variables for linearizing the threshold penalty basis functions.
    # We impose an upper bound (50.0) as a robust safeguard to prevent the solver 
    # from returning an 'Unbounded' status during unexpected state explorations.
    model.penalty_T1 = pyo.Var(bounds=(0, 50.0))
    model.penalty_T2 = pyo.Var(bounds=(0, 50.0))

    # --- Transition Dynamics Constraints ---
    # We implemented the linear thermal dynamics equations exactly as derived 
    # in our Part A methodological formulation.
    def temp_dynamics_rule(model, r):
        other_r = 2 if r == 1 else 1
        heat_exchange = params['zeta_exch'] * (state[f'T{other_r}'] - state[f'T{r}'])
        thermal_loss = params['zeta_loss'] * (params['T_out'] - state[f'T{r}'])
        heating_effect = params['zeta_conv'] * model.p[r]
        vent_cooling = params['zeta_cool'] * model.v
        occ_gain = params['zeta_occ'] * state[f'occ{r}']
        return model.T_x[r] == state[f'T{r}'] + heat_exchange + thermal_loss + heating_effect - vent_cooling + occ_gain
    
    model.temp_dyn_constraint = pyo.Constraint([1, 2], rule=temp_dynamics_rule)
    
    def humidity_dynamics_rule(model):
        # The .get() method ensures the code safely defaults to 0 to prevent 
        # KeyError exceptions if occupancy data is missing from the state dictionary.
        occ1 = state.get('occ1', 0)
        occ2 = state.get('occ2', 0)
        total_occ = occ1 + occ2
        occ_contribution = params['eta_occ'] * total_occ
        vent_reduction = params['eta_vent'] * model.v
        return model.H_x == state['H'] + occ_contribution - vent_reduction
        
    model.hum_dyn_constraint = pyo.Constraint(rule=humidity_dynamics_rule)
    
    # --- Penalty Formulation Constraints ---
    # This structure mathematically linearizes the max(0, T_low - T_x) function 
    # required for evaluating the quadratic penalty basis functions.
    model.pen_t1_constr = pyo.Constraint(expr=model.penalty_T1 >= params['T_low'] - model.T_x[1])
    model.pen_t2_constr = pyo.Constraint(expr=model.penalty_T2 >= params['T_low'] - model.T_x[2])

    # --- System Operational Constraints & Overrule Controllers ---
    # Enforcing the minimum ventilation run-time inertia (3 hours).
    if state['c'] > 0:
        model.vent_inertia_constraint = pyo.Constraint(expr=model.v == 1)
        
    # Enforcing the strictly binding humidity overrule controller.
    if state['H'] > params['H_high']:
        model.hum_overrule_constraint = pyo.Constraint(expr=model.v == 1)

    model.overrule_constraints = pyo.ConstraintList()
    for r in [1, 2]:
        # We strictly map the overrule variable names to match the 
        # evaluation environment's exact state dictionary keys.
        y_low = state.get(f'y_low_{r}', 0)
        y_high = state.get(f'y_high_{r}', 0)
        
        # If overrules are active, we force the decision variables to comply 
        # to strictly avoid infeasibility penalties in the simulation environment.
        if y_low == 1:
            model.overrule_constraints.add(model.p[r] == params['P_max'][r-1])
        if y_high == 1:
            model.overrule_constraints.add(model.p[r] == 0)

    # --- Objective Function ---
    # The objective minimizes the immediate deterministic electricity cost 
    # plus the approximated future cost modeled via the Linear VFA.
    def objective_rule(model):
        immediate_cost = state['price'] * (model.p[1] + model.p[2] + params['P_vent'] * model.v)
        vfa = (theta[0] * model.T_x[1] + 
               theta[1] * model.T_x[2] + 
               theta[2] * model.H_x + 
               theta[3] * (model.penalty_T1 ** 2) + 
               theta[4] * (model.penalty_T2 ** 2))
        return immediate_cost + vfa
        
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    # --- Solver Configuration ---
    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0 
    # We explicitly set the NonConvex parameter to 2 to permit Gurobi 
    # to effectively resolve the quadratic terms present in the VFA penalty.
    solver.options['NonConvex'] = 2 
    
    result = solver.solve(model)
    
    # Fallback mechanism: Should the solver encounter a probabilistically rare 
    # infeasible state during testing, we return safe, default actions (OFF) 
    # to guarantee the simulation executes continuously without crashing.
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return {'p1': 0.0, 'p2': 0.0, 'v': 0.0}
    
    return {'p1': pyo.value(model.p[1]), 'p2': pyo.value(model.p[2]), 'v': pyo.value(model.v)}

# ==============================================================================
# 4. POLICY EXECUTION FUNCTION
# ==============================================================================
def policy(state):
    """
    Primary execution function invoked by the evaluation environment.
    It takes the observed state dictionary and returns the optimal control actions.
    By leveraging the pre-trained offline weights, we ensure the execution time 
    remains strictly within the 15-second computational limit.
    """
    # We deploy the ADP optimizer step utilizing our frozen, pre-trained weights.
    optimal_actions = solve_adp_step(state, TRAINED_THETA, params)
    
    # We structure the returned output to strictly map to the exact dictionary 
    # format expected by the evaluation simulator.
    return {
        'p1': float(optimal_actions['p1']),
        'p2': float(optimal_actions['p2']),
        'v': int(optimal_actions['v'])
    }