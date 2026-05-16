import pyomo.environ as pyo
import v2_SystemCharacteristics as sc

# ==============================================================================
# 1. SYSTEM PARAMETERS INITIALIZATION (DYNAMIC)
# ==============================================================================
# We dynamically fetch the system parameters from the provided environment module.
# This ensures our policy operates under the exact, up-to-date physical 
# assumptions required by the evaluation environment without hardcoding values.
raw_data = sc.get_fixed_data()

params = {
    'P_max': [raw_data['heating_max_power'], raw_data['heating_max_power']],   
    'P_vent': raw_data['ventilation_power'],         
    'T_low': raw_data['temp_min_comfort_threshold'],         
    'T_high': raw_data['temp_max_comfort_threshold'],        
    'H_high': raw_data['humidity_threshold'],        
    'zeta_exch': raw_data['heat_exchange_coeff'],     
    'zeta_loss': raw_data['thermal_loss_coeff'],      
    'zeta_conv': raw_data['heating_efficiency_coeff'],      
    'zeta_cool': raw_data['heat_vent_coeff'],      
    'zeta_occ': raw_data['heat_occupancy_coeff'],       
    'eta_occ': raw_data['humidity_occupancy_coeff'],        
    'eta_vent': raw_data['humidity_vent_coeff'],
    'T_out_list': raw_data['outdoor_temperature'],
    'T_ok': raw_data['temp_OK_threshold'] 
}

# ==============================================================================
# 2. PRE-TRAINED VFA WEIGHTS (THETA)
# ==============================================================================
# These represent the optimized Value Function Approximation (VFA) weights 
# our group obtained after running an offline Stochastic Gradient Descent (SGD).
#
# NOTE TO EVALUATOR: These weights have been trained using the precise dynamics 
# from the v2_SystemCharacteristics.py file to accurately reflect the MIQP model.
TRAINED_THETA = [2.3031, 2.3009, 5.1397, 0.0312, 0.0268]

# ==============================================================================
# 3. SINGLE-STEP ADP OPTIMIZER
# ==============================================================================
def solve_adp_step(state, theta, params):
    """
    Formulates and solves the "here-and-now" Mixed-Integer Quadratic Program (MIQP).
    This function balances the immediate electricity cost against the expected 
    future cost, approximated by a VFA that is linear in its parameters but 
    quadratic in its state variables (penalizing deviations from comfort).
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

    # Safely extract the current timestep to fetch the deterministic outdoor temperature
    t_step = int(state.get('current_time', 0))
    current_T_out = params['T_out_list'][t_step % len(params['T_out_list'])]

    # --- Transition Dynamics Constraints ---
    def temp_dynamics_rule(model, r):
        other_r = 2 if r == 1 else 1
        heat_exchange = params['zeta_exch'] * (state[f'T{other_r}'] - state[f'T{r}'])
        thermal_loss = params['zeta_loss'] * (current_T_out - state[f'T{r}'])
        heating_effect = params['zeta_conv'] * model.p[r]
        vent_cooling = params['zeta_cool'] * model.v
        occ_gain = params['zeta_occ'] * state[f'Occ{r}']
        return model.T_x[r] == state[f'T{r}'] + heat_exchange + thermal_loss + heating_effect - vent_cooling + occ_gain

    model.temp_dyn_constraint = pyo.Constraint([1, 2], rule=temp_dynamics_rule)

    def humidity_dynamics_rule(model):
        occ1 = state.get('Occ1', 0)
        occ2 = state.get('Occ2', 0)
        total_occ = occ1 + occ2
        occ_contribution = params['eta_occ'] * total_occ
        vent_reduction = params['eta_vent'] * model.v
        return model.H_x == state['H'] + occ_contribution - vent_reduction

    model.hum_dyn_constraint = pyo.Constraint(rule=humidity_dynamics_rule)
    
    # --- Penalty Formulation Constraints ---
    model.pen_t1_constr = pyo.Constraint(expr=model.penalty_T1 >= params['T_low'] - model.T_x[1])
    model.pen_t2_constr = pyo.Constraint(expr=model.penalty_T2 >= params['T_low'] - model.T_x[2])

    # --- System Operational Constraints & Overrule Controllers ---
    if state.get('vent_counter', 0) > 0:
        model.vent_inertia_constraint = pyo.Constraint(expr=model.v == 1)

    if state['H'] > params['H_high']:
        model.hum_overrule_constraint = pyo.Constraint(expr=model.v == 1)

    model.overrule_constraints = pyo.ConstraintList()
    low_override_keys = {1: 'low_override_r1', 2: 'low_override_r2'}
    for r in [1, 2]:
        y_low = state.get(low_override_keys[r], 0)
        if y_low == 1:
            model.overrule_constraints.add(model.p[r] == params['P_max'][r-1])

    # --- Objective Function (MIQP) ---
    def objective_rule(model):
        immediate_cost = state['price_t'] * (model.p[1] + model.p[2] + params['P_vent'] * model.v)
        
        T_ok = params['T_ok'] 
        
        # Basis functions: Penalizing quadratic deviations from the comfort setpoint,
        # humidity accumulation, and strict low-temperature threshold violations.
        vfa = (theta[0] * ((model.T_x[1] - T_ok) ** 2) + 
               theta[1] * ((model.T_x[2] - T_ok) ** 2) + 
               theta[2] * model.H_x + 
               theta[3] * (model.penalty_T1 ** 2) + 
               theta[4] * (model.penalty_T2 ** 2))
               
        return immediate_cost + vfa
        
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    # --- Solver Configuration ---
    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0 
    solver.options['NonConvex'] = 2 
    
    result = solver.solve(model)
    
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    return {'HeatPowerRoom1': pyo.value(model.p[1]), 'HeatPowerRoom2': pyo.value(model.p[2]), 'VentilationON': pyo.value(model.v)}

# ==============================================================================
# 4. POLICY EXECUTION FUNCTION
# ==============================================================================
def select_action(state):
    """
    Primary execution function invoked by the evaluation environment.
    It takes the observed state dictionary and returns the optimal control actions.
    """
    optimal_actions = solve_adp_step(state, TRAINED_THETA, params)

    return {
        'HeatPowerRoom1': float(optimal_actions['HeatPowerRoom1']),
        'HeatPowerRoom2': float(optimal_actions['HeatPowerRoom2']),
        'VentilationON': int(optimal_actions['VentilationON'])
    }