# Decision-Making Under Uncertainty: HVAC System Optimization

A comprehensive project for DTU's 02435 Decision Making Under Uncertainty course (Spring 2026), focusing on optimal control of heating, ventilation, and air conditioning (HVAC) systems under uncertain occupancy and electricity prices.

## Project Overview

This project evolves from an offline "optimal-in-hindsight" approach (Part A) to developing robust, real-time decision-making policies (Part B). The system balances minimizing electricity costs with maintaining comfort constraints (temperature and humidity) in a two-room restaurant. Furthermore, the project scales up to a distributed multi-agent system coordinating energy consumption across 15 stores in a mall.

### Key Features

- **MILP-based Optimization**: Uses Pyomo with Gurobi solver for base modeling and deterministic lookaheads.
- **Advanced Decision Policies**: Implements Stochastic Programming (SP), Approximate Dynamic Programming (ADP) with Linear Value Function Approximation, and Hybrid policies.
- **Custom Simulation Environment**: Evaluates policies against unknown future data drawn from stochastic processes for prices and occupancies.
- **Distributed Decision-Making**: Coordinates 15 separate HVAC systems subject to a global peak power constraint using Lagrangian relaxation algorithms.
- **Smart Control Logic**: Handles non-linear overrule controllers (forced heating/ventilation) and minimum runtime constraints.

---

## Project Structure

```text
Decision-Making-under-Uncertainty/
├── README.md                                      # This file
├── Decission Making, Assignment Part A, 2026/
│   ├── Assignment_Decision_Making_Part_A.ipynb    # Main Jupyter notebook with all tasks
│   ├── Assignment_2026 Part A.pdf                 # Original assignment specification
│   ├── SystemCharacteristics.py                   # System parameters 
│   ├── PlotsRestaurant.py                         # Visualization 
│   ├── Data_JSON/
│   │   └── FixedData.json                         # Fixed system data (data visualization)
│   ├── OccupancyRoom1.csv                         # Room 1 occupancy data
│   ├── OccupancyRoom2.csv                         # Room 2 occupancy data
│   ├── PriceData.csv                              # Electricity price
│   └── __pycache__/                               # Python cache files
└── Part_B/
    ├── Assignment_2026_Part_B.pdf                 # Part B specification
    ├── Solution_to_Assignment_A_2026.py           # Official system model used for Part B
    ├── SP_policy_[Group].py                       # Task 3: Stochastic Programming Policy
    ├── ADP_policy_[Group].py                      # Task 4: Approximate Dynamic Programming
    ├── Hybrid_policy_[Group].py                   # Task 5: Hybrid Policy
    ├── Evaluation_Environment.py                  # Task 6: Custom simulation and testing engine
    ├── Distributed_Control.ipynb                  # Task 7: Multi-agent mall coordination
    ├── PolicyRestaurant.py                        # Template for policy implementation
    ├── Checks.py                                  # Validation script for policy constraints & time limits
    ├── PriceProcessRestaurant.py                  # Stochastic model for generating test prices
    ├── OccupancyProcessRestaurant.py              # Stochastic model for generating test occupancies
    ├── v2_System_characteristics.py               # Updated fixed parameters and thresholds
    ├── Task70ccupancies.csv                       # Known occupancies for the 15-store mall
    └── DataTask7.csv                              # Parameters for the distributed problem
## System Dynamics & Physics Model

The core environment relies on the linear dynamics validated in Part A:

**Temperature Dynamics:**
`T_r,t = T_r,t-1 + heat_exchange + thermal_loss + heating_effect - ventilation_cooling + occupancy_gain`

**Humidity Dynamics:**
`H_t = H_t-1 + occupancy_contribution - ventilation_reduction`

### Initial State for Online Policies (Part B)
- Temperature (`T_r,0`): 21°C for each room
- Humidity (`H_0`): 40%
- Ventilation counter (`c_0`): 0
- Overrule status (`y_low_r,0`): 0
- Prices and Occupancy: Drawn from uniform random distributions at start.

---

## Implementation Tasks

### Part A: Offline Optimization
- **Task 1 & 2**: Formulation and implementation of the Optimal-in-Hindsight solution (MILP). Calculates the absolute lower bound of costs assuming perfect knowledge of the 10-hour horizon.

### Part B: Online Policies under Uncertainty
All policies are restricted to a **5-8 second execution time** per hour step and must map decisions to valid operational bounds.

- **Task 3: Stochastic Programming**: A multi-stage lookahead policy relying on scenario generation (reduced via scenario trees) to anticipate price and occupancy paths.
- **Task 4: Approximate Dynamic Programming (ADP)**: Solves a single-step "here-and-now" MILP utilizing a Linear Value Function Approximation (VFA) on the post-decision state to estimate the cost-to-go. Weights are trained offline via stochastic gradient descent.
- **Task 5: Hybrid Policy**: The ultimate control policy combining elements of Lookahead (SP) and Policy/Cost Function Approximations to maximize robustness.
- **Task 6: Simulation & Evaluation**: A Python testbed evaluating the Dummy, Optimal-in-Hindsight, Expected Value, SP, and ADP policies across 100 independent episodes, generating cost histograms and performance comparisons.

### Part B: Distributed Optimization
- **Task 7: The Mall Problem**: Expands the scope to $N=15$ stores. Implements a distributed optimization algorithm (ADMM/Subgradient method) to minimize the deviation from a reference temperature `T_ref` across all stores, while strictly respecting a global power limit `P_mall`. Evaluates convergence via Lagrangian multiplier evolution and adaptive step sizes.

---

## Getting Started

### Prerequisites
- Python 3.8+
- **Gurobi Optimizer** (commercial solver - required for Pyomo MILP)
- Key Python packages: `pyomo`, `numpy`, `pandas`, `matplotlib`

### Installation

1. **Install Gurobi & Dependencies**:
```bash
pip install pyomo numpy pandas matplotlib gurobipy
### Verify Gurobi License
Ensure you have an active academic license from [Gurobi's website](https://www.gurobi.com/).

---

## Running Part B Policies

To test a policy before submission, ensure it passes the rigorous checks defined in the assignment:

```bash
python Checks.py --policy ADP_policy_groupX.py

*Note: Any policy exceeding 15 seconds per step or returning NaN values will automatically default to the Dummy (OFF) policy for that hour.*

---

## Key Results & Evaluation Metrics

* **Average Daily Cost**: The primary metric for Tasks 3-5, evaluated over unseen stochastic paths.
* **Constraint Satisfaction**: Adherence to the minimum ventilation runtime (3 hours) and overrule controller boundaries.
* **Algorithm Convergence**: For Task 7, analyzed via objective value plots against iterations for different step sizes (alpha = 0.001 to 10, plus adaptive steps).

---

## Author & Contributors

Developed for DTU Decision-Making Under Uncertainty course (02435).  
**Last Updated**: Spring 2026
