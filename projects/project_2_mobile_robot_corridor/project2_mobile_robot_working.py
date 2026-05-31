# %% Cell 1 - Imports
import sys
from pathlib import Path

import casadi as ca
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (PROJECT_DIR / "mobile_robot_helpers.py").exists():
    PROJECT_DIR = Path("projects/project_2_mobile_robot_corridor").resolve()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from mobile_robot_helpers import (
    CorridorParams,
    discretize_continuous_system,
    dlqr,
    eta_bounds_horizon,
    global_state_from_error,
    nominal_tracking_controller,
    phi_sequence,
    plot_corridor_preview,
    plot_eta_history,
    plot_input_history,
    plot_top_view,
    print_summary_table,
    tracking_error,
    unicycle_step,
)

# %% Cell 2 - Project parameters
Ts = 0.1
steps = 100
N = 40

radius = 1.8
v_r = 0.45
omega_r = v_r / radius
phi0 = 0.70

K_xi = 1.0
K_eta = 2.0
K_psi = 2.0

nx = 3
nu = 2

x0 = np.array([0.0, -0.15, 0.0])
z0 = global_state_from_error(x0, phi0, radius)

# %% Cell 3 - Constant local prediction model for MPC and LQR correction
A_c = np.array(
    [
        [-K_xi, omega_r, 0.0],
        [-omega_r, 0.0, v_r],
        [0.0, -v_r * K_eta, -v_r * K_psi],
    ]
)

B_c = np.array(
    [
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ]
)

A, B = discretize_continuous_system(A_c, B_c, Ts)

Q = np.diag([0.5, 8.0, 1.0])
R = np.diag([0.15, 0.08])
K_lqr, P_inf = dlqr(A, B, Q, R)

u_min = np.array([-0.30, -1.20])
u_max = np.array([0.30, 1.20])

# %% Cell 4 - Known path-dependent lateral constraints
corridor = CorridorParams(
    phi_center=np.pi / 2.0,
    phi_half_width=0.24,
    eta_min_normal=-0.55,
    eta_max_normal=0.65,
    eta_min_narrow=0.10,
    eta_max_narrow=0.65,
)

phi_values = phi_sequence(phi0, omega_r, Ts, steps)
plot_corridor_preview(phi_values, corridor)

# %% Cell 5 - Nominal nonlinear closed-loop motion, without MPC correction
z_nominal = np.zeros((steps + 1, nx))
x_nominal = np.zeros((steps + 1, nx))
u_nominal = np.zeros((steps, nu))
z_nominal[0] = z0
x_nominal[0] = tracking_error(z_nominal[0], phi_values[0], radius)

for k in range(steps):
    x_k = tracking_error(z_nominal[k], phi_values[k], radius)
    u_c = nominal_tracking_controller(x_k, v_r, omega_r, K_xi, K_eta, K_psi)
    u_e = np.zeros(nu)

    z_nominal[k + 1] = unicycle_step(z_nominal[k], u_c + u_e, Ts)
    x_nominal[k] = x_k
    u_nominal[k] = u_e

x_nominal[-1] = tracking_error(z_nominal[-1], phi_values[-1], radius)
nominal_result = {"x": x_nominal, "z": z_nominal, "u": u_nominal}

plot_top_view({"nominal": nominal_result}, phi_values, corridor, radius)
plot_eta_history({"nominal": nominal_result}, phi_values, corridor)
print_summary_table({"nominal": nominal_result}, phi_values, corridor)

# %% Cell 6 - LQR correction applied to the nonlinear loop
z_lqr = np.zeros((steps + 1, nx))
x_lqr = np.zeros((steps + 1, nx))
u_lqr = np.zeros((steps, nu))
z_lqr[0] = z0
x_lqr[0] = tracking_error(z_lqr[0], phi_values[0], radius)

for k in range(steps):
    x_k = tracking_error(z_lqr[k], phi_values[k], radius)
    u_c = nominal_tracking_controller(x_k, v_r, omega_r, K_xi, K_eta, K_psi)
    u_e = -K_lqr @ x_k

    z_lqr[k + 1] = unicycle_step(z_lqr[k], u_c + u_e, Ts)
    x_lqr[k] = x_k
    u_lqr[k] = u_e

x_lqr[-1] = tracking_error(z_lqr[-1], phi_values[-1], radius)

baseline_results = {
    "nominal": nominal_result,
    "LQR": {"x": x_lqr, "z": z_lqr, "u": u_lqr},
}

plot_top_view(baseline_results, phi_values, corridor, radius)
plot_eta_history(baseline_results, phi_values, corridor)
plot_input_history({"LQR": baseline_results["LQR"]}, Ts)
print_summary_table(baseline_results, phi_values, corridor)

# %% Cell 7 - CasADi MPC correction solver

def solve_mpc_correction(x_meas, phi_k):
    opti = ca.Opti()

    U = opti.variable(nu, N)

    A_ca = ca.DM(A)
    B_ca = ca.DM(B)
    Q_ca = ca.DM(Q)
    R_ca = ca.DM(R)
    P_ca = ca.DM(P_inf)
    x0_ca = ca.DM(x_meas)

    _, eta_min_h, eta_max_h, _ = eta_bounds_horizon(phi_k, omega_r, Ts, N, corridor)

    X = [x0_ca]
    cost = 0

    for j in range(N):
        x_j = X[j]
        u_j = U[:, j]

        cost += ca.mtimes([x_j.T, Q_ca, x_j])
        cost += ca.mtimes([u_j.T, R_ca, u_j])

        opti.subject_to(opti.bounded(float(u_min[0]), U[0, j], float(u_max[0])))
        opti.subject_to(opti.bounded(float(u_min[1]), U[1, j], float(u_max[1])))

        x_next = A_ca @ x_j + B_ca @ u_j
        X.append(x_next)

        opti.subject_to(x_next[1] >= float(eta_min_h[j + 1]))
        opti.subject_to(x_next[1] <= float(eta_max_h[j + 1]))

    x_N = X[N]
    cost += ca.mtimes([x_N.T, P_ca, x_N])

    opti.minimize(cost)
    opti.set_initial(U, 0.0)
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0})

    sol = opti.solve()

    U_opt = np.array(sol.value(U)).T
    X_opt = np.zeros((N + 1, nx))
    X_opt[0] = x_meas
    for j in range(N):
        X_opt[j + 1] = A @ X_opt[j] + B @ U_opt[j]

    return {
        "u0": U_opt[0],
        "U": U_opt,
        "X": X_opt,
        "status": opti.stats()["return_status"],
    }

# %% Cell 8 - MPC correction applied to the nonlinear loop
z_mpc = np.zeros((steps + 1, nx))
x_mpc = np.zeros((steps + 1, nx))
u_mpc = np.zeros((steps, nu))
mpc_status = []
z_mpc[0] = z0
x_mpc[0] = tracking_error(z_mpc[0], phi_values[0], radius)

for k in range(steps):
    phi_k = phi_values[k]
    x_k = tracking_error(z_mpc[k], phi_k, radius)
    result = solve_mpc_correction(x_k, phi_k)

    u_c = nominal_tracking_controller(x_k, v_r, omega_r, K_xi, K_eta, K_psi)
    u_e = result["u0"]

    z_mpc[k + 1] = unicycle_step(z_mpc[k], u_c + u_e, Ts)
    x_mpc[k] = x_k
    u_mpc[k] = u_e
    mpc_status.append(result["status"])

x_mpc[-1] = tracking_error(z_mpc[-1], phi_values[-1], radius)
mpc_result = {"x": x_mpc, "z": z_mpc, "u": u_mpc, "status": mpc_status}

plot_top_view({"MPC": mpc_result}, phi_values, corridor, radius)
plot_eta_history({"MPC": mpc_result}, phi_values, corridor)
plot_input_history({"MPC": mpc_result}, Ts)
print_summary_table({"MPC": mpc_result}, phi_values, corridor)

# %% Cell 9 - Final comparison
all_results = {
    "nominal": nominal_result,
    "LQR": baseline_results["LQR"],
    "MPC": mpc_result,
}

plot_top_view(all_results, phi_values, corridor, radius)
plot_eta_history(all_results, phi_values, corridor)
plot_input_history(
    {
        "LQR": baseline_results["LQR"],
        "MPC": mpc_result,
    },
    Ts,
)
print_summary_table(all_results, phi_values, corridor)

# %% Cell 10 - Solver status check
unique_status = sorted(set(mpc_status))
print("MPC solver status values:")
for status in unique_status:
    print("-", status)
