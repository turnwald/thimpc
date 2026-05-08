"""Scenario code for Project 1: constrained attitude control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

from controllers.lqr import simulate_lqr
from controllers.saturated_lqr import simulate_saturated_lqr
from projects._shared.casadi_mpc import LinearCasadiMPC, MPCInfo
from projects.project_1_attitude_constraints import config
from projects.project_1_attitude_constraints.plots import (
    plot_constraint_activity,
    plot_phase_plane,
    plot_terminal_geometry,
    plot_time_histories,
)


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def count_violations(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-7) -> int:
    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return int(np.sum((values < lower - tol) | (values > upper + tol)))


def status_counts(statuses: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


def solve_attitude_mpc(
    x0: np.ndarray,
    *,
    x_min: np.ndarray = config.X_MIN,
    x_max: np.ndarray = config.X_MAX,
    rate_bound: np.ndarray | None = None,
    u_previous: np.ndarray | None = None,
    soft_theta: bool = False,
    slack_weight: float = 0.0,
) -> tuple[np.ndarray, MPCInfo]:
    _, P = lqr_gain(config.A, config.B, config.Q, config.R)
    mpc = LinearCasadiMPC(
        A=config.A,
        B=config.B,
        Q=config.Q,
        R=config.R,
        horizon=config.HORIZON,
        x_min=x_min,
        x_max=x_max,
        u_min=config.U_MIN,
        u_max=config.U_MAX,
        terminal_cost=P,
        rate_bound=rate_bound,
        soft_state_indices=[0] if soft_theta else None,
        slack_weight=slack_weight,
    )
    return mpc.solve(x0, u_previous=u_previous)


def simulate_receding_mpc(
    x0: np.ndarray,
    steps: int,
    *,
    rate_bound: np.ndarray | None = None,
    soft_theta: bool = False,
    slack_weight: float = 0.0,
) -> dict[str, object]:
    X = np.zeros((steps + 1, config.A.shape[0]))
    U = np.zeros((steps, config.B.shape[1]))
    slack = np.zeros(steps)
    statuses: list[str] = []
    failures = 0
    X[0] = np.asarray(x0, dtype=float).reshape(config.A.shape[0])
    u_previous = np.zeros(config.B.shape[1])

    for k in range(steps):
        u0, info = solve_attitude_mpc(
            X[k],
            rate_bound=rate_bound,
            u_previous=u_previous,
            soft_theta=soft_theta,
            slack_weight=slack_weight,
        )
        statuses.append(info.status)
        if info.success:
            U[k] = u0
            if info.slack.size:
                slack[k] = float(info.slack[0, 0])
        else:
            failures += 1
            U[k] = 0.0
        X[k + 1] = config.A @ X[k] + config.B @ U[k]
        u_previous = U[k]

    return {"X": X, "U": U, "slack": slack, "statuses": statuses, "failures": failures}


def run_project(steps: int = config.STEPS, output_dir: str | Path | None = None, *, show: bool = True, close: bool = False) -> dict[str, object]:
    """Run the full Project 1 comparison and return metrics."""
    K, P = lqr_gain(config.A, config.B, config.Q, config.R)

    X_lqr, U_lqr = simulate_lqr(config.A, config.B, K, config.X0, steps)
    X_sat, U_sat = simulate_saturated_lqr(config.A, config.B, K, config.X0, steps, config.U_MIN, config.U_MAX)
    mpc_run = simulate_receding_mpc(config.X0, steps)
    rate_run = simulate_receding_mpc(config.X0, steps, rate_bound=config.RATE_BOUND)

    _, hard = solve_attitude_mpc(
        np.array([1.5, 0.0]),
        x_min=np.array([-0.8, -2.5]),
        x_max=np.array([0.8, 2.5]),
    )
    _, soft = solve_attitude_mpc(
        np.array([1.5, 0.0]),
        x_min=np.array([-0.8, -2.5]),
        x_max=np.array([0.8, 2.5]),
        soft_theta=True,
        slack_weight=5_000.0,
    )

    runs = {
        "LQR": (X_lqr, U_lqr),
        "saturated LQR": (X_sat, U_sat),
        "MPC": (mpc_run["X"], mpc_run["U"]),
        "MPC + rate": (rate_run["X"], rate_run["U"]),
    }
    t = np.arange(steps + 1) * config.DT

    if output_dir is not None:
        figures_dir = Path(output_dir) / "figures"
        plot_time_histories(figures_dir / "time_plots.png", t, runs, config.X_MIN, config.X_MAX, config.U_MIN, config.U_MAX, show=show, close=close)
        plot_phase_plane(figures_dir / "phase_plane.png", runs, config.X_MIN, config.X_MAX, show=show, close=close)
        plot_terminal_geometry(figures_dir / "terminal_ellipses.png", P, config.X_MIN, config.X_MAX, show=show, close=close)
        plot_constraint_activity(
            figures_dir / "constraint_activity.png",
            t,
            rate_run["X"],
            rate_run["U"],
            rate_run["slack"],
            config.X_MIN,
            config.X_MAX,
            config.U_MIN,
            config.U_MAX,
            show=show,
            close=close,
        )
    else:
        plot_time_histories(Path("/tmp/thimpc_project1_time_plots.png"), t, runs, config.X_MIN, config.X_MAX, config.U_MIN, config.U_MAX, show=show, close=close)
        plot_phase_plane(Path("/tmp/thimpc_project1_phase_plane.png"), runs, config.X_MIN, config.X_MAX, show=show, close=close)
        plot_terminal_geometry(Path("/tmp/thimpc_project1_terminal_ellipses.png"), P, config.X_MIN, config.X_MAX, show=show, close=close)
        plot_constraint_activity(
            Path("/tmp/thimpc_project1_constraint_activity.png"),
            t,
            rate_run["X"],
            rate_run["U"],
            rate_run["slack"],
            config.X_MIN,
            config.X_MAX,
            config.U_MIN,
            config.U_MAX,
            show=show,
            close=close,
        )

    return {
        "final_state_norm": {name: float(np.linalg.norm(X[-1])) for name, (X, _) in runs.items()},
        "max_abs_input": {name: float(np.max(np.abs(U))) for name, (_, U) in runs.items()},
        "input_violation_count": {
            name: count_violations(U[:, 0], config.U_MIN[0], config.U_MAX[0]) for name, (_, U) in runs.items()
        },
        "theta_violation_count": {
            name: count_violations(X[:, 0], config.X_MIN[0], config.X_MAX[0]) for name, (X, _) in runs.items()
        },
        "mpc_solver_failures": int(mpc_run["failures"]),
        "rate_mpc_solver_failures": int(rate_run["failures"]),
        "mpc_status_counts": status_counts(mpc_run["statuses"]),
        "rate_mpc_status_counts": status_counts(rate_run["statuses"]),
        "hard_infeasibility_status": hard.status,
        "soft_recovery_status": soft.status,
        "soft_initial_slack": float(soft.slack[0, 0]) if soft.success and soft.slack.size else None,
    }


def main() -> None:
    metrics = run_project(show=False, close=True, output_dir="/tmp/thimpc_project1")
    print(metrics)


if __name__ == "__main__":
    main()
