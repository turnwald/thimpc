"""Scenario code for Project 2: mobile robot corridor tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

from controllers.lqr import lqr_control
from controllers.nominal_tracker import corrected_yaw_rate
from projects._shared.casadi_mpc import LinearCasadiMPC
from projects.project_2_mobile_robot_corridor import config
from projects.project_2_mobile_robot_corridor.plots import (
    plot_horizon_comparison,
    plot_input,
    plot_lateral_error,
    plot_margin,
    plot_top_view,
)
from systems.mobile_robot import (
    circular_reference,
    corridor_bounds,
    state_from_error,
    tracking_error,
    tracking_error_matrices,
    unicycle_step,
)


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def count_violations(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-9) -> int:
    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return int(np.sum((values < lower - tol) | (values > upper + tol)))


def status_counts(statuses: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


def state_bounds_for_horizon(phi: np.ndarray, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    eta_min, eta_max = corridor_bounds(
        phi,
        narrow_center=config.NARROW_CENTER,
        narrow_width=config.NARROW_WIDTH,
        margin=margin,
    )
    lower = np.column_stack([np.full_like(phi, -2.0), eta_min, np.full_like(phi, -1.2)])
    upper = np.column_stack([np.full_like(phi, 2.0), eta_max, np.full_like(phi, 1.2)])
    return lower, upper


def solve_corridor_mpc(
    error: np.ndarray,
    phi_window: np.ndarray,
    P: np.ndarray,
    *,
    horizon: int = config.HORIZON,
    margin: float = 0.0,
    delta_previous: float = 0.0,
    soft_corridor: bool = False,
) -> tuple[np.ndarray, object]:
    A, B = tracking_error_matrices(config.DT, config.V_REF, config.OMEGA_REF)
    lower, upper = state_bounds_for_horizon(phi_window, margin=margin)
    mpc = LinearCasadiMPC(
        A=A,
        B=B,
        Q=config.Q,
        R=config.R,
        horizon=horizon,
        x_min=lower,
        x_max=upper,
        u_min=config.DELTA_MIN,
        u_max=config.DELTA_MAX,
        terminal_cost=P,
        rate_bound=config.DELTA_RATE,
        soft_state_indices=[1] if soft_corridor else None,
        slack_weight=20_000.0,
    )
    return mpc.solve(error, u_previous=np.array([delta_previous]))


def simulate_tracker(
    mode: str,
    steps: int,
    *,
    horizon: int = config.HORIZON,
    margin: float = 0.0,
    yaw_rate_bias: float = 0.0,
    soft_corridor: bool = False,
) -> dict[str, np.ndarray | int | list[str] | float]:
    A, B = tracking_error_matrices(config.DT, config.V_REF, config.OMEGA_REF)
    K, P = lqr_gain(A, B, config.Q, config.R)

    t = np.arange(steps + horizon + 2) * config.DT
    reference = circular_reference(t, config.RADIUS, config.V_REF, config.OMEGA_REF)
    phi = config.OMEGA_REF * t

    state = state_from_error(config.INITIAL_ERROR, reference[0])
    states = np.zeros((steps + 1, 3))
    errors = np.zeros((steps + 1, 3))
    delta = np.zeros(steps)
    slack = np.zeros(steps)
    statuses: list[str] = []
    failures = 0
    states[0] = state
    errors[0] = tracking_error(state, reference[0])
    delta_previous = 0.0

    for k in range(steps):
        error = tracking_error(state, reference[k])
        if mode == "nominal":
            delta_k = 0.0
            statuses.append("nominal")
        elif mode == "saturated_lqr":
            raw = float(lqr_control(error, np.zeros(3), K)[0])
            delta_k = float(np.clip(raw, config.DELTA_MIN[0], config.DELTA_MAX[0]))
            delta_k = float(np.clip(delta_k, delta_previous - config.DELTA_RATE[0], delta_previous + config.DELTA_RATE[0]))
            statuses.append("saturated_lqr")
        elif mode == "mpc":
            u0, info = solve_corridor_mpc(
                error,
                phi[k : k + horizon + 1],
                P,
                horizon=horizon,
                margin=margin,
                delta_previous=delta_previous,
                soft_corridor=soft_corridor,
            )
            statuses.append(info.status)
            if info.success:
                delta_k = float(u0[0])
                if info.slack.size:
                    slack[k] = float(info.slack[0, 0])
            else:
                failures += 1
                delta_k = 0.0
        else:
            raise ValueError(f"unknown tracker mode: {mode}")

        omega = corrected_yaw_rate(config.OMEGA_REF, delta_k)
        state = unicycle_step(state, config.V_REF, omega, config.DT, yaw_rate_bias=yaw_rate_bias)
        states[k + 1] = state
        errors[k + 1] = tracking_error(state, reference[k + 1])
        delta[k] = delta_k
        delta_previous = delta_k

    eta_min, eta_max = corridor_bounds(
        phi[: steps + 1],
        narrow_center=config.NARROW_CENTER,
        narrow_width=config.NARROW_WIDTH,
        margin=margin,
    )
    margin_history = np.minimum(errors[:, 1] - eta_min, eta_max - errors[:, 1])
    return {
        "states": states,
        "errors": errors,
        "delta": delta,
        "delta_rate": np.diff(np.r_[0.0, delta]),
        "eta_min": eta_min,
        "eta_max": eta_max,
        "margin": margin_history,
        "statuses": statuses,
        "failures": failures,
        "slack": slack,
        "reference": reference[: steps + 1],
        "radius": config.RADIUS,
    }


def summarize_run(run: dict[str, np.ndarray | int | list[str] | float]) -> dict[str, object]:
    return {
        "minimum_margin": float(np.min(run["margin"])),
        "maximum_violation": float(max(0.0, -np.min(run["margin"]))),
        "corridor_violation_count": count_violations(run["errors"][:, 1], run["eta_min"], run["eta_max"]),
        "rms_lateral_error": rms(run["errors"][:, 1]),
        "rms_heading_error": rms(run["errors"][:, 2]),
        "max_abs_input": float(np.max(np.abs(run["delta"]))),
        "max_abs_input_rate": float(np.max(np.abs(run["delta_rate"]))),
        "solver_failures": int(run["failures"]),
        "status_counts": status_counts(run["statuses"]),
        "total_slack": float(np.sum(run["slack"])),
    }


def run_project(steps: int = config.STEPS, output_dir: str | Path | None = None, *, show: bool = True, close: bool = False) -> dict[str, object]:
    nominal = simulate_tracker("nominal", steps)
    saturated = simulate_tracker("saturated_lqr", steps)
    mpc = simulate_tracker("mpc", steps)
    soft = simulate_tracker("mpc", steps, soft_corridor=True)
    mismatch = simulate_tracker("mpc", steps, soft_corridor=True, yaw_rate_bias=0.035)
    tightened = simulate_tracker("mpc", steps, margin=0.08, soft_corridor=True)
    horizon_runs = {N: simulate_tracker("mpc", steps, horizon=N, soft_corridor=True) for N in [5, 15, 30]}

    runs = {
        "nominal": nominal,
        "saturated LQR": saturated,
        "MPC": mpc,
        "soft MPC": soft,
    }
    t = np.arange(steps + 1) * config.DT
    figures_dir = Path(output_dir) / "figures" if output_dir is not None else Path("/tmp")
    plot_top_view(figures_dir / "top_view_trajectory.png", mpc["reference"][:, :2], runs, config.RADIUS, show=show, close=close)
    plot_lateral_error(figures_dir / "lateral_error_corridor.png", t, runs, show=show, close=close)
    plot_input(figures_dir / "yaw_rate_correction.png", t, runs, config.DELTA_MIN[0], config.DELTA_MAX[0], show=show, close=close)
    plot_margin(figures_dir / "minimum_margin.png", t, runs, show=show, close=close)
    plot_horizon_comparison(figures_dir / "horizon_comparison.png", t, horizon_runs, show=show, close=close)

    A, B = tracking_error_matrices(config.DT, config.V_REF, config.OMEGA_REF)
    _, P = lqr_gain(A, B, config.Q, config.R)
    hard_u, hard = solve_corridor_mpc(np.array([0.0, 0.45, 0.0]), np.full(config.HORIZON + 1, config.NARROW_CENTER), P, margin=0.04)
    soft_u, soft_single = solve_corridor_mpc(
        np.array([0.0, 0.45, 0.0]),
        np.full(config.HORIZON + 1, config.NARROW_CENTER),
        P,
        margin=0.04,
        soft_corridor=True,
    )
    del hard_u, soft_u

    metrics = {
        "nominal": summarize_run(nominal),
        "saturated_lqr": summarize_run(saturated),
        "mpc": summarize_run(mpc),
        "soft_mpc": summarize_run(soft),
        "yaw_bias_soft_mpc": summarize_run(mismatch),
        "tightened_soft_mpc": summarize_run(tightened),
        "hard_infeasibility_status": hard.status,
        "soft_recovery_status": soft_single.status,
        "soft_recovery_initial_slack": float(soft_single.slack[0, 0]) if soft_single.success and soft_single.slack.size else None,
        "horizon_minimum_margins": {str(N): float(np.min(run["margin"])) for N, run in horizon_runs.items()},
    }
    return metrics


def main() -> None:
    metrics = run_project(show=False, close=True, output_dir="/tmp/thimpc_project2")
    print(metrics)


if __name__ == "__main__":
    main()
