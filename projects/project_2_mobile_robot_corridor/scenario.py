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
    state_from_error,
    tracking_error,
    tracking_error_matrices,
    unicycle_step,
    wrap_angle,
)

CORRIDOR_HALF_WIDTH = getattr(config, "CORRIDOR_HALF_WIDTH", 0.55)
NARROW_HALF_WIDTH = getattr(config, "NARROW_HALF_WIDTH", 0.22)
NARROW_OFFSET = getattr(config, "NARROW_OFFSET", 0.32)
BASELINE_K_XI = 0.03
BASELINE_K_ETA = 0.65
BASELINE_K_PSI = 0.85


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


def clip_delta(raw_delta: float, delta_previous: float) -> float:
    """Apply the project input and input-rate limits to a yaw-rate correction."""
    limited = float(np.clip(raw_delta, config.DELTA_MIN[0], config.DELTA_MAX[0]))
    return float(np.clip(limited, delta_previous - config.DELTA_RATE[0], delta_previous + config.DELTA_RATE[0]))


def baseline_control(error: np.ndarray, delta_previous: float) -> float:
    """Simple local tracker used as the deliberately non-predictive baseline."""
    xi, eta, psi_error = np.asarray(error, dtype=float).reshape(3)
    raw = -(BASELINE_K_XI * xi + BASELINE_K_ETA * eta + BASELINE_K_PSI * psi_error)
    return clip_delta(raw, delta_previous)


def corridor_error_bounds(phi: np.ndarray, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Project-local smooth shifted corridor bounds for the lateral tracking error."""
    phi = np.asarray(phi, dtype=float)
    wrapped = wrap_angle(phi - config.NARROW_CENTER)
    half_angle = config.NARROW_WIDTH / 2.0
    inside_narrow = np.abs(wrapped) <= half_angle
    blend = np.zeros_like(phi, dtype=float)
    blend[inside_narrow] = 0.5 * (1.0 + np.cos(np.pi * wrapped[inside_narrow] / half_angle))

    half_width = CORRIDOR_HALF_WIDTH - blend * (CORRIDOR_HALF_WIDTH - NARROW_HALF_WIDTH) - margin
    center = blend * NARROW_OFFSET
    half_width = np.maximum(half_width, 0.03)
    return center - half_width, center + half_width


def state_bounds_for_horizon(phi: np.ndarray, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    eta_min, eta_max = corridor_error_bounds(phi, margin=margin)
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
        slack_weight=config.MPC_SLACK_WEIGHT,
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
        if mode in {"nominal", "baseline"}:
            delta_k = baseline_control(error, delta_previous)
            statuses.append("baseline")
        elif mode == "saturated_lqr":
            raw = float(lqr_control(error, np.zeros(3), K)[0])
            delta_k = clip_delta(raw, delta_previous)
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
                fallback_raw = float(lqr_control(error, np.zeros(3), K)[0])
                delta_k = clip_delta(fallback_raw, delta_previous)
        else:
            raise ValueError(f"unknown tracker mode: {mode}")

        omega = corrected_yaw_rate(config.OMEGA_REF, delta_k)
        state = unicycle_step(state, config.V_REF, omega, config.DT, yaw_rate_bias=yaw_rate_bias)
        states[k + 1] = state
        errors[k + 1] = tracking_error(state, reference[k + 1])
        delta[k] = delta_k
        delta_previous = delta_k

    eta_min, eta_max = corridor_error_bounds(phi[: steps + 1], margin=margin)
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
    errors = np.asarray(run["errors"])
    return {
        "minimum_margin": float(np.min(run["margin"])),
        "maximum_violation": float(max(0.0, -np.min(run["margin"]))),
        "corridor_violation_count": count_violations(run["errors"][:, 1], run["eta_min"], run["eta_max"]),
        "rms_position_error": rms(np.linalg.norm(errors[:, :2], axis=1)),
        "rms_lateral_error": rms(run["errors"][:, 1]),
        "rms_heading_error": rms(run["errors"][:, 2]),
        "max_abs_input": float(np.max(np.abs(run["delta"]))),
        "max_abs_input_rate": float(np.max(np.abs(run["delta_rate"]))),
        "solver_failures": int(run["failures"]),
        "status_counts": status_counts(run["statuses"]),
        "total_slack": float(np.sum(run["slack"])),
    }


def run_project(steps: int = config.STEPS, output_dir: str | Path | None = None, *, show: bool = True, close: bool = False) -> dict[str, object]:
    from projects.project_2_mobile_robot_corridor.animation import save_corridor_animation

    nominal = simulate_tracker("baseline", steps)
    saturated = simulate_tracker("saturated_lqr", steps)
    hard_mpc = simulate_tracker("mpc", steps)
    mpc = simulate_tracker("mpc", steps, soft_corridor=True)
    soft = mpc
    mismatch = simulate_tracker("mpc", steps, soft_corridor=True, yaw_rate_bias=0.035)
    tightened = simulate_tracker("mpc", steps, margin=0.08, soft_corridor=True)
    horizon_runs = {N: simulate_tracker("mpc", steps, horizon=N, soft_corridor=True) for N in [5, 15, 30]}

    runs = {
        "baseline": nominal,
        "saturated LQR": saturated,
        "MPC": mpc,
    }
    t = np.arange(steps + 1) * config.DT
    project_dir = Path(output_dir) if output_dir is not None else Path("/tmp/thimpc_project2")
    figures_dir = project_dir / "figures"
    plot_top_view(figures_dir / "top_view_trajectory.png", mpc["reference"][:, :2], runs, config.RADIUS, show=show, close=close)
    plot_lateral_error(figures_dir / "lateral_error_corridor.png", t, runs, show=show, close=close)
    plot_input(figures_dir / "yaw_rate_correction.png", t, runs, config.DELTA_MIN[0], config.DELTA_MAX[0], show=show, close=close)
    plot_margin(figures_dir / "minimum_margin.png", t, runs, show=show, close=close)
    plot_horizon_comparison(figures_dir / "horizon_comparison.png", t, horizon_runs, show=show, close=close)
    animation_stride = max(1, steps // 45)
    animation_path = save_corridor_animation(project_dir / "corridor_comparison.gif", runs, stride=animation_stride, interval=80)

    A, B = tracking_error_matrices(config.DT, config.V_REF, config.OMEGA_REF)
    _, P = lqr_gain(A, B, config.Q, config.R)
    hard_u, hard = solve_corridor_mpc(np.array([0.0, 0.0, 0.0]), np.full(config.HORIZON + 1, config.NARROW_CENTER), P, margin=0.04)
    soft_u, soft_single = solve_corridor_mpc(
        np.array([0.0, 0.0, 0.0]),
        np.full(config.HORIZON + 1, config.NARROW_CENTER),
        P,
        margin=0.04,
        soft_corridor=True,
    )
    del hard_u, soft_u

    metrics = {
        "baseline": summarize_run(nominal),
        "nominal": summarize_run(nominal),
        "saturated_lqr": summarize_run(saturated),
        "mpc": summarize_run(mpc),
        "hard_mpc": summarize_run(hard_mpc),
        "soft_mpc": summarize_run(soft),
        "yaw_bias_soft_mpc": summarize_run(mismatch),
        "tightened_soft_mpc": summarize_run(tightened),
        "hard_infeasibility_status": hard.status,
        "soft_recovery_status": soft_single.status,
        "soft_recovery_initial_slack": float(soft_single.slack[0, 0]) if soft_single.success and soft_single.slack.size else None,
        "horizon_minimum_margins": {str(N): float(np.min(run["margin"])) for N, run in horizon_runs.items()},
        "figures_dir": str(figures_dir),
        "animation_path": str(animation_path),
    }
    return metrics


def main() -> None:
    metrics = run_project(show=False, close=True, output_dir="/tmp/thimpc_project2")
    print(metrics)


if __name__ == "__main__":
    main()
