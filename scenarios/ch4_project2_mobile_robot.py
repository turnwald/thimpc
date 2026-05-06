"""Chapter 4 Project 2: mobile robot local tracking in a safety corridor."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from controllers.lqr import discrete_lqr, lqr_control
from controllers.nominal_tracker import corrected_yaw_rate
from mpc.casadi_linear_mpc import solve_linear_mpc
from mpc.metrics import count_violations, rms, status_counts, write_metrics, write_summary
from mpc.plotting import plot_robot_top_view, savefig
from mpc.terminal_tools import dare_terminal_cost
from systems.mobile_robot import (
    circular_reference,
    corridor_bounds,
    state_from_error,
    tracking_error,
    tracking_error_matrices,
    unicycle_step,
)


def _state_bounds_for_horizon(phi: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    eta_min, eta_max = corridor_bounds(phi, margin=margin)
    lower = np.column_stack([np.full_like(phi, -2.0), eta_min, np.full_like(phi, -1.2)])
    upper = np.column_stack([np.full_like(phi, 2.0), eta_max, np.full_like(phi, 1.2)])
    return lower, upper


def simulate_tracker(
    mode: str,
    steps: int,
    horizon: int,
    margin: float = 0.0,
    yaw_rate_bias: float = 0.0,
    soft_corridor: bool = False,
) -> dict[str, np.ndarray | int | list[str]]:
    """Simulate nominal, saturated-LQR, or MPC correction around a circle."""
    dt = 0.1
    v_r = 0.8
    omega_r = 0.2
    radius = v_r / omega_r
    Q = np.diag([1.0, 25.0, 4.0])
    R = np.array([[0.35]])
    A, B = tracking_error_matrices(dt, v_r, omega_r)
    K, P = discrete_lqr(A, B, Q, R)
    delta_bounds = (np.array([-0.7]), np.array([0.7]))
    delta_rate = np.array([0.12])

    t = np.arange(steps + horizon + 2) * dt
    reference = circular_reference(t, radius, v_r, omega_r)
    phi = omega_r * t

    initial_error = np.array([0.0, 0.38, 0.12])
    state = state_from_error(initial_error, reference[0])
    states = np.zeros((steps + 1, 3))
    errors = np.zeros((steps + 1, 3))
    delta = np.zeros(steps)
    statuses: list[str] = []
    failures = 0
    slack = np.zeros(steps)
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
            delta_k = float(np.clip(raw, delta_bounds[0][0], delta_bounds[1][0]))
            delta_k = float(np.clip(delta_k, delta_previous - delta_rate[0], delta_previous + delta_rate[0]))
            statuses.append("saturated_lqr")
        elif mode == "mpc":
            lower, upper = _state_bounds_for_horizon(phi[k : k + horizon + 1], margin)
            result = solve_linear_mpc(
                A,
                B,
                error,
                horizon,
                Q,
                R,
                P_terminal=P,
                u_bounds=delta_bounds,
                x_bounds_sequence=(lower, upper),
                rate_bound=delta_rate,
                u_previous=np.array([delta_previous]),
                soften_state_indices=[1] if soft_corridor else None,
                slack_penalty=20_000.0,
            )
            statuses.append(result.status)
            if result.success:
                delta_k = float(result.u0[0])
                if result.slack.size:
                    slack[k] = float(result.slack[0, 0])
            else:
                failures += 1
                delta_k = 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        omega = corrected_yaw_rate(omega_r, delta_k)
        state = unicycle_step(state, v_r, omega, dt, yaw_rate_bias=yaw_rate_bias)
        states[k + 1] = state
        errors[k + 1] = tracking_error(state, reference[k + 1])
        delta[k] = delta_k
        delta_previous = delta_k

    eta_min, eta_max = corridor_bounds(phi[: steps + 1], margin=margin)
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
        "radius": radius,
    }


def plot_lateral_error(
    path: Path,
    t: np.ndarray,
    runs: dict[str, dict[str, np.ndarray]],
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    first = next(iter(runs.values()))
    ax.fill_between(t, first["eta_min"], first["eta_max"], color="0.9", label="corridor")
    for label, run in runs.items():
        ax.plot(t, run["errors"][:, 1], label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lateral error eta [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_delta(
    path: Path,
    t: np.ndarray,
    runs: dict[str, dict[str, np.ndarray]],
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for label, run in runs.items():
        ax.step(t[:-1], run["delta"], where="post", label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Delta omega [rad/s]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_input_rate(
    path: Path,
    t: np.ndarray,
    runs: dict[str, dict[str, np.ndarray]],
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for label, run in runs.items():
        ax.step(t[:-1], run["delta_rate"], where="post", label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Delta omega rate [rad/s/sample]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_margin(
    path: Path,
    t: np.ndarray,
    runs: dict[str, dict[str, np.ndarray]],
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for label, run in runs.items():
        ax.plot(t, run["margin"], label=label)
    ax.axhline(0.0, color="k", linewidth=0.9)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("minimum corridor margin [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_horizon_comparison(
    path: Path,
    t: np.ndarray,
    horizon_runs: dict[int, dict[str, np.ndarray]],
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for horizon, run in horizon_runs.items():
        ax.plot(t, run["errors"][:, 1], label=f"N = {horizon}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lateral error eta [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def run_project2(steps: int, output_dir: str | Path, *, show: bool = True, close: bool = False) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    nominal = simulate_tracker("nominal", steps, horizon=15)
    saturated = simulate_tracker("saturated_lqr", steps, horizon=15)
    mpc = simulate_tracker("mpc", steps, horizon=15)
    soft = simulate_tracker("mpc", steps, horizon=15, soft_corridor=True)
    mismatch = simulate_tracker("mpc", steps, horizon=15, soft_corridor=True, yaw_rate_bias=0.035)
    tightened = simulate_tracker("mpc", steps, horizon=15, margin=0.08, soft_corridor=True)
    horizon_runs = {N: simulate_tracker("mpc", steps, horizon=N, soft_corridor=True) for N in [5, 15, 30]}

    dt = 0.1
    t = np.arange(steps + 1) * dt
    runs = {
        "nominal": nominal,
        "saturated LQR": saturated,
        "MPC": mpc,
        "soft MPC": soft,
    }

    radius = float(mpc["radius"])
    plot_robot_top_view(
        figures_dir / "top_view_trajectory.png",
        mpc["reference"][:, :2],
        {label: run["states"] for label, run in runs.items()},
        radius=radius,
        inner_radius=radius - 0.55,
        outer_radius=radius + 0.55,
        show=show,
        close=close,
    )
    plot_lateral_error(figures_dir / "lateral_error_corridor.png", t, runs, show=show, close=close)
    plot_delta(figures_dir / "yaw_rate_correction.png", t, runs, show=show, close=close)
    plot_input_rate(
        figures_dir / "input_rate.png",
        t,
        {"saturated LQR": saturated, "MPC": mpc, "soft MPC": soft},
        show=show,
        close=close,
    )
    plot_margin(figures_dir / "minimum_margin.png", t, runs, show=show, close=close)
    plot_horizon_comparison(figures_dir / "horizon_comparison.png", t, horizon_runs, show=show, close=close)

    A, B = tracking_error_matrices(0.1, 0.8, 0.2)
    Q = np.diag([1.0, 25.0, 4.0])
    R = np.array([[0.35]])
    P = dare_terminal_cost(A, B, Q, R)
    lower, upper = _state_bounds_for_horizon(np.full(16, np.pi), margin=0.04)
    hard = solve_linear_mpc(
        A,
        B,
        np.array([0.0, 0.45, 0.0]),
        15,
        Q,
        R,
        P_terminal=P,
        u_bounds=(np.array([-0.7]), np.array([0.7])),
        x_bounds_sequence=(lower, upper),
    )
    soft_single = solve_linear_mpc(
        A,
        B,
        np.array([0.0, 0.45, 0.0]),
        15,
        Q,
        R,
        P_terminal=P,
        u_bounds=(np.array([-0.7]), np.array([0.7])),
        x_bounds_sequence=(lower, upper),
        soften_state_indices=[1],
        slack_penalty=20_000.0,
    )

    metrics = {}
    for label, run in {
        "nominal": nominal,
        "saturated_lqr": saturated,
        "mpc": mpc,
        "soft_mpc": soft,
        "yaw_bias_soft_mpc": mismatch,
        "tightened_soft_mpc": tightened,
    }.items():
        metrics[label] = {
            "minimum_corridor_margin": float(np.min(run["margin"])),
            "corridor_violation_count": count_violations(run["errors"][:, 1], run["eta_min"], run["eta_max"]),
            "rms_lateral_error": rms(run["errors"][:, 1]),
            "rms_heading_error": rms(run["errors"][:, 2]),
            "max_abs_delta_omega": float(np.max(np.abs(run["delta"]))),
            "max_abs_delta_omega_rate": float(np.max(np.abs(run["delta_rate"]))),
            "solver_failures": int(run["failures"]),
            "status_counts": status_counts(run["statuses"]),
            "total_slack": float(np.sum(run["slack"])),
        }
    metrics["hard_infeasibility_status"] = hard.status
    metrics["soft_recovery_status"] = soft_single.status
    metrics["soft_recovery_initial_slack"] = float(soft_single.slack[0, 0]) if soft_single.success and soft_single.slack.size else None
    metrics["horizon_minimum_margins"] = {str(N): float(np.min(run["margin"])) for N, run in horizon_runs.items()}
    write_metrics(output_dir / "metrics.json", metrics)
    write_summary(
        output_dir / "summary.txt",
        [
            "Chapter 4 Project 2: mobile robot corridor tracking",
            f"steps: {steps}",
            f"MPC solver failures: {metrics['mpc']['solver_failures']}",
            f"soft MPC solver failures: {metrics['soft_mpc']['solver_failures']}",
            f"MPC minimum corridor margin: {metrics['mpc']['minimum_corridor_margin']:.6g}",
            f"hard infeasibility status: {metrics['hard_infeasibility_status']}",
            f"soft recovery status: {metrics['soft_recovery_status']}",
        ],
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output-dir", default="outputs/ch4_project2")
    args = parser.parse_args()
    run_project2(args.steps, args.output_dir, show=False, close=True)


if __name__ == "__main__":
    main()
