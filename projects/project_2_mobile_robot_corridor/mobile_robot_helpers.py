"""Small helper functions for Project 2: mobile robot corridor MPC."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.linalg import expm, solve_discrete_are


@dataclass(frozen=True)
class CorridorParams:
    phi_center: float
    phi_half_width: float
    eta_min_normal: float
    eta_max_normal: float
    eta_min_narrow: float
    eta_max_narrow: float


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def discretize_continuous_system(A_c, B_c, Ts):
    nx, nu = B_c.shape
    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = A_c
    M[:nx, nx:] = B_c

    M_d = expm(M * Ts)
    A_d = M_d[:nx, :nx]
    B_d = M_d[:nx, nx:]
    return A_d, B_d


def dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def phi_sequence(phi0, omega_r, Ts, steps):
    return phi0 + omega_r * Ts * np.arange(steps + 1)


def eta_bounds_from_path(phi_values, corridor):
    phi_values = np.asarray(phi_values, dtype=float)
    eta_min = corridor.eta_min_normal * np.ones_like(phi_values)
    eta_max = corridor.eta_max_normal * np.ones_like(phi_values)

    in_narrow_section = np.abs(wrap_to_pi(phi_values - corridor.phi_center)) <= corridor.phi_half_width
    eta_min[in_narrow_section] = corridor.eta_min_narrow
    eta_max[in_narrow_section] = corridor.eta_max_narrow

    return eta_min, eta_max, in_narrow_section


def eta_bounds_horizon(phi_k, omega_r, Ts, N, corridor):
    phi_horizon = phi_k + omega_r * Ts * np.arange(N + 1)
    eta_min, eta_max, active = eta_bounds_from_path(phi_horizon, corridor)
    return phi_horizon, eta_min, eta_max, active


def reference_from_phi(phi_values, radius):
    phi_values = np.asarray(phi_values, dtype=float)
    px = radius * np.cos(phi_values)
    py = radius * np.sin(phi_values)
    psi = phi_values + 0.5 * np.pi
    return px, py, psi


def global_state_from_error(error, phi, radius):
    """Convert tracking error to global rover state.

    The error is expressed in the rover body frame, consistent with the
    nonlinear tracking controller used in the project.
    """
    error = np.asarray(error, dtype=float)
    px_ref, py_ref, psi_ref = reference_from_phi(np.array([phi]), radius)

    xi, eta, psi_error = error
    psi = wrap_to_pi(float(psi_ref[0]) + psi_error)

    px = float(px_ref[0]) + np.cos(psi) * xi - np.sin(psi) * eta
    py = float(py_ref[0]) + np.sin(psi) * xi + np.cos(psi) * eta
    return np.array([px, py, psi])


def tracking_error(global_state, phi, radius):
    """Compute tracking error in the rover body frame."""
    global_state = np.asarray(global_state, dtype=float)
    px_ref, py_ref, psi_ref = reference_from_phi(np.array([phi]), radius)

    dx = global_state[0] - float(px_ref[0])
    dy = global_state[1] - float(py_ref[0])
    psi = global_state[2]

    xi = np.cos(psi) * dx + np.sin(psi) * dy
    eta = -np.sin(psi) * dx + np.cos(psi) * dy
    psi_error = wrap_to_pi(psi - float(psi_ref[0]))
    return np.array([xi, eta, psi_error])


def nominal_tracking_controller(error, v_r, omega_r, K_xi, K_eta, K_psi):
    xi, eta, psi_error = error
    v_c = -K_xi * xi + v_r * np.cos(psi_error)
    omega_c = -K_psi * np.sin(psi_error) + omega_r - K_eta * v_r * eta
    return np.array([v_c, omega_c])


def unicycle_step(global_state, control, Ts):
    px, py, psi = np.asarray(global_state, dtype=float)
    v, omega = np.asarray(control, dtype=float)

    px_next = px + Ts * v * np.cos(psi)
    py_next = py + Ts * v * np.sin(psi)
    psi_next = wrap_to_pi(psi + Ts * omega)
    return np.array([px_next, py_next, psi_next])


def global_path_from_error(phi_values, error_trajectory, radius):
    phi_values = np.asarray(phi_values, dtype=float)
    error_trajectory = np.asarray(error_trajectory, dtype=float)

    px = np.zeros_like(phi_values)
    py = np.zeros_like(phi_values)

    for k, phi in enumerate(phi_values):
        state = global_state_from_error(error_trajectory[k], phi, radius)
        px[k] = state[0]
        py[k] = state[1]

    return px, py


def eta_margin(phi_values, eta_values, corridor):
    eta_min, eta_max, _ = eta_bounds_from_path(phi_values, corridor)
    return np.minimum(eta_values - eta_min, eta_max - eta_values)


def print_summary_table(results, phi_values, corridor):
    print("controller              min eta margin    final |e|      max |u_e|")
    print("----------------------------------------------------------------")
    for name, data in results.items():
        margin = eta_margin(phi_values, data["x"][:, 1], corridor)
        final_error = np.linalg.norm(data["x"][-1])
        max_input = np.max(np.linalg.norm(data["u"], axis=1)) if len(data["u"]) else 0.0
        print(f"{name:<23s} {np.min(margin):>14.3f}    {final_error:>9.3f}    {max_input:>9.3f}")


def plot_corridor_preview(phi_values, corridor):
    eta_min, eta_max, active = eta_bounds_from_path(phi_values, corridor)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.fill_between(phi_values, eta_min, eta_max, alpha=0.20, label="allowed eta box")
    ax.plot(phi_values, eta_min, linestyle="--", label=r"$\eta_{\min}$")
    ax.plot(phi_values, eta_max, linestyle="--", label=r"$\eta_{\max}$")
    ax.axhline(0.0, linewidth=1.0, label="reference path")
    if np.any(active):
        ax.axvspan(phi_values[active][0], phi_values[active][-1], alpha=0.12, label="narrowed section")
    ax.set_xlabel(r"path variable $\phi$ [rad]")
    ax.set_ylabel(r"lateral error $\eta$ [m]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_top_view(results, phi_values, corridor, radius):
    phi_dense = np.linspace(phi_values[0], phi_values[-1], 300)
    px_ref, py_ref, _ = reference_from_phi(phi_dense, radius)
    eta_min, eta_max, active_dense = eta_bounds_from_path(phi_dense, corridor)

    px_min, py_min = global_path_from_error(
        phi_dense,
        np.column_stack([np.zeros_like(phi_dense), eta_min, np.zeros_like(phi_dense)]),
        radius,
    )
    px_max, py_max = global_path_from_error(
        phi_dense,
        np.column_stack([np.zeros_like(phi_dense), eta_max, np.zeros_like(phi_dense)]),
        radius,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(px_ref, py_ref, linestyle="--", label="reference path")
    ax.plot(px_min, py_min, linewidth=1.0, label=r"$\eta_{\min}$")
    ax.plot(px_max, py_max, linewidth=1.0, label=r"$\eta_{\max}$")

    if np.any(active_dense):
        px_narrow_ref, py_narrow_ref, _ = reference_from_phi(phi_dense[active_dense], radius)
        ax.scatter(px_narrow_ref, py_narrow_ref, s=14, label="narrowed section")

    for name, data in results.items():
        if "z" in data:
            ax.plot(data["z"][:, 0], data["z"][:, 1], label=name)
        else:
            px, py = global_path_from_error(phi_values, data["x"], radius)
            ax.plot(px, py, label=name)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$p_x$ [m]")
    ax.set_ylabel(r"$p_y$ [m]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_eta_history(results, phi_values, corridor):
    eta_min, eta_max, active = eta_bounds_from_path(phi_values, corridor)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(phi_values, eta_min, eta_max, alpha=0.20, label="allowed eta box")
    ax.plot(phi_values, eta_min, linestyle="--", label=r"$\eta_{\min}$")
    ax.plot(phi_values, eta_max, linestyle="--", label=r"$\eta_{\max}$")
    ax.axhline(0.0, linewidth=1.0, label="reference path")

    for name, data in results.items():
        ax.plot(phi_values, data["x"][:, 1], label=name)

    if np.any(active):
        ax.axvspan(phi_values[active][0], phi_values[active][-1], alpha=0.12, label="narrowed section")

    ax.set_xlabel(r"path variable $\phi$ [rad]")
    ax.set_ylabel(r"lateral error $\eta$ [m]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_input_history(results, Ts):
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    for name, data in results.items():
        time_u = Ts * np.arange(data["u"].shape[0])
        axes[0].plot(time_u, data["u"][:, 0], label=name)
        axes[1].plot(time_u, data["u"][:, 1], label=name)

    axes[0].set_ylabel(r"$v_e$")
    axes[1].set_ylabel(r"$\omega_e$")
    axes[1].set_xlabel("time [s]")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()



def create_top_view_animation(results, phi_values, corridor, radius, interval=80):
    """Create a top-view animation using the same result interface as the plots.

    Each result entry is expected to contain either:
    - z: global trajectory with columns [p_x, p_y, psi], or
    - x: local error trajectory, which is reconstructed to global position.

    The returned FuncAnimation object should be assigned to a variable in a
    notebook cell so that it is not garbage-collected before display or saving.
    """
    phi_dense = np.linspace(phi_values[0], phi_values[-1], 300)
    px_ref, py_ref, _ = reference_from_phi(phi_dense, radius)
    eta_min, eta_max, active_dense = eta_bounds_from_path(phi_dense, corridor)

    px_min, py_min = global_path_from_error(
        phi_dense,
        np.column_stack([np.zeros_like(phi_dense), eta_min, np.zeros_like(phi_dense)]),
        radius,
    )
    px_max, py_max = global_path_from_error(
        phi_dense,
        np.column_stack([np.zeros_like(phi_dense), eta_max, np.zeros_like(phi_dense)]),
        radius,
    )

    trajectories = {}
    for name, data in results.items():
        if "z" in data:
            trajectories[name] = (data["z"][:, 0], data["z"][:, 1])
        else:
            trajectories[name] = global_path_from_error(phi_values, data["x"], radius)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(px_ref, py_ref, linestyle="--", label="reference path")
    ax.plot(px_min, py_min, linewidth=1.0, label=r"$\eta_{\min}$")
    ax.plot(px_max, py_max, linewidth=1.0, label=r"$\eta_{\max}$")

    if np.any(active_dense):
        px_narrow_ref, py_narrow_ref, _ = reference_from_phi(phi_dense[active_dense], radius)
        ax.scatter(px_narrow_ref, py_narrow_ref, s=14, label="narrowed section")

    lines = {}
    markers = {}
    for name in trajectories:
        (line,) = ax.plot([], [], label=name)
        (marker,) = ax.plot([], [], marker="o", linestyle="")
        lines[name] = line
        markers[name] = marker

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$p_x$ [m]")
    ax.set_ylabel(r"$p_y$ [m]")
    ax.grid(True)
    ax.legend()

    n_frames = len(phi_values)

    def update(frame):
        artists = []
        for name, (px, py) in trajectories.items():
            lines[name].set_data(px[: frame + 1], py[: frame + 1])
            markers[name].set_data([px[frame]], [py[frame]])
            artists.extend([lines[name], markers[name]])
        return artists

    return FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)


def save_top_view_animation(filename, results, phi_values, corridor, radius, interval=80, fps=15):
    """Save a top-view animation using matplotlib's available writers."""
    animation = create_top_view_animation(results, phi_values, corridor, radius, interval=interval)
    animation.save(filename, fps=fps)
    return animation
