"""Scenario code for Project 3: learning-enhanced prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from projects._shared.mobile_robot import tracking_error_matrices
from projects.project_3_learning_enhanced_prediction import config
from projects.project_3_learning_enhanced_prediction.plots import plot_prediction_eta, plot_residual_error, plot_rmse
from projects.project_3_learning_enhanced_prediction.residual_model import fit_residual_least_squares, predict_residual


def true_transition(error: np.ndarray, delta_omega: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Measured transition with a small systematic residual."""
    error = np.asarray(error, dtype=float)
    delta_omega = np.asarray(delta_omega, dtype=float).reshape(-1)
    nominal = error @ A.T + delta_omega.reshape(-1, 1) @ B.T
    residual = np.column_stack(
        [
            0.015 * error[:, 1] * error[:, 2],
            0.035 * error[:, 2] + 0.025 * delta_omega * error[:, 1],
            0.018 * delta_omega + 0.01 * error[:, 2] * np.abs(error[:, 2]),
        ]
    )
    return nominal + residual


def generate_data(samples: int, seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    errors = np.column_stack(
        [
            rng.uniform(-0.4, 0.4, samples),
            rng.uniform(-0.55, 0.55, samples),
            rng.uniform(-0.35, 0.35, samples),
        ]
    )
    inputs = rng.uniform(-0.6, 0.6, samples)
    return errors, inputs


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def run_project(samples: int = config.SAMPLES, output_dir: str | Path | None = None, *, show: bool = True, close: bool = False) -> dict[str, object]:
    A, B = tracking_error_matrices(config.DT, config.V_REF, config.OMEGA_REF)
    errors, inputs = generate_data(samples)
    measured_next = true_transition(errors, inputs, A, B)
    nominal_next = errors @ A.T + inputs.reshape(-1, 1) @ B.T
    residual = measured_next - nominal_next

    split = max(20, int(0.7 * samples))
    train = slice(0, split)
    test = slice(split, samples)

    W = fit_residual_least_squares(errors[train], inputs[train], residual[train], ridge=1e-6)
    learned_next = nominal_next[test] + predict_residual(W, errors[test], inputs[test])

    nominal_error = measured_next[test] - nominal_next[test]
    learned_error = measured_next[test] - learned_next
    nominal_rmse = np.sqrt(np.mean(nominal_error * nominal_error, axis=0))
    learned_rmse = np.sqrt(np.mean(learned_error * learned_error, axis=0))

    figures_dir = Path(output_dir) / "figures" if output_dir is not None else Path("/tmp")
    plot_prediction_eta(figures_dir / "prediction_vs_measured_eta.png", measured_next[test], nominal_next[test], learned_next, show=show, close=close)
    plot_residual_error(figures_dir / "residual_prediction_error.png", measured_next[test], nominal_next[test], learned_next, show=show, close=close)
    plot_rmse(figures_dir / "rmse_comparison.png", nominal_rmse, learned_rmse, show=show, close=close)

    return {
        "train_samples": int(split),
        "test_samples": int(samples - split),
        "nominal_rmse_by_state": {
            "xi": float(nominal_rmse[0]),
            "eta": float(nominal_rmse[1]),
            "psi": float(nominal_rmse[2]),
        },
        "learned_rmse_by_state": {
            "xi": float(learned_rmse[0]),
            "eta": float(learned_rmse[1]),
            "psi": float(learned_rmse[2]),
        },
        "nominal_rmse_total": rms(nominal_error),
        "learned_rmse_total": rms(learned_error),
        "residual_weight_matrix": W,
    }


def main() -> None:
    metrics = run_project(show=False, close=True, output_dir="/tmp/thimpc_project3")
    print(metrics)


if __name__ == "__main__":
    main()
