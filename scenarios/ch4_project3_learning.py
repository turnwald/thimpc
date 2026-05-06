"""Chapter 4 Project 3: learning-enhanced prediction model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mpc.metrics import rms, write_metrics, write_summary
from mpc.plotting import savefig
from systems.mobile_robot import tracking_error_matrices
from systems.residual_models import fit_residual_least_squares, predict_residual


def true_transition(error: np.ndarray, delta_omega: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Measured transition with small systematic mismatch."""
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


def generate_data(samples: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
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


def plot_prediction_eta(path: Path, measured: np.ndarray, nominal: np.ndarray, learned: np.ndarray) -> None:
    idx = np.arange(measured.shape[0])
    plt.figure(figsize=(8.0, 4.4))
    plt.plot(idx, measured[:, 1], label="measured eta[k+1]", linewidth=1.6)
    plt.plot(idx, nominal[:, 1], label="nominal prediction", linewidth=1.2)
    plt.plot(idx, learned[:, 1], label="learned prediction", linewidth=1.2)
    plt.xlabel("test sample")
    plt.ylabel("next lateral error eta")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    savefig(path)


def plot_residual_error(path: Path, measured: np.ndarray, nominal: np.ndarray, learned: np.ndarray) -> None:
    nominal_error = measured - nominal
    learned_error = measured - learned
    idx = np.arange(measured.shape[0])
    plt.figure(figsize=(8.0, 4.4))
    plt.plot(idx, nominal_error[:, 1], label="nominal eta residual")
    plt.plot(idx, learned_error[:, 1], label="post-learning eta error")
    plt.xlabel("test sample")
    plt.ylabel("prediction error")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    savefig(path)


def plot_rmse(path: Path, nominal_rmse: np.ndarray, learned_rmse: np.ndarray) -> None:
    labels = ["xi", "eta", "psi"]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(6.4, 4.4))
    plt.bar(x - width / 2.0, nominal_rmse, width, label="nominal")
    plt.bar(x + width / 2.0, learned_rmse, width, label="learned residual")
    plt.xticks(x, labels)
    plt.ylabel("one-step RMSE")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="best")
    savefig(path)


def run_project3(samples: int, output_dir: str | Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    A, B = tracking_error_matrices(dt=0.1, v_r=0.8, omega_r=0.2)
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

    plot_prediction_eta(figures_dir / "prediction_vs_measured_eta.png", measured_next[test], nominal_next[test], learned_next)
    plot_residual_error(figures_dir / "residual_prediction_error.png", measured_next[test], nominal_next[test], learned_next)
    plot_rmse(figures_dir / "rmse_comparison.png", nominal_rmse, learned_rmse)

    metrics = {
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
    write_metrics(output_dir / "metrics.json", metrics)
    write_summary(
        output_dir / "summary.txt",
        [
            "Chapter 4 Project 3: learning-enhanced prediction",
            f"samples: {samples}",
            f"train samples: {metrics['train_samples']}",
            f"test samples: {metrics['test_samples']}",
            f"nominal RMSE total: {metrics['nominal_rmse_total']:.6g}",
            f"learned RMSE total: {metrics['learned_rmse_total']:.6g}",
            "Learning improves one-step prediction here; it does not replace constraints.",
        ],
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--output-dir", default="outputs/ch4_project3")
    args = parser.parse_args()
    run_project3(args.samples, args.output_dir)


if __name__ == "__main__":
    main()
