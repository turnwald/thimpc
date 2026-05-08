"""Project-local plots for learning-enhanced prediction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def savefig(path: str | Path, fig: plt.Figure) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def plot_prediction_eta(
    path: str | Path,
    measured: np.ndarray,
    nominal: np.ndarray,
    learned: np.ndarray,
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    idx = np.arange(measured.shape[0])
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(idx, measured[:, 1], label="measured eta[k+1]", linewidth=1.6)
    ax.plot(idx, nominal[:, 1], label="nominal prediction", linewidth=1.2)
    ax.plot(idx, learned[:, 1], label="learned prediction", linewidth=1.2)
    ax.set_xlabel("test sample")
    ax.set_ylabel("next lateral error eta")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_residual_error(
    path: str | Path,
    measured: np.ndarray,
    nominal: np.ndarray,
    learned: np.ndarray,
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    idx = np.arange(measured.shape[0])
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(idx, measured[:, 1] - nominal[:, 1], label="nominal eta residual")
    ax.plot(idx, measured[:, 1] - learned[:, 1], label="post-learning eta error")
    ax.set_xlabel("test sample")
    ax.set_ylabel("prediction error")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_rmse(
    path: str | Path,
    nominal_rmse: np.ndarray,
    learned_rmse: np.ndarray,
    *,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    labels = ["xi", "eta", "psi"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(x - width / 2.0, nominal_rmse, width, label="nominal")
    ax.bar(x + width / 2.0, learned_rmse, width, label="learned residual")
    ax.set_xticks(x, labels)
    ax.set_ylabel("one-step RMSE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    savefig(path, fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax

