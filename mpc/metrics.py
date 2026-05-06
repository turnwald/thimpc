"""Small deterministic metrics helpers for scenario scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def to_builtin(value: Any) -> Any:
    """Convert NumPy values into JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    return value


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    """Write metrics JSON with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def status_counts(statuses: list[str]) -> dict[str, int]:
    """Count solver statuses while keeping JSON output simple."""
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_summary(path: str | Path, lines: list[str]) -> None:
    """Write a short text summary for quick inspection of scenario outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def rms(values: np.ndarray) -> float:
    """Root-mean-square value."""
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def count_violations(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-9) -> int:
    """Count samples outside elementwise lower/upper bounds."""
    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return int(np.sum((values < lower - tol) | (values > upper + tol)))
