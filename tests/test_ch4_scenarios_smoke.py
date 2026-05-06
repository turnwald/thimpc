import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.create_student_material import study_notebooks


SCENARIOS = [
    (
        "scenarios.ch4_project1_attitude",
        ["--steps", "8"],
        [
            "figures/time_plots.png",
            "figures/phase_plane.png",
            "figures/terminal_ellipses.png",
            "figures/constraint_activity.png",
            "metrics.json",
            "summary.txt",
        ],
    ),
    (
        "scenarios.ch4_project2_mobile_robot",
        ["--steps", "8"],
        [
            "figures/top_view_trajectory.png",
            "figures/lateral_error_corridor.png",
            "figures/yaw_rate_correction.png",
            "figures/input_rate.png",
            "figures/minimum_margin.png",
            "figures/horizon_comparison.png",
            "metrics.json",
            "summary.txt",
        ],
    ),
    (
        "scenarios.ch4_project3_learning",
        ["--samples", "50"],
        [
            "figures/prediction_vs_measured_eta.png",
            "figures/residual_prediction_error.png",
            "figures/rmse_comparison.png",
            "metrics.json",
            "summary.txt",
        ],
    ),
]


def run_module(module: str, args: list[str], out_dir: Path, env: dict[str, str] | None = None) -> None:
    if env is None:
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(Path.cwd()), "MPLBACKEND": "Agg"})
    command = [sys.executable, "-m", module, *args, "--output-dir", str(out_dir)]
    subprocess.run(command, check=True, env=env)


@pytest.mark.parametrize(("module", "args", "expected"), SCENARIOS)
def test_scenario_smoke_outputs(tmp_path, module, args, expected):
    pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    if module != "scenarios.ch4_project3_learning":
        pytest.importorskip("casadi")
        pytest.importorskip("scipy")

    out_dir = tmp_path / module.rsplit(".", 1)[-1]
    run_module(module, args, out_dir)

    for name in expected:
        path = out_dir / name
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics
    assert (out_dir / "summary.txt").read_text(encoding="utf-8").strip()


def test_project1_metrics_sanity(tmp_path):
    pytest.importorskip("casadi")

    out_dir = tmp_path / "project1"
    run_module("scenarios.ch4_project1_attitude", ["--steps", "8"], out_dir)
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["mpc_solver_failures"] == 0
    assert metrics["rate_mpc_solver_failures"] == 0
    assert metrics["hard_infeasibility_status"]
    assert metrics["soft_recovery_status"]
    assert "mpc_status_counts" in metrics


def test_project2_metrics_sanity(tmp_path):
    pytest.importorskip("casadi")

    out_dir = tmp_path / "project2"
    run_module("scenarios.ch4_project2_mobile_robot", ["--steps", "8"], out_dir)
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["mpc"]["solver_failures"] == 0
    assert metrics["soft_mpc"]["solver_failures"] == 0
    assert metrics["mpc"]["corridor_violation_count"] == 0
    assert "status_counts" in metrics["mpc"]


def test_project3_metrics_sanity(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")

    out_dir = tmp_path / "project3"
    run_module("scenarios.ch4_project3_learning", ["--samples", "50"], out_dir)
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["train_samples"] > 0
    assert metrics["test_samples"] > 0
    assert metrics["learned_rmse_total"] < metrics["nominal_rmse_total"]


NOTEBOOKS = study_notebooks()
NOTEBOOK_SMOKE_PATHS = [str(path) for path in NOTEBOOKS if path.exists()] + [str(path) for path in NOTEBOOKS.values()]


@pytest.mark.parametrize("notebook", NOTEBOOK_SMOKE_PATHS)
def test_notebook_executes_in_smoke_mode(tmp_path, notebook):
    pytest.importorskip("nbconvert")
    if "study_03" in notebook:
        pytest.importorskip("osqp")
    if "study_04" in notebook or "study_05" in notebook or "study_06" in notebook:
        pytest.importorskip("casadi")
    if "study_07" not in notebook:
        pytest.importorskip("scipy")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(Path.cwd()),
            "MPLBACKEND": "Agg",
            "THIMPC_CH4_PROJECT1_STEPS": "8",
            "THIMPC_CH4_PROJECT2_STEPS": "8",
            "THIMPC_CH4_PROJECT3_SAMPLES": "50",
            "THIMPC_STUDY03_STEPS": "8",
            "THIMPC_STUDY04_STEPS": "4",
            "THIMPC_OUTPUT_ROOT": str(tmp_path / "outputs"),
            "JUPYTER_CONFIG_DIR": str(tmp_path / "jupyter_config"),
            "JUPYTER_DATA_DIR": str(tmp_path / "jupyter_data"),
            "JUPYTER_RUNTIME_DIR": str(tmp_path / "jupyter_runtime"),
            "IPYTHONDIR": str(tmp_path / "ipython"),
        }
    )
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        notebook,
        "--output-dir",
        str(tmp_path),
        "--output",
        Path(notebook).name,
        "--ExecutePreprocessor.kernel_name=python3",
        "--ExecutePreprocessor.timeout=180",
    ]
    subprocess.run(command, check=True, env=env)
    output_name = tmp_path / Path(notebook).name
    assert output_name.exists()
    assert output_name.stat().st_size > 0
