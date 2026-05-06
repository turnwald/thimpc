import os

import pytest


def use_headless_matplotlib() -> None:
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg", force=True)


def test_new_project_modules_import():
    pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")

    import projects.project_1_attitude_constraints.scenario  # noqa: F401
    import projects.project_2_mobile_robot_corridor.scenario  # noqa: F401
    import projects.project_3_learning_enhanced_prediction.scenario  # noqa: F401


def test_project1_reduced_run(tmp_path):
    pytest.importorskip("casadi")
    pytest.importorskip("scipy")
    use_headless_matplotlib()

    from projects.project_1_attitude_constraints import scenario

    metrics = scenario.run_project(steps=3, output_dir=tmp_path / "project1", show=False, close=True)
    assert metrics["mpc_solver_failures"] == 0
    assert metrics["hard_infeasibility_status"]
    assert (tmp_path / "project1" / "figures" / "time_plots.png").exists()


def test_project2_reduced_run(tmp_path):
    pytest.importorskip("casadi")
    pytest.importorskip("scipy")
    use_headless_matplotlib()

    from projects.project_2_mobile_robot_corridor import scenario

    metrics = scenario.run_project(steps=3, output_dir=tmp_path / "project2", show=False, close=True)
    assert metrics["mpc"]["solver_failures"] == 0
    assert "minimum_margin" in metrics["nominal"]
    assert (tmp_path / "project2" / "figures" / "minimum_margin.png").exists()


def test_project3_reduced_run(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    use_headless_matplotlib()

    from projects.project_3_learning_enhanced_prediction import scenario

    metrics = scenario.run_project(samples=50, output_dir=tmp_path / "project3", show=False, close=True)
    assert metrics["learned_rmse_total"] < metrics["nominal_rmse_total"]
    assert (tmp_path / "project3" / "figures" / "rmse_comparison.png").exists()
