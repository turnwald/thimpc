import pytest


def test_ch4_modules_import():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    pytest.importorskip("casadi")

    import projects._shared.casadi_mpc  # noqa: F401
    import projects.project_1_attitude_constraints.scenario  # noqa: F401
    import projects.project_2_mobile_robot_corridor.scenario  # noqa: F401
    import projects.project_3_learning_enhanced_prediction.scenario  # noqa: F401
    import tools.check_student_release  # noqa: F401
    import tools.create_student_material  # noqa: F401
