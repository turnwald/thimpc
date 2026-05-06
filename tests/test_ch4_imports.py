import pytest


def test_ch4_modules_import():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    pytest.importorskip("casadi")

    import controllers.lqr  # noqa: F401
    import controllers.nominal_tracker  # noqa: F401
    import controllers.saturated_lqr  # noqa: F401
    import mpc.casadi_linear_mpc  # noqa: F401
    import mpc.metrics  # noqa: F401
    import mpc.plotting  # noqa: F401
    import mpc.terminal_tools  # noqa: F401
    import systems.double_integrator  # noqa: F401
    import systems.mobile_robot  # noqa: F401
    import systems.residual_models  # noqa: F401
    import tools.check_student_release  # noqa: F401
    import tools.create_student_material  # noqa: F401
