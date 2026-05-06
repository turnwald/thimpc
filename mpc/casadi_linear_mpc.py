"""Transparent finite-horizon linear MPC builder using CasADi Opti.

The function intentionally keeps the formulation visible for teaching:
states and inputs are decision variables, dynamics are equality constraints,
cost terms are added stage by stage, and hard/soft bounds are explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import casadi as ca
except ImportError:  # pragma: no cover - exercised only outside the course env
    ca = None


@dataclass
class LinearMPCResult:
    """Result returned by solve_linear_mpc."""

    success: bool
    status: str
    u0: np.ndarray
    X: np.ndarray
    U: np.ndarray
    slack: np.ndarray
    objective: float


def _as_bound_sequence(
    bounds: tuple[np.ndarray, np.ndarray] | None,
    sequence_bounds: tuple[np.ndarray, np.ndarray] | None,
    horizon: int,
    nx: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if sequence_bounds is not None:
        lower, upper = sequence_bounds
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if bounds is None:
        return None
    lower, upper = bounds
    lower = np.asarray(lower, dtype=float).reshape(nx)
    upper = np.asarray(upper, dtype=float).reshape(nx)
    return np.repeat(lower.reshape(1, nx), horizon + 1, axis=0), np.repeat(upper.reshape(1, nx), horizon + 1, axis=0)


def solve_linear_mpc(
    A: np.ndarray,
    B: np.ndarray,
    x0: np.ndarray,
    horizon: int,
    Q: np.ndarray,
    R: np.ndarray,
    P_terminal: np.ndarray | None = None,
    u_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    x_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    x_bounds_sequence: tuple[np.ndarray, np.ndarray] | None = None,
    rate_bound: np.ndarray | None = None,
    u_previous: np.ndarray | None = None,
    soften_state_indices: list[int] | None = None,
    slack_penalty: float = 0.0,
) -> LinearMPCResult:
    """Solve one finite-horizon linear MPC problem.

    Parameters use row-major NumPy arrays for readability. Internally, CasADi
    columns are used: X[:, k] is the predicted state and U[:, k] is the input.
    """
    if ca is None:
        raise ImportError("casadi is required to solve the Chapter 4 MPC problems")

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    P_terminal = Q if P_terminal is None else np.asarray(P_terminal, dtype=float)

    nx = A.shape[0]
    nu = B.shape[1]
    N = int(horizon)
    soften_state_indices = [] if soften_state_indices is None else list(soften_state_indices)

    opti = ca.Opti()
    X = opti.variable(nx, N + 1)
    U = opti.variable(nu, N)
    if soften_state_indices:
        Eps = opti.variable(len(soften_state_indices), N + 1)
        opti.subject_to(Eps >= 0)
    else:
        Eps = None

    cost = 0
    opti.subject_to(X[:, 0] == x0.reshape(nx, 1))

    if u_bounds is not None:
        u_min, u_max = u_bounds
        u_min = np.asarray(u_min, dtype=float).reshape(nu)
        u_max = np.asarray(u_max, dtype=float).reshape(nu)
    else:
        u_min = u_max = None

    state_bounds = _as_bound_sequence(x_bounds, x_bounds_sequence, N, nx)
    if state_bounds is not None:
        x_min_seq, x_max_seq = state_bounds
    else:
        x_min_seq = x_max_seq = None

    if rate_bound is not None:
        rate_bound = np.asarray(rate_bound, dtype=float).reshape(nu)
        if u_previous is None:
            u_previous = np.zeros(nu)
        u_previous = np.asarray(u_previous, dtype=float).reshape(nu)

    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]

        cost += ca.mtimes([xk.T, Q, xk]) + ca.mtimes([uk.T, R, uk])
        opti.subject_to(X[:, k + 1] == ca.mtimes(A, xk) + ca.mtimes(B, uk))

        if u_bounds is not None:
            opti.subject_to(uk >= u_min)
            opti.subject_to(uk <= u_max)

        if rate_bound is not None:
            previous = u_previous if k == 0 else U[:, k - 1]
            opti.subject_to(uk - previous <= rate_bound)
            opti.subject_to(uk - previous >= -rate_bound)

    terminal = X[:, N]
    cost += ca.mtimes([terminal.T, P_terminal, terminal])

    if state_bounds is not None:
        soft_position = {state_index: row for row, state_index in enumerate(soften_state_indices)}
        for k in range(N + 1):
            for i in range(nx):
                lower = x_min_seq[k, i]
                upper = x_max_seq[k, i]
                if i in soft_position:
                    eps = Eps[soft_position[i], k]
                    opti.subject_to(X[i, k] >= lower - eps)
                    opti.subject_to(X[i, k] <= upper + eps)
                    cost += slack_penalty * eps * eps
                else:
                    opti.subject_to(X[i, k] >= lower)
                    opti.subject_to(X[i, k] <= upper)

    opti.minimize(cost)
    options = {
        "print_time": False,
        "ipopt": {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 200,
        },
    }
    opti.solver("ipopt", options)

    try:
        solution = opti.solve()
        X_raw = np.asarray(solution.value(X), dtype=float)
        U_raw = np.asarray(solution.value(U), dtype=float)
        if X_raw.ndim == 1:
            X_value = X_raw.reshape(N + 1, nx)
        else:
            X_value = X_raw.T
        if U_raw.ndim == 1:
            U_value = U_raw.reshape(N, nu)
        else:
            U_value = U_raw.T
        if Eps is None:
            slack_value = np.zeros((0, N + 1))
        else:
            slack_value = np.asarray(solution.value(Eps), dtype=float)
            if slack_value.ndim == 1:
                slack_value = slack_value.reshape(1, -1)
        return LinearMPCResult(
            success=True,
            status=str(opti.stats().get("return_status", "Solve_Succeeded")),
            u0=U_value[0].copy(),
            X=X_value,
            U=U_value,
            slack=slack_value,
            objective=float(solution.value(cost)),
        )
    except RuntimeError:
        try:
            status = str(opti.stats().get("return_status", "Solve_Failed"))
        except RuntimeError:
            status = "Solve_Failed"
        return LinearMPCResult(
            success=False,
            status=status,
            u0=np.zeros(nu),
            X=np.full((N + 1, nx), np.nan),
            U=np.full((N, nu), np.nan),
            slack=np.full((len(soften_state_indices), N + 1), np.nan),
            objective=float("nan"),
        )
