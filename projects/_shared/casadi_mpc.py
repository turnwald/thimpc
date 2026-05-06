"""Small linear MPC helper for the application projects.

The class is intentionally shallow: it builds the finite-horizon problem in a
readable way, keeps NumPy arrays at the project boundary, and hides only the
repeated CasADi boilerplate from the walkthrough notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import casadi as ca
except ImportError:  # pragma: no cover - depends on the teaching environment
    ca = None


@dataclass
class MPCInfo:
    """Numerical information returned after one MPC solve."""

    success: bool
    status: str
    X: np.ndarray
    U: np.ndarray
    slack: np.ndarray
    objective: float


def _as_2d_bound(bound: np.ndarray | None, horizon: int, width: int) -> np.ndarray | None:
    if bound is None:
        return None
    value = np.asarray(bound, dtype=float)
    if value.shape == (width,):
        return np.repeat(value.reshape(1, width), horizon + 1, axis=0)
    if value.shape == (horizon + 1, width):
        return value
    raise ValueError(f"expected bound shape {(width,)} or {(horizon + 1, width)}, got {value.shape}")


class LinearCasadiMPC:
    """Finite-horizon constrained linear MPC using CasADi Opti."""

    def __init__(
        self,
        *,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        horizon: int,
        x_min: np.ndarray | None = None,
        x_max: np.ndarray | None = None,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        terminal_cost: np.ndarray | None = None,
        rate_bound: np.ndarray | None = None,
        soften_state_constraints: bool = False,
        soft_state_indices: list[int] | None = None,
        slack_weight: float | None = None,
    ) -> None:
        if ca is None:
            raise ImportError("casadi is required for the project MPC helper")

        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.P = self.Q if terminal_cost is None else np.asarray(terminal_cost, dtype=float)
        self.N = int(horizon)
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]

        if self.A.shape != (self.nx, self.nx):
            raise ValueError("A must be square")
        if self.B.shape[0] != self.nx:
            raise ValueError("B row count must match A")
        if self.Q.shape != (self.nx, self.nx):
            raise ValueError("Q shape must match the state dimension")
        if self.R.shape != (self.nu, self.nu):
            raise ValueError("R shape must match the input dimension")
        if self.P.shape != (self.nx, self.nx):
            raise ValueError("terminal_cost shape must match the state dimension")

        self.x_min = _as_2d_bound(x_min, self.N, self.nx)
        self.x_max = _as_2d_bound(x_max, self.N, self.nx)
        self.u_min = None if u_min is None else np.asarray(u_min, dtype=float).reshape(self.nu)
        self.u_max = None if u_max is None else np.asarray(u_max, dtype=float).reshape(self.nu)
        self.rate_bound = None if rate_bound is None else np.asarray(rate_bound, dtype=float).reshape(self.nu)

        if soft_state_indices is not None:
            self.soft_state_indices = list(soft_state_indices)
        elif soften_state_constraints:
            self.soft_state_indices = list(range(self.nx))
        else:
            self.soft_state_indices = []
        self.slack_weight = 0.0 if slack_weight is None else float(slack_weight)

    def _reference_sequence(self, reference: np.ndarray | None) -> np.ndarray:
        if reference is None:
            return np.zeros((self.N + 1, self.nx))
        ref = np.asarray(reference, dtype=float)
        if ref.shape == (self.nx,):
            return np.repeat(ref.reshape(1, self.nx), self.N + 1, axis=0)
        if ref.shape == (self.N + 1, self.nx):
            return ref
        raise ValueError(f"expected reference shape {(self.nx,)} or {(self.N + 1, self.nx)}, got {ref.shape}")

    def solve(self, x0: np.ndarray, reference: np.ndarray | None = None, u_previous: np.ndarray | None = None) -> tuple[np.ndarray, MPCInfo]:
        """Solve one MPC problem and return the first input plus solve details."""
        x0 = np.asarray(x0, dtype=float).reshape(self.nx)
        ref = self._reference_sequence(reference)
        if self.rate_bound is not None:
            if u_previous is None:
                u_previous = np.zeros(self.nu)
            u_previous = np.asarray(u_previous, dtype=float).reshape(self.nu)

        opti = ca.Opti()
        X = opti.variable(self.nx, self.N + 1)
        U = opti.variable(self.nu, self.N)
        if self.soft_state_indices:
            S = opti.variable(len(self.soft_state_indices), self.N + 1)
            opti.subject_to(S >= 0)
        else:
            S = None

        cost: Any = 0
        opti.subject_to(X[:, 0] == x0.reshape(self.nx, 1))

        for k in range(self.N):
            x_error = X[:, k] - ref[k].reshape(self.nx, 1)
            uk = U[:, k]
            cost += ca.mtimes([x_error.T, self.Q, x_error]) + ca.mtimes([uk.T, self.R, uk])
            opti.subject_to(X[:, k + 1] == ca.mtimes(self.A, X[:, k]) + ca.mtimes(self.B, uk))

            if self.u_min is not None:
                opti.subject_to(uk >= self.u_min)
            if self.u_max is not None:
                opti.subject_to(uk <= self.u_max)
            if self.rate_bound is not None:
                previous = u_previous if k == 0 else U[:, k - 1]
                opti.subject_to(uk - previous <= self.rate_bound)
                opti.subject_to(uk - previous >= -self.rate_bound)

        terminal_error = X[:, self.N] - ref[self.N].reshape(self.nx, 1)
        cost += ca.mtimes([terminal_error.T, self.P, terminal_error])

        soft_row = {state_index: row for row, state_index in enumerate(self.soft_state_indices)}
        for k in range(self.N + 1):
            for i in range(self.nx):
                slack = 0 if S is None or i not in soft_row else S[soft_row[i], k]
                if self.x_min is not None and np.isfinite(self.x_min[k, i]):
                    opti.subject_to(X[i, k] >= self.x_min[k, i] - slack)
                if self.x_max is not None and np.isfinite(self.x_max[k, i]):
                    opti.subject_to(X[i, k] <= self.x_max[k, i] + slack)
                if S is not None and i in soft_row:
                    cost += self.slack_weight * slack * slack

        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {
                "print_time": False,
                "ipopt": {"print_level": 0, "sb": "yes", "max_iter": 200},
            },
        )

        try:
            solution = opti.solve()
            X_raw = np.asarray(solution.value(X), dtype=float)
            U_raw = np.asarray(solution.value(U), dtype=float)
            X_value = X_raw.reshape(self.N + 1, self.nx) if X_raw.ndim == 1 else X_raw.T
            U_value = U_raw.reshape(self.N, self.nu) if U_raw.ndim == 1 else U_raw.T
            slack_value = np.zeros((0, self.N + 1)) if S is None else np.asarray(solution.value(S), dtype=float)
            if slack_value.ndim == 1:
                slack_value = slack_value.reshape(1, -1)
            info = MPCInfo(
                success=True,
                status=str(opti.stats().get("return_status", "Solve_Succeeded")),
                X=X_value,
                U=U_value,
                slack=slack_value,
                objective=float(solution.value(cost)),
            )
            return U_value[0].copy(), info
        except RuntimeError:
            try:
                status = str(opti.stats().get("return_status", "Solve_Failed"))
            except RuntimeError:
                status = "Solve_Failed"
            info = MPCInfo(
                success=False,
                status=status,
                X=np.full((self.N + 1, self.nx), np.nan),
                U=np.full((self.N, self.nu), np.nan),
                slack=np.full((len(self.soft_state_indices), self.N + 1), np.nan),
                objective=float("nan"),
            )
            return np.zeros(self.nu), info
