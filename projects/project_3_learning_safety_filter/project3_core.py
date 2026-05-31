"""Core helpers for Project 3: learning an MPC expert with a CBF filter.

Teaching model
--------------
The MPC expert, the learned policy, and the CBF-QP all operate on the simple
2D end-effector point model

    p_{k+1} = p_k + Ts v_k.

The planar arm in the notebook is only a visual replay layer.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 1. Scenario and point model
# ---------------------------------------------------------------------------


def default_scenario() -> dict:
    """Return the deterministic scenario used throughout the walkthrough."""

    return {
        "Ts": 0.10,
        "N": 12,
        "p_min": np.array([-1.05, -0.75], dtype=float),
        "p_max": np.array([1.05, 0.75], dtype=float),
        "v_min": np.array([-1.0, -1.0], dtype=float),
        "v_max": np.array([1.0, 1.0], dtype=float),
        "p_start": np.array([-0.92, -0.42], dtype=float),
        "p_goal": np.array([0.92, 0.42], dtype=float),
        "p_obs": np.array([0.02, 0.02], dtype=float),
        "r_safe": 0.31,
        "Q": np.diag([3.0, 3.0]),
        "R": 0.06 * np.eye(2),
        "P": np.diag([16.0, 16.0]),
        "alpha": 2.0,
        "seed": 7,
    }


def ensure_output_dir(path: str | Path = "outputs/project_3_learning_safety_filter") -> Path:
    """Create and return the generated-output directory."""

    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def point_step(p: np.ndarray, v: np.ndarray, scenario: dict) -> np.ndarray:
    """One clipped Euler step of the end-effector point model."""

    v_clip = np.clip(np.asarray(v, dtype=float), scenario["v_min"], scenario["v_max"])
    p_next = np.asarray(p, dtype=float) + scenario["Ts"] * v_clip
    return np.clip(p_next, scenario["p_min"], scenario["p_max"])


# ---------------------------------------------------------------------------
# 2. Safety function and CBF residual
# ---------------------------------------------------------------------------


def safety_function(p: np.ndarray, scenario: dict) -> np.ndarray:
    """Evaluate h(p) = ||p - p_obs||^2 - r_safe^2."""

    p_array = np.asarray(p, dtype=float)
    d = p_array - scenario["p_obs"]
    return np.sum(d * d, axis=-1) - scenario["r_safe"] ** 2


def safety_gradient(p: np.ndarray, scenario: dict) -> np.ndarray:
    """Gradient of h(p) with respect to p."""

    return 2.0 * (np.asarray(p, dtype=float) - scenario["p_obs"])


def cbf_residual(p: np.ndarray, v: np.ndarray, scenario: dict) -> float:
    """Return rho(p, v) = grad h(p)^T v + alpha h(p).

    A nonnegative residual means the first-order CBF condition is satisfied.
    """

    p = np.asarray(p, dtype=float)
    v = np.asarray(v, dtype=float)
    h_now = float(safety_function(p, scenario))
    return float(safety_gradient(p, scenario) @ v + float(scenario["alpha"]) * h_now)


# ---------------------------------------------------------------------------
# 3. MPC expert and expert-data generation
# ---------------------------------------------------------------------------


def sample_safe_points(scenario: dict, n_samples: int, seed: int = 7) -> np.ndarray:
    """Sample safe points, biased toward the start-goal corridor."""

    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    p_min = scenario["p_min"]
    p_max = scenario["p_max"]
    start = scenario["p_start"]
    goal = scenario["p_goal"]

    while len(samples) < n_samples:
        if rng.random() < 0.70:
            lam = rng.uniform(0.0, 1.0)
            candidate = (1.0 - lam) * start + lam * goal
            candidate += rng.normal(scale=np.array([0.18, 0.22]), size=2)
        else:
            candidate = rng.uniform(p_min, p_max)
        candidate = np.clip(candidate, p_min, p_max)
        if safety_function(candidate, scenario) > 0.025:
            samples.append(candidate)

    return np.vstack(samples)


def build_mpc_expert_solver(scenario: dict):
    """Build the CasADi/Ipopt receding-horizon MPC expert.

    The returned function maps the current end-effector point p to the first
    MPC velocity v_MPC and a solver status string.

    Deliberate teaching choice: this function does not silently replace MPC
    with another controller. If CasADi/Ipopt is unavailable or a solve fails,
    the error is raised. Dataset generation catches such errors and skips the
    affected sample instead of mixing fallback-controller data into the MPC
    dataset.
    """

    try:
        import casadi as ca
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Project 3 expert-data generation requires CasADi. "
            "Use the course devcontainer or install casadi."
        ) from exc

    N = int(scenario["N"])
    Ts = float(scenario["Ts"])
    opti = ca.Opti()

    p = opti.variable(2, N + 1)
    v = opti.variable(2, N)
    p0 = opti.parameter(2)

    p_goal = ca.DM(scenario["p_goal"])
    p_obs = ca.DM(scenario["p_obs"])
    p_min = ca.DM(scenario["p_min"])
    p_max = ca.DM(scenario["p_max"])
    v_min = ca.DM(scenario["v_min"])
    v_max = ca.DM(scenario["v_max"])
    Q = ca.DM(scenario["Q"])
    R = ca.DM(scenario["R"])
    P = ca.DM(scenario["P"])
    r_safe = float(scenario["r_safe"])

    objective = 0
    opti.subject_to(p[:, 0] == p0)
    for k in range(N):
        e = p[:, k] - p_goal
        objective += ca.mtimes([e.T, Q, e]) + ca.mtimes([v[:, k].T, R, v[:, k]])
        opti.subject_to(p[:, k + 1] == p[:, k] + Ts * v[:, k])
        opti.subject_to(p_min <= p[:, k])
        opti.subject_to(p[:, k] <= p_max)
        opti.subject_to(v_min <= v[:, k])
        opti.subject_to(v[:, k] <= v_max)
        d = p[:, k] - p_obs
        opti.subject_to(ca.dot(d, d) >= r_safe**2)

    eN = p[:, N] - p_goal
    objective += ca.mtimes([eN.T, P, eN])
    opti.subject_to(p_min <= p[:, N])
    opti.subject_to(p[:, N] <= p_max)
    dN = p[:, N] - p_obs
    opti.subject_to(ca.dot(dN, dN) >= r_safe**2)
    opti.minimize(objective)

    opti.solver(
        "ipopt",
        {"print_time": False},
        {"print_level": 0, "max_iter": 120, "tol": 1e-5, "acceptable_tol": 1e-4},
    )

    def solve(current_p: np.ndarray) -> tuple[np.ndarray, str]:
        current_p = np.asarray(current_p, dtype=float)
        opti.set_value(p0, current_p)

        # Straight-line initialization only helps the nonlinear solver.  It is
        # not the expert policy.
        direction = scenario["p_goal"] - current_p
        for k in range(N + 1):
            guess = current_p + (k / N) * direction
            opti.set_initial(p[:, k], np.clip(guess, scenario["p_min"], scenario["p_max"]))
        v_guess = np.clip(direction / max(N * Ts, Ts), scenario["v_min"], scenario["v_max"])
        for k in range(N):
            opti.set_initial(v[:, k], v_guess)

        solution = opti.solve()
        return np.array(solution.value(v[:, 0]), dtype=float).reshape(2), "Solve_Succeeded"

    return solve


def proportional_goal_controller(p: np.ndarray, scenario: dict, gain: float = 1.6) -> np.ndarray:
    """Simple diagnostic baseline, not used to generate MPC expert data."""

    v = gain * (scenario["p_goal"] - np.asarray(p, dtype=float))
    return np.clip(v, scenario["v_min"], scenario["v_max"])


def generate_expert_dataset(
    scenario: dict | None = None,
    n_samples: int = 120,
    seed: int = 7,
    max_attempts_factor: int = 5,
) -> dict:
    """Generate a compact imitation-learning dataset with the MPC expert.

    Failed nonlinear MPC solves are not replaced by another controller. They
    are skipped and reported separately as skipped_status.
    """

    scenario = default_scenario() if scenario is None else scenario
    solve_mpc = build_mpc_expert_solver(scenario)

    accepted_p: list[np.ndarray] = []
    accepted_v: list[np.ndarray] = []
    accepted_status: list[str] = []
    skipped_status: list[str] = []
    attempts = 0
    max_attempts = max(n_samples, max_attempts_factor * n_samples)

    while len(accepted_p) < n_samples and attempts < max_attempts:
        # Sample in small chunks to keep the code simple and deterministic.
        candidate = sample_safe_points(scenario, n_samples=1, seed=seed + attempts)[0]
        attempts += 1
        try:
            v_mpc, status = solve_mpc(candidate)
        except RuntimeError as exc:
            skipped_status.append(f"skipped_solve_failure: {str(exc).splitlines()[0]}")
            continue
        accepted_p.append(candidate)
        accepted_v.append(v_mpc)
        accepted_status.append(status)

    if len(accepted_p) < n_samples:
        raise RuntimeError(
            f"Only {len(accepted_p)} of {n_samples} MPC samples could be generated. "
            "Check the scenario, horizon, or solver installation."
        )

    points = np.vstack(accepted_p)
    velocities = np.vstack(accepted_v)
    return {
        "p": points,
        "v_mpc": velocities,
        "features": feature_matrix(points, scenario),
        "status": np.array(accepted_status, dtype="U64"),
        "skipped_status": np.array(skipped_status, dtype="U128"),
    }


# ---------------------------------------------------------------------------
# 4. Dataset I/O and policy features
# ---------------------------------------------------------------------------


def feature_matrix(p: np.ndarray, scenario: dict) -> np.ndarray:
    """Build normalized policy inputs from p and visible scenario quantities."""

    p_batch = np.asarray(p, dtype=float)
    if p_batch.ndim == 1:
        p_batch = p_batch[None, :]

    center = 0.5 * (scenario["p_min"] + scenario["p_max"])
    half_width = 0.5 * (scenario["p_max"] - scenario["p_min"])
    scale = np.maximum(half_width, 1e-9)
    mean_scale = float(np.mean(scale))

    p_norm = (p_batch - center) / scale
    goal_rel = (scenario["p_goal"] - p_batch) / scale
    obs_rel = (scenario["p_obs"] - p_batch) / scale
    radius = np.full((p_batch.shape[0], 1), scenario["r_safe"] / mean_scale)
    h_norm = safety_function(p_batch, scenario)[:, None] / (mean_scale**2)
    return np.hstack([p_norm, goal_rel, obs_rel, radius, h_norm])


def save_dataset(path: str | Path, dataset: dict) -> None:
    """Save a generated expert dataset."""

    np.savez(
        path,
        p=dataset["p"],
        v_mpc=dataset["v_mpc"],
        features=dataset["features"],
        status=dataset["status"],
        skipped_status=dataset.get("skipped_status", np.array([], dtype="U128")),
    )


def load_dataset(path: str | Path) -> dict:
    """Load a saved expert dataset."""

    data = np.load(path, allow_pickle=False)
    result = {
        "p": data["p"],
        "v_mpc": data["v_mpc"],
        "features": data["features"],
        "status": data["status"],
    }
    if "skipped_status" in data.files:
        result["skipped_status"] = data["skipped_status"]
    return result


# ---------------------------------------------------------------------------
# 5. Neural-network MPC surrogate
# ---------------------------------------------------------------------------


def initialize_policy(input_dim: int, hidden_dim: int = 32, seed: int = 7) -> dict:
    """Initialize a two-hidden-layer tanh MLP in plain NumPy."""

    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(scale=0.35, size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "W2": rng.normal(scale=0.25, size=(hidden_dim, hidden_dim)),
        "b2": np.zeros(hidden_dim),
        "W3": rng.normal(scale=0.18, size=(hidden_dim, 2)),
        "b3": np.zeros(2),
    }


def _policy_forward_features(policy: dict, x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Forward pass of the NumPy MLP on already-built features."""

    z1 = x @ policy["W1"] + policy["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ policy["W2"] + policy["b2"]
    a2 = np.tanh(z2)
    y = a2 @ policy["W3"] + policy["b3"]
    cache = {"x": x, "a1": a1, "a2": a2}
    return y, cache


def policy_velocity(policy: dict, p: np.ndarray, scenario: dict) -> np.ndarray:
    """Evaluate the learned MPC surrogate at one point or a batch."""

    x = feature_matrix(p, scenario)
    y, _ = _policy_forward_features(policy, x)
    y = np.clip(y, scenario["v_min"], scenario["v_max"])
    if np.asarray(p).ndim == 1:
        return y[0]
    return y


# ---------------------------------------------------------------------------
# 6. Imitation-learning training loop
# ---------------------------------------------------------------------------


def train_policy(
    dataset: dict,
    scenario: dict,
    epochs: int = 650,
    learning_rate: float = 3e-3,
    batch_size: int = 32,
    seed: int = 7,
    snapshot_epochs: tuple[int, ...] = (0, 1, 3, 8, 15, 30, 60, 100, 180, 300, 450, 650),
) -> tuple[dict, list[dict], dict]:
    """Train the NumPy MLP to imitate the first MPC input.

    This is the learning step of the project:

        pi_theta(features(p, scenario)) ~= v_MPC(p, scenario).

    The loss is the mean squared error between the network output and the MPC
    expert velocity stored in the dataset.  Gradients are backpropagated
    manually and the parameters are updated with Adam.
    """

    rng = np.random.default_rng(seed)
    x_all = np.asarray(dataset["features"], dtype=float)
    y_all = np.asarray(dataset["v_mpc"], dtype=float)
    policy = initialize_policy(x_all.shape[1], seed=seed)

    order = rng.permutation(x_all.shape[0])
    n_train = max(8, int(0.80 * x_all.shape[0]))
    train_idx = order[:n_train]
    val_idx = order[n_train:]
    if val_idx.size == 0:
        val_idx = train_idx

    adam_m = {key: np.zeros_like(value) for key, value in policy.items()}
    adam_v = {key: np.zeros_like(value) for key, value in policy.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step_count = 0
    history: list[dict] = []
    snapshots: dict[int, np.ndarray] = {}

    def loss(indices: np.ndarray) -> float:
        pred, _ = _policy_forward_features(policy, x_all[indices])
        err = pred - y_all[indices]
        return float(np.mean(np.sum(err * err, axis=1)))

    for epoch in range(epochs + 1):
        if epoch in snapshot_epochs:
            snapshots[epoch] = simulate_learned_policy(policy, scenario, steps=55, use_cbf=False)["p"]

        history.append({"epoch": epoch, "train_loss": loss(train_idx), "val_loss": loss(val_idx)})
        if epoch == epochs:
            break

        shuffled = rng.permutation(train_idx)
        for start in range(0, shuffled.size, batch_size):
            batch_idx = shuffled[start : start + batch_size]
            x = x_all[batch_idx]
            y_target = y_all[batch_idx]

            # Forward pass: current surrogate prediction v_NN.
            y_pred, cache = _policy_forward_features(policy, x)
            batch_n = max(1, x.shape[0])

            # Supervised imitation gradient for ||v_NN - v_MPC||^2.
            dy = (2.0 / batch_n) * (y_pred - y_target)
            grads = {}
            grads["W3"] = cache["a2"].T @ dy
            grads["b3"] = np.sum(dy, axis=0)

            da2 = dy @ policy["W3"].T
            dz2 = da2 * (1.0 - cache["a2"] ** 2)
            grads["W2"] = cache["a1"].T @ dz2
            grads["b2"] = np.sum(dz2, axis=0)

            da1 = dz2 @ policy["W2"].T
            dz1 = da1 * (1.0 - cache["a1"] ** 2)
            grads["W1"] = cache["x"].T @ dz1
            grads["b1"] = np.sum(dz1, axis=0)

            step_count += 1
            for key in policy:
                adam_m[key] = beta1 * adam_m[key] + (1.0 - beta1) * grads[key]
                adam_v[key] = beta2 * adam_v[key] + (1.0 - beta2) * (grads[key] ** 2)
                m_hat = adam_m[key] / (1.0 - beta1**step_count)
                v_hat = adam_v[key] / (1.0 - beta2**step_count)
                policy[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    snapshot_epochs_kept = np.array(sorted(snapshots), dtype=int)
    snapshot_rollouts = np.stack([snapshots[int(epoch)] for epoch in snapshot_epochs_kept])
    snapshot_data = {"epochs": snapshot_epochs_kept, "p": snapshot_rollouts}
    return policy, history, snapshot_data


def save_policy(path: str | Path, policy: dict) -> None:
    """Save a NumPy MLP policy."""

    np.savez(path, **policy)


def load_policy(path: str | Path) -> dict:
    """Load a NumPy MLP policy."""

    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def save_training_history(path: str | Path, history: list[dict]) -> None:
    """Save training history as a small CSV file."""

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def load_training_history(path: str | Path) -> dict:
    """Load training history without requiring pandas."""

    rows = np.genfromtxt(path, delimiter=",", names=True)
    if rows.shape == ():
        rows = np.array([rows], dtype=rows.dtype)
    return {name: np.asarray(rows[name]) for name in rows.dtype.names}


def save_training_snapshots(path: str | Path, snapshots: dict) -> None:
    """Save rollout snapshots collected during training."""

    np.savez(path, epochs=snapshots["epochs"], p=snapshots["p"])


def load_training_snapshots(path: str | Path) -> dict:
    """Load rollout snapshots collected during training."""

    data = np.load(path, allow_pickle=False)
    return {"epochs": data["epochs"], "p": data["p"]}


# ---------------------------------------------------------------------------
# 7. CBF-QP safety filter
# ---------------------------------------------------------------------------


def solve_velocity_qp(v_nom: np.ndarray, a: np.ndarray, b: float, scenario: dict) -> tuple[np.ndarray, dict]:
    """Solve min 0.5 ||v - v_nom||^2 subject to a^T v >= b and bounds.

    This helper contains the numerical details.  The teaching-level CBF
    function below only has to form a and b from h(p).
    """

    v_nom = np.asarray(v_nom, dtype=float)
    a = np.asarray(a, dtype=float)
    v_min = scenario["v_min"]
    v_max = scenario["v_max"]
    v0 = np.clip(v_nom, v_min, v_max)

    def margin(v: np.ndarray) -> float:
        return float(a @ v - b)

    if margin(v0) >= -1e-10:
        return v0, {"status": "nominal_feasible", "success": True, "margin": margin(v0)}

    result = minimize(
        fun=lambda v: 0.5 * float(np.sum((v - v_nom) ** 2)),
        x0=v0,
        jac=lambda v: v - v_nom,
        bounds=list(zip(v_min, v_max)),
        constraints=[{"type": "ineq", "fun": margin, "jac": lambda v: a}],
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 60, "disp": False},
    )
    v_safe = np.clip(result.x, v_min, v_max)
    if bool(result.success and margin(v_safe) >= -1e-7):
        return v_safe, {"status": str(result.message), "success": True, "margin": margin(v_safe)}

    v_safe, success = _best_feasible_box_projection(v0, v_nom, a, b, scenario)
    return v_safe, {"status": "projection_fallback", "success": success, "margin": margin(v_safe)}


def _best_feasible_box_projection(
    v0: np.ndarray,
    v_nom: np.ndarray,
    a: np.ndarray,
    b: float,
    scenario: dict,
) -> tuple[np.ndarray, bool]:
    """Deterministic two-dimensional backup for the tiny CBF-QP.

    It tries the half-space projection and all relevant box-boundary
    intersections, then returns the feasible candidate closest to v_nom.
    """

    v_min = scenario["v_min"]
    v_max = scenario["v_max"]

    candidates: list[np.ndarray] = []
    denom = float(a @ a)
    if denom > 1e-12:
        candidates.append(v0 + ((b - float(a @ v0)) / denom) * a)

    # Box corners.
    for vx in (v_min[0], v_max[0]):
        for vy in (v_min[1], v_max[1]):
            candidates.append(np.array([vx, vy], dtype=float))

    # Intersections of a^T v = b with box edges.
    if abs(a[1]) > 1e-12:
        for vx in (v_min[0], v_max[0]):
            candidates.append(np.array([vx, (b - a[0] * vx) / a[1]], dtype=float))
    if abs(a[0]) > 1e-12:
        for vy in (v_min[1], v_max[1]):
            candidates.append(np.array([(b - a[1] * vy) / a[0], vy], dtype=float))

    feasible: list[np.ndarray] = []
    for cand in candidates:
        cand = np.clip(cand, v_min, v_max)
        if float(a @ cand - b) >= -1e-7:
            feasible.append(cand)

    if not feasible:
        return np.clip(v0, v_min, v_max), False

    best = min(feasible, key=lambda cand: float(np.sum((cand - v_nom) ** 2)))
    return best, True


def cbf_safety_filter(p: np.ndarray, v_nn: np.ndarray, scenario: dict) -> tuple[np.ndarray, dict]:
    """Apply the one-step CBF-QP safety filter to the learned velocity.

    The QP is

        minimize_v 0.5 ||v - v_NN||^2
        subject to grad h(p)^T v + alpha h(p) >= 0
                   v_min <= v <= v_max.
    """

    p = np.asarray(p, dtype=float)
    v_nn = np.asarray(v_nn, dtype=float)
    h_now = float(safety_function(p, scenario))
    a = safety_gradient(p, scenario)
    b = -float(scenario["alpha"]) * h_now

    v_safe, qp_info = solve_velocity_qp(v_nn, a, b, scenario)
    p_after = point_step(p, v_safe, scenario)
    h_after = float(safety_function(p_after, scenario))
    rho_safe = cbf_residual(p, v_safe, scenario)

    diagnostics = {
        "h_before": h_now,
        "h_after": h_after,
        "intervention_norm": float(np.linalg.norm(v_safe - v_nn)),
        "qp_status": qp_info["status"],
        "qp_success": bool(qp_info["success"]),
        "active_cbf": bool(rho_safe <= 1e-6),
        "cbf_margin": float(rho_safe),
    }
    return v_safe, diagnostics


# ---------------------------------------------------------------------------
# 8. Closed-loop rollouts and metrics
# ---------------------------------------------------------------------------


def simulate_learned_policy(
    policy: dict,
    scenario: dict,
    steps: int = 60,
    use_cbf: bool = False,
) -> dict:
    """Closed-loop rollout for the learned MPC surrogate, optionally with CBF-QP."""

    p = np.zeros((steps + 1, 2))
    v_nn = np.zeros((steps, 2))
    v_apply = np.zeros((steps, 2))
    h = np.zeros(steps + 1)
    intervention = np.zeros(steps)
    active_cbf = np.zeros(steps, dtype=bool)
    qp_status: list[str] = []

    p[0] = scenario["p_start"]
    h[0] = safety_function(p[0], scenario)
    for k in range(steps):
        v_nn[k] = policy_velocity(policy, p[k], scenario)
        if use_cbf:
            v_apply[k], diagnostics = cbf_safety_filter(p[k], v_nn[k], scenario)
            intervention[k] = diagnostics["intervention_norm"]
            active_cbf[k] = diagnostics["active_cbf"]
            qp_status.append(diagnostics["qp_status"])
        else:
            v_apply[k] = np.clip(v_nn[k], scenario["v_min"], scenario["v_max"])
            qp_status.append("not_used")

        p[k + 1] = point_step(p[k], v_apply[k], scenario)
        h[k + 1] = safety_function(p[k + 1], scenario)

    return {
        "p": p,
        "v_nom": v_nn,  # kept for compatibility with existing visuals
        "v_nn": v_nn,
        "v": v_apply,
        "h": h,
        "intervention": intervention,
        "active_cbf": active_cbf,
        "qp_status": np.array(qp_status),
        "goal_error": float(np.linalg.norm(p[-1] - scenario["p_goal"])),
        "min_h": float(np.min(h)),
    }


def simulate_mpc_expert(scenario: dict, steps: int = 60) -> dict:
    """Closed-loop rollout for the MPC expert."""

    solve_mpc = build_mpc_expert_solver(scenario)
    p = np.zeros((steps + 1, 2))
    v = np.zeros((steps, 2))
    h = np.zeros(steps + 1)
    status: list[str] = []

    p[0] = scenario["p_start"]
    h[0] = safety_function(p[0], scenario)
    for k in range(steps):
        v[k], this_status = solve_mpc(p[k])
        status.append(this_status)
        p[k + 1] = point_step(p[k], v[k], scenario)
        h[k + 1] = safety_function(p[k + 1], scenario)

    return {
        "p": p,
        "v": v,
        "h": h,
        "status": np.array(status),
        "goal_error": float(np.linalg.norm(p[-1] - scenario["p_goal"])),
        "min_h": float(np.min(h)),
    }


def rollout_metrics(name: str, rollout: dict) -> dict:
    """Compact metrics table row for one rollout."""

    row = {
        "controller": name,
        "goal_error": float(rollout["goal_error"]),
        "min_h": float(rollout["min_h"]),
    }
    if "intervention" in rollout:
        row["mean_filter_intervention"] = float(np.mean(rollout["intervention"]))
        row["active_cbf_steps"] = int(np.sum(rollout["active_cbf"]))
    return row
