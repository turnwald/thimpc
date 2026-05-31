# Project 3 — Learning an MPC Expert with a Robust CBF Safety Filter

This project demonstrates a compact advanced MPC pipeline:

1. an MPC expert generates safe end-effector velocities,
2. a small NumPy neural policy learns an approximation of the MPC input map,
3. a CBF-QP safety filter corrects the learned velocity online under bounded execution mismatch.

The project is a teaching demonstration, not a deep-learning or nonlinear-robotics project.

## Model and visual interpretation

The control, learning, and CBF computations use the transparent end-effector point model

\[
p_{k+1}=p_k+T_s v_k.
\]

All spatial plots and animations are shown as a planar two-link arm. The arm is a visual replay layer: each end-effector point \(p_k\) is mapped to a feasible arm configuration \(q_k\). The arm is not the plant used by the MPC or the learned policy.

## What is learned?

The neural policy does not learn an independent hand-designed controller. It learns a fast approximation of the MPC input map:

\[
\pi_\theta(p,s) \approx v^\star_{\mathrm{MPC}}(p,s),
\]

where \(s\) contains the scenario data, such as goal, obstacle, safety radius, and bounds.

## Safety filter

The CBF-QP solves

\[
\min_v \|v-v_{\mathrm{NN}}\|^2
\]

subject to a hard CBF constraint and velocity bounds. For the nominal point model,

\[
\nabla h(p)^T v + \alpha h(p) \ge 0.
\]

For the robust disturbance demo, execution is modeled as

\[
p_{k+1}=p_k+T_s(v_k+d_k),\qquad \|d_k\|\le \bar d,
\]

and the filter enforces the conservative residual

\[
\rho_{\mathrm{rob}}(p,v)
=\nabla h(p)^T v+\alpha h(p)-\|\nabla h(p)\|\bar d\ge 0.
\]

This is the central teaching point: the state can still satisfy \(h(p)>0\), while the command violates the robust CBF residual. The filter then acts preventively, before the trajectory has to collide.

## Files

Expected project folder:

```text
projects/project_3_learning_safety_filter/
  walkthrough.ipynb
  README.md
  project3_core.py
  project3_visuals.py
  assets/
    expert_dataset_small.npz
    policy_pretrained.npz
    training_history_pretrained.csv
    training_rollout_snapshots.npz
```

Generated outputs are written to:

```text
outputs/project_3_learning_safety_filter/
```

Do not commit generated outputs.

## Default run

Open `walkthrough.ipynb` and run all cells with the default switches:

```python
TRAIN_MODEL = False
GENERATE_EXPERT_DATA = False
CREATE_ANIMATIONS = False
```

This loads the committed pretrained NumPy policy and stored training history.

## Optional retraining

Set:

```python
TRAIN_MODEL = True
```

The notebook retrains the small NumPy MLP and writes the new training history, snapshots, and policy to `outputs/project_3_learning_safety_filter/`. It does not overwrite committed assets.

## Optional expert data generation

Set:

```python
GENERATE_EXPERT_DATA = True
```

This regenerates a small expert dataset using the CasADi MPC expert and writes it to `outputs/project_3_learning_safety_filter/`.

## Optional animations

Set:

```python
CREATE_ANIMATIONS = True
```

This creates GIF files in `outputs/project_3_learning_safety_filter/`:

- training snapshot replay as planar-arm panels,
- disturbed learned policy versus robust CBF-filtered policy.

## Dependencies

The project uses the course stack plus lightweight plotting/data helpers:

- numpy
- scipy
- matplotlib
- casadi
- pandas, optional for tabular inspection
- pillow / imageio, optional for GIF/animation output

No PyTorch, TensorFlow, JAX, or heavy learning framework is required. The neural policy is implemented directly in NumPy.

## Teaching interpretation

The intended progression is:

1. MPC can generate high-quality constrained behavior.
2. A learned surrogate can imitate the MPC input map and run cheaply.
3. A learned surrogate is not a hard safety certificate under execution mismatch.
4. The CBF-QP monitors the measured state and enforces a hard online safety condition.
5. The planar arm replay makes the otherwise simple end-effector model visually meaningful.
