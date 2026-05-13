# thimpc

Teaching repository for the THI Master's course on Model Predictive Control.

This branch contains the student-facing teaching material. It does not include
instructor solution notebooks.

## Teaching Structure

```text
plane_code/
  01_lqr.ipynb
  02_mpc.ipynb
  03_mpc_geometry.ipynb
  04_casadi.ipynb

projects/
  _shared/
  project_1_attitude_constraints/
  project_2_mobile_robot_corridor/
  project_3_learning_enhanced_prediction/
```

`plane_code/` contains linear classroom notebooks for live teaching. The
notebook itself is the teaching document, with visible mathematics and code.

- `plane_code/01_lqr.ipynb` teaches LQR and ends with saturated LQR failure.
- `plane_code/02_mpc.ipynb` teaches MPC first by hand/QP, then introduces
  CasADi at the end as a formulation tool.
- `plane_code/03_mpc_geometry.ipynb` teaches feasibility, terminal sets, and
  the stability picture.
- `plane_code/04_casadi.ipynb` introduces CasADi syntax for compact MPC
  formulations.

`projects/` contains Chapter 4 application projects. Students open and run each
project's `walkthrough.ipynb`; project-local modules keep repeated simulation,
plotting, and solver setup readable without turning the repository into a
framework.

- `projects/project_1_attitude_constraints/`: attitude constraints, saturated
  LQR, constrained MPC, rate limits, infeasibility, terminal geometry, and soft
  constraints.
- `projects/project_2_mobile_robot_corridor/`: mobile robot corridor tracking,
  narrowed passages, model mismatch, soft constraints, and margin intuition.
- `projects/project_3_learning_enhanced_prediction/`: least-squares residual
  learning for improved one-step prediction, with validation and closed-loop
  caveats.

## Environment

Use the local virtual environment when available:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start Jupyter from the repository root so notebooks can import the local
packages:

```bash
PYTHONPATH="$PWD" jupyter lab
```

## Validation

For this student-facing branch, run:

```bash
git status
git branch --show-current
git diff --stat
.venv/bin/python -m compileall projects tools tests
.venv/bin/python -m pytest -q -rs
THIMPC_RELEASE_MODE=public .venv/bin/python -m pytest -q -rs
.venv/bin/python tools/check_student_release.py
```

Notebook execution should write temporary outputs outside the repository, for
example:

```bash
.venv/bin/python -m nbconvert --execute --to notebook --output /tmp/thimpc_01_lqr_executed.ipynb plane_code/01_lqr.ipynb
.venv/bin/python -m nbconvert --execute --to notebook --output /tmp/thimpc_02_mpc_executed.ipynb plane_code/02_mpc.ipynb
```

Never push or commit from automation unless the maintainer asks for it
explicitly.
