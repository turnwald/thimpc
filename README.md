# thimpc

Teaching repository for the THI Master's course on Model Predictive Control.

## Teaching Structure

```text
plane_code/
  01_lqr.ipynb
  02_mpc.ipynb
  03_mpc_geometry.ipynb

projects/
  project_1_attitude_constraints/
  project_2_mobile_robot_corridor/
  project_3_learning_enhanced_prediction/

studies/
  study_01_plain_python_foundations/
  ...
```

`plane_code/` contains compact live-teaching notebooks for LQR, MPC, feasibility,
terminal sets, and stability intuition.

`projects/` contains the Chapter 4 application projects. Open each project's
`walkthrough.ipynb` from the repository root and run the cells in order.

`studies/` contains the earlier study-based notebooks. These are retained during
the migration to the project layout.

## Environment

Use a local virtual environment or the VS Code Dev Container.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start Jupyter from the repository root so notebooks can import the local modules:

```bash
PYTHONPATH="$PWD" jupyter lab
```

## Validation

```bash
python -m compileall controllers mpc projects scenarios systems tools tests
python -m pytest -q -rs
python tools/check_student_release.py
```

Generated plots and notebook execution outputs should be written outside the
repository or under ignored output directories.
