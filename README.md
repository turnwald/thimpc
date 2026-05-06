# thimpc

Teaching repository for the THI Master's course on Model Predictive Control.

The repository is organized as a sequence of traceable MPC studies. Each study
is centered on a walkthrough notebook: the notebook is the primary learning
object, and helper modules exist only to keep repeated plotting, simulation, and
solver boilerplate tidy.

## Studies

Student-facing notebooks are named `walkthrough.ipynb`. Instructor branches also
contain `walkthrough_solution.ipynb`, which is stripped into the student version.

```text
studies/
  study_01_plain_python_foundations/
  study_02_lqr_value_iteration_policy_iteration/
  study_03_manual_qp_mpc/
  study_04_casadi_mpc_transition/
  study_05_ch4_attitude_constraints/
  study_06_ch4_mobile_robot_corridor/
  study_07_ch4_learning_residual_mpc/
```

The studies progress from plain Python simulation to LQR and Riccati methods,
manual condensed-QP MPC, a CasADi reformulation of the same MPC problem, and
three Chapter 4 application studies.

The original `notebooks/plane_code/01_open_loop.ipynb` teaching material is
preserved. Its double-integrator examples are integrated into the first four
studies.

## Recommended Live-Teaching Setup: WSL/Ubuntu + Venv

For instructor day-to-day work and live teaching, use a local WSL/Ubuntu Python
virtual environment. This avoids relying on container rebuilds, extension
downloads, or VS Code server setup during a live session.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name thi-mpc --display-name "THI MPC (.venv)"
python -m pytest -q -rs
```

The convenience script does the same setup:

```bash
tools/setup_venv.sh
```

Do not commit `.venv`; it is ignored by `.gitignore`.

Start Jupyter from the repository root so notebooks can import the course
modules:

```bash
source .venv/bin/activate
PYTHONPATH="$PWD" jupyter lab
```

## Reproducible Fallback: VS Code Dev Container

The Dev Container is the reproducible clean environment. It installs the same
`requirements.txt` as local venv setup.

1. Open the repository in VS Code.
2. Choose **Dev Containers: Reopen in Container**.
3. Wait for the image to build from `.devcontainer/Dockerfile`.
4. Open notebooks in VS Code, or use the forwarded JupyterLab port.

The container sets `PYTHONPATH=/workspace`, sets `MPLBACKEND=Agg`, and starts
JupyterLab on port 8888. If you need the JupyterLab token URL, inspect
`/tmp/jupyter.log` inside the container.

## Browser Fallback: GitHub Codespaces

Codespaces uses the same `.devcontainer/` setup. It is intended as a browser
fallback for students without a local setup.

1. Open the repository in GitHub.
2. Create a Codespace.
3. Wait for the container build and post-start setup.
4. Open `studies/.../walkthrough.ipynb`.
5. Run `python -m pytest -q -rs`.
6. Run a reduced scenario or selected notebook cells.

Codespaces should be validated in a fresh Codespace before a public class
release.

## Running Notebooks

Open the relevant `studies/study_*/walkthrough.ipynb` file. Every study follows
the same teaching pattern:

1. Motivation
2. Model
3. Controller or MPC formulation
4. Implementation
5. Simulation
6. Plots
7. Interpretation
8. Student tasks or questions

Important matrices, costs, constraints, solver status, and the first applied
input should remain visible in the notebook.

## Student Notebook Generation

On instructor branches, regenerate student notebooks with:

```bash
python tools/create_student_material.py
```

The generator reads `studies/study_*/walkthrough_solution.ipynb` and writes the
sibling `walkthrough.ipynb`. It removes instructor-only markdown, replaces code
between `SOLUTION_START` and `SOLUTION_END` with TODO scaffolding, clears code
outputs, and rejects private or instructor-only text.

The older `tools/create_student_ch4_material.py` command remains as a wrapper
for compatibility.

## Tests And Validation

Run the lightweight checks:

```bash
python3 -m compileall controllers mpc systems scenarios tools tests
python3 -m pytest -q -rs
python3 tools/create_student_material.py
git diff --exit-code studies
```

On a public/student branch, also run:

```bash
THIMPC_RELEASE_MODE=public python3 -m pytest -q -rs
python3 tools/check_student_release.py
```

`tools/check_student_release.py` is expected to fail on instructor branches
because solution notebooks are present.

## Scenario Scripts

Chapter 4 studies use scenario scripts for repeatable plots and metrics:

```bash
python -m scenarios.ch4_project1_attitude
python -m scenarios.ch4_project2_mobile_robot
python -m scenarios.ch4_project3_learning
```

Reduced smoke runs:

```bash
python -m scenarios.ch4_project1_attitude --steps 40 --output-dir outputs/ch4_project1_smoke
python -m scenarios.ch4_project2_mobile_robot --steps 60 --output-dir outputs/ch4_project2_smoke
python -m scenarios.ch4_project3_learning --samples 100 --output-dir outputs/ch4_project3_smoke
```

Generated outputs are ignored by git unless selected examples are deliberately
curated for teaching.

## Branch Policy

- `main` is the student-facing public branch and should be pushed only to the
  `public` remote when explicitly intended.
- `solutions` is the temporary instructor solution branch.
- `private` is the final private instructor branch.
- `ch4-solution-first-labs` is the current temporary work branch for Chapter 4
  solution-first labs and cleanup.

Never push automatically. Check `git status --short` and `git remote -v` before
switching branches or preparing a release.

## Public Release Checklist

On `main` only:

```bash
git status --short
python3 tools/create_student_material.py
git diff --exit-code studies
THIMPC_RELEASE_MODE=public python3 -m pytest -q -rs
python3 tools/check_student_release.py
find studies -name '*solution*.ipynb'
rg -n "SOLUTION_START|SOLUTION_END|Instructor solution|private note|instructor-only" studies
```

Expected result: no solution notebooks, no forbidden markers, no hidden outputs,
and runnable student study notebooks.

## Troubleshooting

- If `python` is not found on Linux, WSL, or macOS, use `python3` to create the
  venv. After activation, `python` should point to `.venv`.
- If CasADi or OSQP installation fails locally, use the Dev Container for clean
  validation.
- If notebooks cannot import `systems`, `controllers`, `mpc`, or `scenarios`,
  start Jupyter from the repository root with `PYTHONPATH` set as shown above.
- If port 8888 is already in use, stop the existing Jupyter process or inspect
  `/tmp/jupyter.log` for the URL printed by Jupyter.
