# thimpc

Teaching repository for the THI Master's course on Model Predictive Control.

This public version is intentionally narrow. The main learning objects are the
Jupyter notebooks in `plane_code/` and `projects/`, and the recommended run path
is the Dev Container.

## Structure

```text
plane_code/
  01_lqr.ipynb
  02_mpc.ipynb
  03_mpc_geometry.ipynb
  04_casadi.ipynb

projects/
  project_1_attitude_constraints/
    walkthrough.ipynb
    attitude_plotting.py
  project_2_mobile_robot_corridor/
    walkthrough.ipynb
    mobile_robot_helpers.py
  project_3_learning_safety_filter/
    walkthrough.ipynb
    project3_core.py
    project3_visuals.py
    assets/
```

`plane_code/` contains the lecture companion notebooks.

`projects/` contains the Chapter 4 application projects. Project-local helper
files are kept next to the notebooks so students can inspect the model,
constraints, controller logic, simulation loop, and plotting code without
navigating a larger framework.

Project 3 commits small pretrained assets under
`projects/project_3_learning_safety_filter/assets/`. Projects 1 and 2 do not
need committed assets.

## Dev Container

Open the repository in VS Code and choose:

```text
Dev Containers: Reopen in Container
```

The container installs `requirements.txt`, sets the workspace to `/workspace`,
and starts JupyterLab on port 8888 via `.devcontainer/start.sh`.

Inside the container, open notebooks directly from:

```text
plane_code/
projects/
```

## Generated Outputs

Generated plots, HTML files, animations, and regenerated datasets belong under:

```text
outputs/
```

`outputs/` is ignored by Git and by the Docker build context. Existing local
outputs may be kept on disk, but they are not part of the committed teaching
source unless deliberately promoted into a project-local `assets/` folder.

## Local Fallback

The Dev Container is preferred, but a local Python environment can be used:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name thi-mpc --display-name "THI MPC (.venv)"
jupyter lab
```

Start Jupyter from the repository root so relative project paths resolve as
shown in the notebooks.

## Quick Validation

From the repository root:

```bash
python -m compileall plane_code projects
```

For a fuller check, run the notebooks in the Dev Container.
