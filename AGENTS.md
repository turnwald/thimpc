# Agent Rules For THI MPC Teaching Repository

This is a student-facing Master's course repository for Model Predictive Control. Prefer readable teaching material over research-framework architecture.

## Branch Policy

- `main` is the public student-facing branch.
- `solutions` is the temporary instructor solution branch.
- `private` is the final private/instructor branch.
- Temporary feature branches should be deleted after integration.

`main` must not contain instructor-only solutions, solution markers, hidden solution outputs, private notes, or solution notebooks.

## Safety Rules

- Always run `git status --short` before editing, committing, or switching branches.
- Do not create branches unless explicitly requested.
- Stop and report if the current branch is not the expected branch for the task.
- Never switch branches with uncommitted unrelated changes.
- Never commit instructor solutions to `main`.
- Never push without explicit instructor instruction.
- Never delete branches unless they are fully merged and explicitly approved.
- Always generate student notebooks before preparing `main`.
- Always run solution-leakage checks before touching or committing `main`.
- On `main`, use `THIMPC_RELEASE_MODE=public python -m pytest -q` and `python tools/check_student_release.py`.

## Teaching Code Rules

- Prefer explicit, readable code over clever abstractions.
- Keep models, costs, constraints, solver status, simulation loops, and plots visible.
- Avoid deep inheritance, controller factories, generic frameworks, and complicated config systems.
- Use explicit matrices and lecture notation where possible.
- Keep notebooks understandable for students.

## Environment Rules

- Dev Container is the recommended reproducible mode.
- WSL/Ubuntu with local `venv` is the supported lightweight mode.
- GitHub Codespaces is supported only after structural and functional validation.
- `requirements.txt` must support local venv use.
- `.devcontainer/` must support VS Code and Codespaces.

## Solution-First Workflow

- Instructor solution notebooks live as `plane_code/*_solution.ipynb` and `projects/project_*/walkthrough_solution.ipynb` only on `solutions` and `private`.
- Student notebooks are generated/stripped into sibling `plane_code/*.ipynb` and `projects/project_*/walkthrough.ipynb` notebooks.
- Use `python tools/create_student_material.py` to regenerate all student-facing notebooks.
- Generated student notebooks must not contain instructor outputs.
- Tests must verify no `SOLUTION_START`, `SOLUTION_END`, `Instructor solution`, private notes, or hidden solution outputs leak into student material.
