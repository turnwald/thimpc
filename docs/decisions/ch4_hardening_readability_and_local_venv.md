# Chapter 4 Hardening: Readability and Local Venv

The course repository supports three usage modes:

- Recommended for live teaching: WSL/Ubuntu with local Python `venv`.
- Reproducible fallback: VS Code Dev Container.
- Browser fallback: GitHub Codespaces.

The local venv workflow is the day-to-day path because it is fast and reliable
during live teaching. The devcontainer remains the reproducible path because it
pins the operating system base image, Python version, system libraries, Jupyter
setup, and `PYTHONPATH` behavior.

The implementation intentionally prioritizes readability over abstraction. The
labs should show the MPC ingredients directly: dynamics matrices, cost matrices,
input and state bounds, terminal costs, slack variables, solver status, and the
first input applied in receding horizon. Shared helpers are kept small and
explicit, and no general experiment framework or configuration layer is
introduced.

The solution-first workflow is protected by deterministic notebook generation.
Instructor notebooks live as `studies/study_*/walkthrough_solution.ipynb` and
keep solution regions between `SOLUTION_START` and `SOLUTION_END`; student
notebooks are generated as sibling `walkthrough.ipynb` files by replacing those
regions with TODO scaffolding, removing instructor-only markdown, and clearing
code outputs. Tests and CI check for marker leakage and ensure generated
student notebooks are up to date.
