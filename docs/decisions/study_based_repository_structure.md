# Study-Based Repository Structure

The repository is organized around walkthrough studies rather than isolated
scripts. Each study has a local notebook pair:

- `walkthrough_solution.ipynb` on instructor branches.
- `walkthrough.ipynb` as the generated student notebook.

This keeps the course sequence visible in the file tree and makes each study a
self-contained learning object with motivation, model, formulation,
implementation, simulation, plots, interpretation, and student tasks.

Root-level `controllers/`, `systems/`, `mpc/`, and `scenarios/` remain because
flat imports are easier for students than a package-heavy `src/` layout. Shared
helpers are allowed for repeated plotting, simulation, and solver boilerplate,
but they should not hide models, costs, constraints, solver status, or the
first input applied in receding horizon.

The original `notebooks/plane_code/01_open_loop.ipynb` remains preserved as
teaching material. Its double-integrator examples seed the early studies:
plain-Python simulation, LQR/value-function interpretation, manual QP MPC, and
the CasADi transition.
