# Chapter 4 Application Studies: Solution-First Materials

Chapter 4 is a project-based application chapter. Students have already seen
state-space models, LQR, constrained MPC, terminal costs, receding-horizon
implementation, tuning, and CasADi. The chapter now answers: if they want to
use MPC in their next project, how should they start a first application study?

The repository provides infrastructure: container setup, notebooks, scenario
scripts, simulators, plotting, baseline controllers, and reusable MPC helpers.
Student work should stay focused on MPC design and interpretation: models,
costs, constraints, horizons, terminal costs, slack variables, and closed-loop
results.

Solutions are created first. Instructor-only solution notebooks live as
`studies/study_*/walkthrough_solution.ipynb` and use
`SOLUTION_START` and `SOLUTION_END` code markers plus `Instructor solution`
markdown. Student-facing notebooks are generated from those solution notebooks
with `tools/create_student_material.py`.

The three Chapter 4 studies are:

- Project 1: constrained attitude control with a double integrator.
- Project 2: mobile robot local tracking in a safety corridor.
- Project 3: learning-enhanced prediction through least-squares residuals.

The implementation intentionally avoids a large framework. The CasADi MPC
builder keeps dynamics, stage costs, terminal costs, constraints, slack
variables, solver status, and first-input application visible.
