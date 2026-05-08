# Project 1 - Attitude Constraints

This project studies constrained attitude control for a discrete double
integrator. The walkthrough compares LQR, saturated LQR, constrained MPC,
rate-limited MPC, infeasibility, terminal geometry, and soft constraints.

Open `walkthrough.ipynb` from the repository root and run the cells in order.
The main observations are the time histories, phase-plane trajectories,
terminal ellipses, and constraint-activity plot.

Good student modifications:

- Change the initial attitude in `config.py`.
- Tighten or relax the torque and angle bounds.
- Change the MPC horizon.
- Compare hard constraints with soft angle constraints.

