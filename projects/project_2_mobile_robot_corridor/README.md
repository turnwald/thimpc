# Project 2 - Mobile Robot Corridor

This project studies local tracking-error MPC for a mobile robot following a
circular reference through a corridor with a smooth narrowed passage centered at
90 degrees around the circle. In the critical passage, the admissible lateral
error interval is shifted away from zero, so the desired reference path itself is
not inside the allowed state set.

Open `walkthrough.ipynb` from the repository root and run the cells in order.
The main claim is visual: a simple baseline tracker follows the reference into a
constraint conflict, saturated LQR improves the local response but remains
short-sighted, and MPC predicts the shifted narrow passage. The notebook also
keeps a hard-MPC diagnostic to show why abrupt or already-violated state
constraints can produce real solver infeasibility.

The most useful outputs are:

- top-view trajectory with obstacle, wall, and narrowed corridor;
- lateral-error corridor plot;
- yaw-rate correction plot with limits;
- margin plot;
- animated baseline / saturated LQR / MPC replay;
- metrics comparing minimum margin, violation, input use, and tracking error.

Good student modifications:

- Change the initial lateral or heading error in `config.py`.
- Adjust the corridor width or robust margin.
- Compare short and long horizons.
- Add model mismatch and inspect the margin loss.
