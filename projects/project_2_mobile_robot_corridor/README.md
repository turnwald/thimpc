# Project 2 - Mobile Robot Corridor

This project studies local tracking-error MPC for a mobile robot following a
circular path through a corridor with a narrowed passage centered at 90 degrees
around the circle.

Open `walkthrough.ipynb` from the repository root and run the cells in order.
The main claim is visual: nominal tracking drifts into the corridor boundary,
while MPC predicts the narrowed passage and keeps the relevant margin positive
in the nominal scenario.

The most useful outputs are:

- top-view trajectory with obstacle, wall, and narrowed corridor;
- lateral-error corridor plot;
- yaw-rate correction plot with limits;
- margin plot;
- metrics comparing minimum margin, violation, input use, and tracking error.

Good student modifications:

- Change the initial lateral or heading error in `config.py`.
- Adjust the corridor width or robust margin.
- Compare short and long horizons.
- Add model mismatch and inspect the margin loss.
