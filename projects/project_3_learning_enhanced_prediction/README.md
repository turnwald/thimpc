# Project 3 - Learning-Enhanced Prediction

This project fits a least-squares residual model for a mobile robot tracking
error model. The residual model improves one-step prediction on held-out data,
but it does not replace constraints, validation, or closed-loop checks.

Open `walkthrough.ipynb` from the repository root and run the cells in order.
The main outputs are a measured-versus-predicted lateral-error plot, a residual
error plot, and RMSE metrics.

Good student modifications:

- Change the amount of training data.
- Change the random seed.
- Compare nominal and learned prediction errors state by state.
- Ask what would still need to be checked before using the learned model in MPC.

