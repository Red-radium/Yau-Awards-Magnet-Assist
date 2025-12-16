dx = 1e-5  # initial perturbation size (m)
Dx = 0.01  # spacing of different initial x0 positions (m)
N = 30     # number of initial x0 samples
MAX_TIME = 10.0

import numpy as np
import matplotlib.pyplot as plt
# from dipole_one_terry import *
from dipole_terry import Experiment

e = Experiment()

def lyapunov_for_x0(x0, perturbation=dx, max_time=MAX_TIME):
    """Simulate two trajectories separated by 'perturbation' in x and compute λ(t)."""
    base_t, base_pos = e.simulation(x=x0, max_time=max_time, respect_termination=False)
    pert_t, pert_pos = e.simulation(x=x0 + perturbation, max_time=max_time, respect_termination=False)

    base_arr = np.array(base_pos)
    pert_arr = np.array(pert_pos)
    n = min(len(base_arr), len(pert_arr))
    if n == 0:
        return np.array([]), np.array([])

    base_arr = base_arr[:n]
    pert_arr = pert_arr[:n]
    times = np.array(base_t[:n])

    separation = np.linalg.norm(pert_arr - base_arr, axis=1)
    separation[separation == 0] = np.finfo(float).tiny

    lambdas = np.log(separation / separation[0]) / times
    if len(lambdas) > 1:
        lambdas[0] = lambdas[1]  # avoid divide-by-zero at t=0
    else:
        lambdas[0] = 0.0
    return times, lambdas


X0 = np.arange(0, Dx * N, Dx)
time_axis = None
lambda_rows = []
valid_x0 = []

for x0 in X0:
    t_vals, lambdas = lyapunov_for_x0(x0)
    if len(lambdas) == 0:
        continue
    valid_x0.append(x0)
    lambda_rows.append(lambdas)
    if time_axis is None or len(t_vals) < len(time_axis):
        time_axis = t_vals

if not lambda_rows:
    raise RuntimeError("No trajectories produced data for Lyapunov calculation.")

min_len = min(len(row) for row in lambda_rows)
time_axis = time_axis[:min_len]
lambda_grid = np.stack([row[:min_len] for row in lambda_rows], axis=1)

X_mesh, T_mesh = np.meshgrid(valid_x0, time_axis)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("$X_0$ (m)")
ax.set_ylabel("Time (s)")
ax.set_zlabel("Lyapunov Exponent")
surf = ax.plot_surface(X_mesh, T_mesh, lambda_grid, cmap="viridis", alpha=0.8)
# Reference plane λ = 0 to highlight stability threshold
zero_plane = np.zeros_like(lambda_grid)
ax.plot_surface(X_mesh, T_mesh, zero_plane, color="gray", alpha=0.3, linewidth=0, antialiased=False)
cbar = fig.colorbar(surf, ax=ax, 
                   shrink=0.6,  # Adjusts height (0-1)
                   aspect=15,   # Makes bar thinner
                   pad=0.1)
plt.show()
