"""Numerical solver: integrates the equations of motion with scipy.integrate.solve_ivp.

The electric field switches on and off abruptly at the plate edges, so the flight
is integrated region by region (tube, x plates, gap, y plates, drift).  Inside a
region the field is constant and the right-hand side is smooth.  Terminal events
stop the integration exactly when

* the ion reaches the end of a region (z crosses the region boundary),
* the ion touches an x plate or a y plate  (|x| = dx/2 or |y| = d/2),
* the ion arrives at the wafer (end of the last region).

The method (for example 'DOP853' or 'RK45') and the tolerances come from the
input file and are printed by main.py.  The dense output of every region is
sampled on a regular time grid (simulation.dt) so the results form one tidy table.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import physics_model as pm

# column order of the output table (the schema of the guideline)
RESULT_COLUMNS = ("time_t", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
                  "kinetic_energy", "potential_energy", "total_energy")


@dataclass
class SolverResult:
    table: pd.DataFrame            # simulation_results.csv (guideline schema)
    conserved: pd.DataFrame        # extra conserved quantities (p_z, angular momentum)
    outcome: str                   # 'wafer', 'plate_x', 'plate_y' or 'timeout'
    final_state: dict              # time and position/velocity where the flight ended
    exits: dict = field(default_factory=dict)     # angle and position when leaving each plate pair
    method: str = ""
    n_rhs_calls: int = 0


def _edge_event(z_end):
    def event(t, s):
        return s[2] - z_end
    event.terminal = True
    event.direction = 1.0
    return event


def _plate_event(index, half_gap):
    def event(t, s):
        return half_gap - abs(s[index])
    event.terminal = True
    event.direction = -1.0
    return event


def solve(bl, sim, solver):
    """Integrate one ion through the beam line.

    bl      physics_model.BeamLine
    sim     {'dt': output sampling interval (s), 't_max': safety time limit (s)}
    solver  {'method': ..., 'rtol': ..., 'atol': ...}
    """
    s = np.array([bl.x0, bl.y0, bl.z_start, bl.vx0, bl.vy0, bl.v0, 0.0])
    t = 0.0
    pieces, exits, n_calls = [], {}, 0
    outcome = "timeout"

    for index, (name, z_end, ex, ey) in enumerate(bl.segments()):
        events = [_edge_event(z_end)]
        if index == pm.X_PLATES:
            events.append(_plate_event(0, bl.dx / 2.0))
        if index == pm.Y_PLATES:
            events.append(_plate_event(1, bl.d / 2.0))

        sol = solve_ivp(pm.make_rhs(bl, ex, ey), (t, sim["t_max"]), s,
                        method=solver["method"], rtol=solver["rtol"], atol=solver["atol"],
                        events=events, dense_output=True)
        n_calls += sol.nfev
        pieces.append(sol)
        if sol.status != 1:                       # t_max reached before the region ended
            outcome = "timeout"
            t, s = sol.t[-1], sol.y[:, -1]
            break

        fired = [i for i, te in enumerate(sol.t_events) if te.size]
        t, s = sol.t[-1], sol.y[:, -1]
        if len(events) > 1 and 1 in fired:        # a plate event fired first
            outcome = "plate_x" if index == pm.X_PLATES else "plate_y"
            break
        if index == pm.X_PLATES:
            exits["x"] = {"z": s[2], "pos": s[0], "theta": float(np.arctan2(s[3], s[5]))}
        if index == pm.Y_PLATES:
            exits["y"] = {"z": s[2], "pos": s[1], "theta": float(np.arctan2(s[4], s[5]))}
        if index == len(bl.segments()) - 1:
            outcome = "wafer"

    # ---- sample the dense output on a regular time grid (vectorised) ---------
    t_end = t
    grid = np.arange(0.0, t_end, sim["dt"])
    if grid.size == 0 or grid[-1] < t_end:
        grid = np.append(grid, t_end)
    piece_end = np.array([p.t[-1] for p in pieces])
    owner = np.minimum(np.searchsorted(piece_end, grid, side="left"), len(pieces) - 1)
    states = np.empty((7, grid.size))
    for i, piece in enumerate(pieces):
        mask = owner == i
        if mask.any():
            states[:, mask] = piece.sol(grid[mask])
    x, y, z, vx, vy, vz, w = states

    # ---- energies and conserved quantities ------------------------------------
    kinetic = pm.kinetic_energy(bl, vx, vy, vz)
    potential = pm.potential_energy(bl, w)
    table = pd.DataFrame({
        "time_t": grid, "pos_x": x, "pos_y": y, "pos_z": z,
        "vel_x": vx, "vel_y": vy, "vel_z": vz,
        "kinetic_energy": kinetic, "potential_energy": potential,
        "total_energy": kinetic + potential}, columns=RESULT_COLUMNS).astype("float64")

    ang = pm.angular_momentum(bl, x, y, z, vx, vy, vz)
    p_z = bl.m * vz
    conserved = pd.DataFrame({
        "time_t": grid, "momentum_z": p_z,
        "momentum_z_rel_error": np.abs((p_z - p_z[0]) / p_z[0]),
        "angular_momentum_x": ang[:, 0], "angular_momentum_y": ang[:, 1],
        "angular_momentum_z": ang[:, 2]}).astype("float64")

    final = {"time_t": float(t_end), "x": float(x[-1]), "y": float(y[-1]), "z": float(z[-1]),
             "vx": float(vx[-1]), "vy": float(vy[-1]), "vz": float(vz[-1])}
    return SolverResult(table, conserved, outcome, final, exits, solver["method"], n_calls)
