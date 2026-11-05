from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import physics_model as pm

RESULT_COLUMNS = ("time_t", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
                  "kinetic_energy", "potential_energy", "total_energy",
                  "bending_angle_x", "bending_angle_y")

@dataclass
class SolverResult:
    table: pd.DataFrame
    outcome: str
    final_state: dict
    exits: dict = field(default_factory=dict)
    method: str = ""
    n_rhs_calls: int = 0

def edge_event(z_end):
    def event(t, s):
        return s[2] - z_end
    event.terminal = True
    event.direction = 1.0
    return event

def plate_event(index, half_gap):
    def event(t, s):
        return half_gap - abs(s[index])
    event.terminal = True
    event.direction = -1.0
    return event

def solve(bl, sim, solver):
    s = np.array([bl.x0, bl.y0, bl.z_start, bl.vx0, bl.vy0, bl.v0, 0.0])
    t = 0.0
    pieces, exits, n_calls = [], {}, 0
    outcome = "timeout"

    for index, (name, z_end, ex, ey) in enumerate(bl.segments()):
        events = [edge_event(z_end)]
        if index == pm.X_PLATES:
            events.append(plate_event(0, bl.dx / 2.0))
        if index == pm.Y_PLATES:
            events.append(plate_event(1, bl.dy / 2.0))

        sol = solve_ivp(pm.make_rhs(bl, ex, ey), (t, sim["t_max"]), s,
                        method=solver["method"], rtol=solver["rtol"], atol=solver["atol"],
                        events=events, dense_output=True)
        n_calls += sol.nfev
        pieces.append(sol)
        if sol.status != 1:
            outcome = "timeout"
            t, s = sol.t[-1], sol.y[:, -1]
            break

        fired = [i for i, te in enumerate(sol.t_events) if te.size]
        t, s = sol.t[-1], sol.y[:, -1]
        if len(events) > 1 and 1 in fired:
            outcome = "plate_x" if index == pm.X_PLATES else "plate_y"
            break
        if index == pm.X_PLATES:
            exits["x"] = {"z": s[2], "pos": s[0], "theta": float(np.arctan2(s[3], s[5]))}
        if index == pm.Y_PLATES:
            exits["y"] = {"z": s[2], "pos": s[1], "theta": float(np.arctan2(s[4], s[5]))}
        if index == len(bl.segments()) - 1:
            outcome = "wafer"

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

    kinetic = pm.kinetic_energy(bl, vx, vy, vz)
    potential = pm.potential_energy(bl, w)
    bending_x, bending_y = pm.bending_angle(vx, vy, vz)
    table = pd.DataFrame({
        "time_t": grid, "pos_x": x, "pos_y": y, "pos_z": z,
        "vel_x": vx, "vel_y": vy, "vel_z": vz,
        "kinetic_energy": kinetic, "potential_energy": potential,
        "total_energy": kinetic + potential,
        "bending_angle_x": bending_x, "bending_angle_y": bending_y}, columns=RESULT_COLUMNS).astype("float64")

    final = {"time_t": float(t_end), "x": float(x[-1]), "y": float(y[-1]), "z": float(z[-1]),
             "vx": float(vx[-1]), "vy": float(vy[-1]), "vz": float(vz[-1])}
    return SolverResult(table, outcome, final, exits, solver["method"], n_calls)