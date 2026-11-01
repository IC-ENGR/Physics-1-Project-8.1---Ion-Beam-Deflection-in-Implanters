"""Ion beam deflection in an implanter - entry point.

Usage (from the project folder):
    python main.py                       # run with data/inputs/initial_state.json
    python main.py --config my_case.json # run with another input file
    python main.py --interactive         # also open the animated replay window

Pipeline:  input file -> validation (config.py) -> physical model (physics_model.py)
           -> numerical solution (solvers.py) -> CSV tables + figures (visualization.py)
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np

import physics_model as pm
import solvers
import visualization as viz
from config import ConfigError, load_config

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "data" / "inputs" / "initial_state.json"


def print_summary(bl, result, theory, solver_cfg):
    f = result.final_state
    energy = result.table["total_energy"]
    energy_error = float(np.max(np.abs((energy - energy.iloc[0]) / energy.iloc[0])))
    print(f"Solver            : scipy.integrate.solve_ivp, method='{result.method}', "
          f"rtol={solver_cfg['rtol']:g}, atol={solver_cfg['atol']:g}, {result.n_rhs_calls} function calls")
    print(f"Ion               : q = {bl.q / bl.e_charge:g} e, m = {bl.m / bl.amu:g} u, "
          f"v0 = {bl.v0:.4e} m/s, flight time (theory) = {theory['t_flight'] * 1e6:.4f} us")
    print(f"Outcome           : {result.outcome}   (end of flight at t = {f['time_t'] * 1e6:.4f} us, "
          f"z = {f['z']:.4f} m)")
    for axis, label in (("x", "x plates"), ("y", "y plates")):
        if axis in result.exits:
            print(f"Exit of {label:<9} : theta = {math.degrees(result.exits[axis]['theta']):+.4f} deg "
                  f"(theory {theory['theta_' + axis + '_deg']:+.4f} deg)")
    if result.outcome == "wafer":
        r = math.hypot(f["x"], f["y"])
        print(f"Impact on wafer   : x = {f['x'] * 1e3:+.3f} mm, y = {f['y'] * 1e3:+.3f} mm, "
              f"r = {r * 1e3:.3f} mm -> {'ON' if r <= bl.wafer_R else 'OFF'} the wafer")
        print(f"Theory            : x = {theory['x_wafer'] * 1e3:+.3f} mm, y = {theory['y_wafer'] * 1e3:+.3f} mm")
    elif result.outcome in ("plate_x", "plate_y"):
        limit = theory["Vx_limit"] if result.outcome == "plate_x" else theory["Vd_limit"]
        print(f"The ion hit a plate; the voltage limit for passing is about {limit:.0f} V.")
    print(f"Energy check      : max |(E - E0)/E0| = {energy_error:.3e}")
    print(f"Axial momentum    : max relative change = {result.conserved['momentum_z_rel_error'].max():.3e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ion beam deflection simulator")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON input file")
    parser.add_argument("--interactive", action="store_true", help="open the animated replay window")
    args = parser.parse_args(argv)

    # ---- input and validation ------------------------------------------------
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = load_config(args.config)
        for item in caught:
            print(f"WARNING: {item.message}", file=sys.stderr)
    except ConfigError as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 1

    np.random.seed(cfg["random_seed"])          # reproducibility (no random numbers are used here)
    bl = pm.BeamLine.from_config(cfg)
    print(f"System            : {cfg['system']}")

    # ---- solve -----------------------------------------------------------------
    result = solvers.solve(bl, cfg["simulation"], cfg["solver"])
    print_summary(bl, result, pm.theory(bl), cfg["solver"])

    # ---- output ------------------------------------------------------------------
    out = cfg["output"]
    csv_path, conserved_path = viz.save_tables(result, ROOT / out["results_dir"],
                                               out["results_csv"], out["conserved_csv"])
    figures = viz.save_all_plots(result, bl, ROOT / out["figures_dir"], out["dpi"])
    print(f"Saved             : {csv_path.relative_to(ROOT)}, {conserved_path.relative_to(ROOT)}")
    for path in figures:
        print(f"Saved             : {path.relative_to(ROOT)}")

    if args.interactive:
        viz.run_interactive(result, bl, cfg["viewer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
