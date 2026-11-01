"""Input parsing and validation.

Reads the JSON configuration file and checks that every required value is
present, is a finite number of the right type, and is physically sensible
(no negative mass, no negative voltage source V0, no ion starting inside a
plate, ...).  There is NO physics in this module and NO default value: every
number comes from the JSON file.
"""

import json
import math
import warnings
from pathlib import Path


class ConfigError(ValueError):
    """Raised when the input file is missing, malformed or non-physical."""


# section -> required keys.  Any key not listed here is rejected (catches typos).
REQUIRED = {
    "constants": ("elementary_charge", "atomic_mass_unit", "speed_of_light"),
    "parameters": ("V0_kV", "Vd", "l", "d", "Vx", "lx", "dx", "gap_xy", "L",
                   "wafer_mm", "q_e", "m_u", "L_from_centre"),
    "geometry": ("tube_len", "tube_gap"),
    "initial_conditions": ("x0", "y0", "vx0", "vy0"),
    "simulation": ("dt", "t_max"),
    "solver": ("method", "rtol", "atol"),
    "output": ("results_dir", "figures_dir", "results_csv", "conserved_csv", "dpi"),
    "viewer": ("frame_ms", "frames_per_flight"),
}
TOP_LEVEL = ("system", "random_seed") + tuple(REQUIRED)
STRING_KEYS = {"solver": ("method",),
               "output": ("results_dir", "figures_dir", "results_csv", "conserved_csv")}
BOOL_KEYS = {"parameters": ("L_from_centre",)}
SOLVER_METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_structure(cfg):
    if not isinstance(cfg, dict):
        raise ConfigError("The top level of the input file must be a JSON object { ... }.")
    unknown = set(cfg) - set(TOP_LEVEL)
    if unknown:
        raise ConfigError(f"Unknown top-level entries: {sorted(unknown)}")
    if not isinstance(cfg.get("random_seed"), int) or isinstance(cfg.get("random_seed"), bool):
        raise ConfigError("'random_seed' must be an integer.")
    for section, keys in REQUIRED.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ConfigError(f"Missing section '{section}'.")
        missing = [k for k in keys if k not in cfg[section]]
        extra = [k for k in cfg[section] if k not in keys]
        if missing:
            raise ConfigError(f"Section '{section}' is missing: {missing}")
        if extra:
            raise ConfigError(f"Section '{section}' has unknown entries: {extra}")
        for key in keys:
            value = cfg[section][key]
            if key in STRING_KEYS.get(section, ()):
                if not isinstance(value, str) or not value:
                    raise ConfigError(f"'{section}.{key}' must be a non-empty string.")
            elif key in BOOL_KEYS.get(section, ()):
                if not isinstance(value, bool):
                    raise ConfigError(f"'{section}.{key}' must be true or false (lower case).")
            elif not _is_number(value):
                raise ConfigError(f"'{section}.{key}' must be a finite number, got {value!r}.")


def _require(condition, message):
    if not condition:
        raise ConfigError(message)


def _check_physics(cfg):
    c, p, g = cfg["constants"], cfg["parameters"], cfg["geometry"]
    ic, sim, sol, out, view = (cfg["initial_conditions"], cfg["simulation"], cfg["solver"],
                               cfg["output"], cfg["viewer"])

    for key in ("elementary_charge", "atomic_mass_unit", "speed_of_light"):
        _require(c[key] > 0, f"constants.{key} must be positive.")

    _require(p["V0_kV"] > 0, "parameters.V0_kV (accelerating voltage) must be positive.")
    _require(p["m_u"] > 0, "parameters.m_u (ion mass) must be positive: negative mass is non-physical.")
    _require(p["q_e"] > 0, "parameters.q_e (ion charge) must be positive: this model describes positive ions.")
    for key in ("l", "d", "lx", "dx", "L", "wafer_mm"):
        _require(p[key] > 0, f"parameters.{key} must be positive.")
    _require(p["gap_xy"] >= 0, "parameters.gap_xy must not be negative.")
    for key in ("tube_len", "tube_gap"):
        _require(g[key] > 0, f"geometry.{key} must be positive.")

    # the wafer plane must lie behind the y plates
    z_wafer = p["L"] + (p["l"] / 2.0 if p["L_from_centre"] else 0.0)
    _require(z_wafer > p["l"], "The wafer must be behind the y plates: increase parameters.L "
                               "(L must be larger than l, or larger than l/2 when L_from_centre is true).")

    # the ion must start inside the gaps of both plate pairs
    _require(abs(ic["x0"]) < p["dx"] / 2, "initial_conditions.x0 is outside the x-plate gap (ion starts inside a plate).")
    _require(abs(ic["y0"]) < p["d"] / 2, "initial_conditions.y0 is outside the y-plate gap (ion starts inside a plate).")

    # numerical settings
    _require(sim["dt"] > 0, "simulation.dt must be positive.")
    _require(sim["t_max"] > sim["dt"], "simulation.t_max must be larger than simulation.dt.")
    _require(sol["method"] in SOLVER_METHODS, f"solver.method must be one of {SOLVER_METHODS}.")
    _require(0 < sol["rtol"] < 1e-2, "solver.rtol must satisfy 0 < rtol < 0.01.")
    _require(sol["atol"] > 0, "solver.atol must be positive.")
    _require(float(out["dpi"]).is_integer() and out["dpi"] > 0, "output.dpi must be a positive integer.")
    _require(view["frame_ms"] >= 1, "viewer.frame_ms must be at least 1.")
    _require(view["frames_per_flight"] >= 1, "viewer.frames_per_flight must be at least 1.")

    # speed of the ion: the model is non-relativistic
    v0 = math.sqrt(2.0 * p["q_e"] * c["elementary_charge"] * p["V0_kV"] * 1e3
                   / (p["m_u"] * c["atomic_mass_unit"]))
    _require(v0 < c["speed_of_light"], "The ion would move faster than light: check V0_kV and m_u.")
    if v0 > 0.1 * c["speed_of_light"]:
        warnings.warn("The ion speed exceeds 10 % of the speed of light: the non-relativistic "
                      "equations used here become inaccurate.")


def validate(cfg):
    """Raise ConfigError if the configuration is incomplete or non-physical."""
    _check_structure(cfg)
    _check_physics(cfg)


def load_config(path):
    """Read and validate the JSON input file; return it as a dictionary."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Input file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON ({exc}).") from exc
    validate(cfg)
    return cfg
