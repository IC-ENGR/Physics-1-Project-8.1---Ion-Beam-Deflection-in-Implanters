import json
import math
import warnings
from pathlib import Path


class ConfigError(ValueError):
    """"""
REQUIRED = {
    "constants": ("elementary_charge", "atomic_mass_unit", "speed_of_light"),
    "parameters": ("V0", "Vy", "ly", "dy", "Vx", "lx", "dx", "gap_xy", "L",
                   "wafer_dia", "q", "m"),
    "geometry": ("tube_len", "tube_gap"),
    "initial_conditions": ("x0", "y0", "vx0", "vy0"),
    "simulation": ("dt", "t_max"),
    "solver": ("method", "rtol", "atol"),
    "output": ("results_dir", "figures_dir", "results_csv", "dpi"),
    "viewer": ("frame_ms", "frames_per_flight"),
}
TOP_LEVEL = ("system") + tuple(REQUIRED)
STRING_KEYS = {"solver": ("method",),
               "output": ("results_dir", "figures_dir", "results_csv")}
SOLVER_METHODS = ("DOP853")

def CheckFiniteNumber(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def CheckStructure(cfg):
    if not isinstance(cfg, dict):
        raise ConfigError("The top level of the input file must be a JSON object { ... }.")
    unknown = set(cfg) - set(TOP_LEVEL)
    if unknown:
        raise ConfigError(f"Unknown top-level entries: {sorted(unknown)}")
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
            elif not CheckFiniteNumber(value):
                raise ConfigError(f"'{section}.{key}' must be a finite number, got {value!r}.")

def _require(condition, message):
    if not condition:
        raise ConfigError(message)

def CheckPhysics(cfg):
    c, p, g = cfg["constants"], cfg["parameters"], cfg["geometry"]
    ic, sim, sol, out, view = (cfg["initial_conditions"], cfg["simulation"], cfg["solver"],
                               cfg["output"], cfg["viewer"])
    for key in ("elementary_charge", "atomic_mass_unit", "speed_of_light"):
        _require(c[key] > 0, f"constants.{key} must be positive.")
    _require(p["V0"] > 0, "V0 must be positive.")
    _require(p["m"] > 0, "ion mass m must be positive.")
    _require(p["q"] > 0, "ion charge must be positive")
    for key in ("ly", "dy", "lx", "dx", "L", "wafer_dia"):
        _require(p[key] > 0, f"{key} must be positive.")
    _require(p["gap_xy"] >= 0, "The gap between x and y plates must not be negative.")
    for key in ("tube_len", "tube_gap"):
        _require(g[key] > 0, f"geometry.{key} must be positive.")

    z_wafer = p["L"]
    _require(z_wafer > p["ly"], "The wafer must be behind the y plates.")
    _require(abs(ic["x0"]) < p["dx"] / 2, "x0 is outside the x-plates gap.")
    _require(abs(ic["y0"]) < p["dy"] / 2, "y0 is outside the y-plates gap.")
    _require(sim["dt"] > 0, "dt must be positive.")
    _require(sim["t_max"] > sim["dt"], "t_max must be larger than dt.")
    _require(sol["method"] in SOLVER_METHODS, f"method must be one of {SOLVER_METHODS}.")
    _require(0 < sol["rtol"] < 1e-2, "0 < rtol < 0.01.")
    _require(sol["atol"] > 0, "atol")
    _require(float(out["dpi"]).is_integer() and out["dpi"] > 0, "dpi must be a positive integer.")
    _require(view["frame_ms"] >= 1, "frame_ms must be at least 1.")
    _require(view["frames_per_flight"] >= 1, "frames_per_flight must be at least 1.")
    v0 = math.sqrt(2.0 * p["q"] * c["elementary_charge"] * p["V0"] * 1e3
                   / (p["m"] * c["atomic_mass_unit"]))
    _require(v0 < c["speed_of_light"], "The ion would move faster than light")
    if v0 > 0.1 * c["speed_of_light"]:
        warnings.warn("The ion speed exceeds 10 % of the speed of light: the non-relativistic equations used here become inaccurate.")

def validate(cfg):
    CheckStructure(cfg)
    CheckPhysics(cfg)

def LoadConfiguration(path):
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