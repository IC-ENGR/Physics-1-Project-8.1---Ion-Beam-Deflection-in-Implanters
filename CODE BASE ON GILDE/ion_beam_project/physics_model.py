"""Physical model of the ion beam line.

Contains everything that describes the PHYSICS and nothing else:

* the geometry of the beam line (tube, x plates, gap, y plates, drift, wafer),
* the equations of motion (Lorentz force, here F = qE because B = 0),
* the energies and the angular momentum of the ion,
* the closed-form theory (Eq. 3.1 - 3.7 of the report) used only for comparison.

No number is hard-coded here: every constant and parameter is received from the
JSON input file through BeamLine.from_config().

State vector of the ion:  s = [x, y, z, vx, vy, vz, w]
    x, y, z    position (m)            z = beam axis, x and y = deflection axes
    vx, vy, vz velocity (m/s)
    w          work done by the electric field so far, divided by the initial
               kinetic energy K0 (dimensionless; it is integrated together with
               the motion so that the potential energy U = -w*K0 is known)
"""

import math
from dataclasses import dataclass

import numpy as np

# Regions of the beam line, in the order the ion crosses them.
REGION_NAMES = ("tube", "x_plates", "gap", "y_plates", "drift")
X_PLATES, Y_PLATES = 1, 3


@dataclass(frozen=True)
class BeamLine:
    """All constants and parameters of one simulation (immutable, no defaults)."""
    e_charge: float      # C    elementary charge
    amu: float           # kg   atomic mass unit
    c_light: float       # m/s  speed of light
    V0: float            # V    accelerating voltage
    Vd: float            # V    voltage across the y plates
    l: float             # m    length of the y plates
    d: float             # m    gap of the y plates
    Vx: float            # V    voltage across the x plates
    lx: float            # m    length of the x plates
    dx: float            # m    gap of the x plates
    gap_xy: float        # m    free distance between x plates and y plates
    L: float             # m    distance to the wafer
    wafer_mm: float      # mm   wafer diameter
    q: float             # C    ion charge
    m: float             # kg   ion mass
    L_from_centre: bool  # True: L measured from the centre of the y plates
    tube_len: float      # m    drawn length of the acceleration tube
    tube_gap: float      # m    tube exit -> entrance of the x plates
    x0: float            # m    initial x
    y0: float            # m    initial y
    vx0: float           # m/s  initial vx
    vy0: float           # m/s  initial vy

    # ------------------------------------------------------------------ build
    @classmethod
    def from_config(cls, cfg):
        c, p, g, ic = cfg["constants"], cfg["parameters"], cfg["geometry"], cfg["initial_conditions"]
        return cls(
            e_charge=c["elementary_charge"], amu=c["atomic_mass_unit"], c_light=c["speed_of_light"],
            V0=p["V0_kV"] * 1e3, Vd=p["Vd"], l=p["l"], d=p["d"],
            Vx=p["Vx"], lx=p["lx"], dx=p["dx"], gap_xy=p["gap_xy"],
            L=p["L"], wafer_mm=p["wafer_mm"],
            q=p["q_e"] * c["elementary_charge"], m=p["m_u"] * c["atomic_mass_unit"],
            L_from_centre=p["L_from_centre"], tube_len=g["tube_len"], tube_gap=g["tube_gap"],
            x0=ic["x0"], y0=ic["y0"], vx0=ic["vx0"], vy0=ic["vy0"])

    # ------------------------------------------------------- derived quantities
    @property
    def v0(self):                      # Eq. 3.1:  (1/2) m v0^2 = q V0
        return math.sqrt(2.0 * self.q * self.V0 / self.m)

    @property
    def K0(self):                      # initial kinetic energy (J)
        return 0.5 * self.m * (self.v0 ** 2 + self.vx0 ** 2 + self.vy0 ** 2)

    @property
    def wafer_R(self):                 # wafer radius (m)
        return 0.5e-3 * self.wafer_mm

    @property
    def zx0(self):                     # entrance of the x plates
        return -(self.gap_xy + self.lx)

    @property
    def zx1(self):                     # exit of the x plates
        return -self.gap_xy

    @property
    def zcx(self):                     # centre of the x plates
        return 0.5 * (self.zx0 + self.zx1)

    @property
    def z_start(self):                 # ion starts at the exit of the acceleration tube
        return self.zx0 - self.tube_gap

    @property
    def z_wafer(self):                 # the y plates occupy 0 <= z <= l
        return self.L + (self.l / 2.0 if self.L_from_centre else 0.0)

    @property
    def Lx(self):                      # centre of the x plates -> wafer
        return self.z_wafer - self.zcx

    @property
    def Ex(self):                      # electric field between the x plates (V/m)
        return self.Vx / self.dx

    @property
    def Ey(self):                      # electric field between the y plates (V/m)
        return self.Vd / self.d

    @property
    def tan_theta_x(self):             # Eq. 3.6 for the x plates
        return self.Vx * self.lx / (2.0 * self.V0 * self.dx)

    @property
    def tan_theta_y(self):             # Eq. 3.6 for the y plates
        return self.Vd * self.l / (2.0 * self.V0 * self.d)

    # --------------------------------------------------------------- geometry
    def segments(self):
        """Regions in the order the ion crosses them.

        Each entry is (name, z at the end of the region, Ex, Ey) where Ex and Ey
        are the electric fields that act inside that region.
        """
        return (
            ("tube", self.zx0, 0.0, 0.0),
            ("x_plates", self.zx1, self.Ex, 0.0),
            ("gap", 0.0, 0.0, 0.0),
            ("y_plates", self.l, 0.0, self.Ey),
            ("drift", self.z_wafer, 0.0, 0.0),
        )


# --------------------------------------------------------------------------- #
#  Equations of motion
# --------------------------------------------------------------------------- #
def make_rhs(bl, ex, ey):
    """Right-hand side ds/dt of the equations of motion for constant fields (ex, ey).

    Newton + Lorentz force with B = 0:   m dv/dt = q E,   dr/dt = v.
    The last component integrates the work of the field, dw/dt = q E.v / K0.
    """
    ax = bl.q * ex / bl.m
    ay = bl.q * ey / bl.m
    work_rate = bl.q / bl.K0

    def rhs(t, s):
        vx, vy, vz = s[3], s[4], s[5]
        return np.array([vx, vy, vz, ax, ay, 0.0, work_rate * (ex * vx + ey * vy)])

    return rhs


# --------------------------------------------------------------------------- #
#  Energies and conserved quantities (all work on whole numpy arrays)
# --------------------------------------------------------------------------- #
def kinetic_energy(bl, vx, vy, vz):
    return 0.5 * bl.m * (vx ** 2 + vy ** 2 + vz ** 2)


def potential_energy(bl, w):
    """U = -(work done by the field so far).  For a uniform field this is U = -qE.(r - r_in)."""
    return 0.0 - w * bl.K0


def angular_momentum(bl, x, y, z, vx, vy, vz):
    """L = m (r x v) about the origin, shape (N, 3)."""
    r = np.stack([x, y, z], axis=-1)
    v = np.stack([vx, vy, vz], axis=-1)
    return bl.m * np.cross(r, v)


# --------------------------------------------------------------------------- #
#  Closed-form theory (for comparison only)
# --------------------------------------------------------------------------- #
def theory(bl):
    """Eq. 3.1 - 3.7 of the report, extended with the initial position and velocity."""
    flight = bl.z_wafer - bl.z_start
    return {
        "v0": bl.v0,
        "t_flight": flight / bl.v0,
        "tan_theta_x": bl.tan_theta_x,
        "tan_theta_y": bl.tan_theta_y,
        "theta_x_deg": math.degrees(math.atan(bl.tan_theta_x + bl.vx0 / bl.v0)),   # direction after the plates
        "theta_y_deg": math.degrees(math.atan(bl.tan_theta_y + bl.vy0 / bl.v0)),
        "x_wafer": bl.x0 + bl.vx0 / bl.v0 * flight + bl.tan_theta_x * bl.Lx,
        "y_wafer": bl.y0 + bl.vy0 / bl.v0 * flight + bl.tan_theta_y * (bl.z_wafer - bl.l / 2.0),
        "Vx_limit": 2.0 * bl.V0 * bl.dx ** 2 / bl.lx ** 2,   # larger |Vx| hits an x plate
        "Vd_limit": 2.0 * bl.V0 * bl.d ** 2 / bl.l ** 2,     # larger |Vd| hits a y plate
    }
