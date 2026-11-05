import math
from dataclasses import dataclass
import numpy as np
REGION_NAMES = ("tube", "x_plates", "gap", "y_plates", "drift")
X_PLATES, Y_PLATES = 1, 3

@dataclass(frozen=True)
class BeamLine:
    elementary_charge: float  
    amu: float        
    c_light: float      
    V0: float      
    Vy: float       
    ly: float         
    dy: float        
    Vx: float        
    lx: float           
    dx: float           
    gap_xy: float        
    L: float           
    wafer_dia: float     
    q: float             
    m: float
    tube_len: float
    tube_gap: float
    x0: float            
    y0: float            
    vx0: float         
    vy0: float 

    @classmethod
    def from_config(cls, cfg):
        c, p, g, ic = cfg["constants"], cfg["parameters"], cfg["geometry"], cfg["initial_conditions"]
        return cls(
            elementary_charge=c["elementary_charge"], amu=c["atomic_mass_unit"], c_light=c["speed_of_light"],
            V0=p["V0"] * 1e3, Vy=p["Vy"], ly=p["ly"], dy=p["dy"],
            Vx=p["Vx"], lx=p["lx"], dx=p["dx"], gap_xy=p["gap_xy"],
            L=p["L"], wafer_dia=p["wafer_dia"],
            q=p["q"] * c["elementary_charge"], m=p["m"] * c["atomic_mass_unit"],
            tube_len=g["tube_len"], tube_gap=g["tube_gap"],
            x0=ic["x0"], y0=ic["y0"], vx0=ic["vx0"], vy0=ic["vy0"])
    @property
    def v0(self): 
        return math.sqrt(2.0 * self.q * self.V0 / self.m)

    @property
    def K0(self):
        return 0.5 * self.m * (self.v0 ** 2 + self.vx0 ** 2 + self.vy0 ** 2)

    @property
    def wafer_R(self):
        return 0.5e-3 * self.wafer_dia

    @property
    def zx0(self):
        return -(self.gap_xy + self.lx)

    @property
    def zx1(self):
        return -self.gap_xy

    @property
    def zcx(self):
        return 0.5 * (self.zx0 + self.zx1)

    @property
    def z_start(self):
        return self.zx0 - self.tube_gap

    @property
    def z_wafer(self):
        return self.L

    @property
    def Lx(self):
        return self.z_wafer - self.zcx

    @property
    def Ex(self):
        return self.Vx / self.dx

    @property
    def Ey(self):                      
        return self.Vy / self.dy

    @property
    def tan_theta_x(self):             
        return self.Vx * self.lx / (2.0 * self.V0 * self.dx)

    @property
    def tan_theta_y(self):             
        return self.Vy * self.ly / (2.0 * self.V0 * self.dy)

    def segments(self):
        return (
            ("tube", self.zx0, 0.0, 0.0),
            ("x_plates", self.zx1, self.Ex, 0.0),
            ("gap", 0.0, 0.0, 0.0),
            ("y_plates", self.ly, 0.0, self.Ey),
            ("drift", self.z_wafer, 0.0, 0.0),
        )

def make_rhs(bl, ex, ey):
    ax = bl.q * ex / bl.m
    ay = bl.q * ey / bl.m
    work_rate = bl.q / bl.K0
    def rhs(t, s):
        vx, vy, vz = s[3], s[4], s[5]
        return np.array([vx, vy, vz, ax, ay, 0.0, work_rate * (ex * vx + ey * vy)])
    return rhs

def kinetic_energy(bl, vx, vy, vz):
    return 0.5 * bl.m * (vx ** 2 + vy ** 2 + vz ** 2)

def potential_energy(bl, w):
    return 0.0 - w * bl.K0

def bending_angle(vx, vy, vz):
    """Angle (deg) between the velocity and the z axis, projected on the x-z and y-z planes."""
    return np.degrees(np.arctan2(vx, vz)), np.degrees(np.arctan2(vy, vz))

def theory(bl):
    flight = bl.z_wafer - bl.z_start
    return {
        "v0": bl.v0,
        "t_flight": flight / bl.v0,
        "tan_theta_x": bl.tan_theta_x,
        "tan_theta_y": bl.tan_theta_y,
        "theta_x_deg": math.degrees(math.atan(bl.tan_theta_x + bl.vx0 / bl.v0)),
        "theta_y_deg": math.degrees(math.atan(bl.tan_theta_y + bl.vy0 / bl.v0)),
        "x_wafer": bl.x0 + bl.vx0 / bl.v0 * flight + bl.tan_theta_x * bl.Lx,
        "y_wafer": bl.y0 + bl.vy0 / bl.v0 * flight + bl.tan_theta_y * (bl.z_wafer - bl.ly / 2.0),
        "Vx_limit": 2.0 * bl.V0 * bl.dx ** 2 / bl.lx ** 2,
        "Vy_limit": 2.0 * bl.V0 * bl.dy ** 2 / bl.ly ** 2,
    }