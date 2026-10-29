#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ion-beam electrostatic deflection simulator  (Physics 1 project)
=================================================================

Simulates a positive ion that leaves the acceleration tube, passes between two
parallel deflection plates (uniform field E_d along y) and flies on to a target
wafer, exactly like the schematic in the project report.

Physics used
------------
Lorentz force      F = q E + q v x B
Accelerating tube  1/2 m v0^2 = q V0            ->  v0 = sqrt(2 q V0 / m)   (3.1)
Between the plates E_d = V_d / d,  a_t = q E_d / m                            (3.2)
Theory (B = 0)     tan(theta) = (V_d / V0) * l / (2 d)                        (3.6)
                   y ~ L tan(theta)                                           (3.7)

The simulation itself does NOT use these closed-form results: the ion is moved
with a 4th-order Runge-Kutta integrator of the full Lorentz-force equation of
motion, and the theory values are shown next to it for comparison.
(An optional stray magnetic field B_x is included so that you can see why
shielding is required: with B_x = 0 the simulation reproduces Eq. 3.6 / 3.7.)

Geometry (z = beam axis, y = deflection axis, as in the figure)
---------------------------------------------------------------
  z = Z_START   ion leaves the acceleration tube
  z = 0         entrance of the deflection plates
  z = l         exit of the deflection plates
  z = L         target wafer   (L measured from the plate entrance, as in the
                figure; tick the check-box to measure L from the plate centre,
                which is the convention under which Eq. 3.7 is exact)

Requirements:  Python 3.8+,  matplotlib  (pip install matplotlib)
Run:           python ion_beam_deflection.py
Keys:          Space = start/stop,   N = one step
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons, Slider

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
E_CHARGE = 1.602176634e-19      # C   elementary charge
AMU = 1.66053906660e-27         # kg  atomic mass unit
Z_START = -0.08                 # m   ion starts at the exit of the acceleration tube


# --------------------------------------------------------------------------- #
#  Parameters (all the variables that influence the trajectory)
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    V0_kV: float = 20.0         # accelerating voltage            (kV)
    Vd: float = 1000.0          # deflecting voltage              (V)   >0: top plate negative, E_d along +y
    l: float = 0.10             # length of the plates            (m)
    d: float = 0.04             # distance between the plates     (m)
    L: float = 0.50             # distance to the target wafer    (m)
    q_e: float = 1.0            # ion charge                      (multiples of e)
    m_u: float = 31.0           # ion mass                        (atomic mass units)  31 = P+
    Bx_mT: float = 0.0          # stray magnetic field B_x        (millitesla)
    L_from_centre: bool = False  # False: L from plate entrance (figure) / True: from plate centre

    @property
    def V0(self) -> float:
        return self.V0_kV * 1e3

    @property
    def q(self) -> float:
        return self.q_e * E_CHARGE

    @property
    def m(self) -> float:
        return self.m_u * AMU

    @property
    def Bx(self) -> float:
        return self.Bx_mT * 1e-3

    @property
    def v0(self) -> float:      # Eq. 3.1
        return math.sqrt(2.0 * self.q * self.V0 / self.m)

    @property
    def z_wafer(self) -> float:
        return self.L + (self.l / 2.0 if self.L_from_centre else 0.0)

    @property
    def tan_theta(self) -> float:   # Eq. 3.6
        return self.Vd * self.l / (2.0 * self.V0 * self.d)

    @property
    def t_ref(self) -> float:   # ballistic time from tube exit to wafer (used for animation speed)
        return (self.z_wafer - Z_START) / self.v0


# --------------------------------------------------------------------------- #
#  Equation of motion  (state = z, y, vz, vy)
# --------------------------------------------------------------------------- #
def region(p: Params, z: float) -> int:
    """0 = before the plates, 1 = between the plates (E_d present), 2 = after the plates."""
    if z < 0.0:
        return 0
    if z <= p.l:
        return 1
    return 2


def derivs(p: Params, s, reg: int):
    """Lorentz force  F = q E + q v x B   with E = (0, E_d, 0) in the plates and B = (B_x, 0, 0)."""
    _, _, vz, vy = s
    k = p.q / p.m
    ey = p.Vd / p.d if reg == 1 else 0.0
    az = -k * vy * p.Bx
    ay = k * (ey + vz * p.Bx)
    return (vz, vy, az, ay)


def rk4(p: Params, s, dt: float, reg: int):
    k1 = derivs(p, s, reg)
    k2 = derivs(p, [a + 0.5 * dt * b for a, b in zip(s, k1)], reg)
    k3 = derivs(p, [a + 0.5 * dt * b for a, b in zip(s, k2)], reg)
    k4 = derivs(p, [a + dt * b for a, b in zip(s, k3)], reg)
    return tuple(a + dt / 6.0 * (b + 2 * c + 2 * e + f)
                 for a, b, c, e, f in zip(s, k1, k2, k3, k4))


def advance(p: Params, s, dt: float, reg: int, events: list, depth: int = 0):
    """RK4 step that is split exactly where the ion crosses a field boundary (z = 0 or z = l)."""
    s1 = rk4(p, s, dt, reg)
    r1 = region(p, s1[0])
    if r1 == reg or depth >= 4:
        return s1, r1
    lo, hi = 0.0, dt                       # bisection for the crossing time
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if region(p, rk4(p, s, mid, reg)[0]) == reg:
            lo = mid
        else:
            hi = mid
    sb = rk4(p, s, hi, reg)
    rb = region(p, sb[0])
    events.append((reg, rb, sb))
    return advance(p, sb, dt - hi, rb, events, depth + 1)


def micro_step(p: Params, s, reg: int, dt: float):
    """One integration step.  Returns (state, region, events, outcome, fraction_of_dt_used)."""
    events: list = []
    s_new, reg_new = advance(p, s, dt, reg, events)
    outcome: Optional[str] = None
    frac = 1.0

    def lerp(f):
        return tuple(a + f * (b - a) for a, b in zip(s, s_new))

    half_gap = p.d / 2.0
    if reg_new == 1 and abs(s_new[1]) >= half_gap:                  # ion hits a plate
        a0, a1 = abs(s[1]), abs(s_new[1])
        frac = min(1.0, max(0.0, (half_gap - a0) / (a1 - a0))) if a1 > a0 else 1.0
        s_new, outcome = lerp(frac), "plate"
    elif s_new[0] >= p.z_wafer:                                     # ion reaches the wafer
        frac = (p.z_wafer - s[0]) / (s_new[0] - s[0])
        s_new, outcome = lerp(frac), "wafer"
    elif s_new[2] <= 0.0:                                           # ion turned round (huge B)
        outcome = "miss"
    return s_new, reg_new, events, outcome, frac


# --------------------------------------------------------------------------- #
#  Forecast (full trajectory computed from the current parameters)
# --------------------------------------------------------------------------- #
@dataclass
class Forecast:
    z: List[float]
    y: List[float]
    outcome: str
    z_hit: float
    y_hit: float
    t_total: float
    theta_exit: Optional[float]     # rad, direction of v when the ion leaves the plates
    y_exit: Optional[float]


def compute_forecast(p: Params) -> Forecast:
    s = (Z_START, 0.0, p.v0, 0.0)
    reg = region(p, s[0])
    dt = p.t_ref / 700.0
    zs, ys = [s[0]], [s[1]]
    t, theta_exit, y_exit, outcome = 0.0, None, None, None
    for _ in range(4000):
        s_new, reg_new, events, outcome, frac = micro_step(p, s, reg, dt)
        for r_from, r_to, sb in events:
            if r_from == 1 and r_to == 2 and theta_exit is None:
                theta_exit, y_exit = math.atan2(sb[3], sb[2]), sb[1]
        t += frac * dt
        zs.append(s_new[0])
        ys.append(s_new[1])
        s, reg = s_new, reg_new
        if outcome:
            break
    return Forecast(zs, ys, outcome or "miss", s[0], s[1], t, theta_exit, y_exit)


# --------------------------------------------------------------------------- #
#  The live ion (moved step by step by the animation / Step button)
# --------------------------------------------------------------------------- #
class Ion:
    def __init__(self, p: Params):
        self.reset(p)

    def reset(self, p: Params):
        self.s = (Z_START, 0.0, p.v0, 0.0)
        self.reg = region(p, Z_START)
        self.t = 0.0
        self.trail_z = [Z_START]
        self.trail_y = [0.0]
        self.outcome: Optional[str] = None
        self.theta_exit: Optional[float] = None
        self.y_exit: Optional[float] = None

    def advance(self, p: Params, dt: float, n_sub: int = 4):
        h = dt / n_sub
        for _ in range(n_sub):
            if self.outcome:
                break
            s_new, reg_new, events, outcome, frac = micro_step(p, self.s, self.reg, h)
            for r_from, r_to, sb in events:
                if r_from == 1 and r_to == 2 and self.theta_exit is None:
                    self.theta_exit, self.y_exit = math.atan2(sb[3], sb[2]), sb[1]
            self.t += frac * h
            self.s, self.reg, self.outcome = s_new, reg_new, outcome
            self.trail_z.append(s_new[0])
            self.trail_y.append(s_new[1])


# --------------------------------------------------------------------------- #
#  Slider definitions:  key, label, min, max, initial, step, format, column, row
# --------------------------------------------------------------------------- #
SLIDERS = [
    ("V0_kV", r"$V_0$ (kV)", 1, 200, 20, 0.5, "%.1f", 0, 0),
    ("Vd", r"$V_d$ (V)", -5000, 5000, 1000, 10, "%.0f", 0, 1),
    ("q_e", r"$q$ (×e)", 1, 5, 1, 1, "%d", 0, 2),
    ("m_u", r"$m$ (u)", 1, 250, 31, 1, "%d", 1, 0),
    ("l", r"$l$ (m)", 0.02, 0.25, 0.10, 0.005, "%.3f", 1, 1),
    ("d", r"$d$ (m)", 0.01, 0.10, 0.04, 0.001, "%.3f", 1, 2),
    ("L", r"$L$ (m)", 0.30, 1.50, 0.50, 0.01, "%.2f", 2, 0),
    ("Bx_mT", r"$B_x$ (mT)", -20, 20, 0, 0.5, "%.1f", 2, 1),
    ("speed", "speed (×)", 0.1, 5.0, 1.0, 0.1, "%.1f", 2, 2),
]


# --------------------------------------------------------------------------- #
#  Application
# --------------------------------------------------------------------------- #
class App:
    FRAME_MS = 30               # animation tick
    FRAMES_PER_FLIGHT = 165     # at speed x1 the whole flight takes ~5 s
    STEPS_PER_FLIGHT = 50       # one press of "Step" = 1/50 of the flight time

    def __init__(self):
        self.p = Params()
        self.fc = compute_forecast(self.p)
        self.ion = Ion(self.p)
        self.running = False
        self.scene = []         # artists that are rebuilt when a parameter changes
        self.yl = 0.06
        self._build_figure()
        self.rebuild_scene()
        self.refresh()
        self.timer = self.fig.canvas.new_timer(interval=self.FRAME_MS)
        self.timer.add_callback(self.tick)
        self.timer.start()

    # ------------------------------------------------------------------ GUI
    def _build_figure(self):
        self.fig = plt.figure(figsize=(15, 9.2))
        try:
            self.fig.canvas.manager.set_window_title("Ion beam deflection simulator")
        except Exception:
            pass
        self.fig.suptitle("Ion-beam deflection in an ion implanter - uniform electric field "
                          "between the deflection plates", fontsize=13, fontweight="bold", y=0.99)

        # main plot -------------------------------------------------------
        self.ax = self.fig.add_axes([0.055, 0.41, 0.62, 0.515])
        self.ax.set_xlabel("z  (m)   -  beam axis")
        self.ax.set_ylabel("y  (m)   -  deflection axis")
        self.ax.grid(alpha=0.25)
        ax = self.ax
        (self.ln_fc,) = ax.plot([], [], "--", color="#ff7f0e", lw=1.8, zorder=4,
                                label="forecast trajectory")
        (self.ln_trail,) = ax.plot([], [], "-", color="#1f77b4", lw=2.6, zorder=5,
                                   label="ion path so far")
        (self.ln_tan,) = ax.plot([], [], ":", color="k", lw=1.3, zorder=3,
                                 label="tangent at plate exit")
        (self.pt_ion,) = ax.plot([], [], "o", ms=11, color="#1f77b4", mec="k", zorder=7)
        (self.pt_fc,) = ax.plot([], [], "o", ms=12, mfc="none", mec="#d62728", mew=2, zorder=6,
                                label="forecast impact")
        (self.pt_hit,) = ax.plot([], [], "X", ms=11, color="#d62728", mec="k", zorder=8,
                                 label="simulated impact")
        self.ion_lbl = ax.annotate("3: ion (m, q)", xy=(Z_START, 0), xytext=(8, 12),
                                   textcoords="offset points", fontsize=9, zorder=9)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=8.5,
                  frameon=False)

        # results panel -----------------------------------------------------
        self.axi = self.fig.add_axes([0.70, 0.41, 0.29, 0.515])
        self.axi.set_xticks([])
        self.axi.set_yticks([])
        self.axi.set_facecolor("#f5f6fa")
        self.info = self.axi.text(0.03, 0.985, "", va="top", ha="left", family="monospace",
                                  fontsize=8.7, transform=self.axi.transAxes)

        # buttons -----------------------------------------------------------
        bw, bh, by = 0.085, 0.05, 0.295
        self.b_run = Button(self.fig.add_axes([0.055, by, bw, bh]), "Start",
                            color="#b7e4c7", hovercolor="#95d5b2")
        self.b_step = Button(self.fig.add_axes([0.150, by, bw, bh]), "Step",
                             color="#ffe8a3", hovercolor="#ffd966")
        self.b_reset = Button(self.fig.add_axes([0.245, by, bw, bh]), "Reset",
                              color="#ffc9c9", hovercolor="#ffa8a8")
        self.b_def = Button(self.fig.add_axes([0.340, by, bw, bh]), "Defaults",
                            color="#dfe3f7", hovercolor="#c5cbee")
        for b in (self.b_run, self.b_step, self.b_reset, self.b_def):
            b.label.set_fontsize(11)
        self.b_run.on_clicked(self.on_run)
        self.b_step.on_clicked(self.on_step)
        self.b_reset.on_clicked(self.on_reset)
        self.b_def.on_clicked(self.on_defaults)

        # check-box: reference point of L ---------------------------------------
        ax_chk = self.fig.add_axes([0.445, 0.285, 0.25, 0.065])
        ax_chk.set_frame_on(False)
        self.chk = CheckButtons(ax_chk, ["measure L from the plate centre\n(convention of Eq. 3.7)"],
                                [False])
        self.chk.labels[0].set_fontsize(9)
        self.chk.on_clicked(self.on_param)

        # sliders (control panel) --------------------------------------------
        self.fig.text(0.055, 0.245,
                      "CONTROL PANEL  -  any change recomputes the forecast and restarts the ion",
                      fontsize=10.5, fontweight="bold")
        self.sl = {}
        col_left = [0.105, 0.435, 0.765]
        row_y = [0.19, 0.12, 0.05]
        for key, label, vmin, vmax, vinit, step, fmt, col, row in SLIDERS:
            sax = self.fig.add_axes([col_left[col], row_y[row], 0.145, 0.03])
            sld = Slider(sax, label, vmin, vmax, valinit=vinit, valstep=step, valfmt=fmt,
                         color="#74a9e6")
            sld.label.set_fontsize(10)
            sld.valtext.set_fontsize(9.5)
            if key != "speed":
                sld.on_changed(self.on_param)
            self.sl[key] = sld

        self.fig.text(0.99, 0.008, "Keys: Space = start/stop,  N = one step", ha="right",
                      fontsize=8, color="gray")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    # ------------------------------------------------------------ scene
    def rebuild_scene(self):
        """(Re)draw everything that depends on the parameters: tube, plates, wafer, arrows ..."""
        for a in self.scene:
            try:
                a.remove()
            except Exception:
                pass
        self.scene = []
        add = self.scene.append
        ax, p, fc = self.ax, self.p, self.fc

        ymax_path = max((abs(v) for v in fc.y), default=0.0)
        yl = max(0.06, 0.9 * p.d, 1.35 * ymax_path)
        yl = math.ceil(yl * 100 - 1e-9) / 100.0
        self.yl = yl
        zw = p.z_wafer
        ax.set_xlim(Z_START - 0.10, zw + 0.10)
        ax.set_ylim(-yl, yl)
        th = 0.07 * yl                # drawn thickness of a plate
        hh = 0.78 * yl                # half height of tube and wafer
        top = ax.get_xaxis_transform()

        # 1: acceleration tube
        add(ax.add_patch(Rectangle((Z_START - 0.08, -hh), 0.08, 2 * hh, fc="#c46b4d", ec="#5c2e1e",
                                   lw=1.2, zorder=1)))
        add(ax.text(Z_START - 0.04, 0, rf"$V_0$ = {p.V0_kV:g} kV", rotation=90, color="white",
                    ha="center", va="center", fontsize=11, fontweight="bold", zorder=3))
        add(ax.text(Z_START - 0.04, 0.97, "1: acceleration\ntube", transform=top, ha="center",
                    va="top", fontsize=9))

        # 5: target wafer
        add(ax.add_patch(Rectangle((zw - 0.008, -hh), 0.016, 2 * hh, fc="#d9dcf2", ec="#444",
                                   lw=1.2, zorder=1)))
        add(ax.text(zw, 0.97, "5: target\nwafer", transform=top, ha="center", va="top", fontsize=9))

        # 2: deflection plates (colours follow the figure: red = "-", blue = "+")
        neg, pos, grey = "#d62728", "#1f77b4", "#999999"
        if p.Vd > 0:
            ctop, cbot, stop, sbot = neg, pos, "−", "+"
        elif p.Vd < 0:
            ctop, cbot, stop, sbot = pos, neg, "+", "−"
        else:
            ctop = cbot = grey
            stop = sbot = ""
        add(ax.add_patch(Rectangle((0, p.d / 2), p.l, th, fc=ctop, ec="k", lw=0.8, zorder=2)))
        add(ax.add_patch(Rectangle((0, -p.d / 2 - th), p.l, th, fc=cbot, ec="k", lw=0.8, zorder=2)))
        add(ax.text(p.l / 2, p.d / 2 + th / 2, stop, color="white", ha="center", va="center",
                    fontsize=9, fontweight="bold", zorder=3))
        add(ax.text(p.l / 2, -p.d / 2 - th / 2, sbot, color="white", ha="center", va="center",
                    fontsize=9, fontweight="bold", zorder=3))
        add(ax.text(p.l / 2, 0.97, "2: deflection plates", transform=top, ha="center", va="top",
                    fontsize=9))

        # electric field arrows E_d
        if p.Vd != 0:
            n, gap = 8, 0.1 * p.d
            sg = 1.0 if p.Vd > 0 else -1.0
            xs = [p.l * (i + 0.5) / n for i in range(n)]
            add(ax.quiver(xs, [-sg * (p.d / 2 - gap)] * n, [0] * n, [sg * (p.d - 2 * gap)] * n,
                          angles="xy", scale_units="xy", scale=1, color="#2a6fdb", width=0.0022,
                          alpha=0.8, zorder=2))
            add(ax.text(p.l + 0.012, sg * p.d * 0.28, r"$\vec{E}_d$", fontsize=11, color="#2a6fdb",
                        ha="left", va="center"))

        # beam axis
        add(ax.plot([Z_START, zw], [0, 0], "--", color="#444", lw=0.9, zorder=1)[0])

        # dimensions d, l, L
        arrow = dict(arrowstyle="<->", color="k", lw=1.0)
        add(ax.annotate("", xy=(-0.02, p.d / 2), xytext=(-0.02, -p.d / 2), arrowprops=arrow))
        add(ax.text(-0.028, p.d * 0.25, "d", ha="right", va="center", fontsize=11, style="italic"))
        y_l = -p.d / 2 - th - 0.10 * yl
        add(ax.annotate("", xy=(p.l, y_l), xytext=(0, y_l), arrowprops=arrow))
        add(ax.text(p.l / 2, y_l - 0.03 * yl, "l", ha="center", va="top", fontsize=11, style="italic"))
        y_L = y_l - 0.15 * yl
        z_ref = p.l / 2 if p.L_from_centre else 0.0
        add(ax.annotate("", xy=(zw, y_L), xytext=(z_ref, y_L), arrowprops=arrow))
        add(ax.text((zw + z_ref) / 2, y_L - 0.03 * yl, "L", ha="center", va="top", fontsize=11,
                    style="italic"))
        add(ax.plot([z_ref, z_ref], [y_L, 0], ":", color="gray", lw=0.9, zorder=1)[0])

        # 4: trajectory label
        if fc.outcome == "wafer":
            i = int(0.82 * (len(fc.z) - 1))
            add(ax.annotate("4: trajectory", xy=(fc.z[i], fc.y[i]), xytext=(-6, 14),
                            textcoords="offset points", ha="right", fontsize=9))

        # theta label and y bracket
        if fc.theta_exit is not None and abs(math.tan(fc.theta_exit)) > 1e-9:
            slope = math.tan(fc.theta_exit)
            z_v = p.l - fc.y_exit / slope             # where the tangent crosses the axis
            sgn = 1.0 if slope > 0 else -1.0
            add(ax.text(z_v + 0.05, -sgn * 0.09 * yl, rf"$\theta$ = {math.degrees(fc.theta_exit):+.2f}°",
                        fontsize=11, color="k", ha="left", va="center"))
        if fc.outcome == "wafer":
            xb = zw + 0.04
            add(ax.annotate("", xy=(xb, fc.y_hit), xytext=(xb, 0),
                            arrowprops=dict(arrowstyle="<->", color="#2b8a3e", lw=1.4)))
            add(ax.text(xb + 0.008, fc.y_hit / 2, "y", color="#2b8a3e", fontsize=12, style="italic",
                        ha="left", va="center"))
            add(ax.plot([zw, xb + 0.01], [fc.y_hit, fc.y_hit], ":", color="#2b8a3e", lw=1.0)[0])

    # ------------------------------------------------------------ refresh
    def refresh(self):
        p, fc, ion = self.p, self.fc, self.ion
        self.ln_fc.set_data(fc.z, fc.y)
        self.ln_trail.set_data(ion.trail_z, ion.trail_y)
        self.pt_ion.set_data([ion.s[0]], [ion.s[1]])
        self.ion_lbl.xy = (ion.s[0], ion.s[1])
        if fc.outcome in ("wafer", "plate"):
            self.pt_fc.set_data([fc.z_hit], [fc.y_hit])
        else:
            self.pt_fc.set_data([], [])
        if ion.outcome in ("wafer", "plate"):
            self.pt_hit.set_data([ion.s[0]], [ion.s[1]])
        else:
            self.pt_hit.set_data([], [])
        if fc.theta_exit is not None and abs(math.tan(fc.theta_exit)) > 1e-9:
            slope = math.tan(fc.theta_exit)
            z_v = p.l - fc.y_exit / slope
            self.ln_tan.set_data([z_v, p.z_wafer], [0.0, fc.y_exit + slope * (p.z_wafer - p.l)])
        else:
            self.ln_tan.set_data([], [])
        self.info.set_text(self.info_text())
        self.b_run.label.set_text("Stop" if self.running else "Start")
        self.fig.canvas.draw_idle()

    def info_text(self) -> str:
        p, fc, ion = self.p, self.fc, self.ion
        th = math.atan(p.tan_theta)
        L_ = []
        L_.append("BEAM")
        L_.append(f" v0  = {p.v0:.3e} m/s   E_k = {p.q_e * p.V0_kV:.1f} keV")
        L_.append(f" E_d = V_d/d = {p.Vd / p.d:+.3e} V/m")
        L_.append(f" time between plates t = l/v0 = {p.l / p.v0 * 1e9:.1f} ns")
        L_.append("")
        L_.append("THEORY  (Eq. 3.6 / 3.7, valid for B = 0)")
        L_.append(f" tan θ = V_d·l /(2·V0·d) = {p.tan_theta:+.5f}")
        L_.append(f" θ = {math.degrees(th):+.4f} deg = {th:+.5f} rad")
        L_.append(f" y = L·tan θ = {p.L * p.tan_theta:+.5f} m")
        if p.Bx_mT != 0:
            L_.append(" !! B_x ≠ 0: theory above is not exact")
        L_.append("")
        L_.append("SIMULATION  (live ion)")
        status = {"wafer": "ion hit the wafer", "plate": "ION HIT A PLATE",
                  "miss": "ion turned back"}.get(ion.outcome or "")
        if status is None:
            status = "running" if self.running else ("ready" if ion.t == 0 else "paused")
        L_.append(f" status = {status}")
        L_.append(f" t = {ion.t * 1e6:.4f} µs")
        L_.append(f" z = {ion.s[0]:+.5f} m")
        L_.append(f" y = {ion.s[1]:+.5f} m  ({ion.s[1] * 1e3:+.2f} mm)")
        ang_now = math.degrees(math.atan2(ion.s[3], ion.s[2]))
        L_.append(f" θ now  = {ang_now:+.4f} deg")
        if ion.theta_exit is not None:
            L_.append(f" θ exit = {math.degrees(ion.theta_exit):+.4f} deg  (left the plates)")
        else:
            L_.append(" θ exit = -  (still before / inside the plates)")
        L_.append("")
        L_.append("IMPACT POINT ON THE WAFER")
        if fc.outcome == "wafer":
            L_.append(f" forecast : z = {fc.z_hit:.4f} m")
            L_.append(f"            y = {fc.y_hit:+.5f} m  ({fc.y_hit * 1e3:+.2f} mm)")
        elif fc.outcome == "plate":
            L_.append(f" forecast : ion hits a plate at z = {fc.z_hit:.4f} m")
            L_.append("            (reduce V_d or increase d / V0)")
        else:
            L_.append(" forecast : the ion never reaches the wafer")
        if ion.outcome == "wafer":
            L_.append(f" simulated: z = {ion.s[0]:.4f} m")
            L_.append(f"            y = {ion.s[1]:+.5f} m  ({ion.s[1] * 1e3:+.2f} mm)")
        elif ion.outcome == "plate":
            L_.append(f" simulated: hit plate at z = {ion.s[0]:.4f} m")
        else:
            L_.append(" simulated: not reached yet")
        if not p.L_from_centre:
            L_.append("")
            L_.append(" note: Eq. 3.7 measures L from the plate centre;")
            L_.append(" here L starts at the plate entrance, as in the figure.")
        return "\n".join(L_)

    # ------------------------------------------------------------ callbacks
    def read_params(self):
        v = {k: s.val for k, s in self.sl.items()}
        self.p = Params(V0_kV=v["V0_kV"], Vd=v["Vd"], l=v["l"], d=v["d"], L=v["L"],
                        q_e=v["q_e"], m_u=v["m_u"], Bx_mT=v["Bx_mT"],
                        L_from_centre=bool(self.chk.get_status()[0]))

    def on_param(self, _=None):
        self.read_params()
        self.fc = compute_forecast(self.p)
        self.ion.reset(self.p)
        self.rebuild_scene()
        self.refresh()

    def on_run(self, _=None):
        if self.ion.outcome:
            self.ion.reset(self.p)
        self.running = not self.running
        self.refresh()

    def on_step(self, _=None):
        self.running = False
        if self.ion.outcome:
            self.ion.reset(self.p)
        self.ion.advance(self.p, self.p.t_ref / self.STEPS_PER_FLIGHT)
        self.refresh()

    def on_reset(self, _=None):
        self.running = False
        self.ion.reset(self.p)
        self.refresh()

    def on_defaults(self, _=None):
        self.running = False
        for sld in self.sl.values():
            sld.eventson = False
            sld.set_val(sld.valinit)
            sld.eventson = True
        if self.chk.get_status()[0]:
            self.chk.eventson = False
            self.chk.set_active(0)
            self.chk.eventson = True
        self.on_param()

    def on_key(self, event):
        if event.key == " ":
            self.on_run()
        elif event.key in ("n", "N"):
            self.on_step()

    def tick(self):
        if not self.running:
            return
        dt = self.p.t_ref / self.FRAMES_PER_FLIGHT * self.sl["speed"].val
        self.ion.advance(self.p, dt)
        if self.ion.outcome:
            self.running = False
        self.refresh()


def main():
    app = App()          # keep a reference so that widgets/timer are not garbage-collected
    plt.show()
    return app


if __name__ == "__main__":
    main()