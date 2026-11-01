import math
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, CheckButtons, TextBox



# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
E_CHARGE = 1.602176634e-19       # C   elementary charge
AMU = 1.66053906660e-27         # kg  atomic mass unit
Z_START = -0.08                 # m   ion starts at the exit of the acceleration tube
GAP_XY = 0.02                   # m   drift distance between X-plates and Y-plates


# --------------------------------------------------------------------------- #
#  Parameters
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    V0_kV: float = 20.0         # accelerating voltage              (kV)
    Vd: float = 1000.0          # Y deflecting voltage              (V)
    Vdx: float = 1000.0         # X deflecting voltage              (V)
    lx: float = 0.08            # length of X plates                (m)
    ly: float = 0.10            # length of Y plates                (m)
    dx_plate: float = 0.04      # distance between X plates         (m)
    d: float = 0.04             # distance between Y plates         (m)
    L: float = 0.50             # distance to the target wafer      (m)
    q_e: float = 1.0            # ion charge                        (multiples of e)
    m_u: float = 31.0           # ion mass                          (atomic mass units)
    Bx_mT: float = 0.0          # stray magnetic field B_x          (millitesla)
    speed: float = 1.0          # simulation speed multiplier
    L_from_centre: bool = False # False: L from Y-plate entrance / True: from Y-plate centre

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
    def v0(self) -> float:
        return math.sqrt(2.0 * self.q * self.V0 / self.m)

    @property
    def z_x_start(self) -> float:
        return 0.0

    @property
    def z_x_end(self) -> float:
        return self.lx

    @property
    def z_y_start(self) -> float:
        return self.lx + GAP_XY

    @property
    def z_y_end(self) -> float:
        return self.lx + GAP_XY + self.ly

    @property
    def z_wafer(self) -> float:
        y_ref_centre = self.z_y_start + (self.ly / 2.0)
        return self.L + (y_ref_centre if self.L_from_centre else 0.0)

    @property
    def t_ref(self) -> float:
        return (self.z_wafer - Z_START) / self.v0


# --------------------------------------------------------------------------- #
#  Equation of motion  (state = z, y, x, vz, vy, vx)
# --------------------------------------------------------------------------- #
def region(p: Params, z: float) -> int:
    """
    0 = before X plates (z < 0)
    1 = inside X plates (0 <= z <= lx)
    2 = drift between X and Y plates (lx < z < lx + GAP_XY)
    3 = inside Y plates (lx + GAP_XY <= z <= lx + GAP_XY + ly)
    4 = after Y plates / drift to wafer
    """
    if z < 0.0:
        return 0
    if z <= p.z_x_end:
        return 1
    if z < p.z_y_start:
        return 2
    if z <= p.z_y_end:
        return 3
    return 4


def derivs(p: Params, s, reg: int):
    """Lorentz force F = q E + q v x B with E_x in X-plates, E_y in Y-plates, B_x."""
    _, y, x, vz, vy, vx = s
    k = p.q / p.m
    ex = p.Vdx / p.dx_plate if reg == 1 else 0.0
    ey = p.Vd / p.d if reg == 3 else 0.0
    bx = p.Bx

    az = k * (-vy * bx)
    ay = k * (ey + vz * bx)
    ax = k * (ex - vy * bx)
    return (vz, vy, vx, az, ay, ax)


def rk4(p: Params, s, dt: float, reg: int):
    k1 = derivs(p, s, reg)
    k2 = derivs(p, [a + 0.5 * dt * b for a, b in zip(s, k1)], reg)
    k3 = derivs(p, [a + 0.5 * dt * b for a, b in zip(s, k2)], reg)
    k4 = derivs(p, [a + dt * b for a, b in zip(s, k3)], reg)
    return tuple(a + dt / 6.0 * (b + 2 * c + 2 * e + f)
                 for a, b, c, e, f in zip(s, k1, k2, k3, k4))


def advance(p: Params, s, dt: float, reg: int, events: list, depth: int = 0):
    s1 = rk4(p, s, dt, reg)
    r1 = region(p, s1[0])
    if r1 == reg or depth >= 5:
        return s1, r1
    lo, hi = 0.0, dt
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
    events: list = []
    s_new, reg_new = advance(p, s, dt, reg, events)
    outcome: Optional[str] = None
    frac = 1.0

    def lerp(f):
        return tuple(a + f * (b - a) for a, b in zip(s, s_new))

    half_gap_x = p.dx_plate / 2.0
    half_gap_y = p.d / 2.0

    if reg_new == 1 and abs(s_new[2]) >= half_gap_x:
        a0, a1 = abs(s[2]), abs(s_new[2])
        frac = min(1.0, max(0.0, (half_gap_x - a0) / (a1 - a0))) if a1 > a0 else 1.0
        s_new, outcome = lerp(frac), "plate_x"
    elif reg_new == 3 and abs(s_new[1]) >= half_gap_y:
        a0, a1 = abs(s[1]), abs(s_new[1])
        frac = min(1.0, max(0.0, (half_gap_y - a0) / (a1 - a0))) if a1 > a0 else 1.0
        s_new, outcome = lerp(frac), "plate_y"
    elif s_new[0] >= p.z_wafer:
        frac = (p.z_wafer - s[0]) / (s_new[0] - s[0])
        s_new, outcome = lerp(frac), "wafer"
    elif s_new[3] <= 0.0:
        outcome = "miss"
    return s_new, reg_new, events, outcome, frac


# --------------------------------------------------------------------------- #
#  Forecast
# --------------------------------------------------------------------------- #
@dataclass
class Forecast:
    z: List[float]
    y: List[float]
    x: List[float]
    outcome: str
    z_hit: float
    y_hit: float
    x_hit: float
    t_total: float
    theta_exit_y: Optional[float]
    theta_exit_x: Optional[float]
    y_exit: Optional[float]
    x_exit: Optional[float]


def compute_forecast(p: Params) -> Forecast:
    s = (Z_START, 0.0, 0.0, p.v0, 0.0, 0.0)
    reg = region(p, s[0])
    dt = p.t_ref / 700.0
    zs, ys, xs = [s[0]], [s[1]], [s[2]]
    t = 0.0
    theta_exit_y, theta_exit_x, y_exit, x_exit, outcome = None, None, None, None, None
    for _ in range(4000):
        s_new, reg_new, events, outcome, frac = micro_step(p, s, reg, dt)
        for r_from, r_to, sb in events:
            if r_from == 1 and r_to == 2 and theta_exit_x is None:
                theta_exit_x, x_exit = math.atan2(sb[5], sb[3]), sb[2]
            if r_from == 3 and r_to == 4 and theta_exit_y is None:
                theta_exit_y, y_exit = math.atan2(sb[4], sb[3]), sb[1]
        t += frac * dt
        zs.append(s_new[0])
        ys.append(s_new[1])
        xs.append(s_new[2])
        s, reg = s_new, reg_new
        if outcome:
            break
    return Forecast(zs, ys, xs, outcome or "miss", s[0], s[1], s[2], t,
                    theta_exit_y, theta_exit_x, y_exit, x_exit)


# --------------------------------------------------------------------------- #
#  Live Ion
# --------------------------------------------------------------------------- #
class Ion:
    def __init__(self, p: Params):
        self.reset(p)

    def reset(self, p: Params):
        self.s = (Z_START, 0.0, 0.0, p.v0, 0.0, 0.0)
        self.reg = region(p, Z_START)
        self.t = 0.0
        self.trail_z = [Z_START]
        self.trail_y = [0.0]
        self.trail_x = [0.0]
        self.outcome: Optional[str] = None
        self.theta_exit_y: Optional[float] = None
        self.theta_exit_x: Optional[float] = None
        self.y_exit: Optional[float] = None
        self.x_exit: Optional[float] = None

    def advance(self, p: Params, dt: float, n_sub: int = 4):
        h = dt / n_sub
        for _ in range(n_sub):
            if self.outcome:
                break
            s_new, reg_new, events, outcome, frac = micro_step(p, self.s, self.reg, h)
            for r_from, r_to, sb in events:
                if r_from == 1 and r_to == 2 and self.theta_exit_x is None:
                    self.theta_exit_x, self.x_exit = math.atan2(sb[5], sb[3]), sb[2]
                if r_from == 3 and r_to == 4 and self.theta_exit_y is None:
                    self.theta_exit_y, self.y_exit = math.atan2(sb[4], sb[3]), sb[1]
            self.t += frac * h
            self.s, self.reg, self.outcome = s_new, reg_new, outcome
            self.trail_z.append(s_new[0])
            self.trail_y.append(s_new[1])
            self.trail_x.append(s_new[2])


# --------------------------------------------------------------------------- #
#  Text Input Definitions: key, label, initial_value, column, row
# --------------------------------------------------------------------------- #
TEXT_INPUTS = [
    ("V0_kV", r"$V_0$ (kV):", "20.0", 0, 0),
    ("Vd", r"$V_d$ (Y) (V):", "1000.0", 0, 1),
    ("Vdx", r"$V_{dx}$ (X) (V):", "1000.0", 0, 2),
    ("q_e", r"$q$ (×e):", "1.0", 1, 0),
    ("m_u", r"$m$ (u):", "31.0", 1, 1),
    ("lx", r"$l_x$ (m):", "0.08", 1, 2),
    ("ly", r"$l_y$ (m):", "0.10", 2, 0),
    ("dx_plate", r"$d_x$ (m):", "0.04", 2, 1),
    ("d", r"$d_y$ (m):", "0.04", 2, 2),
    ("L", r"$L$ (m):", "0.50", 3, 0),
    ("Bx_mT", r"$B_x$ (mT):", "0.0", 3, 1),
    ("speed", "speed (×):", "1.0", 3, 2),
]


# --------------------------------------------------------------------------- #
#  Application
# --------------------------------------------------------------------------- #
class App:
    FRAME_MS = 16          # Strict ~60 Hz tick interval (~16.67 ms, set to 16ms)
    FRAMES_PER_FLIGHT = 330 # Doubled frames to match the 60 Hz rate for equal flight duration
    STEPS_PER_FLIGHT = 50

    def __init__(self):
        self.p = Params()
        self.fc = compute_forecast(self.p)
        self.ion = Ion(self.p)
        self.running = False
        self.scene = []
        self._build_figure()
        self.rebuild_scene()
        self.refresh()
        self.timer = self.fig.canvas.new_timer(interval=self.FRAME_MS)
        self.timer.add_callback(self.tick)
        self.timer.start()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 10.5))
        try:
            self.fig.canvas.manager.set_window_title("Ion beam deflection simulator (60 Hz)")
        except Exception:
            pass
        self.fig.suptitle("Ion-beam deflection simulator — X & Y scan plates and wafer impact (60 Hz)",
                          fontsize=13, fontweight="bold", y=0.98)

        # 1. Main plot (y-z view) -------------------------------------------
        self.ax = self.fig.add_axes([0.05, 0.54, 0.58, 0.39])
        self.ax.set_title("main: Y-Z Beam Deflection View", fontsize=10, fontweight="bold")
        self.ax.set_xlabel("z (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.grid(alpha=0.25)
        (self.ln_fc,) = self.ax.plot([], [], "--", color="#ff7f0e", lw=1.8, zorder=4, label="forecast")
        (self.ln_trail,) = self.ax.plot([], [], "-", color="#1f77b4", lw=2.6, zorder=5, label="path so far")
        (self.pt_ion,) = self.ax.plot([], [], "o", ms=9, color="#1f77b4", mec="k", zorder=7)
        (self.pt_fc,) = self.ax.plot([], [], "o", ms=10, mfc="none", mec="#d62728", mew=2, zorder=6)
        (self.pt_hit,) = self.ax.plot([], [], "X", ms=10, color="#d62728", mec="k", zorder=8)
        self.ion_lbl = self.ax.annotate("ion", xy=(Z_START, 0), xytext=(6, 10),
                                   textcoords="offset points", fontsize=8.5, zorder=9)
        self.ax.legend(loc="upper right", fontsize=8, frameon=True)

        # 2. X plates window ('x') ------------------------------------------
        self.ax_x = self.fig.add_axes([0.05, 0.28, 0.27, 0.20])
        self.ax_x.set_title("x: X-Scan Plates View (X-Z)", fontsize=9, fontweight="bold")
        self.ax_x.set_xlabel("z (m)")
        self.ax_x.set_ylabel("x (m)")
        self.ax_x.grid(alpha=0.25)
        (self.ln_x_fc,) = self.ax_x.plot([], [], "--", color="#ff7f0e", lw=1.5)
        (self.ln_x_trail,) = self.ax_x.plot([], [], "-", color="#2ca02c", lw=2.0)
        (self.pt_x_ion,) = self.ax_x.plot([], [], "o", ms=8, color="#2ca02c", mec="k")

        # 3. Wafer window ('wafer') -----------------------------------------
        self.ax_wafer = self.fig.add_axes([0.36, 0.28, 0.27, 0.20])
        self.ax_wafer.set_title("wafer: Target Impact (X-Y)", fontsize=9, fontweight="bold")
        self.ax_wafer.set_xlabel("x (m)")
        self.ax_wafer.set_ylabel("y (m)")
        self.ax_wafer.grid(alpha=0.25)
        self.ax_wafer.set_aspect("equal")
        (self.pt_wafer_fc,) = self.ax_wafer.plot([], [], "o", ms=10, mfc="none", mec="#d62728", mew=2, label="forecast")
        (self.pt_wafer_hit,) = self.ax_wafer.plot([], [], "X", ms=10, color="#d62728", mec="k", label="simulated")
        self.ax_wafer.legend(loc="upper right", fontsize=8)

        # 4. Results panel --------------------------------------------------
        self.axi = self.fig.add_axes([0.65, 0.54, 0.32, 0.39])
        self.axi.set_xticks([])
        self.axi.set_yticks([])
        self.axi.set_facecolor("#f5f6fa")
        self.info = self.axi.text(0.03, 0.985, "", va="top", ha="left", family="monospace",
                                  fontsize=8.3, transform=self.axi.transAxes)

        # 5. Buttons --------------------------------------------------------
        bw, bh, by = 0.075, 0.04, 0.225
        self.b_run = Button(self.fig.add_axes([0.65, by, bw, bh]), "Start", color="#b7e4c7")
        self.b_step = Button(self.fig.add_axes([0.735, by, bw, bh]), "Step", color="#ffe8a3")
        self.b_reset = Button(self.fig.add_axes([0.82, by, bw, bh]), "Reset", color="#ffc9c9")
        self.b_def = Button(self.fig.add_axes([0.905, by, bw, bh]), "Defaults", color="#dfe3f7")
        for b in (self.b_run, self.b_step, self.b_reset, self.b_def):
            b.label.set_fontsize(10)
        self.b_run.on_clicked(self.on_run)
        self.b_step.on_clicked(self.on_step)
        self.b_reset.on_clicked(self.on_reset)
        self.b_def.on_clicked(self.on_defaults)

        # Check-box for L reference -----------------------------------------
        ax_chk = self.fig.add_axes([0.65, 0.175, 0.32, 0.045])
        ax_chk.set_frame_on(False)
        self.chk = CheckButtons(ax_chk, ["measure L from Y-plate centre"], [False])
        self.chk.labels[0].set_fontsize(9)
        self.chk.on_clicked(self.on_param)

        # 6. Text inputs control panel --------------------------------------
        self.fig.text(0.05, 0.245, "CONTROL PANEL (type value & press Enter)", fontsize=10, fontweight="bold")
        self.txt_boxes = {}
        col_x = [0.10, 0.25, 0.40, 0.53]
        row_y = [0.18, 0.12, 0.06]
        for key, label, vinit, col, row in TEXT_INPUTS:
            tax = self.fig.add_axes([col_x[col], row_y[row], 0.07, 0.035])
            txt_box = TextBox(tax, label, initial=vinit)
            txt_box.label.set_fontsize(9)
            txt_box.text_disp.set_fontsize(9)
            txt_box.on_submit(self.on_param)
            self.txt_boxes[key] = txt_box

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def rebuild_scene(self):
        for a in self.scene:
            try:
                a.remove()
            except Exception:
                pass
        self.scene = []
        add = self.scene.append
        ax, ax_x, ax_w, p, fc = self.ax, self.ax_x, self.ax_wafer, self.p, self.fc

        ymax_path = max((abs(v) for v in fc.y), default=0.0)
        yl = max(0.06, 0.9 * p.d, 1.35 * ymax_path)
        zw = p.z_wafer

        ax.set_xlim(Z_START - 0.05, zw + 0.05)
        ax.set_ylim(-yl, yl)
        th = 0.07 * yl
        hh = 0.78 * yl

        # Main view: acceleration tube
        add(ax.add_patch(Rectangle((Z_START - 0.06, -hh), 0.06, 2 * hh, fc="#c46b4d", ec="#5c2e1e", lw=1.0, zorder=1)))
        # X-plates in main view (indicated)
        add(ax.add_patch(Rectangle((0, -hh * 0.8), p.lx, 2 * hh * 0.8, fc="#e6f2ff", ec="#2a6fdb", lw=1.0, ls="--", zorder=1)))
        add(ax.text(p.lx / 2, 0, "X-plates\n(hidden y)", color="#2a6fdb", ha="center", va="center", fontsize=8, zorder=2))
        # Y-plates in main view
        add(ax.add_patch(Rectangle((p.z_y_start, p.d / 2), p.ly, th, fc="#d62728", ec="k", lw=0.8, zorder=2)))
        add(ax.add_patch(Rectangle((p.z_y_start, -p.d / 2 - th), p.ly, th, fc="#1f77b4", ec="k", lw=0.8, zorder=2)))
        # Target wafer in main view
        add(ax.add_patch(Rectangle((zw - 0.001, -0.015), 0.002, 0.03, fc="#d9dcf2", ec="#444", lw=1.2, zorder=1)))
        add(ax.plot([Z_START, zw], [0, 0], "--", color="#444", lw=0.8, zorder=1)[0])

        # X window view
        xmax_path = max((abs(v) for v in fc.x), default=0.0)
        xl = max(0.04, 0.9 * p.dx_plate, 1.4 * xmax_path)
        ax_x.set_xlim(-0.02, p.lx + 0.05)
        ax_x.set_ylim(-xl, xl)
        th_x = 0.07 * xl
        ax_x.add_patch(Rectangle((0, p.dx_plate / 2), p.lx, th_x, fc="#d62728", ec="k", lw=0.8))
        ax_x.add_patch(Rectangle((0, -p.dx_plate / 2 - th_x), p.lx, th_x, fc="#1f77b4", ec="k", lw=0.8))
        ax_x.plot([0, p.lx + 0.03], [0, 0], "--", color="#444", lw=0.8)

        # Wafer window view (X-Y)
        w_lim = max(0.02, 1.5 * max(abs(fc.x_hit), abs(fc.y_hit), 0.01))
        self.ax_wafer.set_xlim(-w_lim, w_lim)
        self.ax_wafer.set_ylim(-w_lim, w_lim)
        wafer_circle = Circle((0, 0), w_lim * 0.85, fc="#eef2f7", ec="#333", lw=1.5, zorder=1)
        self.ax_wafer.add_patch(wafer_circle)
        self.ax_wafer.axhline(0, color="#ccc", lw=0.8, ls="--")
        self.ax_wafer.axvline(0, color="#ccc", lw=0.8, ls="--")

    def refresh(self):
        p, fc, ion = self.p, self.fc, self.ion
        self.ln_fc.set_data(fc.z, fc.y)
        self.ln_trail.set_data(ion.trail_z, ion.trail_y)
        self.pt_ion.set_data([ion.s[0]], [ion.s[1]])
        self.ion_lbl.xy = (ion.s[0], ion.s[1])

        self.ln_x_fc.set_data(fc.z[:len(fc.x)], fc.x)
        self.ln_x_trail.set_data(ion.trail_z, ion.trail_x)
        self.pt_x_ion.set_data([ion.s[0]], [ion.s[2]])

        if fc.outcome in ("wafer", "plate_y", "plate_x"):
            self.pt_fc.set_data([fc.z_hit], [fc.y_hit])
            self.pt_wafer_fc.set_data([fc.x_hit], [fc.y_hit])
        else:
            self.pt_fc.set_data([], [])
            self.pt_wafer_fc.set_data([], [])

        if ion.outcome in ("wafer", "plate_y", "plate_x"):
            self.pt_hit.set_data([ion.s[0]], [ion.s[1]])
            self.pt_wafer_hit.set_data([ion.s[2]], [ion.s[1]])
        else:
            self.pt_hit.set_data([], [])
            self.pt_wafer_hit.set_data([], [])

        self.info.set_text(self.info_text())
        self.b_run.label.set_text("Stop" if self.running else "Start")
        self.fig.canvas.draw()

    def info_text(self) -> str:
        p, fc, ion = self.p, self.fc, self.ion
        L_ = []
        L_.append("BEAM & VOLTAGES (60 Hz)")
        L_.append(f" v0  = {p.v0:.3e} m/s  E_k = {p.q_e * p.V0_kV:.1f} keV")
        L_.append(f" V_dx = {p.Vdx:+.0f} V | V_d = {p.Vd:+.0f} V")
        L_.append("")
        L_.append("SIMULATION STATUS")
        status = {"wafer": "reached wafer", "plate_y": "HIT Y PLATE",
                  "plate_x": "HIT X PLATE", "miss": "turned back"}.get(ion.outcome or "")
        if status is None:
            status = "running" if self.running else ("ready" if ion.t == 0 else "paused")
        L_.append(f" status = {status}")
        L_.append(f" t = {ion.t * 1e6:.4f} µs")
        L_.append(f" z = {ion.s[0]:+.4f} m")
        L_.append(f" y = {ion.s[1]:+.4f} m ({ion.s[1]*1e3:+.2f} mm)")
        L_.append(f" x = {ion.s[2]:+.4f} m ({ion.s[2]*1e3:+.2f} mm)")
        L_.append("")
        L_.append("WAFER IMPACT POINT (Forecast)")
        if fc.outcome == "wafer":
            L_.append(f" x = {fc.x_hit * 1e3:+.2f} mm")
            L_.append(f" y = {fc.y_hit * 1e3:+.2f} mm")
        else:
            L_.append(f" outcome: {fc.outcome}")
        return "\n".join(L_)

    def read_params(self):
        v = {}
        for k, tb in self.txt_boxes.items():
            try:
                v[k] = float(tb.text)
            except ValueError:
                v[k] = getattr(self.p, k, 1.0)
        self.p = Params(V0_kV=v["V0_kV"], Vd=v["Vd"], Vdx=v["Vdx"], lx=v["lx"], ly=v["ly"],
                        dx_plate=v["dx_plate"], d=v["d"], L=v["L"], q_e=v["q_e"],
                        m_u=v["m_u"], Bx_mT=v["Bx_mT"], speed=v["speed"],
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
        for key, _, vinit, _, _ in TEXT_INPUTS:
            if key in self.txt_boxes:
                self.txt_boxes[key].set_val(vinit)
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
        dt = self.p.t_ref / self.FRAMES_PER_FLIGHT * self.p.speed
        self.ion.advance(self.p, dt)
        if self.ion.outcome:
            self.running = False
        self.refresh()


def main():
    app = App()
    plt.show()
    return app


if __name__ == "__main__":
    main()