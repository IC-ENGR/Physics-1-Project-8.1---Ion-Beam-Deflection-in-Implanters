import math
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import Button

# state vector of the ion:  s = (z, x, y, vz, vx, vy)
Z, X, Y, VZ, VX, VY = range(6)

# regions of the beam line
R_PRE, R_XPL, R_GAP, R_YPL, R_POST = range(5)
HITS = ("wafer", "plate_x", "plate_y")


# --------------------------------------------------------------------------- #
#  Parameters
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    V0_kV: float = 20.0
    Vd: float = 1000.0
    l: float = 0.10
    d: float = 0.04
    Vx: float = 500.0
    lx: float = 0.10
    dx: float = 0.02
    gap_xy: float = 0.05
    L: float = 0.50
    wafer_mm: float = 150.0
    q_e: float = 1.0
    m_u: float = 31.0
    L_from_centre: bool = False

    E_CHARGE: float = 1.602176634e-19
    AMU: float = 1.66053906660e-27
    TUBE_LEN: float = 0.08
    TUBE_GAP: float = 0.08

    @property
    def V0(self) -> float:
        return self.V0_kV * 1e3

    @property
    def q(self) -> float:
        return self.q_e * self.E_CHARGE

    @property
    def m(self) -> float:
        return self.m_u * self.AMU

    @property
    def v0(self) -> float:
        return math.sqrt(2.0 * self.q * self.V0 / self.m)

    @property
    def wafer_R(self) -> float:
        return 0.5e-3 * self.wafer_mm

    @property
    def zx0(self) -> float:
        return -(self.gap_xy + self.lx)

    @property
    def zx1(self) -> float:
        return -self.gap_xy

    @property
    def zcx(self) -> float:
        return 0.5 * (self.zx0 + self.zx1)

    @property
    def z_start(self) -> float:
        return self.zx0 - self.TUBE_GAP

    @property
    def z_wafer(self) -> float:
        return self.L + (self.l / 2.0 if self.L_from_centre else 0.0)

    @property
    def Lx(self) -> float:
        return self.z_wafer - self.zcx

    @property
    def tan_theta(self) -> float:
        return self.Vd * self.l / (2.0 * self.V0 * self.d)

    @property
    def tan_theta_x(self) -> float:
        return self.Vx * self.lx / (2.0 * self.V0 * self.dx)

    @property
    def t_ref(self) -> float:
        return (self.z_wafer - self.z_start) / self.v0


# --------------------------------------------------------------------------- #
#  Physics Engine
# --------------------------------------------------------------------------- #
def region(p: Params, z: float) -> int:
    if z < p.zx0: return R_PRE
    if z <= p.zx1: return R_XPL
    if z < 0.0: return R_GAP
    if z <= p.l: return R_YPL
    return R_POST


def derivs(p: Params, s, reg: int):
    _, _, _, vz, vx, vy = s
    k = p.q / p.m
    ex = p.Vx / p.dx if reg == R_XPL else 0.0
    ey = p.Vd / p.d if reg == R_YPL else 0.0
    a_x = k * ex
    a_y = k * ey
    a_z = 0.0
    return (vz, vx, vy, a_z, a_x, a_y)


def rk4(p: Params, s, dt: float, reg: int):
    k1 = derivs(p, s, reg)
    k2 = derivs(p, [a + 0.5 * dt * b for a, b in zip(s, k1)], reg)
    k3 = derivs(p, [a + 0.5 * dt * b for a, b in zip(s, k2)], reg)
    k4 = derivs(p, [a + dt * b for a, b in zip(s, k3)], reg)
    return tuple(a + dt / 6.0 * (b + 2 * c + 2 * e + f)
                 for a, b, c, e, f in zip(s, k1, k2, k3, k4))


def advance(p: Params, s, dt: float, reg: int, events: list, depth: int = 0):
    s1 = rk4(p, s, dt, reg)
    r1 = region(p, s1[Z])
    if r1 == reg or depth >= 8:
        return s1, r1
    lo, hi = 0.0, dt
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if region(p, rk4(p, s, mid, reg)[Z]) == reg:
            lo = mid
        else:
            hi = mid
    sb = rk4(p, s, hi, reg)
    rb = region(p, sb[Z])
    events.append((reg, rb, sb))
    return advance(p, sb, dt - hi, rb, events, depth + 1)


def micro_step(p: Params, s, reg: int, dt: float):
    events: list = []
    s_new, reg_new = advance(p, s, dt, reg, events)
    outcome: Optional[str] = None
    frac = 1.0

    def lerp(f):
        return tuple(a + f * (b - a) for a, b in zip(s, s_new))

    def hit_fraction(idx, half_gap):
        a0, a1 = abs(s[idx]), abs(s_new[idx])
        return min(1.0, max(0.0, (half_gap - a0) / (a1 - a0))) if a1 > a0 else 1.0

    if reg_new == R_XPL and abs(s_new[X]) >= p.dx / 2.0:
        frac = hit_fraction(X, p.dx / 2.0)
        s_new, outcome = lerp(frac), "plate_x"
    elif reg_new == R_YPL and abs(s_new[Y]) >= p.d / 2.0:
        frac = hit_fraction(Y, p.d / 2.0)
        s_new, outcome = lerp(frac), "plate_y"
    elif s_new[Z] >= p.z_wafer:
        frac = (p.z_wafer - s[Z]) / (s_new[Z] - s[Z])
        s_new, outcome = lerp(frac), "wafer"
    elif s_new[VZ] <= 0.0:
        outcome = "miss"
    return s_new, reg_new, events, outcome, frac


@dataclass
class PlateExit:
    z: float
    pos: float
    theta: float


def note_exits(events, exits: Dict[str, PlateExit]):
    for r_from, r_to, sb in events:
        if r_to <= r_from: continue
        if r_from == R_XPL and "x" not in exits:
            exits["x"] = PlateExit(sb[Z], sb[X], math.atan2(sb[VX], sb[VZ]))
        elif r_from == R_YPL and "y" not in exits:
            exits["y"] = PlateExit(sb[Z], sb[Y], math.atan2(sb[VY], sb[VZ]))


# --------------------------------------------------------------------------- #
#  Live Ion Object
# --------------------------------------------------------------------------- #
class Ion:
    def __init__(self, p: Params):
        self.reset(p)

    def reset(self, p: Params):
        self.s = (p.z_start, 0.0, 0.0, p.v0, 0.0, 0.0)
        self.reg = region(p, p.z_start)
        self.t = 0.0
        self.trail_z = [p.z_start]
        self.trail_x = [0.0]
        self.trail_y = [0.0]
        self.outcome: Optional[str] = None
        self.exits: Dict[str, PlateExit] = {}

    def advance(self, p: Params, dt: float, n_sub: int = 4):
        h = dt / n_sub
        for _ in range(n_sub):
            if self.outcome: break
            s_new, reg_new, events, outcome, frac = micro_step(p, self.s, self.reg, h)
            note_exits(events, self.exits)
            self.t += frac * h
            self.s, self.reg, self.outcome = s_new, reg_new, outcome
            self.trail_z.append(s_new[Z])
            self.trail_x.append(s_new[X])
            self.trail_y.append(s_new[Y])


# --------------------------------------------------------------------------- #
#  Application
# --------------------------------------------------------------------------- #
class App:
    FRAME_MS = 0.01
    FRAMES_PER_FLIGHT = 50
    STEPS_PER_FLIGHT = 500

    def __init__(self):
        self.p = self.load_params_from_json()
        self.ion = Ion(self.p)
        self.running = False
        self.scene = []
        self.yl = 0.06
        self._build_figure()
        self.rebuild_scene()
        self.refresh()
        self.timer = self.fig.canvas.new_timer(interval=self.FRAME_MS)
        self.timer.add_callback(self.tick)
        self.timer.start()

    def load_params_from_json(self) -> Params:
        with open("data.json", "r") as f:
            data = json.load(f)
            # Merge constants and params into a single dict
            combined = {**data.get("constants", {}), **data.get("params", {})}
            return Params(**combined)

    def _build_figure(self):
        self.fig = plt.figure(figsize=(15, 9.6))
        self.fig.suptitle("Ion-beam scanning simulator (Data-driven)", fontsize=13, fontweight="bold", y=0.99)

        # Main window
        self.ax = self.fig.add_axes([0.055, 0.585, 0.62, 0.335])
        self.ax.set_xlabel("z (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.grid(alpha=0.25)
        (self.ln_trail,) = self.ax.plot([], [], "-", color="#1f77b4", lw=2.6, label="ion path")
        (self.pt_ion,) = self.ax.plot([], [], "o", ms=11, color="#1f77b4", mec="k", zorder=7)
        (self.pt_hit,) = self.ax.plot([], [], "X", ms=11, color="#d62728", mec="k", zorder=8)
        self.ion_lbl = self.ax.annotate("3: ion (m, q)", xy=(0, 0), xytext=(8, 12), textcoords="offset points",
                                        fontsize=9, zorder=9)

        # X-view
        self.axx = self.fig.add_axes([0.055, 0.285, 0.275, 0.215])
        self.axx.set_title("2x: x-scan plates (x-z view)", fontsize=10, fontweight="bold")
        (self.ln_trail_x,) = self.axx.plot([], [], "-", color="#1f77b4", lw=2.6)
        (self.pt_ion_x,) = self.axx.plot([], [], "o", ms=9, color="#1f77b4", mec="k")
        (self.pt_hit_x,) = self.axx.plot([], [], "X", ms=10, color="#d62728", mec="k")

        # Wafer-view
        fw, fh = self.fig.get_size_inches()
        side = 0.215 * fh / fw
        self.axw = self.fig.add_axes([0.385, 0.285, side, 0.215])
        self.axw.set_aspect("equal", adjustable="box")
        self.axw.set_title("5: target wafer (x-y plane)", fontsize=10, fontweight="bold")
        (self.ln_trail_w,) = self.axw.plot([], [], "-", color="#1f77b4", lw=2.2)
        (self.pt_ion_w,) = self.axw.plot([], [], "o", ms=8, color="#1f77b4", mec="k")
        (self.pt_hit_w,) = self.axw.plot([], [], "X", ms=11, color="#d62728", mec="k")

        # Readout
        x_r = 0.385 + side + 0.022
        self.axr = self.fig.add_axes([x_r, 0.285, 0.675 - x_r, 0.215])
        self.axr.set_xticks([]);
        self.axr.set_yticks([])
        self.axr.set_facecolor("#f5f6fa")
        kw = dict(va="top", ha="left", family="monospace", fontsize=8.5, transform=self.axr.transAxes)
        self.axr.text(0.06, 0.95, "SIMULATED IMPACT", fontweight="bold", **kw)
        self.wr_sim = self.axr.text(0.06, 0.42, "", **kw)

        # Results panel
        self.axi = self.fig.add_axes([0.70, 0.285, 0.29, 0.635])
        self.axi.set_xticks([]);
        self.axi.set_yticks([])
        self.axi.set_facecolor("#f5f6fa")
        self.info = self.axi.text(0.03, 0.985, "", va="top", ha="left", family="monospace", fontsize=8.7,
                                  transform=self.axi.transAxes)

        # Buttons
        bw, bh, by = 0.075, 0.04, 0.185
        self.b_run = Button(self.fig.add_axes([0.055, by, bw, bh]), "Start", color="#b7e4c7")
        self.b_reset = Button(self.fig.add_axes([0.225, by, bw, bh]), "Reset", color="#ffc9c9")
        self.b_run.on_clicked(self.on_run)
        self.b_reset.on_clicked(self.on_reset)

    def rebuild_scene(self):
        for a in self.scene:
            try:
                a.remove()
            except:
                pass
        self.scene = []
        self._scene_main()
        self._scene_xz()
        self._scene_wafer()

    def _scene_main(self):
        ax, p = self.ax, self.p
        add = self.scene.append
        ymax_path = max((abs(v) for v in [0.0]), default=0.0)
        yl = max(0.06, 0.9 * p.d, 1.15 * p.wafer_R)
        yl = math.ceil(yl * 100 - 1e-9) / 100.0
        self.yl = yl
        zw, zs = p.z_wafer, p.z_start
        ax.set_xlim(zs - 0.10, zw + 0.10)
        ax.set_ylim(-yl, yl)
        th, hh = 0.07 * yl, 0.78 * yl
        top = ax.get_xaxis_transform()

        add(ax.add_patch(
            Rectangle((zs - Params.TUBE_LEN, -hh), Params.TUBE_LEN, 2 * hh, fc="#c46b4d", ec="#5c2e1e", lw=1.2,
                      zorder=1)))
        add(ax.text(zs - Params.TUBE_LEN / 2, 0, rf"$V_0$ = {p.V0_kV:g} kV", rotation=90, color="white", ha="center",
                    va="center", fontsize=11, fontweight="bold", zorder=3))
        add(ax.add_patch(
            Rectangle((zw - 0.001, -p.wafer_R), 0.002, 2 * p.wafer_R, fc="#d9dcf2", ec="#444", lw=1.2, zorder=1)))

        neg, pos, grey = "#d62728", "#1f77b4", "#999999"
        wy = p.d / 2 + th
        cx, sx = (pos, "+") if p.Vx > 0 else (neg, "−" if p.Vx < 0 else (grey, ""))
        add(ax.add_patch(Rectangle((p.zx0, -wy), p.lx, 2 * wy, fc=to_rgba(cx, 0.28), ec="k", lw=0.8, zorder=2)))

        if p.Vd > 0:
            ctop, cbot, stop, sbot = neg, pos, "−", "+"
        elif p.Vd < 0:
            ctop, cbot, stop, sbot = pos, neg, "+", "−"
        else:
            ctop = cbot = grey; stop = sbot = ""
        add(ax.add_patch(Rectangle((0, p.d / 2), p.l, th, fc=ctop, ec="k", lw=0.8, zorder=2)))
        add(ax.add_patch(Rectangle((0, -p.d / 2 - th), p.l, th, fc=cbot, ec="k", lw=0.8, zorder=2)))

        add(ax.plot([zs, zw], [0, 0], "--", color="#444", lw=0.9, zorder=1)[0])

    def _scene_xz(self):
        ax, p = self.axx, self.p
        add = self.scene.append
        z0, z1, lx, dx = p.zx0, p.zx1, p.lx, p.dx
        zl, zr = z0 - 0.30 * lx, z1 + 0.22 * lx
        xl = 0.95 * dx
        ax.set_xlim(zl, zr)
        ax.set_ylim(-xl, xl)
        th = 0.10 * dx
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        neg, pos, grey = "#d62728", "#1f77b4", "#999999"
        if p.Vx > 0:
            ctop, cbot, stop, sbot = neg, pos, "−", "+"
        elif p.Vx < 0:
            ctop, cbot, stop, sbot = pos, neg, "+", "−"
        else:
            ctop = cbot = grey; stop = sbot = ""
        add(ax.add_patch(Rectangle((z0, dx / 2), lx, th, fc=ctop, ec="k", lw=0.8, zorder=2)))
        add(ax.add_patch(Rectangle((z0, -dx / 2 - th), lx, th, fc=cbot, ec="k", lw=0.8, zorder=2)))

    def _scene_wafer(self):
        ax, p = self.axw, self.p
        add = self.scene.append
        R = p.wafer_R * 1e3
        lim = 1.3 * R
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        add(ax.add_patch(Circle((0, 0), R, fc="#d9dcf2", ec="#444", lw=1.5, zorder=1)))
        add(ax.plot([-lim, lim], [0, 0], "-", color="#8a8a8a", lw=0.7, zorder=2)[0])
        add(ax.plot([0, 0], [-lim, lim], "-", color="#8a8a8a", lw=0.7, zorder=2)[0])

    def refresh(self):
        p, ion = self.p, self.ion
        mm = 1e3
        self.ln_trail.set_data(ion.trail_z, ion.trail_y)
        self.pt_ion.set_data([ion.s[Z]], [ion.s[Y]])
        self.ion_lbl.xy = (ion.s[Z], ion.s[Y])

        if ion.outcome in HITS:
            self.pt_hit.set_data([ion.s[Z]], [ion.s[Y]])
        else:
            self.pt_hit.set_data([], [])

        self.ln_trail_x.set_data(ion.trail_z, ion.trail_x)
        self.pt_ion_x.set_data([ion.s[Z]], [ion.s[X]])
        if ion.outcome == "plate_x":
            self.pt_hit_x.set_data([ion.s[Z]], [ion.s[X]])
        else:
            self.pt_hit_x.set_data([], [])

        self.ln_trail_w.set_data([v * mm for v in ion.trail_x], [v * mm for v in ion.trail_y])
        self.pt_ion_w.set_data([ion.s[X] * mm], [ion.s[Y] * mm])
        if ion.outcome == "wafer":
            self.pt_hit_w.set_data([ion.s[X] * mm], [ion.s[Y] * mm])
        else:
            self.pt_hit_w.set_data([], [])

        # Text Output
        if ion.outcome == "wafer":
            r = math.hypot(ion.s[X], ion.s[Y])
            ok = r <= p.wafer_R
            txt = f"IMPACT\nx = {ion.s[X] * 1e3:+.2f} mm\ny = {ion.s[Y] * 1e3:+.2f} mm\nr = {r * 1e3:.2f} mm\n-> " + (
                "ON" if ok else "OFF")
            self.wr_sim.set_text(txt)
            self.wr_sim.set_color("#2b8a3e" if ok else "#d62728")
        elif ion.outcome in ("plate_x", "plate_y"):
            self.wr_sim.set_text(f"Stopped by {ion.outcome[-1]} plates")
            self.wr_sim.set_color("#d62728")
        else:
            self.wr_sim.set_text("In flight...")
            self.wr_sim.set_color("#666666")

        L_ = ["BEAM", f"v0 = {p.v0:.3e} m/s", f"E_x = {p.Vx / p.dx:+.3e} V/m", f"E_d = {p.Vd / p.d:+.3e} V/m"]
        L_.append("")
        L_.append("SIMULATION")
        if ion.outcome == "wafer":
            L_.append("status = hit wafer")
        else:
            status = {"plate_x": "hit x plate", "plate_y": "hit y plate", "miss": "turned back"}.get(ion.outcome or "")
            L_.append(f"status = {status}")
        L_.append(f"t = {ion.t * 1e6:.4f} µs")
        L_.append(f"z = {ion.s[Z]:+.5f} m")
        L_.append(f"x = {ion.s[X]:+.5f} m")
        L_.append(f"y = {ion.s[Y]:+.5f} m")
        self.info.set_text("\n".join(L_))

        self.b_run.label.set_text("Stop" if self.running else "Start")
        self.fig.canvas.draw_idle()

    def on_run(self, _=None):
        if self.ion.outcome:
            self.ion.reset(self.p)
        self.running = not self.running
        self.refresh()

    def on_reset(self, _=None):
        self.running = False
        self.ion.reset(self.p)
        self.refresh()

    def tick(self):
        if not self.running:
            return
        dt = self.p.t_ref / self.FRAMES_PER_FLIGHT
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