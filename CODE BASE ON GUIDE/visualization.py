"""Output generation: CSV tables, the four mandatory figures and an optional animated replay.

The figures are drawn with matplotlib's object-oriented Figure class, so they can
be produced without opening a window (works on servers and in batch runs).
Every figure has a title, axis labels with units, a legend and grid lines.
"""

import math
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

NEG, POS, GREY = "#d62728", "#1f77b4", "#999999"     # plate colours: red = "-", blue = "+"
PATH_COLOUR, HIT_COLOUR = "#1f77b4", "#d62728"


# --------------------------------------------------------------------------- #
#  CSV export
# --------------------------------------------------------------------------- #
def save_tables(result, directory, results_csv, conserved_csv):
    """Write simulation_results.csv (guideline schema) and the extra conserved-quantity table."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(directory / results_csv, index=False)
    result.conserved.to_csv(directory / conserved_csv, index=False)
    return directory / results_csv, directory / conserved_csv


# --------------------------------------------------------------------------- #
#  Beam-line drawings (shared by the saved figures and the replay window)
# --------------------------------------------------------------------------- #
def _plate_colours(voltage):
    """(colour of the plate at +coordinate, colour of the plate at -coordinate)."""
    if voltage > 0:
        return NEG, POS      # E points to +coordinate: the + plate is at the bottom
    if voltage < 0:
        return POS, NEG
    return GREY, GREY


def _half_height(bl):
    half = max(0.06, 0.9 * bl.d, 0.9 * bl.dx, 1.15 * bl.wafer_R)
    return math.ceil(half * 100 - 1e-9) / 100.0


def draw_side_view(ax, bl):
    """z-y view: tube, x plates (translucent), y plates, wafer edge-on."""
    half = _half_height(bl)
    th, hh = 0.07 * half, 0.78 * half
    ctop, cbot = _plate_colours(bl.Vd)
    ax.set_xlim(bl.z_start - bl.tube_len - 0.02, bl.z_wafer + 0.10)
    ax.set_ylim(-half, half)
    ax.add_patch(Rectangle((bl.z_start - bl.tube_len, -hh), bl.tube_len, 2 * hh,
                           fc="#c46b4d", ec="#5c2e1e", lw=1.2, zorder=1))
    ax.text(bl.z_start - bl.tube_len / 2, 0, rf"$V_0$ = {bl.V0 / 1e3:g} kV", rotation=90,
            color="white", ha="center", va="center", fontsize=10, fontweight="bold", zorder=3)
    wy = bl.d / 2 + th
    ax.add_patch(Rectangle((bl.zx0, -wy), bl.lx, 2 * wy,
                           fc=to_rgba(_plate_colours(bl.Vx)[0], 0.28), ec="k", lw=0.8, zorder=2))
    ax.add_patch(Rectangle((0, bl.d / 2), bl.l, th, fc=ctop, ec="k", lw=0.8, zorder=2))
    ax.add_patch(Rectangle((0, -bl.d / 2 - th), bl.l, th, fc=cbot, ec="k", lw=0.8, zorder=2))
    ax.add_patch(Rectangle((bl.z_wafer - 0.001, -bl.wafer_R), 0.002, 2 * bl.wafer_R,
                           fc="#d9dcf2", ec="#444", lw=1.2, zorder=1))
    ax.plot([bl.z_start, bl.z_wafer], [0, 0], "--", color="#444", lw=0.9, zorder=1)
    ax.set_xlabel("z (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25)


def draw_top_view(ax, bl):
    """z-x view: x plates drawn as plates, y plates seen from above (translucent)."""
    half = _half_height(bl)
    th, hh = 0.07 * half, 0.78 * half
    ctop, cbot = _plate_colours(bl.Vx)
    ax.set_xlim(bl.z_start - bl.tube_len - 0.02, bl.z_wafer + 0.10)
    ax.set_ylim(-half, half)
    ax.add_patch(Rectangle((bl.z_start - bl.tube_len, -hh), bl.tube_len, 2 * hh,
                           fc="#c46b4d", ec="#5c2e1e", lw=1.2, zorder=1))
    ax.add_patch(Rectangle((bl.zx0, bl.dx / 2), bl.lx, th, fc=ctop, ec="k", lw=0.8, zorder=2))
    ax.add_patch(Rectangle((bl.zx0, -bl.dx / 2 - th), bl.lx, th, fc=cbot, ec="k", lw=0.8, zorder=2))
    ax.add_patch(Rectangle((0, -hh), bl.l, 2 * hh,
                           fc=to_rgba(_plate_colours(bl.Vd)[0], 0.28), ec="k", lw=0.8, zorder=2))
    ax.add_patch(Rectangle((bl.z_wafer - 0.001, -bl.wafer_R), 0.002, 2 * bl.wafer_R,
                           fc="#d9dcf2", ec="#444", lw=1.2, zorder=1))
    ax.plot([bl.z_start, bl.z_wafer], [0, 0], "--", color="#444", lw=0.9, zorder=1)
    ax.set_xlabel("z (m)")
    ax.set_ylabel("x (m)")
    ax.grid(alpha=0.25)


def draw_wafer_view(ax, bl):
    """x-y view of the round wafer, axes in millimetres."""
    radius = bl.wafer_R * 1e3
    lim = 1.3 * radius
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.add_patch(Circle((0, 0), radius, fc="#d9dcf2", ec="#444", lw=1.5, zorder=1))
    ax.axhline(0, color="#8a8a8a", lw=0.7, zorder=2)
    ax.axvline(0, color="#8a8a8a", lw=0.7, zorder=2)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(alpha=0.25)


# --------------------------------------------------------------------------- #
#  The four mandatory figures
# --------------------------------------------------------------------------- #
def _impact_label(result, bl):
    f = result.final_state
    if result.outcome == "wafer":
        r = math.hypot(f["x"], f["y"])
        where = "ON the wafer" if r <= bl.wafer_R else "OFF the wafer"
        return f"wafer impact: x = {f['x'] * 1e3:+.2f} mm, y = {f['y'] * 1e3:+.2f} mm ({where})"
    names = {"plate_x": "stopped by an x plate", "plate_y": "stopped by a y plate",
             "timeout": "t_max reached before the wafer"}
    return names[result.outcome]


def plot_trajectory(result, bl, path, dpi):
    """1. Trajectory / spatial plot: side view, top view and wafer view."""
    t = result.table
    fig = Figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.32, wspace=0.22)
    ax_side = fig.add_subplot(gs[0, :])
    ax_top = fig.add_subplot(gs[1, 0])
    ax_waf = fig.add_subplot(gs[1, 1])
    draw_side_view(ax_side, bl)
    draw_top_view(ax_top, bl)
    draw_wafer_view(ax_waf, bl)
    z, x, y = t["pos_z"], t["pos_x"], t["pos_y"]
    ax_side.plot(z, y, color=PATH_COLOUR, lw=2.2, zorder=5, label="ion path")
    ax_top.plot(z, x, color=PATH_COLOUR, lw=2.2, zorder=5, label="ion path")
    ax_waf.plot(x * 1e3, y * 1e3, color=PATH_COLOUR, lw=2.0, zorder=5, label="projected path")
    for ax, u, v, scale in ((ax_side, z, y, 1.0), (ax_top, z, x, 1.0), (ax_waf, x, y, 1e3)):
        ax.plot(u.iloc[0] * scale, v.iloc[0] * scale, "o", ms=8, color="#2b8a3e", mec="k",
                zorder=7, label="start")
        ax.plot(u.iloc[-1] * scale, v.iloc[-1] * scale, "X", ms=11, color=HIT_COLOUR, mec="k",
                zorder=8, label="end of flight")
        ax.legend(loc="upper left", fontsize=8)
    ax_side.set_title("Side view (z-y): y plates deflect the beam up or down")
    ax_top.set_title("Top view (z-x): x plates deflect the beam sideways")
    ax_waf.set_title("Target wafer (x-y)")
    fig.suptitle(f"Ion-beam trajectory - {_impact_label(result, bl)}", fontweight="bold")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_time_series(result, path, dpi):
    """2. Time series of the state variables (positions and velocities)."""
    t = result.table
    time_us = t["time_t"] * 1e6
    fig = Figure(figsize=(12, 8))
    axs = fig.subplots(2, 2, sharex=True)
    axs[0, 0].plot(time_us, t["pos_x"] * 1e3, label="x")
    axs[0, 0].plot(time_us, t["pos_y"] * 1e3, label="y")
    axs[0, 0].set_ylabel("Transverse position (mm)")
    axs[0, 0].set_title("Transverse positions")
    axs[0, 1].plot(time_us, t["pos_z"], color="#2b8a3e", label="z")
    axs[0, 1].set_ylabel("Axial position z (m)")
    axs[0, 1].set_title("Position along the beam axis")
    axs[1, 0].plot(time_us, t["vel_x"], label=r"$v_x$")
    axs[1, 0].plot(time_us, t["vel_y"], label=r"$v_y$")
    axs[1, 0].set_ylabel("Transverse velocity (m/s)")
    axs[1, 0].set_title("Transverse velocities")
    axs[1, 1].plot(time_us, t["vel_z"], color="#2b8a3e", label=r"$v_z$")
    axs[1, 1].set_ylabel("Axial velocity (m/s)")
    axs[1, 1].set_title("Velocity along the beam axis")
    vz_mean = float(t["vel_z"].mean())
    axs[1, 1].set_ylim(0.99 * vz_mean, 1.01 * vz_mean)
    for ax in axs.flat:
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
    axs[1, 0].set_xlabel("Time (µs)")
    axs[1, 1].set_xlabel("Time (µs)")
    fig.suptitle("Time series of the state variables", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)


def plot_phase_space(result, path, dpi):
    """3. Phase-space diagrams: velocity versus position for each axis."""
    t = result.table
    fig = Figure(figsize=(14, 4.8))
    axs = fig.subplots(1, 3)
    for ax, pos, vel, scale, xlabel, ylabel, title in (
            (axs[0], "pos_x", "vel_x", 1e3, "x (mm)", r"$v_x$ (m/s)", "x phase space"),
            (axs[1], "pos_y", "vel_y", 1e3, "y (mm)", r"$v_y$ (m/s)", "y phase space"),
            (axs[2], "pos_z", "vel_z", 1.0, "z (m)", r"$v_z$ (m/s)", "z phase space")):
        ax.plot(t[pos] * scale, t[vel], color=PATH_COLOUR, lw=2, label="trajectory")
        ax.plot(t[pos].iloc[0] * scale, t[vel].iloc[0], "o", color="#2b8a3e", mec="k", label="start")
        ax.plot(t[pos].iloc[-1] * scale, t[vel].iloc[-1], "X", ms=10, color=HIT_COLOUR, mec="k", label="end")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
    vz_mean = float(t["vel_z"].mean())
    axs[2].set_ylim(0.99 * vz_mean, 1.01 * vz_mean)
    fig.suptitle("Phase-space diagrams (velocity vs position)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)


def plot_energy(result, bl, path, dpi):
    """4. Energy conservation and fractional energy error."""
    t = result.table
    time_us = t["time_t"] * 1e6
    unit = bl.e_charge * 1e3                                   # J -> keV
    energy = t["total_energy"]
    error = np.abs((energy - energy.iloc[0]) / energy.iloc[0])
    fig = Figure(figsize=(9, 7))
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(time_us, t["kinetic_energy"] / unit, label="kinetic energy K")
    ax1.plot(time_us, t["potential_energy"] / unit, label="potential energy U")
    ax1.plot(time_us, energy / unit, "k--", label="total energy E = K + U")
    ax1.set_ylabel("Energy (keV)")
    ax1.set_title("Energy of the ion")
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend()
    floor = np.finfo(float).eps / 10                          # exact zeros cannot be drawn on a log axis
    ax2.plot(time_us, np.maximum(error, floor), color="red", label="relative energy error")
    ax2.set_yscale("log")                                      # log scale for numerical error
    ax2.set_title("Integration error over time (values below machine precision are drawn at the bottom)")
    ax2.set_xlabel("Time (µs)")
    ax2.set_ylabel(r"Fractional error ($\Delta E/E_{0}$)")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)


def save_all_plots(result, bl, directory, dpi):
    """Create the four mandatory figures; return the list of files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {"trajectory.png": lambda p: plot_trajectory(result, bl, p, dpi),
             "time_series.png": lambda p: plot_time_series(result, p, dpi),
             "phase_space.png": lambda p: plot_phase_space(result, p, dpi),
             "energy_error.png": lambda p: plot_energy(result, bl, p, dpi)}
    for name, draw in files.items():
        draw(directory / name)
    return [directory / name for name in files]


# --------------------------------------------------------------------------- #
#  Optional animated replay of the computed trajectory
# --------------------------------------------------------------------------- #
def run_interactive(result, bl, viewer):
    """Replay the computed trajectory with Start/Stop, Step and Reset buttons."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    t = result.table
    z, x, y = (t[c].to_numpy() for c in ("pos_z", "pos_x", "pos_y"))
    n = len(t)
    step = max(1, n // int(viewer["frames_per_flight"]))
    state = {"i": 0, "running": False}

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("Ion-beam deflection - replay of the computed trajectory", fontweight="bold")
    ax_side = fig.add_axes([0.055, 0.55, 0.62, 0.35])
    ax_top = fig.add_axes([0.055, 0.17, 0.40, 0.28])
    ax_waf = fig.add_axes([0.50, 0.17, 0.20, 0.28])
    ax_info = fig.add_axes([0.72, 0.17, 0.27, 0.73])
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    ax_info.set_facecolor("#f5f6fa")
    info = ax_info.text(0.04, 0.98, "", va="top", ha="left", family="monospace", fontsize=9,
                        transform=ax_info.transAxes)
    draw_side_view(ax_side, bl)
    draw_top_view(ax_top, bl)
    draw_wafer_view(ax_waf, bl)
    ax_waf.set_title("target wafer", fontsize=10)

    trail_s, = ax_side.plot([], [], "-", color=PATH_COLOUR, lw=2.4, zorder=5)
    trail_t, = ax_top.plot([], [], "-", color=PATH_COLOUR, lw=2.4, zorder=5)
    trail_w, = ax_waf.plot([], [], "-", color=PATH_COLOUR, lw=2.0, zorder=5)
    dot_s, = ax_side.plot([], [], "o", ms=10, color=PATH_COLOUR, mec="k", zorder=7)
    dot_t, = ax_top.plot([], [], "o", ms=10, color=PATH_COLOUR, mec="k", zorder=7)
    dot_w, = ax_waf.plot([], [], "o", ms=8, color=PATH_COLOUR, mec="k", zorder=7)
    hit_w, = ax_waf.plot([], [], "X", ms=11, color=HIT_COLOUR, mec="k", zorder=8)

    def refresh():
        i = state["i"]
        trail_s.set_data(z[:i + 1], y[:i + 1])
        trail_t.set_data(z[:i + 1], x[:i + 1])
        trail_w.set_data(x[:i + 1] * 1e3, y[:i + 1] * 1e3)
        dot_s.set_data([z[i]], [y[i]])
        dot_t.set_data([z[i]], [x[i]])
        dot_w.set_data([x[i] * 1e3], [y[i] * 1e3])
        done = i == n - 1
        hit_w.set_data([x[i] * 1e3], [y[i] * 1e3]) if done and result.outcome == "wafer" else hit_w.set_data([], [])
        lines = ["BEAM", f"v0 = {bl.v0:.3e} m/s", f"E_x = {bl.Ex:+.3e} V/m", f"E_y = {bl.Ey:+.3e} V/m", "",
                 "ION", f"t = {t['time_t'].iloc[i] * 1e6:.4f} µs", f"z = {z[i]:+.5f} m",
                 f"x = {x[i]:+.5f} m", f"y = {y[i]:+.5f} m", "",
                 "STATUS", "running" if state["running"] else ("finished" if done else "paused")]
        if done:
            lines += ["", "RESULT"] + textwrap.wrap(_impact_label(result, bl), 36)
        info.set_text("\n".join(lines))
        b_run.label.set_text("Stop" if state["running"] else "Start")
        fig.canvas.draw_idle()

    def on_run(_=None):
        if state["i"] == n - 1:
            state["i"] = 0
        state["running"] = not state["running"]
        refresh()

    def on_step(_=None):
        state["running"] = False
        state["i"] = 0 if state["i"] == n - 1 else min(n - 1, state["i"] + step)
        refresh()

    def on_reset(_=None):
        state["running"], state["i"] = False, 0
        refresh()

    def tick():
        if state["running"]:
            state["i"] = min(n - 1, state["i"] + step)
            if state["i"] == n - 1:
                state["running"] = False
            refresh()

    b_run = Button(fig.add_axes([0.055, 0.06, 0.075, 0.05]), "Start", color="#b7e4c7")
    b_step = Button(fig.add_axes([0.14, 0.06, 0.075, 0.05]), "Step", color="#ffe8a3")
    b_reset = Button(fig.add_axes([0.225, 0.06, 0.075, 0.05]), "Reset", color="#ffc9c9")
    b_run.on_clicked(on_run)
    b_step.on_clicked(on_step)
    b_reset.on_clicked(on_reset)
    timer = fig.canvas.new_timer(interval=int(viewer["frame_ms"]))
    timer.add_callback(tick)
    timer.start()
    refresh()
    plt.show()
    return timer
