import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import numpy as np

# Physical Constants
E_CHARGE = 1.602e-19
AMU_TO_KG = 1.6605e-27

# Setup Figure & Layout
fig, ax = plt.subplots(figsize=(16, 9))
plt.subplots_adjust(left=0.05, right=0.74, bottom=0.28, top=0.95)

# ==========================================
# 1. SCHEMATIC LAYOUT
# ==========================================
hv_box = patches.Rectangle(
    (-0.5, -0.9),
    2.8,
    5.8,
    linewidth=2,
    edgecolor='gray',
    facecolor='lightgray',
    alpha=0.25,
)
ax.add_patch(hv_box)
ax.text(
    0.9,
    4.6,
    'High Voltage Section',
    fontsize=10,
    weight='bold',
    ha='center',
    color='dimgray',
)

ion_src = patches.Rectangle(
    (0.8, -0.7), 0.4, 0.5, facecolor='gold', edgecolor='black', lw=1.5
)
ax.add_patch(ion_src)
ax.text(1.0, -0.45, 'Ion Source', fontsize=8, ha='center', weight='bold')

ext_left = patches.Rectangle(
    (0.7, -0.05), 0.15, 0.7, facecolor='skyblue', edgecolor='black'
)
ext_right = patches.Rectangle(
    (1.15, -0.05), 0.15, 0.7, facecolor='skyblue', edgecolor='black'
)
ax.add_patch(ext_left)
ax.add_patch(ext_right)
ax.text(
    1.0,
    0.75,
    'Extraction Electrode',
    fontsize=8,
    ha='center',
    style='italic',
)

analyzer_arc = patches.Arc(
    (1.0, 0.8),
    2.4,
    2.4,
    angle=0,
    theta1=0,
    theta2=90,
    color='dodgerblue',
    lw=3,
    ls='--',
)
ax.add_patch(analyzer_arc)
ax.text(
    0.3,
    2.1,
    'Mass Analyzer\n(90° Bend)',
    fontsize=9,
    weight='bold',
    color='dodgerblue',
    ha='center',
)

ax.plot([2.0, 2.4], [1.85, 1.85], color='crimson', lw=4)
ax.plot([2.0, 2.4], [2.15, 2.15], color='crimson', lw=4)
ax.text(
    2.2,
    2.3,
    'Analyzing Slit',
    fontsize=8,
    weight='bold',
    color='crimson',
    ha='center',
)

accel_col = patches.Rectangle(
    (2.5, 1.75),
    1.2,
    0.5,
    facecolor='orange',
    alpha=0.4,
    edgecolor='black',
    hatch='//',
)
ax.add_patch(accel_col)
ax.text(
    3.1,
    1.45,
    'Acceleration Column',
    fontsize=8,
    ha='center',
    style='italic',
)

deflector = patches.Rectangle(
    (3.9, 1.5), 0.6, 1.0, facecolor='cyan', alpha=0.4, edgecolor='black'
)
ax.add_patch(deflector)
ax.text(4.2, 2.65, 'Deflector', fontsize=8, ha='center', style='italic')

optics_box = patches.Rectangle(
    (4.8, 1.65), 1.8, 0.7, facecolor='white', edgecolor='black', lw=1.5
)
ax.add_patch(optics_box)
ax.text(
    5.7,
    2.5,
    'Q-Lens & Scanner',
    fontsize=8,
    ha='center',
    weight='bold',
)
ax.add_patch(
    patches.Rectangle(
        (5.1, 1.85), 0.4, 0.3, facecolor='cornflowerblue', edgecolor='black'
    )
)
ax.add_patch(
    patches.Rectangle(
        (5.9, 1.85), 0.4, 0.3, facecolor='lightpink', edgecolor='black'
    )
)

end_station = patches.Rectangle(
    (7.0, -0.9),
    1.6,
    5.8,
    linewidth=2,
    edgecolor='gray',
    facecolor='lightgreen',
    alpha=0.25,
)
ax.add_patch(end_station)
ax.text(
    7.8,
    4.6,
    'End Station',
    fontsize=10,
    weight='bold',
    ha='center',
    color='darkgreen',
)

target_wafer = patches.Rectangle(
    (8.1, 1.0), 0.15, 2.0, facecolor='gold', edgecolor='orange', lw=2
)
ax.add_patch(target_wafer)
ax.text(
    8.17,
    3.3,
    'Sample (Substrate)',
    fontsize=8,
    weight='bold',
    rotation=90,
    va='bottom',
)

ax.set_xlim(-1.0, 9.0)
ax.set_ylim(-1.2, 5.2)
ax.set_xlabel('Beamline X Position (m)', fontsize=10, weight='bold')
ax.set_ylabel('Y Position (m)', fontsize=10, weight='bold')
ax.set_title(
    'Advanced Ion Implanter Beam Deflection & Animation Model',
    fontsize=13,
    weight='bold',
)
ax.grid(True, linestyle='--', alpha=0.3)

# ==========================================
# 2. TELEMETRY MONITOR PANEL
# ==========================================
ax_monitor = fig.add_axes([0.76, 0.28, 0.21, 0.67])
ax_monitor.set_facecolor('#f8f9fa')
ax_monitor.axis('off')
ax_monitor.add_patch(
    patches.Rectangle(
        (0, 0),
        1,
        1,
        transform=ax_monitor.transAxes,
        facecolor='#f8f9fa',
        edgecolor='#cccccc',
        lw=1.5,
    )
)
monitor_text = ax_monitor.text(
    0.05,
    0.95,
    '',
    fontsize=9.5,
    family='monospace',
    verticalalignment='top',
    transform=ax_monitor.transAxes,
)

# ==========================================
# 3. PLOT ELEMENTS
# ==========================================
(forecast_line,) = ax.plot([], [], 'b--', lw=1.5, alpha=0.6, label='Forecast')
(beam_trail,) = ax.plot([], [], 'r-', lw=3, alpha=0.8, label='Active Beam')
(particle_dot,) = ax.plot([], [], 'ro', markersize=9, zorder=6)
(radius_circle_plot,) = ax.plot([], [], 'c:', lw=1.5, label='Bending Radius R')


class SimulationState:

  def __init__(self):
    self.x_path = np.array([])
    self.y_path = np.array([])
    self.current_frame = 0
    self.is_running = False
    self.passed_slit = True
    self.velocity = 0.0
    self.radius = 0.0
    self.current_x = 1.0
    self.current_y = -0.45


state = SimulationState()


def update_calculations():
  mass_amu = slider_mass.val
  v_ext = slider_v_ext.val * 1e3
  b_field = slider_b.val
  scan_val = slider_scan.val

  q = E_CHARGE
  m = mass_amu * AMU_TO_KG

  v = np.sqrt(2 * q * v_ext / m) if v_ext > 0 else 0
  R = (m * v) / (q * b_field) if b_field > 0 else 999.0
  state.velocity = v
  state.radius = R

  y_src = np.linspace(-0.45, 0.8, 20)
  x_src = np.ones_like(y_src) * 1.0

  theta = np.linspace(np.pi, np.pi / 2, 40)
  x_arc = (1.0 + R) + R * np.cos(theta)
  y_arc = 0.8 + R * np.sin(theta)

  y_exit = 0.8 + R
  state.passed_slit = abs(y_exit - 2.0) < 0.22

  x_exit_actual = 1.0 + R
  x_drift = np.linspace(x_exit_actual, 8.1, 50)
  y_drift = y_exit + (x_drift - x_exit_actual) * scan_val * 0.12

  state.x_path = np.concatenate([x_src, x_arc[1:], x_drift[1:]])
  state.y_path = np.concatenate([y_src, y_arc[1:], y_drift[1:]])

  circ_theta = np.linspace(np.pi, np.pi / 2, 50)
  radius_circle_plot.set_xdata((1.0 + R) + R * np.cos(circ_theta))
  radius_circle_plot.set_ydata(0.8 + R * np.sin(circ_theta))

  forecast_line.set_xdata(state.x_path)
  forecast_line.set_ydata(state.y_path)

  update_monitor_display(
      mass_amu, v_ext, b_field, v, R, state.current_x, state.current_y
  )
  fig.canvas.draw_idle()


def update_monitor_display(mass, v_ext, b_field, v, R, cur_x, cur_y):
  status_str = 'PASSED ✅' if state.passed_slit else 'BLOCKED ❌'
  text_content = (
      f'=== REAL-TIME MONITOR ===\n\n'
      f'[Current Position]\n'
      f'• X : {cur_x:6.2f} m\n'
      f'• Y : {cur_y:6.2f} m\n\n'
      f'[Beam Parameters]\n'
      f'• Mass (m) : {mass:5.1f} amu\n'
      f'• Ext Volt : {v_ext/1e3:5.1f} kV\n'
      f'• B-Field  : {b_field:5.3f} T\n\n'
      f'[Physics Formulae]\n'
      f'• v = sqrt(2qV/m)\n'
      f'  = {v/1e4:5.2f} × 10^4 m/s\n\n'
      f'• R = (mv)/(qB)\n'
      f'  = {R:5.2f} m (Des: 1.20m)\n\n'
      f'[Slit Analysis]\n'
      f'• Status: {status_str}'
  )
  monitor_text.set_text(text_content)


# ==========================================
# 4. CONTROLS
# ==========================================
ax_mass = plt.axes([0.10, 0.18, 0.45, 0.025])
ax_tmass = plt.axes([0.58, 0.18, 0.08, 0.025])
ax_v_ext = plt.axes([0.10, 0.13, 0.45, 0.025])
ax_tv_ext = plt.axes([0.58, 0.13, 0.08, 0.025])
ax_b = plt.axes([0.10, 0.08, 0.45, 0.025])
ax_tb = plt.axes([0.58, 0.08, 0.08, 0.025])
ax_scan = plt.axes([0.10, 0.03, 0.45, 0.025])
ax_tscan = plt.axes([0.58, 0.03, 0.08, 0.025])

slider_mass = Slider(
    ax_mass, 'Mass (amu)', valmin=1, valmax=100, valinit=28, valfmt='%0.1f'
)
slider_v_ext = Slider(
    ax_v_ext, 'Ext Volt (kV)', valmin=5, valmax=50, valinit=20, valfmt='%0.1f'
)
slider_b = Slider(
    ax_b,
    'B-Field (T)',
    valmin=0.05,
    valmax=0.3,
    valinit=0.090,
    valfmt='%0.3f',
)
slider_scan = Slider(
    ax_scan,
    'Scan Deflect',
    valmin=-1.0,
    valmax=1.0,
    valinit=0.0,
    valfmt='%0.2f',
)

textbox_mass = TextBox(ax_tmass, '', initial='28')
textbox_v_ext = TextBox(ax_tv_ext, '', initial='20')
textbox_b = TextBox(ax_tb, '', initial='0.090')
textbox_scan = TextBox(ax_tscan, '', initial='0.0')


def on_slider_changed(val):
  textbox_mass.set_val(f'{slider_mass.val:.1f}')
  textbox_v_ext.set_val(f'{slider_v_ext.val:.1f}')
  textbox_b.set_val(f'{slider_b.val:.3f}')
  textbox_scan.set_val(f'{slider_scan.val:.2f}')
  update_calculations()


slider_mass.on_changed(on_slider_changed)
slider_v_ext.on_changed(on_slider_changed)
slider_b.on_changed(on_slider_changed)
slider_scan.on_changed(on_slider_changed)


def submit_text(text, slider):
  try:
    val = float(text)
    if slider.valmin <= val <= slider.valmax:
      slider.set_val(val)
  except ValueError:
    pass


textbox_mass.on_submit(lambda text: submit_text(text, slider_mass))
textbox_v_ext.on_submit(lambda text: submit_text(text, slider_v_ext))
textbox_b.on_submit(lambda text: submit_text(text, slider_b))
textbox_scan.on_submit(lambda text: submit_text(text, slider_scan))

btn_start = Button(
    plt.axes([0.70, 0.12, 0.12, 0.07]),
    'Start Animation',
    color='gold',
    hovercolor='orange',
)
btn_reset = Button(
    plt.axes([0.84, 0.12, 0.12, 0.07]),
    'Reset View',
    color='lightcoral',
    hovercolor='indianred',
)


def on_start(event):
  update_calculations()
  state.current_frame = 0
  state.is_running = True


def on_reset(event):
  state.is_running = False
  beam_trail.set_xdata([])
  beam_trail.set_ydata([])
  particle_dot.set_xdata([])
  particle_dot.set_ydata([])
  state.current_x = 1.0
  state.current_y = -0.45
  update_calculations()


btn_start.on_clicked(on_start)
btn_reset.on_clicked(on_reset)
ax.legend(loc='upper left', fontsize=8)


# ==========================================
# 5. OPTIMIZED ANIMATION LOOP (NO LAG)
# ==========================================
def update_animation(frame):
  if not state.is_running or len(state.x_path) == 0:
    return beam_trail, particle_dot

  if state.current_frame < len(state.x_path):
    idx = state.current_frame
    state.current_x = state.x_path[idx]
    state.current_y = state.y_path[idx]

    if not state.passed_slit and state.x_path[idx] >= 2.0:
      beam_trail.set_xdata(state.x_path[:idx])
      beam_trail.set_ydata(state.y_path[:idx])
      particle_dot.set_xdata([state.current_x])
      particle_dot.set_ydata([state.current_y])
      update_monitor_display(
          slider_mass.val,
          slider_v_ext.val * 1e3,
          slider_b.val,
          state.velocity,
          state.radius,
          state.current_x,
          state.current_y,
      )
      state.is_running = False
      return beam_trail, particle_dot

    beam_trail.set_xdata(state.x_path[:idx])
    beam_trail.set_ydata(state.y_path[:idx])
    particle_dot.set_xdata([state.current_x])
    particle_dot.set_ydata([state.current_y])

    # PERFORMANCE BOOST: Only update monitor text every 4 frames instead of every frame
    if state.current_frame % 4 == 0:
      update_monitor_display(
          slider_mass.val,
          slider_v_ext.val * 1e3,
          slider_b.val,
          state.velocity,
          state.radius,
          state.current_x,
          state.current_y,
      )

    state.current_frame += 3  # Skip frames slightly for smoother, faster movement
  else:
    state.is_running = False

  return beam_trail, particle_dot


update_calculations()

# Increased interval slightly and enabled blitting safely for better performance
ani = animation.FuncAnimation(
    fig,
    update_animation,
    frames=250,
    interval=30,
    blit=False,
    cache_frame_data=False,
)

plt.show()