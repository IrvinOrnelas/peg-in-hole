import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Desactivar shortcuts de matplotlib
mpl.rcParams["toolbar"] = "None"
for key in mpl.rcParams:
    if key.startswith("keymap."):
        mpl.rcParams[key] = []

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from classes.model.RobotModel import RobotModel
from classes.controllers.ImpedanceController import ImpedanceController
from classes.contact.PegHoleContact import PegHoleContact
from classes.robots.Slave import Slave
from matplotlib.patches import Circle


def main():
    model = RobotModel(
        l1=0.35,
        l2=0.30,
        l3=0.20,
        m1=1.5,
        m2=1.0,
        m3=0.5
    )

    controller = ImpedanceController(
        robot_model=model,
        kd=700.0,
        bd=80.0,
        kq=np.diag([1.5, 1.0, 0.8]),
        bq=np.diag([0.8, 0.6, 0.5]),
        q_rest=np.array([0.6, -0.5, 0.1], dtype=np.float64)
    )

    contact_model = PegHoleContact(
        hole_center=np.array([0.55, 0.10], dtype=np.float64),
        peg_length=0.08,
        peg_radius=0.008,
        hole_radius=0.009,
        contact_k=2000.0,
        force_threshold=2.0
    )

    slave = Slave(
        robot_model=model,
        controller=controller,
        contact_model=contact_model,
        q0=np.array([0.6, -0.5, 0.1], dtype=np.float64),
        dt=0.01
    )

    pressed_keys = set()
    base_speed = 0.30
    fast_speed = 0.80
    current_speed = [base_speed]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.6, 1.0)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Slave Robot 3R")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    line_robot, = ax.plot([], [], "o-", linewidth=3, markersize=8)
    point_ee, = ax.plot([], [], "s", markersize=10)
    point_target, = ax.plot([], [], "x", markersize=12, markeredgewidth=2)
    line_path, = ax.plot([], [], "--", linewidth=1)

    line_force, = ax.plot([], [], linewidth=2)

    text_info = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        verticalalignment="top"
    )

    # Dibujo simple del agujero
    hole_center = contact_model.hole_center
    hole_radius = float(contact_model.hole_radius)

    hole_circle = Circle(
        (hole_center[0], hole_center[1]),
        hole_radius,
        fill=False,
        linewidth=2
    )
    ax.add_patch(hole_circle)

    ax.text(hole_center[0] + 0.02, hole_center[1] + 0.02, "HOLE")

    path_x = []
    path_y = []

    def clamp_target(p):
        p = p.copy()
        p[0] = np.clip(p[0], -0.80, 0.80)
        p[1] = np.clip(p[1], -0.50, 0.90)
        return p

    def on_key_press(event):
        key = event.key
        if key is None:
            return

        key = key.lower()

        if key in ["w", "a", "s", "d"]:
            pressed_keys.add(key)

        elif key == "shift":
            current_speed[0] = fast_speed

        elif key in ["+", "="]:
            current_speed[0] = min(current_speed[0] + 0.05, 2.0)

        elif key == "-":
            current_speed[0] = max(current_speed[0] - 0.05, 0.05)

    def on_key_release(event):
        key = event.key
        if key is None:
            return

        key = key.lower()

        if key in pressed_keys:
            pressed_keys.remove(key)

        if key == "shift":
            current_speed[0] = base_speed

    def on_mouse_press(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        p_mouse = np.array([event.xdata, event.ydata], dtype=np.float64)
        slave.set_cartesian_target(clamp_target(p_mouse))

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)
    fig.canvas.mpl_connect("button_press_event", on_mouse_press)

    def update_target_from_keyboard():
        p = slave.p_des.copy()

        if "w" in pressed_keys:
            p[1] += current_speed[0] * slave.dt
        if "s" in pressed_keys:
            p[1] -= current_speed[0] * slave.dt
        if "a" in pressed_keys:
            p[0] -= current_speed[0] * slave.dt
        if "d" in pressed_keys:
            p[0] += current_speed[0] * slave.dt

        slave.set_cartesian_target(clamp_target(p))

    def animate(_frame):
        update_target_from_keyboard()

        data = slave.step()
        pts = model.forward_kinematics_full(slave.q)

        x = pts[:, 0]
        y = pts[:, 1]

        p_ee = pts[-1]
        F_contact = data["F_contact"]

        line_robot.set_data(x, y)
        point_ee.set_data([p_ee[0]], [p_ee[1]])
        point_target.set_data([data["p_des"][0]], [data["p_des"][1]])

        path_x.append(p_ee[0])
        path_y.append(p_ee[1])
        line_path.set_data(path_x, path_y)

        force_scale = 0.02
        p_force_end = p_ee + force_scale * F_contact
        line_force.set_data(
            [p_ee[0], p_force_end[0]],
            [p_ee[1], p_force_end[1]]
        )

        text_info.set_text(
            f't = {data["t"]:.2f} s\n'
            f'p = [{data["p"][0]:.3f}, {data["p"][1]:.3f}]\n'
            f'p_target = [{data["p_des"][0]:.3f}, {data["p_des"][1]:.3f}]\n'
            f'F_contact = [{data["F_contact"][0]:.3f}, {data["F_contact"][1]:.3f}] N\n'
            f'state = {data["contact_state"]}\n'
            f'speed = {current_speed[0]:.2f} m/s'
        )

        return line_robot, point_ee, point_target, line_path, line_force, text_info

    ani = animation.FuncAnimation(
        fig,
        animate,
        interval=10,
        blit=False,
        cache_frame_data=False
    )

    plt.show()


if __name__ == "__main__":
    main()