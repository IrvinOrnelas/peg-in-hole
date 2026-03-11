import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from classes.model.RobotModel import RobotModel
from classes.controllers.CTC import CTC
from classes.robots.Master import Master


def main():

    for key in mpl.rcParams:
        if key.startswith("keymap."):
            mpl.rcParams[key] = []

    model = RobotModel(
        l1=0.35,
        l2=0.30,
        l3=0.20,
        m1=1.5,
        m2=1.0,
        m3=0.5
    )

    controller = CTC(
        robot_model=model,
        kp=np.diag([120.0, 100.0, 80.0]),
        kv=np.diag([25.0, 20.0, 15.0])
    )

    master = Master(
        robot_model=model,
        controller=controller,
        q0=np.array([np.deg2rad(135), -np.deg2rad(135), -np.deg2rad(45)], dtype=np.float64),
        dt=0.01
    )

    pressed_keys = set()
    base_speed = 0.35
    fast_speed = 0.90
    current_speed = [base_speed]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Master Robot 3R")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    line_robot, = ax.plot([], [], "o-", linewidth=3, markersize=8)
    point_ee, = ax.plot([], [], "s", markersize=10)
    point_target, = ax.plot([], [], "x", markersize=12, markeredgewidth=2)

    line_path, = ax.plot([], [], "--", linewidth=1)

    text_info = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        verticalalignment="top"
    )

    path_x = []
    path_y = []

    def clamp_target(p):
        p = p.copy()
        p[0] = np.clip(p[0], -0.80, 0.80)
        p[1] = np.clip(p[1], -0.80, 0.80)
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
        master.set_cartesian_target(clamp_target(p_mouse))

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)
    fig.canvas.mpl_connect("button_press_event", on_mouse_press)

    def update_target_from_keyboard():
        p = master.p_des.copy()

        if "w" in pressed_keys:
            p[1] += current_speed[0] * master.dt
        if "s" in pressed_keys:
            p[1] -= current_speed[0] * master.dt
        if "a" in pressed_keys:
            p[0] -= current_speed[0] * master.dt
        if "d" in pressed_keys:
            p[0] += current_speed[0] * master.dt

        master.set_cartesian_target(clamp_target(p))

    def animate(_frame):
        update_target_from_keyboard()

        data = master.step()
        pts = model.forward_kinematics_full(master.q)

        x = pts[:, 0]
        y = pts[:, 1]

        line_robot.set_data(x, y)
        point_ee.set_data([pts[-1, 0]], [pts[-1, 1]])
        point_target.set_data([master.p_des[0]], [master.p_des[1]])

        path_x.append(pts[-1, 0])
        path_y.append(pts[-1, 1])
        line_path.set_data(path_x, path_y)

        text_info.set_text(
            f't = {data["t"]:.2f} s\n'
            f'p = [{data["p"][0]:.3f}, {data["p"][1]:.3f}]\n'
            f'p_target = [{data["p_des"][0]:.3f}, {data["p_des"][1]:.3f}]\n'
            f'speed = {current_speed[0]:.2f} m/s'
        )

        return line_robot, point_ee, point_target, line_path, text_info

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