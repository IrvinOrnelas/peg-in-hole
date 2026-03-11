import argparse
import threading
import time
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from classes.model.RobotModel import RobotModel
from classes.controllers.CTC import CTC
from classes.robots.Master import Master
from classes.network.MasterNetClient import MasterNetClient


mpl.rcParams["toolbar"] = "None"
for key in mpl.rcParams:
    if key.startswith("keymap."):
        mpl.rcParams[key] = []


def setup_master_plots():
    fig = plt.figure(figsize=(14, 10))

    fig.suptitle(
        "TE3001B — Robot Maestro 3R | Peg-in-Hole Teleoperado",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    ax_robot = fig.add_subplot(2, 2, 1)
    ax_tau = fig.add_subplot(2, 2, 2)
    ax_force = fig.add_subplot(2, 2, 3)
    ax_q = fig.add_subplot(2, 2, 4)

    for ax in [ax_robot, ax_tau, ax_force, ax_q]:
        ax.grid(True)

    ax_robot.set_title("Vista Cinemática 3R")
    ax_robot.set_xlabel("x [m]")
    ax_robot.set_ylabel("y [m]")
    ax_robot.set_xlim(-0.9, 0.9)
    ax_robot.set_ylim(-0.9, 0.9)
    ax_robot.set_aspect("equal")

    ax_tau.set_title("Torques Articulares τ [Nm]")
    ax_tau.set_xlabel("Tiempo [s]")
    ax_tau.set_ylabel("τ [Nm]")

    ax_force.set_title("Fuerzas de Contacto Reflejadas [N]")
    ax_force.set_xlabel("Tiempo [s]")
    ax_force.set_ylabel("F [N]")

    ax_q.set_title("Ángulos Articulares q [rad]")
    ax_q.set_xlabel("Tiempo [s]")
    ax_q.set_ylabel("q [rad]")

    line_robot, = ax_robot.plot([], [], "o-", linewidth=3, markersize=8)
    point_ee, = ax_robot.plot([], [], "s", markersize=10)
    point_target, = ax_robot.plot([], [], "x", markersize=10, markeredgewidth=2)

    ax_robot.plot([0.55], [0.10], "x", markersize=8, markeredgewidth=2)
    ax_robot.text(0.57, 0.12, "HOLE")

    tau_lines = [
        ax_tau.plot([], [], label="τ1")[0],
        ax_tau.plot([], [], label="τ2")[0],
        ax_tau.plot([], [], label="τ3")[0]
    ]
    ax_tau.legend(loc="upper right")

    force_lines = [
        ax_force.plot([], [], label="Fx")[0],
        ax_force.plot([], [], label="Fy")[0]
    ]
    ax_force.legend(loc="upper right")

    q_lines = [
        ax_q.plot([], [], label="q1")[0],
        ax_q.plot([], [], label="q2")[0],
        ax_q.plot([], [], linestyle=":", label="q3")[0]
    ]
    ax_q.legend(loc="upper right")

    plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))

    return fig, ax_robot, ax_tau, ax_force, ax_q, line_robot, point_ee, point_target, tau_lines, force_lines, q_lines


def main():
    parser = argparse.ArgumentParser(description="Master Robot 3R")
    parser.add_argument("--slave-ip", default="127.0.0.1", help="IP del esclavo")
    args = parser.parse_args()

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
        kp=np.diag([140.0, 120.0, 100.0]),
        kv=np.diag([30.0, 25.0, 20.0])
    )

    q0 = np.deg2rad(np.array([135.0, -100.0, -45.0], dtype=np.float64))

    master = Master(
        robot_model=model,
        controller=controller,
        q0=q0,
        dt=0.01
    )

    net = MasterNetClient(slave_ip=args.slave_ip)

    fig, ax_robot, ax_tau, ax_force, ax_q, line_robot, point_ee, point_target, tau_lines, force_lines, q_lines = setup_master_plots()

    pressed_keys = set()
    base_speed = 0.25
    fast_speed = 0.70
    current_speed = [base_speed]

    speed_step = 0.01
    min_speed = 0.01
    max_speed = 1.0
    N = 500

    hist_t = np.zeros(N, dtype=np.float64)
    hist_q = np.zeros((N, 3), dtype=np.float64)
    hist_tau = np.zeros((N, 3), dtype=np.float64)
    hist_F = np.zeros((N, 2), dtype=np.float64)
    idx = [0]

    running = [True]

    csv_file = open("master_data.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "time",
        "q1", "q2", "q3",
        "dq1", "dq2", "dq3",
        "tau1", "tau2", "tau3",
        "Fx", "Fy",
        "ee_x", "ee_y",
        "target_x", "target_y",
        "in_contact",
        "contact_state"
    ])
    csv_file.flush()

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
            current_speed[0] = min(max_speed, current_speed[0] + speed_step)
            print(f"Velocidad master: {current_speed[0]:.2f}")
        elif key == "-":
            current_speed[0] = max(min_speed, current_speed[0] - speed_step)
            print(f"Velocidad master: {current_speed[0]:.2f}")
        elif key == "escape":
            running[0] = False
            plt.close(fig)

    def on_key_release(event):
        key = event.key
        if key is None:
            return
        key = key.lower()

        if key in pressed_keys:
            pressed_keys.remove(key)
        if key == "shift":
            current_speed[0] = base_speed

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

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

    def sim_loop():
        while running[0]:
            update_target_from_keyboard()

            net.send_target(master.p_des)

            master.q_des = model.inverse_kinematics(master.q_des, master.p_des)

            tau, e, de = controller.compute(
                master.q,
                master.dq,
                master.q_des,
                master.dq_des,
                master.ddq_des,
                F_ext=None
            )

            master.q, master.dq = model.integrate_dynamics(
                master.q,
                master.dq,
                tau,
                master.dt
            )
            master.t += master.dt

            ee = model.forward_kinematics(master.q)

            i = idx[0] % N
            hist_t[i] = master.t
            hist_q[i] = master.q
            hist_tau[i] = tau
            hist_F[i] = net.F_contact
            idx[0] += 1

            in_contact = getattr(net, "in_contact", False)
            contact_state = getattr(net, "contact_state", "")

            csv_writer.writerow([
                master.t,
                master.q[0], master.q[1], master.q[2],
                master.dq[0], master.dq[1], master.dq[2],
                tau[0], tau[1], tau[2],
                net.F_contact[0], net.F_contact[1],
                ee[0], ee[1],
                master.p_des[0], master.p_des[1],
                int(bool(in_contact)),
                contact_state
            ])
            csv_file.flush()

            time.sleep(master.dt)

    threading.Thread(target=sim_loop, daemon=True).start()

    def animate(_frame):
        n = min(idx[0], N)
        i0 = idx[0] % N
        order = np.arange(i0, i0 + n) % N

        t = hist_t[order]
        q_hist = hist_q[order]
        tau_hist = hist_tau[order]
        F_hist = hist_F[order]

        pts = model.forward_kinematics_full(master.q)
        line_robot.set_data(pts[:, 0], pts[:, 1])
        point_ee.set_data([pts[-1, 0]], [pts[-1, 1]])
        point_target.set_data([master.p_des[0]], [master.p_des[1]])

        t_win = 5.0
        mask = (t > master.t - t_win) if master.t > t_win else np.ones(n, dtype=bool)

        for j, ln in enumerate(tau_lines):
            ln.set_data(t[mask], tau_hist[mask, j])

        for j, ln in enumerate(force_lines):
            ln.set_data(t[mask], F_hist[mask, j])

        for j, ln in enumerate(q_lines):
            ln.set_data(t[mask], q_hist[mask, j])

        for ax in [ax_tau, ax_force, ax_q]:
            ax.set_xlim(max(0.0, master.t - t_win), max(t_win, master.t))
            ax.relim()
            ax.autoscale_view(scalex=False)

        return [line_robot, point_ee, point_target] + tau_lines + force_lines + q_lines

    anim=animation.FuncAnimation(
        fig,
        animate,
        interval=50,
        blit=False,
        cache_frame_data=False
    )

    plt.show()
    running[0] = False
    csv_file.close()


if __name__ == "__main__":
    main()