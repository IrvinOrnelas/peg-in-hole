import argparse
import threading
import time
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from classes.model.RobotModel import RobotModel
from classes.controllers.ImpedanceController import ImpedanceController
from classes.contact.PegHoleContact import PegHoleContact
from classes.robots.Slave import Slave
from classes.network.SlaveNetServer import SlaveNetServer
from matplotlib.patches import Circle
from classes.contact.ObstacleContact import ObstacleContact


mpl.rcParams["toolbar"] = "None"
for key in mpl.rcParams:
    if key.startswith("keymap."):
        mpl.rcParams[key] = []


def setup_slave_plots():
    fig = plt.figure(figsize=(14, 10))

    fig.suptitle(
        "TE3001B — Robot Esclavo 3R | Control de Impedancia + Peg-in-Hole",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    ax_robot = fig.add_subplot(2, 2, 1)
    ax_force = fig.add_subplot(2, 2, 2)
    ax_tau = fig.add_subplot(2, 2, 3)
    ax_err = fig.add_subplot(2, 2, 4)

    for ax in [ax_robot, ax_force, ax_tau, ax_err]:
        ax.grid(True)

    ax_robot.set_title("Esclavo — Peg-in-Hole")
    ax_robot.set_xlabel("x [m]")
    ax_robot.set_ylabel("y [m]")
    ax_robot.set_xlim(-0.9, 0.9)
    ax_robot.set_ylim(-0.5, 1.0)
    ax_robot.set_aspect("equal")

    ax_force.set_title("Fuerzas de Contacto [N]")
    ax_force.set_xlabel("Tiempo [s]")
    ax_force.set_ylabel("F [N]")

    ax_tau.set_title("Torques Articulares τ [Nm]")
    ax_tau.set_xlabel("Tiempo [s]")
    ax_tau.set_ylabel("τ [Nm]")

    ax_err.set_title("Error Cartesiano |e| [mm]")
    ax_err.set_xlabel("Tiempo [s]")
    ax_err.set_ylabel("Error [mm]")

    line_robot, = ax_robot.plot([], [], "o-", linewidth=3, markersize=8)
    peg_line, = ax_robot.plot([], [], "-", linewidth=5)
    point_ee, = ax_robot.plot([], [], "s", markersize=10)
    point_target, = ax_robot.plot([], [], "x", markersize=10, markeredgewidth=2)

    state_text = ax_robot.text(
        0.02, 0.96, "",
        transform=ax_robot.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top"
    )

    hole_x = 0.55
    hole_y = 0.10
    hole_r = 0.008
    hole_depth = 0.08

    x_left = hole_x - hole_r
    x_right = hole_x + hole_r
    y_top = hole_y
    y_bottom = hole_y - hole_depth

    ax_robot.plot([x_left, x_left], [y_top, y_bottom], linewidth=2)
    ax_robot.plot([x_right, x_right], [y_top, y_bottom], linewidth=2)
    ax_robot.plot([x_left, x_right], [y_bottom, y_bottom], linewidth=2)
    ax_robot.text(hole_x + 0.02, hole_y + 0.02, "HOLE")

    force_lines = [
        ax_force.plot([], [], label="Fx contacto")[0],
        ax_force.plot([], [], label="Fy contacto")[0]
    ]
    ax_force.axhline(2.0, linestyle="--", linewidth=1.4, label="Umbral 2.0 N")
    ax_force.legend(loc="upper right")

    tau_lines = [
        ax_tau.plot([], [], label="τ1")[0],
        ax_tau.plot([], [], label="τ2")[0],
        ax_tau.plot([], [], label="τ3")[0]
    ]
    ax_tau.legend(loc="upper right")

    err_lines = [
        ax_err.plot([], [], label="|e_x|")[0],
        ax_err.plot([], [], label="|e_y|")[0]
    ]
    ax_err.axhline(1.0, linestyle=":", linewidth=1.2, label="1 mm (meta)")
    ax_err.legend(loc="upper right")

    circle_obstacle = Circle(
        (0.25, 0.05),
        0.1,
        fill=False,
        linewidth=2
    )
    ax_robot.add_patch(circle_obstacle)

    plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))

    return fig, ax_robot, ax_force, ax_tau, ax_err, line_robot, peg_line, point_ee, point_target, state_text, force_lines, tau_lines, err_lines


def main():
    parser = argparse.ArgumentParser(description="Slave Robot 3R")
    parser.add_argument("--master-ip", default="127.0.0.1", help="IP del master")
    args = parser.parse_args()

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
        kd=400.0,
        bd=40.0,
        kq=np.diag([0.05, 0.05, 0.05]),
        bq=np.diag([0.05, 0.05, 0.05]),
        q_rest=np.array([0.6, -0.5, 0.1], dtype=np.float64)
    )

    contact_model = PegHoleContact(
        hole_center=np.array([0.55, 0.10], dtype=np.float64),
        peg_length=0.08,
        peg_radius=0.0075,
        hole_radius=0.0080,
        contact_k=2000.0,
        force_threshold=2.0
    )

    obstacle = ObstacleContact(
        center=np.array([0.25, 0.05]),
        radius=0.08,
        stiffness=2000.0
    )

    q0 = np.deg2rad(np.array([135.0, -100.0, -45.0], dtype=np.float64))

    slave = Slave(
        robot_model=model,
        controller=controller,
        contact_model=contact_model,
        q0=q0,
        dt=0.01,
        obstacles=[obstacle]
    )

    net = SlaveNetServer(master_ip=args.master_ip)

    fig, ax_robot, ax_force, ax_tau, ax_err, line_robot, peg_line, point_ee, point_target, state_text, force_lines, tau_lines, err_lines = setup_slave_plots()

    N = 500
    hist_t = np.zeros(N, dtype=np.float64)
    hist_tau = np.zeros((N, 3), dtype=np.float64)
    hist_F = np.zeros((N, 2), dtype=np.float64)
    hist_e = np.zeros((N, 2), dtype=np.float64)
    idx = [0]

    running = [True]

    csv_file = open("slave_data.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "time",
        "tau1", "tau2", "tau3",
        "Fx", "Fy",
        "ee_x", "ee_y",
        "target_x", "target_y",
        "ex", "ey",
        "ex_mm", "ey_mm",
        "in_contact",
        "contact_state"
    ])
    csv_file.flush()

    def sim_loop():
        while running[0]:
            if net.has_received_target:
                slave.set_cartesian_target(net.p_des)

            data = slave.step()

            net.send_contact_data(
                F_contact=data["F_contact"],
                in_contact=data["in_contact"],
                contact_state=data["contact_state"]
            )

            ee = model.forward_kinematics(slave.q)

            i = idx[0] % N
            hist_t[i] = slave.t
            hist_tau[i] = data["tau"]
            hist_F[i] = data["F_contact"]
            hist_e[i] = data["e_p"]
            idx[0] += 1

            csv_writer.writerow([
                slave.t,
                data["tau"][0], data["tau"][1], data["tau"][2],
                data["F_contact"][0], data["F_contact"][1],
                ee[0], ee[1],
                slave.p_des[0], slave.p_des[1],
                data["e_p"][0], data["e_p"][1],
                abs(data["e_p"][0]) * 1000.0,
                abs(data["e_p"][1]) * 1000.0,
                int(bool(data["in_contact"])),
                data["contact_state"]
            ])
            csv_file.flush()

            time.sleep(slave.dt)

    threading.Thread(target=sim_loop, daemon=True).start()

    def animate(_frame):
        n = min(idx[0], N)
        i0 = idx[0] % N
        order = np.arange(i0, i0 + n) % N

        t = hist_t[order]
        tau_hist = hist_tau[order]
        F_hist = hist_F[order]
        e_hist = hist_e[order]

        pts = model.forward_kinematics_full(slave.q)
        line_robot.set_data(pts[:, 0], pts[:, 1])
        point_ee.set_data([pts[-1, 0]], [pts[-1, 1]])
        point_target.set_data([slave.p_des[0]], [slave.p_des[1]])

        peg_start = pts[-1]
        peg_dir = pts[-1] - pts[-2]
        if np.linalg.norm(peg_dir) > 1e-9:
            peg_dir = peg_dir / np.linalg.norm(peg_dir)
        peg_end = peg_start + peg_dir * 0.08
        peg_line.set_data([peg_start[0], peg_end[0]], [peg_start[1], peg_end[1]])

        if not net.has_received_target:
            state_text.set_text("Estado: ESPERANDO A MASTER...")
        else:
            state_text.set_text(f"Estado: {slave.contact_state}")

        t_win = 5.0
        mask = (t > slave.t - t_win) if slave.t > t_win else np.ones(n, dtype=bool)

        for j, ln in enumerate(force_lines):
            ln.set_data(t[mask], F_hist[mask, j])

        for j, ln in enumerate(tau_lines):
            ln.set_data(t[mask], tau_hist[mask, j])

        err_lines[0].set_data(t[mask], np.abs(e_hist[mask, 0]) * 1000.0)
        err_lines[1].set_data(t[mask], np.abs(e_hist[mask, 1]) * 1000.0)

        for ax in [ax_force, ax_tau, ax_err]:
            ax.set_xlim(max(0.0, slave.t - t_win), max(t_win, slave.t))
            ax.relim()
            ax.autoscale_view(scalex=False)

        return [line_robot, peg_line, point_ee, point_target, state_text] + force_lines + tau_lines + err_lines

    anime = animation.FuncAnimation(
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