import numpy as np
import numpy.typing as npt


class Slave:

    def __init__(
        self,
        robot_model,
        controller,
        contact_model,
        q0: npt.NDArray[np.float64],
        dt: float,
        obstacles=None
    ):
        self.robot_model = robot_model
        self.controller = controller
        self.contact_model = contact_model
        self.obstacles = obstacles if obstacles is not None else []

        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.dq = np.zeros(3, dtype=np.float64)

        if self.q.shape != (3,):
            raise ValueError("q0 must be a 3-element vector")

        self.p_des = self.robot_model.forward_kinematics(self.q).copy()
        self.dp_des = np.zeros(2, dtype=np.float64)

        self.dt = np.float64(dt)
        self.t = np.float64(0.0)

        self.F_contact = np.zeros(2, dtype=np.float64)
        self.F_total = np.zeros(2, dtype=np.float64)

        self.contact_state = "APPROACH"
        self.in_contact = False

    def set_cartesian_target(self, p_des: npt.NDArray[np.float64]):
        p_des = np.asarray(p_des, dtype=np.float64)

        if p_des.shape != (2,):
            raise ValueError("p_des must be a 2-element vector")

        self.p_des = p_des.copy()

    def step(self):
        p_cur = self.robot_model.forward_kinematics(self.q)

        # Fuerza del agujero
        F_hole, state_str, in_hole_contact = self.contact_model.compute_contact_force(p_cur)

        # Fuerza total de obstáculos
        F_obstacles = np.zeros(2, dtype=np.float64)
        in_obstacle_contact = False

        for obstacle in self.obstacles:
            F_obs, in_obs = obstacle.compute_force(p_cur)
            F_obstacles += F_obs
            in_obstacle_contact = in_obstacle_contact or in_obs

        # Fuerza total de contacto que sí afecta al controlador
        F_contact_total = F_hole + F_obstacles

        self.F_contact = F_contact_total
        self.contact_state = state_str
        self.in_contact = in_hole_contact or in_obstacle_contact

        tau, F_total, e_p = self.controller.compute(
            self.q,
            self.dq,
            self.p_des,
            self.dp_des,
            self.F_contact
        )

        self.F_total = F_total

        self.q, self.dq = self.robot_model.integrate_dynamics(
            self.q,
            self.dq,
            tau,
            self.dt
        )

        self.t += self.dt

        return {
            "t": self.t,
            "q": self.q.copy(),
            "dq": self.dq.copy(),
            "tau": tau.copy(),
            "p": self.robot_model.forward_kinematics(self.q).copy(),
            "p_des": self.p_des.copy(),
            "F_contact": self.F_contact.copy(),
            "F_total": self.F_total.copy(),
            "e_p": e_p.copy(),
            "contact_state": self.contact_state,
            "in_contact": self.in_contact
        }