import numpy as np
import numpy.typing as npt


class Master:

    def __init__(
        self,
        robot_model,
        controller,
        q0: npt.NDArray[np.float64],
        dt: float
    ):
        self.robot_model = robot_model
        self.controller = controller

        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.dq = np.zeros(3, dtype=np.float64)

        if self.q.shape != (3,):
            raise ValueError("q0 must be a 3-element vector")

        self.q_des = self.q.copy()
        self.dq_des = np.zeros(3, dtype=np.float64)
        self.ddq_des = np.zeros(3, dtype=np.float64)

        self.p_des = self.robot_model.forward_kinematics(self.q).copy()

        self.dt = np.float64(dt)
        self.t = np.float64(0.0)

    def set_cartesian_target(self, p_des: npt.NDArray[np.float64]):
        p_des = np.asarray(p_des, dtype=np.float64)

        if p_des.shape != (2,):
            raise ValueError("p_des must be a 2-element vector")

        self.p_des = p_des.copy()

    def step(self):
        self.q_des = self.robot_model.inverse_kinematics(self.q_des, self.p_des)

        tau, e, de = self.controller.compute(
            self.q,
            self.dq,
            self.q_des,
            self.dq_des,
            self.ddq_des
        )

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
            "q_des": self.q_des.copy(),
            "tau": tau.copy(),
            "e": e.copy(),
            "de": de.copy(),
            "p": self.robot_model.forward_kinematics(self.q).copy(),
            "p_des": self.p_des.copy()
        }