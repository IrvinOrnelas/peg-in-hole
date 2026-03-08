import numpy as np
import numpy.typing as npt


class ImpedanceController:

    def __init__(
        self,
        robot_model,
        kd: float,
        bd: float,
        kq: npt.NDArray[np.float64],
        bq: npt.NDArray[np.float64],
        q_rest: npt.NDArray[np.float64]
    ):
        self.robot_model = robot_model

        self.kd = np.float64(kd)
        self.bd = np.float64(bd)

        self.kq = np.asarray(kq, dtype=np.float64)
        self.bq = np.asarray(bq, dtype=np.float64)
        self.q_rest = np.asarray(q_rest, dtype=np.float64)

        if self.kq.shape != (3, 3):
            raise ValueError("kq must be a 3x3 matrix")
        if self.bq.shape != (3, 3):
            raise ValueError("bq must be a 3x3 matrix")
        if self.q_rest.shape != (3,):
            raise ValueError("q_rest must be a 3-element vector")

    def compute(
        self,
        q: npt.NDArray[np.float64],
        dq: npt.NDArray[np.float64],
        p_des: npt.NDArray[np.float64],
        dp_des: npt.NDArray[np.float64],
        F_contact: npt.NDArray[np.float64] | None = None
    ):
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        p_des = np.asarray(p_des, dtype=np.float64)
        dp_des = np.asarray(dp_des, dtype=np.float64)

        if q.shape != (3,):
            raise ValueError("q must be a 3-element vector")
        if dq.shape != (3,):
            raise ValueError("dq must be a 3-element vector")
        if p_des.shape != (2,):
            raise ValueError("p_des must be a 2-element vector")
        if dp_des.shape != (2,):
            raise ValueError("dp_des must be a 2-element vector")

        if F_contact is None:
            F_contact = np.zeros(2, dtype=np.float64)
        else:
            F_contact = np.asarray(F_contact, dtype=np.float64)

        if F_contact.shape != (2,):
            raise ValueError("F_contact must be a 2-element vector")

        p_cur = self.robot_model.forward_kinematics(q)
        J = self.robot_model.jacobian(q)

        dp_cur = J @ dq

        e_p = p_des - p_cur
        e_dp = dp_des - dp_cur

        F_imp = self.kd * e_p + self.bd * e_dp
        F_total = F_imp + F_contact

        tau_task = J.T @ F_total

        e_q = self.q_rest - q
        tau_posture = self.kq @ e_q - self.bq @ dq

        g_vec = self.robot_model.gravity_vector(q)
        C_mat = self.robot_model.coriolis_matrix(q, dq)

        tau = tau_task + tau_posture + g_vec + C_mat @ dq
        tau = np.clip(tau, -20.0, 20.0)

        return tau, F_total, e_p