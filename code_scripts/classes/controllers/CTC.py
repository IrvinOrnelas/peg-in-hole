import numpy as np
import numpy.typing as npt

class CTC:

    def __init__(
        self,
        robot_model,
        kp: npt.NDArray[np.float64],
        kv: npt.NDArray[np.float64]
    ):
        self.robot_model = robot_model

        self.kp = np.asarray(kp, dtype=np.float64)
        self.kv = np.asarray(kv, dtype=np.float64)

        if self.kp.shape != (3, 3):
            raise ValueError("kp must be a 3x3 matrix")
        if self.kv.shape != (3, 3):
            raise ValueError("kv must be a 3x3 matrix")

    def compute(
        self,
        q: npt.NDArray[np.float64],
        dq: npt.NDArray[np.float64],
        q_des: npt.NDArray[np.float64],
        dq_des: npt.NDArray[np.float64],
        ddq_des: npt.NDArray[np.float64],
        F_ext: npt.NDArray[np.float64] | None = None
    ):
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        q_des = np.asarray(q_des, dtype=np.float64)
        dq_des = np.asarray(dq_des, dtype=np.float64)
        ddq_des = np.asarray(ddq_des, dtype=np.float64)

        if q.shape != (3,):
            raise ValueError("q must be a 3-element vector")
        if dq.shape != (3,):
            raise ValueError("dq must be a 3-element vector")
        if q_des.shape != (3,):
            raise ValueError("q_des must be a 3-element vector")
        if dq_des.shape != (3,):
            raise ValueError("dq_des must be a 3-element vector")
        if ddq_des.shape != (3,):
            raise ValueError("ddq_des must be a 3-element vector")

        e = q_des - q
        de = dq_des - dq

        a_d = ddq_des + self.kv @ de + self.kp @ e

        M_mat = self.robot_model.inertia_matrix(q)
        C_mat = self.robot_model.coriolis_matrix(q, dq)
        g_vec = self.robot_model.gravity_vector(q)

        tau = M_mat @ a_d + C_mat @ dq + g_vec

        if F_ext is not None:
            F_ext = np.asarray(F_ext, dtype=np.float64)

            if F_ext.shape != (2,):
                raise ValueError("F_ext must be a 2-element vector")

            if np.linalg.norm(F_ext) > 0.01:
                J = self.robot_model.jacobian(q)
                tau = tau + J.T @ F_ext

        tau = np.clip(tau, -20.0, 20.0)

        return tau, e, de