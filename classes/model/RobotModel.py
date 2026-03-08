import numpy as np
import numpy.typing as npt
from typing import Optional


class RobotModel:
    def __init__(
        self,
        l1: float,
        l2: float,
        l3: float,
        m1: float,
        m2: float,
        m3: float,
        base: Optional[npt.NDArray[np.float64]] = None,
        g_grav: Optional[float] = None,
    ):
        self.l1 = np.float64(l1)
        self.l2 = np.float64(l2)
        self.l3 = np.float64(l3)

        self.m1 = np.float64(m1)
        self.m2 = np.float64(m2)
        self.m3 = np.float64(m3)

        if base is None:
            self.base = np.array([0.0, 0.0], dtype=np.float64)
        else:
            self.base = np.asarray(base, dtype=np.float64)
            if self.base.shape != (2,):
                raise ValueError("base must be a 2-element vector")

        if g_grav is None:
            self.g_grav = np.float64(9.81)
        else:
            self.g_grav = np.float64(g_grav)

    def forward_kinematics_full(
        self, q: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (3,):
            raise ValueError("q must be a 3-element vector")

        q1, q2, q3 = q

        x0, y0 = self.base

        x1 = x0 + self.l1 * np.cos(q1)
        y1 = y0 + self.l1 * np.sin(q1)

        x2 = x1 + self.l2 * np.cos(q1 + q2)
        y2 = y1 + self.l2 * np.sin(q1 + q2)

        x3 = x2 + self.l3 * np.cos(q1 + q2 + q3)
        y3 = y2 + self.l3 * np.sin(q1 + q2 + q3)

        p0 = np.array([x0, y0], dtype=np.float64)
        p1 = np.array([x1, y1], dtype=np.float64)
        p2 = np.array([x2, y2], dtype=np.float64)
        p3 = np.array([x3, y3], dtype=np.float64)

        return np.array([p0, p1, p2, p3], dtype=np.float64)

    def forward_kinematics(
        self, q: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (3,):
            raise ValueError("q must be a 3-element vector")

        q1, q2, q3 = q

        x0, y0 = self.base

        x1 = x0 + self.l1 * np.cos(q1)
        y1 = y0 + self.l1 * np.sin(q1)

        x2 = x1 + self.l2 * np.cos(q1 + q2)
        y2 = y1 + self.l2 * np.sin(q1 + q2)

        x3 = x2 + self.l3 * np.cos(q1 + q2 + q3)
        y3 = y2 + self.l3 * np.sin(q1 + q2 + q3)

        return np.array([x3, y3], dtype=np.float64)

    def inverse_kinematics(
        self,
        q: npt.NDArray[np.float64],
        p: npt.NDArray[np.float64],
        damp: float = 0.01,
    ) -> npt.NDArray[np.float64]:
        q = np.asarray(q, dtype=np.float64).copy()
        p = np.asarray(p, dtype=np.float64)

        if q.shape != (3,):
            raise ValueError("q must be a 3-element vector")
        if p.shape != (2,):
            raise ValueError("p must be a 2-element vector")

        for _ in range(8):
            e = p - self.forward_kinematics(q)

            if np.linalg.norm(e) < 1e-4:
                break

            J = self.jacobian(q)
            Jp = J.T @ np.linalg.inv(J @ J.T + damp**2 * np.eye(2, dtype=np.float64))
            q = q + Jp @ e

        return q

    def jacobian(self, q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (3,):
            raise ValueError("q must be a 3-element vector")

        q1, q2, q3 = q

        s1 = np.sin(q1)
        s12 = np.sin(q1 + q2)
        s123 = np.sin(q1 + q2 + q3)

        c1 = np.cos(q1)
        c12 = np.cos(q1 + q2)
        c123 = np.cos(q1 + q2 + q3)

        x = np.array(
            [
                -self.l1 * s1 - self.l2 * s12 - self.l3 * s123,
                -self.l2 * s12 - self.l3 * s123,
                -self.l3 * s123,
            ],
            dtype=np.float64,
        )

        y = np.array(
            [
                self.l1 * c1 + self.l2 * c12 + self.l3 * c123,
                self.l2 * c12 + self.l3 * c123,
                self.l3 * c123,
            ],
            dtype=np.float64,
        )

        return np.vstack((x, y)).astype(np.float64)

    def inertia_matrix(self, q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (3,):
            raise ValueError("q must be a 3-element vector")

        _, q2, q3 = q

        c2 = np.cos(q2)
        c3 = np.cos(q3)
        c23 = np.cos(q2 + q3)

        m11 = (
            self.m1 * self.l1**2
            + self.m2 * (self.l1**2 + self.l2**2 + 2 * self.l1 * self.l2 * c2)
            + self.m3
            * (
                self.l1**2
                + self.l2**2
                + self.l3**2
                + 2 * self.l1 * self.l2 * c2
                + 2 * self.l1 * self.l3 * c23
                + 2 * self.l2 * self.l3 * c3
            )
        )

        m12 = (
            self.m2 * (self.l2**2 + self.l1 * self.l2 * c2)
            + self.m3
            * (
                self.l2**2
                + self.l3**2
                + self.l1 * self.l2 * c2
                + self.l1 * self.l3 * c23
                + 2 * self.l2 * self.l3 * c3
            )
        )

        m13 = self.m3 * (
            self.l3**2 + self.l1 * self.l3 * c23 + self.l2 * self.l3 * c3
        )

        m22 = self.m2 * self.l2**2 + self.m3 * (
            self.l2**2 + self.l3**2 + 2 * self.l2 * self.l3 * c3
        )

        m23 = self.m3 * (self.l3**2 + self.l2 * self.l3 * c3)

        m33 = self.m3 * self.l3**2

        return np.array(
            [
                [m11, m12, m13],
                [m12, m22, m23],
                [m13, m23, m33],
            ],
            dtype=np.float64,
        )

    def coriolis_matrix(
        self, q: npt.NDArray[np.float64], dq: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)

        if q.shape != (3,):
            raise ValueError("q must be a 3-element vector")
        if dq.shape != (3,):
            raise ValueError("dq must be a 3-element vector")

        eps = 1e-5
        n = len(q)
        C = np.zeros((n, n), dtype=np.float64)

        for k in range(n):
            qp = q.copy()
            qm = q.copy()
            qp[k] += eps
            qm[k] -= eps

            dM_dk = (self.inertia_matrix(qp) - self.inertia_matrix(qm)) / (2 * eps)
            C += 0.5 * dM_dk * dq[k]

        return C

    def gravity_vector(self, q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (3,):
            raise ValueError("q must be a 3-element vector")

        q1, q2, q3 = q

        c1 = np.cos(q1)
        c12 = np.cos(q1 + q2)
        c123 = np.cos(q1 + q2 + q3)

        g1 = self.g_grav * (
            (self.m1 + self.m2 + self.m3) * self.l1 * c1
            + (self.m2 + self.m3) * self.l2 * c12
            + self.m3 * self.l3 * c123
        )
        g2 = self.g_grav * (
            (self.m2 + self.m3) * self.l2 * c12
            + self.m3 * self.l3 * c123
        )
        g3 = self.g_grav * self.m3 * self.l3 * c123

        return np.array([g1, g2, g3], dtype=np.float64)

    def integrate_dynamics(
        self,
        q: npt.NDArray[np.float64],
        dq: npt.NDArray[np.float64],
        tau: npt.NDArray[np.float64],
        dt: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        dt = float(dt)

        if q.shape != (3,):
            raise ValueError("q must be a 3-element vector")
        if dq.shape != (3,):
            raise ValueError("dq must be a 3-element vector")
        if tau.shape != (3,):
            raise ValueError("tau must be a 3-element vector")

        M_mat = self.inertia_matrix(q)
        C_mat = self.coriolis_matrix(q, dq)
        g_vec = self.gravity_vector(q)

        ddq = np.linalg.solve(M_mat, tau - C_mat @ dq - g_vec)

        dq_new = dq + ddq * dt
        q_new = q + dq_new * dt

        q_limits = np.array([np.deg2rad(145), np.deg2rad(145), np.deg2rad(145)], dtype=np.float64)
        q_new = np.clip(q_new, -q_limits, q_limits)
        dq_new = np.clip(dq_new, -3.0, 3.0)

        return q_new, dq_new