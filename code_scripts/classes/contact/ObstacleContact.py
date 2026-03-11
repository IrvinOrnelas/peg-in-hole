import numpy as np
import numpy.typing as npt


class ObstacleContact:

    def __init__(
        self,
        center: npt.NDArray[np.float64],
        radius: float,
        stiffness: float = 1500.0
    ):
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        self.k = float(stiffness)

    def compute_force(self, p: npt.NDArray[np.float64]):

        diff = p - self.center
        dist = np.linalg.norm(diff)

        if dist < self.radius:

            penetration = self.radius - dist

            if dist > 1e-6:
                normal = diff / dist
            else:
                normal = np.array([1.0, 0.0])

            F = self.k * penetration * normal

            return F, True

        return np.zeros(2), False