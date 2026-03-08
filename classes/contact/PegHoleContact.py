import numpy as np
import numpy.typing as npt


class PegHoleContact:

    APPROACH = 0
    CONTACT = 1
    INSERTION = 2
    COMPLETE = 3

    STATE_NAMES = {
        APPROACH: "APPROACH",
        CONTACT: "CONTACT",
        INSERTION: "INSERTION",
        COMPLETE: "COMPLETE"
    }

    def __init__(
        self,
        hole_center: npt.NDArray[np.float64],
        peg_length: float,
        peg_radius: float,
        hole_radius: float,
        contact_k: float,
        force_threshold: float,
        hole_depth: float = 0.08
    ):
        self.hole_center = np.asarray(hole_center, dtype=np.float64)

        if self.hole_center.shape != (2,):
            raise ValueError("hole_center must be a 2-element vector")

        self.peg_length = np.float64(peg_length)
        self.peg_radius = np.float64(peg_radius)
        self.hole_radius = np.float64(hole_radius)

        self.contact_k = np.float64(contact_k)
        self.force_threshold = np.float64(force_threshold)
        self.hole_depth = np.float64(hole_depth)

        self.phase = self.APPROACH
        self.depth = np.float64(0.0)

        # Geometría del slot
        self.left_wall = self.hole_center[0] - self.hole_radius
        self.right_wall = self.hole_center[0] + self.hole_radius
        self.top_y = self.hole_center[1]
        self.bottom_y = self.hole_center[1] - self.hole_depth

    def compute_contact_force(self, p_ee: npt.NDArray[np.float64]):
        p_ee = np.asarray(p_ee, dtype=np.float64)

        if p_ee.shape != (2,):
            raise ValueError("p_ee must be a 2-element vector")

        x = p_ee[0]
        y = p_ee[1]

        F_contact = np.zeros(2, dtype=np.float64)
        in_contact = False

        # profundidad medida desde la entrada superior
        self.depth = max(np.float64(0.0), self.top_y - y)

        # radio efectivo libre dentro del agujero
        clearance = self.hole_radius - self.peg_radius
        if clearance < 0.0:
            clearance = np.float64(0.0)

        # límites efectivos para el centro del peg
        x_min = self.hole_center[0] - clearance
        x_max = self.hole_center[0] + clearance

        # zona vertical relevante
        near_hole_vertical = (y <= self.top_y + self.peg_length * 0.5) and (y >= self.bottom_y - self.peg_length)

        if not near_hole_vertical:
            self.phase = self.APPROACH
            return F_contact, self.STATE_NAMES[self.phase], in_contact

        inside_lateral = (x_min <= x <= x_max)
        below_top = y <= self.top_y
        above_bottom = y >= self.bottom_y

        # contacto con pared izquierda
        if x < x_min:
            penetration = x_min - x
            F_contact[0] += self.contact_k * penetration
            in_contact = True

        # contacto con pared derecha
        if x > x_max:
            penetration = x - x_max
            F_contact[0] -= self.contact_k * penetration
            in_contact = True

        # contacto con el fondo
        if y < self.bottom_y:
            penetration = self.bottom_y - y
            F_contact[1] += self.contact_k * penetration
            in_contact = True

        # lógica de estados
        if inside_lateral and below_top and above_bottom:
            if self.depth >= self.hole_depth * 0.85:
                self.phase = self.COMPLETE
            else:
                self.phase = self.INSERTION
        elif in_contact:
            self.phase = self.CONTACT
        else:
            self.phase = self.APPROACH

        return F_contact, self.STATE_NAMES[self.phase], in_contact