import socket
import threading
import json
import numpy as np
import numpy.typing as npt


class MasterNetClient:

    def __init__(
        self,
        slave_ip: str,
        port_tx: int = 9001,
        port_rx: int = 9002
    ):
        self.slave_ip = slave_ip
        self.port_tx = port_tx
        self.port_rx = port_rx

        self.sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.sock_rx.bind(("", self.port_rx))
        self.sock_rx.settimeout(0.005)

        self.F_contact = np.zeros(2, dtype=np.float64)
        self.in_contact = False
        self.contact_state = "APPROACH"

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def send_target(self, p_des: npt.NDArray[np.float64]):
        p_des = np.asarray(p_des, dtype=np.float64)

        if p_des.shape != (2,):
            raise ValueError("p_des must be a 2-element vector")

        msg = json.dumps({
            "p_des": p_des.tolist()
        })

        self.sock_tx.sendto(msg.encode(), (self.slave_ip, self.port_tx))

    def _recv_loop(self):
        while True:
            try:
                data, _ = self.sock_rx.recvfrom(512)
                parsed = json.loads(data.decode())

                self.F_contact = np.asarray(parsed["F_contact"], dtype=np.float64)
                self.in_contact = bool(parsed["in_contact"])
                self.contact_state = str(parsed["contact_state"])

            except (socket.timeout, json.JSONDecodeError, KeyError, ValueError):
                pass