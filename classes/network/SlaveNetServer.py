import socket
import threading
import json
import numpy as np
import numpy.typing as npt


class SlaveNetServer:

    def __init__(
        self,
        master_ip: str = "127.0.0.1",
        port_rx: int = 9001,
        port_tx: int = 9002
    ):
        self.master_ip = master_ip
        self.port_rx = port_rx
        self.port_tx = port_tx

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", self.port_rx))
        self.sock.settimeout(0.005)

        self.p_des = np.zeros(2, dtype=np.float64)
        self.has_received_target = False

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(512)
                parsed = json.loads(data.decode())

                self.p_des = np.asarray(parsed["p_des"], dtype=np.float64)
                self.master_addr = addr
                self.has_received_target = True

            except (socket.timeout, json.JSONDecodeError, KeyError, ValueError, AttributeError):
                pass

    def send_contact_data(
        self,
        F_contact: npt.NDArray[np.float64],
        in_contact: bool,
        contact_state: str
    ):
        F_contact = np.asarray(F_contact, dtype=np.float64)

        if F_contact.shape != (2,):
            raise ValueError("F_contact must be a 2-element vector")

        msg = json.dumps({
            "F_contact": F_contact.tolist(),
            "in_contact": int(in_contact),
            "contact_state": contact_state
        })

        try:
            self.sock.sendto(msg.encode(), (self.master_ip, self.port_tx))
        except Exception:
            pass