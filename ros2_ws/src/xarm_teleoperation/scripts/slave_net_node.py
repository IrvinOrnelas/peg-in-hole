#!/usr/bin/env python3

import socket
import threading
import json
import time
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool, String
from xarm_msgs.srv import MoveJoint


class SlaveNetServer:
    def __init__(
        self,
        master_ip: str = "127.0.0.1",
        port_rx: int = 9001,
        port_tx: int = 9002,
        transport: str = "udp",
        tcp_server_mode: bool = True,
        timeout: float = 0.005
    ):
        self.master_ip = master_ip
        self.port_rx = port_rx
        self.port_tx = port_tx
        self.transport = transport.lower()
        self.tcp_server_mode = tcp_server_mode
        self.timeout = timeout

        if self.transport not in ("udp", "tcp"):
            raise ValueError("transport must be 'udp' or 'tcp'")

        self.q_des = np.zeros(6, dtype=np.float64)
        self.has_received_target = False
        self.master_addr: Optional[Tuple[str, int]] = None

        self._running = True

        # UDP
        self.sock: Optional[socket.socket] = None

        # TCP
        self.server_sock: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None

        self._setup_socket()

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _setup_socket(self):
        if self.transport == "udp":
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("", self.port_rx))
            self.sock.settimeout(self.timeout)

        elif self.transport == "tcp":
            if self.tcp_server_mode:
                self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_sock.bind(("", self.port_rx))
                self.server_sock.listen(1)
                self.server_sock.settimeout(1.0)
            else:
                self._connect_as_client()

    def _connect_as_client(self):
        while self._running:
            try:
                self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.conn.settimeout(2.0)
                self.conn.connect((self.master_ip, self.port_rx))
                self.conn.settimeout(self.timeout)
                return
            except Exception:
                time.sleep(1.0)

    def _recv_loop(self):
        if self.transport == "udp":
            self._recv_loop_udp()
        else:
            self._recv_loop_tcp()

    def _recv_loop_udp(self):
        while self._running:
            try:
                data, addr = self.sock.recvfrom(512)
                parsed = json.loads(data.decode())

                self.q_des = np.asarray(parsed["q_des"], dtype=np.float64)
                if self.q_des.shape != (6,):
                    raise ValueError("q_des must be a 6-element vector")

                self.master_addr = addr
                self.has_received_target = True

            except (socket.timeout, json.JSONDecodeError, KeyError, ValueError, AttributeError):
                pass
            except Exception:
                pass

    def _recv_loop_tcp(self):
        buffer = b""

        while self._running:
            try:
                if self.tcp_server_mode:
                    if self.conn is None:
                        try:
                            self.conn, self.master_addr = self.server_sock.accept()
                            self.conn.settimeout(self.timeout)
                        except socket.timeout:
                            continue
                else:
                    if self.conn is None:
                        self._connect_as_client()
                        continue

                data = self.conn.recv(1024)
                if not data:
                    self._close_tcp_connection()
                    continue

                buffer += data

                while b"\n" in buffer:
                    raw_msg, buffer = buffer.split(b"\n", 1)
                    if not raw_msg.strip():
                        continue

                    parsed = json.loads(raw_msg.decode())
                    self.q_des = np.asarray(parsed["q_des"], dtype=np.float64)
                    
                    if self.q_des.shape != (6,):
                        raise ValueError("q_des must be a 6-element vector")

                    self.has_received_target = True
                    
                    #print(f"Received TCP: {parsed}")

            except (socket.timeout, json.JSONDecodeError, KeyError, ValueError, AttributeError):
                pass
            except Exception:
                self._close_tcp_connection()
                time.sleep(0.2)

    def _close_tcp_connection(self):
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass
        self.conn = None

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
            if self.transport == "udp":
                self.sock.sendto(msg.encode(), (self.master_ip, self.port_tx))
            else:
                if self.conn is not None:
                    self.conn.sendall((msg + "\n").encode())
        except Exception:
            pass

    def shutdown(self):
        self._running = False

        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass

        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass

        try:
            if self.server_sock is not None:
                self.server_sock.close()
        except Exception:
            pass


class SlaveNetNode(Node):
    def __init__(self):
        super().__init__("slave_net_node")

        # Parameters
        self.declare_parameter("master_ip", "127.0.0.1")
        self.declare_parameter("port_rx", 9001)
        self.declare_parameter("port_tx", 9002)
        self.declare_parameter("transport", "udp")
        self.declare_parameter("tcp_server_mode", True)

        master_ip = self.get_parameter("master_ip").value
        port_rx = self.get_parameter("port_rx").value
        port_tx = self.get_parameter("port_tx").value
        transport = self.get_parameter("transport").value
        tcp_server_mode = self.get_parameter("tcp_server_mode").value

        self.net = SlaveNetServer(
            master_ip=master_ip,
            port_rx=port_rx,
            port_tx=port_tx,
            transport=transport,
            tcp_server_mode=tcp_server_mode
        )

        # Internal contact state
        self.F_contact = np.zeros(2, dtype=np.float64)
        self.in_contact = False
        self.contact_state = "no_contact"
        
        # Services
        self.follower_client = self.create_client(MoveJoint, '/follower/set_servo_angle')
        
        while not self.follower_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando al servicio del brazo follower...')
            
        self.waiting_for_response = False
        self.message_counter = 0
        

        # Publishers

        # Subscribers
        self.create_subscription(Float64MultiArray, "/slave/contact_force", self.cb_force, 10)
        self.create_subscription(Bool, "/slave/in_contact", self.cb_in_contact, 10)
        self.create_subscription(String, "/slave/contact_state", self.cb_contact_state, 10)

        # Timer
        self.create_timer(0.02, self.update_loop)  # 50 Hz

        self.get_logger().info(
            f"SlaveNetNode running with transport={transport}, "
            f"master_ip={master_ip}, port_rx={port_rx}, port_tx={port_tx}"
        )

    def cb_force(self, msg: Float64MultiArray):
        arr = np.asarray(msg.data, dtype=np.float64)
        if arr.shape[0] >= 2:
            self.F_contact = arr[:2]

    def cb_in_contact(self, msg: Bool):
        self.in_contact = bool(msg.data)

    def cb_contact_state(self, msg: String):
        self.contact_state = msg.data

    def update_loop(self):
        # Publish received p_des
        self.net.send_contact_data(
            F_contact=self.F_contact,
            in_contact=self.in_contact,
            contact_state=self.contact_state
        )

        # 2. Lógica para mover el robot esclavo con los datos recibidos por red
        if not self.net.has_received_target or self.waiting_for_response:
            return  # Si no hay datos de red o estamos esperando al robot, abortamos

        request = MoveJoint.Request()
        # Tomamos los ángulos que llegaron por el socket TCP/UDP
        request.angles = self.net.q_des.tolist() 
        request.speed = 0.0
        request.acc = 0.0
        request.mvtime = 0.0

        # Bloqueamos el envío de nuevos comandos
        self.waiting_for_response = True

        # Enviamos el comando asíncrono
        future = self.follower_client.call_async(request)
        future.add_done_callback(self.service_done_callback)

        self.message_counter += 1
        if self.message_counter % 10 == 0:
            formatted_angles = [round(angle, 3) for angle in request.angles]
            self.get_logger().info(f'Enviados por red {self.message_counter}. Ángulos: {formatted_angles}')
            
    def service_done_callback(self, future):
        """Se ejecuta automáticamente cuando el robot esclavo confirma que recibió el comando."""
        self.waiting_for_response = False

    def destroy_node(self):
        self.net.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SlaveNetNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()