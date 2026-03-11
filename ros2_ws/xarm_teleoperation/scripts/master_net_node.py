#!/usr/bin/env python3

import json
import socket
import threading
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray, String
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class MasterNetNode(Node):
    """
    ROS 2 Humble node for master-side communication with a slave process.

    Features:
    - Supports UDP or TCP
    - Subscribes to desired target p_des from ROS 2
    - Sends p_des to slave
    - Receives contact data from slave
    - Publishes contact data into ROS 2
    - Transport selected by parameter: 'udp' or 'tcp'
    """

    def __init__(self):
        super().__init__('master_net_node')

        # --------------------------------
        # Parameters
        # --------------------------------
        self.declare_parameter('transport', 'udp')         # 'udp' or 'tcp'
        self.declare_parameter('slave_ip', '127.0.0.1')
        self.declare_parameter('port_tx', 9001)
        self.declare_parameter('port_rx', 9002)
        self.declare_parameter('tcp_client_mode', True)    # True: master connects, False: master listens
        self.declare_parameter('recv_timeout_sec', 0.01)
        self.declare_parameter('send_period_sec', 0.02)    # 50 Hz

        self.transport = self.get_parameter('transport').get_parameter_value().string_value.lower()
        self.slave_ip = self.get_parameter('slave_ip').get_parameter_value().string_value
        self.port_tx = self.get_parameter('port_tx').get_parameter_value().integer_value
        self.port_rx = self.get_parameter('port_rx').get_parameter_value().integer_value
        self.tcp_client_mode = self.get_parameter('tcp_client_mode').get_parameter_value().bool_value
        self.recv_timeout_sec = self.get_parameter('recv_timeout_sec').get_parameter_value().double_value
        self.send_period_sec = self.get_parameter('send_period_sec').get_parameter_value().double_value

        if self.transport not in ('udp', 'tcp'):
            raise ValueError("Parameter 'transport' must be 'udp' or 'tcp'")

        # --------------------------------
        # Internal state
        # --------------------------------
        self.running = True

        self.q_des = np.zeros(6, dtype=np.float64)

        self.F_contact = np.zeros(3, dtype=np.float64)
        self.in_contact = False
        self.contact_state = "APPROACH"

        # UDP sockets
        self.udp_sock_tx: Optional[socket.socket] = None
        self.udp_sock_rx: Optional[socket.socket] = None

        # TCP sockets
        self.tcp_sock: Optional[socket.socket] = None
        self.tcp_server_sock: Optional[socket.socket] = None
        self.tcp_conn_addr: Optional[Tuple[str, int]] = None

        # --------------------------------
        # ROS interfaces
        # --------------------------------
        # Input target from ROS
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(
            JointState,
            '/leader/joint_states',
            self.q_des_cb,
            best_effort_qos
        )

        # Output received contact info to ROS
        self.contact_force_pub = self.create_publisher(Float64MultiArray, 'master/F_contact', 10)
        self.in_contact_pub = self.create_publisher(Bool, 'master/in_contact', 10)
        self.contact_state_pub = self.create_publisher(String, 'master/contact_state', 10)

        self.timer = self.create_timer(self.send_period_sec, self.timer_cb)

        # --------------------------------
        # Network setup
        # --------------------------------
        self._setup_network()

        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()

        self.get_logger().info(
            f"MasterNetNode started | transport={self.transport} | "
            f"slave_ip={self.slave_ip} | tx={self.port_tx} | rx={self.port_rx}"
        )

    # ============================================================
    # ROS callbacks
    # ============================================================
    def q_des_cb(self, msg):
        if len(msg.position) >= 6:
            self.q_des = np.array(msg.position[:6], dtype=np.float64)
        else:
            self.get_logger().warn("Received master/q_des with less than 6 elements")

    def timer_cb(self):
        # Send q_des instead of p_des
        self.send_target(self.q_des)
        self.publish_contact_data()

    def publish_contact_data(self):
        f_msg = Float64MultiArray()
        f_msg.data = self.F_contact.tolist()
        self.contact_force_pub.publish(f_msg)

        c_msg = Bool()
        c_msg.data = self.in_contact
        self.in_contact_pub.publish(c_msg)

        s_msg = String()
        s_msg.data = self.contact_state
        self.contact_state_pub.publish(s_msg)

    # ============================================================
    # Network setup
    # ============================================================
    def _setup_network(self):
        if self.transport == 'udp':
            self._setup_udp()
        else:
            self._setup_tcp()

    def _setup_udp(self):
        self.udp_sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock_rx.bind(('', self.port_rx))
        self.udp_sock_rx.settimeout(self.recv_timeout_sec)

        self.get_logger().info(
            f"UDP master ready | sending to {self.slave_ip}:{self.port_tx} | listening on 0.0.0.0:{self.port_rx}"
        )

    def _setup_tcp(self):
        if self.tcp_client_mode:
            self._connect_tcp_client()
        else:
            self.tcp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_server_sock.bind(('', self.port_rx))
            self.tcp_server_sock.listen(1)
            self.tcp_server_sock.settimeout(1.0)
            self.get_logger().info(f"TCP server listening on 0.0.0.0:{self.port_rx}")

    def _connect_tcp_client(self):
        while rclpy.ok() and self.running:
            try:
                self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_sock.settimeout(2.0)
                self.tcp_sock.connect((self.slave_ip, self.port_tx))
                self.tcp_sock.settimeout(self.recv_timeout_sec)
                self.get_logger().info(f"Connected to TCP slave at {self.slave_ip}:{self.port_tx}")
                return
            except Exception as e:
                self.get_logger().warn(f"TCP connect failed: {e}")
                time.sleep(1.0)

    # ============================================================
    # Sending
    # ============================================================
    def send_target(self, q_des: np.ndarray):
        q_des = np.asarray(q_des, dtype=np.float64)

        # Check for 6 elements
        if q_des.shape != (6,):
            self.get_logger().warn("q_des must be a 6-element vector")
            return

        # Update JSON payload key
        msg = json.dumps({
            "type": "target",
            "q_des": q_des.tolist()
        })

        try:
            if self.transport == 'udp':
                if self.udp_sock_tx is None:
                    return
                self.udp_sock_tx.sendto(msg.encode(), (self.slave_ip, self.port_tx))
            else:
                if self.tcp_client_mode:
                    if self.tcp_sock is None:
                        return
                    self.tcp_sock.sendall((msg + '\n').encode())
                else:
                    if self.tcp_sock is None:
                        return
                    self.tcp_sock.sendall((msg + '\n').encode())

        except Exception as e:
            self.get_logger().warn(f"Failed to send target: {e}")
            if self.transport == 'tcp':
                try:
                    if self.tcp_sock is not None:
                        self.tcp_sock.close()
                except Exception:
                    pass
                self.tcp_sock = None

    # ============================================================
    # Receive loops
    # ============================================================
    def _recv_loop(self):
        if self.transport == 'udp':
            self._recv_loop_udp()
        else:
            self._recv_loop_tcp()

    def _recv_loop_udp(self):
        while rclpy.ok() and self.running:
            try:
                data, _ = self.udp_sock_rx.recvfrom(1024)
                self._process_incoming_message(data)
            except socket.timeout:
                pass
            except Exception as e:
                self.get_logger().warn(f"UDP recv error: {e}")

    def _recv_loop_tcp(self):
        while rclpy.ok() and self.running:
            try:
                if self.tcp_client_mode:
                    if self.tcp_sock is None:
                        self._connect_tcp_client()
                        continue

                    data = self.tcp_sock.recv(1024)
                    if not data:
                        self.get_logger().warn("TCP slave disconnected, reconnecting...")
                        self.tcp_sock.close()
                        self.tcp_sock = None
                        continue

                    for line in data.splitlines():
                        if line.strip():
                            self._process_incoming_message(line)

                else:
                    if self.tcp_sock is None:
                        try:
                            conn, addr = self.tcp_server_sock.accept()
                            conn.settimeout(self.recv_timeout_sec)
                            self.tcp_sock = conn
                            self.tcp_conn_addr = addr
                            self.get_logger().info(f"TCP client connected from {addr}")
                        except socket.timeout:
                            continue

                    data = self.tcp_sock.recv(1024)
                    if not data:
                        self.get_logger().warn("TCP client disconnected")
                        self.tcp_sock.close()
                        self.tcp_sock = None
                        self.tcp_conn_addr = None
                        continue

                    for line in data.splitlines():
                        if line.strip():
                            self._process_incoming_message(line)

            except socket.timeout:
                pass
            except Exception as e:
                self.get_logger().warn(f"TCP recv error: {e}")
                try:
                    if self.tcp_sock is not None:
                        self.tcp_sock.close()
                except Exception:
                    pass
                self.tcp_sock = None
                time.sleep(0.5)

    # ============================================================
    # Message parsing
    # ============================================================
    def _process_incoming_message(self, raw_data: bytes):
        try:
            parsed = json.loads(raw_data.decode())

            msg_type = parsed.get("type", "contact")
            if msg_type != "contact":
                return

            F_contact = np.asarray(parsed["F_contact"], dtype=np.float64)
            if F_contact.shape != (2,):
                raise ValueError("F_contact must be a 2-element vector")

            self.F_contact = F_contact
            self.in_contact = bool(parsed["in_contact"])
            self.contact_state = str(parsed["contact_state"])

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            self.get_logger().warn(f"Invalid incoming message: {e}")

    # ============================================================
    # Shutdown
    # ============================================================
    def destroy_node(self):
        self.running = False

        try:
            if self.udp_sock_tx is not None:
                self.udp_sock_tx.close()
        except Exception:
            pass

        try:
            if self.udp_sock_rx is not None:
                self.udp_sock_rx.close()
        except Exception:
            pass

        try:
            if self.tcp_sock is not None:
                self.tcp_sock.close()
        except Exception:
            pass

        try:
            if self.tcp_server_sock is not None:
                self.tcp_server_sock.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MasterNetNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()