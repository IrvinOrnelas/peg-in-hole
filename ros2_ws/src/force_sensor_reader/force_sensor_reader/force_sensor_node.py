#!/usr/bin/env python3
"""
ROS2 Humble node that reads force & compression data from the force_sensor.ino
Arduino sketch over USB serial and publishes it as ROS2 topics.

Serial format (115200 baud, ~10 Hz):
    "Fuerza medida: X.XX N  Compresion medida: Y.YY"

Published topics:
    /force_sensor/force       [std_msgs/Float32]   Force in Newtons
    /force_sensor/compression [std_msgs/Float32]   Normalized compression [0-1]
    /force_sensor/status      [std_msgs/String]    Raw line (debug / calibration msgs)
"""

import re
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64, String

try:
    import serial
except ImportError:
    raise SystemExit(
        "pyserial is not installed. Install it with: pip install pyserial"
    )

# Regex that matches the data lines produced by the Arduino sketch
_DATA_RE = re.compile(
    r"Fuerza medida:\s*([\d.]+)\s*N\s+Compresion medida:\s*([\d.]+)",
    re.IGNORECASE,
)


class ForceSensorNode(Node):
    def __init__(self):
        super().__init__("force_sensor_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("timeout", 1.0)
        self.declare_parameter("frame_id", "force_sensor")

        port      = self.get_parameter("port").get_parameter_value().string_value
        baud_rate = self.get_parameter("baud_rate").get_parameter_value().integer_value
        timeout   = self.get_parameter("timeout").get_parameter_value().double_value
        self._frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        # ── QoS ───────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_force = self.create_publisher(
            Float64, "force_sensor/force", sensor_qos
        )
        self._pub_compression = self.create_publisher(
            Float64, "force_sensor/compression", sensor_qos
        )
        self._pub_status = self.create_publisher(
            String, "force_sensor/status", 10
        )

        # ── Serial port ───────────────────────────────────────────────────────
        self._serial: serial.Serial | None = None
        self._open_serial(port, baud_rate, timeout)

        # Read in a background thread so the ROS2 executor is never blocked
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f"force_sensor_node started – port={port}, baud={baud_rate}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Serial helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _open_serial(self, port: str, baud_rate: int, timeout: float) -> None:
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=baud_rate,
                timeout=timeout,
            )
            self.get_logger().info(f"Opened serial port {port}")
        except serial.SerialException as exc:
            self.get_logger().error(f"Cannot open serial port {port}: {exc}")
            self._serial = None

    # ──────────────────────────────────────────────────────────────────────────
    # Background reader
    # ──────────────────────────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        """Continuously read lines from serial and publish parsed values."""
        while self._running and rclpy.ok():
            if self._serial is None or not self._serial.is_open:
                self.get_logger().warning("Serial port not available, waiting...")
                import time; time.sleep(1.0)
                continue

            try:
                raw = self._serial.readline()
            except serial.SerialException as exc:
                self.get_logger().error(f"Serial read error: {exc}")
                self._serial = None
                continue

            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            self._process_line(line)

    def _process_line(self, line: str) -> None:
        """Try to parse a data line; fall back to publishing it as status."""
        match = _DATA_RE.search(line)
        if match:
            try:
                force       = float(match.group(1))
                compression = float(match.group(2))
            except ValueError:
                self.get_logger().warning(f"Could not parse floats in: '{line}'")
                return

            force_msg       = Float64(data=force)
            compression_msg = Float64(data=compression)

            self._pub_force.publish(force_msg)
            self._pub_compression.publish(compression_msg)

            self.get_logger().debug(
                f"Force={force:.4f} N  Compression={compression:.4f}"
            )
        else:
            # Calibration / info messages from setup()
            if line:
                status_msg = String(data=line)
                self._pub_status.publish(status_msg)
                self.get_logger().info(f"[Arduino] {line}")

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()
            self.get_logger().info("Serial port closed.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ForceSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
