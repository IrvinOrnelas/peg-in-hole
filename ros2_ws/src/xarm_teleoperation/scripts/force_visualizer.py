#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

import matplotlib.pyplot as plt
import time


class ForceVisualizer(Node):

    def __init__(self):

        super().__init__('force_visualizer')

        self.force = 0.0
        self.threshold = 1.0

        self.t_data = []
        self.f_data = []
        self.contact_times = []

        self.start_time = time.time()

        self.in_contact = False

        self.create_subscription(
            Float64,
            "/force_sensor/force",
            self.force_callback,
            10
        )

        self.get_logger().info("Force visualizer started")

        # Matplotlib setup
        plt.ion()
        self.fig, self.ax = plt.subplots()

        self.line, = self.ax.plot([], [], label="Force (N)")
        self.ax.axhline(
            y=self.threshold,
            linestyle="--",
            label="Contact threshold"
        )

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force (N)")
        self.ax.set_title("Force Sensor Contact Detection")

        self.ax.legend()
        self.ax.grid()

        # Timer para actualizar gráfica
        self.create_timer(0.05, self.update_plot)

    def force_callback(self, msg):

        self.force = msg.data
        t = time.time() - self.start_time

        self.t_data.append(t)
        self.f_data.append(self.force)

        if abs(self.force) > self.threshold and not self.in_contact:

            self.get_logger().info("CONTACT EVENT DETECTED")

            self.contact_times.append(t)
            self.in_contact = True

        elif abs(self.force) <= self.threshold:

            self.in_contact = False

    def update_plot(self):

        if len(self.t_data) == 0:
            return

        self.line.set_data(self.t_data, self.f_data)

        for ct in self.contact_times:
            self.ax.axvline(ct, color="red", linestyle="--")

        self.ax.set_xlim(max(0, self.t_data[-1] - 10), self.t_data[-1] + 1)

        self.ax.set_ylim(min(self.f_data + [-1]), max(self.f_data + [1]))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):

    rclpy.init(args=args)

    node = ForceVisualizer()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()