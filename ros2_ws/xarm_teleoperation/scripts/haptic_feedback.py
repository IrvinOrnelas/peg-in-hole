
import csv
import os
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from .kinematics import position_jacobian


class HapticFeedbackNode(Node):

    def __init__(self):

        super().__init__('haptic_feedback_node')

        self.beta = 0.3
        self.K_virtual = 50.0

        self.q_master = np.zeros(6)
        self.force_z = 0.0

        self.create_subscription(JointState,"/follower/joint_states" , self.joint_callback, 10)
        self.create_subscription(Float64,   "force_sensor/force", self.force_callback, 10)

        self.cmd_pub = self.create_publisher(JointState, '/master_cmd', 10)

        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info("Haptic feedback node started")

    def joint_callback(self, msg):

        if len(msg.position) >= 6:
            self.q_master = np.array(msg.position[:6])

    def force_callback(self, msg):

        self.force_z = msg.data


    def control_loop(self):

        q = self.q_master
        J = position_jacobian(q) 
        F = np.array([0.0, 0.0, self.force_z])
        tau = J.T @ (self.beta * F)
        dq = tau / self.K_virtual
        q_cmd = q - dq

        msg = JointState()
        msg.position = q_cmd.tolist()
        self.cmd_pub.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = HapticFeedbackNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()