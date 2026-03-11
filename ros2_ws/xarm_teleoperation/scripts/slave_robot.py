#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from xarm.wrapper import XArmAPI
import time

class MasterRobotNode(Node):
    def __init__(self):
        super().__init__('master_robot_node')
        
        # 1. Connect to the Master xArm
        master_ip = '192.168.1.XXX' # REPLACE WITH REAL IP
        self.arm = XArmAPI(master_ip)
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0) # Mode 0 is standard position mode
        self.arm.set_state(0) # Sport state
        
        self.get_logger().info(f"Connected to Master xArm at {master_ip}")

        # 2. Setup ROS Publisher
        self.pub_q = self.create_publisher(Float64MultiArray, 'master/q_des', 10)
        
        # 3. Read loop (e.g., 100 Hz)
        self.timer = self.create_timer(0.01, self.read_and_publish)

    def read_and_publish(self):
        # Read joint states from the SDK
        code, joint_states = self.arm.get_joint_states(is_radian=True)
        
        if code == 0: # 0 means success
            msg = Float64MultiArray()
            # Extract just the 6 joint angles (ignore the velocities/efforts for now)
            msg.data = joint_states[0][:6] 
            self.pub_q.publish(msg)

    def destroy_node(self):
        self.arm.disconnect()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MasterRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()