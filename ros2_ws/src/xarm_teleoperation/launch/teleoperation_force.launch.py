from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    haptic_feedback_node = Node(
        package='xarm_teleoperation',
        executable='haptic_feedback.py',
        name='haptic_feedback_node',
        output='screen'
    )
    sensor_force_node = Node(
        package='force_sensor_reader',
        executable='force_sensor_node',
        name='sensor_force_node',
        output='screen'
    )

    return LaunchDescription([
        haptic_feedback_node,
        sensor_force_node,
    ])