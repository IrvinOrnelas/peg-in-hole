from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port_arg = DeclareLaunchArgument(
        "port",
        default_value="/dev/ttyUSB0",
        description="Serial port the Arduino is connected to (e.g. /dev/ttyUSB0 or /dev/ttyACM0)",
    )

    baud_rate_arg = DeclareLaunchArgument(
        "baud_rate",
        default_value="115200",
        description="Serial baud rate (must match the value in force_sensor.ino)",
    )

    frame_id_arg = DeclareLaunchArgument(
        "frame_id",
        default_value="force_sensor",
        description="TF frame id attached to sensor messages",
    )

    node = Node(
        package="force_sensor_reader",
        executable="force_sensor_node",
        name="force_sensor_node",
        output="screen",
        parameters=[
            {
                "port":      LaunchConfiguration("port"),
                "baud_rate": LaunchConfiguration("baud_rate"),
                "frame_id":  LaunchConfiguration("frame_id"),
            }
        ],
    )

    return LaunchDescription([port_arg, baud_rate_arg, frame_id_arg, node])
