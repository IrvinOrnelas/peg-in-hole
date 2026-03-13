ros2 service call /leader/motion_enable xarm_msgs/srv/SetInt16ById "{id: 8, data: 1}"
ros2 service call /leader/set_mode xarm_msgs/srv/SetInt16 "{data: 2}"
ros2 service call /leader/set_mode xarm_msgs/srv/SetInt16 "{data: 2}"
ros2 service call /leader/set_state xarm_msgs/srv/SetInt16 "{data: 0}"