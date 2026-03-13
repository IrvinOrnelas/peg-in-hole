ros2 service call /follower/motion_enable xarm_msgs/srv/SetInt16ById "{id: 8, data: 1}"
ros2 service call /follower/set_mode xarm_msgs/srv/SetInt16 "{data: 0}"
ros2 service call /follower/set_mode xarm_msgs/srv/SetInt16 "{data: 0}"
ros2 service call /follower/set_state xarm_msgs/srv/SetInt16 "{data: 0}"