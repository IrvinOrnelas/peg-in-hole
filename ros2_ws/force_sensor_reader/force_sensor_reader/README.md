# force_sensor_reader

ROS2 Humble node that reads force and compression data from the `force_sensor.ino`
Arduino sketch over USB serial and publishes them as ROS2 topics.

## Published Topics

| Topic | Type | Description |
|---|---|---|
| `/force_sensor/force` | `std_msgs/Float32` | Force in Newtons |
| `/force_sensor/compression` | `std_msgs/Float32` | Normalized compression [0–1] |
| `/force_sensor/status` | `std_msgs/String` | Raw calibration/status lines |

## Launch Parameters

| Parameter | Default | Description |
|---|---|---|
| `port` | `/dev/ttyUSB0` | Serial port (try `/dev/ttyACM0` for Uno/Mega) |
| `baud_rate` | `115200` | Must match `Serial.begin()` in the sketch |
| `frame_id` | `force_sensor` | TF frame id for sensor messages |

---

## Setup

All commands below assume you are at the **root of the cloned repository**:

```bash
git clone <repo-url>
cd <repo-name>
```

### 1. Serial port permissions

Grant your user access to the serial port, then **log out and back in**:

```bash
sudo usermod -aG dialout $USER
```

### 2. Create the virtual environment

The venv **must** use `--system-site-packages` so it can see the ROS2 Python
packages (`rclpy`, `std_msgs`, etc.) installed system-wide.

```bash
python3 -m venv venv --system-site-packages
```

### 3. Activate the venv and install dependencies

```bash
source venv/bin/activate
pip install -r force_sensor_reader/requirements.txt
```

### 4. Build the package

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select force_sensor_reader
source install/setup.bash
```

### 5. Run

```bash
# Default port (/dev/ttyUSB0)
ros2 launch force_sensor_reader force_sensor.launch.py

# Override the port if needed
ros2 launch force_sensor_reader force_sensor.launch.py port:=/dev/ttyACM0
```

---

## Optional: auto-activate on every terminal session

Add these lines to your `~/.bashrc`:

```bash
source /opt/ros/humble/setup.bash
source <path/to/repo>/venv/bin/activate
```
