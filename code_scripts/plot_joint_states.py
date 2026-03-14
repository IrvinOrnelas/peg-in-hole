#!/usr/bin/env python3
"""
Compare leader vs follower joint_states from MCAP rosbag.
Plots position, velocity, and effort for all 6 joints over time.
Contact intervals are derived from /force_sensor/force using the same
thresholds as haptic_feedback.py: contact when 1.0 < |force| < 10.0 N.
"""

import numpy as np
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

BAG_PATH = "teleop_haptic/teleop_haptic/teleop_haptic_0.mcap"
LEADER_TOPIC = "/leader/joint_states"
FOLLOWER_TOPIC = "/follower/joint_states"
FORCE_TOPIC = "/force_sensor/force"

# Contact thresholds matching haptic_feedback.py
FORCE_MIN = 1.0   # N
FORCE_MAX = 10.0  # N


def _open_bag(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    return reader


def read_joint_states(bag_path: str, topic: str) -> dict:
    """Read all JointState messages from a topic. Times are absolute epoch seconds."""
    reader = _open_bag(bag_path)
    times, positions, velocities, efforts = [], [], [], []

    while reader.has_next():
        t_topic, data, _ = reader.read_next()
        if t_topic != topic:
            continue
        msg = deserialize_message(data, JointState)
        times.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        positions.append(list(msg.position) if msg.position else [np.nan] * 6)
        velocities.append(list(msg.velocity) if msg.velocity else [np.nan] * 6)
        efforts.append(list(msg.effort) if msg.effort else [np.nan] * 6)

    return {
        "time": np.array(times),
        "position": np.array(positions),
        "velocity": np.array(velocities),
        "effort": np.array(efforts),
    }


def align_times(leader: dict, follower: dict) -> float:
    """Normalize both time arrays to a common t=0. Returns global_t0 (epoch s)."""
    global_t0 = min(leader["time"][0], follower["time"][0])
    leader["time"] -= global_t0
    follower["time"] -= global_t0
    return global_t0


def read_contact_intervals(bag_path: str, global_t0: float) -> list:
    """Return [(t_start, t_end), ...] intervals when force sensor detects contact.
    Contact is defined as FORCE_MIN < |force| < FORCE_MAX, matching haptic_feedback.py.
    Uses bag receive timestamps since /force_sensor/force has no header stamp.
    """
    reader = _open_bag(bag_path)
    intervals = []
    in_contact = False
    t_start = None
    t_last = 0.0

    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        if topic != FORCE_TOPIC:
            continue
        msg = deserialize_message(data, Float64)
        t = ts_ns * 1e-9 - global_t0
        t_last = t
        force = abs(msg.data)
        is_contact = FORCE_MIN < force < FORCE_MAX

        if is_contact and not in_contact:
            t_start = t
            in_contact = True
        elif not is_contact and in_contact:
            intervals.append((t_start, t))
            in_contact = False

    if in_contact:
        intervals.append((t_start, t_last))

    return intervals


def shade_contact(ax, intervals):
    """Shade contact intervals in red on an axes. Adds legend entry once."""
    for i, (t0, t1) in enumerate(intervals):
        ax.axvspan(t0, t1, color="red", alpha=0.18,
                   label="In contact" if i == 0 else None)


def compute_error(leader: dict, follower: dict) -> np.ndarray:
    """Interpolate follower onto leader timestamps, return position error array (N,6)."""
    errors = np.full_like(leader["position"], np.nan)
    for j in range(6):
        errors[:, j] = np.interp(
            leader["time"], follower["time"], follower["position"][:, j],
            left=np.nan, right=np.nan,
        ) - leader["position"][:, j]
    return errors


def plot_field(leader, follower, field, unit, title_prefix, out_file, contact_intervals):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{title_prefix} — Leader vs Follower", fontsize=14, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        shade_contact(ax, contact_intervals)
        ax.plot(leader["time"], leader[field][:, idx],
                label="Leader", linewidth=1.2, color="#1f77b4", zorder=3)
        ax.plot(follower["time"], follower[field][:, idx],
                label="Follower", linewidth=1.2, color="#ff7f0e", linestyle="--", zorder=3)
        ax.set_title(f"Joint {idx + 1}")
        ax.set_ylabel(unit)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")


def read_force(bag_path: str, global_t0: float) -> dict:
    """Read /force_sensor/force. Uses bag receive timestamps (no header stamp)."""
    reader = _open_bag(bag_path)
    times, values = [], []
    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        if topic != FORCE_TOPIC:
            continue
        msg = deserialize_message(data, Float64)
        times.append(ts_ns * 1e-9 - global_t0)
        values.append(msg.data)
    return {"time": np.array(times), "force": np.array(values)}


def plot_force(force_data, out_file, contact_intervals):
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle("Force Sensor — Raw Signal with Contact Zones", fontsize=14, fontweight="bold")

    shade_contact(ax, contact_intervals)
    ax.plot(force_data["time"], force_data["force"],
            color="#2ca02c", linewidth=1.2, label="Force (N)", zorder=3)

    ax.axhline(FORCE_MIN, color="orange", linewidth=1.2, linestyle="--",
               label=f"Contact threshold min ({FORCE_MIN} N)")
    ax.axhline(FORCE_MAX, color="red",    linewidth=1.2, linestyle="--",
               label=f"Contact threshold max ({FORCE_MAX} N)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")


def plot_error(leader, follower, out_file, contact_intervals):
    errors = compute_error(leader, follower)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Tracking Error (Follower − Leader) — Joint Positions",
                 fontsize=14, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        shade_contact(ax, contact_intervals)
        ax.plot(leader["time"], np.degrees(errors[:, idx]),
                color="#d62728", linewidth=1.0, zorder=3)
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        rms = np.sqrt(np.nanmean(errors[:, idx] ** 2))
        ax.set_title(f"Joint {idx + 1}  (RMS = {np.degrees(rms):.3f}°)")
        ax.set_ylabel("Error (deg)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")


def main():
    print("Reading leader joint states...")
    leader = read_joint_states(BAG_PATH, LEADER_TOPIC)
    print(f"  {len(leader['time'])} messages")

    print("Reading follower joint states...")
    follower = read_joint_states(BAG_PATH, FOLLOWER_TOPIC)
    print(f"  {len(follower['time'])} messages")

    print("Aligning timestamps...")
    global_t0 = align_times(leader, follower)

    print("Reading contact intervals from force sensor...")
    contact_intervals = read_contact_intervals(BAG_PATH, global_t0)
    print(f"  {len(contact_intervals)} contact interval(s) found")
    for i, (a, b) in enumerate(contact_intervals):
        print(f"    [{i+1}] t={a:.2f}s → {b:.2f}s  (duration={b-a:.2f}s)")

    print("Reading raw force signal...")
    force_data = read_force(BAG_PATH, global_t0)
    print(f"  {len(force_data['time'])} messages")

    # --- Plots ---
    plot_force(force_data, "force_sensor.png", contact_intervals)
    plot_field(leader, follower, "position", "rad",
               "Joint Positions", "joint_positions.png", contact_intervals)

    has_vel = not np.all(np.isnan(leader["velocity"]))
    if has_vel:
        plot_field(leader, follower, "velocity", "rad/s",
                   "Joint Velocities", "joint_velocities.png", contact_intervals)

    plot_error(leader, follower, "joint_tracking_error.png", contact_intervals)

    # --- Summary stats ---
    errors = compute_error(leader, follower)
    print("\n--- RMS Tracking Error per Joint ---")
    for j in range(6):
        rms = np.degrees(np.sqrt(np.nanmean(errors[:, j] ** 2)))
        peak = np.degrees(np.nanmax(np.abs(errors[:, j])))
        print(f"  Joint {j+1}: RMS = {rms:.4f}°   Peak = {peak:.4f}°")

    plt.show()


if __name__ == "__main__":
    main()
