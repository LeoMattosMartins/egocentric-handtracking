import argparse
import json
import math
from pathlib import Path

import numpy as np

from retarget_hand import estimate_joint_targets, imu_attitude, landmark_points, nearest_imu, normalize_hand, palm_frame
from unitree_targets import DEX3_JOINTS, dex3_targets_from_retargeted


def iter_coordinate_frames(coordinates):
    if "frames" in coordinates:
        for frame_key, frame_data in coordinates["frames"].items():
            yield int(frame_key), frame_data.get("timestamp_ms"), frame_data.get("hands", [])
    else:
        for frame_key, hands in coordinates.items():
            yield int(frame_key), None, hands


def choose_hand(hands, preferred_label):
    if not hands:
        return None
    if preferred_label:
        matches = [hand for hand in hands if hand.get("label") == preferred_label]
        if matches:
            return matches[0]
    return hands[0]


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return quat_normalize(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_to_euler_np(q):
    w, x, y, z = quat_normalize(q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def quat_multiply_np(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return quat_normalize(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_inverse(q):
    q = quat_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def matrix_to_quat(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return quat_normalize(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ]
        )
    axis = int(np.argmax(np.diag(matrix)))
    if axis == 0:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return quat_normalize(
            [
                (matrix[2, 1] - matrix[1, 2]) / s,
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
            ]
        )
    if axis == 1:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return quat_normalize(
            [
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
            ]
        )
    s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return quat_normalize(
        [
            (matrix[1, 0] - matrix[0, 1]) / s,
            (matrix[0, 2] + matrix[2, 0]) / s,
            (matrix[1, 2] + matrix[2, 1]) / s,
            0.25 * s,
        ]
    )


def palm_orientation_quat(hand):
    points = normalize_hand(landmark_points(hand))
    palm_x, palm_y, palm_z = palm_frame(points)
    matrix = np.asarray([palm_x, palm_y, palm_z], dtype=np.float64).T
    return matrix_to_quat(matrix)


def palm_delta_root_quat(palm_quat, palm_reference, max_roll=0.85, max_pitch=0.38, max_yaw=0.55):
    delta = quat_multiply_np(palm_quat, quat_inverse(palm_reference))
    roll, pitch, yaw = quat_to_euler_np(delta)
    roll = max(-max_roll, min(max_roll, roll))
    pitch = max(-max_pitch, min(max_pitch, pitch))
    yaw = max(-max_yaw, min(max_yaw, yaw))
    return euler_to_quat(roll, pitch, yaw)


def wrist_root_position(hand):
    wrist = next((lm for lm in hand["landmarks"] if lm["id"] == 0), None)
    if wrist is None:
        return [np.nan, np.nan, np.nan]
    return [
        (float(wrist["x"]) - 0.5) * 0.45,
        0.0,
        (0.5 - float(wrist["y"])) * 0.35 + 0.05,
    ]


def interpolate_missing(values):
    values = values.astype(np.float64)
    x = np.arange(values.shape[0])
    out = values.copy()
    for col in range(values.shape[1]):
        valid = np.isfinite(values[:, col])
        if valid.sum() == 0:
            out[:, col] = 0.0
        elif valid.sum() == 1:
            out[:, col] = values[valid, col][0]
        else:
            out[:, col] = np.interp(x, x[valid], values[valid, col])
    return out


def zero_phase_ema(values, alpha):
    forward = values.copy()
    for i in range(1, len(forward)):
        forward[i] = alpha * values[i] + (1.0 - alpha) * forward[i - 1]
    backward = forward.copy()
    for i in range(len(backward) - 2, -1, -1):
        backward[i] = alpha * forward[i] + (1.0 - alpha) * backward[i + 1]
    return backward


def rolling_median(values, window=5):
    if window <= 1 or len(values) < 3:
        return values
    radius = window // 2
    out = values.copy()
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        out[i] = np.median(values[lo:hi], axis=0)
    return out


def smooth_values(values, alpha):
    interpolated = interpolate_missing(values)
    return zero_phase_ema(rolling_median(interpolated, window=5), alpha)


def smooth_quaternions(values, alpha):
    values = values.astype(np.float64)
    for i in range(1, len(values)):
        if np.all(np.isfinite(values[i])) and np.all(np.isfinite(values[i - 1])):
            if np.dot(values[i], values[i - 1]) < 0:
                values[i] *= -1.0
    filled = interpolate_missing(values)
    norms = np.linalg.norm(filled, axis=1, keepdims=True)
    filled = np.divide(filled, norms, out=np.tile([1.0, 0.0, 0.0, 0.0], (len(filled), 1)), where=norms > 1e-9)
    smoothed = zero_phase_ema(rolling_median(filled, window=5), alpha)
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    return np.divide(smoothed, norms, out=np.tile([1.0, 0.0, 0.0, 0.0], (len(smoothed), 1)), where=norms > 1e-9)


def build_smoothed_track(folder, preferred_label="Right", alpha=0.18):
    coordinates = json.loads((folder / "coordinates.json").read_text())
    imu = json.loads((folder / "imu.json").read_text())
    attitude = imu_attitude(imu["samples"])
    imu_start_ms = imu["samples"][0][0] if imu["samples"] else 0

    frames = sorted(iter_coordinate_frames(coordinates), key=lambda item: item[0])
    frame_count = len(frames)
    video_duration_ms = imu["samples"][-1][0] - imu_start_ms if imu["samples"] else 0
    frame_interval_ms = video_duration_ms / max(1, frame_count - 1)

    joint_values = np.full((frame_count, len(DEX3_JOINTS)), np.nan)
    root_positions = np.full((frame_count, 3), np.nan)
    root_quats = np.full((frame_count, 4), np.nan)
    detected = np.zeros(frame_count, dtype=bool)
    frame_ids = []
    times = []
    palm_reference = None

    for row_idx, (frame_index, video_timestamp_ms, hands) in enumerate(frames):
        frame_time_ms = imu_start_ms + (
            video_timestamp_ms if video_timestamp_ms is not None else frame_index * frame_interval_ms
        )
        imu_row = nearest_imu(attitude, frame_time_ms)
        frame_ids.append(frame_index)
        times.append(frame_time_ms)
        camera_quat = None
        if imu_row:
            q = imu_row["ahrs_quat_wxyz"]
            camera_quat = [q["w"], q["x"], q["y"], q["z"]]

        hand = choose_hand(hands, preferred_label)
        if not hand:
            continue
        detected[row_idx] = True
        root_positions[row_idx] = wrist_root_position(hand)
        palm_quat = palm_orientation_quat(hand)
        if palm_reference is None:
            palm_reference = palm_quat
        root_quats[row_idx] = palm_delta_root_quat(palm_quat, palm_reference)
        targets = estimate_joint_targets(hand, camera_quat=None)
        dex3 = dex3_targets_from_retargeted(targets)
        joint_values[row_idx] = [dex3[name] for name in DEX3_JOINTS]

    smoothed_joints = smooth_values(joint_values, alpha)
    smoothed_positions = smooth_values(root_positions, alpha)
    smoothed_quats = smooth_quaternions(root_quats, alpha)

    rows = []
    for i, frame in enumerate(frame_ids):
        rows.append(
            {
                "frame": int(frame),
                "time_ms": float(times[i]),
                "detected": bool(detected[i]),
                "root_pos": [float(v) for v in smoothed_positions[i]],
                "root_quat_wxyz": [float(v) for v in smoothed_quats[i]],
                "dex3": {name: float(value) for name, value in zip(DEX3_JOINTS, smoothed_joints[i])},
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Create a temporally smoothed Dex3 IK track from hand landmarks.")
    parser.add_argument("--root", default="task_01")
    parser.add_argument("--folder", help="One video folder. Defaults to all folders under --root.")
    parser.add_argument("--label", choices=["Left", "Right"], default="Right")
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--output-name", default="hand_ik_smoothed.json")
    args = parser.parse_args()

    folders = [Path(args.folder)] if args.folder else sorted(Path(args.root).glob("VID_*"))
    for folder in folders:
        if not (folder / "coordinates.json").exists():
            continue
        rows = build_smoothed_track(folder, preferred_label=args.label, alpha=args.alpha)
        output = folder / args.output_name
        output.write_text(json.dumps(rows, indent=2))
        detected = sum(row["detected"] for row in rows)
        print(f"{output}: wrote {len(rows)} frames, {detected} detected, {len(rows) - detected} IK-filled")


if __name__ == "__main__":
    main()
