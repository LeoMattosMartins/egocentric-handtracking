import argparse
import csv
import json
import math
from bisect import bisect_left
from pathlib import Path


FINGERS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}

FINGER_JOINTS = {
    "thumb": ["cmc", "mcp", "ip"],
    "index": ["mcp", "pip", "dip"],
    "middle": ["mcp", "pip", "dip"],
    "ring": ["mcp", "pip", "dip"],
    "pinky": ["mcp", "pip", "dip"],
}


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def norm(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    length = norm(v)
    if length < 1e-9:
        return [0.0, 0.0, 0.0]
    return [x / length for x in v]


def clamp(value, low, high):
    return max(low, min(high, value))


def angle_between(a, b):
    denom = norm(a) * norm(b)
    if denom < 1e-9:
        return 0.0
    return math.acos(clamp(dot(a, b) / denom, -1.0, 1.0))


def signed_angle_about_axis(a, b, axis):
    a_n = normalize(a)
    b_n = normalize(b)
    axis_n = normalize(axis)
    unsigned = angle_between(a_n, b_n)
    sign = 1.0 if dot(cross(a_n, b_n), axis_n) >= 0 else -1.0
    return sign * unsigned


def quat_normalize(q):
    length = math.sqrt(sum(value * value for value in q))
    if length < 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    return [value / length for value in q]


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quat_from_axis_angle(axis, angle):
    axis_n = normalize(axis)
    half = 0.5 * angle
    scale = math.sin(half)
    return quat_normalize([math.cos(half), axis_n[0] * scale, axis_n[1] * scale, axis_n[2] * scale])


def quat_rotate(q, v):
    q_conj = [q[0], -q[1], -q[2], -q[3]]
    rotated = quat_multiply(quat_multiply(q, [0.0, *v]), q_conj)
    return rotated[1:]


def quat_from_gyro(gx, gy, gz, dt):
    angle = math.sqrt(gx * gx + gy * gy + gz * gz) * dt
    if angle < 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    return quat_from_axis_angle([gx, gy, gz], angle)


def quat_from_accel(ax, ay, az):
    gravity_body = normalize([ax, ay, az])
    if norm(gravity_body) < 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    world_down = [0.0, 0.0, -1.0]
    axis = cross(gravity_body, world_down)
    axis_length = norm(axis)
    if axis_length < 1e-9:
        return [1.0, 0.0, 0.0, 0.0] if dot(gravity_body, world_down) > 0 else [0.0, 1.0, 0.0, 0.0]
    angle = math.acos(clamp(dot(gravity_body, world_down), -1.0, 1.0))
    return quat_from_axis_angle(axis, angle)


def quat_to_euler(q):
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def landmark_points(hand):
    source = hand.get("world_landmarks") or hand["landmarks"]
    use_world = bool(hand.get("world_landmarks"))
    points = {}
    for landmark in source:
        if use_world:
            points[landmark["id"]] = [float(landmark["x"]), float(landmark["y"]), float(landmark["z"])]
        else:
            # MediaPipe image landmarks are x-right, y-down, and z roughly image-scaled.
            # Convert into a wrist-relative right/up/forward-ish frame before normalizing.
            points[landmark["id"]] = [
                float(landmark["x"]),
                -float(landmark["y"]),
                -float(landmark["z"]),
            ]
    return points


def normalize_hand(points):
    wrist = points[0]
    palm_scale = 0.5 * (norm(sub(points[5], wrist)) + norm(sub(points[17], wrist)))
    if palm_scale < 1e-9:
        palm_scale = 1.0
    return {idx: [(value - wrist[dim]) / palm_scale for dim, value in enumerate(point)] for idx, point in points.items()}


def palm_frame(points):
    wrist = points[0]
    index_mcp = points[5]
    middle_mcp = points[9]
    pinky_mcp = points[17]
    palm_x = normalize(sub(index_mcp, pinky_mcp))
    palm_y = normalize(sub(middle_mcp, wrist))
    palm_z = normalize(cross(palm_x, palm_y))
    if norm(palm_z) < 1e-9:
        palm_z = [0.0, 0.0, 1.0]
    return palm_x, palm_y, palm_z


def flexion_angle(points, prev_id, joint_id, next_id):
    incoming = sub(points[prev_id], points[joint_id])
    outgoing = sub(points[next_id], points[joint_id])
    return math.pi - angle_between(incoming, outgoing)


def estimate_joint_targets(hand, camera_quat=None):
    points = normalize_hand(landmark_points(hand))
    _, palm_y, palm_z = palm_frame(points)
    middle_base = sub(points[9], points[0])

    targets = {}
    for finger, chain in FINGERS.items():
        joint_names = FINGER_JOINTS[finger]
        for name, prev_id, joint_id, next_id in zip(joint_names, chain[:-2], chain[1:-1], chain[2:]):
            targets[f"{finger}_{name}_flex"] = flexion_angle(points, prev_id, joint_id, next_id)

        base_id = chain[1]
        base_vector = sub(points[base_id], points[0])
        if finger == "middle":
            abduction = 0.0
        else:
            abduction = signed_angle_about_axis(middle_base, base_vector, palm_z)
        targets[f"{finger}_abduction"] = abduction

    palm_normal = palm_z
    targets["wrist_pitch_from_camera"] = math.atan2(palm_y[2], math.sqrt(palm_y[0] ** 2 + palm_y[1] ** 2))
    targets["wrist_roll_from_camera"] = math.atan2(palm_normal[0], palm_normal[1])

    if camera_quat:
        palm_y_world = quat_rotate(camera_quat, palm_y)
        palm_z_world = quat_rotate(camera_quat, palm_z)
        targets["wrist_pitch_world"] = math.atan2(
            palm_y_world[2],
            math.sqrt(palm_y_world[0] ** 2 + palm_y_world[1] ** 2),
        )
        targets["wrist_roll_world"] = math.atan2(palm_z_world[0], palm_z_world[1])
        targets["wrist_yaw_world"] = math.atan2(palm_y_world[1], palm_y_world[0])
    return targets


def imu_attitude(samples, correction_gain=0.02):
    if not samples:
        return []

    attitude = []
    q = quat_from_accel(samples[0][1], samples[0][2], samples[0][3])
    last_t = samples[0][0]
    for row in samples:
        t, ax, ay, az, gx, gy, gz = row
        dt = max(0.0, (t - last_t) / 1000.0)
        last_t = t
        q = quat_normalize(quat_multiply(q, quat_from_gyro(gx, gy, gz, dt)))

        accel_q = quat_from_accel(ax, ay, az)
        predicted_down = quat_rotate(q, [0.0, 0.0, -1.0])
        measured_down = quat_rotate(accel_q, [0.0, 0.0, -1.0])
        correction_axis = cross(predicted_down, measured_down)
        correction_angle = math.asin(clamp(norm(correction_axis), -1.0, 1.0))
        if norm(correction_axis) > 1e-9:
            q = quat_normalize(
                quat_multiply(quat_from_axis_angle(correction_axis, correction_gain * correction_angle), q)
            )

        roll, pitch, yaw = quat_to_euler(q)
        attitude.append(
            {
                "time_ms": t,
                "accel": {"x": ax, "y": ay, "z": az},
                "gyro": {"x": gx, "y": gy, "z": gz},
                "ahrs_quat_wxyz": {"w": q[0], "x": q[1], "y": q[2], "z": q[3]},
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
            }
        )
    return attitude


def nearest_imu(attitude, time_ms):
    if not attitude:
        return None
    times = [row["time_ms"] for row in attitude]
    idx = bisect_left(times, time_ms)
    candidates = []
    if idx < len(attitude):
        candidates.append(attitude[idx])
    if idx > 0:
        candidates.append(attitude[idx - 1])
    return min(candidates, key=lambda row: abs(row["time_ms"] - time_ms))


def choose_primary_hand(hands, preferred_label):
    if not hands:
        return None
    if preferred_label:
        matches = [hand for hand in hands if hand.get("label") == preferred_label]
        if matches:
            return matches[0]
    return hands[0]


def iter_coordinate_frames(coordinates):
    if "frames" in coordinates:
        for frame_key, frame_data in coordinates["frames"].items():
            yield int(frame_key), frame_data.get("timestamp_ms"), frame_data.get("hands", [])
    else:
        for frame_key, hands in coordinates.items():
            yield int(frame_key), None, hands


def retarget_folder(folder, preferred_label=None):
    coordinates_path = folder / "coordinates.json"
    imu_path = folder / "imu.json"
    if not coordinates_path.exists() or not imu_path.exists():
        return []

    coordinates = json.loads(coordinates_path.read_text())
    imu = json.loads(imu_path.read_text())
    attitude = imu_attitude(imu["samples"])
    imu_start_ms = imu["samples"][0][0] if imu["samples"] else 0

    frames = sorted(iter_coordinate_frames(coordinates), key=lambda item: item[0])
    frame_count = len(frames)
    video_duration_ms = imu["samples"][-1][0] - imu_start_ms if imu["samples"] else 0
    frame_interval_ms = video_duration_ms / max(1, frame_count - 1)

    rows = []
    for frame_index, video_timestamp_ms, hands in frames:
        hand = choose_primary_hand(hands, preferred_label)
        if not hand:
            continue

        frame_time_ms = imu_start_ms + (video_timestamp_ms if video_timestamp_ms is not None else frame_index * frame_interval_ms)
        imu_row = nearest_imu(attitude, frame_time_ms)
        camera_quat = None
        if imu_row:
            quat = imu_row["ahrs_quat_wxyz"]
            camera_quat = [quat["w"], quat["x"], quat["y"], quat["z"]]
        targets = estimate_joint_targets(hand, camera_quat=camera_quat)
        rows.append(
            {
                "frame": frame_index,
                "time_ms": frame_time_ms,
                "hand_label": hand.get("label", "Unknown"),
                "imu": imu_row,
                "joint_targets_rad": targets,
            }
        )
    return rows


def write_json(path, rows):
    path.write_text(json.dumps(rows, indent=2))


def write_csv(path, rows):
    target_names = sorted({name for row in rows for name in row["joint_targets_rad"]})
    fieldnames = [
        "frame",
        "time_ms",
        "hand_label",
        "imu_roll",
        "imu_pitch",
        "imu_yaw",
        "imu_quat_w",
        "imu_quat_x",
        "imu_quat_y",
        "imu_quat_z",
        *target_names,
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            imu = row["imu"] or {}
            quat = imu.get("ahrs_quat_wxyz") or {}
            out = {
                "frame": row["frame"],
                "time_ms": f"{row['time_ms']:.3f}",
                "hand_label": row["hand_label"],
                "imu_roll": imu.get("roll", ""),
                "imu_pitch": imu.get("pitch", ""),
                "imu_yaw": imu.get("yaw", ""),
                "imu_quat_w": quat.get("w", ""),
                "imu_quat_x": quat.get("x", ""),
                "imu_quat_y": quat.get("y", ""),
                "imu_quat_z": quat.get("z", ""),
            }
            out.update(row["joint_targets_rad"])
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(description="Retarget MediaPipe hand landmarks into generic humanoid hand joint targets.")
    parser.add_argument("--root", default="task_01", help="Root containing video folders.")
    parser.add_argument("--label", choices=["Left", "Right"], help="Prefer this detected hand label when two hands are visible.")
    parser.add_argument("--json-name", default="retargeted_hand.json", help="Per-folder JSON output filename.")
    parser.add_argument("--csv-name", default="retargeted_hand.csv", help="Per-folder CSV output filename.")
    args = parser.parse_args()

    root = Path(args.root)
    folders = sorted(path.parent for path in root.glob("*/coordinates.json"))
    if not folders:
        raise SystemExit(f"No coordinates.json files found under {root}")

    for folder in folders:
        rows = retarget_folder(folder, preferred_label=args.label)
        write_json(folder / args.json_name, rows)
        write_csv(folder / args.csv_name, rows)
        print(f"{folder}: wrote {len(rows)} retargeted frames")


if __name__ == "__main__":
    main()
