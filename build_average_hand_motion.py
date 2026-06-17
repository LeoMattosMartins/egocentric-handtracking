import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from render_interpolated_annotation import build_smoothed_landmarks
from retarget_hand import estimate_joint_targets
from smooth_hand_ik import build_smoothed_track


TARGET_KEYS = [
    "thumb_cmc_flex",
    "thumb_abduction",
    "thumb_mcp_flex",
    "thumb_ip_flex",
    "index_mcp_flex",
    "index_abduction",
    "index_pip_flex",
    "index_dip_flex",
    "middle_mcp_flex",
    "middle_abduction",
    "middle_pip_flex",
    "middle_dip_flex",
    "ring_mcp_flex",
    "ring_abduction",
    "ring_pip_flex",
    "ring_dip_flex",
    "pinky_mcp_flex",
    "pinky_abduction",
    "pinky_pip_flex",
    "pinky_dip_flex",
]


def landmarks_to_hand(landmarks):
    return {
        "landmarks": [
            {"id": idx, "x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
            for idx, point in enumerate(landmarks)
        ]
    }


def quat_average(quats):
    ref = quats[0]
    aligned = []
    for quat in quats:
        quat = np.asarray(quat, dtype=np.float64)
        if np.dot(ref, quat) < 0:
            quat = -quat
        aligned.append(quat)
    mean = np.mean(aligned, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


def video_frame_count(folder):
    cap = cv2.VideoCapture(str(folder / "base.mp4"))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {folder / 'base.mp4'}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def resample_indices(length, output_frames):
    if length <= 1:
        return np.zeros(output_frames, dtype=np.int32)
    return np.round(np.linspace(0, length - 1, output_frames)).astype(np.int32)


def load_sequence(folder, output_frames, label, alpha):
    frame_count = video_frame_count(folder)
    landmarks, detected = build_smoothed_landmarks(folder / "coordinates.json", frame_count, label, alpha)
    track = build_smoothed_track(folder, preferred_label=label, alpha=alpha)
    indices = resample_indices(frame_count, output_frames)

    sequence = []
    for idx in indices:
        targets = estimate_joint_targets(landmarks_to_hand(landmarks[idx]))
        sequence.append(
            {
                "root_pos": np.asarray(track[idx]["root_pos"], dtype=np.float64),
                "root_quat_wxyz": np.asarray(track[idx]["root_quat_wxyz"], dtype=np.float64),
                "detected": bool(detected[idx]),
                "joint_targets_rad": {key: float(targets.get(key, 0.0)) for key in TARGET_KEYS},
            }
        )
    return sequence


def build_average(root, output_frames, label, alpha):
    folders = [folder for folder in sorted(Path(root).glob("VID_*")) if (folder / "coordinates.json").exists()]
    if not folders:
        raise SystemExit(f"No video folders found under {root}")

    sequences = [load_sequence(folder, output_frames, label, alpha) for folder in folders]
    rows = []
    for frame_idx in range(output_frames):
        root_positions = np.asarray([sequence[frame_idx]["root_pos"] for sequence in sequences])
        quats = [sequence[frame_idx]["root_quat_wxyz"] for sequence in sequences]
        joint_targets = {}
        for key in TARGET_KEYS:
            joint_targets[key] = float(np.mean([sequence[frame_idx]["joint_targets_rad"][key] for sequence in sequences]))
        detected_count = sum(sequence[frame_idx]["detected"] for sequence in sequences)
        rows.append(
            {
                "frame": frame_idx,
                "source_count": len(sequences),
                "detected_count": int(detected_count),
                "root_pos": [float(value) for value in root_positions.mean(axis=0)],
                "root_quat_wxyz": [float(value) for value in quat_average(quats)],
                "joint_targets_rad": joint_targets,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Average repeated egocentric hand videos into one smoothed hand motion.")
    parser.add_argument("--root", default="task_01")
    parser.add_argument("--output", default="training_data/average_hand_motion.json")
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--label", choices=["Left", "Right"], default="Right")
    parser.add_argument("--alpha", type=float, default=0.12)
    args = parser.parse_args()

    rows = build_average(args.root, args.frames, args.label, args.alpha)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    print(f"wrote {output}")
    print(f"frames: {len(rows)}  averaged videos: {rows[0]['source_count'] if rows else 0}")


if __name__ == "__main__":
    main()
