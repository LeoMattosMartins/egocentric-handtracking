import argparse
import json
from pathlib import Path

import numpy as np

from render_interpolated_annotation import build_smoothed_landmarks
from retarget_hand import estimate_joint_targets
from smooth_hand_ik import build_smoothed_track
from unitree_targets import DEX3_JOINTS, G1_WRIST_JOINTS
from unitree_targets import dex3_targets_from_retargeted, g1_wrist_targets_from_retargeted


def iter_coordinate_frames(coordinates):
    if "frames" in coordinates:
        for frame_key, frame_data in coordinates["frames"].items():
            yield int(frame_key), frame_data.get("hands", [])
    else:
        for frame_key, hands in coordinates.items():
            yield int(frame_key), hands


def choose_primary_hand(hands, preferred_label=None):
    if not hands:
        return None
    if preferred_label:
        matches = [hand for hand in hands if hand.get("label") == preferred_label]
        if matches:
            return matches[0]
    return hands[0]


def landmark_vector(hand):
    landmarks = hand.get("world_landmarks") or hand.get("landmarks") or []
    by_id = {landmark["id"]: landmark for landmark in landmarks}
    values = []
    for idx in range(21):
        landmark = by_id.get(idx, {})
        values.extend(
            [
                float(landmark.get("x", 0.0)),
                float(landmark.get("y", 0.0)),
                float(landmark.get("z", 0.0)),
            ]
        )
    return values


def smoothed_landmark_vector(landmarks):
    return [float(value) for point in landmarks for value in point]


def landmarks_to_hand(landmarks):
    return {
        "landmarks": [
            {"id": idx, "x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
            for idx, point in enumerate(landmarks)
        ]
    }


def retargeted_by_frame(path):
    rows = json.loads(path.read_text())
    return {int(row["frame"]): row for row in rows}


def build_dataset(root, preferred_label=None):
    observations = []
    g1_targets = []
    dex3_targets = []
    sequence_ids = []
    frame_ids = []
    sequence_names = []

    for sequence_id, folder in enumerate(sorted(Path(root).glob("VID_*"))):
        coordinates_path = folder / "coordinates.json"
        video_path = folder / "base.mp4"
        if not coordinates_path.exists() or not video_path.exists():
            continue

        sequence_names.append(folder.name)
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        smoothed_landmarks, detected = build_smoothed_landmarks(
            coordinates_path,
            frame_count,
            preferred_label,
            alpha=0.22,
        )
        track = build_smoothed_track(folder, preferred_label=preferred_label or "Right", alpha=0.22)

        for row_index, track_row in enumerate(track[:frame_count]):
            landmarks = smoothed_landmarks[row_index]
            root_quat = track_row["root_quat_wxyz"]
            joint_targets = estimate_joint_targets(landmarks_to_hand(landmarks), camera_quat=root_quat)
            g1 = g1_wrist_targets_from_retargeted(joint_targets, preferred_label or "Right")
            dex3 = dex3_targets_from_retargeted(joint_targets)

            observations.append(
                [
                    *smoothed_landmark_vector(landmarks),
                    *track_row["root_pos"],
                    *root_quat,
                    1.0 if detected[row_index] else 0.0,
                    row_index / max(1, frame_count - 1),
                ]
            )
            g1_targets.append([g1[name] for name in G1_WRIST_JOINTS])
            dex3_targets.append([dex3[name] for name in DEX3_JOINTS])
            sequence_ids.append(sequence_id)
            frame_ids.append(track_row["frame"])

    if not observations:
        raise SystemExit(f"No training rows found under {root}. Run retarget_hand.py first.")

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "g1_wrist_targets": np.asarray(g1_targets, dtype=np.float32),
        "dex3_targets": np.asarray(dex3_targets, dtype=np.float32),
        "sequence_ids": np.asarray(sequence_ids, dtype=np.int32),
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "sequence_names": np.asarray(sequence_names),
        "observation_names": np.asarray(
            [f"landmark_{idx}_{axis}" for idx in range(21) for axis in ("x", "y", "z")]
            + [
                "root_x",
                "root_y",
                "root_z",
                "root_quat_w",
                "root_quat_x",
                "root_quat_y",
                "root_quat_z",
                "was_detected",
                "action_phase",
            ]
        ),
        "g1_wrist_names": np.asarray(G1_WRIST_JOINTS),
        "dex3_names": np.asarray(DEX3_JOINTS),
    }


def main():
    parser = argparse.ArgumentParser(description="Build imitation-learning arrays from MediaPipe, IMU, and Unitree retargets.")
    parser.add_argument("--root", default="task_01")
    parser.add_argument("--output", default="training_data/unitree_intent_dataset.npz")
    parser.add_argument("--label", choices=["Left", "Right"], help="Prefer this detected hand when two hands are visible.")
    args = parser.parse_args()

    dataset = build_dataset(args.root, preferred_label=args.label)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)
    print(f"wrote {output}")
    print(f"observations: {dataset['observations'].shape}")
    print(f"g1_wrist_targets: {dataset['g1_wrist_targets'].shape}")
    print(f"dex3_targets: {dataset['dex3_targets'].shape}")


if __name__ == "__main__":
    main()
