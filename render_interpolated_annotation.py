import argparse
import json
from pathlib import Path

import cv2
import numpy as np


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def iter_coordinate_frames(coordinates):
    if "frames" in coordinates:
        for frame_key, frame_data in coordinates["frames"].items():
            yield int(frame_key), frame_data.get("hands", [])
    else:
        for frame_key, hands in coordinates.items():
            yield int(frame_key), hands


def choose_hand(hands, preferred_label):
    if not hands:
        return None
    if preferred_label:
        matches = [hand for hand in hands if hand.get("label") == preferred_label]
        if matches:
            return matches[0]
    return hands[0]


def interpolate_missing(values):
    out = values.copy().astype(np.float64)
    x = np.arange(out.shape[0])
    flat = out.reshape(out.shape[0], -1)
    for col in range(flat.shape[1]):
        valid = np.isfinite(flat[:, col])
        if valid.sum() == 0:
            flat[:, col] = 0.0
        elif valid.sum() == 1:
            flat[:, col] = flat[valid, col][0]
        else:
            flat[:, col] = np.interp(x, x[valid], flat[valid, col])
    return flat.reshape(out.shape)


def zero_phase_ema(values, alpha):
    forward = values.copy()
    for i in range(1, len(forward)):
        forward[i] = alpha * values[i] + (1.0 - alpha) * forward[i - 1]
    backward = forward.copy()
    for i in range(len(backward) - 2, -1, -1):
        backward[i] = alpha * forward[i] + (1.0 - alpha) * backward[i + 1]
    return backward


def build_smoothed_landmarks(coordinates_path, frame_count, preferred_label, alpha):
    coordinates = json.loads(coordinates_path.read_text())
    raw = np.full((frame_count, 21, 3), np.nan)
    detected = np.zeros(frame_count, dtype=bool)

    for frame_idx, hands in iter_coordinate_frames(coordinates):
        if frame_idx >= frame_count:
            continue
        hand = choose_hand(hands, preferred_label)
        if not hand:
            continue
        detected[frame_idx] = True
        by_id = {lm["id"]: lm for lm in hand["landmarks"]}
        for landmark_id in range(21):
            lm = by_id.get(landmark_id)
            if lm:
                raw[frame_idx, landmark_id] = [float(lm["x"]), float(lm["y"]), float(lm["z"])]

    smoothed = zero_phase_ema(interpolate_missing(raw), alpha)
    return smoothed, detected


def draw_hand(frame, landmarks, detected, trail):
    height, width = frame.shape[:2]
    points = []
    for x, y, _ in landmarks:
        px = int(np.clip(x, -0.1, 1.1) * width)
        py = int(np.clip(y, -0.1, 1.1) * height)
        points.append((px, py))

    line_color = (60, 235, 80) if detected else (255, 210, 60)
    point_color = (40, 40, 255) if detected else (0, 180, 255)
    text_color = (50, 220, 80) if detected else (0, 210, 255)

    overlay = frame.copy()
    for a, b in HAND_CONNECTIONS:
        cv2.line(overlay, points[a], points[b], line_color, 3, cv2.LINE_AA)
    for idx, point in enumerate(points):
        radius = 6 if idx in (0, 4, 8, 12, 16, 20) else 4
        cv2.circle(overlay, point, radius, point_color, -1, cv2.LINE_AA)

    for i in range(1, len(trail)):
        cv2.line(overlay, trail[i - 1], trail[i], (220, 220, 220), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, dst=frame)
    label = "detected + smoothed" if detected else "IK filled / interpolated"
    cv2.putText(frame, label, (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)


def render_folder(folder, output, preferred_label, alpha, max_frames):
    video_path = folder / "base.mp4"
    coordinates_path = folder / "coordinates.json"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        frame_count = min(frame_count, max_frames)

    landmarks, detected = build_smoothed_landmarks(coordinates_path, frame_count, preferred_label, alpha)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not write {output}")

    trail = []
    frame_idx = 0
    try:
        while frame_idx < frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            wrist = landmarks[frame_idx, 0]
            trail.append((int(wrist[0] * width), int(wrist[1] * height)))
            trail = trail[-45:]
            draw_hand(frame, landmarks[frame_idx], bool(detected[frame_idx]), trail)
            cv2.putText(
                frame,
                f"frame {frame_idx:04d}",
                (32, height - 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print(f"wrote {output}")
    print(f"frames: {frame_idx}  detected: {int(detected[:frame_idx].sum())}  filled: {frame_idx - int(detected[:frame_idx].sum())}")


def main():
    parser = argparse.ArgumentParser(description="Render smoothed/interpolated 21-point hand annotations over base.mp4.")
    parser.add_argument("--folder", default="task_01/VID_20260616_183409_968_529")
    parser.add_argument("--root", help="Render every VID_* folder under this root.")
    parser.add_argument("--output", default="outputs/interpolated_annotation_holdout.mp4")
    parser.add_argument("--output-name", default="interpolated_annotated.mp4")
    parser.add_argument("--label", choices=["Left", "Right"], default="Right")
    parser.add_argument("--alpha", type=float, default=0.12)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if args.root:
        for folder in sorted(Path(args.root).glob("VID_*")):
            if (folder / "base.mp4").exists() and (folder / "coordinates.json").exists():
                render_folder(folder, folder / args.output_name, args.label, args.alpha, args.max_frames)
    else:
        render_folder(Path(args.folder), Path(args.output), args.label, args.alpha, args.max_frames)


if __name__ == "__main__":
    main()
