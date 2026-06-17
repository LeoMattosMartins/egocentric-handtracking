import json
import os

import cv2
import mediapipe as mp
from tqdm import tqdm


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def draw_skeleton(frame, landmarks, width, height):
    pixel_points = []
    for lm in landmarks:
        px = int(lm["x"] * width)
        py = int(lm["y"] * height)
        pixel_points.append((px, py))

    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(pixel_points) and end_idx < len(pixel_points):
            cv2.line(frame, pixel_points[start_idx], pixel_points[end_idx], (0, 255, 0), 2)

    for px, py in pixel_points:
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)


def process_video_folder(folder_path, landmarker, pbar_outer):
    video_path = os.path.join(folder_path, "base.mp4")
    output_video_path = os.path.join(folder_path, "annotated.mp4")
    output_json_path = os.path.join(folder_path, "coordinates.json")

    folder_name = os.path.basename(folder_path)
    pbar_outer.set_description(f"Processing folder: {folder_name[:20]}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n[Error] Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    all_frames_data = {
        "metadata": {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": total_frames,
            "coordinate_space": "mediapipe_normalized_image_landmarks",
            "world_coordinate_space": "mediapipe_hand_world_landmarks",
        },
        "frames": {},
    }
    frame_count = 0

    with tqdm(total=total_frames, desc="  └─ Frames", leave=False, unit="fr") as pbar_inner:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = landmarker.detect(mp_image)
            frame_data = {
                "frame": frame_count,
                "timestamp_ms": cap.get(cv2.CAP_PROP_POS_MSEC),
                "hands": [],
            }

            if detection_result.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    hand_label = "Unknown"
                    if detection_result.handedness and hand_idx < len(detection_result.handedness):
                        hand_label = detection_result.handedness[hand_idx][0].category_name

                    landmarks_list = []
                    for lm_id, lm in enumerate(hand_landmarks):
                        landmarks_list.append(
                            {
                                "id": lm_id,
                                "x": float(lm.x),
                                "y": float(lm.y),
                                "z": float(lm.z),
                            }
                        )

                    world_landmarks_list = []
                    if detection_result.hand_world_landmarks and hand_idx < len(detection_result.hand_world_landmarks):
                        for lm_id, lm in enumerate(detection_result.hand_world_landmarks[hand_idx]):
                            world_landmarks_list.append(
                                {
                                    "id": lm_id,
                                    "x": float(lm.x),
                                    "y": float(lm.y),
                                    "z": float(lm.z),
                                }
                            )

                    frame_data["hands"].append(
                        {
                            "hand_index": hand_idx,
                            "label": hand_label,
                            "landmarks": landmarks_list,
                            "world_landmarks": world_landmarks_list,
                        }
                    )

                    draw_skeleton(frame, landmarks_list, width, height)

            all_frames_data["frames"][str(frame_count)] = frame_data
            out.write(frame)
            frame_count += 1
            pbar_inner.update(1)

    cap.release()
    out.release()

    with open(output_json_path, "w") as f:
        json.dump(all_frames_data, f, indent=4)


def main():
    root_dir = "./task_01"
    model_path = "hand_landmarker.task"

    if not os.path.exists(root_dir):
        print(f"Root directory {root_dir} not found.")
        return

    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' missing. Please download it first.")
        return

    target_folders = [d for d, _, f in os.walk(root_dir) if "base.mp4" in f]
    if not target_folders:
        print("No folders containing 'base.mp4' found.")
        return

    print(f"Found {len(target_folders)} video folders to process.\n")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        with tqdm(target_folders, desc="Total Progress", unit="folder") as pbar_outer:
            for folderpath in pbar_outer:
                process_video_folder(folderpath, landmarker, pbar_outer)


if __name__ == "__main__":
    main()
