import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from unitree_targets import DEX3_JOINTS


def joint_qpos_address(model, joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(joint_name)
    return model.jnt_qposadr[joint_id]


def configure_camera():
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.02, 0.0, 0.08]
    camera.distance = 0.55
    camera.azimuth = 145.0
    camera.elevation = -18.0
    return camera


def predict(observations, policy):
    x = (observations - policy["x_mean"]) / policy["x_std"]
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ policy["weights"]


def apply_policy_dex3(rows, dataset_path, policy_path, sequence_id):
    dataset = np.load(dataset_path, allow_pickle=True)
    policy = np.load(policy_path, allow_pickle=True)
    chosen_sequence = sequence_id if sequence_id >= 0 else int(policy["holdout_sequence"])
    mask = dataset["sequence_ids"] == chosen_sequence
    if not mask.any():
        raise SystemExit(f"sequence id {chosen_sequence} not found")

    order = np.argsort(dataset["frame_ids"][mask])
    frame_ids = dataset["frame_ids"][mask][order]
    predictions = predict(dataset["observations"][mask][order], policy)
    output_names = [str(name) for name in policy["output_names"]]
    pred_by_name = {name: predictions[:, idx] for idx, name in enumerate(output_names)}
    all_frames = np.asarray([row["frame"] for row in rows], dtype=np.float64)

    for joint_name in DEX3_JOINTS:
        if joint_name not in pred_by_name:
            continue
        values = np.interp(all_frames, frame_ids.astype(np.float64), pred_by_name[joint_name])
        for row, value in zip(rows, values):
            row["dex3"][joint_name] = float(value)
    return chosen_sequence


def set_hand_pose(model, data, row, joint_addresses, root_offset):
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hand_root_joint")
    root_qpos = model.jnt_qposadr[root_id]
    data.qpos[root_qpos : root_qpos + 3] = np.asarray(row["root_pos"], dtype=np.float64) + root_offset
    data.qpos[root_qpos + 3 : root_qpos + 7] = row["root_quat_wxyz"]
    for name in DEX3_JOINTS:
        data.qpos[joint_addresses[name]] = row["dex3"][name]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description="Render a smoothed Unitree Dex3 hand IK track to MP4.")
    parser.add_argument("--mode", choices=["ik", "policy"], default="ik")
    parser.add_argument("--scene", default="unitree_mujoco/unitree_robots/g1/dex3_right_hand_scene.xml")
    parser.add_argument("--track", default="task_01/VID_20260616_183017_035_764/hand_ik_smoothed.json")
    parser.add_argument("--dataset", default="training_data/unitree_intent_dataset.npz")
    parser.add_argument("--policy", default="training_data/intent_ridge_model.npz")
    parser.add_argument("--sequence-id", type=int, default=-1)
    parser.add_argument("--output", default="outputs/dex3_hand_ik_smoothed.mp4")
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.08])
    args = parser.parse_args()

    rows = json.loads(Path(args.track).read_text())
    if args.max_frames:
        rows = rows[: args.max_frames]
    if args.mode == "policy":
        sequence_id = apply_policy_dex3(rows, args.dataset, args.policy, args.sequence_id)
        print(f"using trained policy Dex3 predictions for sequence {sequence_id}")
    root_positions = np.asarray([row["root_pos"] for row in rows], dtype=np.float64)
    root_offset = np.asarray(args.center, dtype=np.float64) - np.median(root_positions, axis=0)

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    joint_addresses = {name: joint_qpos_address(model, name) for name in DEX3_JOINTS}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {output}")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = configure_camera()
    try:
        for i, row in enumerate(rows):
            set_hand_pose(model, data, row, joint_addresses, root_offset)
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if i % 100 == 0 or i == len(rows) - 1:
                print(f"rendered frame {i + 1}/{len(rows)}")
    finally:
        renderer.close()
        writer.release()

    detected = sum(1 for row in rows if row["detected"])
    print(f"wrote {output}")
    print(f"frames: {len(rows)}  detected: {detected}  filled: {len(rows) - detected}")


if __name__ == "__main__":
    main()
