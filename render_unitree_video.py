import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from unitree_targets import G1_STANDING_JOINT_TARGETS
from unitree_targets import g1_full_body_targets_from_retargeted


def object_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise KeyError(name)
    return idx


def qpos_index_for_joint(model, joint_name):
    joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id]


def actuator_for_joint(model, joint_name):
    for actuator_id in range(model.nu):
        trnid = model.actuator_trnid[actuator_id]
        if trnid[0] < 0:
            continue
        actuator_joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(trnid[0]))
        if actuator_joint == joint_name:
            return actuator_id
    raise KeyError(joint_name)


def load_retargeted_targets(path, max_frames):
    rows = json.loads(Path(path).read_text())
    if max_frames:
        rows = rows[:max_frames]
    return [
        g1_full_body_targets_from_retargeted(row["joint_targets_rad"], row.get("hand_label", "Right"))
        for row in rows
    ]


def predict(observations, policy):
    x = (observations - policy["x_mean"]) / policy["x_std"]
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ policy["weights"]


def load_policy_targets(dataset_path, policy_path, sequence_id, max_frames):
    dataset = np.load(dataset_path, allow_pickle=True)
    policy = np.load(policy_path, allow_pickle=True)
    chosen_sequence = sequence_id if sequence_id >= 0 else int(policy["holdout_sequence"])
    mask = dataset["sequence_ids"] == chosen_sequence
    if not mask.any():
        raise SystemExit(f"sequence id {chosen_sequence} not found")

    order = np.argsort(dataset["frame_ids"][mask])
    observations = dataset["observations"][mask][order]
    predictions = predict(observations, policy)
    output_names = [str(name) for name in policy["output_names"]]
    if max_frames:
        predictions = predictions[:max_frames]

    targets = []
    for row in predictions:
        target = dict(G1_STANDING_JOINT_TARGETS)
        target.update({name: float(value) for name, value in zip(output_names, row) if name.endswith("_joint")})
        targets.append(target)
    return targets, chosen_sequence


def configure_camera(model):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.8]
    camera.distance = 3.0
    camera.azimuth = 140.0
    camera.elevation = -15.0
    return camera


def reset_robot(model, data):
    mujoco.mj_resetData(model, data)
    # Lift and square up the floating base so wrist motion is visible and stable.
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[0:3] = [0.0, 0.0, 0.78]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for joint_name, value in G1_STANDING_JOINT_TARGETS.items():
        try:
            qpos_adr, _ = qpos_index_for_joint(model, joint_name)
        except KeyError:
            continue
        data.qpos[qpos_adr] = value
    mujoco.mj_forward(model, data)


def pin_floating_base(model, data):
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[0:3] = [0.0, 0.0, 0.78]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[0:6] = 0.0


def render_video(args):
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    if args.mode == "policy":
        targets, sequence_id = load_policy_targets(args.dataset, args.policy, args.sequence_id, args.max_frames)
        print(f"rendering trained policy for sequence {sequence_id}")
    else:
        targets = load_retargeted_targets(args.retargeted, args.max_frames)
        print(f"rendering retargeted labels from {args.retargeted}")

    controlled = []
    first_target = targets[0]
    for joint_name in sorted(first_target):
        try:
            qpos_adr, dof_adr = qpos_index_for_joint(model, joint_name)
            actuator_id = actuator_for_joint(model, joint_name)
        except KeyError:
            continue
        controlled.append((joint_name, qpos_adr, dof_adr, actuator_id))

    if not controlled:
        raise SystemExit("No target joints exist in this MuJoCo scene.")

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

    reset_robot(model, data)
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = configure_camera(model)

    try:
        for frame_idx, target in enumerate(targets):
            for _ in range(args.steps_per_frame):
                if args.pin_base:
                    pin_floating_base(model, data)
                data.ctrl[:] = 0.0
                for joint_name, qpos_adr, dof_adr, actuator_id in controlled:
                    error = target.get(joint_name, 0.0) - data.qpos[qpos_adr]
                    kp = args.leg_kp if any(part in joint_name for part in ("hip", "knee", "ankle", "waist")) else args.kp
                    kd = args.leg_kd if any(part in joint_name for part in ("hip", "knee", "ankle", "waist")) else args.kd
                    torque = kp * error - kd * data.qvel[dof_adr]
                    low, high = model.actuator_ctrlrange[actuator_id]
                    data.ctrl[actuator_id] = np.clip(torque, low, high)
                mujoco.mj_step(model, data)
                if args.pin_base:
                    pin_floating_base(model, data)

            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if frame_idx % 100 == 0 or frame_idx == len(targets) - 1:
                print(f"rendered frame {frame_idx + 1}/{len(targets)}")
    finally:
        renderer.close()
        writer.release()

    print(f"wrote {output}")
    print("controlled joints:", ", ".join(name for name, *_ in controlled))


def main():
    parser = argparse.ArgumentParser(description="Render Unitree G1 MuJoCo playback to MP4.")
    parser.add_argument("--mode", choices=["policy", "retargeted"], default="policy")
    parser.add_argument("--scene", default="unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
    parser.add_argument("--dataset", default="training_data/unitree_intent_dataset.npz")
    parser.add_argument("--policy", default="training_data/intent_ridge_model.npz")
    parser.add_argument("--sequence-id", type=int, default=-1)
    parser.add_argument("--retargeted", default="task_01/VID_20260616_183017_035_764/retargeted_hand.json")
    parser.add_argument("--output", default="outputs/unitree_g1_policy.mp4")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--steps-per-frame", type=int, default=6)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--kd", type=float, default=0.25)
    parser.add_argument("--leg-kp", type=float, default=45.0)
    parser.add_argument("--leg-kd", type=float, default=2.0)
    parser.add_argument("--pin-base", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    render_video(args)


if __name__ == "__main__":
    main()
