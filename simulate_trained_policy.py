import argparse
from pathlib import Path

import mujoco
import numpy as np


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


def predict(observations, model):
    x = (observations - model["x_mean"]) / model["x_std"]
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ model["weights"]


def main():
    parser = argparse.ArgumentParser(description="Drive Unitree G1 MuJoCo from the trained intent model.")
    parser.add_argument("--scene", default="unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
    parser.add_argument("--dataset", default="training_data/unitree_intent_dataset.npz")
    parser.add_argument("--model", default="training_data/intent_ridge_model.npz")
    parser.add_argument("--sequence-id", type=int, default=-1, help="Default: model holdout sequence.")
    parser.add_argument("--steps-per-frame", type=int, default=6)
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--kd", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=True)
    policy = np.load(args.model, allow_pickle=True)
    sequence_id = args.sequence_id if args.sequence_id >= 0 else int(policy["holdout_sequence"])
    mask = dataset["sequence_ids"] == sequence_id
    if not mask.any():
        raise SystemExit(f"sequence id {sequence_id} not found")

    order = np.argsort(dataset["frame_ids"][mask])
    observations = dataset["observations"][mask][order]
    predictions = predict(observations, policy)
    output_names = [str(name) for name in policy["output_names"]]
    if args.max_frames:
        predictions = predictions[: args.max_frames]

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    controlled = []
    for output_idx, name in enumerate(output_names):
        if not name.endswith("_joint"):
            continue
        try:
            qpos_adr, dof_adr = qpos_index_for_joint(model, name)
            actuator_id = actuator_for_joint(model, name)
        except KeyError:
            continue
        controlled.append((output_idx, name, qpos_adr, dof_adr, actuator_id))

    if not controlled:
        raise SystemExit("No model output joints exist in this MuJoCo scene.")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    for frame_idx, row in enumerate(predictions):
        for _ in range(args.steps_per_frame):
            data.ctrl[:] = 0.0
            for output_idx, name, qpos_adr, dof_adr, actuator_id in controlled:
                error = row[output_idx] - data.qpos[qpos_adr]
                torque = args.kp * error - args.kd * data.qvel[dof_adr]
                low, high = model.actuator_ctrlrange[actuator_id]
                data.ctrl[actuator_id] = np.clip(torque, low, high)
            mujoco.mj_step(model, data)

        if frame_idx % 100 == 0 or frame_idx == len(predictions) - 1:
            summary = ", ".join(f"{name}={data.qpos[qpos_adr]:.3f}" for _, name, qpos_adr, _, _ in controlled)
            print(f"frame {frame_idx:04d}: {summary}")

    sequence_name = dataset["sequence_names"][sequence_id] if sequence_id < len(dataset["sequence_names"]) else sequence_id
    print(f"simulated trained policy for sequence {sequence_id} ({sequence_name})")
    print("controlled joints:", ", ".join(name for _, name, *_ in controlled))


if __name__ == "__main__":
    main()
