import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from unitree_targets import g1_wrist_targets_from_retargeted


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


def load_targets(path):
    rows = json.loads(Path(path).read_text())
    return [
        g1_wrist_targets_from_retargeted(row["joint_targets_rad"], row.get("hand_label", "Right"))
        for row in rows
    ]


def main():
    parser = argparse.ArgumentParser(description="Replay retargeted hand/wrist motion on Unitree G1 MuJoCo.")
    parser.add_argument("--scene", default="unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
    parser.add_argument("--retargeted", default="task_01/VID_20260616_183017_035_764/retargeted_hand.json")
    parser.add_argument("--steps-per-frame", type=int, default=6)
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--kd", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    targets = load_targets(args.retargeted)
    if args.max_frames:
        targets = targets[: args.max_frames]

    controlled = []
    for joint_name in sorted(targets[0]):
        try:
            qpos_adr, dof_adr = qpos_index_for_joint(model, joint_name)
            actuator_id = actuator_for_joint(model, joint_name)
        except KeyError:
            continue
        controlled.append((joint_name, qpos_adr, dof_adr, actuator_id))

    if not controlled:
        raise SystemExit("No target joints from retargeted data exist in this MuJoCo model.")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    for frame_idx, target in enumerate(targets):
        for _ in range(args.steps_per_frame):
            data.ctrl[:] = 0.0
            for joint_name, qpos_adr, dof_adr, actuator_id in controlled:
                error = target[joint_name] - data.qpos[qpos_adr]
                torque = args.kp * error - args.kd * data.qvel[dof_adr]
                low, high = model.actuator_ctrlrange[actuator_id]
                data.ctrl[actuator_id] = np.clip(torque, low, high)
            mujoco.mj_step(model, data)

        if frame_idx % 100 == 0 or frame_idx == len(targets) - 1:
            summary = ", ".join(f"{name}={data.qpos[qpos_adr]:.3f}" for name, qpos_adr, _, _ in controlled)
            print(f"frame {frame_idx:04d}: {summary}")

    print(f"simulated {len(targets)} retargeted frames on {Path(args.scene).name}")
    print("controlled joints:", ", ".join(name for name, *_ in controlled))


if __name__ == "__main__":
    main()
