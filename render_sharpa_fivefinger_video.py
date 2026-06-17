import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np

from render_interpolated_annotation import build_smoothed_landmarks
from retarget_hand import estimate_joint_targets
from smooth_hand_ik import build_smoothed_track


FIVE_FINGER_MAP = {
    "right_thumb_CMC_FE": "thumb_cmc_flex",
    "right_thumb_CMC_AA": "thumb_abduction",
    "right_thumb_MCP_FE": "thumb_mcp_flex",
    "right_thumb_MCP_AA": None,
    "right_thumb_IP": "thumb_ip_flex",
    "right_index_MCP_FE": "index_mcp_flex",
    "right_index_MCP_AA": "index_abduction",
    "right_index_PIP": "index_pip_flex",
    "right_index_DIP": "index_dip_flex",
    "right_middle_MCP_FE": "middle_mcp_flex",
    "right_middle_MCP_AA": "middle_abduction",
    "right_middle_PIP": "middle_pip_flex",
    "right_middle_DIP": "middle_dip_flex",
    "right_ring_MCP_FE": "ring_mcp_flex",
    "right_ring_MCP_AA": "ring_abduction",
    "right_ring_PIP": "ring_pip_flex",
    "right_ring_DIP": "ring_dip_flex",
    "right_pinky_CMC": "pinky_abduction",
    "right_pinky_MCP_FE": "pinky_mcp_flex",
    "right_pinky_MCP_AA": "pinky_abduction",
    "right_pinky_PIP": "pinky_pip_flex",
    "right_pinky_DIP": "pinky_dip_flex",
}


def clamp(value, low, high):
    return max(low, min(high, value))


def landmarks_to_hand(landmarks):
    return {
        "landmarks": [
            {"id": idx, "x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
            for idx, point in enumerate(landmarks)
        ]
    }


def make_floating_scene(source_xml, output_dir):
    source_xml = Path(source_xml)
    tree = ET.parse(source_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = (source_xml.parent / compiler.get("meshdir")).resolve()
        compiler.set("meshdir", str(meshdir))
    root_body = root.find("./worldbody/body")
    if root_body is None:
        raise SystemExit(f"No root body found in {source_xml}")
    root_body.insert(0, ET.Element("freejoint", {"name": "floating_hand_root"}))

    scene = ET.Element("mujoco", {"model": "floating_sharpa_wave_scene"})
    include = ET.SubElement(scene, "include", {"file": str((output_dir / "right_hand_floating.xml").resolve())})
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "720"})
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    asset = ET.SubElement(scene, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient", "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {"type": "2d", "name": "groundplane", "builtin": "checker", "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {"name": "groundplane", "texture": "groundplane", "texuniform": "true", "texrepeat": "4 4", "reflectance": "0.2"})
    worldbody = ET.SubElement(scene, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "pos": "0 0 -0.08", "size": "0 0 0.05", "type": "plane", "material": "groundplane"})

    floating_xml = output_dir / "right_hand_floating.xml"
    scene_xml = output_dir / "scene_right_floating.xml"
    tree.write(floating_xml, encoding="unicode")
    ET.ElementTree(scene).write(scene_xml, encoding="unicode")
    return scene_xml


def joint_address(model, name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(name)
    return model.jnt_qposadr[joint_id], model.jnt_range[joint_id]


def configure_camera():
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.12]
    camera.distance = 0.55
    camera.azimuth = 135.0
    camera.elevation = -18.0
    return camera


def set_pose(model, data, addresses, targets, root_pos=None, root_quat=None, root_offset=None):
    if root_pos is not None and root_quat is not None:
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_hand_root")
        if root_id >= 0:
            qadr = model.jnt_qposadr[root_id]
            data.qpos[qadr : qadr + 3] = np.asarray(root_pos, dtype=np.float64) + np.asarray(root_offset, dtype=np.float64)
            data.qpos[qadr + 3 : qadr + 7] = root_quat
    for joint_name, target_key in FIVE_FINGER_MAP.items():
        qadr, joint_range = addresses[joint_name]
        value = 0.0 if target_key is None else targets.get(target_key, 0.0)
        data.qpos[qadr] = clamp(value, float(joint_range[0]), float(joint_range[1]))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description="Render a five-finger Sharpa Wave hand from smoothed MediaPipe landmarks.")
    parser.add_argument("--scene", default="mujoco_menagerie/sharpa_wave/right_hand.xml")
    parser.add_argument("--folder", default="task_01/VID_20260616_183409_968_529")
    parser.add_argument("--average-track", help="Render an averaged track JSON instead of one video folder.")
    parser.add_argument("--output", default="outputs/sharpa_fivefinger_interpolated_holdout.mp4")
    parser.add_argument("--label", choices=["Left", "Right"], default="Right")
    parser.add_argument("--alpha", type=float, default=0.12)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.12])
    args = parser.parse_args()

    averaged_rows = None
    if args.average_track:
        import json

        averaged_rows = json.loads(Path(args.average_track).read_text())
        frame_count = len(averaged_rows)
        if args.max_frames:
            frame_count = min(frame_count, args.max_frames)
    else:
        folder = Path(args.folder)
        cap = cv2.VideoCapture(str(folder / "base.mp4"))
        if not cap.isOpened():
            raise SystemExit(f"Could not open {folder / 'base.mp4'}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if args.max_frames:
            frame_count = min(frame_count, args.max_frames)
        landmarks, detected = build_smoothed_landmarks(folder / "coordinates.json", frame_count, args.label, args.alpha)
        root_rows = build_smoothed_track(folder, preferred_label=args.label, alpha=args.alpha)[:frame_count]

    if averaged_rows is not None:
        root_positions = np.asarray([row["root_pos"] for row in averaged_rows[:frame_count]], dtype=np.float64)
    else:
        root_positions = np.asarray([row["root_pos"] for row in root_rows[:frame_count]], dtype=np.float64)
    root_offset = np.asarray(args.center, dtype=np.float64) - np.median(root_positions, axis=0)

    temp_dir = tempfile.TemporaryDirectory()
    scene = make_floating_scene(args.scene, Path(temp_dir.name))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    addresses = {name: joint_address(model, name) for name in FIVE_FINGER_MAP}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise SystemExit(f"Could not write {output}")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = configure_camera()
    try:
        for frame_idx in range(frame_count):
            if averaged_rows is not None:
                row = averaged_rows[frame_idx]
                targets = row["joint_targets_rad"]
                root_pos = row["root_pos"]
                root_quat = row["root_quat_wxyz"]
            else:
                targets = estimate_joint_targets(landmarks_to_hand(landmarks[frame_idx]))
                root_pos = root_rows[frame_idx]["root_pos"]
                root_quat = root_rows[frame_idx]["root_quat_wxyz"]
            set_pose(model, data, addresses, targets, root_pos=root_pos, root_quat=root_quat, root_offset=root_offset)
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if frame_idx % 100 == 0 or frame_idx == frame_count - 1:
                print(f"rendered frame {frame_idx + 1}/{frame_count}")
    finally:
        renderer.close()
        writer.release()

    print(f"wrote {output}")
    if averaged_rows is None:
        print(f"frames: {frame_count}  detected: {int(detected.sum())}  filled: {frame_count - int(detected.sum())}")
    else:
        print(f"frames: {frame_count}  averaged from {averaged_rows[0].get('source_count', 'unknown')} videos")
    temp_dir.cleanup()


if __name__ == "__main__":
    main()
