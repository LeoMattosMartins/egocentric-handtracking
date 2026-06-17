G1_WRIST_JOINTS = [
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_STANDING_JOINT_TARGETS = {
    "left_hip_pitch_joint": -0.10,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.25,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.10,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.25,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.65,
    "left_shoulder_roll_joint": 0.35,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.95,
    "right_shoulder_pitch_joint": 0.65,
    "right_shoulder_roll_joint": -0.35,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.95,
    **{joint: 0.0 for joint in G1_WRIST_JOINTS},
}

DEX3_JOINTS = [
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "index_0",
    "index_1",
    "middle_0",
    "middle_1",
]


def clamp(value, low, high):
    return max(low, min(high, value))


def dex3_targets_from_retargeted(joint_targets):
    """Map MediaPipe-derived finger angles to Unitree Dex3-1 command order."""
    return {
        "thumb_0": clamp(joint_targets.get("thumb_abduction", 0.0), -1.2, 1.2),
        "thumb_1": clamp(joint_targets.get("thumb_cmc_flex", 0.0), 0.0, 1.8),
        "thumb_2": clamp(
            0.5 * (joint_targets.get("thumb_mcp_flex", 0.0) + joint_targets.get("thumb_ip_flex", 0.0)),
            0.0,
            1.8,
        ),
        "index_0": clamp(joint_targets.get("index_mcp_flex", 0.0), 0.0, 1.8),
        "index_1": clamp(
            0.5 * (joint_targets.get("index_pip_flex", 0.0) + joint_targets.get("index_dip_flex", 0.0)),
            0.0,
            1.8,
        ),
        "middle_0": clamp(joint_targets.get("middle_mcp_flex", 0.0), 0.0, 1.8),
        "middle_1": clamp(
            0.5 * (joint_targets.get("middle_pip_flex", 0.0) + joint_targets.get("middle_dip_flex", 0.0)),
            0.0,
            1.8,
        ),
    }


def g1_wrist_targets_from_retargeted(joint_targets, hand_label):
    """Map retargeted wrist pose to the G1 29DOF wrist joints present in MJCF."""
    side = "left" if hand_label == "Left" else "right"
    targets = {name: 0.0 for name in G1_WRIST_JOINTS}
    roll = joint_targets.get("wrist_roll_world", joint_targets.get("wrist_roll_from_camera", 0.0))
    pitch = joint_targets.get("wrist_pitch_world", joint_targets.get("wrist_pitch_from_camera", 0.0))
    yaw = joint_targets.get("wrist_yaw_world", 0.0)
    targets[f"{side}_wrist_roll_joint"] = clamp(roll, -1.6, 1.6)
    targets[f"{side}_wrist_pitch_joint"] = clamp(pitch, -1.2, 1.2)
    targets[f"{side}_wrist_yaw_joint"] = clamp(yaw, -1.2, 1.2)
    return targets


def g1_full_body_targets_from_retargeted(joint_targets, hand_label):
    targets = dict(G1_STANDING_JOINT_TARGETS)
    targets.update(g1_wrist_targets_from_retargeted(joint_targets, hand_label))
    return targets
