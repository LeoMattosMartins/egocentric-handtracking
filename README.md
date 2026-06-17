# egocentric-handtracking

Dimenso 2026 hackathon experiments for turning egocentric hand videos into
robot-hand motion targets.

## Current Pipeline

1. `batch_processing.py` scans `task_01/*/base.mp4`, runs MediaPipe Hand
   Landmarker, writes an `annotated.mp4`, and exports per-frame 21-point hand
   detections to `coordinates.json`.
2. Each video folder also has `imu.json`, sampled at 100 Hz. Each sample row is
   treated as `[time_ms, ax, ay, az, gx, gy, gz]`.
3. `retarget_hand.py` reads `coordinates.json` and `imu.json`, aligns every
   detected hand frame to the nearest IMU sample, runs a quaternion AHRS filter
   on the IMU stream, estimates generic humanoid hand joint targets, and writes:
   - `retargeted_hand.json`
   - `retargeted_hand.csv`
4. `smooth_hand_ik.py` and `render_interpolated_annotation.py` fill missing
   detections, smooth hand motion, and write an improved handout video for every
   source video.
5. `prepare_unitree_dataset.py` converts the repeated videos into imitation
   learning arrays from the full-frame smoothed/interpolated landmarks,
   AHRS-derived camera orientation, root motion, G1 wrist targets, and Unitree
   Dex3-1 hand targets.
6. `build_average_hand_motion.py` resamples the repeated videos to a common
   action timeline and averages the hand pose, root position, and IMU/AHRS root
   orientation into one canonical motion.
7. `train_intent_model.py` trains a small baseline behavioral-cloning model.
8. `render_unitree_video.py` renders retargeted labels or the trained model on
   the Unitree G1 29DOF MuJoCo model with a standing full-body controller.
9. `render_dex3_hand_video.py` and `render_sharpa_fivefinger_video.py` render
   articulated hand meshes before the motion is attached to the robot.

Run the retargeting step with:

```bash
uv run python retarget_hand.py --root task_01
```

If both hands are visible and you want to prefer one:

```bash
uv run python retarget_hand.py --root task_01 --label Right
```

## Retargeting Notes

The retargeted outputs are model-agnostic intermediate controls plus Unitree
mapping inputs. They contain radian values named like `index_mcp_flex`,
`index_pip_flex`, `thumb_cmc_flex`, and `pinky_abduction`, plus AHRS fields
(`imu_quat_*`, `imu_roll`, `imu_pitch`, `imu_yaw`) and world-frame wrist
estimates (`wrist_roll_world`, `wrist_pitch_world`, `wrist_yaw_world`).

Important caveat: the current `coordinates.json` stores MediaPipe normalized
image landmarks. Its `z` value is useful for relative hand shape, but it is not
metric world depth. The AHRS pass helps orient the camera frame with respect to
gravity, but it cannot recover absolute hand position by itself. For better
Unitree/MuJoCo retargeting, the next extraction pass should also store:

- MediaPipe `hand_world_landmarks`, if available from the model output.
- Per-frame timestamps from the video stream.
- Camera intrinsics or at least focal length / field of view.
- The exact Unitree hand MJCF joint and actuator names.

Once the Unitree MJCF is in the repo, map these generic targets to the
simulation joints, clamp them to each joint range, and drive the MuJoCo controls
from `retargeted_hand.csv` or `retargeted_hand.json`.

## Unitree MuJoCo

This workspace now includes a clone of
`https://github.com/unitreerobotics/unitree_mujoco` in `unitree_mujoco/`.

The clean G1 scene to use is:

```bash
unitree_mujoco/unitree_robots/g1/scene_29dof.xml
```

The checked-out `unitree_mujoco/unitree_robots/g1/scene.xml` appears to contain
generated terrain XML that is malformed on this machine, so the scripts default
to `scene_29dof.xml`.

The cloned G1/H2/H1-2 MJCF files include wrist joints and rubber/handless end
effectors, but not articulated finger bodies. Unitree's G1 docs do list the
Dex3-1 hand command order:

```text
thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1
```

For that reason, the current pipeline drives G1 legs/waist/arms/wrists in MuJoCo
and preserves Dex3-1 finger intent as training targets until a dexterous-hand
MJCF is added. The renderer defaults to `--pin-base` so the robot stands for
visualization while the upper body performs the intended action. Turn that off
with `--no-pin-base` when adding a real balance controller.

This repo also contains a hand-only Dex3 scene at:

```bash
unitree_mujoco/unitree_robots/g1/dex3_right_hand_scene.xml
```

It uses the Unitree G1 Dex3 palm/thumb/index/middle STL meshes already present
in `unitree_mujoco/unitree_robots/g1/meshes` and adds the seven documented
Dex3 hinge joints. This is a pre-robot visualization rig for validating hand
motion in space before mapping it onto the full humanoid.

For a true five-finger alternative, this workspace also includes
`mujoco_menagerie/`. The best current candidate is Sharpa Wave:

```bash
mujoco_menagerie/sharpa_wave/scene_right.xml
```

Sharpa Wave is a five-finger tactile hand with 22 DoF. It maps cleanly from the
21 MediaPipe landmarks because it has thumb, index, middle, ring, and pinky
joints. `render_sharpa_fivefinger_video.py` drives this model from the smoothed
interpolated hand landmarks. It can render either one source video or the
averaged action motion.

## Training And Playback

Generate or refresh retargeted labels:

```bash
uv run python retarget_hand.py --root task_01
```

Build the training dataset:

```bash
uv run python prepare_unitree_dataset.py --root task_01 --label Right
```

Train the baseline model:

```bash
uv run python train_intent_model.py --dataset training_data/unitree_intent_dataset.npz
```

Render AHRS-retargeted labels directly on standing Unitree G1:

```bash
uv run python render_unitree_video.py \
  --mode retargeted \
  --retargeted task_01/VID_20260616_183017_035_764/retargeted_hand.json \
  --output outputs/unitree_g1_ahrs_retargeted.mp4
```

Render the trained baseline policy on standing Unitree G1:

```bash
uv run python render_unitree_video.py \
  --mode policy \
  --output outputs/unitree_g1_ahrs_policy_standing.mp4
```

Generate smooth full-frame hand IK tracks:

```bash
uv run python smooth_hand_ik.py --root task_01 --label Right
```

Render improved interpolated handout videos into every video folder:

```bash
uv run python render_interpolated_annotation.py \
  --root task_01 \
  --label Right \
  --output-name interpolated_annotated.mp4
```

Render the smoothed IK hand motion before robot playback:

```bash
uv run python render_dex3_hand_video.py \
  --mode ik \
  --track task_01/VID_20260616_183409_968_529/hand_ik_smoothed.json \
  --output outputs/dex3_hand_ik_smoothed_holdout.mp4
```

Render trained-policy Dex3 predictions on the same smoothed hand path:

```bash
uv run python render_dex3_hand_video.py \
  --mode policy \
  --track task_01/VID_20260616_183409_968_529/hand_ik_smoothed.json \
  --sequence-id 5 \
  --output outputs/dex3_hand_policy_holdout.mp4
```

Render the five-finger Sharpa Wave alternative from the smoothed/interpolated
landmarks:

```bash
uv run python render_sharpa_fivefinger_video.py \
  --folder task_01/VID_20260616_183409_968_529 \
  --output outputs/sharpa_fivefinger_interpolated_holdout.mp4
```

Build the averaged motion from all repeated videos:

```bash
uv run python build_average_hand_motion.py \
  --root task_01 \
  --frames 360 \
  --output training_data/average_hand_motion.json
```

Render that averaged motion on the five-finger Sharpa Wave hand:

```bash
uv run python render_sharpa_fivefinger_video.py \
  --average-track training_data/average_hand_motion.json \
  --max-frames 360 \
  --output outputs/sharpa_fivefinger_average_motion.mp4
```

Current baseline output from the default leave-one-video-out split:

```text
train rows: 6111
holdout rows: 1013
holdout sequence: 5
observation rows: 7124
observation features: 72
Dex3 finger MAE: roughly 0.06-0.19 rad
G1 right wrist roll MAE: roughly 1.18 rad
G1 right wrist pitch MAE: roughly 0.81 rad
G1 right wrist yaw MAE: roughly 1.76 rad
```
