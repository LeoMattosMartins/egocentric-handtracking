# Egocentric Hand Motion To Robot Simulation

## Summary

This project explores whether first-person hand videos, plus smartglasses IMU
data, can be turned into robot imitation targets. The current pipeline takes
egocentric videos of repeated hand actions, extracts 21 MediaPipe landmarks,
fills missing detections with interpolation and smoothing, estimates hand and
wrist motion, averages repeated demonstrations, and renders that intent on
MuJoCo hand and Unitree robot assets.

At the beginning, I experimented with the idea of using full hand meshes instead
of point tracking. The intuition was biological: hands are not just keypoints,
they are soft 3D structures with constraints, surfaces, and occlusions, so a mesh
should describe the action more faithfully than sparse points. That is still
probably true. The practical blocker was compute. Reliable mesh reconstruction
from egocentric video would need heavier models, more tuning, and likely better
hardware than I had available for this stage, so I left mesh-based recovery as a
future improvement and built the current system around 21-point tracking.

## Goal

The final goal is to train a model that can infer the intended action from a set
of egocentric hand demonstrations and make a humanoid robot reproduce that
action in simulation before it ever runs on hardware.

The project has three concrete outputs right now:

- An interpolated handout video for every source video, showing what the system
  believes the hand did even through missing detections.
- A five-finger hand simulation showing the averaged action motion in 3D.
- A first imitation-learning dataset and baseline model that connect smoothed
  hand observations to robot wrist and hand targets.

## Data

Each video folder under `task_01/` contains:

- `base.mp4`: egocentric video of the hand action.
- `coordinates.json`: MediaPipe hand detections, with up to 21 landmarks per
  detected frame.
- `imu.json`: IMU samples used to estimate camera orientation.
- Generated outputs such as `retargeted_hand.json`, `hand_ik_smoothed.json`,
  and `interpolated_annotated.mp4`.

The current dataset contains six demonstrations of the same general action. The
videos do not have reliable detections on every frame, which became one of the
main engineering problems.

## Pipeline

1. `batch_processing.py` runs MediaPipe Hand Landmarker and exports
   `coordinates.json`.
2. `retarget_hand.py` aligns video frames to IMU samples and runs a quaternion
   AHRS-style complementary filter over the IMU stream.
3. `smooth_hand_ik.py` fills missing hand detections, smooths finger targets,
   estimates wrist/root position, and estimates wrist rotation from the hand
   landmarks. The glasses IMU is treated as camera/head context, not as a
   wrist-mounted sensor.
4. `render_interpolated_annotation.py` creates the interpolated handout videos.
5. `prepare_unitree_dataset.py` builds full-frame observations from smoothed
   landmarks, root position, root quaternion, detection flags, and action phase.
6. `build_average_hand_motion.py` resamples all videos onto a common action
   timeline and averages pose, root translation, and root orientation.
7. `render_sharpa_fivefinger_video.py` renders the averaged motion on a
   five-finger MuJoCo hand model.
8. `train_intent_model.py` trains a ridge-regression baseline as a first sanity
   check for imitation learning.

## What Changed During The Journey

The earliest version treated MediaPipe landmarks almost directly as the motion
source. That produced visible motion, but the robot/hand looked too stiff and
the missing frames created discontinuities.

The next step was to use the IMU stream. The IMU does not solve absolute
position by itself, but it gives a better estimate of camera orientation and
gravity context. Running an AHRS filter before retargeting made the egocentric
data less detached from the real world.

Then the focus moved to interpolation. The model should not learn only from the
frames where MediaPipe succeeded; it should learn from a consistent time series
that includes the system's best estimate between detections. That led to the
full-frame `hand_ik_smoothed.json` files and the updated training dataset.

The Unitree assets were another turning point. The available Unitree G1 scene in
`unitree_mujoco` has robot wrist joints, but not a true five-finger articulated
hand. The Unitree Dex3 hand has articulated fingers, but it is a three-finger
hand. To test five-finger motion, I added MuJoCo Menagerie and used the Sharpa
Wave five-finger hand as an alternative visualization model.

The most recent correction was wrist rotation. The first five-finger simulation
could flex fingers and move the hand through space, but it did not visibly rotate
the wrist. A first attempt used the IMU too aggressively, which made the wrist
bend back and wander because the IMU is mounted on smartglasses, not on the hand.
The latest version uses the IMU as camera/head context and estimates wrist
rotation primarily from the palm landmarks: the first reliable palm pose becomes
a calibration reference, and later palm poses become relative wrist rotation.

## Current Results

Generated files:

- `outputs/sharpa_fivefinger_average_motion.mp4`
- `outputs/sharpa_fivefinger_average_motion.preview.jpg`
- `training_data/average_hand_motion.json`
- `training_data/unitree_intent_dataset.npz`
- `training_data/intent_ridge_model.npz`
- `task_01/VID_*/interpolated_annotated.mp4`

The rebuilt dataset has:

- 7,124 observation rows.
- 72 observation features.
- 6 G1 wrist target values.
- 7 Unitree Dex3 target values.

The current baseline model is intentionally simple. After adding stronger wrist
rotation targets, the wrist prediction errors increased:

- Right wrist roll MAE: about 0.61 rad.
- Right wrist pitch MAE: about 0.08 rad.
- Right wrist yaw MAE: about 0.22 rad.
- Dex3 finger MAE: about 0.13 to 0.26 rad.

That is not a failure of the whole pipeline. It is evidence that a linear ridge
model is not expressive enough once the target includes real wrist orientation
dynamics.

## Limitations

The biggest limitation is depth. The current `coordinates.json` uses normalized
MediaPipe image landmarks. The `z` coordinate helps describe relative hand
shape, but it is not calibrated metric depth. As a result, the hand can move in
3D for visualization, but the translation is still an approximation.

The second limitation is occlusion. Egocentric hand videos often hide fingers
behind the palm or object. Interpolation smooths over those gaps, but it cannot
recover information that was never observed.

The third limitation is model quality. The ridge baseline is useful as a
debugging tool because it is fast and easy to inspect, but it is too weak for
orientation-heavy motion. A temporal model should replace it.

The fourth limitation is robot embodiment. The Unitree G1 assets in this setup
do not include a true five-finger hand. The current five-finger simulation uses
Sharpa Wave as a proxy, while the Unitree path still uses G1 wrist targets and
Dex3-style hand targets.

Finally, standing balance is not solved. The G1 renderer can pin the base for
visualization, but a hardware-ready version needs a whole-body controller that
keeps the robot stable while the arm and hand perform the action.

## Cost And Feasibility

The current prototype is feasible on a local machine because most of the work is
CPU-friendly. MediaPipe landmark extraction, interpolation, ridge-regression
training, and MuJoCo rendering can run without a dedicated GPU. That makes the
pipeline practical for iteration: I can regenerate `coordinates.json`, rebuild
the smoothed tracks, create the per-video handouts, train the baseline model, and
render the five-finger average without needing cloud infrastructure.

The cost is mainly time and engineering complexity rather than paid compute. The
pipeline has many fragile transforms: egocentric camera coordinates to hand
coordinates, hand coordinates to robot targets, repeated demonstrations to an
average action, and then robot targets to simulation. Each transform can look
reasonable by itself while still producing motion that is wrong in the final
render. That is why the visual handout videos are important. They make the
intermediate estimate inspectable before it is fed into the robot model.

In practical terms, the current direct cost is low if a capable laptop or desktop
is already available: the software stack is open source, the videos are local,
and the baseline model trains quickly. The hidden cost is iteration time. Every
change to detection, smoothing, orientation, or retargeting has to be rerun and
watched because the numerical outputs alone do not reveal whether the simulated
hand actually looks plausible.

For this project stage, the most feasible path is to keep the perception model
lightweight, keep the robot simulation offline, and use the rendered videos as
the main validation tool. Running on the real robot is not yet feasible without
better balance control, stronger wrist estimation, and more reliable handling of
occlusion.

The less feasible path, at least with the available time and compute, is full
GPU-heavy reconstruction. Mesh models, image segmentation models, and temporal
deep learning models would likely improve the result, but they introduce higher
setup cost, longer training/inference time, and more failure modes. They are
better framed as extensions once the simpler coordinate pipeline is already
stable.

## Next Steps

The most important next step is to replace the linear baseline with a temporal
model such as a small Transformer, TCN, or GRU that predicts wrist and finger
targets from a window of smoothed observations.

The data pipeline should also store better source information:

- MediaPipe world landmarks, if available.
- Video timestamps per frame.
- Camera intrinsics or field of view.
- A calibrated transform from camera/IMU frame to robot frame.
- Object or contact cues if the action involves manipulating something.

The simulation path should move from a floating hand to a whole-body robot with
a real balance controller. The current five-finger hand render is the right
debugging stage before robot attachment, but it is not the final embodiment.

Mesh-based hand recovery should also return as a future line of work when more
compute is available. A mesh or MANO-style hand model could enforce anatomical
constraints, reduce impossible finger poses, and give a better bridge from
egocentric perception to robot imitation.

With more time, I would revisit the left-hand blocking idea in a more careful
way. The goal was reasonable: hide or recolor the non-target hand before running
MediaPipe so the detector tracks the right hand more consistently. The quick
version was not reliable enough, so it was removed from the final run. A better
version would use a GPU image model or segmentation model to identify the left
hand per frame, then either paint it a high-contrast color, blur it, or replace
it with a solid mask before landmark extraction. This would need validation
against the original video so it does not accidentally destroy right-hand
evidence or create artifacts that confuse MediaPipe.

A stronger GPU-based extension would combine several models: a detector or
segmenter for hand identity, a mesh or MANO-style hand estimator for anatomical
pose, and a temporal model that learns motion continuity across frames. That
would make the system less dependent on frame-by-frame landmark success and
could reduce jitter, identity swaps, and impossible wrist poses. The tradeoff is
that this becomes a compute project as much as a robotics project; it would
likely require a good CUDA GPU or cloud instance, more tuning, and a more formal
evaluation set.

## Bottom Line

The project has moved from raw egocentric hand detections to a working
retargeting and visualization pipeline. It now produces smoothed per-video
handouts, an averaged five-finger motion in MuJoCo, and a training dataset that
uses interpolated hand motion earlier in the learning path. The result is not yet
robot-ready, but it is a much clearer prototype: the system can show what it
thinks the action was, where the uncertainty is, and what needs to improve before
the motion can transfer to a standing humanoid robot.
