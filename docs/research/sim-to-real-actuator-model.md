# Sim-to-real actuator modeling for Le Snail

Date: 2026-10-14

## Summary and recommendation

**What the "NN that models the joint responses" is.** It is Strategy 4 of NVIDIA's course *Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac* [C15]. The strategy pairs two tools:
- **SAGE** replays the same motion files on the real and the simulated SO-101 and measures the per-joint gap.
- **GapONet** is a learned model that maps commanded actions to real motion and outputs a compensated (delta) action.

The idea descends from the ANYmal actuator network [H19]: an MLP from position-error and velocity history to torque, trained on under 4 minutes of real data. Isaac Lab v2.3.2 ships that idea as `ActuatorNetMLP` and `ActuatorNetLSTM` [IL-net].

**Why it does not fit Le Snail directly.**
- Our STS3215 servos run their own position loop and cannot measure torque, so a torque-output network is a poor first fit.
- GapONet's public code has no SO-101 task. It targets Isaac Sim 4.5, and its SO-101 data was collected on a fixed-root arm, while Le Snail carries its body weight through its joints.

**Our provisional actuator values looked wrong.** The datasheet gives about 1.91 N·m stall torque and about 5.45 rad/s no-load speed at 7.4 V [FT-DS], against our 0.8 N·m effort limit and 2 rad/s simulated velocity limit. Published SO-101 sim gains also disagree by about 100× (Q4).

> **Update 2026-10-15/16.** Le Snail's servos are the **12 V** STS3215 (C018/C047, sticker-checked), not the 7.4 V C001 this study first assumed: stall ≈ 2.94 N·m, rated 0.98 N·m, no-load ≈ 4.7 rad/s, bench-tested stable dynamic torque ≈ 1.47 N·m, backlash ≈ 0.87° (#12, 2026-10-15). Physics-v2 replaced the 0.8 N·m / 2 rad/s placeholders with 1.5 N·m nominal (0.98–2.94 randomized) and 4 rad/s (3.0–4.7). Measured register state and bus rate are in [Stage 0 and Stage 1 dry-run results](#stage-0-and-stage-1-dry-run-results-2026-10-16).

**Recommendation: measure first, then identify a parametric model, and escalate to a learned one only if structured residuals remain.**
1. Measure the gap with SAGE-style paired replay. Use SAGE's motion files and metrics, but our own logger and our own Isaac Lab env.
2. Identify delay, target rate/acceleration limits, PD, effort limit, friction and armature, and re-measure. At a 20 Hz policy this is likely to close most of the gap.
3. Only then, if needed, fit a small learned position-response model in a custom 200 Hz actuator class, validated on floor-contact data.

The course itself says "small gaps: tune parameters; large gaps: use GapONet" [C15].

## Which NVIDIA material?

The user confirmed it is the course **"Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac"**, base URL https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/ [C-index]. The course targets Isaac Sim 5.1.0, Isaac Lab 2.3.0, LeRobot 0.4.3 and GR00T N1.6 [C06]. Its workshop repo's simulation Dockerfile builds on the `nvcr.io/nvidia/isaac-lab:2.3.2` image [WS, `docker/sim/Dockerfile:1`], which matches our pin.

Relevant chapters:

| Chapter | What it is | Relevance to Le Snail |
|---|---|---|
| 07 Calibrating SO-101 [C07] | Runs `lerobot-calibrate` for the leader and follower, plus a calibration check script. The follower runs at 12 V. | Calibration procedure only. It documents no servo PID or register settings. |
| 09 Strategy 1, domain randomization + teleop [C09] | Randomizes lighting, camera pose and object placement. Physics randomization (mass, friction, damping, actuator delay) is listed only as examples, with no values. | Little. No SO-101 actuator parameters are given. |
| 13 Strategy 2, co-training [C13] | Fine-tunes the GR00T N1.6 VLA on about 5 real teleop episodes plus 70–100 sim demos. No results or mixing ratio are given. | This is imitation learning, not an actuator model. All that carries over to PPO locomotion is the idea of keeping some real data in the loop. We have no real demonstrations of crawling to co-train on. |
| 14 Strategy 3, Cosmos augmentation [C14] | Visual augmentation. The page states: "We have not measured or addressed the actuation gap." | None. |
| **15 Strategy 4, SAGE + GapONet [C15]** | SAGE does paired real/sim replay and quantifies the per-joint gap. GapONet "learns a neural network model of actuator behavior". The page says: "Our SO-101 setup collected 8 hours of real trajectory data for such training." | **This is the material the user meant.** Details are below. |

Other NVIDIA material on learned actuator models, for completeness:
- The *Getting Started With Isaac Lab* course has an "Actuator Modeling" page. It describes Isaac Lab actuator networks (command plus joint-state history to torque) that are frozen and replace the PD loop in simulation [NV-AM].
- The Isaac Lab actuator API reference [IL-api].
- NVIDIA's Newton "Neural Dynamics" (NeRD) blog [NV-NeRD] describes a learned model that replaces the whole physics solver, not an actuator model.
- We found no NVIDIA developer-blog post that describes an actuator network for SO-101. The Spot sim-to-real blog does not describe one either.

### SAGE (Sim2Real Actuator Gap Estimator)

Repo: `isaac-sim2real/sage` at commit `5b31cfa35416d1300b5099557049bd8e9cb4507d` [SAGE].

- **Targets:** Isaac Lab 2.2.0, Isaac Sim 5.0.0, Python 3.10 [SAGE, `README.md` badges/prerequisites]. The simulation half uses the Isaac Sim core `Articulation` API directly, not an Isaac Lab env:
  - `set_gains(kps, kds)` and `set_effort_modes("force")` [SAGE, `sage/simulation.py:227-236`];
  - an optional fixed root [`sage/simulation.py:175-181`];
  - position targets updated every `render_freq/control_freq` physics steps [`sage/simulation.py:420-450`].
- **Simulation parameters disagree in three places.**
  - The course runs `--fix-root --physics-freq 200 --render-freq 200 --control-freq 50 --kp 100 --kd 2` [C15].
  - At this commit, `sage/assets.py:52-58` sets SO-101 `default_kp=1000.0`, `default_kd=200.0` and `default_control_freq=20`. Command-line arguments override these.
  - The real-robot collector defaults to `control_freq=50` [`sage/real_so101/so101_lerobot_collector.py:608-618`].
  - SAGE's README itself warns that mismatched controller settings "will inflate the measured gap" [SAGE, `README.md` "Starting from Real Data"].
- **How real SO-101 data is logged** [`sage/real_so101/so101_lerobot_collector.py`]:
  - It uses `lerobot.motors.feetech.FeetechMotorsBus` directly at 1 Mbps (lines 151, 200-213). It does not go through `SO101Follower`.
  - On connect it writes only `P_Coefficient=16`, `I_Coefficient=0` and `D_Coefficient=32`, plus gripper `Max_Torque_Limit=500` and `Protection_Current=250` (lines 216-227). Every other register keeps its stored value.
  - Each loop it writes `Goal_Position` to each motor individually (lines 331-350).
  - It then reads `Present_Position`, `Present_Velocity` and `Present_Current` **one motor and one register at a time**: 18 transactions, no sync read (lines 242-271).
  - Velocity is converted as raw × 2π/4096 (lines 258-264).
  - "Torque" is `Present_Current × 0.022814`, which the code comments as "STS3215 12V, estimated by measuring stall current" (lines 266-270). Le Snail uses the 7.4 V variant, so this constant does not apply.
  - **Timestamps:** one `time.monotonic()` per loop, taken *before* the write, is used for both the command row and the state row (lines 497-514). They are saved as microseconds in `control.csv` and `state_motor.csv`, plus an `event.csv` with `MOTION_START` (lines 541-605).
  - The loop runs at 50 Hz by default, with motion files assumed to be 50 Hz (lines 139, 446-466; `docs/LEROBOT_REAL.md:70`).
  - Our estimate: at roughly 1 ms per transaction (Q4), 24 sequential transactions take about 20+ ms, i.e. the whole 20 ms period at 50 Hz. The real replay probably overruns and plays slower and irregularly, which inflates the measured gap independently of actuator mismatch. This is another reason to use our own logger.
  - Joint zero is taken as the midpoint of the calibrated encoder range. Offsets and inversions are all zero or false by default (lines 71-92, 306-329).
- **Gap metrics** [`sage/analysis.py`]:
  - Real and sim series are aligned on the `MOTION_START` event and linearly resampled to a common `sample_dt` (lines 332-386).
  - For position, velocity and torque, per joint, it computes:
    - RMSE (lines 626-627);
    - MAPE, excluding samples where |real| < 0.1 (lines 629-649);
    - Pearson correlation (line 652);
    - cosine similarity (line 656);
    - cross-correlation lag, `max_lag` in seconds (lines 659-666);
    - for velocity and torque only, derivative-magnitude "smoothness" statistics (lines 669-677).
- **Motion files:** 50 SO-101 files in `motion_files/so101/custom/`, including `step_response`, `chirp_sweep`, `multisine`, `prbs_excitation`, `backlash_detection`, `friction_gravity`, `hold_under_gravity`, `velocity_ramp` and `random_waypoints`. Each is a CSV of joint angles in radians at 50 Hz. Joint names are `Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw` [`configs/so101_valid_joints.txt`]. By our count the files hold about 225k frames, roughly 75 min at 50 Hz.
- **Asset:** `assets/so101/SO-ARM101-USD.usd`. It is byte-identical (same MD5) to the workshop's `SO-ARM101-USD.usd`.

### GapONet

Repo: `jiemingcui/gaponet` at commit `91eb8819c3156b46fdb3266a048600c67c9adf5d` [GAPO]. The paper is "Bridging the Sim-to-Real Gap in Humanoid Dynamics via Learned Nonlinear Operators", NeurIPS 2025 [GAPO-paper]. Its abstract reports a 257,493-frame humanoid dataset (SimLifter) and "over 40%" better zero-shot tracking than default PD. We read only the abstract.

- **Targets:** "Isaac Sim 4.5.0+" per the README, whose install command pins `isaacsim==4.5.0`. It vendors its own Isaac Lab copy: `isaaclab` extension version 0.36.0, with classifier "Isaac Sim :: 4.5.0" [GAPO, `README.md`; `source/isaaclab/setup.py`, `config/extension.toml`]. This is not our Isaac Lab 2.3.2 / Isaac Sim 5.1.0 stack. Isaac Lab's joint-friction semantics also changed between Sim 4.5 (a coefficient) and 5.x (an effort) [IL-basecfg, lines 152-153]. Friction values identified under GapONet's stack therefore do not port directly to ours.
- **No SO-101 support in the code.** At this commit the repo contains only humanoid tasks (Unitree H1-2 and "Fourior") and no SO-101 task, config or data. The course says "its integration in SAGE is in progress" [C15].
- **What it is: a delta-action wrapper, not an actuator replacement.** It is trained with PPO inside a DirectRLEnv:
  - The action is a per-joint position offset, added to the *recorded real command* before `set_joint_position_target` [GAPO, `.../humanoid_operator/humanoid_operator_env.py:218-226`].
  - The reward is −(position error² + 0.01 × velocity error²) against the *recorded real* joint position and velocity (lines 1007-1025).
  - Physics runs at 200 Hz with decimation 4, i.e. 50 Hz actions [`humanoid_operator_env_cfg.py:46,75`].
  - The DeepONet trunk input is the current commanded action (10 joints) (lines 824-859; `agents/rsl_rl_operator_cfg.py:20`).
  - The branch input is a 400-dimensional "sensor" embedding of probe responses, which a model network can predict from current position plus a **4-step history of (position, velocity, target)** [`humanoid_operator_env_cfg.py:173-176`; `rsl_rl_operator_cfg.py:31-36`; `humanoid_operator_env.py:891-910`].
  - Training data is `.npz` with `real_dof_positions`, `real_dof_velocities`, `real_dof_positions_cmd` and `real_dof_torques` [GAPO, `README.md` "Motion Data"].
  - We did not verify the exact semantics of the branch "sensor" sub-environments beyond this reading.
- **Course framing:** the model can be used "inside sim so that the environment better matches real actuation" or "on the real robot at inference time" [C15]. The SO-101 compensation loop "is under active development" [C15]. No SO-101 dataset is published; the published datasets cover H1-2 and RealMan [SAGE, `README.md`].

### Do their SO-101 assets match our URDF?

**No, not directly.**
- **Joint names differ.** SAGE and the workshop use `Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw` [WS, `source/sim_to_real_so101/assets/so101.py:42-49`]. Our TheRobotStudio URDF (SO-ARM100 commit `a758567`) uses `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
- **Joint conventions differ between NVIDIA's own tools:**
  - SAGE uses "zero = calibrated mid-range" with no sign changes [SAGE collector, lines 306-329].
  - The workshop linearly maps LeRobot's −100..100 range onto per-joint USD ranges in degrees [WS, `source/sim_to_real_so101/utils/lerobot_interface.py:44-52,139-168`].
  - Le Snail already requires its own measured LeRobot-to-URDF mapping (`docs/hardware.md`).
- **Not verified:** the USD's axis directions and zero poses. The USD is a binary crate file that we did not inspect.
- **Do not reuse the workshop's gains.** The workshop `SO101_CFG` uses implicit actuators with hand-set gains: stiffness 4–55 and damping 0.3–0.8 per joint, and `effort_limit_sim=30` for every joint [WS, `source/sim_to_real_so101/assets/so101.py:54-101`]. Its comments quote joint torques of "34.4–62.1 N-m", which are an order of magnitude above the STS3215's rating (see Q4). Neither set of values is a measurement.

## 1. What is an actuator network? (Hwangbo et al. 2019)

The source is Hwangbo et al., "Learning agile and dynamic motor skills for legged robots", *Science Robotics* 4(26), eaau5872 (2019), arXiv:1901.08652 [H19].

- **Problem.** ANYmal's series-elastic actuators have delays and dynamics that a rigid-body simulator does not capture. An "ideal" actuator model assumes zero latency and infinite bandwidth [H19, Results].
- **Inputs:**
  - The network takes a history of joint **position error** (commanded minus actual) and joint **velocity** [H19, Methods, "Modeling the actuation"].
  - "We use a history consisting of the current state and two past states that correspond to t − 0.01 and t − 0.02 seconds". That is 6 inputs per joint.
  - The history should be "sufficiently longer than the sum of all communication delays and the mechanical response time", and was tuned on validation error [H19].
  - Each joint is modeled independently, with one shared network for all 12 joints [H19].
- **Output:** joint torque [H19, Fig. 5].
- **Architecture:** an MLP with 3 hidden layers of 32 units and softsign activation. Evaluating it for 12 joints takes 12.2 µs [H19].
- **Data collection on hardware:**
  - A parameterized controller drove sinusoidal foot trajectories through inverse kinematics while the feet made and broke ground contact. Amplitude ranged over 5–10 cm and frequency over 1–25 Hz, and the robot was disturbed by hand [H19].
  - "The excitation must cover a wide range of frequency spectra", otherwise the model produced unnatural oscillation [H19].
  - Torque labels came from the actuators' own torque measurement, which has 0.2 Nm resolution [H19].
- **Amount of data:** "less than 4 min" at 400 Hz from 12 actuators in parallel, more than a million samples, split 90/10 into train and validation [H19].
- **Accuracy:**
  - Validation average error was 0.740 Nm, against 3.55 Nm for the ideal model.
  - On test data from learned policies the error was 0.966 Nm, against 5.74 Nm [H19].
- **Use in simulation:** at each step the network maps the buffered joint-state history and the policy's position targets to torques for the rigid-body simulator [H19, Methods]. The paper does not state the simulation timestep. The deployed policy ran at 200 Hz [H19].
- **Sim-to-real effect:**
  - Policies trained with the ideal model or a hand-tuned analytical model "could not make a single step without falling", even after "more than a week" of tuning the analytical model [H19].
  - With the actuator net, the policy broke ANYmal's previous speed record by 25%, reaching 1.5 m/s on hardware, and tracked commands better than the best model-based controller [H19].
- **Follow-up work:**
  - Lee et al. 2020 reused the same 6-input feature set [Lee20].
  - Rudin et al. 2022 (Isaac Gym, legged_gym) switched to an **LSTM** fed only current measurements. They found 0.005 s (200 Hz) to be the largest stable step, because "the actuator network (approximating a discrete-time PD-controller) becom[es] unstable" [Rudin22].

## 2. Isaac Lab v2.3.2 actuator support

Code is at tag `v2.3.2`, commit `37ddf626871758333d6ed89cf64ad702aef127d0`. Paths are under `source/isaaclab/isaaclab/` [IL].

**Classes and configs.** Package exports are in `actuators/__init__.py:25-36`.

| Class (file:lines) | Config (file:lines) | Model |
|---|---|---|
| `ImplicitActuator` (`actuator_pd.py:36-142`) | `ImplicitActuatorCfg` (`actuator_pd_cfg.py:18-26`) | PhysX's continuous PD drive. `compute()` passes the targets through and only estimates torque for logging (117-142). The docstring cautions that it is "only provided for consistency" (49-53). |
| `IdealPDActuator` (`actuator_pd.py:150-200`) | `IdealPDActuatorCfg` (`actuator_pd_cfg.py:34-38`) | Explicit τ = kp·e + kd·ė + τ_ff, clipped to `effort_limit`. |
| `DCMotor` (`actuator_pd.py:203-307`) | `DCMotorCfg` (`actuator_pd_cfg.py:41-48`) | Ideal PD plus a four-quadrant linear torque-speed clip using `saturation_effort` (stall torque), `velocity_limit` (no-load speed) and `effort_limit` (continuous torque) (295-307). Both `saturation_effort` and `velocity_limit` are required (266-277). |
| `DelayedPDActuator` (`actuator_pd.py:310-367`) | `DelayedPDActuatorCfg` (`actuator_pd_cfg.py:51-61`) | Ideal PD with the position, velocity and effort commands delayed by an integer number of **physics steps**. The delay is drawn uniformly from `[min_delay, max_delay]` on each reset (343-357). |
| `RemotizedPDActuator` (`actuator_pd.py:370-451`) | `RemotizedPDActuatorCfg` (`actuator_pd_cfg.py:64-81`) | Delayed PD with an angle-dependent torque limit taken from a lookup table. |
| `ActuatorNetMLP` (`actuator_net.py:101-188`) | `ActuatorNetMLPCfg` (`actuator_net_cfg.py:29-64`) | History MLP (Hwangbo). Subclass of `DCMotor`. |
| `ActuatorNetLSTM` (`actuator_net.py:31-98`) | `ActuatorNetLSTMCfg` (`actuator_net_cfg.py:16-26`) | LSTM (Rudin). Subclass of `DCMotor`. |

**Actuator-net details:**
- **File format.** The network must be TorchScript. It is loaded with `torch.jit.load(read_file(cfg.network_file))` (`actuator_net.py:49-51`, `126-128`). `read_file` accepts a local path or a Nucleus URL (`utils/assets.py:114`).
- **MLP input layout** (`actuator_net.py:146-174`):
  - Two histories are kept: position error `q_des − q` and velocity `q̇`, each of length `max(input_idx)+1` (131-135). Both are rolled by one slot on every `compute()` call (151-157).
  - `input_idx` selects which history slots feed the network, where 0 is the current slot and n is n calls ago (`actuator_net_cfg.py:58-64`).
  - The input is `[pos_err[idx]·pos_scale, vel[idx]·vel_scale]` for `input_order="pos_vel"`, or the reverse for `"vel_pos"` (166-174). The network sees one row per joint, so the batch is `num_envs·num_joints` (162-165).
  - The output is multiplied by `torque_scale` (179).
- **LSTM input and state:**
  - The input is `[q_des − q, q̇]` with shape (N·J, 1, 2) (81-82).
  - The scripted module must expose `.lstm`, from which the layer count and hidden size are read (54-55), and must return `(torque, (h, c))` (86-88).
  - Hidden state is reset per environment in `reset()` (71-75).
- **Output meaning:**
  - The output is joint **effort (torque)**. It is clipped by the `DCMotor` torque-speed curve.
  - The class returns efforts only, setting position and velocity targets to `None` (91-98, 182-188).
  - The configs set `stiffness=None` and `damping=None` because the network replaces the PD loop (`actuator_net_cfg.py:21-23`, `34-37`).
- **Run rate: every physics step.**
  - `DirectRLEnv.step()` loops `decimation` times. Each iteration calls `_apply_action()`, then `scene.write_data_to_sim()`, then `sim.step()` (`envs/direct_rl_env.py:370-375`).
  - `Articulation.write_data_to_sim()` calls `_apply_actuator_model()`, which calls each actuator's `compute()` (`assets/articulation/articulation.py:218-264`, `1869-1906`).
  - For Le Snail (200 Hz physics, decimation 10) the network runs at **200 Hz**. One history slot is 5 ms, so `input_idx=[0,2,4]` would mean 0/10/20 ms.
- **Explicit vs implicit:**
  - At init, implicit actuators write their stiffness and damping into PhysX.
  - For explicit actuators, including the actuator nets, the PhysX gains are set to **zero** and the actuator computes torque itself (`articulation.py:1760-1769`). `effort_limit_sim` then defaults to 1e9 so PhysX does not clip a second time (`actuator_base.py:106-111,174-175`).
  - PhysX position and velocity targets are written only when implicit actuators exist (`articulation.py:262-264`).
  - Since Isaac Sim 5, joint `friction`, `dynamic_friction` and `viscous_friction` are efforts rather than coefficients (`actuator_base_cfg.py:138-165`).

**Notes for our DirectRLEnv:**
1. **Gain randomization must change.** Our env randomizes gains with `write_joint_stiffness_to_sim` and `write_joint_damping_to_sim` (`src/le_snail/sim/env.py:316-317`). That is correct for `ImplicitActuator`. With an explicit actuator it would re-enable a PhysX PD drive on top of the actuator's torque. Randomize the actuator's own `stiffness` and `damping` tensors instead.
2. **Actuator state must be reset.** Actuator histories and LSTM state reset only when `robot.reset(env_ids)` is called. Check that `_reset_idx` reaches it.
3. **No torque labels.** `ActuatorNet*` needs torque labels, and the STS3215 cannot measure torque (`Present_Current` and `Present_Load` are proxies; see Q4).
4. **Lower-risk custom actuator.** A cleaner custom actuator for a position servo is a subclass of `ImplicitActuator`. It must be a subclass, not just `ActuatorBase` with an implicit flag, because `Articulation` checks `isinstance(actuator, ImplicitActuator)` before writing PhysX gains and enabling position targets (`articulation.py:1760-1764`). Its `compute()` would rewrite `control_action.joint_positions`, applying delay, slew or acceleration limits, a deadband, or a learned position offset. `_apply_actuator_model()` copies non-`None` positions into the PhysX targets (`articulation.py:1892-1897`), so the stable implicit PD is kept at 200 Hz. The `ImplicitActuator` docstring discourages relying on the class, so verify this with a step test.
5. **Stability check.** An explicit PD at 5 ms on the light SO-101 links may be stiff or unstable. Step-test it in sim, and set `armature`, before committing to an explicit model.
6. **Required config fields.** Because `ActuatorNet*Cfg` inherits `DCMotorCfg`, `saturation_effort` and `velocity_limit` are required.

## 3. Existing assets that use an actuator net

The asset configs are in `source/isaaclab_assets/isaaclab_assets/robots/` [IL-assets].

- **ANYmal-B, C and D** use `ANYDRIVE_3_LSTM_ACTUATOR_CFG` = `ActuatorNetLSTMCfg` [IL-assets, `anymal.py:45-52`; used at lines 87, 122, 157].
  - `network_file`: `{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt`.
  - `saturation_effort` 120, `effort_limit` 80, `velocity_limit` 7.5.
  - A `DCMotorCfg` fallback, `ANYDRIVE_3_SIMPLE_ACTUATOR_CFG`, is at lines 34-42.
  - The net is hosted on NVIDIA's asset server under `.../Isaac/IsaacLab/` (`utils/assets.py`).
  - **Unverified:** its training data and provenance. Presumably it is ANYbotics/ETH work, following [Rudin22], but no source we read states this.
- **Unitree Go1** uses `GO1_ACTUATOR_CFG` = `ActuatorNetMLPCfg` [IL-assets, `unitree.py:32-49`].
  - `network_file`: `{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/Unitree/unitree_go1.pt`.
  - `input_idx=[0,1,2]`, `input_order="pos_vel"`, `pos_scale=-1.0`, `vel_scale=1.0`, `torque_scale=1.0`, effort/saturation 23.7, velocity 30.
  - The docstring says: "This model is taken from: https://github.com/Improbable-AI/walk-these-ways".
  - `pos_scale=-1` means that network was trained on `q − q_des`, the opposite sign to Isaac Lab's buffer. Keep the sign convention explicit when we train our own.
- **No Isaac Lab asset ships an actuator net for SO-101 or any hobby servo.**

## 4. Applying this to STS3215 position servos

### STS3215 facts (7.4 V variant, ST-3215-C001)

> **Wrong variant for Le Snail.** Our servos are the 12 V STS3215; use the 12 V figures in the summary's update note. The 7.4 V table is kept for the BAM comparison below, whose fits are for the 7.4 V servo.

**Official Feetech datasheet**, edition A/0, 2020-04-10 [FT-DS]:

| Item | Value |
|---|---|
| Stall torque | 19.5 kg·cm ≈ **1.91 N·m** at 7.4 V; 16.5 kg·cm ≈ 1.62 N·m at 6 V |
| Rated torque | 5 kg·cm ≈ 0.49 N·m at 7.4 V |
| Stall current | 2.5 A at 7.4 V |
| No-load speed | 0.192 s/60° (52 rpm ≈ **5.45 rad/s**) at 7.4 V |
| Gear ratio | 1/345 |
| Backlash | ≤ 0.5° |
| Terminal resistance | 2.5 Ω |
| Torque constant | 8 kg·cm/A ≈ 0.785 N·m/A |
| Encoder | 12-bit magnetic, 0.088° (4096 steps per 360°) |
| Bus | half-duplex TTL serial, 38.4 kbps to 1 Mbps, factory default 1 Mbps |
| Internal position update rate | 1 ms |
| Overload protection | above 80% of stall for 2 s |
| Over-temperature protection | above 70 °C |

Conflicts and context:
- **Feetech's own pages disagree.** The product page and the catalog listing give somewhat different torque/voltage pairs. We treat the PDF as primary.
- **Supply voltage matters.** The SO-ARM100 README notes that the stock kit powers these 7.4 V servos from a **5 V** supply [SOARM, `README.md`]. Measure what Le Snail actually supplies under load, because stall torque and speed scale with voltage.
- **Follower gearing.** The SO-101 follower uses 6 × 1/345 servos. The mixed 1/147 and 1/191 gearing is on the leader arm only [SOARM, `README.md`; LeRobot `so101.md`].
- **Encoder placement unknown.** The datasheet does not say whether the encoder reads the output shaft. Horn, bracket and gear play may therefore be partly invisible to `Present_Position` (unverified).
- **Register units are from a reseller.** Units come from the Waveshare ST3215 memory map, a secondary source [WS-map]:
  - `Present_Load` (address 60) is the **PWM duty cycle in 0.1% units, not torque**.
  - `Present_Current` (69) is in 6.5 mA units.
  - `Acceleration` (41) is in 100 step/s² units.
  - Speed is in step/s, with 3400 step/s ≈ 5.2 rad/s at 7.4 V.

### What LeRobot does on connect

LeRobot `main` at commit `ae2b57f12223c8c5184fcd814eb63cd9ce7de203` [LR]:
- **On connect.** `SOFollower.connect()` → `configure()` [LR, `src/lerobot/robots/so_follower/so_follower.py:114-132,184-196`]. Inside `configure()`:
  - It calls `bus.configure_motors()`, which writes `Return_Delay_Time=0`, `Maximum_Acceleration=254` and `Acceleration=254` (≈ 39 rad/s², our conversion) [LR, `src/lerobot/motors/feetech/feetech.py:209-225`].
  - It then sets `Operating_Mode` to position on every motor, with `P_Coefficient=16`, `I_Coefficient=0` and `D_Coefficient=32` (`config_so_follower.py:45-47`).
  - On the gripper only, it writes `Max_Torque_Limit=500`, `Protection_Current=250` and `Overload_Torque=25`.
- **Observations.** `get_observation()` does a single `sync_read("Present_Position")` [LR, `so_follower.py:204-211`]. `send_action()` does a `sync_write("Goal_Position")` (244-255).
- **Reading other registers.** Any register can be sync-read by name, one register per transaction [LR, `motors_bus.py:1128-1170`]. The private `_sync_read(addr, length, ids)` can read a contiguous block such as position plus speed in one transaction (1173-1205). Sign-magnitude decoding is applied to position, velocity and load (`tables.py:206-216`).
- **Upcoming change.** Open PR #4672 would replace the bus implementation, so pin the commit.

### Bus rate

- **Measured.** A community measurement on a real SO-101 found `sync_read` of positions takes about **1.0 ms mean** (Jetson Orin NX) [LR-4672]. A lost reply costs about 50 ms (the SDK timeout) [LR-4672].
- **Our wire-time estimate** for 6 servos at 1 Mbps, 10 bits per byte, return delay 0:

  | Transaction | Wire time |
  |---|---|
  | Position-only sync read | ≈ 0.62 ms |
  | Contiguous block, addresses 56–70 (position, speed, load, voltage, temperature, …, current) | ≈ 1.4 ms |
  | Goal sync write | ≈ 0.26 ms |

  USB-serial turnaround (1 ms USB frames) dominates. Expect **a few hundred Hz** for one block read plus one write in Python. By the same estimate, SAGE's 18 single-register reads plus 6 writes per cycle would cost on the order of 20+ ms.

### Model structure and implied parameters (BAM)

BAM models the 7.4 V STS3215 like this [BAM, `bam/feetech/actuator.py:22-175`]:
- **Rate-limited internal target.** The firmware rate-limits its internal target position to `max_velocity`, by default 3400 step/s ≈ 5.2 rad/s.
- **P-only controller.** It runs a P controller on that internal target: duty = clip((q_target_smooth − q)·kp·0.166·ratio, ±0.97), and voltage = V_in × duty.
- **Motor torque:** τ = (kt/R)·V − (kt²/R)·q̇ − friction.

Caveats:
- **The STS3215 parameters are repo-only.** The BAM paper itself identifies Dynamixel MX-64/106 and eRob actuators, not the STS3215 [BAM-paper]. There is no published accuracy figure for the STS3215 fits.
- **Different register settings.** The recordings set the firmware **D coefficient to 0** and swept P over 4/8/16/32 (`bam/feetech/record.py`, `all_record.py`). LeRobot runs P = 16 with **D = 32**, and BAM's model has no D term.
- **Supply voltage of the fit is unverified.** The scripts default to 5.1 V or 15 V; the per-log `vin` would settle it.

**Gains implied by the BAM STS3215 M4 parameters** [BAM, `bam/params/feetech_sts3215_7_4V/m4.json`]. These are our derivations, assuming V_in = 7.4 V. They are not measurements of our servos.

| Quantity | Estimate | Le Snail provisional value |
|---|---|---|
| Position stiffness at P = 16 | ≈ 12 N·m/rad (M4; ≈ 9 N·m/rad via `to_mujoco` with M1) | 12 N·m/rad |
| Back-EMF damping kt²/R | ≈ 0.65 N·m·s/rad, plus viscous friction 0.03–0.06 | 0.5 N·m·s/rad |
| Max torque at 97% duty | ≈ 3.5–3.7 N·m, **above the 1.91 N·m datasheet stall torque** | 0.8 N·m effort limit |
| Internal-target speed limit | ≈ 5.1–5.3 rad/s, consistent with the datasheet 5.45 rad/s | 2 rad/s `velocity_limit_sim` |
| Command delay | ≈ 1 ms (M1) to 5 ms (M3–M6) | one-step (50 ms) action-delay option |
| Armature | ≈ 0.023–0.026 kg·m² (a pendulum-bench fit that may absorb other unmodeled effects; compare against link inertias, especially the wrist, before copying it into Isaac Lab) | — |
| Duty saturation onset at P = 16 | position error ≈ 0.3 rad | — |

Reading the table:
- Our provisional 0.8 N·m sits between the datasheet rated torque (0.49 N·m) and stall torque (1.91 N·m).
- Our 2 rad/s simulated velocity limit is well below the servo's roughly 5.4 rad/s.
- The 1 rad/s target slew limit is different: it is a command filter in the policy-to-hardware contract (`src/le_snail/contract.py:87`), applied on both the sim and the robot side. It is not an actuator parameter, so it is not a mismatch.
- All of these values must be measured on our own arm, at our own supply voltage and register settings.

### Existing SO-101 simulation parameters disagree wildly

- **SO-ARM100 repo, two MuJoCo `sts3215` classes** [SOARM, `Simulation/SO101/`]:
  - `joints_properties.xml`: kp 17.8, damping 0.60, frictionloss 0.052, armature 0.028, ±3.35 N·m. The repo's `Simulation/SO101/README.md` says it was adapted from Open Duck Mini, i.e. BAM-derived.
  - `so101_new_calib.xml`: kp 998.22, kv 2.731. A comment says it was calculated from a 12 V servo, assuming P = 16. That is about 100× stiffer than the first class.
  - Our URDF from the same repo carries only joint limits (effort 10, velocity 10), not gains.
- **MuJoCo Menagerie `trs_so_arm100`:** kp 50, frictionloss 0.1, armature 0.1, ±3.5 N·m, with no derivation given [MENAG].
- **Community Isaac Lab ports:**
  - `isaac_so_arm101`: hand-tuned implicit stiffness 50–200, effort 1.9 [ISO101].
  - LeIsaac: 17.8 / 0.60, the BAM / Open Duck numbers [LEISAAC].
  - The NVIDIA workshop: stiffness 4–55, effort 30 (above).

**None of these values is measured on a LeRobot-configured SO-101, which is why Le Snail should identify its own.**

### Modeling options for a servo we cannot torque-command
1. **Position-domain model (recommended form for any learned model).**
   - Learn (or identify) the map from the history of commanded goal and measured position/velocity to the next measured position/velocity. Equivalently, learn a corrected target that the sim PD should track. All signals are directly measurable on the STS3215.
   - In Isaac Lab, implement it as a custom `ImplicitActuator` subclass that rewrites `joint_positions` every physics step (Q2, note 4).
   - GapONet and ASAP are learned variants of this idea: delta actions trained so sim tracks real [GAPO; ASAP25].
2. **Physics servo model (BAM-style) as an explicit actuator.**
   - Port BAM's equations into an `ActuatorBase` subclass with `is_implicit_model=False`: a rate-limited internal target, then P-control to duty to voltage, then the DC-motor torque minus friction.
   - Set `armature` from the fit, and check that the 200 Hz explicit integration is stable. BAM's own simulations use MuJoCo, and it ships a vectorized `bam/mjlab.py` wrapper that can serve as a template.
   - This yields torque without needing torque sensing, because torque is inferred through the motor model.
3. **Torque-output actuator net (Hwangbo).**
   - Only possible with torque labels. `Present_Current` and `Present_Load` are weak proxies (SAGE converts current to torque with a 12 V stall-current constant), and a load cell or pendulum bench would be needed for ground truth.
   - Not recommended as the first step.

### Prior SO-100/SO-101 sim-to-real material
- **NVIDIA course and SAGE** (above).
- **The workshop's hand-set implicit gains** (above).
- **Open Duck Mini** (STS3215 servos) used BAM to identify its servos, then converted the parameters to MuJoCo damping, kp, frictionloss, armature and forcerange ("We use BAM") [ODM, `docs/sim2real.md`].
- **Squint** (an SO-101 visual RL sim-to-real paper, arXiv 2602.21203) reports no actuator model [Squint]. We found no peer-reviewed paper that identifies SO-101 joint dynamics specifically.

## 5. Alternatives and complements

| Approach | Primary evidence | Fit for Le Snail |
|---|---|---|
| **Classical system identification, then domain randomization** | **Tan 2018 (Minitaur)** [Tan18]: identified a DC-motor model plus *measured* latency (15–19 ms policy, 3 ms motor PD), then randomized latency 0–40 ms, control step 3–20 ms and motor strength 80–120%. Result: "both accurate actuator model and latency simulation are important. Without either of them, the learned controllers do not work on the real robot."<br><br>**Haarnoja 2024 (OP3, Dynamixel position servos)** [Haar24]: fitted damping, armature, friction, max torque and Kp to sinusoid-driven trajectories of a loaded servo. Randomization was deliberately kept small, including 10–50 ms observation delay. "Zero-shot transfer did not work" without randomization and perturbations. Their limitations section admits the servos were approximated as ideal position-feedback torque sources, which left behaviour sensitive to battery charge.<br><br>**Lin 2025 "autotune"** [Lin25]: a parallel-sim population search over gains, friction, mass and joint limits, scored against one set of real calibration trajectories, took under 4 min. Real grasp success was 8/10 with the best-fit parameters versus 0/10 with the worst. | **Best first step.** It matches our hardware class (position servos, low policy rate) and our data budget. |
| **Servo-specific physics model (BAM)** | [BAM-paper; BAM]: models a voltage-controlled servo, U = clip(PID, ±Umax) and τ = (kt/R)·U − (kt²/R)·q̇, plus Stribeck and load-dependent friction. It is identified on a pendulum bench with CMA-ES. On Dynamixel MX-64/106 it cut position MAE by 2–3× versus Coulomb-viscous friction. **STS3215 7.4 V parameters ship in the repo**, though with no published accuracy and recorded with D = 0. The repo also has MuJoCo / mjlab actuator wrappers. Open Duck Mini (STS3215) used it [ODM]. | **Strong candidate for stage 2 or 3.** It is physically grounded and already identified for our servo model. It would need porting to an Isaac Lab actuator class. Our PID registers must be re-checked against BAM's assumptions (see Q4). |
| **DC-motor torque-speed clip** | Isaac Lab `DCMotor` [IL-pd, 203-307]. The STS3215 is internally a voltage-controlled DC motor, so its stall torque and no-load speed bound the reachable torque and velocity. | A cheap upgrade from a constant `effort_limit` to a velocity-dependent one. But `DCMotor` is explicit, so it only helps if we move to an explicit or custom actuator. A custom implicit-actuator class can apply the same clip through `effort_limit_sim` only as a constant. |
| **Delay / latency modeling** | Tan18 and Haar24 (above). **Peng 2018** [Peng18]: randomizing the action timestep was critical; fixing it dropped real success from 0.89 to 0.29. Isaac Lab `DelayedPDActuator` delays by an integer number of physics steps, re-drawn at each reset [IL-pd, 310-367]. **Sandha 2020** [Sandha20]: put the delay into the observation for stochastic delays. | **Required.** Our one-step (50 ms) action-delay option is coarse. Measure the real command-to-motion latency and randomize at 5 ms resolution around it. |
| **Random force/torque perturbations** | legged_gym pushes [Rudin22]. Haar24 used 5–15 Nm impulses every 1–3 s. Disney BDX applied random body wrenches [Grandia24]. | A complement for robustness, especially for side-fall recovery. It does not replace actuator fidelity. |
| **Learned delta action / real-to-sim** | **ASAP** [ASAP25]: a PPO delta-action model trained so sim matches real, then frozen while the policy is fine-tuned. It needed more than 400 real clips for 23 DoF, and two robots were damaged collecting data. **GAT** [GAT17] grounds the simulator's actions with a learned real forward model. **SimOpt** [SimOpt19] adapts the randomization distribution from a few real rollouts. **GapONet** [GAPO] is ASAP-like, with a DeepONet. | Powerful but data- and engineering-heavy. It only pays off once a parametric model is shown to be insufficient. Floor-contact data would be needed for Le Snail. |
| **Actuator network (torque)** | [H19; Rudin22] | Requires torque labels, which we do not have, plus an explicit actuator at 200 Hz. The position-output variant (Q4) is the relevant form. |
| **Low-cost designs that avoid actuator nets** | Berkeley Humanoid [Liao24] designed its hardware so a simple joint model with armature and friction suffices, and called "motor strength" randomization a "lazy approach". Disney BDX [Grandia24] used identified first-principles models with backlash, friction and a velocity-dependent torque limit. | Supports a parametric model plus targeted randomization over a black-box network. |

**What fits Le Snail best:**
- Start with a measured delay, plus the servo's rate and acceleration limiting, plus an identified PD/effort/friction/armature model, with modest randomization around the identified values.
- Then compare BAM's STS3215 M4 model (ported) against that baseline.

Reasons:
1. The servo hides torque, so a torque network cannot be trained directly.
2. Our 20 Hz policy (50 ms) is slow compared with the servo's dynamics; the BAM fit has a command delay of about 5 ms. Most of the gap should therefore show up as latency, rate limits, torque saturation under load, and friction or backlash. Those are parameters, not black-box dynamics.
3. Literature on both position-servo robots (OP3, Minitaur, Disney) and low-cost designs transferred with parametric models.
4. It needs minutes, not hours, of real data, and it extrapolates more safely to floor-contact loads than a network fitted on a suspended arm.

## 6. Staged study plan for Le Snail

### Stage 0: freeze the servo configuration (½ day)
- Connect through the **same code path as deployment**: `SO101Follower.connect()` → `configure()`, at the pinned LeRobot commit. Then dump every RAM/EEPROM register on every servo to JSON and store it with each log.
- SAGE's collector bypasses `configure()` (see above), so its register state can differ from deployment.
- At commit `ae2b57f`, `configure()` writes the following [LR]:
  - P/I/D = 16/0/32
  - `Acceleration` = `Maximum_Acceleration` = 254
  - `Return_Delay_Time` = 0
  - gripper only: torque limit 500, protection current 250, overload torque 25
- Re-dump after any LeRobot upgrade (PR #4672 rewrites the bus layer). If `configure()` changes P/D/acceleration, all identified parameters are invalid.
- BAM's STS3215 fits were recorded with D = 0 [BAM]. Identify under LeRobot's settings, not BAM's.
- Record supply voltage under load, servo temperatures, the LeRobot commit, and the USB adapter model.

### Stage 1: measure the gap (SAGE-style, 1–2 days)
**(a) Data.**
- **Rig:** clamp the base link to a bench (fixed root, matching the sim `fix_root` case). First suspend the arm so that no link touches anything.
- **Motions:**
  - Start from SAGE's SO-101 motion files: `step_response`, `chirp_sweep`, `multisine`, `prbs_excitation`, `velocity_ramp`, `backlash_detection`, `friction_gravity`, `hold_under_gravity` and `random_waypoints` [SAGE].
  - Rename the joints `Rotation..Jaw` → `shoulder_pan..gripper`, map them through our measured LeRobot↔URDF mapping, and clip to URDF limits minus a margin.
  - Add Le Snail-specific sequences: replays of 20 Hz command sequences from sim crawling policies, and goal steps of 2–30° at several speeds.
- **Command timing:** commands go out at 20 Hz (the deployment rate) and also at 50 Hz (for bandwidth).
- **Logger:** write our own small logger instead of SAGE's per-motor loop.
  - Use one `sync_write` of `Goal_Position`, then one `_sync_read` of the contiguous block at addresses 56–70 (position, speed, load, voltage, temperature, current) per cycle.
  - Run at the fastest rate the bus sustains: an estimated 1.4 ms of wire time per read plus USB turnaround, so a target of 100–300 Hz (see Q4).
  - Count and log lost replies, which cost about 50 ms each [LR-4672].
  - Timestamp with `time.monotonic_ns()` immediately before and after each bus transaction. Store both, and log the command send time separately.
  - Log raw register values; convert offline.
- **Safety:**
  - Limit joint ranges in software.
  - Ramp to the start pose, as SAGE does with a 3 s interpolation [SAGE collector, lines 362-383].
  - Abort on `Present_Load` (PWM duty)/current above a threshold, temperature above about 60 °C, or tracking error above a threshold. The servo's own protections trip at 80% of stall for 2 s and at 70 °C [FT-DS].
  - Keep a hand on the physical power cutoff.
  - Limit each run to a few minutes and let the servos cool. Do not run the gripper against an object.
- **Sim twin:** replay the *logged post-filter commands that actually went on the bus, with their real timestamps*, in our own Isaac Lab env. Switch off the env's own slew filter and action delay for this replay, so the real latency is not counted twice.
  - Use the same URDF, the same `ImplicitActuatorCfg`, 200 Hz physics, a zero-order hold at the real send times, and a fixed root.
  - Do not use SAGE's `run_simulation.py`: it loads a different USD with different joint names, through the Isaac Sim core API.
  - Log sim joint position, velocity and applied torque at 200 Hz.

**(b) Metrics.**
- Use SAGE's metrics per joint: RMSE, MAPE, correlation, cosine similarity and cross-correlation lag [SAGE analysis, 608-679].
- Add step-response metrics: dead time, 10–90% rise time, overshoot, steady-state error under gravity, and peak velocity. Also add the frequency response (gain and phase) from the chirp and multisine runs.
- Compute lag from *corrected* timestamps. SAGE stamps state rows at command time, which biases `max_lag`.
- This baseline tells us whether the gap is mostly lag, mostly amplitude/saturation, or mostly friction/backlash.

### Stage 2: parametric identification (2–3 days)
- **Fit, per joint group** (shoulder_lift/elbow loaded by gravity, wrist, gripper):
  - pure delay (in 5 ms steps);
  - target slew rate and acceleration limit (the servo firmware rate-limits its internal target [BAM]; LeRobot's `Acceleration` register setting matters);
  - stiffness and damping, or an equivalent servo P/D;
  - effort limit and a velocity-dependent torque limit;
  - joint friction (static/dynamic/viscous are efforts in Isaac Sim 5);
  - armature.
- **Method:** a batched sim search (CMA-ES or population search, as in Lin 2025's autotune [Lin25]) over parallel Isaac Lab envs, minimizing position and velocity error on the training trajectories.
- **Held-out validation:** whole held-out motion types (for example, train on chirp/PRBS/steps and validate on `random_waypoints` and policy replays), not random time-splits.
- **Pass criterion:** held-out position RMSE and lag substantially below the Stage 1 baseline, with no systematic residual pattern.
- **Optional:** port BAM's STS3215 M4/M6 model [BAM] as a candidate and compare it on the same held-out set.
- **What Stage 2 delivers (6c), one of two paths:**
  - **Implicit path (default).** An `ImplicitActuator` subclass that applies the identified delay (in 5 ms steps), target rate limit and acceleration limit to `joint_positions`. PhysX keeps the identified stiffness, damping, armature and friction, and a *constant* `effort_limit_sim`. Gain DR can stay on `write_joint_*_to_sim`. This path cannot represent a velocity-dependent torque limit.
  - **Explicit path (only if loaded or fast motion needs it).** `DCMotor`-style velocity-dependent torque limits, or a full BAM port, as an explicit actuator. This means explicit PD at 5 ms, a stability check, armature tuning, and moving gain DR into the actuator's tensors (Q2).

### Stage 3 (only if Stage 2 residuals are structured): learned model
- **Target:** predict next-step joint position and velocity (or a position correction) from a history of commanded goal, measured position and velocity.
- **History:** at least the identified delay plus about 50–100 ms, sampled at the bus rate.
- **Network:** a small MLP like [H19] (for example 3×32, softsign) or a GRU.
- **Data:** 15–60 minutes of varied data, more if load conditions vary. Hwangbo needed less than 4 min, but with 12 parallel actuators and torque labels [H19].
- **Integration:** a custom subclass of `ImplicitActuator` whose `compute()` keeps a 200 Hz history and outputs an adjusted `joint_positions` target. This keeps PhysX's stable PD. Use `ActuatorNetMLP` only if we choose a torque-output model trained against sim-inferred torques, and accept the explicit-PD stability issues.
- **Rate mismatch:** the bus samples at roughly 100–300 Hz (irregularly) while the actuator runs at 200 Hz. Either resample the logs to 200 Hz (interpolating measured states and zero-order-holding commands) and train the network at 5 ms steps, or train at the bus period and pick `input_idx`-style slots spaced to match. The model must be invariant to the policy holding a target for 10 physics steps.
- **Gain randomization:** move it from `write_joint_stiffness_to_sim` into the actuator's tensors (see Q2).
- **No double counting.** A custom actuator's delay and rate stage *replaces* the env's current one-step action delay (`src/le_snail/sim/env.py:152-160,312`) rather than stacking on top of it. Keep the contract slew filter, because it also runs on the robot.
- **Alternative:** a GapONet/ASAP-style delta-action model, trained by RL to make our sim reproduce real logs. This means writing a new SO-101 task on top of the GapONet code, which targets Isaac Sim 4.5. Treat it as a research spike, not the main path.

### Stage 4: load-bearing validation on the floor (essential)
- A fixed-root, suspended-arm gap does **not** cover Le Snail's operating regime. During crawling the base and gripper are ground supports, and shoulder_lift, elbow_flex and wrist_flex carry body weight and contact impulses.
- **Logging:** log the same registers, plus the IMU, while replaying identical 20 Hz command sequences on the floor (low-friction and high-friction surfaces) and in sim with a floating base.
- **Compare:**
  - joint tracking and lag under load;
  - `Present_Load`/current saturation;
  - base displacement and orientation per gait cycle from the IMU and a tape or video measurement.
- **If loaded tracking differs from suspended tracking:** refit the effort limit and torque-speed limit using the loaded data. Load-dependent friction (BAM M3+) is one candidate explanation.

### Did it improve transfer? (decision gates)
1. **Open-loop replay:** sim-vs-real joint RMSE and lag on held-out sequences, before versus after the model. SAGE metrics are enough here.
2. **Closed-loop policy:** train two policies with identical seeds and rewards, (A) the current provisional actuator plus DR and (B) the identified model plus DR around it. Deploy both on the real robot for N≥5 trials each.
   - Compare forward distance in 120 s (target 50 cm), success rate, joint-tracking error, servo temperature rise, and side-fall recovery rate.
   - Also compare sim-predicted against real distance per trial. A smaller sim-real distance gap is the direct measure.
3. **Ablations:** delay only; delay plus rate limit; full parametric; learned. This tells us which part mattered.

### Risks
- **Bus read rate.**
  - Sequential per-motor reads (as in SAGE) cost a round-trip each, roughly 1 ms or more. Use sync read of contiguous blocks; a measured positions-only `sync_read` takes about 1 ms [LR-4672]. Measure the actual loop period and jitter, and log it.
  - Any read rate below about 100 Hz limits identification of dynamics above roughly 20–50 Hz. That is acceptable for a 20 Hz policy.
- **Gear backlash.**
  - The course notes hobby servos "can introduce significant backlash" that accumulates along the chain [C15].
  - The datasheet rates gear backlash at ≤ 0.5° per servo [FT-DS]. Whether the encoder sees that play is not documented. Horn and bracket play is invisible to `Present_Position` either way.
  - Measure it with SAGE's `backlash_detection` motion plus an external reference (camera or IMU on the link). Model it as a deadband, or randomize it.
- **Load-dependent behaviour.** Torque saturation and friction change with gravity and contact load, which is why the floor stage exists.
- **Temperature and voltage.** Torque falls as the motor heats and the supply sags; Haar24 saw battery-level sensitivity [Haar24].
  - Log `Present_Temperature` and voltage.
  - Identify cold and warm conditions separately.
  - Randomize the effort limit by about ±15% around the identified value.
- **LeRobot overwriting servo PID registers.**
  - `configure()` rewrites P/I/D and acceleration on every connect.
  - Pin the LeRobot commit, dump registers on every run, and refuse to run if they differ from the identified set.
  - Never identify with one register set and deploy with another. SAGE writes P=16/I=0/D=32 but not acceleration [SAGE collector, 216-227].
- **Is a simple delay plus PD identification sufficient at 20 Hz?**
  - Probably, for the arm joints.
  - A 50 ms policy period is about 10 times the BAM-identified servo command delay (about 5 ms) [BAM]. Between policy steps the servo has largely converged unless it is torque- or speed-limited.
  - The dominant effects are therefore saturation (torque, speed, acceleration), end-to-end latency (USB, inference, bus), and friction or backlash. All of these are parametric.
  - The acceleration limit is worth checking. `Acceleration=254` × 100 step/s² ≈ 39 rad/s² (our derivation from the reseller unit [WS-map]). From rest, a 0.05 rad move under that profile takes about 50 ms, which is one full policy period. The firmware's acceleration profile may therefore be a first-order effect at 20 Hz. Include it in the Stage 2 fit.
  - Decide with the Stage 1 data: if a delay plus PD plus limits model reaches held-out RMSE near the encoder and backlash floor, stop there.
- **Version drift.** SAGE targets Isaac Lab 2.2 / Sim 5.0 and GapONet targets Sim 4.5, versus our Isaac Lab 2.3.2 / Sim 5.1. Joint friction changed meaning between Sim 4.5 and 5.x. Keep all identification inside our pinned stack.

## Stage 0 and Stage 1 dry-run results (2026-10-16)

Read-only measurements on IDs 1–5 (the gripper servo, ID 6, was unplugged). Adapter: the ESP32-S3 CH343 USB-serial bridge used as a pass-through to the Waveshare bus board at 1 Mbps, with the ESP32 held in reset. Tools (no LeRobot needed; framing and register table from LeRobot `ae2b57f`):
- `scripts/servo_dump.py PORT`: PING + READ of addresses 0–70 and 80–86 to `runs/actuator-id/stage0/registers-<utc>.json`, with the registers LeRobot `configure()` would change.
- `scripts/actuator_log.py PORT`: one `sync_read` of the 56–70 block per cycle, `monotonic_ns` before and after each transaction, raw bytes to `.npz`. Dry run by default. `--motion FILE --allow-motion` writes goals (operator approval per run).
- `scripts/actuator_motions.py`: one-joint step and chirp offset files for the logger.

**Registers (`runs/actuator-id/stage0/registers-20261016T170902Z.json`).** All five: model 777, firmware 3.10, 12.0–12.1 V, 33–34 °C.

| Register | Value | Note |
|---|---|---|
| P / I / D | 16 / 0 / 32 | already LeRobot's values |
| `Maximum_Acceleration` (85) | 254 | already LeRobot's value |
| `Return_Delay_Time` | 0 | already LeRobot's value |
| `Operating_Mode` | 0 (position) | |
| `Phase` | 0x0C | bit 4 clear, so `configure()` leaves it |
| **`Acceleration` (41, RAM)** | **0** | `configure()` writes 254. RAM resets at power-up, so the servo runs with a different acceleration setting until LeRobot connects. Identify after that write. |
| `Max_Torque_Limit` / `Torque_Limit` | 1000 / 1000 | |
| `Protection_Current` / `Overload_Torque` | 310 / 80 | |
| `Protective_Torque` / `Protection_Time` | 20 / 200 | |
| `Maximum_Velocity_Limit` (84) | 65 | × `Velocity_Unit_factor` 50 = 3250 step/s ≈ 5.0 rad/s, if the unit is step/s (unverified) |
| `Min_` / `Max_Position_Limit` | 806–3520, 947–3303, 789–3010, 1493–3254, 0–4095 | calibration-set; the logger's software window is these minus a margin |
| **`Goal_Position`** | **0** while torque is off | never enable torque before writing goal = present position |

**Bus rate.** 60 s unpaced dry run: **488 Hz**, 29,299 cycles, **0 lost replies**. Transaction (request to last reply) p50 1.94 ms, p99 2.92 ms, max 4.26 ms; period p50 2.00 ms, p99 3.09 ms. That is about 0.5 ms over the 1.4 ms wire estimate, so a goal write at 50 Hz plus a block read stays well above the 100–300 Hz target.

## Not verified

- **Which NVIDIA page the user had in mind.** It is assumed to be Strategy 4 (SAGE + GapONet) of the confirmed course.
- **GapONet internals.**
  - Paper details beyond the NeurIPS abstract.
  - The "8 hours of real SO-101 data" claim, which appears only on the course page; no SO-101 code or data is public.
  - The semantics of the branch "sensor" sub-environments.
- **SAGE's SO-101 USD.** Its joint axes and zero poses relative to the TheRobotStudio URDF; the binary USD was not inspected.
- **ANYdrive LSTM net.** The provenance and training data of `anydrive_3_lstm_jit.pt`.
- **Hwangbo et al. 2019.** The simulation timestep, and whether the actuator net ran every simulation substep.
- **BAM STS3215 fits.**
  - The supply voltage, load masses and accuracy of the STS3215 parameter fits.
  - Whether the firmware D term (LeRobot sets D = 32) changes behaviour relative to BAM's D = 0 recordings.
  - BAM's max-torque estimate (≈ 3.5 N·m) conflicts with the datasheet stall torque (1.91 N·m).
- **STS3215 register details.**
  - Register units come from the Waveshare reseller map, because Feetech's official memory table was not retrieved.
  - Whether the encoder reads the output shaft.
  - The default `Return_Delay_Time` and the maximum `Acceleration` value (secondary sources disagree).
- ~~**Bus timing.**~~ Measured on the Le Snail laptop on 2026-10-16; see Stage 0 and Stage 1 dry-run results.
- **Acceleration effect.** The derived 39 rad/s² acceleration limit, and its effect at 20 Hz.
- **Implicit-actuator subclass.** That a custom `ImplicitActuator` subclass rewriting `joint_positions` behaves as described. This is inferred from the source and not run.

## Sources

Code was read at the commits listed. Line numbers refer to those commits.

**NVIDIA course and NVIDIA material**
- [C-index] *Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac*, course index. https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/
- [C06] Get the code (versions). https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/06-get-the-code.html
- [C07] Calibrating SO-101. https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/07-calibrating-so101.html
- [C09] Strategy 1, domain randomization + teleop. https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html
- [C13] Strategy 2, co-training. https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/13-strategy2-cotraining.html
- [C14] Strategy 3, Cosmos. https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/14-strategy3-cosmos.html
- [C15] Strategy 4, SAGE + GapONet. https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/15-strategy4-sage.html
- [WS] isaac-sim/Sim-to-Real-SO-101-Workshop, commit `ce807d99724cb65671abec01f908a2fcb4a6eab7`. https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop
- [SAGE] isaac-sim2real/sage, commit `5b31cfa35416d1300b5099557049bd8e9cb4507d`. https://github.com/isaac-sim2real/sage (see `docs/LEROBOT_REAL.md`, `sage/real_so101/so101_lerobot_collector.py`, `sage/analysis.py`, `sage/simulation.py`, `sage/assets.py`)
- [GAPO] jiemingcui/gaponet, commit `91eb8819c3156b46fdb3266a048600c67c9adf5d`. https://github.com/jiemingcui/gaponet
- [GAPO-paper] Cui et al., "Bridging the Sim-to-Real Gap in Humanoid Dynamics via Learned Nonlinear Operators", NeurIPS 2025. https://neurips.cc/virtual/2025/124008, https://openreview.net/forum?id=Zi1oawm4cA (abstract only read)
- [NV-AM] NVIDIA, *Getting Started With Isaac Lab*, "Actuator Modeling". https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/transferring-robot-learning-policies-from-simulation-to-reality/04-bridging-the-gap-real-world-data-integration/01-actuator-modeling.html
- [NV-NeRD] NVIDIA Technical Blog, "Advancing Robotics Development with Neural Dynamics in Newton", 2025-09-29. https://developer.nvidia.com/blog/advancing-robotics-development-with-neural-dynamics-in-newton/

**Isaac Lab** (tag `v2.3.2`, commit `37ddf626871758333d6ed89cf64ad702aef127d0`, https://github.com/isaac-sim/IsaacLab/tree/v2.3.2)
- [IL-net] `source/isaaclab/isaaclab/actuators/actuator_net.py`, `actuator_net_cfg.py`
- [IL-pd] `source/isaaclab/isaaclab/actuators/actuator_pd.py`, `actuator_pd_cfg.py`
- [IL-basecfg] `source/isaaclab/isaaclab/actuators/actuator_base_cfg.py`, `actuator_base.py`
- [IL] `source/isaaclab/isaaclab/assets/articulation/articulation.py`, `source/isaaclab/isaaclab/envs/direct_rl_env.py`, `source/isaaclab/isaaclab/utils/assets.py`
- [IL-assets] `source/isaaclab_assets/isaaclab_assets/robots/anymal.py`, `unitree.py`
- [IL-api] Isaac Lab API reference, `isaaclab.actuators`. https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html

**Papers**
- [H19] Hwangbo et al., "Learning agile and dynamic motor skills for legged robots", *Science Robotics* 4(26) eaau5872, 2019. https://doi.org/10.1126/scirobotics.aau5872, https://arxiv.org/abs/1901.08652
- [Lee20] Lee et al., "Learning quadrupedal locomotion over challenging terrain", *Science Robotics* 5(47) eabc5986, 2020. https://arxiv.org/abs/2010.11251
- [Rudin22] Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL", CoRL 2021. https://arxiv.org/abs/2109.11978
- [Tan18] Tan et al., "Sim-to-Real: Learning Agile Locomotion For Quadruped Robots", RSS 2018. https://arxiv.org/abs/1804.10332
- [Peng18] Peng et al., "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization", ICRA 2018. https://arxiv.org/abs/1710.06537
- [Haar24] Haarnoja et al., "Learning Agile Soccer Skills for a Bipedal Robot with Deep RL", *Science Robotics* 2024. https://arxiv.org/abs/2304.13653
- [Lin25] Lin et al., "Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids", 2025. https://arxiv.org/abs/2502.20396
- [ASAP25] He et al., "ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills", 2025. https://arxiv.org/abs/2502.01143
- [GAT17] Hanna and Stone, "Grounded Action Transformation for Robot Learning in Simulation", AAAI 2017. https://ojs.aaai.org/index.php/AAAI/article/view/11044
- [SimOpt19] Chebotar et al., "Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience", ICRA 2019. https://arxiv.org/abs/1810.05687
- [Sandha20] Sandha et al., "Sim2Real Transfer for Deep RL with Stochastic State Transition Delays", CoRL 2020. https://proceedings.mlr.press/v155/sandha21a.html
- [Liao24] Liao et al., "Berkeley Humanoid: A Research Platform for Learning-based Control", 2024. https://arxiv.org/abs/2407.21781
- [Grandia24] Grandia et al., "Design and Control of a Bipedal Robotic Character", 2024 (arXiv 2025). https://arxiv.org/abs/2501.05204
- [BAM-paper] Duclusaud, Passault et al., "Extended Friction Models for the Physics Simulation of Servo Actuators". https://arxiv.org/abs/2410.08650
- [ODM] apirrone/Open_Duck_Mini, commit `b23317a485b3cec7d8417f352478778b3475173c`, `docs/sim2real.md`. https://github.com/apirrone/Open_Duck_Mini
- [FT-DS] Feetech STS3215 7.4 V (ST-3215-C001) product specification, edition A/0, 2020-04-10. https://www.feetechrc.com/Data/feetechrc/upload/file/20260622/6391772523943436695270694.pdf (product page: https://www.feetechrc.com/74v-19-kgcm-plastic-case-metal-tooth-magnetic-code-double-axis-ttl-series-steering-gear)
- [WS-map] Waveshare, "ST3215 Memory Table Parsing V3.7" (reseller register map; secondary). https://files.waveshare.com/upload/2/27/ST3215%20memory%20register%20map-EN.xls
- [SOARM] TheRobotStudio/SO-ARM100, commit `a758567c3978dfeefe282ede0500085a48fe8f78` (`README.md`, `Simulation/SO101/joints_properties.xml`, `Simulation/SO101/so101_new_calib.xml`). https://github.com/TheRobotStudio/SO-ARM100
- [LR] huggingface/lerobot, commit `ae2b57f12223c8c5184fcd814eb63cd9ce7de203` (`src/lerobot/robots/so_follower/so_follower.py`, `config_so_follower.py`, `src/lerobot/motors/feetech/feetech.py`, `tables.py`, `src/lerobot/motors/motors_bus.py`, `docs/source/so101.md`). https://github.com/huggingface/lerobot
- [LR-4672] lerobot PR #4672 and its benchmark comment (community measurement). https://github.com/huggingface/lerobot/pull/4672#issuecomment-5951759502
- [MENAG] google-deepmind/mujoco_menagerie `trs_so_arm100`, commit `d7b481bac63d3d600ab404ee74f7474c5efe8a67`. https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100
- [ISO101] MuammerBay/isaac_so_arm101, commit `e4624dea075b00a36dbc66bebd531d191c92e8cd`. https://github.com/MuammerBay/isaac_so_arm101
- [LEISAAC] LightwheelAI/leisaac, commit `24d3bcd3f1e4585740fc79921782c41617237812`. https://github.com/LightwheelAI/leisaac
- [Squint] "Squint", SO-101 visual RL sim-to-real. https://arxiv.org/abs/2602.21203
- [BAM] Rhoban/bam, commit `e9a619d56da5236206f4de6ceec2c1ee1b497b5c`. https://github.com/Rhoban/bam (see `bam/feetech/actuator.py`, `bam/params/feetech_sts3215_7_4V/m1..m6.json`, `bam/mjlab.py`)
