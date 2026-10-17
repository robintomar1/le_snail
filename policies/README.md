# Deployed policies

The four learned policies run on the real SO-101 on 2026-10-17, together with the open-loop crawl gait that first moved the real arm. Each folder holds the exported bundle the robot runs, the training checkpoint to continue from, the sim evaluation (report and video), and every real-arm run log with a generated summary.

The scripted gait behind all of this is frozen separately in [`runs/best_forward_move_1/`](../runs/best_forward_move_1/) (tag `best_forward_move_1`).

| Policy | What it is | Rate | Sim crawl (robust dev) | Sim 3-min endurance: path / net | Get-up | Real arm |
|---|---|---|---|---|---|---|
| [`gait-clone-fast-001`](gait-clone-fast-001/) | MLP cloned (DAgger) from a faster phase-locked version of the hybrid-001 gait | 50 Hz | **5/5**, 50 cm in 9.2–11.2 s | **11.25 m / 10.90 m**, 0 falls | no | arm 2: 6 live runs, 4 ran their full time; 2 stopped on lost servo replies (one after 27 s) |
| [`gait-clone-001`](gait-clone-001/) | MLP cloned (DAgger) from the phase-locked hybrid-001 D-balanced gait | 50 Hz | **5/5**, 50 cm in 15.4–18.0 s | 6.29 m / 5.96 m, 0 falls | no | arm 1: 11 live runs (4 full, 6 refused as not upright after the ramp); arm 2: 8 live runs (2 full, 1 refused as not upright, 5 stopped on lost servo replies or deadlines) |
| [`diversity-seed101-051-100`](diversity-seed101-051-100/) | RL (PPO), crawl stage, fresh seed 101, 100 iterations | 20 Hz | 1/5 (lane exits; it circles) | 10.12 m / 3.82 m, 0 falls | no | arm 2: 3 live runs, all ran the full 12 s |
| [`two-portion-hybrid-002-500`](two-portion-hybrid-002-500/) | RL (PPO) with a hybrid-controller teacher, two-portion reward, 500 iterations | 50 Hz | 1/5 (turns out of the lane) | 8.48 m / 0.40 m (tight circles) | **left 5/5, right 5/5** | arm 2 on the Raspberry Pi: 3 live runs, all stopped by the BNO085 IMU; arm 1: 8 live runs, 7 ran the full 12 s |

Path is the distance along the track, net the start-to-end distance: the RL policies keep moving but circle, the clones go roughly straight. Endurance tracks are in [`sim-endurance-002/`](sim-endurance-002/).

[`crawl-gait-open-loop/`](crawl-gait-open-loop/) holds the hybrid-001 crawl as 50 Hz joint targets (the sim export) and as the servo-tick motion files that `scripts/actuator_log.py` played on arm 1 (best_forward_move_1).

**None of these has validated physics.** Every bundle says `physics_validated: false`; real runs need `--allow-unvalidated-physics`, which is an operator decision, with the operator on the 12 V cutoff.

## Running one

On the computer wired to the arm (the Rubik Pi), from the repo root:

```bash
.venv-deploy/bin/python scripts/policy_run.py --hardware configs/hardware/arm2.json \
  --bundle policies/gait-clone-fast-001/exported --mode shadow --seconds 10      # read-only check first
.venv-deploy/bin/python scripts/policy_run.py --hardware configs/hardware/arm2.json \
  --bundle policies/gait-clone-fast-001/exported --mode live --allow-unvalidated-physics \
  --require-upright --seconds 12 --out runs/policy-runs
```

- `--hardware` picks the arm: [`arm1.json`](../configs/hardware/arm1.json) (snail_1) or [`arm2.json`](../configs/hardware/arm2.json) (the hackathon arm, snail_2). Each names its servo port, IMU, joint map and IMU mounting. The runner refuses an arm whose servo homing offsets differ from its joint map.
- `--require-upright` stops before the policy starts if the base is more than ~45° from upright after the ramp to home (crawl tests). Leave it off for get-up tests.
- The runner ramps from the present pose to the contract home at ≤ 0.5 rad/s, holds 1 s, runs the policy at the bundle's rate, then holds until Enter and releases torque. It aborts with torque off on the #12 servo envelope (temperature, voltage, sustained load, current, status, tracking, lost replies), a stale IMU sample, or a missed control deadline.

## What the runner does (sim parity)

`scripts/policy_run.py` drives the servos through the repo's own servo code, not LeRobot. It reproduces the training environment's observation and action path: `[q − home, filtered joint velocity, base-frame gravity, base-frame gyro, (applied target − home)/scale]`, actions clipped to ±1 before scaling, contract limits and slew, a velocity filter updated once per policy step, and the sim reset state at the start (previous target = home, velocity 0). Rebuilding the sim's steps 0–10 of 002-500 with the runner's code reproduced the sim observations within 1e-5 and the targets within 2e-7, and the policy outputs on the Pi match the sim's within 2e-6.

## Hardware notes from these runs

- **Arms.** arm 1 = snail_1 (gripper servo added 2026-10-17, uncalibrated, not driven); arm 2 = the hackathon arm, snail_2, with the LeRobot calibration `configs/calibration/snail_2/so101_follower_new.json`. Joint maps: [`configs/joint-maps/`](../configs/joint-maps/).
- **IMU.** The Xsens MTi on the base back face works well (100 Hz, ~12 ms old at use). The BNO085 on the Raspberry Pi's bit-banged I2C dropped reports and locked up whenever the servos were working, so the 002-500 runs on that setup (11:59–12:06 UTC) aborted on IMU staleness.
- **Rubik Pi CPU.** Unpinned, the 50 Hz loop ran on slow cores and a step took ~9–12 ms; pinned to core 7 (`cpu_affinity` in the arm config) ~2 ms.
- **Lost servo replies.** Several runs stopped on "no fresh complete servo state" or "6 consecutive cycles lost a reply": the bus dropped replies mid-run. Worth checking cabling and supply before longer runs.
- **Real joints lag ~100 ms** behind their targets (Stage 1, #12); the 50 Hz policies trained with at most 20 ms of action delay, the 20 Hz one with up to 50 ms.

`real-arm-runs.json` lists every run with the fields of the per-policy `real-arm/RUNS.md` tables.
