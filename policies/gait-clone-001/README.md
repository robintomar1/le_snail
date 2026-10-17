# gait-clone-001

A learned crawl policy that copies the hybrid-001 "D-balanced" rock-and-drag gait (the gait of best_forward_move_1). No get-up: if it falls it stays down.

## How it was made (GPU host, 2026-10-17, #15)

- **Teacher:** `PhaseGait` (`src/le_snail/gait_teacher.py`, commits a9af1bc, f843d8b): the hybrid-001 gait with its phase read from the applied target instead of a clock. A policy has no clock, so the teacher has to label any state from what a policy can see; the closest distinct-phase pair on the slewed cycle is 0.026 rad apart. Played closed loop in a kinematic check it reproduces the open-loop targets within 0.02 rad, and in sim it crawls 5/5 (robust dev 5600–5614, 3.3–4.0 cm/s, final |yaw| 4–8°).
- **Clone:** DAgger (`scripts/clone_gait.py`, commit d42aed3) in Isaac Lab, robust physics `configs/experiments/physics-v2-50hz-001.json`, upright starts, 1024 envs, 6 rounds of 15 s with teacher mix β = 1, 0.5, 0.25, 0, 0, 0, refit 4000 Adam steps per round. Fit MSE 5.6e-5 (action RMS ~0.0075). Export parity passed.
- **Network:** frozen observation normalizer + MLP 24-128-128-6 (ELU), stateless, contract `policy-50hz-001` (24 inputs, 50 Hz, 1 rad/s slew).

## Sim results

- Robust dev 5620–5634 (`sim-evaluation/report.json`, `crawl-0.mp4`): **crawl 5/5**, 50 cm in 15.4–18.0 s (3.2–3.8 cm/s), final yaw −2..+6°, max lateral 0.5–2.5 cm, lowest upright cosine 0.70. Left/right 0/5 (no get-up). No speed flags.
- 3-minute endurance (robust, lane off, 32 envs; `../sim-endurance-002/`): path 6.29 m, net 5.96 m, 0 falls.

## Real arm

19 live runs and 2 shadow runs on 2026-10-17 (`real-arm/RUNS.md`):
- **arm 1 (snail_1), Rubik Pi + Xsens:** 4 runs ran their full 12 or 30 s; 6 were refused by `--require-upright` because the base was already tipped ~45–50° nose-down after the ramp to home (gravity ≈ [0.73, −0.09, −0.68], against the sim crawl start's [0.56, 0, −0.83]); 1 ran its full 30 s and then lost servo replies during the final hold. Heading change over the completed runs ranged from −52° to +110° (12 s runs: +3° to +8°).
- **arm 2 (hackathon arm), Rubik Pi + Xsens:** 2 full 12 s runs, 1 refused as not upright; the others stopped on lost servo replies (steps 390–542) or a missed control deadline (two runs right after a restart, before the cause was clear).
- Distance was not measured.

## Files

| Path | What |
|---|---|
| `exported/` | the bundle the robot runs (`policy.pt` sha256 fc98f82d548f…, `policy.json` contract, `manifest.json`, `sim.json`) |
| `training/student.pt`, `training/dagger.json` | DAgger student weights and the per-round log |
| `sim-evaluation/` | robust dev evaluation report and one crawl video |
| `real-arm/` | every real-arm run log and the summary table |
