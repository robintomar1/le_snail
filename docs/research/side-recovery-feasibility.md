# Side-recovery feasibility: can the SO-101 snail right itself from side-lying?

Independent check, 2026-10-14. It used no learned policy for righting. The question: from the recorded LEFT and RIGHT settled side poses, can the arm's own six joints bring the base back to the crawling posture (upright cosine >= 0.8 held for 0.5 s) and then crawl 5 cm within 30 s?

## Verdict

| Condition | LEFT | RIGHT |
|---|---|---|
| **Sim, as configured**: 0.8 N·m, 2 rad/s joint cap, 1 rad/s target slew, policy action envelope | **Feasible, demonstrated.** Keyframe set p9 passed 6/6 and p34 5/6 on full acceptance (stable upright, then +5 cm with the T100 crawl policy). | **Feasible, demonstrated.** Keyframe set p48 passed 6/6 and p62 3/6. |
| Sim, 0.49 N·m (servo rated torque), same caps | Feasible: p9 6/6, p34 6/6 | Feasible: p48 6/6, p62 5/6 |
| Sim, 1.9 N·m and 5.45 rad/s (stall torque, no-load speed) | Feasible, and more robust: 31/32 over all candidates | Feasible: 28/32 |
| **Real arm, rated 0.49 N·m** | **Uncertain, leaning feasible.** Torque is not the limit. Contact geometry, friction, cable and open-loop fragility are unvalidated. | **Uncertain, leaning feasible.** Same reasons. |
| Real arm, stall 1.91 N·m (at 7.4 V only) | Torque margin is large, but the same non-torque unknowns apply. Stall torque drops on a 5 V supply. | Same |

Each "6/6" counts 3 nominal-physics plus 3 robust-randomization trials. The 3 nominal trials are near-identical repeats, so 6/6 is effectively 1 nominal plus 3 independent randomized trials. Robust randomization covers friction 0.35–0.75, mass ±10 %, PD gains ±20 %, a 0–1 step action delay and tether drag.

**Bottom line: recovery training is worth doing.** Righting is physically possible in the simulator at the configured 0.8 N·m, and even at 0.49 N·m. It takes about 3–4 s at the policy's 1 rad/s slew, and it stays inside the policy's action envelope, so no contract change is needed.

What keeps the current policy fallen is not torque or joint limits. The successful motion is a narrow, coordinated multi-joint sequence:
- Only about 7–26 % of random ±0.1 rad perturbations of the analytic plan succeed in Isaac. Left is 12–26 %; right is narrower at 7–9 %.
- The unperturbed analytic plan itself failed in Isaac under nominal physics.

So this is a hard exploration problem, not an impossible one.

## The robot as modelled

These facts come from `assets/so101/snail.urdf`, parsed directly. The FK, mass and collision-hull code is archived with the results (see Reproducibility).

- **Masses.** Total 0.632 kg: base 0.147 (line 12), shoulder 0.100 (71), upper arm 0.103 (117), lower arm 0.104 (150), wrist 0.079 (196), gripper 0.087 (229), jaw 0.012 (276). The arm beyond the shoulder is 0.385 kg, heavier than base and shoulder together (0.247 kg).
- **Joints:**
  - shoulder_pan (lines 378–384): axis along base −z, at (0.039, 0, 0.062).
  - shoulder_lift (361–367), elbow_flex (344–350) and wrist_flex (327–333): axes all along base +y.
  - wrist_roll (310–316): axis along the forearm.
  - gripper (293–299): the moving jaw.
- **Policy limits.** Contract limits are in `configs/policy.json` lines 5–6. The policy can reach less than these: `home + clip(a, −1, 1)·action_scale`, clipped to the limits (`src/le_snail/sim/env.py:161-162`, `src/le_snail/contract.py:85-88`). The reachable envelope is:

| Joint | Reachable range (rad) |
|---|---|
| pan | [−1.00, 1.00] |
| lift | [−1.65, 0.60] |
| elbow | [−0.10, 1.60] |
| wrist_flex | [−1.55, 0.70] |
| wrist_roll | [−1.50, 1.50] |
| gripper | [0, 0.95] |

  Every result in this report stays inside this envelope.
- **Base geometry.** The union of the base collision meshes spans x −0.031…0.065 m, y ±0.0555 m and z −0.0024…0.070 m in the base frame: a block about 96 × 111 × 72 mm.
  - Upright crawling pose: the base rests on an edge of its bottom face, pitched about 34° (settled root quaternion, `configs/experiments/recover-t100-001.json` `settled_starts`, line 42 on). The gripper is on the floor. Upright cosine is 0.827, only 0.027 above the 0.8 threshold.
  - Side-lying pose: the base lies on its ±y face (roll about 96°). The arm column (base +z) lies flat along the floor. Supports are base + lower arm + gripper/jaw (left) and base + lower arm + wrist (right).
  - The edge that matters for righting is the base's bottom ±y edge, z = −0.0024, y = ∓0.0555. In practice the body rolls about the line from that edge to the planted jaw tip (see Mechanism).
- **Side-lying joint axes.** In the side pose, the lift, elbow and wrist_flex axes point almost vertically in the world (axis z-component ±0.994). These joints can only sweep the arm across the floor. Only shoulder_pan (axis horizontal, along the arm column) and wrist_roll (horizontal, along world x) act out of the floor plane. This is why naive in-plane shuffling, which is what the trained policy does, can never right the body.
- **Model validation.** The quasi-static model rests the robot on the face of its convex collision hull that is stable under gravity. It reproduces the three recorded settled orientations within 0.1° (upright) and 0.7° (left and right), with the same support links as the inspection report (`runs/inspection-20261014T075415Z/inspection-report.md`). Rendered reference poses: `runs/inspection-20261014T075415Z/visual-final/{left,right,upright}-5s.png`.

## Part 1: quasi-static analysis

**Method:**
1. FK and the collision hull of every link (STL meshes decimated to a 3 mm grid).
2. Quasi-static rolling. For each joint configuration along a path, the body rests on the stable hull face it reaches by rolling from its previous orientation.
3. Self-collision check between part hulls of non-adjacent links. Hulls over-approximate concave parts, so this check is conservative.
4. A linear program over floor contact forces (friction pyramid) gives, for each configuration, the **smallest possible peak joint torque** consistent with static equilibrium. This is a lower bound: above the limit means infeasible; below it does not prove feasibility.
5. CEM search over three keyframes plus a return to home, in the policy envelope and also in the full joint limits.

**Result.** Both sides have collision-free quasi-static righting paths inside the policy envelope. In the model they end on the upright base + gripper face, cosine 0.85, with the same pitch as the settled upright pose and a different yaw. They survive keyframe perturbations: ±0.1 rad gives 66/66 (left) and 64/66 (right); ±0.2 rad gives 62/66 and 46/66.

**Mechanism.** The planted jaw tip is the pivot; the shoulder supplies the righting rotation:
1. Fold the arm so that only the base edge and the jaw tip, plus briefly the upper arm, touch the floor. This needs lift strongly negative, wrist_flex strongly negative, wrist_roll turned toward the floor, and pan about −0.6 (left) or +0.6 (right).
2. Sweep pan across to the opposite sign while driving lift from about −1.6 to about +0.5. The body rolls continuously about the base-edge–jaw line. Upright cosine climbs from about 0.05 to about 1.0 while supported only on base and jaw. A final short tip lands on the base + gripper face.
3. Slew back to home. The robot settles into the crawling posture.

The two sides are mirror images in pan and wrist_roll. This contact sequence is what the quasi-static model shows. Isaac confirmed the outcome (stable upright, then +5 cm), not the contact sequence, because nothing was rendered.

Keyframes that passed in Isaac (rad, order pan, lift, elbow, wrist_flex, wrist_roll, gripper), each followed by home `[0, −0.6, 1.1, −0.5, 0, 0.35]`:

| Plan | K1 | K2 | K3 |
|---|---|---|---|
| LEFT p9 | [−0.634, −1.569, 0.226, −1.095, −0.257, 0.371] | [−0.010, −1.605, 0.456, −1.184, −0.848, 0.950] | [0.455, 0.587, 0.748, −0.376, −1.371, 0.639] |
| LEFT p34 | [−0.648, −1.650, 0.227, −1.151, −0.255, 0.431] | [−0.150, −1.609, 0.463, −1.135, −0.886, 0.850] | [0.439, 0.537, 0.704, −0.382, −1.389, 0.686] |
| RIGHT p48 | [0.462, −1.146, 0.281, −1.550, 1.250, 0.950] | [−0.680, 0.481, 0.523, −0.034, 1.500, 0.745] | [0.182, 0.128, 0.832, −0.689, 0.662, 0.668] |
| RIGHT p62 | [0.462, −1.189, 0.226, −1.550, 1.268, 0.950] | [−0.675, 0.411, 0.660, −0.118, 1.398, 0.690] | [0.273, 0.150, 0.912, −0.664, 0.650, 0.689] |

Each row is the analytic CEM plan plus a seeded ±0.1 rad perturbation, clipped to the envelope. The unperturbed CEM plans are in `analytic/cem_search.log` and in the probe script (`KEYS`); the four plans above are in `analytic/confirmed_keys.json`.

**Torque.** Static lower bounds along the confirmed paths, per joint, peak in N·m:

| Plan | Friction | pan | lift | elbow | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|---|
| LEFT p9 | μ ≈ 0 | 0.07 | 0.22 | 0.26 | 0.25 | 0.16 | 0.06 |
| LEFT p9 | μ = 0.35 | 0.06 | 0.14 | 0.12 | 0.18 | 0.15 | 0.09 |
| RIGHT p48 | μ ≈ 0 | 0.05 | 0.21 | 0.25 | 0.22 | 0.11 | 0.09 |
| RIGHT p48 | μ = 0.35 | 0.05 | 0.12 | 0.11 | 0.19 | 0.10 | 0.08 |

The μ ≈ 0 rows are the frictionless case. The worst joint is about 0.27 N·m, against 0.8 (sim), 0.49 (rated) and 1.91 (stall). For comparison, holding the whole 0.385 kg arm out horizontally from the lift joint would need about 0.45 N·m; the righting path never does that, because the floor carries most of the weight. Within the model's assumptions there is no static torque obstacle on either side, even at rated torque.

**Time.** Summing the largest joint change per segment at the 1 rad/s slew gives 5.2 s (left) and 4.5 s (right), home return included. Upright crawling covers 5 cm in about 0.75–1.15 s (`runs/evaluation-t100-development-3000/crawl-*.jsonl`). The 30 s budget is not binding.

**Dynamic flinging is not an option.**
- Righting raises the COM from 0.040 m to about 0.104 m above the floor, a potential-energy barrier of about 0.39 J (m·g·Δh, both sides).
- Upper-bound arm kinetic energy (fixed-base, distal inertia about the joint, home or extended arm):

| Joint speed | Kinetic energy |
|---|---|
| 1 rad/s (policy slew) | 0.003–0.009 J |
| 2 rad/s (sim cap) | 0.013–0.036 J |
| 5.45 rad/s (real no-load speed) | 0.09–0.27 J |

- In the side pose, three of the joints swing about vertical axes, so their momentum does not act about the righting axis at all.
- wrist_roll's axis is useful, but the gripper's inertia about it is tiny.

Righting must therefore be quasi-static, with work done slowly through floor contact. The 1 rad/s slew suits that.

## Part 2: Isaac Lab scripted probe

The script is `scripts/feasibility/side_recovery_probe.py`, run on the project container with 64 envs, headless.

**Setup:**
- `SnailEnv` with `configs/experiments/recover-t100-001.json` physics and `evaluation=True`, starting from the recorded settled LEFT and RIGHT poses after the evaluator's 2 s zero-action settle.
- Keyframes go through the **normal action path**: envelope clip, 1 rad/s target slew, implicit PD (stiffness 12, damping 0.5).
- Per-env PhysX effort and velocity limits are overridden to (0.49 N·m, 2 rad/s), (0.8, 2), (1.9, 5.45) and, in the second run, (1.9, 2).
- After the final home keyframe has been held for 1 s, control passes to the frozen T100 crawl policy (`runs/scale-treatment-051-100/exported/policy.pt`, the same policy as the evaluation report).
- Each env is scored with the acceptance `Trial` (`src/le_snail/evaluation.py`: stable ≥ 0.8 for 0.5 s at line 49, +5 cm at line 60, 30 s timeout at line 25), using the pre-reset transition.
- Each side runs under nominal physics (stage `recover`) and robust randomization (stage `robust`).

**Run 1, sweep** (`runs/feasibility-side-recovery-20261014T111208Z/results.json`). The unperturbed analytic plans plus 58 random ±0.1 rad perturbations, with perturbation and effort level tied to env index, so this run is not a matched comparison.

| Side | Unperturbed plan | Perturbations passing |
|---|---|---|
| LEFT | Nominal physics: cosine 0.64–0.67, failed. Robust: 3/3 passed. | 7/58 nominal, 15/58 robust |
| RIGHT | Cosine up to 0.17 only, failed in both stages | 4/58 nominal, 5/58 robust |

The sim-versus-model gap is real: dynamics, sliding and cooked contact hulls narrow the analytic basin a lot. Run 1's torque counters were not masked to the scripted phase; do not use them.

**Run 2, matched confirmation** (`runs/feasibility-side-recovery-20261014T111801Z/results.json`). The passing perturbations from run 1 are replayed with identical keyframes at every effort and velocity setting. Each cell is passes over trials (nominal + robust):

| Plan | 0.49 N·m, 2 rad/s | 0.8, 2 | 1.9, 2 | 1.9, 5.45 |
|---|---|---|---|---|
| LEFT p9 | 6/6 | 6/6 | 5/6 | 6/6 |
| LEFT p34 | 6/6 | 5/6 | 5/6 | 6/6 |
| LEFT p14 | 4/6 | 5/6 | 4/6 | 6/6 |
| LEFT p44 | 2/4 | 4/4 | 4/4 | 4/4 |
| LEFT p35 | 1/4 | 1/4 | 0/4 | 3/4 |
| LEFT p29 | 0/6 | 0/6 | 0/6 | 6/6 |
| RIGHT p48 | 6/6 | 6/6 | 6/6 | 6/6 |
| RIGHT p62 | 5/6 | 3/6 | 5/6 | 6/6 |
| RIGHT p14 | 0/8 | 0/8 | 0/8 | 8/8 |
| RIGHT p26 | 0/6 | 0/6 | 0/6 | 6/6 |
| RIGHT p56 | 0/6 | 0/6 | 0/6 | 2/6 |

- Stable upright was credited at a median 3.6 s (left) and 2.95 s (right). Full passes, including the 5 cm crawl, finished at a median of about 6.7–9.6 s. The 11 left failures that were faults all left the lane (lateral > 0.15 m).
- **Torque is not what decides success.** Results at 0.49, 0.8 and 1.9 N·m with the same 2 rad/s cap are statistically indistinguishable.
  - The limits were in force: PhysX readback of the drive max force gave 0.49, 0.8 and 1.9 per group (`applied_effort_limit` in results.json). This was read for shoulder_pan; all six joints were written in the same call.
  - The deterministic nominal-physics upright traces for the same plan at 0.49 vs 0.8 N·m differ by at most 0.004 (LEFT p9) and 0.010 (RIGHT p48) in upright cosine over the first 4 s. That is the same as the 0.8 vs 1.9 N·m difference (0.008), so it is within the noise floor.
  - So the righting motion rarely needs more than 0.49 N·m, consistent with the quasi-static bound of about 0.27 N·m.
- **The 2 rad/s PhysX joint velocity cap is what decides success.** Several plans pass only when it is raised to 5.45 rad/s, at the same 1.9 N·m. Commanded targets never exceed 1 rad/s, so the cap must be acting on joints that contact or the tip-over drives faster than 2 rad/s, braking them. This is a hypothesis; it was not instrumented. Treat the velocity cap as a sim-fidelity parameter that affects recovery.
- **PD torque estimates.** These are nominal-gain estimates, clipped at each env's limit and counted only during the scripted phase of unfinished trials. Robust-stage gains differ by ±20 %, so they are approximate.
  - At 0.8 N·m, shoulder, elbow and wrist_flex peaks stay below or brush the limit and are saturated ≤ 0.2 % of the time.
  - The gripper is saturated 21–53 % of the scripted phase and wrist_roll up to 3.5 %. This reflects the jaw being commanded into the floor (tracking error), not load the motion needs.
  - At 0.49 N·m, left elbow saturates 8.7 % and wrist_roll 15 %, and the robot still recovers.

## What blocks recovery today, and minimal changes

- **No physical blocker in sim.** No joint-limit, torque-limit or slew change is required.
- **Learning difficulty is the blocker.**
  - The successful motion is a 3–4 s coordinated sequence that first plants the jaw and then sweeps pan across while raising lift.
  - The basin is narrow: about 7–26 % of ±0.1 rad perturbations work (right side 7–9 %), and the unperturbed model plan failed under nominal physics.
  - While fallen, the dense reward gives almost no gradient. Tilt is only −0.1·(1−cos) per second (`src/le_snail/sim/env.py:228`), and forward progress is gated off below cosine 0.5 (lines 221–225).
- Options, roughly cheapest first. These are suggestions, not decisions; some may conflict with existing project choices.
  1. **Scripted righting maneuver.** Trigger it when the IMU shows side-lying, play a side-specific keyframe set (LEFT p9 or p34, RIGHT p48), then hand over to the crawl policy. In this probe that passed 6/6 per side at 0.8 N·m, including the crawl. Whether a scripted maneuver satisfies acceptance is a project decision.
  2. **Seed recovery RL with these keyframes**, as demonstrations, a residual policy, or start states partway through the roll.
  3. **Shape the fallen phase**: reward upright improvement, possibly potential-based so the crawl objective is not distorted.
- **Sim fidelity.** The provisional 2 rad/s joint velocity cap matters for which motions work. Raise it toward the measured servo speed, or measure it, before trusting fine-grained recovery results.
- **Thin threshold margin.** The settled upright cosine is 0.827 against the 0.8 threshold. Small model or hardware differences in base pitch could push a correct recovery below the threshold.

## Real robot

- **Torque.** The static demand along the working paths is about 0.12–0.27 N·m (lower bound), against 0.49 N·m rated and 1.91 N·m stall (`docs/research/sim-to-real-actuator-model.md:211-214`). The sim also passes at 0.49 N·m. Stall torque is quoted at 7.4 V; the stock kit may run the servos from 5 V (line 227), which lowers both torque and speed. Even so, torque is unlikely to be the limiting factor.
- **Unvalidated risks, which are why the verdict is "uncertain":**
  - Collision and contact geometry is unvalidated (`contact_geometry_validated: false`). The mechanism depends on the jaw tip and one base edge acting as pivots.
  - Real friction and sliding at the jaw tip.
  - Cable and tether routing at the base (tether attachment, config line 33), which could block the roll or snag. Cable stiffness is not modelled; only planar drag is.
  - Servo compliance and backlash, and real velocity behaviour under backdriving.
  - Fragility: open-loop plans work only in narrow basins in sim. A closed-loop policy using the IMU, or a scripted maneuver with check-and-retry, is much more likely to transfer than a single open-loop replay.
- **Hardware check worth doing early, before investing heavily:** hand-script LEFT p9 and RIGHT p48 at low speed on the real arm, and watch whether the jaw pivot and the pan sweep behave as in sim.

## Alternative: crawling 5 cm while still side-lying

This is physically possible but does not meet acceptance, which requires regaining the posture first. While fallen, T100 moved:
- LEFT: +3.8 to +5.1 cm in 30 s.
- RIGHT: −2.4 to +4.7 cm.

Two of the ten trials crossed 5 cm, late, at 28.5 s and 26.5 s (`runs/evaluation-t100-development-3000/{left,right}-*.jsonl`; videos alongside). Since righting is feasible, this fallback should not be needed.

## Limitations

- The quasi-static model ignores sliding and dynamics. Its LP torques are lower bounds. Its self-collision check uses convex part hulls (conservative).
- Isaac PD torques are estimates, not PhysX joint forces.
- No video of the scripted probe was rendered, to avoid adding GPU load.
- Success counts are small (4–8 trials per cell). The nominal-physics stage is near-deterministic, so its repeats are not independent.

## Reproducibility

The probes ran against the version of `src/le_snail/sim/env.py` on disk at about 11:12–11:22 UTC. The file has changed since. Line numbers cited here refer to the current file.

| Item | Path |
|---|---|
| Analytic code (pure numpy/scipy): FK and hull model, rolling, LP, self-collision, CEM, robustness, energy | `runs/feasibility-side-recovery-20261014T112358Z/analytic/` (model.py, qs.py, roll.py, analyze.py, search2.py, robust.py, tq.py, tq2.py, energy.py, confirmed_keys.json, cem_search.log) |
| Isaac probe script | `scripts/feasibility/side_recovery_probe.py` (`--mode sweep` reproduces run 1, `--mode confirm` reproduces run 2) |
| Probe results | `runs/feasibility-side-recovery-20261014T111208Z/` and `runs/feasibility-side-recovery-20261014T111801Z/` (results.json, upright_traces.npz) |
