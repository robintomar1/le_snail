# Le Snail continuation spec

Status: ready for implementation under the phase gates below; saved locally at the user's request. Testing approach confirmed on 2026-10-14. GitHub publication is deferred.

## Problem Statement

Le Snail must make the existing tethered SO-101 arm crawl forward on tiles, recover from representative sideways falls, and eventually reproduce that behavior on the physical arm. The project already has an implementation and a working Isaac runtime, but it has not demonstrated crawling, recovery, or hardware transfer.

The immediate problem is deciding what to validate and execute next without restarting the design, losing existing work, or treating working software as evidence of workable mechanics. Training against uninspected contacts or inconsistent control semantics could consume GPU time and produce a policy that exploits the simulator. Hardware transfer also depends on measurements and sensor integration that software tests cannot supply.

### Current baseline

The latest handoff supersedes older pending-preflight statements in the design and README.

| Area | Established evidence | Remaining work |
| --- | --- | --- |
| Runtime | Docker and GPU access work. Isaac Sim 5.1.0, Isaac Lab v2.3.2 and RSL-RL 3.0.1 rebuilt successfully; launcher and CUDA checks passed. | Recheck after relevant changes; preserve the working image and logs. |
| Asset and environment | Recorded SO-101 source and prepared asset hashes; floating-base and six-joint guards passed. | Render and inspect collision fidelity, mass/inertia, home posture, side poses and contact behavior. |
| Smoke test | Four CUDA environments completed 100 control steps from each of upright, left and right resets with finite 24-value observations and rewards; exit code 0. | Exercise meaningful actions and inspect resulting motion. Zero-action finite state does not establish locomotion. |
| Portable tests | 51 tests and 4 subtests passed inside the container. | Add behavior-focused checks for the untested simulation/export/deployment boundaries. |
| Physics and control | Provisional gains, limits, friction, noise, delay and drag are implemented. | Validate simulated behavior and collect physical measurements. All measurement/contact flags remain false; protected-point locations are empty. |
| Learning and evaluation | Staged PPO, checkpoint/export, fifteen-trial evaluation, video and reporting code exist. | Execute and verify the complete pipeline, then demonstrate acceptable crawling and recovery. |
| Hardware | Assembled, calibrated arm with user-confirmed 7.4 V servos; laptop controller and Xsens IMU reader source exist. | Obtain/integrate the selected IMU, compile firmware, measure mappings and timing, validate fault handling, and run supervised trials. |
| Reproducibility | GPU workspace, assets, configs, logs and backups are available. | The workspace is not a Git checkout and source has not been pushed; experiments need an identifiable project-source snapshot as well as dependency and asset versions. |

## Solution

Continue from the successful runtime baseline through explicit evidence gates. First establish trustworthy geometry and control behavior, then prove a short training-to-export-to-evaluation cycle, then increase training in the accepted crawl → recover → robust sequence. Prepare hardware measurements and the IMU in parallel with simulation work. Use the same exported policy contract and acceptance rules throughout.

Each phase produces a reviewable artifact, records its inputs and failures, and states whether the next phase may start. Exploratory training may use explicitly provisional parameters after basic simulation checks pass. Formal milestone approval and live policy deployment require measured parameters, validated contacts and protected points, and reviewed gait evidence.

### Acceptance criteria that remain fixed

- **Distance:** at least 50 cm net forward movement of the base within 120 seconds, including recovery time, while remaining within ±15 cm lateral displacement throughout the trial; at least 3 successes in 5 trials.
- **Recovery:** separate sets of five trials from recorded left and right fallen poses. Regain a stable crawling posture, then advance the base at least 5 cm forward within 30 seconds total; at least 3 successes in 5 trials on each side.
- **Gait and contacts:** crawling may include sliding, temporary tilting, falls and recovery. Continuous rolling is unacceptable. The base and gripper are intended supports; occasional other plastic contact is acceptable; exposed wiring, connectors and electronics stay off the floor.
- **Evidence:** evaluate one exact export with held-out seeds, preserve all fifteen outcomes and associated traces/videos, and record the visual review separately from numerical success. Higher training reward is not an acceptance result.
- **Hardware:** use the same distance/recovery requirements on tiles with external displacement measurement and video. A simulation result does not establish a hardware result.

## User Stories

1. As the project owner, I want the plan to start from the verified runtime state, so that completed setup is not repeated unnecessarily.
2. As the project owner, I want the agreed requirements preserved, so that implementation difficulties do not silently weaken the target.
3. As the project owner, I want completed, provisional and untested work distinguished, so that progress claims reflect evidence.
4. As the project owner, I want a clear next task with a deliverable and exit check, so that each session advances the project.
5. As the project owner, I want existing files, calibration, assets and experiment logs preserved, so that new work does not erase earlier evidence.
6. As a simulation engineer, I want to inspect the imported floating arm and its collision geometry, so that apparent motion comes from plausible contacts.
7. As a simulation engineer, I want a stable folded start posture and representative left/right fallen poses, so that training and evaluation begin from repeatable conditions.
8. As a simulation engineer, I want all six joint directions, ranges and target mappings checked, so that commands move the intended joints within bounds.
9. As a simulation engineer, I want actuator response and saturation inspected, so that the simulator does not rely on unphysical strength or speed.
10. As a simulation engineer, I want bounded tether drag that opposes attachment motion, so that the cable approximation cannot propel the arm.
11. As a robot operator, I want vulnerable parts represented by measured protected points and reviewed contacts, so that a learned gait avoids damaging support locations.
12. As a simulation engineer, I want bounded motion probes before a large training run, so that obvious mechanical infeasibility is discovered early.
13. As a policy developer, I want simulation and deployment to agree on observation order, timing, filtering and applied-action history, so that the exported policy receives the inputs it learned from.
14. As a policy developer, I want invalid observations and actions to fail visibly, so that numerical faults cannot become apparently successful experiments.
15. As a policy developer, I want a short PPO run to save, resume and export successfully, so that the learning pipeline is proven before more compute is allocated.
16. As a policy developer, I want exported inference to agree with training inference on the same observations, so that serialization and normalization preserve policy behavior.
17. As a policy developer, I want upright crawling learned first, so that recovery and robustness build on demonstrated forward motion.
18. As a policy developer, I want recovery introduced from both sides, so that a policy does not pass by succeeding on only one favorable pose.
19. As a policy developer, I want bounded variations in friction, mass, gains, delay, noise and tether drag, so that the policy is evaluated beyond one nominal configuration.
20. As a project owner, I want training budgets increased only after reviewing results, so that repeated failures trigger diagnosis rather than uncontrolled spending.
21. As an evaluator, I want distance measured from the base's scored start position, so that gripper motion, backtracking and settling do not inflate progress.
22. As an evaluator, I want recovery trials to begin genuinely fallen, so that passive settling into an upright pose cannot masquerade as learned recovery.
23. As an evaluator, I want every success and failure recorded under fixed trial rules, so that weak results cannot be hidden by selective reruns.
24. As an evaluator, I want videos synchronized with scored traces, so that numerical success can be checked for rolling, invalid contact and reset artifacts.
25. As a project owner, I want provisional results distinguished from validated milestones, so that a useful experiment is not mistaken for deployment readiness.
26. As a robot operator, I want the selected IMU and bridge to provide correctly framed, timestamped gravity and gyro readings, so that the laptop can reproduce the policy's sensor inputs.
27. As a robot operator, I want separate verified mappings for the body joints and gripper, so that their different native units cannot produce incorrect motion.
28. As a robot operator, I want measured servo response, friction, tether resistance and control latency fed back into simulation, so that training reflects the assembled robot.
29. As a robot operator, I want offline replay on the laptop before motion, so that policy loading and target generation can be checked without commanding servos.
30. As a robot operator, I want stale feedback, missed deadlines, telemetry faults and tracking errors to stop further commands, so that known control faults do not continue motion.
31. As a robot operator, I want supported first-motion checks and a verified physical cutoff, so that initial transfer is supervised and stoppable.
32. As a project owner, I want runs tied to exact source, asset, contract and physics snapshots, so that a result can be reproduced and compared after changes.
33. As a future contributor, I want a handoff that identifies the latest evidence and next gate, so that work can resume without repeating the design interview.

## Implementation Decisions

### Architecture and contracts to preserve

- Continue the existing asset preparation, direct Isaac environment, policy contract, staged training, bundle export, evaluation, hardware adapter and IMU bridge modules. Prefer extending their existing entry points over adding a second simulator, controller or evaluation framework.
- Retain Isaac Sim 5.1.0, Isaac Lab v2.3.2 and RSL-RL 3.0.1 unless a demonstrated blocker requires a recorded change. Train on the remote RTX 6000 Ada; execute the physical policy locally on the Ubuntu 22 laptop. The training host is outside the live control loop.
- Use one feedforward PPO policy controlling six bounded joint-position targets, including the gripper. The initial policy rate is 20 Hz and physics rate is 200 Hz. Any measured need to change timing must propagate through training, export, replay and deployment.
- Preserve the 24-value actor interface: six joint offsets from home, six filtered finite-difference joint velocities, three base-frame gravity components, three base-frame gyro rates and six previous applied actions. Ground-truth position, linear velocity and contacts may support rewards/evaluation but are not actor inputs.
- Keep ordered joints, units, quaternion convention, limits, filtering and timing explicit in the versioned contract. Reject incompatible resume/export/deployment combinations. Changes to observation or action meaning require a new contract and fresh training rather than silently reusing a checkpoint.
- Preserve ordinary falls as recoverable episode events. Keep fault termination and acceptance measurement separate from reward shaping. Record a diagnosis if a reward change improves return while degrading acceptable crawling.
- Retain the bounded dissipative tether approximation at the attachment point, with torque about the base center of mass. Measure the assembled three-cable routing before declaring its range representative.
- Keep provisional, numerical-pass, physics-validated and visually-approved statuses distinct. Validation flags summarize evidence; they do not create it. Do not change them merely to enable export or live mode.

### Ordered work and exit gates

| Phase | Dependency and owner | Work and deliverable | Exit gate |
| --- | --- | --- | --- |
| 0 — Preserve the working baseline | Ready now; agent | Inventory the successful image, source snapshot, assets, contract, physics config and logs. Preserve unique run outputs. Reconcile stale runtime statements when updating project docs. | A future run can identify its exact inputs without overwriting the authoritative workspace. Build/smoke work remains completed unless inputs change. |
| 1 — Inspect physics and control semantics | Phase 0; agent, with operator measurements supplied when available | Record rendered upright/left/right scenes, collision views and bounded joint-motion probes. Inspect masses/inertias, fixed-link merging, joint directions/limits, support contacts, actuator saturation and tether forces. Verify observation/action consistency and evaluation start semantics. Produce a validation report, videos and a list of corrections. | No unexplained penetration, fixed-base behavior, incorrect joint mapping, propulsive drag or observation-contract mismatch. Start poses are reproducible. Any missing physical evidence remains explicitly provisional. |
| 2 — Prove training, export and evaluation execution | Phase 1; agent | Run the agreed 64-environment, 20-iteration crawl probe in a new output location. Exercise checkpoint save/load, export, reload, inference comparison, short rendered evaluation and offline replay. Extend the existing error-reporting behavior to these entry points where needed. | Finite learning signals and actions; usable checkpoint/export; matching training/export inference within a documented tolerance; visible nonzero failures; usable video and trace artifacts. No convergence claim is required. |
| 3 — Establish crawling and recovery | Phase 2; agent | Train upright crawling without tether drag first. Inspect videos and base displacement, then introduce representative left/right recovery starts. Record a finite budget and stopping condition before each extension. Preserve the best acceptable checkpoints and failed probes. | Evidence of repeatable acceptable forward motion, followed by evidence of recovery on both sides. If progress relies on rolling or invalid contact, return to physics/reward diagnosis before increasing compute. |
| 4 — Train robustness and qualify the simulation result | Phase 3; agent; final qualification also requires Phase H measurements | Introduce tether resistance and the accepted randomization. Evaluate the exact exported policy on fifteen held-out trials. If measurements change relevant dynamics or the policy contract, retrain and reevaluate. Produce a report, all trial traces/videos and a visual gait/contact review. | Exploratory numerical success is reported as provisional when measurements are missing. A validated simulation milestone requires the fixed 3/5 distance and 3/5-per-side recovery criteria, validated physics/protected points and acceptable reviewed gait. |
| H — Prepare hardware and identify parameters | Can proceed alongside Phases 1–3; operator for physical work, agent for software/log analysis | Mount the Xsens MTi-1s (replaces the BNO085/BNO086 + RP2040 plan, 2026-10-14); confirm its product code, orientation filter and output configuration on the device. Measure joint/gripper mappings, home/limits, servo response, sustained bounds, telemetry units, tile friction, tether resistance, protected-point locations and complete control-cycle timing. Record a hardware evidence package. | Required mappings and measurements are recorded; IMU frames/freshness are validated; actual timing supports the contract or prompts a documented contract change; cutoff/stop behavior is checked with support. |
| 5 — Rehearse deployment without policy-driven motion | Phase 2 for preliminary replay; Phases H and 4 for final readiness; agent and operator | Install and record the laptop adapter/runtime versions, replay recorded joint/IMU samples, compare observations/actions with simulation, and exercise fault cases through the controller boundary. Tie the exact export to the verified hardware configuration. | Final bundle loads unchanged; replay respects limits and timing; faults suppress subsequent commands; configuration, calibration, measurements and contract agree. |
| 6 — Demonstrate physical crawling and recovery | Final Phases 4, H and 5 gates; operator present | Perform supported bounded first-motion checks, inspect logs, then progress to supervised tile trials using a marked lane and video. Run all distance and side-recovery trials with the same cable routing. | Same numerical and gait/contact acceptance as simulation, measured externally. Preserve all failures; unexplained transfer failures return to identification, simulation and retraining. |

The initial critical path is **0 → 1 → 2 → 3 → 4**. Physical measurement work H should start as soon as the operator and parts are available; final qualification in 4 and live work in 6 cannot bypass it. Preliminary replay can start after 2 to expose software incompatibilities early.

### Specific checks identified in the existing implementation

These are implementation review findings to resolve through behavior tests before relying on training or evaluation; they are not claims that a trained policy has already failed.

1. **Applied-action history under delay.** The simulation applies a queued, potentially one-step-delayed target, while its observation currently uses the latest target. Confirm the intended previous-applied-action semantics with controlled command sequences and make simulation, replay and deployment agree. The zero-action smoke test cannot reveal this difference.
2. **Observation initialization and reset.** Compare finite-difference velocity/filter initialization, gravity conventions and action history across the simulation and portable contract, including partial resets and repeated observation requests. Test deterministic sensor inputs first, then separately check the configured noise and delay behavior.
3. **Scored trial starts.** Evaluation currently settles each start for two seconds before scoring. Inspect all three settled poses, reject invalid or already-upright recovery starts, exclude settling displacement consistently, and ensure automatic environment resets cannot contribute progress or hide a failed transition. Preserve terminal-state evidence in both traces and video.
4. **Boundary conditions in evaluation.** Verify the simulator's episode timeout against the acceptance evaluator at exactly 120 seconds and 30 seconds. Keep the existing stable-posture convention—upright cosine at least 0.8 continuously for 0.5 seconds—as an explicit operational proxy subject to contact/video review, not proof of a feasible crawling posture on its own.
5. **Physics beyond configuration parsing.** Existing portable physics tests reject invalid parameter values; they do not prove force direction, drag power, actuator saturation or collision fidelity. Use actual simulated transitions to establish those properties.
6. **Training/export/evaluation failures.** The smoke test now preserves completion messages and nonzero failures through Kit shutdown. Check the equivalent behavior in training and evaluation; a process exit code without its expected artifact set is insufficient evidence of success.
7. **Artifact identity.** Record the project-source snapshot, exact image/dependency versions, assets, configuration, seed, curriculum stage and checkpoint lineage. Existing asset provenance and bundle checks should be retained and strengthened where the live asset or project source is not yet tied to the result.

### Experiment and decision rules

- Use fresh output locations for every probe, training continuation and evaluation. Preserve unsuccessful runs and the pre-change baseline; never overwrite the downloaded asset provenance or the user's unrelated files.
- Before extending training, record what the next run is intended to answer, its maximum iterations/environment count, the preceding checkpoint, the changed variables and the review criterion. The twenty-iteration probe establishes execution only; later budgets follow observed progress.
- Fix invalid mechanics or control semantics before trying to compensate with reward tuning. If bounded probes expose a likely inability to crawl or escape a selected side pose, produce evidence and bring the contact/body or pose decision back to the owner. Do not silently add pads, replace parts or redefine success.
- Physical measurements may refine home posture, gains, limits, friction, delay, noise and tether ranges. Document each update and its measurement basis, preserve earlier results as provisional, and repeat affected validation/training/evaluation.
- Source publication/version-control setup must preserve the transferred workspace. Establish a backup and inspect the remote repository before any import; do not replace this workspace with an unchecked clone or overwrite remote history.
- Update the runtime handoff after each completed phase with evidence, remaining uncertainty and the next executable task. Publishing this spec later does not itself prove or complete any phase.

## Testing Decisions

The user confirmed the main testing seam: **the existing simulation → training/checkpoint → exported policy → evaluation flow**, supplemented by focused observation/action consistency and hardware fault-handling checks. Prefer this integrated behavior boundary to a parallel mock simulator or a new family of internal test hooks.

A good test establishes externally observable behavior: which targets were applied, which observations the actor received, whether displacement and recovery were scored correctly, whether a bundle can be loaded, whether a fault prevents a command, and whether a failed process reports failure. Avoid tests that merely copy implementation formulas or assert private buffer layouts.

| Boundary and modules | Behavior to verify | Existing prior art |
| --- | --- | --- |
| Isaac environment and policy contract | Finite states under meaningful bounded actions; correct shape/order/units; target and slew bounds; previous applied action under zero/one-step delay; reset/filter consistency. | Current three-pose GPU smoke test and portable contract/filter tests. |
| Imported asset and physics | Floating base, expected joints, credible contacts and start poses; bounded actuator behavior; tether force opposes planar attachment motion, respects its bound and produces the appropriate moment. Include zero translation, rotation and zero-speed cases. | Asset preparation tests for collisions/anchoring and physics configuration rejection tests; extend through observed simulation behavior. |
| PPO checkpoint, export and bundle loader | Short real learning run; save/resume; incompatible contract rejection; normalizer included; training/export action agreement on recorded observations; checksum/provenance checks. | Existing bundle tamper and contract-mismatch tests, plus the actual training/export entry point. |
| Trial evaluator and rendered evaluation | Net base displacement, strict deadlines, lane violations, stable recovery before subsequent 5 cm progress, genuine fallen starts, terminal transitions, complete five-trial groups and interpretable videos. | Existing tests for backtracking, deadline boundaries, lane reentry, transient upright poses, initially upright recovery starts and separate recovery groups. |
| Laptop replay and hardware adapter | Same recorded inputs yield compatible observations and bounded targets; distinct body/gripper mappings; incorrect contracts or unfilled calibration prevent live operation. | Existing joint-mapping, sensor-frame, timestamp and IMU replay/reboot rejection tests; the full offline policy-replay path still needs execution. |
| Controller command boundary | Stale/reordered/restarted IMU stream, slow feedback, missed cycle, invalid/nonfinite policy output, tracking error, missing/bad telemetry and operator interruption prevent subsequent commands as appropriate; shutdown attempts torque disable when communication is available. | Existing IMU decoder/config tests; add controller-level fault injection with recorded/fake transport at the external I/O boundary, then verify actual stop behavior under operator supervision. |

Verification layers remain explicit:

1. Run relevant portable behavior tests for changed contracts, scoring, bundle integrity and hardware parsing/control logic.
2. Run the GPU smoke and bounded inspection checks for changes to the environment, asset, reset logic or dynamics.
3. Run the small real learning/export/reload/evaluation cycle before scaling training or relying on a changed pipeline.
4. Run formal held-out evaluation on the exact final export, retaining all fifteen trial outcomes and visual review. If those seeds are used to tune the policy, reserve a fresh held-out set for final evaluation.
5. Validate real measurements, sensor timing, physical stop behavior and tile trials on the actual hardware. Portable and GPU tests cannot substitute for this evidence.

Any new test seam should sit at the existing environment, bundle or controller I/O boundary. The plan does not require a broad architecture rewrite solely for testability. Numerical tolerances for inference and measured-response comparisons should be documented with their measurement precision rather than invented to make a check pass.

## Out of Scope

- Turning, navigation and the 8×8 IR obstacle sensor.
- Tabletop surfaces and commercial carpet before the tile milestone.
- Full cable dynamics, winding, snagging, untangling and recovery from every upside-down pose.
- New contact pads, protective parts or replacement printed body components in the initial configuration; a demonstrated mechanics problem may motivate a separate decision.
- Alternative simulators, recurrent policies, extra deployed ground-truth observations or additional compute infrastructure without a demonstrated requirement.
- Treating a short PPO run as convergence, a numerical pass as a reviewed milestone, or a simulation pass as proof of hardware transfer.
- Automatic recalibration or overwriting existing robot calibration.
- Unsupervised physical motion, unmeasured operating limits or bypassing provisional-physics and hardware-readiness gates.
- Treating the original two-day simulation/four-day hardware targets as guaranteed delivery dates. Remaining timing depends on mechanics, learning progress, parts availability and measurements.

## Further Notes

### Immediate next work package

Execute Phase 1 after preserving the current baseline: produce a physics-and-control inspection report with upright/left/right videos, bounded joint-motion observations, tether-force evidence, and the applied-action/reset consistency results. Record corrections and remaining physical measurements. The report must state whether the 64-environment, 20-iteration training probe may begin and why.

### Evidence and source map

This spec synthesizes the accepted design and current code; it does not record new simulation or hardware experiments. Local source material:

- [Runtime handoff](../HANDOFF.md): authoritative runtime status, successful image identity, evidence locations and continuation rules.
- [Project README](../README.md): existing training, export, evaluation and deployment workflows.
- [Accepted design](design.md): accepted behavior, success criteria, hardware choices and deferred scope.
- [Simulation validation](simulation.md): contact, actuator, tether and measurement requirements.
- [Hardware integration](hardware.md): IMU, joint/gripper mapping, calibration, timing and deployment checks.
- [Glossary](../GLOSSARY.md), [issue tracker instructions](agents/issue-tracker.md), [triage labels](agents/triage-labels.md) and [domain instructions](agents/domain.md): domain vocabulary and tracker conventions.
- Existing simulation, contract, evaluation, bundle and hardware modules; training/evaluation/deployment entry points; the Xsens IMU reader; and their portable tests.

The successful runtime evidence is retained in `.setup/logs/resume-access-20261014-jFlmsL/`, including `smoke-final.log`, `smoke-final-state.json` and `container-tests.log`. The successful image ID is `sha256:00004264b9b940782ef515b1252ff866f33152a7c1d7c57e0a13f4f625bcf10e`. These artifacts are local to the authoritative GPU workspace and are not assumed to be available in GitHub.

### Ownership and publication

The agent can perform simulation inspection, code corrections, tests, training probes, evaluation, replay tooling and evidence analysis. The operator supplies physical access, hardware measurements, sensor acquisition/mounting and supervision for movement. Missing human measurements block the relevant qualification/deployment gate, while independent simulation work can continue.

The user requested a completed local spec for now. When publication is requested and GitHub authentication is available, create a spec issue in `robintomar1/le_snail` with the `ready-for-agent` label, following the project tracker instructions and checking for an existing equivalent issue first. Human-dependent gates remain explicit even when the overall spec is labeled ready for agent work. No issues, labels or comments were created for this local spec.
