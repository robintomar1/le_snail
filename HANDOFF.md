# Le Snail: GPU-machine handoff

Updated 2026-10-14. Resume in `/home/clutterbot/robin/le_snail_ws` on the GPU host.
The user requested moving work here so the agent can run Docker and inspect the GPU directly.

## Start here

1. Read `AGENTS.md`, this document, and `README.md`. Read `docs/simulation.md` before changing the environment or validating contacts. The accepted requirements are in `docs/design.md`; the runtime status below supersedes its older pending-preflight notes.
2. The latest Dockerfile has now been rebuilt successfully, and launcher/CUDA validation and the smoke test passed on 2026-10-14. See the current runtime status below for the exact image and logs. Rebuild after relevant environment/source changes.
3. Retain the smoke test as a regression check. Diagnose failures against the pinned upstream APIs and retain complete logs. Completion means finite observations/rewards for upright, left and right resets, not merely a successful Docker build.
4. Inspect the imported arm's floating base, collisions, initial/fallen poses, joint order, actuator behavior, and tether force. A smoke-test pass alone does not validate physics or demonstrate crawling.
5. Only after those checks, start a short PPO crawl run and inspect results before increasing the training budget. Continue with recovery, tether/friction variation and held-out evaluation as described in the README.

The user has approved implementation, remote workspace setup, Docker builds, simulation work and the initial training plan. They ran the setup with `--accept-eula`. Continue that work without repeating the design interview. Physical robot motion still requires actual hardware measurements, calibration and the deployment checks described in `docs/hardware.md`.

## Current runtime status and exact blocker

### Straightness continuation approved; large pair held: 2026-10-14

- Laptop GO for `crawl-straight-001`: resume `runs/crawl-supported-001/model_final.pt`, seed42,256 environments,100 NEW iterations maximum. Only reward change is yaw cost `-dt * (1-cos(yaw-yaw_at_reset))`, weight1.0. Config: `configs/experiments/crawl-straight-001.json`. Contract/physics/reset states/other rewards unchanged. This diagnostic continuation retains the supported gait; critic estimates must adapt to the changed reward.
- Real-environment red/green regression: `runs/straight-reward-before-001` fails the missing yaw cost; `runs/straight-reward-after-001` passes35 checks including aligned/quarter-turn/half-turn/full-turn costs and independent heading origins on partial reset. Portable suite70+4subtests passes. Rebuilt image d9eb69c58ea9e6c5927130e7bb4e87de35b691d54cdd151cc648e55c901462c8 and smoke passed; the small continuation is running. Check `.setup/crawl-straight-001.exit` and artifacts before resuming.
- User also requested a larger matched treatment/control pair via laptop. **Do not launch that pair until explicit GO-SCALE/GPU-sharing confirmation in communication file.** Pre-stage only: both arms should start from the SAME checkpoint, same seed/environment/budget/minibatches; checkpoint/video review every50 new iterations. Do not confound treatment with extra pretraining. Small-run outcome must inform the treatment choice.
- Scaled preparation is concrete in `runs/scaled-pair-plan-001/plan.md`: candidate2x4096 envs with16 minibatches, symmetric2x2048/8 fallback only if measured resources require it. `train.py` exposes minibatch/save interval controls with old defaults. `scripts/export_checkpoint.py` exports exact intermediate ActorCritic snapshots without training; final/iteration0 supported checkpoints both match2,048 recorded observations exactly on CPU. **Preparation is not GO-SCALE.**
- Communication: use `python3 .setup/communication-watch/read_new.py` to read ALL unread appended messages using a persisted cursor, rather than a short tail that can miss long messages. Detached watcher still records updates. Continue responding in the append-only communication file.


### Supported-crawl experiment complete; straightness diagnosis next: 2026-10-14

- `runs/crawl-supported-001` completed its 256-env x100-iteration cap (1,638,400 transitions, exit0). Export parity passes 2,048 observations with maximum error 7.153e-7. No budget extension was launched. Training log: `.setup/crawl-supported-001.log`.
- Final randomized diagnostic: `runs/evaluation-supported-003/` (videos plus yaw traces), seed2000. Crawl stays above upright cosine0.8379, reaches peak27.50cm, then leaves the lane at8.9s with net23.75cm and lateral-15.03cm. Unwrapped heading changes -192.26 degrees. Videos show repeated supported movement with substantial turning, not the old fallen lurch; it still fails the agreed task. Left/right remain fallen for the full30s and fail recovery. No qualification/milestone passed.
- Fixed evaluator timer accounting after a real long-trial reproduction: excluded settling was still counted in fallen_time and ended side trials at28.05 scored seconds. Reset fallen_time with the episode timer at scoring start. Both sides now reach30.0s without early simulator termination; failing and corrected runs are retained as evaluation-supported-001 and -002/-003.
- Added diagnostic base_yaw_rad and unwrapped yaw_from_start_rad to evaluation traces/overlay, captured before automatic reset. These are scoring diagnostics only; the actor still has24 inputs. `--nominal-probe` allows comparison with nominal training physics and requires `--probe`; qualification stays randomized.
- Laptop relayed robin's decision: **Option A first, reward-only straightness, keep the24-input contract**. Next compare the same export under nominal vs randomized physics, then propose ONE reward change and a separate bounded experiment. Heading observations remain out of scope without a later decision. Nominal comparison completed at `runs/evaluation-supported-nominal-001` (exit0): crawl also exits the lane at8.8s, net21.13cm, yaw -206.81 degrees. Turning therefore exists without evaluation randomization.
- Evidence summary/contact sheets: `runs/reset-reward-analysis-001/`. All three final randomized videos are `runs/evaluation-supported-003/{crawl,left,right}-0.mp4`. Portable tests remain68+4subtests. Source implementation311c05c pushed by laptop; diagnostic/timer changes are included with this handoff update. A one-change yaw-cost continuation proposal is posted in the communication file; do not infer it has run.


### Phase 3 reset/reward correction and capped experiment: 2026-10-14

- Active coordination is in `agents_communication.txt`. Read and answer laptop messages between work steps. Detached file watcher PID 2674421 appends to `.setup/communication-watch/updates.log`; it records changes but cannot start an agent turn. Laptop handles GitHub tracking/pushes and hardware; GPU handles simulation.
- Standing user requirement relayed by laptop: **render every trained export**, including probes/resumes, and post exact video/contact-sheet paths with an honest gait description. Evaluation now defaults to a close whole-arm tracking camera with progress overlay; `--camera wide` retains fixed framing. Both existing exports have close three-start videos: `runs/evaluation-original-close-001/` and `runs/evaluation-resume-close-002/`. Both still fall; no gait milestone.
- Diagnosis: `runs/reset-reward-diagnosis-001/` (15 nominal actual-physics conditions), interpretation/counterfactuals in `runs/reset-reward-analysis-001/`. Passive upright settling earns +0.3185 reward from 12.99 mm in two seconds. A settled hold has negligible drift; the existing policy still earns positive reward while falling/sliding after settlement.
- Experimental config only: `configs/experiments/crawl-supported-001.json`, preserving the original `configs/sim.json` and policy contract. Optional simulated settled root/joint states are tied to the original asset hash and contract fingerprint; they are not physical measurements. Positive progress gets a posture ramp from zero at cosine 0.5 to full at 0.8; all backward progress remains penalized. Existing other costs remain unchanged and reward components are separately logged.
- Red/green evidence: `runs/supported-reset-before-001/` reproduces free reset progress; `runs/supported-reset-after-002/` passes 30 actual-physics checks including signed motion and contract/reset history. Passive drift <6 micrometres over two seconds, no positive return. Portable tests: 68 passed + 4 subtests.
- Agreed bounded experiment: fresh crawl initialization, seed42, 256 environments x **100 iterations maximum**, output `runs/crawl-supported-001`. Laptop GO and conditions recorded in communication file. Require rebuilt image/smoke before launch; then export/parity, three close videos and a longer upright diagnostic. Stop after this budget and report if supported sustained motion is absent; do not extend automatically.
- Rebuilt image: `sha256:231d4952ab3b011e394f35a367f8c37177c500b46b74d090725c7a3421d8fc3d`. Build log `.setup/supported-rebuild.log`; smoke status `.setup/supported-smoke.exit`. Capped learning has not yet started at this entry; check run artifacts before resuming.


### Phase 2 pipeline executed: 2026-10-14

- Final rendered regression: **25 checks passed**, exit 0 (`visual-regression/`), including the terminal-render and deadline paths. Portable suite remains **51 tests and 4 subtests passed** (`portable-phase2.log`).

- The capped crawl run completed: `runs/crawl-probe-001`, 64 environments, 20 iterations, seed 42, 81,920 transitions, exit 0. Checkpoint and exported bundle exist; source/config/asset snapshots are retained. `export-parity.json` records 512 actual rollout observations with maximum CPU-export/CUDA-inference action difference **8.345e-7** (float32 tolerance 1e-5).
- One-iteration checkpoint resume completed at `runs/crawl-resume-probe-001`, exit 0; lineage SHA-256 matches the original checkpoint and another 512-sample export check passes (maximum error **7.153e-7**).
- Rendered exported evaluation passes execution across crawl/left/right: `runs/evaluation-probe-002`, seed 2000, one five-second diagnostic per start, exit 0. New `--probe` mode keeps acceptance thresholds and cannot qualify incomplete groups. Default evaluation still runs all fifteen full-duration trials. These are development seeds, not held-out qualification.
- **No crawling or recovery milestone achieved.** Five-second net forward displacement is 13.0 / 9.0 / 9.9 cm, but final upright cosine is -0.684 / -0.009 / -0.021: all fallen. Contact sheets and videos were reviewed; movement is not sufficient evidence of acceptable crawling.
- Fixed a real evaluator bug exposed on the second trial: Isaac caches created under `torch.inference_mode()` could not be mutated by a subsequent reset outside that scope. `torch.no_grad()` preserves gradient-free inference and mutable simulator state. Failed `evaluation-probe-001` is retained; it needed a forced stop after the exception (exit 137). Environment/video cleanup now runs on failure too. A deliberately malformed five-action export fails visibly with exit 1 after environment creation. Missing-asset training and missing-bundle evaluation also return 1 with tracebacks.
- Laptop agent reports preliminary replay passed: 200 synthetic static-home samples, finite outputs within joint/slew limits. This is synthetic evidence, not real sensor/actuator timing or hardware readiness.
- Laptop agent relayed the user's updated hardware choice: **Xsens MTi-1s-DEV with native Xbus USB serial**, replacing the earlier BNO085/RP2040 plan. The observation contract is unchanged. Laptop owns transport and sensor documentation updates (#11); old calibration is unverified and must be preserved. Simulation noise/mass/drag remain provisional until measured.
- Source is now a Git repository; laptop imported/pushed baseline `1bc9c56`. GPU commits only its owned files locally; laptop handles pushes/tracking. Continue append-only coordination; do not commit local communication, original notes, assets, experiment outputs or credentials.
- Next GPU work: Phase 3 (#6), first resolve/review airborne reset settling and reward effects, contact support and posture margin. Record a bounded experiment before increasing training. Physics/protected-point measurements and acceptable gait remain open; no physical motion is enabled by these exports.
- Evidence logs and inspection report: `runs/inspection-20261014T075415Z/`. The runtime image remains the Phase 1 rebuild below; evaluation fixes run from the bind-mounted source and will be included in the next image rebuild.


### Phase 1 complete; Phase 2 execution probe starting: 2026-10-14

- Continuation plan: `docs/le-snail-forward-spec.md`. Coordination with the laptop agent is append-only in `agents_communication.txt`; GPU owns simulation/training, laptop owns deployment and GitHub tracking. Issues #1–#10 were created by the laptop agent; #3 is inspection, #4 applied-action correction, #5 pipeline probe.
- Inspection report: `runs/inspection-20261014T075415Z/inspection-report.md`. Baseline source archive and asset hashes retained under `baseline/`; three pose videos, collision overlays and joint traces under `visual-final/`. All 22 rendered-inspection checks pass. A subsequent checks-only run (`boundary-after/`) passes 19 checks including exact timeout and pre-reset terminal snapshot.
- Fixed two reproduced bugs in `src/le_snail/sim/env.py`: actor history now uses the actual delayed commanded target, and the 120-second timeout no longer occurs at 119.95 seconds. Red/green evidence retained. Rendered evaluation now saves terminal images before auto-reset; this last render path awaits the exported-policy evaluation probe.
- Floating model: 6 joints, 7 bodies, 0.632006 kg; 17 authored convex-decomposition collision shapes. Upright settles on base+gripper; both side starts remain fallen. Raw starts are airborne; evaluator excludes two seconds of settling. Cooked hull fidelity, protected points, physical parameters and meaningful crawling remain unvalidated. All flags/configs/assets remain unchanged.
- Rebuilt image after source changes: `sha256:22a27b381002feeb590bc746c6f3157375b9e8f578ac08f3b1dbb851898e2b71`. Prior working image retained as `le-snail:baseline-20261014`. Three-pose smoke passes again; portable suite is 51 tests and 4 subtests. Logs: `rebuild.log`, `smoke-after.log`, `portable-final.log` in the inspection evidence directory.
- GO only for the planned 64-environment, 20-iteration crawl execution probe, seed 42, fresh output `runs/crawl-probe-001`. Training now records source and checkpoint lineage and checks exported CPU inference against training CUDA inference including normalization. Those new training paths await this probe. Require checkpoint resume and rendered exported evaluation before extending the training budget.


### Successful rebuild and smoke test: 2026-10-14, 07:37 UTC

This status supersedes the historical access failures below.

- The session permission profile changed to unrestricted. Docker and NVIDIA device access now work: the process has the host `docker` group, Docker Server responds, and `nvidia-smi` sees the RTX 6000 Ada with driver 580.173.02. No host driver or socket-permission changes were needed.
- Rebuilt the existing Dockerfile successfully, including the flatdict fix and all six Isaac Lab package checks. Final image `le-snail:isaac-2.3.2`: **`sha256:00004264b9b940782ef515b1252ff866f33152a7c1d7c57e0a13f4f625bcf10e`**. Isaac Lab remains at commit `37ddf626871758333d6ed89cf64ad702aef127d0` (v2.3.2; its core Python distribution reports 0.54.2); RSL-RL is 3.0.1.
- `from isaaclab.app import AppLauncher`, flatdict import, and CUDA availability/device-name checks passed inside the container. Asset preparation reused the recorded source; the prepared URDF hash remains unchanged.
- **Smoke test passed with exit code 0:** four CUDA environments, 100 control steps each from upright, left and right resets, observations shaped `(4, 24)`, finite observations and rewards. The environment's floating-base and six-joint-order guards also passed. This is execution validation, not a crawl or contact-fidelity result.
- **Container portable tests: 51 passed, 4 subtests passed.**
- Updated only `scripts/smoke_sim.py` runtime diagnostics: flush completion messages/output before Kit fast shutdown, print exceptions before cleanup, and request Kit exit code 1 on failure. A deliberate missing-asset test retained its traceback and returned exit code 1. The original script is backed up in the evidence directory.
- Diagnostic history: the first two smoke runs exited 0 without completion messages because buffered output was lost. A full-shutdown diagnostic run printed all three successful checks but segfaulted during cleanup. The final script retains Isaac's default fast shutdown; the final run both printed all completion markers and exited 0.
- Requirements, policy/physics configs, simulation environment implementation, Dockerfile, assets and `robin.txt` were preserved. No measurement/contact-validation flags were changed, and no training was started.
- **Next:** inspect rendered upright/side poses, collision fidelity, actuator response and tether force before the initial PPO probe. Protected points and hardware parameters remain unmeasured. Headless GLFW/no-viewport warnings and a fixed-link inertia-merge deprecation warning remain in the logs; investigate their physical relevance during geometry review.
- Evidence directory: `.setup/logs/resume-access-20261014-jFlmsL/`. It contains both rebuild logs, launcher/asset logs, container test results, smoke attempts and final log, deliberate failure log/exit code, final Docker state, copied Isaac Lab internal logs, upstream source snapshots, original script/environment backups, checksums and the previous handoff. The stopped diagnostic containers `le-snail-smoke-20261014-002` and `le-snail-smoke-20261014-final` are retained for further inspection.

### Resume attempt: 2026-10-14, 07:31 UTC

- Re-read `AGENTS.md`, this handoff, README, domain glossary, accepted design and simulation validation instructions. Attempted the requested rebuild with the existing Dockerfile; Docker exited **1** before building because access to `/var/run/docker.sock` is denied.
- This session also exposes no `/dev/nvidia*` devices, and `nvidia-smi` cannot communicate with the driver. Managed session permissions prohibit sandbox escalation; Docker/GPU access must be provided by the session or the direct commands below run from a host terminal.
- Launcher validation and the smoke test remain **not run**. No training was started. The last portable-test result remains the 51-pass result recorded below; it was not repeated in this attempt.
- Existing source, Dockerfile, configs, assets and `robin.txt` were preserved. Checksums for the prepared URDF, Dockerfile, policy/physics configs and `robin.txt` were recorded and verified unchanged. Only this runtime note and new evidence files were added.
- Evidence: `.setup/logs/resume-20261014-ntJjCp/` contains `rebuild.log`, `session-checks.log`, `preserved-files.sha256`, and `HANDOFF.before.md`. Resume with the rebuild once Docker/GPU access is available.

### Resume attempt: 2026-10-14, 07:28 UTC

- Read the project instructions, README, accepted design and simulation checks. The latest Dockerfile fix is present, but the rebuild could not start: this agent session receives `permission denied` connecting to `/var/run/docker.sock`.
- This session also has no usable NVIDIA device nodes; `nvidia-smi` cannot communicate with the driver. The host driver version remains readable as 580.173.02. This is evidence of session access limits, not evidence that the host driver has failed.
- The session's managed permissions do not allow sandbox escalation. Continue the direct commands below from a host terminal or a session with Docker/GPU access. Launcher import and the smoke test remain **not run** in this attempt; the failed baseline image is still unvalidated.
- Portable tests passed again: **51 passed**, using `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q -c /dev/null tests`. The default Conda `python3` lacks pytest; system Python has pytest 6.2.5, so this is not a check of the declared dev dependency environment.
- The prepared URDF SHA-256 still matches `224e2afa2c0528140717bdf530b2f3334daaf87e7072bf4b913d0ab1ce3f2984`. Assets, manifest, requirements, source code, Dockerfile and `robin.txt` were left unchanged. This directory is still not a Git checkout.
- Evidence and the previous handoff are preserved in `.setup/logs/resume-20261014-SPKpno/`: `rebuild.log`, `session-checks.log`, `portable-tests.log`, and `HANDOFF.before.md`.

### Recorded host preflight and prior image failure

- Remote preflight passed: Ubuntu 24.04, NVIDIA RTX 6000 Ada (49,140 MiB reported VRAM), driver 580.173.02, 123 GiB RAM, approximately 446 GiB free SSD at preflight, Docker 29.6.2. A GPU-enabled Docker container successfully ran `nvidia-smi`.
- Isaac Sim **5.1.0**, Isaac Lab **v2.3.2**, and RSL-RL **3.0.1** are the chosen stack. Isaac Lab resolved to `37ddf626871758333d6ed89cf64ad702aef127d0` in the build log.
- The existing `le-snail:isaac-2.3.2` image built successfully but is **incomplete**. Its ID was `sha256:a0888721adc8c6032515643efcd693a071bf0c7a81e3288a3f641c185e524d9e`. Treat that image as the failed baseline, not a validated environment.
- Robot asset fetch succeeded. `assets/so101/snail.urdf` and meshes exist remotely. Source: TheRobotStudio/SO-ARM100 commit `a758567c3978dfeefe282ede0500085a48fe8f78`. Prepared URDF SHA-256: `224e2afa2c0528140717bdf530b2f3334daaf87e7072bf4b913d0ab1ce3f2984`. The downloaded URDF already had collisions; none needed restoration. Contacts remain unvalidated.
- Latest smoke-test failure: `scripts/smoke_sim.py:4`, `from isaaclab.app import AppLauncher`, **`ModuleNotFoundError: No module named 'isaaclab'`**. Simulation never launched.
- Root-cause evidence is in the preceding build: installing `isaaclab` failed while building `flatdict==4.0.1`, with **`ModuleNotFoundError: No module named 'pkg_resources'`**. Pip's isolated build used newer setuptools that removed `pkg_resources`. The upstream installer uses `find ... -exec` and continued after that package failure, allowing Docker to export an incomplete image.

### Dependency fixes (now verified by the successful rebuild above)

`docker/Dockerfile` now:

- Retains the earlier proven fix `TERM=xterm` for Isaac Lab's unconditional `tabs 4`. The original image's `ansi+tabs` terminal definition was unavailable.
- Installs `setuptools==78.1.1` and `wheel==0.45.1`, then preinstalls **only flatdict** with `--no-build-isolation --use-pep517`. Other dependencies retain normal build isolation.
- Checks that the root `isaaclab` module imports and all six Isaac Lab distributions/modules are installed before the image can be exported. This check does not launch Kit or need a GPU.

`scripts/setup_remote.py` now also checks `from isaaclab.app import AppLauncher` in the accepted-license container before fetching assets or running the smoke test. This script is a **laptop-to-remote transfer helper**; on the GPU host use the direct commands below instead.

Validation already performed on the laptop:

- Real Python 3.11 / pip 24.3.1 container reproduction: isolated `flatdict==4.0.1` build failed with the exact `pkg_resources` error despite setuptools 78.1.1 being installed in the outer environment.
- The same reproduction passed with `--no-build-isolation`; the built wheel installed and a real `FlatDict` operation passed.
- Earlier terminal-type reproduction failed with `ansi+tabs` and passed with `xterm`.
- **51 portable tests pass**. These do not validate Isaac APIs, GPU physics, firmware or physical motion.

Full logs copied here for the new agent:

- `.setup/handoff-logs/setup-20261014T070340Z.log`: original terminal-type failure.
- `.setup/handoff-logs/setup-20261014T071539Z.log`: image export succeeded but the core install failed internally; search `pkg_resources` and inspect its preceding `flatdict` build.
- `.setup/handoff-logs/setup-20261014T071942Z.log`: successful asset fetch, then missing `isaaclab` import.

## Next commands: run directly on the GPU host

Use a terminal/session that can stay alive during the build; `tmux` is suitable if installed.
These commands stop on a failed step and retain output in the remote workspace:

```bash
cd /home/clutterbot/robin/le_snail_ws
set -euo pipefail
mkdir -p .setup/logs
docker build --progress=plain -t le-snail:isaac-2.3.2 -f docker/Dockerfile . \
  2>&1 | tee .setup/logs/rebuild.log

export ACCEPT_EULA=Y
bash scripts/container.sh -c \
  'from isaaclab.app import AppLauncher; import flatdict; print("launcher import passed")' \
  2>&1 | tee .setup/logs/launcher-import.log

bash scripts/container.sh scripts/fetch_asset.py \
  2>&1 | tee .setup/logs/asset.log
bash scripts/container.sh scripts/smoke_sim.py --headless --device cuda:0 \
  2>&1 | tee .setup/logs/smoke.log
```

The asset fetch reuses the recorded source revision. Preserve the downloaded asset and manifest.
Keep actual import/build errors visible; the previous build's zero exit code was insufficient evidence.

Portable checks can also run inside the built image:

```bash
bash scripts/container.sh -m pytest -q tests
```

After the smoke test and geometry inspection pass, a short initial training probe is:

```bash
bash scripts/container.sh scripts/train.py --headless --device cuda:0 \
  --stage crawl --num-envs 64 --iterations 20 --output runs/crawl-probe-001
```

Use a new output directory for each run. Twenty iterations are an execution probe, not a convergence or locomotion claim. The README describes the crawl → recover → robust progression, evaluation and export.

## Accepted task and constraints

The SO-101 arm itself must crawl like an inchworm/snail. Use the stock printed body initially; base and gripper are the intended supports. Occasional other plastic contact is acceptable, while exposed wiring/electronics must stay clear. Falls are allowed if it recovers and resumes crawling; continuous rolling is not the intended gait.

- Tiles first; tabletop and commercial carpet later.
- Distance trial: **50 cm net forward base movement within 120 seconds**, including recovery, while remaining within **±15 cm lateral displacement**, passing **3/5 trials**.
- Separate left/right recovery trials: regain a stable crawling posture then move **5 cm forward within 30 seconds**, passing **3/5 per side**. Not every upside-down or cable-entangled pose is required.
- Permanent power and USB tether, plus an accepted extra IMU USB connection. Model bounded dissipative drag at the cable attachment; full cable dynamics and snagging are deferred.
- One feedforward PPO policy, six bounded joint-position targets, initial control rate **20 Hz**, physics **200 Hz**.
- Actor observations: six joint offsets from home, six filtered finite-difference joint velocities, three base-frame gravity components, three gyro rates, six previous applied actions (**24 total**). Global position/velocity/contact truth is not a deployed policy input; position can measure reward and evaluation progress.
- Existing arm is assembled/calibrated; user confirms **7.4 V Feetech servos**. Motor dynamics, sustained limits and body/gripper mappings still need physical measurement.
- Current IMU choice (supersedes the original BNO085/RP2040 plan): Xsens MTi-1s-DEV over native Xbus USB serial, as relayed by the laptop agent from the user on 2026-10-14. Base-frame gravity and gyro policy inputs remain unchanged; transport, mounting and physical validation are pending.
- Physical policy execution belongs on the Ubuntu 22 laptop with RTX 4050; the training GPU is not in the live control loop.
- Turning and 8×8 IR obstacle sensing remain later scope. Original two-day simulation/four-day hardware targets were feasibility targets, not achieved milestones or guarantees.

## Code, validity and collaboration notes

The source includes the Isaac environment, staged PPO training, evaluation/video/export, laptop deployment checks and IMU firmware. None of the simulation, firmware or hardware paths has yet demonstrated a successful crawl.

- `configs/policy.json` and `configs/sim.json` contain **provisional** poses, gains, friction, torque/velocity limits and noise/drag values. All measurement/contact flags remain false; `protected_points` is empty. Establish evidence before setting those flags.
- `docs/simulation.md` describes geometry and physics validation; `docs/hardware.md` describes calibration, timing and IMU integration. In particular, LeRobot's gripper native range differs from its body-joint angle units and needs an explicit mapping.
- An evaluation numerical pass is not sufficient to mark the milestone complete: video must demonstrate an acceptable gait and contacts. The current evaluator intentionally leaves milestone approval pending that review.
- GitHub tracking: `https://github.com/robintomar1/le_snail`. Read the tracker instructions linked by `AGENTS.md` before issue operations. Source was transferred as files and **has not been pushed**; verify Git state before committing or pushing. The laptop workspace was not a valid Git checkout.
- Treat this GPU directory as the authoritative simulation workspace after handoff. Avoid editing the laptop copy in parallel. Preserve `robin.txt`, assets, logs and any new experiments.
- The laptop's `remote_gpu/` was an SSHFS view of this directory through localhost port 2222 → remote SSH port 22. The old agent could edit mounted files but its sandbox blocked SSH sockets. Those access limits do not describe the new GPU-local session: inspect its actual permissions.
- `.setup/` contains ignored transfer snapshots, backups and logs. `.remote-mount/` is laptop-local connection state and was not copied. No passwords, private keys or agent authentication/session stores were transferred.

## Useful upstream references

- [Isaac Lab v2.3.2 installer](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/isaaclab.sh)
- [Isaac Lab core dependencies](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/setup.py)
- [flatdict 4.0.1 legacy build](https://github.com/gmr/flatdict/blob/4.0.1/setup.py)
- [Setuptools removal of pkg_resources](https://setuptools.pypa.io/en/stable/history.html#v82-0-0)
- [SO-101 source assets](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)

Update this handoff's runtime status after the next successful rebuild and smoke test.
