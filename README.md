# Le Snail

SO-101 crawling and recovery with reinforcement learning in Isaac Lab, followed by local policy execution on a tethered physical arm. The accepted task is 50 cm of forward base displacement on tiles within 120 seconds, including recovery, in three of five trials. Left and right recovery are tested separately.

The implementation includes asset preparation, three PPO training stages, policy export, evaluation, an IMU bridge, and a laptop controller. **No successful crawl or hardware transfer has been demonstrated yet.** Isaac, Docker builds, firmware, and hardware execution require validation on the target machines. The bundled physics values and initial poses are provisional.

See [the design](docs/design.md), [simulation checks](docs/simulation.md), and [hardware integration](docs/hardware.md).

## Local development

Python 3.10 or later is sufficient for the portable package and tests. Isaac runs in its own Python 3.11 container; the current LeRobot hardware adapter uses a separate Python 3.12 environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

For a machine with an older system pytest and ROS plugins, the equivalent isolated test invocation is:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -c /dev/null tests
```

## Remote RTX 6000 Ada setup

For mounting the remote workspace through the local TCP tunnel, see
[remote workspace setup and cleanup](docs/remote-workspace.md).

Copy this workspace to the remote Ubuntu 24 machine. These files have not been pushed to the GitHub repository; cloning that repository alone will not retrieve this implementation.

```bash
bash scripts/remote_preflight.sh
docker build -t le-snail:isaac-2.3.2 -f docker/Dockerfile .
```

Review the [Isaac Sim license terms](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html) before accepting them for container execution:

```bash
export ACCEPT_EULA=Y
bash scripts/container.sh scripts/fetch_asset.py
bash scripts/container.sh scripts/smoke_sim.py --headless --device cuda:0
```

Asset preparation resolves the designer's source to a commit and records URDF/mesh hashes. Subsequent runs reuse that revision. To choose a known revision, provide `--revision COMMIT` on the first fetch. `--source /path/to/SO-ARM100` also supports an existing clean checkout. Downloaded assets retain the upstream license.

The importer uses a movable base, existing collisions, and restored visual geometry only where collision geometry is missing. A successful smoke test checks API execution and finite state, **not** contact fidelity or locomotion feasibility. Inspect and adjust the poses and contact model before training; see [simulation checks](docs/simulation.md).

## Train and export

Each stage uses a new output directory. Start small, inspect results, then increase environment count and iterations. The iteration counts below are starting budgets, not estimates of convergence.

```bash
bash scripts/container.sh scripts/train.py --headless --device cuda:0 \
  --stage crawl --num-envs 256 --iterations 1000 --output runs/crawl

bash scripts/container.sh scripts/train.py --headless --device cuda:0 \
  --stage recover --num-envs 256 --iterations 1000 \
  --resume runs/crawl/model_final.pt --output runs/recover

bash scripts/container.sh scripts/train.py --headless --device cuda:0 \
  --stage robust --num-envs 256 --iterations 1000 \
  --resume runs/recover/model_final.pt --output runs/robust
```

`crawl` starts upright without tether resistance; `recover` adds side starts; `robust` adds drag, friction/mass/gain variation, action delay, and sensor noise. Ordinary falls do not immediately end episodes. Leaving the lane, protected-point floor contact, invalid state, or remaining fallen too long ends an episode.

Each run saves the contract, physics configuration, source revision, asset hashes, training settings, checkpoints, and an `exported/` bundle. The bundle contains the TorchScript actor including its observation normalizer, its interface contract, checksums, installed versions, and physics-validation status. Resume rejects changes to the policy contract; start a new run after changing observation/action semantics.

## Evaluate

```bash
bash scripts/container.sh scripts/evaluate.py --headless --device cuda:0 \
  --bundle runs/robust/exported --video --output runs/evaluation
```

Evaluation runs five distance trials and five trials from each side, using held-out seeds. It writes video, time-series JSONL, and `report.json`. A distance trial requires 50 cm net forward base displacement within 120 seconds, within a 15 cm lateral boundary. Recovery requires stable upright posture followed by 5 cm forward displacement within 30 seconds, with three of five successes on each side.

`passed` is the numerical test result. `milestone_candidate` additionally requires validated physics and a robust-stage policy. `milestone_passed` remains false until gait/contact videos have been reviewed; continuous rolling or contact with unmodeled vulnerable parts must not be passed merely because displacement increased. Keep the review alongside the report. Provisional parameters never qualify as validated physics.

## Deploy

Copy the entire `runs/robust/exported/` directory to the Ubuntu 22 laptop. Follow [hardware integration](docs/hardware.md) to acquire the IMU, flash the RP2040, map joints and frames, and verify timing. The training machine is not part of the live control loop.

Offline replay opens no hardware:

```bash
python scripts/deploy.py --bundle runs/robust/exported \
  --replay recorded_observations.jsonl --output runs/replay_targets.jsonl
```

Real motor commands require explicit live mode and a verified hardware configuration:

```bash
python scripts/deploy.py --bundle runs/robust/exported \
  --hardware configs/hardware.local.json --live --seconds 5 \
  --output runs/first_hardware_trial.jsonl
```

The runner refuses provisional physics, mismatched contracts, unfilled calibration, stale IMU data, excessive joint tracking error, and missed control deadlines. It disables torque on normal shutdown and on caught failures when communication is available. A physical power cutoff is still needed for communication loss or a process/machine failure.

## File layout

- `src/le_snail/contract.py`: ordered observations, units, joint limits, and target slew limits.
- `src/le_snail/sim/env.py`: Isaac environment, reset modes, reward, contact probes, and tether force.
- `src/le_snail/evaluation.py`: distance and recovery acceptance rules.
- `src/le_snail/hardware.py`: gripper/body mapping and timestamped IMU transport.
- `configs/`: policy interface, provisional physics, and an unfilled hardware template.
- `firmware/snail_imu/`: BNO08x reports over RP2040 USB serial.
- `scripts/`: asset fetch, simulation smoke test, training, evaluation, and deployment.
