# Simulation validation

The Isaac implementation targets Lab 2.3.2, Sim 5.1.0 and RSL-RL 3.0.1. The rebuilt container passed launcher/CUDA validation and the upright/left/right finite-state smoke test on the RTX 6000 Ada on 2026-10-14; see `HANDOFF.md` for the exact image and logs. Contact geometry, actuator behavior and tether force still require inspection before training. Run the smoke test again after relevant changes.

## Asset and contacts

The designer's [SO-101 URDF](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf) includes base collision entries in the inspected source. The earlier missing-base-collision warning applied to the inspected MuJoCo model and its README. Asset preparation examines the actual downloaded URDF rather than assuming both formats match.

Inspect the converted asset with self-collision enabled. Confirm that the base is free to translate and rotate, that no meshes intersect at rest, and that plastic supports make contact at their visible surfaces. Convex decomposition is an approximation; it can close cavities or change traction. Inspect upright, left-side and right-side starts. The provisional side spawn height is only a way to initialize simulation, not a hardware drop-test instruction.

The default folded posture in `configs/policy.json` has not been checked against the physical arm. Choose a stable feasible posture and update the contract before meaningful training. Verify imported joint axes, limits, home angles, and the gripper's angular range. Source URDF effort/velocity values are not measured servo capabilities.

Place protected-point probes on exposed wiring, connector, and electronics locations that must stay off the floor. Each entry in `configs/sim.json` has this shape:

```json
{"body": "base_link", "position_m": [0.0, 0.0, 0.05], "radius_m": 0.005}
```

That is a format example, not a measured location. Probe coordinates use the named link's frame. The environment ends a trial when a probe sphere intersects the floor. These probes do not detect tether snags or every possible self-contact; review videos and the complete CAD alongside them.

## Actuators and randomization

The initial 0.8 Nm effort limit, 12 Nm/rad stiffness, 0.5 Nm s/rad damping, 2 rad/s simulated velocity limit, friction ranges, and 0.05–0.3 N drag range are **provisional experimental values**. They are not measured SO-101 limits or continuous torque ratings. Replace them using the actual 7.4 V arm's behavior before drawing sim-to-real conclusions.

Measure unloaded and loaded joint response to small position changes. Fit response time, saturation, deadband, and delay. Confirm that a position-drive approximation is adequate; if it is not, replace it with an identified actuator model. Check the gripper separately because its controller protection and leverage differ from the other joints. Match hardware configuration, including PID and acceleration settings, to these measurements.

The robust stage varies material friction, mass and corresponding inertia, drive gains, zero/one-step action lag, encoder noise, gravity-direction noise, angular-rate noise, and tether resistance. It currently approximates sensor imperfections with noise, not a full asynchronous sensor-delay model. Measured timing may require adding observation latency before final training.

Tether force opposes the planar velocity at the attachment point, with a bounded magnitude. Torque accounts for its offset from the base center of mass. This model cannot pull the robot forward or explicitly represent slack loops, winding, or snagging. Measure actual resistance with all three cables and use the same routing in hardware trials.

## Policy and reward

The actor sees 24 values: six joint offsets from home, six filtered finite-difference joint velocities, three unit gravity components in the base frame, three angular rates in rad/s, and six previous applied actions. Joint order and units are in the exported contract. No global position, true base velocity, or contact force is fed to the deployed actor.

Actions are six normalized values mapped to position targets, clipped to joint limits, and rate-limited. The initial policy period is 50 ms and physics period 5 ms. The reward favors net forward base progress and charges for lateral drift, tilt, abrupt commands, elapsed time, and protected contact. Reward is not an acceptance metric; inspect for exploitative rolling, sliding, or collision behavior.

## Evidence before the milestone

Save the source revision, measured values, stable start and fallen poses, and videos showing contact behavior. Set the corresponding validation fields only after these checks:

- `policy.json`: measured home, angular limits and action timing; set `parameters_measured`.
- `sim.json`: identified parameters, inspected contacts, and measured protected-point locations; set `parameters_measured`, `contact_geometry_validated`, and `protected_points_validated`.

These fields summarize recorded evidence; changing a flag is not a substitute for doing the checks. Train/export again after relevant changes. Evaluate the exact export and preserve all fifteen trial outcomes, including failures. Review videos for an acceptable gait and contact behavior before recording a completed milestone.
