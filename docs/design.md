# Le Snail locomotion design

Train an RL policy in simulation to make an SO-101 arm crawl, then reproduce the behavior on physical hardware. Power and USB remain connected during physical operation, with some tether drag expected.

The user confirmed the shared design and implementation defaults. Implementation and run instructions are in the [README](../README.md); target-machine validation and physical measurements remain pending.

## Accepted scope

- Crawling may resemble an inchworm, alternating support between the base and gripper. Sliding and temporary tilting are acceptable. Falls are also acceptable if the robot recovers without assistance and continues crawling. Continuous rolling is an unwanted gait. Initial recovery targets sideways falls encountered while crawling, rather than every possible inverted or tangled posture.
- Start with the existing printed plastic body. Protective parts or replacement printed parts may be added later, but added contact pads are outside the initial configuration.
- The base and gripper are the intended supports. Occasional contact by other plastic body parts is allowed; exposed wiring, connectors, and electronics must stay clear of the floor.
- Target tiles first. Tabletop surfaces and commercial carpet are later targets.
- Milestone one is a 50 cm crawl in simulation within two days. Milestone two is reproducing a 50 cm crawl on hardware within four days. These are target deadlines from the planning session, not validated delivery estimates.
- A successful trial advances the base 50 cm net in the intended forward direction within 120 seconds, including recovery time. The initial demonstration requires at least three successful trials out of five. This threshold does not establish reliable deployment.
- Establish motion without drag first, then require the simulation milestone under varied resisting forces near the tether attachment. This is an initial approximation; cable routing and snagging require separate checks.
- The physical arm is assembled and calibrated, using the Feetech servos supplied with its SO-101 kit. The servos are the **12 V** STS3215 variant (servo sticker checked by the user, 2026-10-14; this corrects an earlier 7.4 V note) on a 12 V supply; exact actuator parameters still need validation against this arm.
- Training compute is an RTX 6000 Ada GPU on a separate Ubuntu 24 machine, with Docker available. Driver, RAM, disk, and container GPU access still require a remote preflight.
- A base-mounted Xsens MTi-1s development board connected over USB is the IMU (2026-10-14; supersedes the earlier BNO085/BNO086 + RP2040 selection). Represent its observations during training; deployed policy inputs must be reproducible on the physical robot.
- Use Isaac Lab 2.3.2 with Isaac Sim 5.1.0 in Docker, subject to the remote machine preflight. Adapt the SO-101 asset for a movable base and valid floor contacts.
- Train one PPO policy for crawling and recovery, using bounded joint-position targets. Start with upright crawling, then introduce sideways falls and drag.
- Evaluate recovery separately from the distance trials. Test representative left- and right-side fallen poses, requiring return to a crawling posture followed by forward movement without assistance. Verify that the stock body can escape those poses before assuming they are learnable.
- Run the trained policy locally on the user's Ubuntu 22 laptop with an RTX 4050, connected directly to the arm and IMU. The remote training GPU is outside the real-time control loop.
- An additional IMU USB cable is acceptable for the prototype. Bundle it loosely with the arm USB and power tether and include it in drag testing.
- The first controller handles forward crawling and recovery. Turning commands and navigation remain later work. Use a marked straight course and video to measure base displacement on hardware.
- Simulator ground-truth position may measure progress and calculate rewards. The deployed policy relies on joint feedback, IMU readings, and previous actions.
- Obstacle sensing is deferred until after crawling transfers to hardware and remains in the intended later scope.
- Feasibility testing may lead to revised contact hardware or gait requirements before committing to full RL training.

## Accepted implementation defaults

- Begin each distance trial at rest in one reproducible, stable folded posture. Keep base lateral displacement within 15 cm of the starting centerline throughout the trial. Recovery tests use a recorded pair of representative left- and right-side poses established during the physical feasibility check.
- For each side, require recovery followed by at least 5 cm net forward movement within 30 seconds in three of five trials. Report left and right results separately. Recovery tests supplement the accepted 50 cm distance trials; success on one test does not replace success on the other.
- Use a small feedforward policy with joint positions, joint velocities estimated from timestamped position differences, base-frame gravity direction, base-frame angular rate, and the previous action. Match filtering and timing between simulation and hardware. Keep global position, true base linear velocity, and contact forces out of deployed policy inputs.
- Control all six actuator channels, including the gripper, with joint-position targets mapped through the verified joint and gripper calibration. Bound positions and target slew rates. Begin with a proposed 20 Hz policy rate, then adjust to measured USB and sensor timing and use the same rate in simulation and deployment.
- Reward net forward progress, with costs for excessive sideways drift, abrupt commands, sustained failure to recover, and contacts with excluded parts. Avoid a repeatable fall/recovery bonus that could make falling more rewarding than crawling. Reward weights are tunable implementation parameters; acceptance tests remain fixed.
- Model realistic actuator limits, response lag, and contact geometry before training. Add variation in floor friction, mass, actuator response, sensor noise and delay, and tether resistance as the crawling policy improves. Establish ranges from measurements or explicitly label them provisional; simulation success alone does not validate those ranges.
- Start with slack cables routed behind the robot, without lifting or pulling it forward. Apply a bounded resisting force near the physical cable entry in simulation. Measure the assembled robot's drag, including sensor wiring, to set the final test range. Treat snags and winding separately from this force approximation.
- Provide an operator stop and stop issuing motion commands on stale feedback or a controller fault. Define the servo stop behavior against the real controller before unsupported motion tests.

## Four day execution plan

1. Day one: verify the remote container/GPU setup; adapt and inspect the movable-base asset, collisions, masses, and actuator behavior; check whether the stock arm can propel itself and escape representative side falls. Obtain and assemble the IMU hardware in parallel. A failed mechanics check triggers a scope or contact-hardware decision.
2. Day two: train and evaluate upright crawling, then recovery and varied drag. Save the policy, environment configuration, seeds, and videos with separate distance and recovery results. If recovery or drag tests fail, report the partial result rather than marking the simulation milestone complete.
3. Day three: integrate the IMU with the laptop, verify coordinate frames and calibration, measure command/feedback timing, and run bounded hardware trials. Adjust the simulation model using observed mismatch and retrain there as needed.
4. Day four: evaluate the unchanged acceptance criteria on tiles and record outcomes. Tabletop, carpet, turning, and obstacle sensing remain subsequent work.

The schedule is a timeboxed feasibility attempt. Both milestones depend on workable floor contacts and actuator behavior; hardware transfer also depends on obtaining and integrating the selected sensor in time.

## Measurements and setup still required

- Remote access, driver/RAM/disk checks, and working container GPU access.
- Exact servo identifiers, calibrated joint limits, gripper mapping, speed and response measurements, and suitable sustained operating limits for the physical arm.
- IMU acquisition, rigid mounting and protection, microcontroller readout, coordinate conventions, and timestamp/latency checks.
- Recorded initial and fallen poses, recovery feasibility, tile friction, and tether force measurements.
- Numerical reward and PPO settings, derived from experiments without silently weakening acceptance criteria.
- The exact 8 by 8 IR sensor and obstacle-avoidance behavior, when that later milestone begins.

## Simulation starting points

NVIDIA's [SO-101 workshop](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/01-overview.html) demonstrates a GR00T manipulation workflow using teleoperation data. It provides reference material for sim-to-real work, but does not establish crawling feasibility.

The hardware designer provides [SO-101 simulation assets](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101). The [MuJoCo model](https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.xml) inspected during planning has a fixed base and missing base collisions. Direct inspection during implementation found collision entries in the current URDF. The implementation therefore validates the downloaded URDF, preserves existing contacts, and restores missing ones only when necessary. Base motion, floor contacts, and actuator behavior still require validation before learned movement is evidence of physical feasibility.

Isaac Lab's [versioned installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html) documents the selected Lab and Sim pairing. Isaac Sim 5.1's [requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) include Ubuntu 24.04 and list the tested Linux driver. Verify the remote machine against these versioned requirements before installation.

## IMU integration

Use the Xsens MTi-1s development board the user already has (decided 2026-10-14). It connects directly to the laptop over USB and streams Xsens Xbus `MTData2` packets, so no intermediate microcontroller or custom firmware is needed. The earlier BNO085/BNO086 + RP2040 selection is superseded; its firmware is retained only as history.

Derive base-frame gravity direction from the device's filtered orientation quaternion (ENU output, heading unused) and use its calibrated rate of turn for angular rate. Raw acceleration is not a substitute for filtered orientation during crawling. Whether this particular board runs an orientation filter must be confirmed on the device (`scripts/xsens_info.py`); if it does not, an attitude estimator becomes a separate decision. Include the sensor, microcontroller, mounting, and accepted additional USB cable in mass and tether measurements.
