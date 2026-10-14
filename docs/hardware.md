# Laptop and sensor integration

The Ubuntu 22 laptop runs the exported policy locally. The arm uses its existing USB servo adapter and 7.4 V supply; a BNO085/BNO086 breakout and RP2040 add a second USB data connection. Include the sensor, bridge, mount and extra cable in mass and drag measurements.

## Python environment

The adapter imports the current LeRobot `so_follower` implementation. The [inspected LeRobot source](https://github.com/huggingface/lerobot/blob/main/pyproject.toml) requires Python 3.12; keep that separate from Isaac's Python 3.11 and the laptop's system Python. With `uv` installed:

```bash
uv venv --python 3.12 .venv-deploy
source .venv-deploy/bin/activate
git clone --depth 1 https://github.com/huggingface/lerobot.git vendor/lerobot
uv pip install -e 'vendor/lerobot[feetech]' -e '.[deploy]'
git -C vendor/lerobot rev-parse HEAD > vendor/lerobot-revision.txt
python -c 'from lerobot.robots.so_follower.so_follower import SO101Follower; from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig'
```

Keep the resolved LeRobot commit and installed package versions with hardware logs. The exported manifest records the training Torch version; verify a replay on the laptop before live use. This setup has not yet been installed or exercised on the laptop.

## IMU and firmware

Use the breakout manufacturer's wiring and voltage guidance for the specific boards acquired. `firmware/snail_imu/snail_imu.ino` uses the RP2040 Arduino core, its default `Wire` I2C bus, and the Adafruit BNO08x library. Configure the board's I2C pins as needed. Install the library and its declared dependencies, select the actual RP2040 board, compile and upload in Arduino IDE or the equivalent CLI. Firmware compilation remains pending until the board/toolchain is available.

The bridge requests Game Rotation Vector and calibrated gyro reports at 100 Hz and emits newline-delimited JSON over USB serial. Quaternion order is **w,x,y,z**, angular-rate units are rad/s, and `sample_us` is an increasing 64-bit microcontroller timestamp. It reports the older arrival time of the paired reports; validate actual sensor and transport latency rather than treating this timestamp as perfect measurement time.

```json
{"seq": 1, "sample_us": 1234567, "q_wxyz": [1,0,0,0], "gyro_rad_s": [0,0,0]}
```

Mount the sensor rigidly and measure `imu_rotation_base_from_sensor`, the proper 3 by 3 rotation mapping sensor-frame vectors to the robot base. At upright rest, the computed base gravity vector should be near `[0,0,-1]`. Rotate about each base axis and confirm gyro signs; place the robot on each side and confirm gravity directions. Game Rotation Vector avoids a magnetometer but does not supply an absolute heading reference. See the [sensor datasheet](https://www.ceva-ip.com/wp-content/uploads/BNO080_085-Datasheet.pdf) and [library guidance](https://learn.adafruit.com/adafruit-9-dof-orientation-imu-fusion-breakout-bno085/arduino).

## Joint mapping

Copy `configs/hardware.example.json` to `configs/hardware.local.json`. Enter the existing LeRobot calibration directory and robot ID; the controller refuses to generate or overwrite calibration automatically.

For every joint, determine the affine mapping:

```text
URDF radians = radians_per_native_unit * LeRobot value + offset_rad
```

With `use_degrees=True`, the five body joints use degrees, but **the gripper still uses 0–100**, as shown in the [LeRobot follower source](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py). Measure corresponding positions in the model and on the arm; confirm direction, offset, range, and at least a third point to check the mapping. Fill all coefficients and native limits. A degrees-to-radians conversion alone is insufficient for the gripper and does not establish body-joint zero offsets.

The template intentionally contains null mapping values. Copy the exact `contract_fingerprint` from the exported manifest after verifying all channels. Check the stop behavior and physical power cutoff with the robot supported. LeRobot configures servos on connection, including PID and acceleration parameters; measure the behavior under that configuration. The numerical temperature, voltage, and tracking thresholds in the template are starting bounds to validate, not manufacturer-approved operating limits.

`Present_Voltage` is read as a raw register; `voltage_v_per_register_unit` explicitly converts it before range checking. Verify the conversion and temperature units against the actual firmware and a meter. Mark `calibration_verified` and `telemetry_register_units_verified` only after recording the checks. These flags do not override a mismatched policy contract or provisional training physics.

## Timing and first movement

First inspect stationary and manually moved sensor/joint readings. Measure the full USB feedback, inference and command cycle. The initial rate is 20 Hz; if the cycle cannot meet it consistently, change the contract and train with the new timing. The loop rejects stale samples, delayed cycles, and excessive tracking error before sending the next command. This is a software check, not an independent hardware watchdog.

Place the robot in the verified starting posture, support it for the first bounded motion check, and keep the physical power cutoff accessible. Live mode starts targets at measured joint positions and requires proximity to the trained home posture. Interrupting the script attempts to disable torque; loss of the USB connection can prevent that attempt from reaching the servos.

Start with a short run using the README command, inspect the log and behavior, then extend trials on tiles. Keep power and both USB cables slack behind the robot. Use a marked lane and video to measure forward/lateral base displacement. Joint/IMU logs alone cannot establish the 50 cm hardware result.

Recorded observation JSONL for offline policy replay uses:

```json
{"time_s": 0.0, "joint_rad": [0,-0.6,1.1,-0.5,0,0.35], "gravity_base": [0,0,-1], "gyro_base_rad_s": [0,0,0]}
```

This is a format example, not measured hardware data. Replay never opens hardware. Preserve failed trials and compare identical settings in simulation before changing training parameters.
