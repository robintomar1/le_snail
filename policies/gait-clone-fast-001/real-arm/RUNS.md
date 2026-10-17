# Real-arm runs of gait-clone-fast-001

Generated from the run logs in this folder (scripts/policy_run.py output: `.json` meta, `.npz` per-step and per-bus-cycle data, `.jsonl` replayable inputs). base cos = upright cosine of the base from the IMU (1 = level, 0.83 = the sim crawl posture, < 0 = on its side). Heading change is integrated base-z gyro over the policy steps. Distance travelled was not measured on these runs.

| UTC (2026-10-17) | mode | arm | computer / IMU | policy steps | base cos at step 0 | lowest base cos | heading change (deg, gyro) | peak servo current (A) | lowest V | stop reason |
|---|---|---|---|---|---|---|---|---|---|---|
| 14:59:00 | shadow | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 200 | 0.49 | 0.49 | 0 | 0.01 | 11.1 | finished |
| 15:00:04 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 360 | 0.71 | 0.27 | 0 | 0.19 | 10.6 | no fresh complete servo state for step 360 |
| 15:01:29 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 600 | 0.71 | 0.30 | 5 | 0.20 | 10.6 | finished |
| 15:02:41 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 600 | 0.70 | 0.35 | -0 | 0.30 | 10.5 | finished |
| 15:03:22 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 600 | 0.71 | 0.30 | -11 | 0.19 | 10.5 | finished |
| 15:05:00 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 600 | 0.71 | -0.01 | -75 | 0.23 | 10.4 | finished |
| 15:06:09 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 1337 | 0.72 | 0.29 | -12 | 0.25 | 10.5 | no fresh complete servo state for step 1337 |
