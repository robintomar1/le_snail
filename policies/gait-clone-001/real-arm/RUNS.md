# Real-arm runs of gait-clone-001

Generated from the run logs in this folder (scripts/policy_run.py output: `.json` meta, `.npz` per-step and per-bus-cycle data, `.jsonl` replayable inputs). base cos = upright cosine of the base from the IMU (1 = level, 0.83 = the sim crawl posture, < 0 = on its side). Heading change is integrated base-z gyro over the policy steps. Distance travelled was not measured on these runs.

| UTC (2026-10-17) | mode | arm | computer / IMU | policy steps | base cos at step 0 | lowest base cos | heading change (deg, gyro) | peak servo current (A) | lowest V | stop reason |
|---|---|---|---|---|---|---|---|---|---|---|
| 14:17:49 | shadow | arm1 (snail_1) | Rubik Pi / xsens | 300 | 0.84 | 0.84 | 0 | 0.01 | 11.4 | finished |
| 14:19:13 | live | arm1 (snail_1) | Rubik Pi / xsens | 0 | – | – | – | 0.06 | 10.9 | base not upright after the ramp (gravity in base frame [0.73, -0.09, -0.68], tilt 47 deg); policy not started |
| 14:19:37 | live | arm1 (snail_1) | Rubik Pi / xsens | 0 | – | – | – | 0.05 | 10.9 | base not upright after the ramp (gravity in base frame [0.73, -0.09, -0.68], tilt 47 deg); policy not started |
| 14:20:20 | live | arm1 (snail_1) | Rubik Pi / xsens | 600 | 0.75 | 0.71 | 3 | 0.16 | 10.9 | finished |
| 14:25:52 | live | arm1 (snail_1) | Rubik Pi / xsens | 600 | 0.72 | 0.64 | 8 | 0.31 | 11.0 | finished |
| 14:26:24 | live | arm1 (snail_1) | Rubik Pi / xsens | 0 | – | – | – | 0.16 | 10.9 | base not upright after the ramp (gravity in base frame [0.75, -0.1, -0.65], tilt 49 deg); policy not started |
| 14:26:43 | live | arm1 (snail_1) | Rubik Pi / xsens | 0 | – | – | – | 0.10 | 11.1 | base not upright after the ramp (gravity in base frame [0.75, -0.1, -0.65], tilt 49 deg); policy not started |
| 14:28:42 | live | arm1 (snail_1) | Rubik Pi / xsens | 1500 | 0.71 | -0.15 | 13 | 0.30 | 11.0 | finished |
| 14:29:27 | live | arm1 (snail_1) | Rubik Pi / xsens | 0 | – | – | – | 0.12 | 10.9 | base not upright after the ramp (gravity in base frame [0.78, -0.11, -0.61], tilt 52 deg); policy not started |
| 14:30:05 | live | arm1 (snail_1) | Rubik Pi / xsens | 0 | – | – | – | 0.10 | 11.2 | base not upright after the ramp (gravity in base frame [0.72, -0.04, -0.7], tilt 46 deg); policy not started |
| 14:31:27 | live | arm1 (snail_1) | Rubik Pi / xsens | 1500 | 0.72 | -0.25 | 110 | 0.31 | 10.9 | finished |
| 14:33:45 | live | arm1 (snail_1) | Rubik Pi / xsens | 1500 | 0.73 | -0.20 | -52 | 0.35 | 4.3 | 6 consecutive cycles lost a reply |
| 14:43:57 | shadow | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 200 | 0.96 | 0.96 | -0 | 0.01 | 10.7 | finished |
| 14:44:32 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 0 | – | – | – | 0.08 | 10.5 | base not upright after the ramp (gravity in base frame [0.68, -0.21, -0.7], tilt 46 deg); policy not started |
| 14:45:07 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 390 | 0.72 | 0.64 | 4 | 0.27 | 10.5 | no fresh complete servo state for step 390 |
| 14:46:04 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 94 | 0.73 | 0.65 | 4 | 0.18 | 10.7 | control deadline missed: step 94 command 13.0 ms after its slot |
| 14:46:35 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 1 | 0.72 | 0.72 | – | 0.09 | 10.4 | control deadline missed: step 1 command 13.2 ms after its slot |
| 14:47:23 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 600 | 0.73 | 0.65 | 7 | 0.21 | 10.4 | finished |
| 14:47:49 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 493 | 0.72 | 0.64 | 5 | 0.20 | 10.4 | no fresh complete servo state for step 493 |
| 14:48:26 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 542 | 0.72 | 0.64 | -0 | 0.22 | 10.6 | no fresh complete servo state for step 542 |
| 14:49:26 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 600 | 0.72 | -0.25 | -48 | 0.19 | 10.7 | finished |
