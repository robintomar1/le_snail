# Real-arm runs of two-portion-hybrid-002-500

Generated from the run logs in this folder (scripts/policy_run.py output: `.json` meta, `.npz` per-step and per-bus-cycle data, `.jsonl` replayable inputs). base cos = upright cosine of the base from the IMU (1 = level, 0.83 = the sim crawl posture, < 0 = on its side). Heading change is integrated base-z gyro over the policy steps. Distance travelled was not measured on these runs.

| UTC (2026-10-17) | mode | arm | computer / IMU | policy steps | base cos at step 0 | lowest base cos | heading change (deg, gyro) | peak servo current (A) | lowest V | stop reason |
|---|---|---|---|---|---|---|---|---|---|---|
| 11:55:54 | shadow | arm2 (hackathon, snail_2) | Raspberry Pi / bno085 | 500 | 1.00 | 1.00 | 0 | 0.01 | 10.9 | finished |
| 11:59:57 | live | arm2 (hackathon, snail_2) | Raspberry Pi / bno085 | 188 | -0.26 | -0.26 | 50 | 0.25 | 10.5 | IMU: Stale IMU sample |
| 12:04:36 | live | arm2 (hackathon, snail_2) | Raspberry Pi / bno085 | 40 | 0.64 | 0.64 | -29 | 0.27 | 10.6 | IMU: Stale IMU sample |
| 12:06:44 | live | arm2 (hackathon, snail_2) | Raspberry Pi / bno085 | 0 | – | – | – | 0.06 | 10.6 | IMU: IMU stream failed |
| 13:09:53 | shadow | arm1 (snail_1) | Rubik Pi / xsens | 500 | 1.00 | 1.00 | 0 | 0.01 | 11.0 | finished |
| 13:10:08 | shadow | arm1 (snail_1) | Rubik Pi / xsens | 500 | 1.00 | 1.00 | 0 | 0.01 | 11.2 | finished |
| 13:11:23 | live | arm1 (snail_1) | Rubik Pi / xsens | 0 | – | – | – | 0.10 | 11.0 | control deadline missed: step 0 command 33.9 ms after its slot |
| 13:13:54 | shadow | arm1 (snail_1) | Rubik Pi / xsens | 200 | 0.86 | 0.86 | 0 | 0.01 | 11.0 | finished |
| 13:16:46 | shadow | arm1 (snail_1) | Rubik Pi / xsens | 500 | 0.86 | 0.86 | 1 | 0.01 | 11.5 | finished |
| 13:24:50 | live | arm1 (snail_1) | Rubik Pi / xsens | 600 | 0.78 | -0.12 | -25 | 0.61 | 10.9 | finished |
| 13:25:31 | live | arm1 (snail_1) | Rubik Pi / xsens | 600 | -0.13 | -0.43 | -41 | 0.29 | 11.0 | finished |
| 13:45:40 | live | arm1 (snail_1) | Rubik Pi / xsens | 600 | 0.77 | -0.09 | -109 | 0.57 | 10.8 | finished |
| 13:46:25 | live | arm1 (snail_1) | Rubik Pi / xsens | 600 | 0.77 | 0.64 | -36 | 0.55 | 10.7 | finished |
| 13:47:09 | live | arm1 (snail_1) | Rubik Pi / xsens | 600 | 0.76 | 0.71 | 19 | 0.38 | 10.8 | finished |
| 13:49:00 | live | arm1 (snail_1) | Rubik Pi / xsens | 600 | 0.73 | 0.60 | 24 | 0.47 | 10.8 | finished |
| 14:10:34 | live | arm1 (snail_1) | Rubik Pi / xsens | 600 | 0.76 | 0.75 | 9 | 0.42 | 10.8 | finished |
