# Real-arm runs of diversity-seed101-051-100

Generated from the run logs in this folder (scripts/policy_run.py output: `.json` meta, `.npz` per-step and per-bus-cycle data, `.jsonl` replayable inputs). base cos = upright cosine of the base from the IMU (1 = level, 0.83 = the sim crawl posture, < 0 = on its side). Heading change is integrated base-z gyro over the policy steps. Distance travelled was not measured on these runs.

| UTC (2026-10-17) | mode | arm | computer / IMU | policy steps | base cos at step 0 | lowest base cos | heading change (deg, gyro) | peak servo current (A) | lowest V | stop reason |
|---|---|---|---|---|---|---|---|---|---|---|
| 15:13:54 | shadow | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 100 | 0.45 | 0.45 | 0 | 0.01 | 11.1 | finished |
| 15:14:02 | shadow | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 100 | 0.45 | 0.45 | 0 | 0.02 | 10.7 | finished |
| 15:16:11 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 240 | 0.74 | 0.74 | 16 | 0.43 | 10.5 | finished |
| 15:16:48 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 240 | 0.75 | -0.06 | -17 | 0.62 | 10.5 | finished |
| 15:17:47 | live | arm2 (hackathon, snail_2) | Rubik Pi / xsens | 240 | 0.74 | 0.74 | 23 | 0.42 | 10.6 | finished |
