# Sim endurance comparison 002 (2026-10-17)

`scripts/endurance.py` on the GPU host: 32 robust-physics envs per bundle, 180 s, lane limit off; a run ends only after 30 s lying fallen. `50hz.*` and `20hz.*` hold the per-env tracks (sampled every 1 s) and the summary.

| policy | path | net | falls |
|---|---|---|---|
| gait-clone-fast-001 | 11.25 m | 10.90 m (7.3–12.7) | 0 |
| hybrid-001 | 6.26 | 6.13 | 0 |
| gait-clone-001 | 6.29 | 5.96 | 0 |
| B-clean-drag | 2.80 | 2.77 | rare |
| two-portion-hybrid-002-500 | 8.48 | 0.40 | 0 |
| two-portion-hybrid-004-550 | 7.59 | 0.55 | 2.2 per run, 2.1 recovered |
| two-portion-hybrid-004-600 | 8.20 | 0.43 | 0.3 per run, 0.3 recovered |
| diversity-seed101-051-100 (20 Hz) | 10.12 | 3.82 | 0 |
| recover-shaped-051-100 (20 Hz) | 8.12 | 0.73 | |
| handover-truncation-001-050 (20 Hz) | 5.93 | 1.38 | 53 % end fallen |
| handover-reward-001-050 (20 Hz) | 5.55 | 0.56 | 47 % end fallen |
| scale-treatment-051-100 (20 Hz) | 7.13 | 0.34 | |
