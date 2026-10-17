# gait-clone-fast-001

A faster version of [gait-clone-001](../gait-clone-001/): the same cloning method, with a teacher gait tuned for speed after robin asked for "another version of it which is faster". No get-up.

## How it was made (GPU host, 2026-10-17, #14 #15; commit 075498e)

- **Gait search:** sweeps over the hybrid-001 gait family (robust and nominal physics) picked "A-fast"; a sideways slide turned out to be set by the shoulder_pan centre, and a fine sweep chose pan centre +0.02 rad (robust, 128 envs, 40 s):

  | pan offset | +0.01 | **+0.02** | +0.03 | +0.04 | ≥ +0.05 |
  |---|---|---|---|---|---|
  | lateral at the end (cm) | +7.0 | **−2.6** | −9.6 | −13.8 | −15 |
  | survive 40 s in the lane | 70 % | **74 %** | 52 % | 22 % | 0 % |

  Result: `configs/controllers/fast-gait-001.json`, 6.3 cm/s in robust physics (50 cm in 8.9 s; the D gait does 3.6 cm/s), upright fraction 0.51 (D: 0.70). The lane exits in the 40 s test came after ~2 m.
- **Teacher:** `PhaseGait` on that gait (`runs/phase-gait-fast-001`): robust dev 5640–5654 crawl 5/5, 50 cm in 8.7–9.9 s.
- **Clone:** DAgger as for gait-clone-001 (`training/dagger.json` has the exact arguments: 1024 envs, 750 steps per round, β = 1, 0.5, 0.25, 0, 0, 0, 4000 updates, batch 8192, seed 7, robust physics-v2-50hz-001). Frozen normalizer + MLP 24-128-128-6 ELU, contract `policy-50hz-001`.

## Sim results

- Robust dev 5660–5674 (`sim-evaluation/`): **crawl 5/5**, 50 cm in **9.2–11.2 s**, final yaw −10..+18°, max lateral ≤ 2 cm, lowest upright cosine 0.52–0.55 (it rocks deeper than the D gait). No speed flags.
- 3-minute endurance (robust, lane off, 32 envs): **path 11.25 m, net 10.90 m** (7.3–12.7 m), 0 falls: the farthest of every policy compared (`../sim-endurance-002/`).

## Real arm

6 live runs and 1 shadow run on arm 2 (the hackathon arm), Rubik Pi + Xsens, 2026-10-17 (`real-arm/RUNS.md`): 4 ran the full 12 s with heading changes of +5°, 0°, −11° and −75°; 2 stopped on lost servo replies, one of them after 27 s of a longer run. Distance was not measured.

## Files

| Path | What |
|---|---|
| `exported/` | the bundle the robot runs (`policy.pt` sha256 235ffe1d50ec…) |
| `training/student.pt`, `training/dagger.json` | DAgger student weights and the per-round log |
| `sim-evaluation/` | robust dev evaluation report and one crawl video |
| `real-arm/` | every real-arm run log and the summary table |
