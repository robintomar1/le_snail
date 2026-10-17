# diversity-seed101-051-100

An early RL crawl policy from the crawl-only stage (#6), trained from a fresh seed to look for a different gait. It moves a lot but circles, and it has no get-up.

## How it was made (GPU host, 2026-10-14, #6; commits 58fedbd, d1e09d8, 6e132fd)

- **Decision:** robin chose a two-seed "fresh diversity sweep" alongside the recovery stage (2026-10-14 10:14Z): fresh seeds 101 and 202 with the then-current crawl "treatment" reward, a first-50-iteration gate, and continuation only for promising seeds.
- **Training:** RSL-RL PPO, Isaac Lab 2.3.2, stage `crawl` (upright starts only), seed 101, 2048 envs × 64 steps, 8 minibatches, learning rate 3e-4 (adaptive), γ 0.995, λ 0.95, entropy 0.01, initial action noise 0.5, actor and critic MLP 128-128 ELU with observation normalization (`training/run.json`). Iterations 1–50 from scratch, then 51–100 resumed from that checkpoint (the folder name's `051-100`).
- **Contract:** the older **20 Hz** contract (fingerprint 10395fc5…): the same 24 inputs and joint limits as the 50 Hz policies, velocity filter α 0.25, one action-delay step = 50 ms in training.

## Sim results

- Robust dev 4000–4014 with physics v2 (`sim-evaluation/report.json`, `crawl-0.mp4`): crawl 1/5. The four others left the 15 cm lane after 11–18 s; left/right 0/5.
- 3-minute endurance (robust, lane off, 32 envs): path 10.12 m but net only 3.82 m, 0 falls. It keeps moving at ~5–6 cm/s along its path, in circles.

## Real arm

3 live runs and 2 shadow runs on arm 2 (the hackathon arm), Rubik Pi + Xsens, 2026-10-17 (`real-arm/RUNS.md`). All three live runs ran the full 12 s (240 steps at 20 Hz) without a stop; heading changes +16°, −17°, +23°. Distance was not measured.

## Files

| Path | What |
|---|---|
| `exported/` | the bundle the robot runs (`policy.pt` sha256 5065ae94d577…, 20 Hz `policy.json`) |
| `training/model_final.pt` | RSL-RL checkpoint at iteration 100 (actor, critic, optimizer), for continued training |
| `training/run.json`, `recovery-diagnostics.json`, `export-parity.json` | training arguments, diagnostics, export parity |
| `sim-evaluation/` | robust dev evaluation report and one crawl video |
| `real-arm/` | every real-arm run log and the summary table |
