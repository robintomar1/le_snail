# two-portion-hybrid-002-500

The Track B RL policy (#15) at the end of its 500-iteration run: it gets up from either side in sim, but its crawl turns out of the lane.

## How it was made (GPU host, 2026-10-16/17, #15)

- **Experiment:** `configs/experiments/two-portion-hybrid-002.json` (identical to two-portion-hybrid-001 except that robin dropped the heading penalty): contract `policy-50hz-001`, robust stage with mass randomization 0.9–1.4 and distal mass 0.7–1.6.
- **Reward ("two-portion", `src/le_snail/rewards.py`):** a get-up potential (kernel-regressed progress along the successful scripted get-ups) plus a standing term, and forward pay for new ground gained while upright and crawling; effort counts only while upright.
- **Teacher prefixes:** in 50 % of episodes the hybrid-001 scripted controller drives the first part of the episode and hands over to the learner at a random point (a reverse curriculum: success moved from late to early handovers, then to the official side starts).
- **Training:** RSL-RL PPO, 8192 envs, 5 chunks of 100 iterations with a pre-registered review after each (`HANDOFF.md`, comms log). Chunk 1 resumed `two-portion-hybrid-001-100`; this bundle is the final chunk (iterations 396–495).
- **Review highlights:** official-start get-ups appeared at chunk 4 (left) and reached ~100 % for right by iteration 436; action noise std rose to 2.92 by the end.

## Sim results

- Robust dev 5480–5494 (`sim-evaluation/report.json`, videos `crawl-0.mp4`, `left-0.mp4`, `right-0.mp4`): crawl 1/5, **left 5/5, right 5/5**. Every crawl turned 107–157° clockwise into the lane edge.
- Nominal physics (deployment reference): crawl 0/5 (turns), left 5/5 (up in 2.9 s), right 5/5 (4.5–5.9 s).
- 3-minute endurance (robust, lane off): path 8.48 m, net 0.40 m: it keeps moving in tight circles.

## Real arm (2026-10-17, `real-arm/RUNS.md`)

- **arm 2 (hackathon arm) on the Raspberry Pi 5 with a BNO085 IMU, 3 live runs:** all stopped by the IMU. The BNO085 on bit-banged I2C dropped reports and then locked up while the servos worked. The one get-up attempt (robot on its side) ran 3.8 s: the policy's first actions matched the sim's for the same start, and the base rocked but did not rise before the stop.
- **arm 1 (snail_1) on the Rubik Pi with the Xsens, 8 live runs:** 1 stopped before step 0 (a CPU-pinning issue, since fixed); 7 ran the full 12 s. In one of them (13:25:31 UTC) the base was already on its side when the policy started, and the policy ran its get-up for 12 s without rising (best upright cosine 0.37); robin reported it kept falling. In the other six the base started upright (cosine 0.73–0.78), with heading changes of −109° to +24°.
- Shadow runs reproduced the sim's observation layout and timing; the policy outputs on the Pi match the sim's within 2e-6.

## Files

| Path | What |
|---|---|
| `exported/` | the bundle the robot runs (`policy.pt` sha256 9ee1846eaead…) |
| `training/model_final.pt` | RSL-RL checkpoint at iteration 495 (actor, critic, optimizer), for continued training (two-portion-hybrid-003 and -004 resumed it) |
| `training/*.json` | handover per-bin get-up rates, action and handover diagnostics, export parity |
| `sim-evaluation/` | robust dev evaluation report and crawl, left and right videos |
| `real-arm/` | every real-arm run log and the summary table |
