# Training curves worksheet: answer key

Answers to the [worksheet](training-curves-worksheet.md). Outcomes (evaluation counts, stop decisions) come from `HANDOFF.md` and [`docs/research/reward-design-study.md`](../research/reward-design-study.md). Where an answer is an explanation the logs don't prove, it says so.

---

## Exercise 1: `crawl-supported-001`

1. **When forward progress starts.** `Reward/progress` is slightly negative for iterations 1–4. It turns positive at about 5 and climbs fastest between 6 and 15 (0.004 → 0.018). That panel is the direct measure; `Train/mean_reward` follows it.
2. **The episode-length hump.** Two different effects:
   - **Rising half:** only *finished* episodes are averaged. At iteration *n* no episode can be longer than about (*n* + 1) × 64 steps; at iteration 14 that cap is 960 and the measured value is 877. Early on the robots barely move, so few of them reach the lane edge, and the only episodes finishing are long ones. The average grows with elapsed time, the same survivorship effect as after a resume.
   - **Falling half:** from about iteration 15 the robots crawl, drift sideways and leave the lane, so episodes now end at about 300 steps.
3. **Improving after 60?** Barely.
   - `Reward/progress` hovers between 0.025 and 0.032.
   - `Train/mean_reward` drifts from about 5.3 to 6–7.
   - Episode length is flat at about 300.
   - Only noise std keeps rising (0.64 → 0.82), which says the signal is weak, not that the policy is improving.
4. **Reward per step and episode duration.** 6.3 ÷ 317 ≈ **0.020 per step**. 317 steps ≈ **16 s**, against a 120 s limit, so lane exits end almost every episode.
5. **Straight crawling?** No. Nothing in the reward charges for turning, and episodes end at about 16 s by lane exit. The policy is never trained on a long, straight episode. Outcome: "gait appears, turns". The next experiment added a yaw cost (exercise 2).

---

## Exercise 2: scale-control vs scale-treatment

1. **Is the treatment worse?** Not on this evidence. The two runs optimise **different reward functions**, so their totals aren't comparable. The treatment's total includes a new term of about −0.04 per step; over about 300 steps that is −12, roughly the whole gap (control ≈ +9.5, treatment ≈ −2). Compare them on what they share: `Reward/progress`, `Reward/lateral`, episode length, and above all the evaluation.
2. **Faster crawler.** The treatment. Progress per step over the last five iterations is about 0.048 vs 0.040; the treatment kept rising in its second chunk while the control stayed flat.
3. **What the treatment learns.** The penalty shrank to less than half its worst value, so base yaw error fell. The policy can't see heading, but it sees the gyro, so it can learn to keep its turning rate lower and more balanced. The evaluation adds a twist: net yaw rate was −10.8°/s for the treatment vs −11.5°/s for the control, nearly the same. The treatment's passes came mainly from **crawling faster**, reaching 50 cm before drifting out of the lane. This gait pivots its base, so base yaw and direction of travel are partly decoupled, and the `1 − cos(yaw)` term partly charges the gait itself (research note, section 3.2).
4. **Noise std.** The control's grows faster (1.40 vs 1.14 at the end).
   - *Inferred:* random actions produce random turning, and in the treatment turning now costs reward. That gives the reward gradient a reason to pull std down, partly offsetting the entropy bonus. The control has no such cost.
5. **Learning rate at the ceiling.** Each update changes the policy less than the KL target, even at the maximum step size. With std near or above 1, clipping and the rate limit absorb most changes to the network output, so the measured KL stays small (research note, section 3.4).
6. **Evaluation (seeds 3000–3014).** Treatment crawl **3/5**, control **0/5**; both 0/5 on side starts. This is one matched pair with one seed: good evidence that the yaw term helped this lineage, not proof that it always helps (exercise 3 shows how much seeds vary).

---

## Exercise 3: seed 101 vs seed 202

1. **Difference at 98.** Progress 0.093 vs 0.058 per step (seed 101 is about 60 % faster). Yaw penalty about −0.002 vs −0.004.
2. **Yaw ten times larger.** Seed 202's gait turned far more, so its base heading wandered much further from the start. Evaluation: net yaw rate +2.4°/s for seed 101 vs +8.7°/s for seed 202. The gaits are different families, too: seed 101 found a fast small-amplitude stepping gait (3–6 Hz), seed 202 a slower one (1.4–2 Hz).
3. **Shorter episodes.** More turning means reaching the lane edge sooner, and the lane exit ends the episode. Shorter episodes then mean lower totals, independent of speed.
4. **The early peak.** The fresh-start survivorship effect from exercise 1. Early averages include only finished episodes, which are capped by elapsed time; the peak collapses once the robots start crawling and leaving the lane.
5. **Value loss.** Seed 101's is higher (about 2.7 vs 1.3), but that doesn't make it the worse run. Value loss is not normalised, so it scales with the size of the returns. Seed 101 collects about three times more reward per episode and survives longer, so its targets are larger and its prediction errors larger in absolute terms. The lane-exit blindness from notes section 6 applies to both.
6. **One seed is not enough.** You would have rejected a recipe that, with seed 101, produced the faster stepping-gait family and the policy later run on arm 2 (development evaluation on seeds 3300–3314: crawl 3/5 vs 0/5). Seed-to-seed spread here is as large as the effect of a reward change. Comparing two configs with one seed each can't separate the config from luck. Use matched seeds and more than one of them, or treat a single comparison as a hint.

---

## Exercise 4: `two-portion-50hz-001-100`

1. **Stable get-ups.** Zero on both sides, in every iteration. About 21,000 side-start episodes produced no get-up; the only upright-fall recovery was at iteration 0.
2. **Forward and reward after 40.** Flat. `Reward/forward` has stayed at about 0.006 per step since iterations 30–40, and `Train/mean_reward` has moved between 4.3 and 7 around 5.5 since about 50.
3. **Std rising with the learning rate at its ceiling.** Notes section 5 applies exactly: the reward signal is too weak to sharpen the policy, and updates are as large as allowed yet change little. Nothing in the reward is pulling the policy towards a get-up.
4. **Episode length.** 1300 steps ÷ 50 ≈ **26 s**. The average mixes two very different populations:
   - upright starts, which crawl and exit the lane;
   - side starts, which lie there until a time limit ends them.

   A shift in the proportion of each changes the average with no change in either behaviour. Read the per-population counters, not the mixed mean.
5. **Rule verdict.** Get-ups were flat at 0 and forward was flat, so the rule says **STOP**, and the run was stopped there. Evaluation on seeds 5300–5314: crawl 0/5, left 0/5, right 0/5. Crawl trials stayed upright but turned about 210° and left the lane at about 10.5 s, having gone about 24 cm forward.
6. **What changed next.** If a get-up never happens, PPO has nothing to reinforce. The next design (exercise 5) used a scripted teacher, the hybrid controller, which drives the first part of half the episodes. The policy then starts from states that are already partway through a get-up, where success is reachable by exploration.

---

## Exercise 5: the hybrid lineage

1. **Resume lines (iterations 100, 199, 298 and 396).** At each one, episode length drops to about 250 and climbs back, and value loss dips close to 0 before returning to its level. Both are the restart effects from notes section 7. `Reward/forward` and `getup_potential` also spike briefly, because every robot is at its start pose at once.
2. **When get-ups appear.**
   - Left get-ups start at about iteration **386** and reach about 0.4 per step by 396.
   - Right get-ups start at about **416** and reach about 0.6 by 431.
   - Over the same stretch, `Reward/getup_potential` jumps from about 0.7 to 0.94, `Train/mean_reward` steps from about 6 to 9, and `Reward/forward` from 0.008 to 0.012.

   Policies often learn this way: a long flat stretch, then a fast rise once the first successes feed the update.
3. **The 3.7 spike.** Almost certainly a resume artifact. It occurs only in the first iteration after the resume and the curve returns to its trend at once. The same jump opens both runs resumed from 002-500 in exercise 6: 3.6 for 003 and 4.1 for 004-600. As always, ignore the first iterations after a resume.
4. **Early hints.** Only weakly on this dashboard: `getup_potential` creeps from about 0.62 to 0.69 over iterations 100–390. The real leading indicator was a counter not drawn here. Handover-start get-ups, from robots the teacher had left partway up, rose from left 42 % / right 53 % after chunk 2 to 95 % / 87 % after chunk 3, while get-ups from the official side-lying start were still 0. The pre-registered review watched those counters, which is why the run continued. The lesson: decide in advance which leading indicator will justify continuing, and log it.
5. **Does it break the std rule?** Yes, as it was first stated. Std rose the whole time and the policy still learned get-ups, which passed a noise-free evaluation. The version that survives:

   > Rising std means the reward's pull on std is weak compared with the entropy bonus, largely because clipping and the rate limit hide std from the reward. It is a warning about training dynamics, not proof that the network output learned nothing. Steadily falling std with rising reward is the healthy pattern. The verdict comes from a noise-free evaluation.

6. **High forward reward, 1/5 crawl.** `Reward/forward` counts motion per step, averaged over everything that happens in training, including the noise. The evaluation pass needs 50 cm *inside the 15 cm lane* from the noise-free policy. All chunk-5 crawls moved well but turned between −107° and −157° into the lane edge. Training reward measures what the reward pays for, and this reward didn't pay for heading.
7. **Why it didn't get up on the real arm.** These curves can't tell you. They only describe the simulator, where the get-up works. The gap is **sim-to-real**: actuator strength and delay, friction, the mass and tether, the IMU frame and the start pose. Compare the real-arm logs (`policies/two-portion-hybrid-002-500/real-arm/`) with the sim evaluation traces, and use the actuator model study (#12).

---

## Exercise 6: 003 vs 004

1. **Upright falls.**
   - **003:** jumps from about 0.4 to 1.6 per step in the first quarter and stays high through the middle, easing to about 0.6 by the end.
   - **004:** stays near 0.1–0.6 and ends at its lowest. By the stop rule, 003 exceeded twice the 002-500 baseline in 6 of 10 windows, so it was **stopped**.
2. **Stable get-ups.**
   - **003:** left get-ups collapse from about 0.56 to 0.04 per step, and right ones fall to about 0.3–0.4. Measured per window, official-start left get-ups fell from about 1,100 to 12–57.
   - **004:** both sides stay around 0.7–0.8. 004 kept the skill; 003 lost it.
3. **Is 3000 steps good news?** No. Over the same stretch `Reward/forward` is at its lowest (about 0.0036, down from 0.011). The robot crawls less, so it reaches the lane edge later. Episodes got longer because the policy **did less**. Episode length only tells you how long until a termination, not whether anything useful happened.
4. **Comparing −3 with +1.** No. 003's total includes a new heading penalty, and 004 changes how forward motion is paid, so the two totals measure different things. Compare the counters they share (falls, get-ups, forward) and the evaluation.
5. **Which to keep, and the evaluations** (crawl / left / right, out of 5):

   | Run | Result |
   |---|---|
   | 002-500 (parent) | 1/5/5 |
   | 003-600 | **2/0/1**, stopped |
   | 004-550 | 2/5/4 |
   | 004-600 | 2/5/5 |

   004 is clearly the better direction. Even so, it wasn't a deployment candidate: its crawls still left the lane, at 42–49 cm, short of the 50 cm pass.
6. **Penalty vs scaled pay (inferred).** A penalty that applies only in some states can be dodged by leaving those states: 003 charged for heading only *while crawling*, so the policy could avoid it by crawling less or by not being upright, and it did both. Scaling down a payment can never make crawling worse than not crawling, so it doesn't give the policy a reason to abandon the activity.
