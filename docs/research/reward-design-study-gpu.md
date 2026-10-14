# Independent GPU study: reward, information and recovery exploration

Date: 2026-10-14. Status: independent report frozen after both GPU return audits, before reading the laptop reward study. This study was written without reading `docs/research/reward-design-study.md`, the laptop reward study, or messages reporting its findings. No training, reward, physics, policy contract or acceptance criteria were changed for this report.

The completed audits favor a bounded demonstration-assisted exploration test before increasing environment count or adding more reward terms. Successful scripted recovery plus the current crawl policy can beat passive holding under the existing reward, while the learned policy never discovers stable recovery and incurs heavily clipped, slew-limited actions. It does **not** establish that PPO has converged or that more training cannot help. Crawling already works on one five-trial development set; recovery has never appeared in the two measured recovery-training chunks despite scripted simulation paths existing at the same actuator limits. A successful script establishes existence of a simulated path, not its discoverability by the present policy or reproducibility on hardware.

## Evidence and scope

The accepted task is one 24-input, six-output feedforward policy for crawling and recovery. Global position, heading and true base velocity remain absent from actor inputs. Actuator limits stay at 0.8 N·m effort, 2 rad/s actual velocity and 1 rad/s target slew. Acceptance remains 50 cm within 120 s and a ±15 cm lane, plus separate left/right stable upright recovery and 5 cm forward within 30 s; each requires three of five trials. Contacts, wiring clearance and hardware parameters remain unvalidated. [Accepted design](../design.md), [environment](../../src/le_snail/sim/env.py), [current physics](../../configs/experiments/recover-upright-progress-001.json).

| Local observation | What it supports | What it does not establish |
| --- | --- | --- |
| Current shaped candidate: crawl 5/5, left 0/5, right 0/5 on development seeds 3400–3414 | Preserve a useful crawling checkpoint; recovery is unresolved | Reliable deployment, qualification or an improvement attributable to shaping |
| Original recovery 50: 9,691 completed side episodes, zero stable upright; shaped next 50: 9,829, also zero | No discovered stable recovery across 19,520 completed side episodes, all ending at the fallen limit | An independent-trial statistical bound or convergence; these are correlated samples from changing policies |
| Each recovery chunk: 4,096 × 64 × 50 = 13,107,200 transitions | Recovery has substantial simulation sampling but only 100 new PPO collection/update iterations across both chunks (each contains multiple gradient steps) | That 100 iterations is sufficient for this task |
| Confirmed scripted paths at unchanged limits; stronger and weaker sides | Exploration has a physically feasible simulation target | Broad feasibility of arbitrary actions or a hardware torque margin |
| Intermediate static bank: left 0/13, right 0/8 valid | The tested cosine 0.5–0.7 holding-state bank is unavailable under its agreed gates | That every possible static bank is impossible, or permission to relax the gates |

The specific seed101 progression is recorded in [review50](../../runs/recovery-review-050/diversity-comparison.json), [review100 diagnostic](../../runs/diversity-review-100/comparison.json) and [fresh full evaluation](../../runs/evaluation-diversity-seed101-100-development-3300/report.json).

Evidence: [evaluation](../../runs/evaluation-recover-shaped-100-development-3400/report.json), [unshaped counters](../../runs/recover-t100-001-050/recovery-diagnostics.json), [shaped counters](../../runs/recover-shaped-051-100/recovery-diagnostics.json), [script confirmation](../../runs/feasibility-side-recovery-20261014T111801Z/results.json), [bank audit](../../runs/recovery-bank-validation-001/report.json).

The CPU curve extraction independently finds that the shaped chunk's mean Gaussian action standard deviation increases from 1.497 to 1.853 between its first and last ten iterations. Its average command-rate reward is −0.00548 to −0.00529 per control transition, close to the six-channel slew-saturation maximum magnitude of 0.006. Upright shaping averages only 4.22e−5 to 3.06e−5 per transition across the whole mixed population. These aggregate values motivate action-clipping and side-specific audits; they do not identify the cause of failure. Gaussian entropy can increase while the actual clipped, slew-limited actuator sequences remain unhelpful. Fresh seed101 also improved from a lane-exit diagnostic at 28.38 cm after50 iterations to a 50.17 cm pass after100, then passed3/5 fresh full crawl trials. That is direct evidence that more updates helped that crawl lineage, without proving that every lineage will improve. The curve extraction shows continued changing progress/noise in several runs, so a universal “already converged” claim is unsupported. [Curve evidence and extraction](../../runs/reward-study-gpu-curves-001/curves.json).

## The reward actually optimized

Let Δt = 0.05 s, u = upright cosine, y = base lateral displacement from reset, ψ = base yaw minus reset yaw, Δx = one-step forward displacement, and d = (new bounded target − old target)/Δt. Define g(u) = clip((u − 0.5)/0.3, 0, 1). Apart from protected-contact cost, the current per-transition reward is exactly:

```text
r = 30 [min(Δx, 0) + g(u) max(Δx, 0)]
    − Δt [0.1(1 − u) + 0.3 y² + 0.02 Σ_j d_j² + 0.02
           + (1 − cos ψ)]
    − 1[protected contact]
    + C (b_next − b),                       C = 1
p(u) = 1[side-start] clip((u − u_reset)/(0.8 − u_reset), 0, 1)
b_next = max(b, p(u)),                     b_reset = 0
```

The implementation clamps the posture denominator away from zero. The original side-start poses make it positive. Shaping is eligible only for side-start episodes, and its lifetime sum is at most one; falling again does not refill it. The training return has no explicit 50 cm completion reward or recovery-plus-5 cm reward. Stable upright is recorded as a diagnostic, while training normally continues beyond an evaluation success. [Reward and termination implementation](../../src/le_snail/sim/env.py).

Useful dimensional checks, derived from this formula:

- At the ±15 cm lane boundary the direct lateral cost is only 0.0003375 per transition, versus yaw cost 0.05 at a quarter-turn and 0.1 at a half-turn. Lane failure also terminates the episode; that discontinuity matters more than comparing these immediate costs alone.
- Six simultaneously slew-limited channels cost 0.006 per transition. This penalizes bounded target velocity, not jerk; reducing it does not bypass the hard slew limit, but could worsen actuator chatter.
- Staying at u ≈ −0.1 without movement costs approximately 0.0065 per transition before yaw and command costs. Thirty seconds accumulates approximately −3.9 undiscounted. A full upright-progress cap is therefore neither automatically dominant nor automatically negligible.
- Positive x movement below u = 0.5 receives zero progress reward, while backward displacement is always penalized. A useful recovery can therefore require unrewarded motion and temporary negative return before upright crawling pays. The completed audit below confirms an early return barrier; it does not establish a global objective contradiction.
- Euler yaw about a nearly side-lying body need not correspond intuitively to useful crawl heading. The actual scripted transition trace must decide whether yaw cost penalizes the righting maneuver materially.

The decisive audit is to compare current-component discounted returns of successful scripted recoveries, holding home and the learned policy over common horizons, including a separately reported maneuver prefix. If a successful maneuver already dominates holding over a matched horizon, increasing its bonus is not the first justified change. If it loses only on the short prefix, bootstrap quality or coherent exploration is more directly implicated than the long-run objective.

## Time horizon and information

The runner uses 64 transitions per environment, γ = 0.995, λ = 0.95 and five PPO learning epochs. The 64-transition rollout spans 3.2 s; the episode does **not** reset at each PPO iteration. GAE uses the recursion `A_t = δ_t + γλ A_(t+1)` with value bootstrapping. [Local runner](../../scripts/train.py), [pinned RSL-RL return implementation](https://github.com/leggedrobotics/rsl_rl/blob/v3.0.1/rsl_rl/storage/rollout_storage.py).

Derived at 20 Hz, the reward discount has a 9.97 s e-fold horizon and 6.91 s half-life. γ^600 ≈ 0.0494 at the 30 s recovery deadline; γ^2400 ≈ 5.96e−6 at 120 s. The direct GAE weighting of a TD residual has a 0.888 s e-fold horizon: (γλ)^60 ≈ 0.0341 at 3 s and (γλ)^80 ≈ 0.0111 at 4 s. This is **not** a hard policy memory or learning horizon. A good value estimate can transmit later benefits through short TD residuals; a poor one makes multi-second recovery sequences difficult to credit. GAE deliberately trades variance against bias from approximate value estimates. [GAE paper, Sections 2–3](https://arxiv.org/html/1506.02438).

Both networks currently receive the same 24 observations. Relative yaw, lateral displacement, base linear velocity, elapsed/fallen time, side-start eligibility and spent running-max budget affect returns but are missing from those inputs. This produces plausible value aliasing. It does not prove those features are the dominant bottleneck. Nor can a privileged critic give the deployed feedforward actor missing information: two identical actor observations that demand different steering actions remain indistinguishable to that actor.

A concrete, bounded critic-only proposal is `V(o_24, z_14)`, with:

```text
z_14 = [sin ψ, cos ψ, y/0.15, v_world_xyz,
        base_height, fallen_time/30, episode_time/120,
        one_hot(initial_side: upright,left,right), u_reset, b]
```

This 38-input critic is an empirical representation candidate, **not a claim to contain the full Markov state**. It omits, for example, physical joint velocities, detailed contact state and simulator history. The existing actor, its normalizer and action standard deviations transfer unchanged; use fresh critics and fresh optimizers in both symmetric and privileged comparison arms. The earlier 32-input CPU transfer check demonstrates a tooling path, not learning efficacy. [Local feasibility assessment](../../runs/critic-feasibility-001/assessment.json).

Asymmetric RL theory warns that simply substituting a latent-state critic for a history-based critic under partial observability can introduce bias. A critic conditioned on both observed information and relevant latent state is the safer design direction; neither that theory nor the proposed 38 features guarantees success here. [Baisero and Amato, 2022](https://arxiv.org/html/2105.11674).

## Shaping, curricula and demonstrations

The current running-max reward prevents repeated rise/fall farming **within one episode**, but it changes the training objective and creates hidden reward memory. Actual resets replenish the cap, so termination/reset incentives still require checking. Its boundedness is useful; it is not the policy-invariance theorem for discounted potential shaping.

For a bounded potential Φ, the discount-consistent alternative is `F = γ Φ(s_next) − Φ(s)`, with correct terminal handling. The discounted sum telescopes to `−Φ(s_0) + γ^T Φ(s_T)`; the terminal term must vanish for a truly terminal episode. Potential shaping can redistribute credit while preserving the underlying objective under its assumptions; it cannot repair a base objective that prefers the wrong outcome. [Ng, Harada and Russell, 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

An exact candidate for a later single-variable ablation is to **replace**, not stack with, the current cap:

```text
Φ(s) = 1[side-start] clip((u − u_reset)/(0.8 − u_reset), 0, 1)
F_t  = 0.995 Φ(s_(t+1)) − Φ(s_t)
Φ(true absorbing terminal) = 0
```

Do not clip negative F or keep only positive differences. Do not use the auto-reset observation as the previous episode's terminal state. A rollout boundary is not a terminal. A bootstrapped time limit needs a consistent continuing-task convention, including the value transformation; setting Φ to zero only because a 64-step batch ends is wrong. Audit the pinned RSL timeout adapter before implementing this proposal: its PPO path adds a timeout value correction. The existing fallen-limit event is a termination, distinct from the 120 s truncation. [Pinned PPO timeout code](https://github.com/leggedrobotics/rsl_rl/blob/v3.0.1/rsl_rl/algorithms/ppo.py).

This replacement is lower priority than diagnosing successful traces and exploration. It changes the objective relative to the current capped-reward baseline, and the current monotonic upright potential may still be uninformative while a necessary preparatory movement lowers upright cosine. Replacing it should not be described as a guaranteed fix.

Reverse curricula train on reachable states near successful outcomes and expand towards harder initial states; their effectiveness depends on meaningful state initialization and sampling starts of intermediate difficulty. They do not require a physically dynamic transition to become a static equilibrium. [Florensa et al., 2017](https://proceedings.mlr.press/v78/florensa17a/florensa17a.pdf). Reference-state initialization in DeepMimic similarly exposes promising intermediate states, but that method's actor state, reference information and early-termination choices differ from this project. Its fall termination must not be copied into a task whose goal explicitly includes getting up. [Peng et al., 2018, Sections 6.1–6.2](https://arxiv.org/pdf/1804.02717).

The failed static bank should remain failed. A lower-band static audit is a separate hypothesis with an unknown yield. A more direct alternative is a **scripted prefix followed by learner control in the same live simulation**: begin at the official fallen pose, execute a verified prefix, then hand over before stable upright. This preserves the physical and controller history without pretending that zeroing velocities reproduces a moving state. It requires a new explicitly scoped experiment; it is not a workaround for the rejected static bank.

For that alternative, score teacher contribution separately, mask prefix transitions out of PPO policy loss, retain the actual observation/filter/applied-target history, and never record the script's recovery as learned success. Do not reset the shaping budget at handover. Compare holding the last command against learner continuation; passive completion cannot count as evidence that the learner solved the maneuver. Official evaluation always starts from the original pose with no script. Retain 65% original upright resets; a proposed recovery-pool split is 75% prefix handovers and 25% original side starts, equal left/right, but the prefix bank must first pass reproducible active-continuation and policy-versus-hold diagnostics.

A demonstration-loss alternative avoids intermediate resets altogether. Record exact actor observations and normalized desired actions from successful scripts, preserving the action clip/slew path, and retain an equal-sized upright crawl-distillation set. An explicit candidate actor loss is:

```text
L_actor = −L_PPO_clip − 0.01 H
          + 0.1 E_Dside mean_j[(mu_theta(o)_j − a_script_j)^2]
          + 0.1 E_Dcrawl mean_j[(mu_theta(o)_j − mu_frozen(o)_j)^2]
```

Use balanced left/right demonstrations; keep the PPO rollout on-policy. Expert actions are supervised targets, not fabricated on-policy PPO samples. Check near-identical 24-input observations for incompatible required actions before training: scripted phase is not an actor input, and the student may not represent a phase-dependent expert. The coefficients above are preregistered pilot values, not measured optimal weights. This is a distinct ablation from prefix curricula, not something to combine with critic, potential and horizon changes in one run.

## Does more parallel training solve this?

Massive parallelism can reduce wall time and improve data diversity, but batch size, rollout length and update count jointly determine learning behavior. Rudin et al. explicitly report a task-dependent tradeoff and failure with overly short per-environment trajectories; their quadruped results are not a recommended universal environment count for this arm. [Rudin et al., 2022, Sections 2.2 and 4.1](https://arxiv.org/html/2109.11978).

Here 4,096 × 64 already gives 262,144 transitions per iteration. Increasing environment count with fixed rollout length increases samples per update, not update count or per-environment sequence duration. Keeping total batch fixed while increasing environments would shorten the sequence, which is unattractive for this recovery hypothesis. More iterations can still help. The defensible distinction is between **insufficient successful exploration**, **an objective/credit barrier**, and **insufficient optimization**, not between “RL works” and “RL does not work.”

The cheap experiments below intentionally use at most 256 environments; they diagnose directional learning and mechanisms, not asymptotic superiority to the 4,096-env baseline. A 100-iteration 256 × 64 pilot has only 1,638,400 transitions, one eighth of a previous 50-iteration large chunk. Report both transitions and PPO iterations; never compare equal iteration counts as equal sample budgets.

## Prioritized bounded experiment menu

All entries are proposals. No new ablation is authorized or launched by this report. Maintain a ledger of the sum of owned GPU-process elapsed time, including diagnostics, startup and evaluation; hard overall ceiling **120 GPU-minutes**, with **256 environments and 100 new PPO iterations maximum per training arm**. Stop an unfinished arm when its time allocation expires and preserve its checkpoint. Do not use reserved qualification seeds 9000–9014.

1. **Reward/exploration audit first, at most 64 environments and 20 GPU-minutes.** Matched left/right official starts; compare hold, deterministic current policy, stochastic policy and confirmed scripted paths under the same physics. Save pre-reset component traces, γ-discounted prefix and matched-horizon returns, maximum upright, stable duration, net x/y, applied commands, raw clipping and slew-saturation rates. Record successful-script count rather than assuming all replays succeed. CPU calculations and existing artifacts cost no GPU time. Stop on state/action mismatch or inability to reproduce the positive control; resolve that before PPO.
2. **One matched control/treatment pair, 256 environments per arm, at most 100 iterations each and 25 GPU-minutes per arm.** Choose exactly one treatment after the audit: demonstrated-action loss if successful paths outscore idle but remain undiscovered; critic38 if value aliasing is supported and successful exploration exists; or a separately preregistered reward correction if the current objective ranks successful recovery below failure. Reuse the same frozen starting actor, seed, 65/35 reset mix, 64-step rollout, four minibatches, γ/λ and unmodified mechanics. For a critic comparison reset both critics/optimizers; other comparisons use identical restart semantics. Review at50 on the original starts; extend to100 only if at least one of five trials on either side reaches stable upright and crawl remains≥3/5, with video excluding continuous rolling. Continue both matched arms if this pair-level discovery gate passes. A larger shaping total or a momentary upright peak alone is insufficient.
3. **Matched exported-policy evaluation, at most 25 GPU-minutes.** Five crawl and five per side on the same fresh development seeds for each arm; reserve the actual range with the coordinator first. Save videos and pre-reset traces. Require crawl ≥3/5 for retention. Report all counts, not only promising trials, and assess supported crawling versus continuous rolling. A single side success is discovery evidence, not the 3/5 recovery milestone. Export/action parity and actor-input separation are required before using a candidate.
4. **Reserve at most 25 GPU-minutes, used only by a new explicit decision.** Replicate the selected treatment in a second training seed if it has a real side-recovery signal; otherwise stop. Do not spend this reserve automatically on extra environment scaling, a new reset search or a stack of reward changes. If the first pair has no useful signal, the report result is a failed bounded test, not an invitation to expand the search silently.

If the audit instead points specifically to delayed credit with adequate exploration, a later isolated horizon proposal is λ = 0.99 with γ unchanged and the same rollout length. Its TD-residual e-fold horizon becomes 3.32 s; its variance also increases. Changing rollout64→128 is a separate experiment because it doubles samples per update at fixed environments. Neither modification is bundled into the first pair.

Immediate stop conditions throughout: non-finite observations/rewards, changed actor contract or actuator limits, reward from reset placement, repeated fall/reset farming, unscored passive get-up, loss of crawl retention, unexpected terminal/bootstrap behavior, export mismatch, or the time/iteration ceiling. Preserve failed traces and checkpoints. Failure of a small pilot cannot establish impossibility or convergence.

## Completed empirical audit and recommendation

Two GPU-owned, read-only probes used64 nominal environments per process, unchanged mechanics/reward/actor, seed4242 and exact unrounded confirmed script keyframes. Audit001 tested holding home, script then originalT100 crawling, the current deterministic policy, and the current stochastic policy for each side. Audit002 tested the same scripts followed by the latest shaped crawl policy. Returns below are γ =0.995 discounted over a30 s observation window, with zero absorbing reward after an environment fault. Script control continues after the recovery acceptance event so that subsequent behavior remains visible.

| Controller | LEFT recovery+5cm | LEFT mean G30 | RIGHT recovery+5cm | RIGHT mean G30 |
| --- | ---: | ---: | ---: | ---: |
| Hold home | 0/64 | −1.237 | 0/64 | −1.244 |
| Current deterministic policy | 0/64 | −3.473 | 0/64 | −3.725 |
| Current stochastic policy | 0/64 | −5.132 | 0/64 | −5.405 |
| Confirmed script → originalT100 | 62/64 | −1.572 | 57/64 | −1.155 |
| Confirmed script → latest shaped policy | 63/64 | −0.392 | 56/64 | −0.975 |

These are64 spatial replicas under nominal physics, **not64 independent randomized task seeds or qualification trials**. There are two scripted keyframe variants per side. Different prior group histories/contact numerics mean audit001 and002 are not bitwise paired even with identical keys, seed and origins. For example, exact LEFT p9's first3.2 s return changes from−1.190 to−1.319, before the continuation policy is used. Consequently the precise between-audit delta cannot be attributed exclusively to the continuation-policy switch. Both audits independently demonstrate successful paths; audit002 demonstrates a successful recovery controller whose sampled mean return beats holding under the current reward. [Audit001 summary](../../runs/reward-study-gpu-analysis-001/reward-study-gpu-returns-001-summary.json), [audit002 summary](../../runs/reward-study-gpu-analysis-001/reward-study-gpu-returns-002-summary.json).

The early barrier remains. At4 s the hold return is approximately−0.430/−0.432 for LEFT/RIGHT, whereas the audit002 script means are−1.281/−1.764. The exact confirmed source cases all recover and pass5 cm in audit002:

| Exact source | G4 | G30 | Recovery+5cm time | Later lane exit |
| --- | ---: | ---: | ---: | ---: |
| LEFT p9, env1 | −1.356 | −0.595 | 7.15 s | 12.90 s |
| LEFT p34, env13 | −1.257 | −0.239 | 7.15 s | 13.00 s |
| RIGHT p48, env9 | −1.493 | −0.206 | 6.85 s | 10.15 s |
| RIGHT p62, env57 | −1.451 | −0.416 | 6.80 s | 9.90 s |

Component traces from audit001 show why a one-number bonus argument is inadequate. At4 s, exact LEFT p9 pays progress−0.737, yaw−0.749 and command rate−0.172 against upright shaping+0.797; RIGHT p48 pays progress−1.337, yaw−0.477 and command rate−0.220 against shaping+0.837, with additional tilt/time/lateral costs. Simply blaming the command penalty or declaring the cap intrinsically too small is unsupported. Dropping the posture gate would again reward fallen sliding; dropping negative displacement accounting would risk cycles. Neither is recommended.

All script environment terminations observed in these probes are lane exits **after** the accepted recovery+5cm event. Zero reward after a fault avoids collecting auto-reset rewards but still means G30 compares variable active durations; escaping further negative costs can improve a return. The table therefore does not establish sustained lane crawling after recovery or a perfectly aligned training objective. Hold and learned-policy groups stay active through30 s, so their finite-window returns omit the small discounted tail and the next fallen-limit transition. Preserve this caveat when comparing returns or proposing terminal costs.

The learned stochastic action stream exceeds the raw normalized action interval in65.25% of LEFT and62.92% of RIGHT channel samples, with slew saturation73.81%/81.99%; neither side recovers. Deterministic actions also clip heavily (70.35%/36.05%). This supports a coherent-exploration problem more directly than a lack of action variance. It does not prove that reducing Gaussian standard deviation alone would solve it.

Initial shaped-critic means for its own stochastic policy are approximately−4.569/−4.311. The corresponding64-step discounted returns plus γ^64 V(s64), minus V(s0), average+0.133/−0.185. This initial-state test does **not** demonstrate a large critic failure. It also does not certify accurate values throughout the maneuver; the policy has not visited successful recovery trajectories on-policy. Off-policy scripted or hold returns must not be mistaken for calibration targets of this policy's critic. [Raw reports and traces](../../runs/reward-study-gpu-returns-001/report.json), [continuation-control report](../../runs/reward-study-gpu-returns-002/report.json).

**Recommended next decision:** authorize only the preparation and, if its data gates pass, the matched demonstration-loss versus unchanged-PPO pair described above. Keep the current actor architecture, reward, official starts, actuator model and γ/λ fixed. Capture fresh teacher actor24 observations and desired normalized actions through the normal action path, plus upright preservation data, before any optimization. The audit traces contain actions, physical summaries and reward components but **do not contain actor24 observations**, so they are not already a training demonstration dataset. Freeze the dataset's source hashes, validate closed-loop teacher replay and observation/action timing, and inspect observable-state/action conflicts before using it. Stop if the fixed feedforward actor cannot represent the teacher behavior without a hidden script phase. Do not silently add phase, heading, recurrence or teacher actions to the deployed policy interface.

This recommendation exposes known successful sequences while retaining the task objective; it is an exploration hypothesis, not a promised solution. A critic-only pair remains a secondary hypothesis if later on-policy calibration supports it. Potential replacement, a longer GAE horizon, prefix curricula and a lower-band static bank remain separate proposals. None should be bundled into the first pair, and none is authorized by this report. Existing simulation success flags remain development evidence; all hardware/contact/protected-point validation flags remain false.

Compute completed: audit001 used186.924 s and audit00267.161 s, totaling254.085 s =4.235 GPU-process minutes =0.070579 GPU-process hours, including container/Kit startup and cleanup. Each process used64 environments; **zero training iterations** ran. Audit002 exited successfully at2026-10-14 12:12:30.711 UTC. This accounting is elapsed time of owned GPU processes, not utilization-weighted device time, and belongs inside the proposed120-minute study ceiling. [Compute ledger](../../runs/reward-study-gpu-analysis-001/compute-ledger.json).

Independent report frozen at 2026-10-14T12:15:03.532284+00:00. The laptop reward study and its findings were not read during preparation. Any later comparison should be a separately labeled reconciliation that preserves this frozen report's provenance.
