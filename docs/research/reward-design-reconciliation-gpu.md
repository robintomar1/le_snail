# GPU reconciliation of the two independent reward studies

2026-10-14. This comparison was prepared only after the laptop coordinator posted `LAPTOP STUDY FROZEN` at12:27:23 UTC and the file hash was verified. Both independent reports remain unchanged:

- [Laptop study](reward-design-study.md): SHA256 `f880ac6fa55529344313d42b145e00f42a084981354e2e4113ab754e21fb4a06`, announced commit71408a6.
- [GPU study](reward-design-study-gpu.md): SHA256 `be8496e40de714e43607bbd09a662c323d609cf465c0f7b349aa81f1764e2a24`, frozen12:15:03 UTC.

The laptop's12:27:58 follow-up already accepts discount-consistent potential shaping, treats the privileged critic as a candidate, and prefers live scripted-prefix handover to dynamic state restoration. These are helpful post-freeze clarifications, recorded separately from its original report. This reconciliation does not authorize implementation or experiments.

**Recommendation: test demonstration-assisted recovery against unchanged PPO first, with the current reward and actor intact.** Both studies identify an undiscovered multi-second recovery sequence and substantial action saturation. The new GPU probes show that successful recovery plus crawl continuation can outperform holding under the existing reward. They support an early exploration/credit barrier, rather than the categorical claim that recovery is impossible under that reward. The proposed reward redesign contains mathematical and incentive risks that need separate checks before adoption.

## Agreements supported by both studies

| Finding | Joint interpretation |
| --- | --- |
| Zero stable get-ups over19,520 completed training side episodes; scripted paths exist at the same limits | Recovery discovery is the immediate missing capability. Preserve the scripts as positive controls and possible training demonstrations. |
| Crawling works on development trials; seed101 improved after additional updates | More updates can help crawling. Neither a universal plateau nor a reliable future pass-rate forecast is established. |
| No controlled environment-count experiment | More parallel environments are a throughput choice whose learning effect remains unmeasured here. Do not present a universal optimum. |
| High action means/std, clip/slew saturation and near-maximal command costs | Audit useful action diversity, rather than treating Gaussian entropy as evidence of coherent exploration. |
| Actor lacks heading/lateral position and reward-history fields | A privileged critic is a reasonable hypothesis; it cannot supply information to a deployed feedforward actor. |
| Static bank failed its agreed gates | Preserve that failure. Dynamic restoration or live-prefix curricula require new, explicit validation and scope. |
| Actor24, actuator limits and official acceptance criteria remain fixed | No hardware motion, hidden actor inputs, changed success definition or qualified deployment claim follows from these studies. |

Evidence and exact data paths appear in the two frozen reports. The GPU report additionally links the two executable return audits and their raw per-step traces.

## New evidence resolves part of the reward disagreement

Under unchanged reward and mechanics, nominal64-environment probes measured:

| Controller | LEFT mean discounted30 s return | RIGHT mean discounted30 s return | Recovery+5cm outcomes |
| --- | ---: | ---: | --- |
| Hold home | −1.237 | −1.244 | 0/64 each |
| Current stochastic learned policy | −5.132 | −5.405 | 0/64 each |
| Confirmed scripts, then originalT100 | −1.572 | −1.155 | 62/64 left;57/64 right |
| Confirmed scripts, then latest shaped policy | −0.392 | −0.975 | 63/64 left;56/64 right |

At4 s, the last controller averages−1.281/−1.764, versus hold−0.430/−0.432. The maneuver initially costs more than waiting, but successful continuation can repay it. At the exact previously confirmed source indices, all four scripts pass recovery+5cm and their30 s returns beat holding in audit002. [Audit001](../../runs/reward-study-gpu-analysis-001/reward-study-gpu-returns-001-summary.json), [audit002](../../runs/reward-study-gpu-analysis-001/reward-study-gpu-returns-002-summary.json).

These are spatial replicas of nominal starts, not64 independent randomized evaluation seeds. Runs are not bitwise paired: contact numerics and prior group histories affect some scripted prefixes before the continuation changes. All recorded script environment failures are later lane exits after the recovery acceptance event. The analysis assigns zero reward after a failure, so different active durations and the benefit of escaping future negative costs remain confounders. This is evidence that successful recovery **can** have a better return than passive holding; it is not proof of a perfectly aligned objective, sustained lane crawling, or a learned recovery policy.

The current stochastic policy's initial64-step bootstrap residual averages+0.133 left/−0.185 right. This does not demonstrate a large initial-state value failure, and it cannot certify value accuracy on unseen recovery states. Heavy clipping is directly measured: raw out-of-range action fractions65.25%/62.92%, slew-saturated channel fractions73.81%/81.99%. These results favor testing access to successful action sequences before attributing failure to the critic or changing several rewards at once.

## Corrections and unresolved disagreements

**Recovery cannot be ruled out for all future training.** The laptop's rule-of-three calculation treats19,520 episodes as independent draws from one fixed success probability. They instead come from changing, correlated PPO policies and two reward settings. Even under the independent stationary assumption, an upper confidence bound does not prove impossibility or bound the performance of a future improved policy. The supported conclusion is no discovered recovery in these runs and an unfavorable empirical case for indiscriminate continuation. The laptop follow-up's description of an early local barrier is more defensible than its frozen report's global impossibility claim.

**Discounted potential shaping is not interchangeable with an undiscounted difference.** With γ=.995, the invariance construction is `F=γΦ(s_next)−Φ(s)`, plus consistent terminal/truncation handling. For the proposed `Φ=3u`, an illustrative low→high→low two-transition cycle from−.1 to.8 and back earns `2.7−.995×2.7=.0135` discounted under `3Δu`, despite zero undiscounted sum. This is an algebraic counterexample, not an observed physical maneuver. An undiscounted difference creates an additional preference for earlier upright occupancy; it is not automatically an invalid heuristic, but cannot claim the discounted invariance or anti-cycle guarantee. A negative immediate shaping term while stationary upright does not invalidate the telescoping theorem. The laptop follow-up accepts the corrected form; adoption is still a separate experiment. [Ng, Harada and Russell, Theorem1](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

**Applying the posture gate to both displacement signs creates a new accounting loophole.** The proposed `30g(u)Δx` gives+0.3 for1 cm forward while upright and zero charge for returning1 cm while fallen. A closed displacement/posture loop can therefore earn positive progress credit. Other terms might discourage a realizable loop, but the progress term itself no longer ensures net-displacement accounting. The current one-sided gate preserves the backward debit precisely while withholding fallen forward credit. Removing that debit is an understandable response to the measured recovery barrier, but is not a narrowly safe fix without explicit closed-loop and reset tests. This is a synthetic reward counterexample, not a claim that the present policy has learned the exploit.

**The proposed alive bonus cannot guarantee nonnegative total rewards or remove termination incentives.** At u=1, zero displacement/command rate/turn-rate and heading errorπ, the laptop formula gives `.016−.5×.05×2=−.034` before other terms. Negative displacement, posture change and contacts provide further counterexamples. The fallen alive term `.006/step` equals.12/s at20 Hz, not.18/s. A positive baseline changes preferences over episode duration; it needs direct counterfactual and active tests rather than a general claim that termination is never attractive.

**Noise saturation is real; the proposed trainer changes are still hypotheses.** The GPU stage-specific probes also show that saturation is not97% uniformly in every context. State-dependent clipping/slew limits and nonzero policy means mean the action process is not universally six unbiased ±.05 rad random walks onceσ exceeds.1. Entropy reduction, std caps/reset, correlated exploration and a bounded distribution should each be tested separately. A conventional policy gradient using raw sampled actions through a deterministic clipping environment is not automatically biased: Fujita and Maeda derive a lower-variance estimator that exploits clipping. A squashed distribution requires consistent likelihood handling; merely changing an output nonlinearity is not a blanket fix. [Clipped Action Policy Gradient](https://proceedings.mlr.press/v80/fujita18a/fujita18a.pdf).

**A three-second turning-rate average is not an observed actor feature.** Instantaneous `−gyro_b·gravity_b` is available, but its three-second EMA depends on history absent from the fixed actor24 input. Adding that EMA to a critic can help value estimation; it does not make it directly observable to this feedforward actor. World-vertical angular velocity also differs from Euler heading rate during tilt. The frozen studies agree that turning and lane margins need attention, but the proposed rate cost, alive bonus, heading reduction and trainer changes constitute a bundle, not an isolated causal test. [Actor observation implementation](../../src/le_snail/sim/env.py), [partial-observability critic limitations](https://arxiv.org/html/2105.11674).

**Keep both crawl candidates.** Seed101's faster and straighter motion merits later matched evaluation; latest shaped policy's5/5 and seed101's3/5 came from different development blocks. Neither proves which is the better recovery-training base. Switching lineage, noise settings and rewards together would obscure the first recovery test. Use one frozen actor in both arms now, preserve seed101 and revisit it with matched evidence.

**Dynamic banks remain an option, not a validated dependency.** Replaying a scripted prefix in a live environment avoids reconstructing hidden simulator/contact history, but still needs teacher-credit masking and passive-completion controls. The laptop's estimated dynamic-bank yield is a planning estimate, not a measured result. We agree that live-prefix handover is preferable to unverified dynamic restore if a curriculum is later selected. The first experiment below instead tests supervised demonstrations without changing starts or needing a bank.

## One proposed first experiment

This is one matched pilot of **demonstration assistance**, with dataset collection as its prerequisite. It is not approval to run either study's full roadmap.

**Initialization and invariants.** Both arms start from `runs/recover-shaped-051-100/model_final.pt`, checkpoint SHA256 `b748cc9cc9e5a4dbc029f3bf1292b7c4ca4cfcaa07d98eb80faa64cb8543b340`; exported-policy SHA256 `368d6d7f43dd51bb1fab03ba7b3591cb173e893e07afd93766eb1915d13777c4`. Use identical full checkpoint/optimizer initialization, seed42, nominal recover stage, original65% upright/35% side reset mixture with balanced sides, current cap1 reward, actor/critic24 architectures and normalizer initialization, existing std/entropy settings, γ=.995, λ=.95,64 rollout steps and four minibatches. Hard actuator/action limits and the official evaluation remain unchanged. No privileged critic, state bank, prefix handover, new potential or trainer repair is bundled in.

**Prerequisite dataset, at most64 simulation environments.** Capture actual24-input observations and normalized desired teacher actions from the exact confirmed LEFT p9/p34 and RIGHT p48/p62 scripts, followed by the current crawl policy through the normal action path. Capture a frozen-policy upright preservation set as well. Save script/config/asset/policy hashes, controller history and full outcomes; separate successes and failures. Verify timing, clipping/slew equivalence, reproducible positive-control recovery on both sides, and whether similar actor observations demand materially conflicting next applied commands. Teacher script phase stays metadata, never an actor feature. Stop for unexplained contradictory labels or failed replay rather than adding hidden inputs. Existing audit NPZ files do not contain actor24 observations and cannot substitute for this collection.

**Arms, each at most256 environments and100 new PPO iterations.** Control A uses unchanged PPO. Treatment B differs only by adding the two demonstration-assistance terms to its actor loss:

```text
L_B = L_A
      + 0.1 E_Dside mean_j[(mu_theta(o)_j − a_script_j)^2]
      + 0.1 E_Dupright mean_j[(mu_theta(o)_j − mu_frozen(o)_j)^2]
```

Balance left/right and script variants in `Dside`; use a fixed, versioned upright set. These two terms together are the tested assistance recipe; this pair cannot separately attribute improvement to the side loss versus the retention loss. PPO trajectories remain on-policy, and expert transitions are never presented as if sampled from the learner. The actor must complete every official-start test with no teacher or script at evaluation/deployment.

**Review and stop gates.** Evaluate both exports at50 iterations on the same proposed development seeds3600–3614, first reserving that range with the coordinator. Continue both arms to100 only if the pair has at least one stable get-up in the five official-start trials for either side, crawl retention≥3/5, valid export parity and no unacceptable gait on video. Proposed final matched block3620–3634 also requires reservation. Report left/right5cm successes separately; a single stable get-up is discovery evidence, not recovery qualification. Stop immediately for non-finite state/action/reward, actor/action mismatch, leaked teacher observations, reset/passive reward credit, crawl collapse, continuous rolling or an exhausted budget. Preserve checkpoints and failed traces.

**Budget.** Allow at most20 GPU-process minutes for data/replay checks,25 minutes per training arm,35 minutes for exports/evaluations and10 minutes contingency. Together with the completed4.235-minute audits, this stays below120 GPU-process minutes. These are ceilings, not throughput promises; stop and report incomplete evidence if a ceiling is reached. The256-env pilot is intentionally diagnostic and cannot establish the best production-scale environment count. A later2048–4096-env extension needs its own decision after a positive result. Reserved qualification seeds9000–9014 remain unused.

## Decision remaining

The owner/coordinator must choose whether to authorize this single dataset-gated demonstration-assistance pilot. If accepted, implementation, dataset validation and the matched pair become the concrete scope; neither future scaling nor alternative reward/critic/curriculum changes are implied. If rejected, preserve the reports and current checkpoints and identify one alternative hypothesis explicitly. No experiments have been launched during this reconciliation, and no source, hardware or handoff files have been changed.
