# Day 11–12 plan: Reinforcement learning (Day 11) and alignment (Day 12)

Eleven lessons in two clean groups. **Day 11 RL group** (rl-basics, q-learning, policy-gradients, deep-rl, ppo; 30 + 30 + 35 + 35 + 35 = 165 min) runs on `src/lib/rl`. **Day 12 alignment group** (sft, rlhf, dpo, constitutional-ai, rlvr) runs on `src/lib/align`. `evaluations` stands alone. Lesson skeletons (step-by-step briefs with exact props) are in `src/content/lessons/alignment/*.mdx` as MDX comments. Task 0 only rebalances these skeletons; the modules and widgets below are future implementation work.

## 1. Shared core modules

Both are pure TS (no DOM), `Float64Array` math, `createRng` from `src/lib/math/rng.ts`, Adam/SGD from `src/lib/nn/optim.ts`. Each gets one test file (`tests/unit/rl-core.test.ts`, `tests/unit/align-core.test.ts`) written **before** the widgets, like `src/lib/cnn/conv.ts` on Day 7.

**`src/lib/rl/` (env, tabular algorithms, DQN and PPO)**
- `gridworld.ts`: the Russell and Norvig 4×3 world (wall at (2,2), +1 goal, -1 pit, start bottom-left). Props: `stepReward` (default -0.04), `slip` (0.2: intended move 0.8, each perpendicular 0.1), `gamma`. API: `makeGrid(opts)`, `transitions(s,a)` → `[s', p, r][]`, `step(s,a,rng)`, `rollout(policy, rng, maxSteps)`.
- `bandit.ts`: k-armed Gaussian bandit, `makeBandit(k, seed)`, `pull(arm, rng)`.
- `tabular.ts`: `policyEvaluation(grid, pi, sweeps)` (records V per sweep), `valueIteration(grid, sweeps)` (records V, Q per sweep), `qLearning(grid, {alpha, epsilon, episodes, seed})` (records every transition: s, a, r, s', TD target, TD error, Q row after), `reinforce(env, {lr, baseline, episodes, seed})` and `actorCritic(...)` for both the bandit and the grid with a tabular softmax policy (logits per state-action), recording logits, returns and per-episode gradient estimates.
- `dqn.ts`: reuse `gridworld.ts`; a small `Float64Array` 2→16→4 ReLU MLP takes normalized coordinates and produces four Q-values, with pure forward/backward helpers and flat weights. Semi-gradient TD loss with a detached target and a terminal mask; finite FIFO replay buffer (capacity 256, uniform seeded batches of 16); separate frozen target weights copied every 50 updates. Record selected transitions, online/target predictions, losses, weight distance and evaluation returns. Comparison variants: neither mechanism, replay only, target only, both; also a supervised fixed-Q-star fit to separate moving targets from ordinary network training. Do not edit the shared NN engine.
- `ppo.ts`: reuse the grid, tabular softmax actor and value critic; collect 64 transitions, compute termination-aware GAE (gamma 0.9, lambda 0.95), freeze old log-probabilities and advantages, optimize shuffled mini-batches of 16 for four epochs (epsilon 0.2), then collect a fresh batch. Keep value loss separate from the policy surrogate. Record ratio extrema, out-of-band fraction, favourable-direction plateau fraction, exact categorical KL(old || new) averaged over the batch's states and evaluation return. Export a pure clipped-surrogate helper reusable by `src/lib/align/ppo.ts`; keep the Day 12 fixed-SFT-reference KL reward shaping in the alignment module.
- Tests: value iteration matches the textbook 4×3 utilities (VERIFY the published values before asserting them, otherwise assert the Bellman fixed point to 1e-9); `transitions` probabilities sum to 1; REINFORCE gradient on the bandit is unbiased (mean of many samples ≈ exact gradient); baseline does not change the mean but lowers variance; determinism under a seed.
- New algorithm tests when implemented: numerical gradients for the MLP and detached DQN loss; terminal targets equal the terminal reward; replay capacity/sampling boundaries and seed determinism; target weights unchanged between copies and equal to online weights on copy; PPO surrogate/gradient for both advantage signs inside/outside the clip band; frozen old log-probabilities and advantages across epochs; GAE terminal/truncation boundaries; KL zero for identical policies. Clipping is not a hard probability-ratio or KL bound.

**`src/lib/align/` (the tiny policy)** The key decision. The policy is a **log-linear categorical policy over a fixed bank of candidate responses**:
`π_θ(y|x) = softmax_y( θ·φ(x,y) + c(x,y) )` over K = 8 candidates per prompt, 10 prompts (8 train, 2 held-out).
- `bank.ts`: prompts are short checkable questions ("What is 7 + 5?", "Spell 'cat' backwards.", "How many legs does a spider have?"). Every prompt has the same 8 candidate templates: *continues the question list* (base-model behaviour), *terse correct*, *clear correct*, *padded correct* ("Great question! ... 12. I hope this helps!"), *rambling wrong* (very long, flattering, wrong), *terse wrong*, *refusal*, *hedged correct*. Feature vector φ (d = 6): `answers`, `correct`, `length` (words / 10), `flattery`, `refuses`, `hedges`. `c(x,y)` is a fixed seeded per-candidate offset (base-model quirks, never trained) so prompts differ.
- **Gold utility** (hidden, "what the person actually wanted"): `g(y) = w_gold·φ`. **Simulated annotators**: Bradley-Terry on `w_gold·φ + bias·φ + noise`, where `bias` rewards length and flattery (the documented length bias of human raters). The rambling candidate's length lies far outside the range seen in preference pairs, so a linear reward model extrapolates it upward. That is the honest source of reward hacking.
- `policy.ts`: `probs`, `logp`, exact `kl(θ, θref)` (sum over 8 candidates, averaged over prompts), exact expected reward and its gradient `E[(r - E r) φ]`.
- `sft.ts`: max-likelihood on demonstrations (Adam). Exponential-family fact used by the SFT "cap" step: at the optimum `E_π[φ] = E_demo[φ]` (moment matching), so SFT correctness converges to the demonstrators' rate.
- `reward.ts`: `samplePairs(policy, n, rng, labeler)`, `fitRM(pairs)` (logistic regression on `φ_w - φ_l`), `labeler`: `'human'` (biased BT) or `'ai'` (constitution rule checker, used by constitutional-ai).
- `ppo.ts`: per iteration, sample B = 64 (prompt, response) pairs from π_old; shaped reward `r = RM(y) - β (log π_old - log π_ref)(y)`; advantage = r minus a per-prompt running-mean baseline (stands in for PPO's value head, say so in one line); clipped surrogate ε = 0.2, 4 epochs, Adam. Records per iteration: θ, exact proxy reward, exact gold, exact KL, clip fraction, ratio histogram.
- `dpo.ts`: DPO loss on the same pairs, records θ, loss, chosen/rejected log-probs, implicit reward `β (θ - θref)·φ`.
- `optimal.ts`: closed form `π* ∝ π_ref exp(r/β)`; in this family `θ* = θref + w_RM/β` exactly, which is the DPO lesson's anchor.
- `verifier.ts` (rlvr): addition problems with 1 to 4 digits. Tabular policy per digit count over actions = (k worked columns, k = 0..4) × (format: one answer | list of guesses). A fixed solver writes k columns with carries and adds the rest without carrying; the response text is generated, and the verifier **parses the text**: `strict` (last line is exactly `Answer: N`) or `lenient` (N appears anywhere). GRPO: G = 8 samples per prompt, advantage `(r - mean)/std` per group (0 when std = 0), optional KL to the start policy.
- Tests: analytic gradients of log-likelihood, DPO loss and PPO surrogate agree with `src/lib/nn` autograd (`Value`) to 1e-9; KL ≥ 0 and 0 at θ = θref; moment matching at the SFT optimum; `θ*` closed form equals long PPO-free exact ascent on `E r - β KL`; DPO with infinite pairs drawn from BT(r) recovers `w_RM/β` direction; verifier parses edge cases; GRPO advantages sum to 0 per group.

Determinism: one seed per lesson step prop (`seed` default fixed), every run precomputed once on mount, cached by props. The Day 12 candidate-policy runs are tiny (PPO 200 iterations × 4 epochs × 64 samples × d = 6); measure setup rather than assuming the new DQN/PPO comparisons are equally cheap. Move a run to a worker if setup exceeds 50 ms. Evaluation rollouts use separate fixed seeds and no training updates. Probe all comparison claims before writing the lessons: naive DQN may oscillate without diverging, and neither DQN mechanisms nor a smaller PPO epsilon guarantee a better measured run.

**Why this policy (honesty):** gradients, KL, rewards and the reward-hacking run are real computations; the only designed part is the bank, which deliberately contains an exploitable answer, as real response spaces do. Each lesson says once, plainly: "a real model writes its response token by token; here it picks one of eight, so we can compute everything exactly". Optional: if Day 10 ships a tiny LM, `c(x,y)` could be its log-probabilities of the candidate strings. Not required.

## 2. Widgets

| Widget | sim.ts computes | Renderer | Budget risk |
|---|---|---|---|
| `GridWorldLab` | runs from `src/lib/rl`; frames over recorded transitions/sweeps | SVG (12 cells, Q triangles, arrows; agent dot by transform) | low |
| `BanditLab` | bandit runs (greedy/ε/UCB over 50 seeded runs), softmax PG, REINFORCE gradient samples | SVG bars + polylines | low |
| `PPOLab` | shared clipped surrogate for signed advantages; PPO batch drift diagnostics from `src/lib/rl/ppo.ts` | SVG curves, probability bars + ratio histogram | low-medium |
| `SFTLab` | base/SFT policies, demos, moment matching; word-bigram LM for the mask step | SVG | medium (two views) |
| `PreferenceLab` | pairs, RM, PPO, β sweep, over-optimization, DPO, optimal policy | SVG | medium-high (many modes). Split DPO into `DPOLab` if > 36 KB |
| `CritiqueRevise` | rule checks, critique-revise rounds, SL on revisions, principle edits | SVG / HTML text | low |
| `VerifierRL` | generated responses, verifier, GRPO training, length curves | SVG + text | low |
| `EvalLab` | IRT benchmark, binomial CIs, arena BT fit, contamination, adaptive overfitting, saturation | SVG (Canvas for arena vote particles if > 300 nodes) | low-medium |

Modes per lesson step (full briefs in the MDX comments):

| Lesson | Steps → widget `mode` (props) |
|---|---|
| rl-basics | 1 GridWorldLab `loop` · 2 `mdp` · 3 `return` (gamma 0.9) · 4 `discount` (gamma 0.5 vs 0.99) · 5 `value` · 6 BanditLab `explore` · 7 `optimal-policy` (stepReward slider) |
| q-learning | 1 GridWorldLab `qvalues` · 2 `bellman` · 3 `qlearn` · 4 `qlearn-curve` · 5 `qlearn-terminal` (goal, pit, ordinary; gamma 0.9) |
| policy-gradients | 1 BanditLab `softmax-pg` · 2 BanditLab `baseline` · 3 GridWorldLab `reinforce` · 4 `reinforce-update` (state-value baseline, episode 10) · 5 `actor-critic` |
| deep-rl | 1 GridWorldLab `dqn-network` (coordinate inputs, hidden 16) · 2 `dqn-update` (no replay/target) · 3 `dqn-instability` (fixed-target comparison, lr 0.01/0.1) · 4 `dqn-replay` (capacity 256, batch 16) · 5 `dqn-target` (copy every 50) · 6 `dqn-compare` (four variants, 10 runs) |
| ppo | 1 GridWorldLab `ppo-advantage` (gamma 0.9, lambda 0.95) · 2 `ppo-ratio` (frozen old probabilities) · 3 PPOLab `clip-positive` (A +1, epsilon 0.2) · 4 `clip-negative` (A -1, epsilon 0.2) · 5 GridWorldLab `ppo-epochs` (4 epochs, batch 64, mini-batch 16) · 6 PPOLab `drift` (epsilon 0.1/0.2/0.4) · 7 GridWorldLab `ppo-train` (100 fresh-batch iterations) |
| sft | 1 SFTLab `base` · 2 `data` · 3 `train` · 4 `mask` · 5 `heldout` · 6 `cap` |
| rlhf | 1 PreferenceLab `pairs` · 2 `reward-model` · 3 `rm-bias` · 4 `ppo` (beta 0.1, applying Day 11 PPO) · 5 `ppo-clip` (response-level application; collection policy versus SFT reference) · 6 `kl-sweep` · 7 `ppo` (beta 0) · 8 `overopt` |
| dpo | 1 PreferenceLab `optimal` · 2 `dpo-pair` · 3 `dpo-train` · 4 `implicit` · 5 `compare` |
| constitutional-ai | 1 CritiqueRevise `rules` · 2 `revise` · 3 `sl` · 4 PreferenceLab `pairs` (labels ai) · 5 `ppo` (labels ai, beta 0.1) · 6 CritiqueRevise `edit` |
| rlvr | 1 VerifierRL `verify` · 2 `sample` · 3 `grpo` · 4 `train` · 5 `length` · 6 `train` (verifier lenient) · 7 `domains` |
| evaluations | 1 EvalLab `benchmark` · 2 `arena` · 3 `contamination` · 4 `goodhart` · 5 `saturation` · 6 `errorbars` |

Motion over `t` (all via pure `frameAt(t, run)` over precomputed records; readouts snap to the recorded iteration shown; bars ease between iterations with statistics recomputed at the displayed θ): grid agent glides cell to cell, values fill per sweep, Q triangles recolour per update; bandit pull counters tick and curves draw; candidate probability bars and θ feature bars tween per training iteration, sampled responses flow as dots into the batch; reward/KL/gold curves draw left to right; verifier ticks/crosses appear per sample; arena votes stream into ratings. Markers name segments ("Sample", "Score", "Update").

## 3. Reuse

- `_shared`: FigureCard, Timeline, createLoop, readColors, advance (all widgets).
- `src/lib/nn/optim.ts` `adam`/`sgd` (plain arrays), `src/lib/nn/value.ts` only in tests as the gradient oracle. Do not edit `src/lib/nn`.
- `SoftmaxBars` / `ProbabilityLab` are visual references for probability bars; no code import needed.
- Colours: `--viz-forward` policy probabilities, `--viz-gradient` updates/advantages, `--viz-pos`/`--viz-neg` signed reward and Q, `--viz-seq-*` value heatmaps and vote density, `--accent` the highlighted candidate or cell. Gold (true) reward versus proxy reward need two distinct marks: proxy = `--viz-forward` line, gold = `--text` dashed line. No new tokens.

## 4. Build order

1. Task 0: rebalance the five Day 11 stubs and links, verify its checks, then stop for review. No algorithm/widget implementation in Task 0.
2. When Day 11 reaches its turn, build `src/lib/rl` and its tests, then GridWorldLab + BanditLab for rl-basics. Write and review one lesson per run, in this order: rl-basics → q-learning → policy-gradients → deep-rl → ppo. Add the DQN modes for deep-rl and the PPO modes + PPOLab for ppo only after their pure helpers are tested. Reuse the existing widgets; no second gridworld.
3. After Day 11 is reviewed, build `src/lib/align` and its tests, then SFTLab + sft and PreferenceLab + rlhf. RLHF imports the shared clipped-surrogate helper and links back to the Day 11 PPO lesson; it teaches the response reward model and fixed-reference penalty rather than introducing PPO again.
4. Continue Day 12 in curriculum order, one lesson per run: dpo (only one owner touches PreferenceLab at a time), constitutional-ai + CritiqueRevise (PreferenceLab labels="ai"), rlvr + VerifierRL, evaluations + EvalLab.

## 5. Risks and open questions for the owner

- **Day split and rebalance.** Day 11 now has five RL stubs, 165 min; Day 12 has six alignment stubs. RLHF lists `alignment/ppo` as its RL prerequisite. The only planned algorithm import across cores is the pure clipped-surrogate helper; the environments, policies and reward shaping remain specific to each day.
- **DQN effect must be measured.** The 4×3 grid and coordinate MLP may not diverge under naive training. Probe moving-target drift, oscillations and all four variants; show only the effect present in the run. A divergence counterexample belongs in further reading until its assumptions are independently verified.
- **PPO clip semantics.** The favourable side depends on the advantage sign. Do not label every out-of-band ratio a zero-gradient sample, or promise that clipping bounds ratios/KL. The Day 12 fixed reference is distinct from each batch's old policy.
- **Is a candidate-bank policy acceptable?** It makes KL, the optimal policy and moment matching exact and legible, at the cost of no token-level generation. The alternative (tiny tabular token Markov policy with exact DP KL) is about 3× the code and harder to read. Recommend the bank.
- **Designed exploit.** Reward hacking requires a candidate the reward model overrates. The rambling candidate is designed in; the hacking itself is computed. The sim author must tune annotator bias and the bank so that, with β = 0, proxy reward rises while gold falls after a peak, and with β ≈ 0.1 gold ends above SFT. Print the numbers before writing prose.
- **RLVR solver.** "More worked columns, fewer carry slips" is an assumption of the toy, stated in the lesson. Length growth is then a real RL outcome, not scripted.
- **Constitutional AI's AI labeler** is a rule checker, not a language model. Say so once.
- **SFT mask step** needs a second tiny model (word bigram). If the measured response-loss gap between masked and unmasked training is negligible, show the gradient mask itself plus the measured numbers, and drop the comparison claim.
- PreferenceLab may approach 40 KB gz; split out `DPOLab` if so.
- Day 15 safety lesson may want to reuse `EvalLab` and the over-optimization mode.

## 6. Core results (measured)

Measured 2026-10-05 by the Day 11 core agent from `src/lib/rl` (Node, vitest probes, deleted afterwards). **These numbers override the MDX briefs.** Re-probe before quoting anything not listed here.

**Reward convention.** Every move costs `stepReward`; a move into a terminal also pays +1 (goal) or −1 (pit), so the goal move earns 0.96 by default. Terminals have value 0. At gamma 1 this reproduces the book exactly.

**Value iteration, gamma 1, stepReward −0.04 (46 sweeps to 1e-12, 7 ms).** These match AIMA 2nd/3rd ed. Fig. 17.3 (verified by web search): (1,3) 0.812, (2,3) 0.868, (3,3) 0.918, (1,2) 0.762, (3,2) 0.660, (1,1) 0.705, (2,1) 0.655, (3,1) 0.611, (4,1) 0.388. Optimal policy: (1,1)↑ (2,1)← (3,1)← (4,1)← (1,2)↑ (3,2)↑ (1,3)→ (2,3)→ (3,3)→. Q*(1,1) = up 0.705, right 0.631, down 0.660, left 0.671. Gaps between the best and second-best Q* are small everywhere (0.018 at (4,1), 0.019 at (3,1) and (3,2), at most 0.041).

**rl-basics.**
- `loop` (uniform random policy): seed 34 gives 15 moves, 1 slip (move 1), reaches the goal, total reward +0.40 (this is the reference mode's default). Seed 1 (the brief's) gives 12 moves, slips on moves 2 and 5, ends in the pit, total −1.48. Seed 13: 11 moves, 1 slip (move 5), pit, −1.44. Most random episodes end in the pit.
- `return` (textbook policy, gamma 0.9): seed 3 has 5 moves, no slip, G_0 = 0.492, plain sum 0.800. Seed 4: 7 moves, 1 slip, G_0 = 0.323, sum 0.720.
- `discount` (gammas 0.5 to 0.99): the optimal policy changes at gamma = 0.66005 ((4,1) ↓ → ←; at low gamma it bumps the wall below the pit rather than walk away) and 0.93781 ((2,1) → → ←). gamma 0.5: V(start) −0.045, policy ↑ → ↑ ↓ ↑ ↑ → → → (cells in order (1,1) (2,1) (3,1) (4,1) (1,2) (3,2) (1,3) (2,3) (3,3)). gamma 0.99: V(start) 0.660, policy ↑ ← ↑ ← ↑ ↑ → → →. Note (3,1) is ↑ at 0.99 but ← at gamma 1, so one more switch lies above 0.99. Use `kit.job('gammaSwitches', …)` (scan ≈ 100 ms).
- `value` (policy evaluation of the textbook policy, gamma 1): V(start) by sweep 1..12 = −0.040, −0.080, −0.120, −0.160, 0.128, 0.326, 0.482, 0.579, 0.636, 0.669, 0.686, 0.696; converges to 0.705. It stays within 0.005 of its final value from sweep 13. Max change per sweep falls from 0.76 to about 2e−7 by sweep 30.
- `optimal-policy` (gamma 1, stepReward −2 to −0.0001): the policy changes at stepReward −1.64971, −1.56426, −0.73114, −0.45262, −0.08499, −0.04483, −0.02736 and −0.02214. The book prints the regions as R < −1.6248, −0.4278 < R < −0.085, −0.0221 < R < 0 and R > 0. −0.085 and −0.0221 reproduce; the others do not reproduce exactly (we find finer regions), so quote our values. At R = 0 every value is 1 and actions tie. For R > 0, gamma 1 values grow without bound, so `rewardSwitches` uses gamma 0.99 there. Then V(start) = 5.0 at R = 0.05, and the policy avoids both exits ((3,3)←, (3,2)←, (4,1)↓). The scan takes about 190 ms, so use `kit.job('rewardSwitches', …)`.
- `explore` (5 arms, 50 runs, 500 steps, seed 7; 37 ms): percent optimal over the last 100 steps: greedy 74.0%, epsilon 0.1 89.3%, UCB c 2 91.0%. Runs stuck on a suboptimal arm (most-pulled arm over the last 100 steps): greedy 13/50, epsilon 0.1 1/50, UCB 2/50. Average reward over the last 100 steps: greedy 1.137, epsilon 0.1 1.178, UCB 1.274.

**q-learning.**
- The first terminal update for seed 11 (alpha 0.1, epsilon 0.1) is transition 21 of episode 1: at (3,3), action right, r 0.96, target 0.96, error 0.96, new Q 0.096. 300 episodes take 5 ms.
- After 300 episodes the start cell's max Q is 0.293, against V* 0.705. The max |Q − Q*| is 0.84, and only 5 of 9 greedy actions are optimal. **Tabular Q-learning with alpha 0.1, epsilon 0.1 and gamma 1 is far from converged at 300 or even 3000 episodes.** Over 20 runs × 3000 episodes, only 1 run settled on the full optimal policy. Getting 8 of 9 cells right takes a median of about 2000 episodes, varying widely. The near-tie cells keep flipping because constant alpha leaves noise larger than the 0.02 Q* gaps. Other settings, 20 runs, 3000 episodes:
  - epsilon 0.3: max error 0.38 at episode 3000; 8/9 cells after a median of about 300 episodes.
  - alpha 0.5: 8/9 cells sooner (median about 150 episodes), but the error stays 0.86.
  - gamma 0.9: 8/9 cells after about 25 episodes (median), error 0.52.
  - Honest framing: "most cells right within a few hundred episodes; the near-tie cells need far more and keep wobbling at constant alpha". For `qlearn-curve` (epsilons 0/0.1/0.3, alpha 0.1, 500 episodes, 20 runs), final max error to Q* averages 0.88 / 0.66 / 0.53. epsilon 0 never matched the optimal policy in any run, and its final agreement was 4 to 8 of 9 cells. Each epsilon takes 30–45 ms; memoise it or use a job.

**policy-gradients.**
- `baseline`: means 1, 2, 3, theta 0, 400 samples, baseline 2.0 (the expected reward). The exact gradient is (−0.333, 0, 0.333). Sample means without / with the baseline: arm 1 −0.403 / −0.322, arm 2 −0.043 / −0.021, arm 3 0.446 / 0.343. Variances: arm 1 0.727 / 0.232 (3.1×), arm 2 1.268 / 0.274 (4.6×), arm 3 1.788 / 0.305 (5.9×).
- `softmax-pg` (means 1, 2, 3, lr 0.1, seed 5): the best arm's probability is 0.333, 0.835 and 0.964 at steps 0, 50 and 200.
- Grid REINFORCE, gradient variance at the uniform policy over 3000 episodes: total variance 75.1 without a baseline and 34.7 with the exact V^π baseline (2.2×). Means agree within 0.05.
- `reinforce` (lr 0.1, 10 runs; exact V(start) of the current policy, gamma 1, V* 0.705). No baseline: −1.587, 0.346, 0.621, 0.644 at episodes 0, 100, 500, 1000. Learned state-value baseline: −1.587, 0.345, 0.630, 0.660. **The baseline gives only a small gain on the curve here, while the gradient variance halves.** 10 runs take about 100 ms per variant: use a job.
- `actor-critic` (lrActor 0.1, lrCritic 0.2, 10 runs): 0.375, 0.632, 0.666 at episodes 100, 500, 1000. It is slightly ahead of REINFORCE at every point, but the gap is small (0.03 at episode 100). It takes 86 ms, so use a job.

**deep-rl** (2→16→4 ReLU MLP, 116 parameters, gamma 0.9, V*(start) 0.351, Q*((3,1), up) 0.397). Initial Q-values (init seed 17) at (1,1): −0.47 / 0.16 / 0.79 / −0.92.
- Instability, on a fixed 1000-transition random-walk stream (`randomStream(g, 1000, 17)`), diagnostic transition (3,1) up:
  - SGD lr 0.01: no divergence. Peak |Q| 1.65, Bellman RMS 0.29 at the end. The target y drifts over −0.02..0.38 (total variation 0.86), and Q reverses direction on 446 of 1000 updates.
  - SGD lr 0.1: no divergence. Peak |Q| 2.95, y range −0.03..0.66, total drift 5.9. The fixed-Q* fit at the same lr reaches loss 2e−4 with peak |Q| 1.34.
  - **Adam lr 0.1: the moving-target run diverges.** Peak |Q| 218, |Q| 109 at the end, Bellman error rising from 0.58 to 13.8, y drifting to 109. The fixed-Q* fit with the same optimiser, lr and data stays bounded (peak 5.3, Bellman 0.30).
  - Adam lr 0.01: stable, peak |Q| 2.8.
  - **So the textbook moving-target divergence appears with Adam at lr 0.1, not with plain SGD.** The `trainDQN` default optimiser is Adam; SGD is `optimizer: 'sgd'`. Each run takes 60–80 ms (use a job).
- Naive online DQN that acts with its own network (1000 updates): no divergence with SGD at 0.01 or 0.1, or with Adam at 0.01. Its exact greedy evaluation return oscillates between −0.12 and 0.33 (Adam 0.01). Adam 0.1 stays at about −0.40, with peak |Q| 6.4.
- Target network (replay, copy every 50, seed 19, lr 0.01): the diagnostic target is 0.3209 at update 48 and 0.1312 at updates 49, 50 and 51. Because updates are 0-based in the arrays, index 49 is the 50th update, where the copy happens. Weight distance falls from 1.488 to 0 at the copy and is 0.024 one update later.
- `dqn-compare` (4 variants × 10 runs × 500 episodes, epsilon 0.1, capacity 256, batch 16, copy every 50). Final exact greedy return, mean (sd), V* 0.351:
  - Adam lr 0.01: naive 0.331 (0.014), replay 0.183 (0.245), target 0.279 (0.113), both 0.190 (0.269). **The mechanisms do not help at this lr; naive is best.**
  - **Adam lr 0.001:** naive 0.260 (0.220), replay 0.299 (0.102), target 0.285 (0.139), both 0.331 (0.019). This is the textbook ordering: both mechanisms together give the best and most consistent result. Recommend lr 0.001 and say plainly that at lr 0.01 the ordering reverses.
  - SGD lr 0.01: all variants about 0.22–0.26 with large spread. SGD lr 0.001: all about −0.25 (not learned).
  - No variant reaches the exact optimal policy in any run (0/10).
  - **Cost: 7–9 s for 500 episodes and 3.7 s for 200 episodes** (Node, one thread). Run it with `kit.job('compareDQN', grid, opts)` in the worker and show the loading text, or cut runs/episodes. Do not run it on the main thread.

**ppo** (tabular softmax actor, tabular critic, gamma 0.9, lambda 0.95, batch 64, mini-batch 16, 4 epochs, Adam lr 0.03, critic SGD 0.1).
- `ppo-train` (seed 37, 100 iterations, 66 ms, so use a job): the exact start value of the stochastic policy is −0.410, −0.076, 0.264, 0.330 and 0.341 at iterations 0, 10, 20, 50 and 100 (V* 0.351). Critic RMS error against the exact V^π is 0.443, 0.227 and 0.157 at iterations 0, 20 and 99.
- `drift` and `ppo-epochs`: **a batch collected from the uniform policy has all 64 advantages negative** (one long episode that ends in the pit). Use `driftRuns(g, { warmIterations: 10, seed: 31, lr: 0.03 })`, whose batch has 32 positive and 32 negative advantages. End of epoch 4 for epsilon 0.1 / 0.2 / 0.4:
  - ratio range [0.735, 1.316] / [0.719, 1.443] / [0.709, 1.579]
  - out-of-band fraction 0.59 / 0.31 / 0.14
  - plateau fraction (favourable side, zero gradient) 0.42 / 0.27 / 0.13
  - KL(old || new) per epoch: epsilon 0.1 0.0012, 0.0044, 0.0081, 0.0114; epsilon 0.2 0.0012, 0.0055, 0.0116, 0.0176; epsilon 0.4 0.0012, 0.0056, 0.0134, 0.0237
  - **A smaller epsilon gives a smaller drift here, but even epsilon 0.1 lets ratios reach 1.32 (outside its 1.1 band).** Clipping is not a wall.
  - At lr 0.1 ratios reach about 2.0, and the ordering of drift by epsilon is not monotone in every epoch.
- `clip-positive` / `clip-negative` are exact: for A = 1 and epsilon 0.2 the objective is 1.2 at both ratio 1.2 and ratio 1.5 (derivative 0 above 1.2). For A = −1 the objective is flat below 0.8, and at 1.4 it is −1.4 with derivative −1.

**Setup times (Node):**

| Run | Time |
|---|---|
| value iteration | 7 ms |
| gamma scan | 100 ms |
| stepReward scan | 190 ms |
| bandit testbed (3 strategies × 50 runs × 500 steps) | 37 ms |
| Q-learning, 300 episodes | 5 ms |
| Q-learning curves, 20 × 500 episodes | 30–45 ms per epsilon |
| REINFORCE, 10 × 1000 episodes | 97–111 ms |
| actor-critic, 10 × 1000 episodes | 86 ms |
| one REINFORCE run with 1000 snapshots | 19 ms |
| DQN, single 1000-update run | 60–80 ms |
| DQN compare | 3.7–9 s |
| PPO, 100 iterations | 66 ms |
| PPO drift runs | 3–8 ms |

Anything over about 30 ms goes through `kit.job`.

**Shell status (core agent).** `src/components/widgets/_rl/` (Shell.svelte, kit.ts, svg.ts, worker.ts) holds the generic figure. GridWorldLab, BanditLab and PPOLab are thin wrappers that use it, and the mode contract is at the top of `GridWorldLab/sim.ts`. See the core agent's report for which reference modes were finished.

## 7. Day 12 core results (measured)

Measured 2026-10-06 by the Day 12 core agent from `src/lib/align` (Node, probes in the scratchpad, deleted). **These numbers override the MDX briefs.** Re-probe before quoting anything not listed here, and re-probe at your exact props. Tests: `tests/unit/align-core.test.ts` (27, including assertions on every tuned headline effect) and `tests/unit/align-shells.test.ts` (16, the four reference modes).

**The bank (as built).** 10 prompts (8 train, 2 held-out), 8 candidates in this fixed order (colour `--viz-cat-1..8`): continues the list, terse correct, clear correct, padded correct, rambling wrong, terse wrong, refusal, hedged correct. phi = answers, correct, length (words / 10, read off the text), flattery, refuses, hedges. Gold `W_GOLD = [1, 2, -0.3, -1, -1, -0.5]`. Human labelers: Bradley-Terry (temperature 1) on gold plus `0.9 * min(length, 2.5)` (a saturating length bias; no flattery bias). Base model `THETA_BASE = [-2.5, 0, 0, 0, -1.5, 0]`; base offsets c(x, y) are N(0, 0.5) per candidate plus a mean of -10 for the rambling answer. Demonstrators, when right: clear 0.75, terse 0.15, hedged 0.1 (they never pad); when wrong: terse wrong.

**Designed exploit, honestly stated.** The rambling answer is 48 words, overconfident and wrong (not flattering: see deviations). The SFT model almost never writes it (log-odds 11.7 below clear correct; P < 1e-4), so it never appears in the 200 preference pairs (lengths seen in pairs: 1 to 28 words). The linear reward model learns +0.76 per 10 words and extrapolates to 48 words, so it scores rambling highest (4.72 on prompt 0, gold -0.44). Everything else is computed.

**sft.**
- `base`: P(continues the list) averaged over the 8 train prompts 0.575; P(a correct answer) 0.185; "continues the list" is the most likely response on all 8 prompts (refusal, 0.18 on average, is second). Prompt 0: P(continue) 0.71.
- `data` (demos 24, demoAccuracy 0.8, seed 2): exactly 19 of 24 correct (0.792; `makeDemos` uses exact counts), mean length 5.4 words, flattery 0.
- `train` (150 steps, lr 0.05): NLL 2.973 -> 1.431; P(continue) 0.575 -> 0.057. Theta (answers, correct, length, flattery, refuses, hedges) goes from [-2.5, 0, 0, 0, -1.5, 0] to [0.50, 0.85, -0.24, -3.27, -3.48, 0.03]. All six coordinates move at the same speed for the first ~25 steps (Adam); "correct" peaks at 1.30 (step 50) and settles back, "answers" keeps climbing. SFT policy averaged: continue 0.06, terse 0.30, clear 0.28, terse wrong 0.14, hedged 0.21; correctness 0.801, gold 2.273.
- `mask` (word bigram, 41-word vocabulary, 16 train + 4 held-out transcripts, Adam lr 0.1): counted tokens per example 12.6 (all) vs 5.9 (response only). Held-out response-token loss: start 3.714; step 100: 2.146 (all) vs 2.150 (response only); step 300: 2.256 vs 2.317; best over training 2.064 vs 2.000. **The gap is negligible and changes sign** (the plan anticipated this): show the mask and the counted-token numbers; do not claim response-only training is better here. 96–130 ms per run: `kit.job('trainBigram', 'all' | 'response')`.
- `heldout`: P(a correct answer) on "How many sides does a hexagon have?" 0.224 -> 0.811; on "What is 15 − 8?" 0.159 -> 0.679.
- `cap`: **recommend demos=40** (brief: unspecified): the demo sets are then exactly 0.600 / 0.800 / 0.950 correct and SFT correctness after 300 steps is 0.602 / 0.804 / 0.955. With 24 demos the sets are 0.583 / 0.792 / 0.958 and SFT ends at 0.586 / 0.797 / 0.962. Moment matching holds to within 0.02–0.04 on every feature at 300 steps (exactly at the optimum: tested to 1e-4 on an interior demo set).

**rlhf** (pairs seed 4, PPO seed 6, beta 0.1, 200 iterations, batch 64, 4 epochs, mini-batch 16, eps 0.2, Adam lr 0.04).
- `pairs` (200): human labels agree with gold 0.685 (69%), the longer response is chosen 0.485 (49%), 58 choices are longer-and-worse. Most pairs are terse vs clear (43), clear vs hedged (34), terse vs hedged (32), terse vs terse wrong (28).
- `reward-model` (200 steps, Adam 0.05): loss 0.693 -> 0.487; weights RM vs gold: answers 1.09 vs 1, correct 2.64 vs 2, length 0.76 vs -0.3, flattery -2.93 vs -1, refuses -3.22 vs -1, hedges -0.75 vs -0.5. Held-out pairwise accuracy (1000 pairs, seed 99) 0.680 (train 0.720; gold itself as a scorer gets 0.649 on the same labels).
- `rm-bias`: Spearman rank correlation RM vs gold 0.871 on the 68 candidates within the seen length range, 0.478 over all 80. Top RM candidate: rambling wrong (prompt 0), RM 4.72, gold -0.44.
- `ppo` beta 0.1: proxy / gold / KL 3.345 / 2.273 / 0 at iteration 0 and 4.076 / 2.808 / 0.802 at 200 (gold +0.535 over SFT). End probabilities: terse 0.14, clear 0.82, hedged 0.04. Gold rises 2.27 -> 2.63 (iteration 5) -> 2.75 (10), then wobbles between 2.66 and 2.83.
- `ppo-clip`: **recommend iter=0 or 1, not 20** (by iteration 20 the policy has settled: no ratio leaves the band). Per epoch 1..4, out-of-band fraction / favourable-side plateau fraction: iteration 0: 0.078/0.078, 0.391/0.359, 0.422/0.359, 0.359/0.328 (ratios 0.43–1.29 at epoch 4; 45 of 64 advantages positive); iteration 1: 0/0, 0.125/0.125, 0.375/0.344, 0.375/0.344 (ratios 0.56–1.34). Out-of-band always exceeds plateau: the extra samples sit outside the band on the penalised side and still get gradient. `record: [it]` keeps per-epoch ratios.
- `kl-sweep` final KL / proxy / gold: beta 0.5: 0.290 / 3.971 / 2.762; beta 0.1: 0.802 / 4.076 / 2.808 (best gold); beta 0.02: 13.01 / 4.719 / -0.440; beta 0: 13.01 / 4.719 / -0.439. beta 0.02 and 0 both end hacked.
- `ppo` beta 0 (reward hacking): gold peaks at **iteration 19, 2.819** (proxy 3.994, KL 0.426; policy there: terse 0.30, clear 0.63, hedged 0.07). P(rambling) is 0.00 at 20, 0.15 at 25, 0.99 at 30. Final: proxy 4.719, gold -0.439, KL 13.01, P(rambling) 1.000. Gold by iteration: 0: 2.27, 10: 2.76, 18: 2.81, 20: 2.81, 25: 2.31, 30: -0.42, then flat at -0.44; the proxy rises until 30 (4.00 at 20, 4.12 at 25, 4.71 at 30) and is flat after. The collapse takes about 10 of the 200 iterations.
- `overopt` beta 0, 400 iterations: gold peaks at sqrt(KL) 0.652 (iteration 19, gold 2.819); final sqrt(KL) 3.608, final proxy minus gold 5.159.
- Robustness: the beta 0 collapse plus a beta 0.1 gold above SFT holds for 8 of 8 PPO seeds; across preference-pair seeds it holds for 4 of 8 (the hack needs the RM's length weight to be large enough). Keep seed 4.

**dpo** (same 200 pairs; `trainDPO`'s lr defaults to 0.05 / beta, which makes the DPO path in v = beta (theta - theta_ref) the reward model's Adam path, step for step: tested).
- `optimal` beta 0.1: pi* gold 2.812, KL 0.994; prompt 0 pi* = terse 0.07, clear 0.91, hedged 0.02; max |PPO end - pi*| = 0.126 (terse 0.14 vs 0.04, clear 0.82 vs 0.92). Z(x0) = 2.1e17 (r / beta reaches about 47; Z depends on the reward's arbitrary offset, so quote it only with that caveat, or quote log Z). pi* gold / KL by beta: 1: 2.69 / 0.18; 0.5: 2.74 / 0.27; 0.2: 2.78 / 0.52; 0.1: 2.81 / 0.99; 0.05: 1.09 / 6.89 (rambling 0.53); 0.02: -0.44 / 13.0 (rambling 1.00). Dragging beta to 0.02 hands the policy to rambling wrong.
- `dpo-pair` (pair 0: "What is half of 30?", chosen terse correct (gold 2.97) over clear correct (2.85), beta 0.1, 40 steps, default lr 0.5): loss 0.693 -> 0.383, margin 0 -> 0.762, gradient weight sigma(-margin) 0.500 -> 0.318.
- `dpo-train` beta 0.1, 300 steps: loss 0.693 -> 0.487, gold 2.819, KL 0.968, held-out preference accuracy 0.680, max |DPO - pi*| 0.005. Gold by step 0 / 25 / 50 / 100 / 200 / 300: 2.27 / 2.19 / 2.70 / 1.97 / 2.81 / 2.82 (KL at step 100: 3.55): mid-run the length weight is temporarily large relative to "correct", so the policy visits the rambling answer and comes back. Show it; it is the RM fit's own path.
- `implicit`: beta (theta - theta_ref) = [1.05, 2.66, 0.74, -2.88, -3.78, -0.73] vs RM [1.09, 2.64, 0.76, -2.93, -3.22, -0.75]; Pearson correlation of implicit reward vs RM score over the 80 candidates 0.999. Largest weight difference: refuses (0.56; refusals are almost never in the pairs, so this weight is barely identified).
- `compare`: beta 0.1: PPO gold / KL 2.808 / 0.802, DPO 2.819 / 0.968; beta 0.02: PPO -0.440 / 13.01, DPO -0.440 / 13.02 (both hacked). **Chosen log-probability falls in DPO**: mean log pi(chosen) -1.402 -> -3.862 at beta 0.1 and -> -31.17 at beta 0.02 (never above -1.398); rejected -1.762 -> -15.30 / -86.92. The brief's "chosen likelihood falls" claim holds.

**constitutional-ai** (default principles answer, accurate, no-flattery, no-refuse; cautious off).
- `rules` prompt 0: violations per candidate 1 0 0 1 1 1 2 0; only terse correct, clear correct and hedged correct have none.
- `revise` (4 rounds, seed 3): starting from SFT samples only 12 of 64 starts violate anything (0.188 -> 0.109 violations; 248 of 256 rounds find nothing to fix). **Recommend sampling the starts from the base model** (`slCAI(sft, { from: THETA_BASE })`): 1.156 -> 0.766 violations, 206 of 256 rounds find nothing to fix. A chain from "continues the list" on prompt 0: no-flattery (nothing) -> answer (-> terse wrong) -> accurate (-> terse correct) -> answer (nothing).
- `sl` (SFT on 64 revisions of SFT samples, 150 steps): expected violations 0.207 -> 0.110; gold 2.273 -> 2.508; terse wrong 0.14 -> 0.09. **P(padded) is 0.005 -> 0.000**, so the brief's padded comparison is moot: compare terse wrong instead.
- `pairs` labels ai (same 200 pairs): AI agrees with human 0.71, AI with gold 0.705 (71%), human with gold 0.685 (69%). Ties (equal violations) go to a seeded coin; many hedged-vs-clear pairs are coin flips.
- `ppo` labels ai, beta 0.1: proxy / gold / KL 4.775 / 2.611 / 0.403 vs human labels 4.076 / 2.808 / 0.802. **AI labels end lower in gold at beta 0.1**: no principle penalises hedging, so the policy hedges (hedged 0.36). But **AI labels avoid the length exploit even at beta 0**: final gold 2.173 and P(rambling) 0.000, vs -0.439 and 1.000 for human labels. Recommend the CAI ppo step show beta 0 (or both) if its point is the exploit.
- `edit` (single toggles; 6 runs of about 45 ms, 250–330 ms total): P(refuse) + P(hedge) / gold: default 0.358 / 2.611; without answer 0.127 / 2.866; without accurate 0.225 / 2.602; without no-flattery 0.000 / 1.153 (the policy goes all-in on padded correct); without no-refuse 0.016 / 2.858; with cautious 0.999 / 2.169 (all hedged; refusal stays 0 because refusing breaks two other principles). SFT is 0.217.

**rlvr** (worked lines now read "7 + 5 = 12, write 2, carry 1").
- `verify`: six fixed examples (47+85 worked, 47+85 unworked = 22, 36+21 unworked, 58+67 worked with a guess list, 365+278 one column = 533, 9+4 worked): strict 3 of 6 pass, lenient 4 of 6 (the guess list passes).
- `sample` (G 8, problems 6, seed 2, start policy): problems 7+3, 80+28, 923+155, 5211+4134, 1+9, 95+16; pass@1 0.250, pass@8 0.833 over the 6 problems; one zero-signal row (923+155, all fail; 5211+4134 has 6 of 8 pass). Largest advantage 2.646 (one pass among 8). No 3-digit row has a pass at seed 2: pick another seed if the brief's TRY is kept.
- `grpo`: advantages (r - mean) / std per group; uniform rows give exactly 0.
- `train` strict (150 iterations, G 8, seed 5, Adam 0.1): strict accuracy per digit count 1 / 2 / 3 / 4 at iterations 0 -> 50 -> 150: 0.54 -> 0.99 -> 1.00, 0.34 -> 0.99 -> 1.00, 0.20 -> 0.98 -> 1.00, 0.09 -> 0.95 -> 1.00.
- `length`: mean words 5.7 -> 10.0, 7.3 -> 18.8, 8.1 -> 27.7, 8.4 -> 37.2 (lines 1.33 -> 2.00, 1.51 -> 3.00, 1.60 -> 3.99, 1.63 -> 4.99). Length grows more for more digits.
- `train` lenient: lenient reward ends at 1.00 for every size; **the guess-list hack wins only for 1- and 2-digit problems** (list share 0.56 and 0.71 at the end; strict accuracy 0.54 -> 0.44 and 0.34 -> 0.29). For 3 and 4 digits four guesses rarely cover the slips, so working out wins (list 0.02 / 0.01, strict 0.98 / 0.99). Show a 2-digit sample response.
- `domains`: pass@8 at the start (mean over digit counts) 0.653 vs pass@1 at the end 0.997; the unverifiable rows change by exactly 0.

**Setup times (Node) and where they run:**

| Run | Time | Where |
|---|---|---|
| sftTheta / trainSFT 150 steps | 10 ms | kit.memo (SFTLab `sftRun`) |
| three 300-step cap runs | 13 ms | kit.memo |
| trainBigram 300 steps | 96–130 ms per mask | kit.job |
| pairs + fitRM | 4–9 ms | kit.memo (PreferenceLab `prefData`) |
| trainAlignPPO 200 iterations | 43–72 ms | kit.job (`ppoRun`) |
| trainAlignPPO 400 iterations (overopt) | 75–95 ms | kit.job |
| trainDPO 300 steps | 43–54 ms | kit.job |
| slCAI | 5 ms | kit.memo |
| edit: 6 RLAIF runs | 250–330 ms | one kit.job per toggle set |
| trainGRPO 150 iterations | 44–87 ms | kit.job |
| evalTable (first call) | about 15 ms | cached in the module |

**Honest deviations from §1 and the briefs.**
- The rambling candidate is not flattering. With flattery, the RM's flattery weight (identified from about 3 pairs that contain the padded answer) decided whether the exploit existed and flipped with tiny changes; without it the exploit rides on the length weight.
- The exploit needs the rambling answer to be very rare under SFT (base offset -10) so that beta 0.1 cannot reach it while beta 0 still finds it through the growing length weight. pi* at beta 0.05 is already half rambling: the beta scale is sensitive. The 0.5 / 0.1 / 0.02 / 0 ordering holds at pair seed 4 (4 of 8 pair seeds).
- PPO uses Adam lr 0.04 (Day 11 used 0.03); it gives the latest visible gold peak while beta 0.1 still ends near pi*. A per-prompt running mean (rate 0.2, initialised with the prompt's first batch mean) stands in for the value head, as planned.
- Mask gap negligible; AI labels hedge more at beta 0.1; the lenient hack is limited to small problems; revise from SFT samples rarely has anything to fix; ppo-clip needs an early iteration; padded is almost never written by SFT.
- Align's PPO surrogate helpers are `alignSurrogate` / `alignSurrogateGrad` (no name clash with `src/lib/rl` in the shared worker).

**Shared code.** `src/lib/align/` (bank, policy, sft, bigram, constitution, reward, ppo, dpo, optimal, verifier, index). `src/components/widgets/_align/draw.ts`: candidateBars (ghost reference bars, ticks, real-state `values` while bars tween), featureBars, rewardChart (proxy, gold, faint comparison gold, peak mark), responseCard, wrap / clip / charsFor, candColor. The only `_rl/` change: `worker.ts` now exposes `src/lib/rl` and `src/lib/align` (plus one doc line in `kit.ts`). Shells: SFTLab, PreferenceLab, CritiqueRevise, VerifierRL, each a 13-line `.svelte` plus `sim.ts` (contract, align exports, costs, data helpers `sftRun`, `prefData` / `ppoRun`, `principlesOf`, `tick` / `cross`). Finished reference modes: `SFTLab/m-base`, `PreferenceLab/m-pairs` (labels human | ai, clickable squares), `CritiqueRevise/m-rules`, `VerifierRL/m-verify` (verifier strict | lenient).

**Mode-file ownership (each = `m-<mode>.ts` + `m-<mode>.meta.ts`; test file in brackets):**
- sft: `SFTLab/m-data`, `m-train`, `m-mask`, `m-heldout`, `m-cap` [`tests/unit/sft-lab.test.ts`]. `m-base` exists (core).
- rlhf: `PreferenceLab/m-reward-model`, `m-rm-bias`, **`m-ppo`** (must accept `labels` and `compare="human"`, drawing that run's gold with rewardChart `goldCompare`, and any `beta`), `m-ppo-clip`, `m-kl-sweep`, `m-overopt` [`tests/unit/preference-lab.test.ts`]. `m-pairs` exists (core; serves rlhf and constitutional-ai).
- dpo (after rlhf, since it shares PreferenceLab): `PreferenceLab/m-optimal`, `m-dpo-pair`, `m-dpo-train`, `m-implicit`, `m-compare` [`tests/unit/preference-dpo.test.ts`].
- constitutional-ai: `CritiqueRevise/m-revise`, `m-sl`, `m-edit` [`tests/unit/critique-revise.test.ts`]; uses PreferenceLab `pairs` (labels ai) and `ppo` (labels ai, compare human) without editing them. `m-rules` exists (core).
- rlvr: `VerifierRL/m-sample`, `m-grpo`, `m-train`, `m-length`, `m-domains` [`tests/unit/verifier-rl.test.ts`]. `m-verify` exists (core; pass `verifier="lenient"` for the gamed-verifier step).
- evaluations: its own EvalLab (not part of this core).

## 8. Day 12 continuation and final review

The interrupted implementation was resumed on 2026-10-06/07. All six lessons are written; final coordinator evidence and review status are in [day-12-review.md](day-12-review.md).

The planned bundle fallback was needed: PreferenceLab initially measured 43.44 KB gzip. The five DPO modes listed above now live in **`DPOLab/`**, with their existing mathematics and shared preference data retained. RLHF and Constitutional AI continue to share PreferenceLab. Actual lesson tests are `preference-lab-rlhf.test.ts` and `preference-lab-dpo.test.ts`. The pure clipping helper moved to `src/lib/rl/surrogate.ts`; `src/lib/rl/ppo.ts` re-exports it for compatibility. Every Day 12 island now passes the 40 KB dependency-closure budget.

**Owner instruction, 2026-10-07:** local verification servers no longer need additional permission. Finish and review Day 12, then stop. Obtain explicit permission before starting Day 13 or its core agent.
