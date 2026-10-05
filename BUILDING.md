# Build Playbook (for whoever continues the build)

This is the coordinator's handbook. Anyone continuing the build reads it: a new Claude session, another agent, or a human. `LESSONS.md` tells you **how to write one lesson**. This file tells you **how to run the build**: what's done, what's next, how to delegate, how to review, and the judgement calls made so far. Read `CLAUDE.md` → `IDEA.md` → `LESSONS.md` → this file → `dos&donts/` (every file in it) before doing anything.

---

## 1. Status (update this table as you go)

Status values: **accepted** means reviewed by screenshot and fixed. **draft?** means written but never reviewed by the coordinator. **stub** means not written yet.

| Day | Track / lessons | Status |
|---|---|---|
| Math primer | math/{vectors, dot-product, matrices, derivatives, chain-rule, probability} | accepted |
| 1 | foundations/{what-is-a-model, data-features-labels, loss-functions, linear-regression, gradient-descent} | accepted |
| 2 | foundations/{logistic-regression, softmax-cross-entropy, decision-boundaries, overfitting-bias-variance, train-val-test, regularization} | accepted |
| 3 | classic-ml/{k-nearest-neighbors, decision-trees, ensembles, svm-kernels, k-means, pca, classic-vs-deep} | accepted + wave-checked (2026-10-01) |
| 4 | neural-networks/{neuron, activation-functions, mlp, hidden-layers, universal-approximation} | accepted + wave-checked (2026-10-01; original five widgets preserved) |
| 5 | neural-networks/{computational-graphs, chain-rule-visually, forward-backward-pass, backprop-mlp, autodiff, vanishing-exploding-gradients} | accepted + wave-checked (2026-10-01) |
| 6 | neural-networks/{optimizers, learning-rate-schedules, weight-initialization, normalization, dropout-regularizers, loss-landscapes} | clarity revision 2 by coordinator (2026-10-02, see §5a); owner clarity review pending |
| 7 | cnns/{convolution, channels-stride-padding, receptive-fields, resnets, feature-visualization} | accepted + wave-checked (2026-10-03); owner clarity review pending |
| 8 | sequences/{why-memory, rnn-cell, unrolling, bptt, vanishing-gradients-time} | accepted + wave-checked (2026-10-03, coordinator review of the early-wave drafts, see the Day 8 note in §6); owner clarity review pending |
| 8 | sequences/{lstm-gru, seq2seq-attention} | accepted + wave-checked (2026-10-03); owner clarity review pending. Written by coordinator. lstm-gru reuses RNNUnroller (gated view redrawn as belt + gate meters; new opt-in `forgetBias`/`showForget` props, default unchanged). seq2seq-attention has a new Seq2SeqAttention widget: two tiny tanh encoder-decoders (fixed vector vs Bahdanau attention) trained per page in workers on string reversal. Decoder gets no previous-output input (with it the alignment learned a one-step shift). exp/tanh are built from + - * / and Adam uses running products instead of pow, so Chrome and Node train bit-identically and prose numbers match. 11 tests in `tests/unit/seq2seq-attention.test.ts`. Wave evidence: autofixer 0 issues on both components; Astro check 0/0/0 (after typing `tanhInPlace` generically in seq2seq `sim.ts`); **537 tests pass**; 101-page build, no source maps; closures Seq2SeqAttention **29.0 KB**, RNNUnroller **32.4 KB** gzip; check-page `--steps --mobile` clean on both lessons in light and dark (no errors, no overflow, about 75 fps). check-page screenshots of seq2seq can catch the in-browser training before it finishes; that is expected |
| 9 | transformers/{tokenization, embeddings, self-attention, multi-head-attention, positional-encodings, transformer-block, causal-masking} | accepted + wave-checked (2026-10-04); owner clarity review pending. See the Day 9 note in §6 |
| 10 | llms/{pretraining, sampling, scaling-laws, inference-kv-cache, emergent-icl, hallucination} | accepted + wave-checked (2026-10-05); owner clarity review pending. See the Day 10 note in §6 |
| 11 | alignment/{rl-basics, q-learning, policy-gradients, deep-rl, ppo} (Reinforcement learning) | accepted + wave-checked (2026-10-05); owner clarity review pending. See the Day 11 note in §6 |
| 12 | alignment/{sft, rlhf, dpo, constitutional-ai, rlvr, evaluations} (Alignment) | skeleton (2026-10-03), `plans/day-11.md` (one plan for both days; split from old Day 11 on 2026-10-03, owner decision) |
| 13 | graphs-generative/gnn | draft? (GNNMessagePassing widget). Review checklist in `plans/day-13.md` §0 |
| 13 | graphs-generative/{autoencoders, vaes, gans, diffusion, flow-matching} | skeleton (2026-10-03), `plans/day-13.md`. LatentSpace, GANDuel, FlowField and GenCommon folders are empty; DiffusionDenoiser is reusable with the listed fixes |
| 14–15 | applied/*, frontier/* | skeleton (2026-10-03), `plans/day-14.md`, `plans/day-15.md`. Day 15 must still be researched from primary sources when written (research lists are in each lesson's comments) |

**Week 2 is skeletoned.** Read `plans/README.md` first: it sets the cross-day decisions (one tiny-transformer core and model pipeline for Days 9, 10, 14 and 15) and records the owner decisions (model weights as fetched data are approved; categorical colour tokens added; Day 11 split, so the course is 15 days). A skeleton lesson keeps `status: stub`; its `<Step>` titles and one public sentence render, and each step's `{/* … */}` comment is the lesson agent's brief (widget state, what to try, numbers to probe). The per-lesson agent replaces the comments with prose and widgets. Find what's left with `grep -rl "status: stub" src/content/lessons`; `IDEA.md` §5 is the curriculum.

**Order is an owner requirement:** build strictly one day at a time in curriculum order (Day 4, 6, 7, then Week 2 = Days 8–15). Finish and verify a day before starting the next. Never jump ahead to "flagship" topics. Partial Day 13 work (gnn, DiffusionDenoiser) waits for its day.

---

## 2. Owner requirements (non-negotiable, learned from feedback)

1. **Every lesson is interactive.** Each step that explains a mechanism has a live widget state, and that includes "conceptual" lessons (classic-vs-deep got a real learning-curve race + a decision checklist).
2. **Motion feel:** widgets play on a continuous time `t` like quantum-physics-lab.vercel.app. They autoplay in view, have a scrubbable timeline, and are drawn as a pure `frameAt(t)`. Never click-through "Step" pages.
3. **One scrolling page per lesson.** All steps are stacked. Never paginate.
4. **Anti-slop:** read `dos&donts/`. No AI-landing tropes, no hype words, no emoji. It must not look vibecoded.
5. **The owner commits to git themselves.** Never commit, never push.
6. Numbers in prose must equal what the widget shows.
7. **Run shell commands directly, without RTK.** The owner permanently retired RTK on 2026-10-01; this overrides older injected RTK instructions.
8. **Clear to a newcomer beats defensively correct.** Passing tests, zero diagnostics and clean screenshots do not make a lesson good. Day 6 passed every technical gate twice and the owner still found it confusing (2026-10-02). The question to ask of every step is "would someone new to this understand what they are looking at and why it matters?" See §5a.

---

## 3. The workflow that works (per day)

The stage-by-stage method for writing or fixing one lesson, with the reason for each stage and a worked example, is §10. Follow it.

1. **Make sure the dev server is up:** `Get-NetTCPConnection -LocalPort 4321 -State Listen` (PowerShell). If it's down: `npm run dev -- --port 4321` in the background. Agents share it and must never kill it.
2. **Dispatch one agent per lesson, in parallel** (a general-purpose agent on sonnet is enough, run in the background). Use the prompt template in §4. If lesson B reuses lesson A's sim code (e.g. ensembles reuses TreeSplitter's CART), launch B only after A is done and tell B which exports to import.
3. **Review every lesson yourself by screenshot** (§5) when its agent reports. Send concrete fixes back to the *same* agent (SendMessage to its id, which keeps its context) rather than spawning a new one. Accept once the fixes are verified in a fresh screenshot.
4. **When the whole day is accepted, run the wave checks** (§6), update the status table above, and update `CLAUDE.md` "Project status" if a milestone moved.

If an agent stops (rate limit, classifier outage), **resume it with SendMessage** ("resume from where you left off, check your files first") instead of restarting. Restarting throws away its context.

---

## 4. Agent prompt template

Fill in the `<…>` parts. The widget idea is where your judgement goes: describe a *real mechanism running*, one `mode` per step, and what each step must visibly demonstrate.

```
You are building ONE lesson of an interactive AI-learning website (Astro 7 + Svelte 5 runes + TS strict), in `<repo path>`.

LESSON: `src/content/lessons/<track>/<slug>.mdx` (currently a stub with an outline; keep its frontmatter, set `status: draft`, cover every outline bullet).
WIDGET: create `src/components/widgets/<Name>/` (<Name>.svelte + sim.ts [+ worker.ts if training is heavy]) and a unit test `tests/unit/<name>.test.ts`. <Reuse notes: which existing widget/sim exports or src/lib/nn to import.>
Widget idea: <the mechanism, what animates over t, one mode per step, sliders/toggles, readouts, what each step must demonstrate>.

BEFORE WRITING, READ FULLY: `CLAUDE.md`, `LESSONS.md` (mandatory), `src/components/widgets/_shared/README.md` (reuse FigureCard, Timeline, createLoop, readColors), everything in `dos&donts/`, and a finished reference lesson + widget close to yours: <pick one or two, e.g. classic-ml/decision-trees + TreeSplitter, foundations/logistic-regression + LogisticFit>. Match their structure, prose style and code idiom. Token colours only. Seeded RNG. All math in sim.ts (no DOM), visuals from a pure frameAt(t); autoplay on view, Play/Pause + scrubbable timeline + speed; reduced motion starts paused.

Lesson = ONE scrolling page of several <Step>s, each with a widget instance (different `mode` props). Numbers in prose MUST match what the widget shows (compute them from sim.ts, never guess). In MDX, bare `<`, `{`, `}` break the build; math goes in `$…$`.

CLARITY RULES (BUILDING.md §5a; these are where past agents failed):
- Each core step: plain-language intuition first, define every term, then ONE concrete thing to look at or try, quoting a real number the widget shows. Write like a teacher, not a widget manual. At most one short caveat where it really matters; no "this is a toy, not a claim…" disclaimers, no meta-notes about how the widget computes ("not interpolated", "recorded", "seeded").
- Before writing a claim about what a demo shows ("momentum overshoots", "warmup helps"), print the simulation's numbers and check. If the sim shows something else, change the prose (or the demo), never the other way round.
- No repeated figures: misconceptions / check-yourself / going-deeper steps get NO widget, and no two core steps may show the same widget with identical props.
- Widget labels in plain words (no "recorded update", "reduction group", "÷ size"). Defaults must match the lesson's claim at that step (e.g. gain 1 / shift 0 until the gain-and-shift step).
- If the effect spans orders of magnitude, use a log axis; if it is small, tighten the domain. Never a tiny shape in a huge empty plane.
- Temporary probe scripts go in the scratchpad or are deleted immediately: a `tests/unit/zz-*.test.ts` left behind runs in every `npm test`. Widget folders hold only `<Name>.svelte`, `sim.ts`, optional `worker.ts`.

VERIFY before reporting done:
1. Svelte MCP autofixer (mcp__svelte__svelte-autofixer, load via ToolSearch) on your .svelte file until clean (pass it in parts if large), AND 0 compiler warnings.
2. `npx vitest run tests/unit/<name>.test.ts` passes; `npm run check` has 0 errors in your files.
3. Dev server is already on port 4321 (shared: do NOT kill it, do NOT run `npm run build`). `node scripts/check-page.mjs learn/<track>/<slug>/ --steps` → errors: [], ~60+ fps, overflowX: false.
4. Screenshot every figure at SEVERAL timeline points (not just t=0), desktop 1280 and mobile 390, and LOOK critically: readouts agree with the canvas, the active timeline marker matches the state shown, no overlapping/clipped labels, readable plot sizes, narration matches visuals, the demo visibly shows the claimed effect. Fix and re-check.
Only touch your own lesson, widget folder and test file. Do not commit. Report: files changed, test count, concerns, final screenshot paths.
```

---

## 5. Review checklist (what agents actually get wrong)

Open the agent's screenshots (Read the PNGs). Look at **mid-animation** frames too, not just the final state. These are real bugs caught in review so far, most common first:

- **Readout disagrees with the canvas.** "support vectors 1" while 2 points are ringed. "n = 62 rows" while the panels say n = 64. A readout throttled to 10 Hz lags the drawing. Everything shown must come from the same frame. If values interpolate between precomputed runs, snap the readout to a real run.
- **Active timeline marker contradicts the state.** The marker says "Pass 1" while the readout says "pass 4". It says "75-NN" while k = 51. Timeline highlights the last marker passed, so marker labels must name a *segment* or be placed exactly where the state changes.
- **Narration doesn't match the data.** It said "two moons", but the data was two blobs.
- **Charts clip data.** The hinge-loss chart range hid the worst point, so the chart and the readout's sum disagreed. Points were cut in half at plot edges, so pad the domain.
- **Overlapping labels.** Timeline marker labels that sit too close, value labels colliding with dots.
- **Wasted or cramped layout.** Big empty areas, or tiny plots on mobile.
- **Demo doesn't show the claim.** Check that the effect is visible, not just asserted in text.
- **Number sanity.** Recompute a couple of prose numbers in your head from the picture: sums, percentages (93% of 28 = 2 wrong), eigen identities, etc.

Also check the report: autofixer run (agents often skip it because the file is big, so ask them to pass it in parts), tests pass, check-page clean. `ponytail:` comments mark deliberate shortcuts. They're fine if they're honest.

### 5a. Clarity review (the Day 6 lesson, 2026-10-02)

Day 6 was delegated to Codex (GPT-6 Luna). It passed 462 tests, zero diagnostics, budget checks and 558 screenshots, and was still confusing. A coordinator review found these, all fixed in the second Day 6 revision. Check every new lesson for each one:

- **Manual-plus-disclaimer prose.** Paragraphs spent on control names ("Recorded update chooses a different batch; Example differences changes…") and hedges ("not a universal ranking", "coordinates are never clamped", "these hand-set objectives are a mechanism demonstration"). Compare with `classic-ml/decision-trees.mdx`: intuition, then the picture, then one thing to try.
- **Claims the simulation contradicts.** The race prose implied SGD was slow; the sim had SGD reaching the valley first (loss 1.046 after 10 updates) and Adam slowest. The warmup step implied warmup helped; on that bowl it only delays. Probe the sim, then write.
- **Duplicate figures.** Misconceptions, check-yourself, going-deeper and summary steps each embedded another copy of the widget (8 figures, 5 repeats, in weight-initialization). Pairs of core steps used identical props (schedules had 7 figures but only 4 distinct states). Fix with props such as `unstable`, `scaled` or `only={[…]}` that give each step its own state.
- **The effect is invisible on the chosen axis.** Linear bars made 0.01× and 10⁻⁶× look the same (initialization). A linear loss axis hid the noise floor that batch size and decay change (schedules). Use log axes around the reference value.
- **No context for a single update.** Optimizer arrows were drawn on an empty plane, so "downhill" was invisible. Now local loss contours sit behind the arrow, and a "read the slope" phase shows the uphill gradient before it flips.
- **Defaults that contradict the step.** Normalization used gain 1.2 and shift 0.3 everywhere, so step 1's "mean zero" claim was false on screen.
- **Off-by-one readouts and point-named markers.** The worked box said "update 13" while the run list said "update 12/80". The "Start" marker stayed highlighted at update 18 of 120. Readouts showed layer-12 results before that bar had appeared.
- **Leftovers.** Scratch tests in `tests/unit/zz-*.test.ts` (one rewrote `.scratch-out.txt` in the repo on every `npm test`) and per-widget `verify.mjs` files.

**Delegating to another model (Codex, GPT, a cheap subagent):** hand it this file, `LESSONS.md` and the §4 template, clarity rules included. Its self-reported verification is evidence of *technical* health only. The coordinator still reads the screenshots and the prose with the §5/§5a lists before the status table says "accepted".

The Astro dev toolbar floats over the bottom of screenshots. It's dev-only, so ignore it.

---

## 6. Wave checks (coordinator only, after a day is accepted)

```bash
npm test                          # all unit tests
npm run check                     # 0 errors
npm run build                     # only the coordinator builds; agents must not (dist/ collisions)
node scripts/check-page.mjs learn/<track>/<slug>/ --steps            # every lesson of the day
node scripts/check-page.mjs learn/<track>/<slug>/ --steps --mobile
```

After building, check the island sizes in `dist/_astro/` (gzip each widget's chunk and its shared JavaScript dependency closure). Each must be under 40 KB gz. Then update the status table in §1 and the repository handoff documents. Do not update personal agent memory unless the owner explicitly asks.

Last recorded totals (2026-10-01, after Day 4): 402 tests in 39 files pass; Astro check has 0 errors, 0 warnings and 0 hints; static build produces 101 pages. All five Day 4 lessons pass desktop/mobile light/dark walkthroughs (0 browser errors, about 75 fps at this machine's display rate, no horizontal overflow). Svelte compiles every Day 4 widget with 0 warnings.

Day 4 continuation review: finished the MLP and hidden-layer lessons from partial files; reviewed the neuron, activation and approximation drafts. Fixed same-frame captions, rounded-number explanations, activation transitions, timeline labels, reduced-motion resets, worker failure handling and the shared header at 320 px. The approximation timeline traces a real network snapshot instead of blending two networks while reporting only one network's unit count/error. Its maximum error is explicitly a sampled measurement, not a rigorous continuous-domain bound.

`node scripts/verify-day4.mjs` checks every figure at three timeline positions in both themes at 1280 and 390 px, keyboard seeking, reduced-motion pausing/reset, normal autoplay/pause, live theme switching and expanded mathematics. `--base http://127.0.0.1:4322 --width 320` also passed against the production preview. Reports and reviewed screenshots are under `.playwright-mcp/check/day4-acceptance/`; per-lesson review captures remain in their usual check folders.

Day 8 review (2026-10-03, coordinator, no subagents): the five early-wave drafts (why-memory, rnn-cell, unrolling, bptt, vanishing-gradients-time) were reviewed like new work. Every number in their prose was re-derived from `RNNUnroller/sim.ts` and `SeqOrderDemo/sim.ts` with a deleted probe; all matched except one rounding (bptt sum of norms 0.628 → 0.629). Fixes: RNNUnroller narration and inspector now update every frame (they lagged the canvas by up to 100 ms, so unrolling showed "Step 1" while step 2 was highlighted); on the backward pass the narration names the earliest step whose bar is fully drawn, not the step the gradient is still travelling to; forward-only figures no longer list "gradient" in the legend. Duplicate figures removed: rnn-cell steps 2, 3 and 6 rendered the same picture (step 3 now pins step 3, step 6 hides the panel), and unrolling steps 1 and 2 were both the fold animation (step 1 now loops only the unroll, step 2 shows the three-call chain for "abc" that its $f(f(f(\cdot)))$ formula describes). Smaller prose fixes: rnn-cell step 4 got a real thing to try, the s = 3 slope sentence in vanishing-gradients now quotes the pinned step's values, and SeqOrderDemo's recurrent label uses $W_{xh}$, $W_{hh}$ like the rest of the day. Evidence: **537 tests**, Astro check 0/0/0, 101-page build without source maps, closures SeqOrderDemo **27.1 KB**, RNNUnroller **32.4 KB**, Seq2SeqAttention **29.0 KB** gzip, check-page `--steps` and `--mobile` clean on all seven Day 8 lessons (about 75 fps, no overflow, no errors), autofixer 0 issues on the changed components.

Day 9 (2026-10-03/04, coordinator plus one sonnet agent per lesson; the self-attention agent also wrote multi-head because it owns AttentionExplorer): widgets EmbeddingSpace, AttentionExplorer (13 modes), PositionLab, BlockFlow, CausalMask on the shared `src/lib/transformer/` core and the fetched `public/models/tiny-lm*` weights. Review fixes, most common first: readouts and labels showing tweened in-between numbers instead of real states (block stream, causal-mask softmax row 0.308 vs the real 0.288, multi-head compare labels on the previous word's bars; rule now: numbers switch at the segment boundary, only geometry eases or cross-fades); unverified causal sentences (the "first LayerNorm" explanation, "this is how sheep carries plural"); position numbering that disagreed between lessons (names or 0-based with [start] = 0 only); monospace narration sentences; mobile canvas text shrunk to 6–7 px (now drawn at CSS size, ≥ 11 px). Budget: AttentionExplorer first measured 47 KB, so it now loads one lazily imported chunk per mode (`m-<mode>.ts`); `scripts/check-widget-budgets.mjs` reports the worst single island for such widgets. `eigSym`/`covariance` moved to `src/lib/math/linalg.ts` (PCAProjector re-exports them) so EmbeddingSpace no longer pulls the whole PCAProjector sim. Evidence: **628 tests in 59 files**, Astro check 0/0/0, 104-page build without source maps, closures TokenizerLab 27.0, EmbeddingSpace 38.0, AttentionExplorer 39.3 (worst mode, explorer), PositionLab 35.7, BlockFlow 36.9, CausalMask 35.4 KB gzip; check-page `--steps` and `--mobile` clean on all seven lessons in light and dark (no errors, no overflow, about 68 fps with a build running). Every prose number was checked against the shipped weights by the agents, and the coordinator spot-recomputed the softmax rows, parameter counts and sinusoid periods.

Day 10 (2026-10-04/05, coordinator plus one opus core agent and one sonnet agent per lesson): the core agent added `src/lib/lm/` (charset, samplers, cost), the Alice corpus (`scripts/data/`), and the `house`, `scaling`, `icl`, `icl-sweep` and `facts` jobs to `scripts/train-models.ts`, all on the Day 9 transformer core; measured results are in `plans/day-10.md` §6, which overrides the briefs. Widgets: PretrainLab, SamplingLab (worker), ScalingLab, KVCacheLab, ICLLab (worker), HallucinationLab (worker). Review found main-thread freezes the old gates missed (KV cache 33–37 s per figure, and 300 ms–12.5 s on four Day 9 widgets): weights in plain `$state` became deep proxies; fixed with `$state.raw`, shared memoised runs and chunking, and check-page now fails any step stall above 100 ms (§8). Other review fixes: monospace narration, mobile canvas text under 11 px, tweened numbers, a 39-symbol miscount, unclear log axes. Honest deviations kept in the prose: no clean induction head, a non-monotone ICL size sweep, no interior IsoFLOP optimum (one-epoch data cap), no facts "unknown" model. Evidence: **715 tests**, Astro check 0/0/0, 104-page build without source maps, Day 10 closures PretrainLab 38.8, SamplingLab 37.2, ScalingLab 30.9, KVCacheLab 37.3, ICLLab 35.2, HallucinationLab 35.4 KB gzip; check-page `--steps` and `--mobile` clean on all six lessons (75 fps, no overflow, worst step stall 67 ms), and Day 9 rechecked with the stall gate (worst 67 ms).

Day 11 (2026-10-05, coordinator plus one opus core agent and one sonnet agent per lesson, all five in parallel): the core agent built `src/lib/rl` (gridworld, bandit, tabular, DQN, PPO; 30 tests) and three thin shells, GridWorldLab, BanditLab and PPOLab, on a shared `src/components/widgets/_rl/` (Shell.svelte, kit.ts with `memo`/`job` worker helpers, svg.ts). Each mode is `m-<mode>.ts` plus a bundled `m-<mode>.meta.ts` (reserves size, CLS 0), so lesson agents owned disjoint files and ran in parallel; mode-local helpers (`q-common.ts`, `dqn-common.ts`, `pg-runs.ts`, `ppo-batch.ts`) sit beside the modes. Measured results in `plans/day-11.md` §6 override the briefs; honest deviations kept in the prose: Q-learning far from converged at the brief's settings (demo uses ε 0.3), the baseline lowers gradient variance 2–6× but not the REINFORCE learning curve, actor-critic only faster early, DQN divergence only with Adam lr 0.1, the replay/target ordering only at lr 0.001 and reversed at lr 0.01, PPO clipping bounds neither ratios nor KL. Review fixes: a backwards sentence about the softmax update, an unprobed cross-lesson comparison, colour words that named the wrong hue, the PPO drift figure reusing the advantage-sign colours for ε, faded rows the narration quoted, a readout one pull behind its narration, a mobile table that cut off the discussed step, `1e-1` log ticks, the dqn-target overshoot explained from a probe (the move is in 5 of 49 batches; shared weights cause 0.30 of the 0.36 rise), and a policy-gradients test that only printed numbers (now asserts them). Dev-server gotcha: `import.meta.glob` can go stale when new mode files appear; touch the shell `.svelte` to refresh. Evidence: **811 tests in 73 files**, Astro check 0/0/0, 104-page build without source maps, worst-mode closures GridWorldLab **39.7 KB** (24 modes; at the limit, so new GridWorldLab modes must be lean), BanditLab 31.1, PPOLab 36.0 KB gzip; check-page `--steps` and `--mobile` clean on all five lessons (68 fps, no overflow, worst step stall 58 ms).

Day 4 widget chunks are 5.5–9.7 KB gzipped; their JavaScript dependency closures are 26.0–35.0 KB, below the 40 KB target. SpaceFolding's worker chunk adds about 2.0 KB. No production source maps were emitted. These checks do not establish the separate Lighthouse, LCP, INP or full accessibility release gates. Lessons retain `status: draft`, following the authoring workflow; “accepted” here records coordinator review, not production release approval. Day 6 was next at this checkpoint; see the later Day 6 acceptance below.

---

## 7. Decisions and conventions made so far

Day 5 continuation review (2026-10-01): all six existing lessons were audited before Day 6 began. Fixed the expression overshoot example (eta 0.1 produces c = 9.2 and loss = 121.8816), scalar-loss/autodiff wording, residual/normalization guarantee claims, chain-rule product reveal and loop endpoint, mobile observer targets and overlapping network weight labels. Forward values and gradient labels now reveal exact computed numbers rather than counting through fictional intermediate derivatives; update loss is recomputed at displayed weights. Long inline equations and code remain within the mobile page. Missing chain-rule mechanism figures and the graph fan-out demonstration reuse the existing widgets.

Day 5 wave evidence: 405 unit tests in 39 files pass, Astro diagnostics are zero errors/warnings/hints, and the static build produces 101 pages. All six lessons passed light/dark 1280/390 px multi-time checks, keyboard seeking, reduced-motion pause/reset, autoplay/pause, theme switching and expanded mathematics. Fresh reviewed screenshots and per-lesson reports live under `.playwright-mcp/check/day5-acceptance/`. `node scripts/verify-day4.mjs --day 5` selects this day (default remains Day 4). Day 5 widget chunks are 4.4–7.6 KB gzip; including their shared imports, 24.2–32.8 KB. `node scripts/check-widget-budgets.mjs BackpropViz ChainRuleLinks AutodiffTape GradientDepth` reproduces the budget check after a build. Release performance/accessibility gates remain unmeasured. The 2026-10-02 continuation also checked all six lessons at 320 px and fixed narrow pager links and the autodiff input-count control.

Day 6 continuation (2026-10-02): six bounded lesson assignments used GPT-6 Luna, with coordinator integration and screenshot review. Built OptimizerLab, ScheduleLab, InitializationLab, NormalizationLab, DropoutLab and LandscapeLab, each with a pure simulation and scoped tests. Optimizer and schedule comparisons share optimizer-state math; actual autograd gradients are cross-checked against the analytic toy objective. Normalization displays exact reduction/affine values and running-stat inference. Dropout uses seeded Bernoulli masks and probability-weighted expectation, actual decay updates and a computed early-stop history. Landscape demonstrations qualify toy curvature, coordinate dependence and residual-path claims rather than asserting universal neural-network behavior.

Day 6 wave evidence: **456 tests in 46 files pass**, Astro diagnostics are zero errors/warnings/hints, all six Svelte components compile without warnings, and the static build produces **101 pages**. All lessons passed desktop/mobile light/dark walkthroughs and multi-time checks at 1280/390/320 px, keyboard seeking, reduced-motion pause/reset, autoplay/pause, live theme switching and expanded mathematics. Additional layout checks cover both themes at 768/1024/1440 px; numeric-input validation and the early-stop terminal state are checked separately. Walkthrough frame rates were about 75 fps on this machine, not a general performance guarantee. Reports/screenshots: `.playwright-mcp/check/day6-acceptance/` and the per-lesson check folders. Commands: `node scripts/verify-day4.mjs --day 6 --base http://127.0.0.1:4322` (add `--width 320`), and `node scripts/verify-day6-layout.mjs http://127.0.0.1:4322`.

Day 6 original widget chunks were approximately **4.6–8.1 KB gzip**, with dependency closures **24.2–28.2 KB**, below 40 KB. Reproduce with `node scripts/check-widget-budgets.mjs OptimizerLab ScheduleLab InitializationLab NormalizationLab DropoutLab LandscapeLab` after building. No production source maps were emitted. Optimizer setup measured roughly 22–39 ms for the checked batch configurations on this machine; this does not establish LCP/INP. Lessons remain `status: draft`; coordinator acceptance is not release approval. Lighthouse, LCP, INP and a full WCAG audit remain unmeasured.

Day 6 teaching revision (2026-10-02): the owner found the widgets confusing despite the earlier technical acceptance. Reworked all six widgets while retaining their simulations and curriculum order. Optimizers now stretch one recorded update into reading the slope, building the update and moving the weights, with magnified comparison arrows and coordinate calculations. The simple update view bounds its rate at 0.16 so doubling the default rate visibly lengthens the arrow; the race keeps its wider rate range. Schedules put rate above resulting loss with a shared cursor and a worked recorded update; the unstable run is opt-in, and hiding it retains a valid inspected run. Initialization uses forward and backward layer-size bars against a 1× reference, with full statistics behind details. Its slider now preserves the exact He preset instead of rounding √2 to 1.40. Normalization follows one highlighted value through four exact calculations; unrevealed table results use an ellipsis, and advanced controls retain their open state while editing. Dropout shows each unit's contribution to the output; early stopping reports only checkpoints observed so far. Landscapes start with readable loss-height cross sections, offer a left/right starting comparison and optional contour maps, and explain sign counting and residual block calculations. Every figure includes a concrete interaction to try. Updated lesson instructions to match the views and corrected the cosine equation's malformed eta command.

Revision evidence: **462 tests in 47 files pass** (including six new worked-explanation regressions), Astro reports zero errors/warnings/hints, the six Svelte components compile without warnings and have no autofixer issues, and the build produces **101 pages**. The build still emits Astro/Vite head-inject directive warnings. `scripts/verify-day6-teaching.mjs http://127.0.0.1:4322` checks 43 widget instances at three timeline positions in both themes at 320/390/1280 px: 36 lesson layouts pass with no browser errors, overflow or stage-height overflow, plus keyboard seeking and autoplay/pause. Animation measured 76 fps on this machine. `scripts/verify-day6-controls.mjs http://127.0.0.1:4322` checks 32 further layouts at 320/768/1024/1440 px in both themes, including exact He scale, extreme race rates, schedule selection and expanded details. Final control screenshots and reports are under `.playwright-mcp/check/day6-teaching/`; screenshots were reviewed in light/dark mobile and desktop views. Final dependency closures are **25.3–29.7 KB gzip**, below 40 KB. A separate production preview is on port 4322 because the development page timed out during review. These checks establish technical behavior; owner feedback on clarity remains pending. Review Day 6 before continuing to Day 7.

Day 6 clarity revision 2 (2026-10-02, coordinator plus two subagents for dropout and landscapes): the owner asked for Codex's Day 6 work to be fixed. All six lessons were rewritten in the teaching voice of Days 1–3, and every widget got the fixes listed in §5a: log axes where the effect spans decades, local loss contours behind single updates, plain labels, segment-named markers, readouts tied to what is drawn, and duplicate figures removed. Prose claims were re-derived from printed simulation numbers. Evidence: **464 tests in 45 files**, Astro check 0/0/0, 101-page build without source maps, Day 6 widget closures **25.5–30.1 KB gzip**. `verify-day6-teaching` (36 layouts), `verify-day6-controls` (32) and `verify-day6-layout` (48) all pass against the 4322 preview. Desktop and mobile screenshots of every lesson were reviewed by the coordinator. Owner review of clarity is still pending.

Day 7 (2026-10-03, coordinator plus one sonnet agent per lesson, all in parallel): the coordinator first wrote a shared CNN core, `src/lib/cnn/conv.ts` (conv2d, pooling and ReLU with backward passes, gradient-checked in `tests/unit/cnn-core.test.ts`), so every widget reuses one implementation. Widgets: ConvSlider, ConvShapes, ReceptiveField, ResidualTrainer (worker; trains 4-layer plain, 20-layer plain and 20-layer residual tanh nets on two spirals to show degradation) and FeatureViz (worker; trains a tiny two-conv CNN once per page for filters, activation maximisation, top images and saliency with the Adebayo randomisation check). Review fixes: readouts one frame behind the drawing (receptive fields, feature-viz hierarchy), a clipped pooling title, cramped side-by-side grids on mobile (now stacked under 540 px), a learn-step caption contradicting the map, and an honesty sentence that the toy exaggerates He et al.'s degradation gap. The channels agent corrected the outline: padding gives corner pixels 4 windows instead of 1, not as many as the centre's 9. Evidence: **525 tests in 51 files**, Astro check 0/0/0, 101-page build without source maps, Day 7 widget closures **27.2–29.3 KB gzip** (`node scripts/check-widget-budgets.mjs ConvSlider ConvShapes ReceptiveField ResidualTrainer FeatureViz`), check-page `--steps` and `--mobile` clean on all five lessons (about 75 fps, no overflow). Training results are chaotic at the last digits: Chrome and Node differ slightly for ResidualTrainer, so its prose quotes ranges. Owner review of clarity is pending.

- **Widget reuse across lessons:** the TreeSplitter `sim.ts` exports CART (`fitTree`, `predict`, `bestSplit`, `impurity`, `accuracy`, `nodesOf`, `leavesOf`). The shared autograd engine in `src/lib/nn` serves MLP/optimizer lessons (Day 4, 6), so don't edit it. The landing page's `LossLandscape3D` (SGD/Momentum/Adam in WebGL) and `TrainingHero` are candidates for the Day 6 optimizers lesson. `F1Shared` has plot helpers (`setupStage`, `F1Slider`).
- **One widget, several `mode` props, one per step** is the standard shape. It's better than several near-duplicate widgets.
- **Heavy training goes in a Web Worker** (ModelChooser does this). Init must not block the main thread for more than 50 ms. Cost readouts use counted operations, not milliseconds, so the numbers are deterministic.
- **Colours:** semantic tokens only. Categorical tokens `--viz-cat-1..8` were added on 2026-10-03 (owner decision) for identity with no order or sign: word classes, experts, datasets, log entry kinds. Use them in fixed order, never cycled; fold past 8 into "other"; always add a label or shape, because slots 3–5 are under 3:1 contrast in light mode and only slots 1–3 separate every pair on scatter-like plots. k-means and DataSplitter still borrow semantic tokens; switching them is optional.
- **Citations:** empirical claims are stated as findings with the paper, e.g. Grinsztajn et al. 2022 for "trees still win on tabular data". Never invent statistics.
- **Screenshots:** agents should use `scripts/check-page.mjs` or their own headless-Chrome script. The Playwright MCP browser is shared and sometimes returns blank or mis-cropped shots, or is left in reduced-motion emulation (fix with `browser_emulate_media reducedMotion: no-preference`). Playwright here reports DPR 0.75.

## 8. Gotchas

- Day 5's `results-320.json` retains the initial autodiff-overflow failure after the first four lessons passed. The final production `results-autodiff-320.json` and `results-vanishing-exploding-gradients-320.json` both pass and complete that six-lesson check; do not mistake the retained diagnostic for an unresolved failure.
- Astro diagnostics, Svelte compilation and the autofixer are not a complete Svelte TypeScript audit: they missed undefined drawing helpers during Day 6 review. A clean report does not replace runtime checks and inspection of rendered charts. Keep displayed values, drawing and timeline phase on the same frame.
- Vitest uses `maxWorkers: 2` to avoid Windows CPU contention; the normal five-second test timeout is unchanged. Restricted-shell `spawn EPERM` can be an environment failure, not a code failure. Run the authorized check outside that restriction before diagnosing source code.
- The production preview on 4322 gives stable screenshot evidence while development HMR is active. Do not rebuild `dist/` while preview browser checks are running. A sandbox-visible empty TCP listing is not conclusive if browser access succeeds.
- The dev server sometimes serves stale content or CSS. Touch the `.mdx` file. If that fails, restart it (kill whatever listens on 4321 via `Get-NetTCPConnection` → `Stop-Process`, then `npm run dev -- --port 4321`), but only when no agents are running.
- **Outages:** the auto-mode safety classifier and API rate limits have both interrupted agents mid-task. Afterwards, resume each agent and verify its files. Half-finished edits happen, like a literal newline in a header line or a type error left unfixed.
- A typical agent-left type error is Jacobi or other array code: `Array.from(..., () => (i === j ? 1 : 0))` infers `(0|1)[][]`. Annotate it as `number[][]`.
- MDX: bare `<`, `{`, `}` in prose break the build.
- **Main-thread stalls hide from the fps check.** check-page's fps reading is taken at the first step, before later islands hydrate. On 2026-10-04 the KV-cache page froze for 33–37 s per figure and positional-encodings for 12.5 s, with every gate green. Cause on KVCacheLab: model weights in plain `$state` became deep reactive proxies, so the forward-pass loops read through proxies (about 150 times slower). Hold weights, traces and precomputed runs in `$state.raw` or plain variables, share one memoised run per page, and keep every task under ~50 ms (yield or use a worker). `check-page --steps` now records each step's longest stall (`stepMaxGapMs`) and fails above 100 ms.

## 9. Open questions for the owner

- The real domain name (the placeholder is `ailearner.dev`).
- The owner hasn't browser-reviewed Days 1–3 yet.

---

## 10. The method, step by step (follow it exactly)

This is how Day 6 was repaired after it failed review, written down so nobody has to rediscover it. It applies to building a new lesson and to fixing an old one. Do the stages in order. Each one exists because skipping it caused a real failure.

### Stage 1: Read before touching anything

Read `CLAUDE.md`, `LESSONS.md`, this file, every file in `dos&donts/`, the lesson's `.mdx`, its widget `.svelte` and its `sim.ts`, start to finish.

**Why:** the cause is rarely where the symptom shows. The schedules readout that said "update 13" while the list beside it said "update 12/80" was not a typo. One used `index + 1` (the update in progress) and the other `step` (updates completed). Only reading the sim showed that both were "right" and that one shared definition was needed.

### Stage 2: Look at the page as a newcomer, before reading the code's intentions

Run `node scripts/check-page.mjs learn/<track>/<slug>/ --steps` and open the screenshots (read the PNG files). For each figure ask, in this order:

1. What is this figure trying to show me?
2. Can I actually see it, or can I only read it in a caption?
3. Do the canvas, readout, narration and highlighted timeline marker agree?
4. Is any space wasted, cramped or overlapping?
5. Have I already seen this figure on this page?

Write the answers down. That list of observed problems is the work order.

**Why:** Day 6 passed every automated gate. Every real problem was visible in the screenshots to anyone who asked these questions: the empty plane around the optimizer arrow, invisible linear bars, the "Start" marker at update 18, five repeated figures on one page.

### Stage 3: Probe the simulation before writing a single claim

For each sentence that says what a demo shows, print the numbers. Put a throwaway test in `tests/unit/zz-tmp-probe.test.ts`, run it with `npx vitest run tests/unit/zz-tmp-probe.test.ts < /dev/null > "$TEMP/probe.txt" 2>&1`, print from inside it with `process.stdout.write('PROBE …\n')`, read the result with `grep PROBE "$TEMP/probe.txt"`, and **delete the test file in the same command**. Example body:

```ts
import { it } from 'vitest';
import { makeRace, objective, OPTIMIZERS } from '../../src/components/widgets/OptimizerLab/sim';
it('probe', () => {
  const r = makeRace({ landscape: 'ravine', lr: 0.08, batchSize: 4, noise: 1 });
  for (const n of OPTIMIZERS) process.stdout.write(`PROBE ${n} ` + [10, 20, 40, 120]
    .map((k) => `${k}:${objective(r[n].states[k].point, 'ravine').toFixed(3)}`).join(' ') + '\n');
});
```

`< /dev/null` matters: without it, a stray `cat` or prompt can hang the shell.

**Why:** that probe showed SGD at loss 1.046 after 10 updates and Adam at 6.728, the opposite of what the race paragraph implied. The warmup probe showed warmup was slower on the easy bowl. The decay probe showed the three decays ending close together, which led to adding a constant-rate baseline that does show the difference. If the numbers and the sentence disagree, the sentence is wrong. Change the prose, or change the demo so it shows the claim honestly.

### Stage 4: Fix at the layer that causes the problem

For each problem from Stage 2, pick the smallest fix at the right layer:

| Symptom | Fix it in | Day 6 example |
|---|---|---|
| Effect invisible (values span orders of magnitude) | drawing: log axis | init bars; schedules loss chart |
| Motion without context | drawing: draw the landscape | contours behind optimizer arrows |
| Two steps show the same figure | a prop that gives each step its own state | `unstable`, `scaled`, `only={[…]}` |
| A step's claim is false at default settings | defaults in `sim.ts` | normalization gain 1 / shift 0 outside the gain step |
| Two readouts disagree | one shared helper used by both | `updateNumber(step, run)` |
| Marker contradicts state | segment labels, or move the marker | Start/Midway → Early/Middle/Late |
| Demo doesn't show the claim | add the missing comparison to the sim | constant-rate baseline in decay mode |
| Labels overlap at 320 px | shorter marker labels; full name stays in the readout | "4 directions" → "4" |
| Jargon | UI strings | "recorded update" → "update number" |

**Why:** a fix at the wrong layer creates a second bug. Rewording the race paragraph without probing would have kept a false claim. Adding a second widget instead of a prop would have doubled the bundle and the maintenance.

### Stage 5: Rewrite the prose last, around what the figure now shows

Each core step has exactly this shape:

1. **The idea in plain words.** Define every term on first use. Use an everyday analogy if an honest one exists.
2. **What the figure shows,** in one or two sentences.
3. **One thing to do or watch,** with a real number from Stage 3.

Math and code go in the `<details>` blocks. At most one short caveat where it changes how the reader should think. Reference steps (misconceptions, check yourself, going deeper) contain no widgets.

**Why:** prose written before the figure is fixed describes the old figure. Prose written without numbers drifts into hedging, because the writer is unsure what the demo shows.

### Stage 6: Verify, then look again

1. Compile with zero warnings (command in `LESSONS.md` §3) and run the Svelte MCP autofixer.
2. Run the widget's tests. Add one test for any new non-trivial helper (e.g. contour endpoints lie on their level).
3. Run `check-page --steps` and `--mobile`, open the new screenshots, and re-ask the five Stage 2 questions.
4. The coordinator runs the wave checks (§6) and the day's verifier scripts against the 4322 preview. Stop the preview before every rebuild.

**Why:** after the init readout fix, the fresh screenshot showed the result appearing before its bar was drawn. After the first marker shortening, the 320 px verifier still caught an overlap. Every fix gets its own fresh look.

### Worked example: schedules, "A batch-size scaling rule has conditions"

**Before (Codex):**

> Some large-batch SGD experiments increase the learning rate as the batch grows, then warm up the rate to avoid early training instability. In the control below, switch the batch-8 rate from 0.08 to 0.32 to see an actual linearly scaled SGD run beside batch 2. That empirical rule was studied for particular distributed ImageNet training setups. This small quadratic does not establish that scaling will help; it is not a universal law or a guarantee for Adam, other models, or other data.

The figure was identical to the previous step's and defaulted to the unscaled rate. The loss axis was linear, so both runs looked flat after update 10.

**Diagnosis (Stage 2):** a control manual (where to switch, what the control does) plus three hedges, and no number. The figure doesn't show the idea until the reader finds a dropdown, and even then the axis hides the difference.

**Probe (Stage 3):** batch 8 at rate 0.32 is 0.255 above the minimum after one update; batch 2 at 0.08 is 7.93 above.

**Fix (Stage 4):** a `scaled` prop so this step opens on the scaled run; the loss drawn as "loss above its minimum" on a log scale.

**After (Stage 5):**

> If a bigger batch takes fewer, less noisy steps, maybe each step can be longer. The **linear scaling rule** says: multiply the batch by $k$, multiply the learning rate by $k$ too. Goyal et al. (2017) used it, together with warmup, to train an ImageNet model with batches of 8,192 images.
>
> Here batch 8 uses four times the rate, 0.32. After its first update it is already 0.26 above the minimum, while batch 2 at 0.08 is still 7.9 above. It is a rule of thumb that works in some settings, not a law: push the rate past the stability limit and it fails like the 0.7 run above.

This has the idea, its source, one number pair taken from the probe, and one caveat that teaches something (the stability limit) instead of hedging. That is the standard for every step.
