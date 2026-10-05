# Week 2 plans (Days 9–15)

Each `day-NN.md` is that day's design (`day-11.md` covers both Day 11, RL, and Day 12, alignment): shared core, widgets and their modes per step, reuse, build order, risks. The per-step brief for every lesson lives in its `.mdx` as MDX comments (`{/* … */}`, not rendered). Lessons stay `status: stub` until a lesson agent writes them. Skeletons written 2026-10-03, one agent per day; coordinator reviewed.

Lesson agents: read your lesson's comments + your day plan + this file. Where a day plan and this file disagree, **this file wins**.

## Cross-day decision 1: one tiny-transformer core, one model pipeline

Days 9, 10, 14 and 15 each planned a small transformer. They are one core, built at the start of Day 9 by the coordinator:

- **Core:** `src/lib/transformer/` (Day 9 plan §1 is the base). Parameters in one flat `Float64Array` with a named offset table (Day 10 need). `Config` per model: vocab (word, char or BPE ids), `d`, heads, layers, context, positions `'learned' | 'rope'` with a pluggable RoPE scaling function (Day 15 long-context), causal on/off.
- **API, one module:** `forward(w, ids, opts) → Trace` (residual stream, q/k/v, scores, weights per head; `opts.ablate`, `opts.patch` hooks for Day 9 multi-head and Day 15 mech-interp); `prefill` / `decodeStep` with a per-layer KV cache, tested equal to `forward` (Day 10); `lossAndGrad`, gradient-checked (Days 9, 10, 14); `generate(w, prompt, { sampler, logitMask })` (Day 10 sampling, Day 14 constrained decoding).
- **Determinism:** training uses the arithmetic-only `exp`/`tanh` from `Seq2SeqAttention/sim.ts` so Node and Chrome agree bit for bit; inference may use `Math.exp` (Day 9 plan) only where the prose rounds to 3 decimals.
- **One training script:** `scripts/train-models.ts` (bundled with esbuild, run with Node), seeded, one entry per model: `tiny-lm` (Day 9 grammar), `tiny-lm-nomask` (Day 9 causal masking), the Day 10 models (Alice char LM, ICL cipher, facts), the Day 14 `prompt-model`, the Day 15 task models (key-value retrieval, repeated sequences). Each entry prints the report its lessons quote (head report, metrics); lesson agents claim nothing the report does not show.
- **One weight format:** `public/models/<name>.bin` (float16 by default; int8 with per-row scales when a model would exceed 40 KB) + `public/models/<name>.json` manifest (config, vocab, offsets, dtype, metrics, loss curve). One loader `load.ts`, memoised per name, fetched when the widget becomes visible. Day 9's "float16 JSON" and Day 10's "int8 .bin" both collapse into this.
- **Day 10's `src/lib/lm/`** keeps only LM-specific helpers (charset, samplers, cost counting) and imports the core; no second transformer. **Day 14's `tinylm.ts` fallback is cancelled.**
- Models that need a worker for training stay offline; widgets do inference only, except where a lesson's point is watching training (Day 13 generative models, Day 7/8 precedent), which train per page in a worker.

## Cross-day decision 2: other shared pieces

- **Tokenizer:** Day 9's `TokenizerLab/sim.ts` BPE is the one tokenizer; Day 14 counts tokens with it.
- **RL:** Day 11's `src/lib/rl` (Q-learning, REINFORCE) is reused by Day 15 SafetyLab rather than re-implemented.
- **Determinism, motion contract, tokens, budgets:** unchanged from `CLAUDE.md` / `LESSONS.md`.

## Handoff: where the build stands and how to continue (updated 2026-10-03, 20:20)

**Done:** Task 0, the Day 11 rebalance to 5 RL lessons (Codex, coordinator-verified). Task 1, the Day 9 shared core `src/lib/transformer/` with models in `public/models/` (Codex, 24 tests, coordinator-verified; probe results are in `plans/day-09.md`). `transformers/tokenization` (Codex, fixed after review, **accepted**).

**Day 9 accepted and wave-checked (2026-10-04).** **Day 10 accepted and wave-checked (2026-10-05).** Then the Day 10 wave check. Lesson-agent prompt for Day 10 onward: `plans/lesson-agent-prompt.md` (BUILDING.md §4 plus the Day 10 rules (body-font readouts, accent sliders, canvas text >= 11 px at 390 px, real-state numbers only, `$state.raw` for weights, no main-thread task over ~50 ms)). **Day 11 accepted and wave-checked (2026-10-05)** (BUILDING.md §6 Day 11 note). GridWorldLab's worst mode is 39.7 KB gzip, so any future GridWorldLab mode must stay lean. **Next: Day 12** (alignment): build `src/lib/align` and its tests first (plans/day-11.md §1, which reuses `clippedSurrogate` from `src/lib/rl/ppo.ts`), then the lessons per its build order (PreferenceLab has one owner at a time). For a parallel-agent prompt, adapt `plans/day-11-agent-prompt.md`. **Remaining, strictly one day at a time (owner rule):** Days 12, 13, 14, 15. Day 11 idea: the core agent builds `src/lib/rl` plus GridWorldLab/BanditLab/PPOLab shells with one lazily loaded module per mode (as AttentionExplorer does), so the five lesson agents can each own only their own mode files and run in parallel.

**How a Claude coordinator runs each day** (BUILDING.md §3–§6 in full; the short version):
1. Read `CLAUDE.md`, `BUILDING.md` (§2, §3, §4, §5, §5a), `LESSONS.md`, this file, and the day's `plans/day-NN.md`. Make sure the dev server is up on port 4321 and start it if it is down; agents share it and never kill it.
2. **Shared core first, by the coordinator or one agent, before any lesson agent starts**: Day 10 `src/lib/lm/` and the Day 10 models added to `scripts/train-models.ts`; Day 11 `src/lib/rl`; Day 12 `src/lib/align`; Day 13 `src/lib/gen/` (plus deleting the empty `GenCommon` folder); Day 14 `src/lib/applied/` and `prompt-model`; Day 15 `src/lib/frontier/seqcost.ts`. Gate each on its tests and the plan's acceptance bars.
3. **Then one subagent per lesson, in parallel**, using the BUILDING.md §4 prompt template plus: "Your brief is the `{/* … */}` comments in your lesson file and `plans/day-NN.md`; delete the comments when the lesson is written." Respect each plan's build order: a lesson that extends another lesson's widget waits for that lesson (Day 9: multi-head after self-attention; Day 13: vaes after autoencoders, flow-matching after diffusion). Never let two agents edit one widget at the same time.
4. **Review every lesson yourself** against BUILDING.md §5 and §5a: probe the prose numbers against the sim, look at screenshots at several timeline points, desktop and mobile, light and dark. Send fixes back to the same agent with SendMessage. Technical gates passing is not acceptance.
5. **Wave checks** (BUILDING.md §6), then update the BUILDING.md status table and the CLAUDE.md status line, then start the next day.
- Day 13: review `gnn.mdx` with the checklist in `plans/day-13.md` §0. Day 15: every factual claim comes from primary sources found by web search, with links and dates; set `lastReviewed`.
- Expect usage limits: a day is 5–8 lessons. If the limit is near, finish and review the lessons in flight, update the status table, and stop cleanly instead of starting new agents.

## Owner decisions (2026-10-03)

1. **Model weights outside the JS budget: approved.** Weight files in `public/models/` are fetched lazily as data when a widget becomes visible, cached across pages, and do not count toward the 40 KB gz per-island JS budget. Keep each file as small as the lesson allows (float16, int8 past 40 KB).
2. **Categorical colours: added.** `--viz-cat-1..8` in `src/styles/tokens.css` (rules in `LESSONS.md` "Look"). Plans that borrowed pos/neg or used shapes for categories should use them.
3. **Day 11 split: done.** Day 11 is Reinforcement learning (rl-basics, q-learning, policy-gradients, deep-rl, ppo; 165 min after Task 0); Day 12 is Alignment (sft, rlhf, dpo, constitutional-ai, rlvr, evaluations). Later days moved up one: graphs and generative models are Day 13, applied AI Day 14, the frontier Day 15. The course is 15 days.
4. **Smaller calls, by default unless the owner objects.** The tokenization-quirks step uses Bengali. The Day 13 GAN collapse may use an 8-Gaussian ring, labelled. The Day 9 self-attention step shows causal triangles with a one-line forward reference. Corpora are public domain (Alice) and invented (Day 14 handbook), committed under `scripts/data/`. Day 15 `whats-new` is re-reviewed every 6 months by the coordinator.
