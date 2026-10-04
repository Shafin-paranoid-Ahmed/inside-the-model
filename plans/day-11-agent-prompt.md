# Day 11 lesson-agent prompt (adapted from `lesson-agent-prompt.md`, 2026-10-05)

Fill `__SLUG__`, `__MODES__`, `__TEST__`, `__EXTRA__` per lesson. All five can run in parallel (each owns only its mode files). Each mode = `m-<mode>.ts` + `m-<mode>.meta.ts` in the named widget folder.

| Lesson | `__MODES__` | `__TEST__` | `__EXTRA__` |
|---|---|---|---|
| rl-basics | GridWorldLab: loop (reference mode, refine it), mdp, return, discount, value, optimal-policy; BanditLab: explore (reference mode, refine it; curves may need a moving average) | rl-basics | loop seed 34 per §6 |
| q-learning | GridWorldLab: qvalues, bellman, qlearn, qlearn-curve, qlearn-terminal | q-learning | §6: alpha 0.1 / eps 0.1 is far from converged at 300 episodes; pick honest settings and teach near-tie flipping |
| policy-gradients | BanditLab: softmax-pg, baseline; GridWorldLab: reinforce, reinforce-update, actor-critic | policy-gradients | §6: baseline cuts variance 2-6x but learning curves barely differ; actor-critic only slightly ahead. Say so |
| deep-rl | GridWorldLab: dqn-network, dqn-update, dqn-instability, dqn-replay, dqn-target, dqn-compare | deep-rl | §6: divergence needs Adam lr 0.1; textbook variant ordering only at lr 0.001; the comparison takes 3.7-9 s, so it must run via `kit.job` (worker) behind the loading text |
| ppo | GridWorldLab: ppo-advantage, ppo-ratio, ppo-epochs, ppo-train; PPOLab: clip-positive (reference mode, refine it), clip-negative, drift | ppo | drift uses `driftRuns({ warmIterations: 10 })` (uniform-policy batch has all-negative advantages); clipping does not bound ratios (eps 0.1 still reaches 1.32) |

---

You are building ONE lesson of an interactive AI-learning website (Astro 7 + Svelte 5 runes + TS strict), in `G:\PROJECTS\RECOVERY\Ai learner website`. Run shell commands directly (never prefix with `rtk`). Do not use Explore agents; read files yourself.

LESSON: `src/content/lessons/alignment/__SLUG__.mdx` (currently a stub; keep its frontmatter, set `status: draft`). Your brief is the `{/* … */}` comments in your lesson file and `plans/day-11.md` (§1 core, §2 widgets, and §6 "Core results (measured)", which overrides the MDX briefs: if a brief asks for an effect §6 says is absent, teach what is present). `plans/README.md` wins where they disagree. Delete the comments when the lesson is written.
YOUR MODE FILES (you own only these): __MODES__
TEST: `tests/unit/__TEST__.test.ts` (frameAt endpoints, determinism, and the numbers your prose quotes).
__EXTRA__

SHARED CORE (do NOT edit `src/lib/rl/`, nor `src/components/widgets/_rl/` (Shell.svelte, kit.ts, svg.ts, worker.ts), nor the shells `GridWorldLab.svelte`, `BanditLab.svelte`, `PPOLab.svelte`, their `sim.ts` or `draw.ts`, or any mode file you do not own; other lesson agents are working in the same folders in parallel). `src/lib/rl/` has the 4x3 gridworld, bandit, tabular algorithms (policy evaluation, value iteration, Q-learning, REINFORCE, actor-critic), DQN and PPO, tested in `tests/unit/rl-core.test.ts`. Each shell loads one lazily imported module per `mode` (found by `import.meta.glob`), so you add a mode by creating two files: `m-<mode>.meta.ts` (title, legend, aspect, controls; bundled with the shell so the card reserves its size) and `m-<mode>.ts` (setup, timeline, frameAt, draw, label). Heavy runs go through `kit.memo` or `kit.job(...)` (worker). The full contract is the comment block at the top of `GridWorldLab/sim.ts`; the reference modes `GridWorldLab/m-loop.ts`, `BanditLab/m-explore.ts`, `PPOLab/m-clip-positive.ts` (+ their meta files) show the idiom. If you need a change to a shell, a shared helper or src/lib/rl, STOP and describe it in a message/report to me (the coordinator) instead of editing; I route shared changes. Never let a mode file grow its own copy of a shared helper; small mode-local helpers are fine.

BEFORE WRITING, READ FULLY: `CLAUDE.md`, `LESSONS.md` (mandatory), `BUILDING.md` §2, §5, §5a and §10, everything in `dos&donts/`, `plans/day-11.md`, the shell contract and reference modes above, the `src/lib/rl` files you use, and these finished references: `classic-ml/decision-trees` + `TreeSplitter` (prose voice) and `transformers/self-attention` + `AttentionExplorer` (lazy-mode widget idiom). Match their structure, prose style and code idiom. Token colours only (`--viz-forward` policy probabilities, `--viz-gradient` updates/advantages, `--viz-pos/--viz-neg` signed rewards/Q/values, `--viz-seq-*` heatmaps, `--accent` highlight, `--viz-cat-1..8` for identities like variants, always labelled). Seeded RNG. All math in mode/sim code with no DOM, visuals from a pure `frameAt(t, …)`; autoplay on view, Play/Pause + scrubbable timeline + speed; reduced motion starts paused. Keep each island (shell + your mode chunk) under 40 KB gzip.

Lesson = ONE scrolling page of several <Step>s, each core step with a widget instance (different `mode` props). Numbers in prose MUST match what the widget shows (compute them from the sim, never guess, never copy from the brief or §6 without re-probing in your exact props). In MDX, bare `<`, `{`, `}` break the build; math goes in `$…$`. Link related lessons with `/learn/alignment/<slug>/` and earlier days where relevant. Published numbers (textbook utilities, paper results such as DQN Atari scores or PPO hyperparameters) must be checked against the primary source by web search before quoting, with links.

CLARITY RULES (BUILDING.md §5a; these are where past agents failed):
- Each core step: plain-language intuition first, define every term, then ONE concrete thing to look at or try, quoting a real number the widget shows. Write like a teacher, not a widget manual. At most one short caveat where it really matters; no "this is a toy, not a claim…" disclaimers, no meta-notes about how the widget computes ("not interpolated", "recorded", "seeded").
- Before writing a claim about what a demo shows, print the simulation's numbers and check (BUILDING.md §10 Stage 3 probe recipe; delete the probe file in the same command). If the sim shows something else, change the prose (or the demo), never the other way round.
- No repeated figures: misconceptions / check-yourself / going-deeper steps get NO widget, and no two core steps may show the same widget with identical props.
- Widget labels in plain words. Defaults must match the lesson's claim at that step.
- If the effect spans orders of magnitude, use a log axis; if it is small, tighten the domain. Never a tiny shape in a huge empty plane.
- Narration and readout sentences use the body font with tabular figures (mono only for numbers and code); range sliders use `accent-color: var(--accent)`; canvas/SVG text stays ≥ 11 px CSS at 390 px; numbers shown during an animation are always real states (switch at segment boundaries; only geometry eases or cross-fades).
- PERFORMANCE: hold runs and traces in `$state.raw` or plain variables, never plain `$state`. No main-thread task over ~50 ms: use the shell's memoised/worker runs as the contract says; `check-page --steps` reports each step's longest stall and fails above 100 ms.
- Timeline markers name segments, never points; the readout, narration, drawing and highlighted marker must agree on every frame.
- Temporary probe scripts go in the scratchpad or are deleted immediately: a `tests/unit/zz-*.test.ts` left behind runs in every `npm test`.

VERIFY before reporting done:
1. `npx vitest run tests/unit/__TEST__.test.ts tests/unit/rl-core.test.ts` passes; `npm run check` has 0 errors in your files.
2. Dev server is already on port 4321 (shared: do NOT kill or restart it, do NOT run `npm run build`). `node scripts/check-page.mjs learn/alignment/__SLUG__/ --steps` and `--steps --mobile` → errors: [], ~60+ fps, overflowX: false, no stall failure. Do NOT use the Playwright MCP browser (shared).
3. Screenshot every figure at SEVERAL timeline points (not just t=0), desktop 1280 and mobile 390, light and dark, and LOOK critically (Read the PNGs): readouts agree with the drawing, the active timeline marker matches the state shown, no overlapping/clipped labels, readable plot sizes, narration matches visuals, the demo visibly shows the claimed effect. Fix and re-check.
Only touch your own lesson, your own mode files and your test file. Do not commit. Report: files changed, test count, every number quoted in the prose with where it comes from, any shared change you need, concerns, final screenshot paths.
