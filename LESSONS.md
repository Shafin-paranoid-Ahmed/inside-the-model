# Lesson & Widget Authoring Guide

Every lesson and widget on the site follows this guide. It exists so 95 lessons read and feel like one book. Read `CLAUDE.md` (rules, anti-slop, motion feel, perf budget) and `IDEA.md` §6–§8 first. The reference implementation is `src/content/lessons/neural-networks/backprop-mlp.mdx` + `src/components/widgets/BackpropViz/`.

## 1. Lesson file

Path `src/content/lessons/<track>/<slug>.mdx`. Keep the existing frontmatter (title, day, track, order, prereqs, duration, level); refine `summary` if needed; set `status: draft` when the lesson is written. Never rename files or ids — other lessons link to them.

```mdx
---
(frontmatter)
status: draft
---
import Step from '../../../components/ui/Step.astro';
import MyWidget from '../../../components/widgets/MyWidget/MyWidget.svelte';

<Step title="Hook as a question or surprise">
Short text. Then the widget, configured for this step.

<MyWidget client:visible title="What this figure shows" loopPhase="…" />
</Step>

<Step title="…"> … </Step>          ← 5–12 core steps, ONE new idea each

<Step title="Common misconceptions"> ≥3 "You might think X. Actually Y." </Step>
<Step title="Check yourself"> 3–5 questions as <details><summary>Q</summary>A</details> </Step>
<Step title="Going deeper"> primary papers + 2–4 best resources, as links </Step>
```

- **Show the math**: `<details><summary>Show the math</summary> … KaTeX … </details>` next to the step it belongs to.
- **Show the code**: `<details><summary>Show the code</summary>` + a short NumPy/PyTorch fenced block that matches the widget.
- Math only inside `$…$` / `$$…$$`. Bare `<`, `{`, `}` break MDX — write `\lt`, or put it in math/code.
- **Every lesson is interactive — no exceptions (owner requirement).** Each lesson has at least one widget the learner can play with, and every step that explains a mechanism shows it in a widget state. Even "conceptual" lessons (e.g. classic ML vs deep learning, train/val/test, evaluations) get a hands-on figure: a simulator, a comparison you can toggle, a dataset you can split, a decision you can make and see the consequence of.
- Every step with a mechanism gets a widget state. Text says "watch", "drag the timeline", "click a node" — never "press Step".

### Writing
- Intuition → picture → math → code. Plain words first; define every term the first time.
- Clear, precise, warm. No hype, no emoji, no "revolutionary/unlock/supercharge/delve".
- Technically rigorous: check every equation, sign, dimension and number. Numbers in the text must match what the widget shows.
- Label simplifications: "In real models …". One short caveat where it matters; never a paragraph of disclaimers ("this is a toy, not a claim…") or notes about how the widget computes ("recorded", "not interpolated").
- Teach, don't narrate the UI: each step explains the idea first, then names ONE thing to look at or try, with a real number from the widget. A step that is mostly control names is a manual, not a lesson.
- Check a demo's claim against the simulation's printed numbers before writing it. If they disagree, the prose is wrong.
- Misconceptions, Check yourself and Going deeper steps contain no widget. No two core steps show the same widget with the same props. (Full list of past failures: `BUILDING.md` §5a.)
- Link prerequisites and related lessons with `/learn/<track>/<slug>/`.
- Frontier content (Day 15 and anything about 2025–2026 models) must come from primary sources found with web search, cited with links and dates. Never from memory.

## 2. Widgets

Folder `src/components/widgets/<Name>/`: `<Name>.svelte` (render + input only), `sim.ts` (pure logic, no DOM), optional `worker.ts`. Tests in `tests/unit/<name>.test.ts`.

**Reuse first.** Before building a widget, check `src/components/widgets/` — one good widget with a few props beats three similar ones. Shared building blocks live in `src/components/widgets/_shared/` (figure card, timeline controls, animation loop, theme colors) — use them; don't re-implement them.

### The motion contract (owner requirement — non-negotiable)
- Driven by a continuous time `t`. `sim.ts` exports a pure `frameAt(t, …)` (or equivalent) returning everything needed to draw; precompute numbers once, the timeline only interpolates.
- **Autoplays** when scrolled into view; pauses off-screen and when the tab is hidden.
- Play/Pause, a **scrubbable timeline** with labeled phase markers, speed (Slow/Normal/Fast). Discrete "step" controls are secondary.
- Values count/tween, colors and widths ease, particles flow along edges; cycles loop smoothly.
- `prefers-reduced-motion`: start paused, no particles; scrubbing and all numbers still work.
- Props let each lesson step configure the widget (e.g. `loopPhase`, `mode`, `title`, `select`).

### Look
- Colors only from CSS tokens (`src/styles/tokens.css`), re-read on theme change. Semantic meaning is fixed site-wide: `--viz-forward` activations/forward, `--viz-gradient` gradients/backward, `--viz-pos`/`--viz-neg` signed values, `--viz-seq-1..5` intensity (attention, weights), `--accent` highlight, `--viz-cat-1..8` categories (identity only: fixed order, never cycled, always with a label or shape; see `tokens.css`).
- Titled figure card with a compact legend; numbers in the mono font with tabular figures.
- Cheapest renderer that holds 60 fps: SVG for small diagrams, Canvas 2D for dense/fast animation, WebGL2 (raw, no three.js) only for 3D.
- Precise and restrained, like a Distill.pub figure. No glow, gradients-for-decoration, blobs or emoji.

### Engineering
- Svelte 5 runes only. Hydrate with `client:visible`. Fixed aspect-ratio box (CLS 0). DPR-aware canvas (cap 2).
- One `requestAnimationFrame` loop; don't write Svelte `$state` every frame for canvas-only data (update readouts ≤ 10×/s).
- Heavy compute (training loops) in a Web Worker. Typed arrays. Seeded RNG (`createRng` from `src/lib/math/rng.ts`).
- Use the shared engine in `src/lib/nn` when you need autograd/MLP/optimizers — never edit it; put helpers in your widget folder.
- Accessible: real buttons with labels, keyboard operable, a `role="img"` + `aria-label` on the figure, an `aria-live="polite"` region that narrates phase changes (not every frame).
- Budget: < 40 KB gz per island (Svelte runtime is shared).
- **One source of truth for numbers:** readouts, bar labels, narration and phase names must all come from the same frame values. A throttled readout must never disagree with what the canvas draws, and the narration must describe what is visible for the whole phase. Timeline labels are sentence case, because the Timeline only uppercases the first letter.
- Animate reveal opacity or geometry, not fictional intermediate derivatives. If weights or coordinates interpolate, recompute the displayed loss/statistics at those displayed values. Distinguish cached pre-update values, final results being revealed, and genuinely computed intermediate states in captions.
- **Fit the view to the content:** axes and plot domains fit what the scene draws (no tiny arrows in a huge empty plane, no clipped bars or shapes), and labels never overlap at desktop or mobile widths. When values span orders of magnitude, use a log axis so the effect is visible. Labels use plain words, and legends name exactly what is drawn.
- Unit-test `sim.ts`: correctness of the math (compare to closed forms or numerical gradients), `frameAt` endpoints, determinism.
- Compile with zero Svelte warnings.

## 3. Definition of done (per lesson)

Several agents work in parallel, so:

1. `npx astro check` (0 errors in your files) and `npx vitest run tests/unit/<your tests>` pass. **Do not run `npm run build`** (parallel builds collide in `dist/`); the coordinator builds after each wave.
2. Every `.svelte` file compiles with zero warnings:
   `node -e "const r=require('svelte/compiler').compile(require('fs').readFileSync(process.argv[1],'utf8'),{filename:'x.svelte'});console.log(r.warnings.map(w=>w.code+': '+w.message).join('\n')||'0 warnings')" <file>`
3. Check the page in an isolated headless Chrome — **do not use the Playwright MCP browser** (it is shared between agents). The dev server runs at http://localhost:4321 (never kill or restart it):
   `node scripts/check-page.mjs learn/<track>/<slug>/ --steps` (no leading slash). It walks every step in light + dark with motion on, reports console errors, fps and horizontal overflow, and saves figure screenshots to `.playwright-mcp/check/<slug>/`. Require: `errors: []`, fps ≈ display rate, `overflowX: false`. **Look at** a few of the screenshots (Read the PNG) and fix what looks wrong. If a page looks stale (old content), touch the .mdx file and retry; if still stale, say so in your report.
4. Frontmatter `status: draft`.
5. Do NOT git commit. Don't touch files outside your assignment. If you need a change to a shared file, describe it in your report instead.
