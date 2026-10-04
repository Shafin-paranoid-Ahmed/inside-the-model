# Repository Guidelines

## Project Structure & Module Organization

This static AI-learning site uses Astro, Svelte 5 islands, strict TypeScript, and MDX. Routes live in `src/pages/`, layouts in `src/layouts/`, and lessons in `src/content/lessons/<track>/<slug>.mdx`. Curriculum ordering belongs in `src/lib/curriculum.ts`.

Widgets live in `src/components/widgets/<Name>/`: `<Name>.svelte` renders the interface, `sim.ts` contains DOM-free mathematics, and optional `worker.ts` handles heavy computation. Reuse `_shared/` controls and `src/lib/nn/` autograd utilities. Styles live in `src/styles/`, assets in `public/`, tests in `tests/unit/`, and verification scripts in `scripts/`.

## Build, Test, and Development Commands

Use Node.js **22.12+**. Run commands directly; the owner permanently retired RTK on 2026-10-01, so never prefix commands with `rtk`.

- `npm ci`: install locked dependencies.
- `npm run dev`: start local development; obtain owner permission before launching a server.
- `npm run check`: check Astro, TypeScript, and content diagnostics.
- `npm test`: run Vitest once; `npm run test:watch` watches changes.
- `npm run build`: generate `dist/`; reserve shared-checkout builds for the coordinator.
- `npm run preview`: serve the production build, with owner permission.

## Coding Style & Naming Conventions

Match existing two-space indentation, single-quoted TypeScript strings, and semicolons. Use Svelte 5 runes, PascalCase component/widget names, and kebab-case lesson slugs and test filenames. No formatter or lint script is currently configured.

Use semantic tokens from `src/styles/tokens.css`, seeded randomness, pure `frameAt(t)` simulations, and `client:visible` hydration. Include playback, scrubbing, speed controls, and reduced-motion support. Avoid React, UI kits, and CSS frameworks.

## Testing Guidelines

Write Vitest tests as `tests/unit/<name>.test.ts`. Cover mathematical invariants, deterministic results, boundaries, and numerical gradients where relevant. No numeric coverage threshold is configured.

Run relevant tests and type checks. For UI changes, use `node scripts/check-page.mjs learn/<track>/<slug>/ --steps --mobile` against an authorized running server. Inspect desktop/mobile, light/dark, and intermediate animation states; prose numbers must match visuals.

## Commit & Pull Request Guidelines

The repository has no commits yet, so no historical message convention exists. Prefer concise imperative subjects. Agents must not commit or push; the owner handles Git publication. PR descriptions should explain behavior, scope, validation, linked issues when applicable, and screenshots for visual changes.

## Lesson Authoring

Read `CLAUDE.md`, `IDEA.md`, `LESSONS.md`, and `BUILDING.md`; inspect `dos&donts/` before UI work. Preserve curriculum order and one scrolling page per lesson. Teach intuition, mechanism, and visible consequences before advanced statistics.

**Week 2 (Days 9–15) is skeletoned.** Start at `plans/README.md`: it holds the ordered task list and the cross-day decisions. Each stub lesson's `{/* … */}` comments are your brief, and `plans/day-NN.md` is the day's design. Replace the comments with real prose and widgets, and delete every brief comment when the lesson is written.

## Rules from past failures (read before writing any lesson)

Day 6 was written by Codex and passed every technical check (462 tests, zero diagnostics, budget checks, 558 screenshots). The owner still found it confusing, and it had to be rewritten. `BUILDING.md` §5a lists what went wrong. In short:

- **Clarity beats passing checks.** Write each step as a teacher: plain-language intuition, define every term, then ONE thing to look at or try, quoting a number the widget shows. A paragraph of control names is a manual, not a lesson. No hedging paragraphs, no "this is a toy" disclaimers, and no notes about how the widget computes ("recorded", "seeded", "the unit tests check").
- **Probe before you write.** Print the simulation's numbers first and write the claim from them. If the sim contradicts the brief or the textbook, change the prose (or the demo), never the numbers. Never claim a head role, a phase change or an effect that the probe does not show.
- **No repeated figures.** Misconceptions, check-yourself and going-deeper steps have no widget. No two core steps use the same widget with the same props, and near-identical states count as repeats.
- **Make the effect visible.** Use log axes for effects that span orders of magnitude, tight domains for small effects, and context behind single updates. Defaults must match the claim at that step.
- **Readouts must match the drawing in the same frame.** Timeline markers name segments, and narration never runs ahead of or behind the canvas.
- **Leave nothing behind.** No `tests/unit/zz-*` probe tests, no `verify.mjs` in widget folders, no scratch files in the repo.
- **One lesson per run, then stop.** Report the files changed, the test count, the probe numbers you quoted, and the screenshot paths. A coordinator reviews the result before the lesson counts as accepted.
