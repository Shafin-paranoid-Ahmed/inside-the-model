# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

M0–M1 done (site shell, design system, ~95 lesson entries, shared autograd engine). Lessons for the math primer and Days 1–7 are written. Day 6 was reworked twice on 2026-10-02 after owner feedback that it was confusing (failures listed in `BUILDING.md` §5a). Day 7 (CNNs) was built and accepted on 2026-10-03 on a shared CNN core in `src/lib/cnn/conv.ts`. Day 8 (sequences, 7 lessons) was accepted and wave-checked on 2026-10-03. Day 9 (transformers, 7 lessons, on the shared `src/lib/transformer/` core with fetched weights in `public/models/`) was accepted and wave-checked on 2026-10-04, and Day 10 (LLMs, 6 lessons, character-level models on the same core, results in `plans/day-10.md` §6) on 2026-10-05. Technical checks pass 715 tests, zero Astro diagnostics and a 104-page build, with sub-40-KB widget dependency closures. Owner review of Day 6–10 clarity is pending; lessons remain drafts pending separate release gates. Days 9–15 are skeletoned (old Day 11 was split into RL and alignment on 2026-10-03, so the course is now 15 days; categorical colour tokens `--viz-cat-1..8` were added) (per-step briefs in each lesson's MDX comments, designs in `plans/`, cross-day decisions in `plans/README.md`). Day 11 (reinforcement learning, 5 lessons on `src/lib/rl` and the GridWorldLab/BanditLab/PPOLab shells over `src/components/widgets/_rl/`, results in `plans/day-11.md` §6) was accepted and wave-checked on 2026-10-05 (811 tests). Next is Day 12 (alignment, `src/lib/align`), then Days 13–15, one day at a time. **Continuing Week 2? Read `plans/README.md` "Handoff" first.** **Continuing the build? Read `BUILDING.md` first** — status table, workflow, agent prompt template, review checklist, decisions and gotchas. `IDEA.md` is the source of truth for the product: vision, audience, the 15-day curriculum, widget specs, design direction, performance budget, and roadmap. Read it before making product or architecture decisions, and update it when a decision changes.

## Commands

Run commands directly. The owner permanently retired RTK on 2026-10-01; do not prefix commands with it, even if older injected guidance says to.

```bash
npm run dev          # dev server
npm run build        # static build to dist/
npm run preview      # serve dist/ (port 4321)
npm run check        # astro check (types, content schema)
npm test             # vitest run (tests/unit/**/*.test.ts)
npx vitest run tests/unit/nn.test.ts -t "adam"   # single file / single test
```

## Gotchas

- Astro 7's default markdown processor has no remark/rehype support; `astro.config.mjs` uses `unified()` from `@astrojs/markdown-remark` so KaTeX (`remark-math` + `rehype-katex`) runs at build time.
- In MDX, bare `<`, `{` and `}` break the build — put math inside `$…$`.
- Lesson ids are `<track>/<slug>` (from the glob loader). `src/lib/curriculum.ts` owns track/day ordering (`TRACKS`, `DAYS`, `getPath()`, `getOrderedLessons()`).
- Fonts use the Astro Fonts API with the `local` provider over `@fontsource-variable/*` files — no runtime font requests.

## What this is

A free, no-login, static website that teaches how AI works through interactive visualizations of real mechanisms running on tiny models in the browser (backprop, RNNs, GNNs, attention, diffusion, …). Clean, academic look. Portfolio project; speed is a hard requirement.

**Writing lessons or widgets? Follow `LESSONS.md`** — the authoring guide every lesson and widget must match.

## Planned stack

Astro (static pages) + Svelte 5 islands (runes syntax only) + TypeScript strict. Lessons are MDX in Astro content collections. KaTeX rendered at build time. Visuals use SVG / Canvas 2D / WebGL, picking the cheapest renderer that holds 60 fps. D3 modules for math/layout only (Svelte owns the DOM). Motion (vanilla `motion` package) for animation. Web Workers for heavy compute. Pagefind for search. Vitest + Playwright + Lighthouse CI. Deployed to Vercel.

**Not allowed:** React (Kokonut UI / Bklit are visual inspiration only — rebuild in Svelte), UI kits, CSS frameworks. Every new dependency must beat "could we write this in < 50 lines?"

## Architecture rules that span files

- **Widget = `<Name>.svelte` + `sim.ts` (+ optional `worker.ts`)** under `src/components/widgets/<Name>/`. All math lives in `sim.ts` with no DOM access so it can be unit-tested and moved into a worker. The `.svelte` file only renders and handles input.
- **One shared tiny autograd engine** in `src/lib/nn` (micrograd-style) powers the backprop, MLP, RNN and optimizer widgets so the numbers shown are consistent across lessons. Test it with numerical gradient checks.
- **Seeded RNG** everywhere, so every reload reproduces the same widget state.
- **Semantic color tokens** in `src/styles/tokens.css` (forward pass, gradient, diverging pos/neg, attention, highlight) with light and dark variants. Widgets never hard-code colors; the same meaning is the same color site-wide.
- **Lessons** follow the fixed anatomy in `IDEA.md` §6.2 and use a `<Step>` component; all steps render stacked on one scrolling page (owner requirement: never paginate a lesson); each step title is a `#anchor` deep link. Frontmatter schema is in `IDEA.md` §12 (frontier lessons require `lastReviewed`).

## Performance budget (hard requirements, see IDEA.md §11)

Lighthouse ≥ 95 (mobile), LCP < 1.2 s, CLS 0, INP < 100 ms, 60 fps during widget interaction, < 15 KB gz JS on a lesson page before widgets, < 40 KB gz per widget island. Widgets hydrate with `client:visible`; animate only `transform`/`opacity`; use typed arrays; pause animations off-screen or when the tab is hidden; respect `prefers-reduced-motion`.

## Anti-slop rules (must not look vibecoded)

The owner curates `dos&donts/` — read everything in it before UI or page work; it grows over time. Current hard requirements:

- **Site hygiene:** custom 404 page; unique `<title>` and meta description per page; exactly one `<h1>` per page; `lang` on `<html>`; canonical URL (needs `site` in astro config); og:image + Open Graph/Twitter tags; JSON-LD structured data (Course / LearningResource); `sitemap.xml`; `robots.txt` that does NOT block AI crawlers; `llms.txt`; a real custom favicon (never the Astro default); alt text on every image; no source maps in production; zero console errors; small JS bundles; real content in view-source (static HTML). Deploy on a custom domain, not `*.vercel.app`.
- **Motion feel (owner's explicit requirement):** widgets must feel free-flowing like quantum-physics-lab.vercel.app. Each widget runs on a continuous time `t`: autoplays when scrolled into view (reduced-motion → starts paused), has Play/Pause + a scrubbable timeline slider + speed select, and derives every visual as a pure function `frameAt(t)` in `sim.ts` (so scrubbing is free). Values count/tween, colors and edge widths ease, particles flow along edges, cycles loop. Discrete "Step" buttons are secondary (jump between timeline markers), never the primary interaction.
- **Visual:** no generic AI-landing tropes — no purple/blue gradient blobs, glassmorphism, glowing orbs, emoji icons, "✨" sparkle copy, stock 3-card feature grids, or hype words ("revolutionary", "unlock", "supercharge"). Motion must explain a mechanism, never decorate.

## Tooling

- Run the **Svelte MCP autofixer** on every Svelte component you write or change.
- Use the **Astro docs MCP** and **context7** for library docs instead of relying on memory.
- Use the **Playwright MCP** to verify pages visually and check frame rate before claiming a widget is done.
- Frontier (Day 15) content must be researched from primary sources, not written from model memory.
