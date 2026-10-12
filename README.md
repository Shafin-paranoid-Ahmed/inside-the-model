# Inside the Model

A free website that teaches how AI works by letting you watch the real mechanisms run. Every lesson is built around small interactive figures: real models, small enough to train in your browser, showing backpropagation, attention, diffusion and the rest step by step. No login, no tracking, no video.

**Live site:** [inside-model.vercel.app](https://inside-model.vercel.app)

## What you will find

A 15-day course, plus a math primer, written for curious beginners who are comfortable with a little algebra.

| Week | Days |
|---|---|
| Week 1 | 1 What is learning? · 2 Classification & generalization · 3 Classic machine learning · 4 Neural networks · 5 Backpropagation · 6 Training deep networks · 7 Convolutional neural networks |
| Week 2 | 8 Sequences & recurrent networks · 9 Transformers · 10 Large language models · 11 Reinforcement learning · 12 Alignment · 13 Graphs & generative models · 14 Applied AI · 15 The frontier |

Each lesson is one scrolling page. Every step works through a single running example, so each number on the page can be reproduced by hand. A figure plays on its own when you scroll to it. You can pause it, scrub its timeline, or change its inputs.

Days 1–13 and the math primer are written. Days 14–15 are in progress. All lessons are drafts under review.

## How it is built

- **[Astro](https://astro.build)** builds static pages, so every lesson's text is real HTML.
- **[Svelte 5](https://svelte.dev)** runs the interactive figures. Each one loads only when it scrolls into view.
- **TypeScript** (strict) throughout. Lessons are written in MDX.
- **KaTeX** renders math when the site is built, not in the browser.
- **SVG and Canvas 2D** draw the figures. There is no chart or UI library.
- **Vitest** tests the math behind the figures.

The design rules:

- **The math lives apart from the drawing.** Each figure is a folder in `src/components/widgets/<Name>/`. Its `sim.ts` holds all the math, with no page access, so it can be tested directly. Its `.svelte` file only draws and handles input.
- **Each figure is a function of time.** A figure's state is computed from its timeline position by a function called `frameAt(t)`, so scrubbing and replaying cost nothing.
- **Results are repeatable.** Every random choice uses a fixed seed, so a reload shows exactly the same figure.
- **The neural-network lessons share one engine.** `src/lib/nn` is a tiny automatic-differentiation engine (the code that computes gradients), checked against numerical gradients. Later topics have shared cores of their own in `src/lib/` (CNNs, transformers, RL, alignment, generative models).
- **Speed is a requirement.** Each figure ships under 40 KB of gzipped JavaScript. Pages without figures need almost no JavaScript at all.

## Running it locally

Requires Node.js 22.12 or newer.

```bash
npm install
npm run dev        # development server at http://localhost:4321
npm test           # unit tests
npm run check      # type and content checks
npm run build      # static site in dist/
npm run preview    # serve the built site
```

## Project layout

```text
src/content/lessons/   lesson pages (MDX), one folder per track
src/components/widgets/ interactive figures: <Name>.svelte + sim.ts
src/lib/               shared engines (nn, cnn, transformer, rl, align, gen, ...)
src/styles/tokens.css  colour tokens shared by every figure, light and dark
tests/unit/            Vitest tests for the simulations
plans/                 per-day design notes and review audits
```

`IDEA.md` describes the product and curriculum. `LESSONS.md` is the guide every lesson and figure follows.
