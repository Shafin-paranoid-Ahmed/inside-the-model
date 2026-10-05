# Inside the Model — Project Idea & Specification

> **Name:** Inside the Model (chosen 2026-10-05; live at https://inside-model.vercel.app until a custom domain exists)
> **Status:** In build. Days 1–11 drafted; see `BUILDING.md` §1 for the live status table.
> **Owner:** Shafin Ahmed
> **Last updated:** 2026-09-29

This document is the single source of truth for **what** we are building and **why**. Anyone — human or AI agent — should be able to read this file and start contributing without any other context.

---

## Table of Contents

1. [The Idea in One Paragraph](#1-the-idea-in-one-paragraph)
2. [Inspiration](#2-inspiration)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Audience](#4-audience)
5. [Core Principles](#5-core-principles)
6. [The Learning Experience](#6-the-learning-experience)
7. [Curriculum — The 15-Day Path](#7-curriculum--the-15-day-path)
8. [Signature Interactive Widgets](#8-signature-interactive-widgets)
9. [Design Direction](#9-design-direction)
10. [Tech Stack](#10-tech-stack)
11. [Performance Budget](#11-performance-budget)
12. [Architecture & Project Structure](#12-architecture--project-structure)
13. [Quality, Testing & Accessibility](#13-quality-testing--accessibility)
14. [Roadmap](#14-roadmap)
15. [Open Questions](#15-open-questions)
16. [How to Contribute](#16-how-to-contribute)

---

## 1. The Idea in One Paragraph

A free, fast, beautifully clean website where anyone can **learn how AI actually works in about two weeks** — not by reading walls of text, but by **playing with interactive visualizations** of the real mechanisms. You drag a weight and watch the loss change. You press "step" and watch gradients flow backward through a network, node by node. You watch an RNN unroll through time, a graph neural network pass messages between nodes, attention heads light up over a sentence, and a diffusion model pull an image out of noise. It covers everything from "what is a model?" to the frontier of AI as of late 2026. No login, no paywall, no lag.

## 2. Inspiration

**[Quantum Physics Lab](https://quantum-physics-lab.vercel.app/)** — an interactive quantum physics course that runs in the browser. What we borrow from it:

- Concepts taught **one step at a time**, with adjustable parameters at every step.
- Abstract objects made **visible** (they color wavefunctions by phase; we color gradients by magnitude and sign).
- A **sandbox** where inputs update visuals in real time.
- **No login**, runs fully in the browser.
- **Presentation mode** for classrooms.
- **Deep links** to a specific step of a specific lesson.
- Dark / light theme.
- Clean, academic look.

Other reference points for quality of explanation: Distill.pub, 3Blue1Brown, Bret Victor's "Explorable Explanations", TensorFlow Playground, Karpathy's micrograd.

## 3. Goals & Non-Goals

### Goals

- **G1 — Intuition first.** Every hard concept gets an interactive visual that makes the mechanism *obvious*, before any equation.
- **G2 — Complete.** Covers beginner fundamentals through current frontier research (late 2026).
- **G3 — Two weeks.** A motivated learner can go through the whole core path in ~15 days at ~2–3 hours/day.
- **G4 — Refresher-friendly.** A researcher can land on any lesson from a search and get the key idea, the key equations and the key visual in under 2 minutes.
- **G5 — Fast.** No lag, ever. Pages load near-instantly; animations hold 60 fps on a mid-range laptop.
- **G6 — Portfolio-grade.** Code, design and content quality good enough to show employers.

### Non-Goals

- ❌ User accounts, logins, profiles, or a backend database.
- ❌ Payments, ads, or paywalls. The site is free forever.
- ❌ Running real large models in the browser. We use **small, honest toy models** that show the real mechanism.
- ❌ Being a coding bootcamp. We show short code snippets for grounding, but the focus is understanding, not exercises in an IDE.
- ❌ A news site. The frontier track is curated and updated deliberately, not a feed.

## 4. Audience

| Persona | Who | What they need |
|---|---|---|
| **The Curious Public** | No ML background, maybe high-school math | Plain language, visuals first, math hidden behind "Show the math" |
| **The Struggling Student** | Taking a university ML/DL course, stuck on backprop / RNNs / GNNs | Step-through visuals that connect the pictures to the equations they see in lectures |
| **The Engineer** | Can code, uses LLM APIs, wants to know what's under the hood | Transformers, LLMs, RAG, agents, inference optimization |
| **The Researcher** | Knows the field, needs a fast refresher on a specific topic | Refresher cards: TL;DR + key equations + the widget + key papers |

Design rule: **every lesson must work for the first and last persona at once.** Beginners follow the full lesson; researchers use the refresher card at the top.

## 5. Core Principles

1. **Show the mechanism, not a metaphor.** Visuals must show what really happens (real numbers, real gradients, real attention weights from a real tiny model), not a cartoon.
2. **One idea per step.** Lessons are split into small steps. Each step introduces exactly one new thing.
3. **Touch it.** Every lesson has at least one thing the learner can drag, toggle, or step through.
4. **Intuition → Picture → Math → Code.** Always in that order. Math and code are available but never block the picture.
5. **Consistent visual language.** The same color always means the same thing across the whole site (e.g. forward pass is always one color, gradients always another).
6. **Fast is a feature.** Performance budgets (§11) are requirements, not goals.
7. **Honest about limits.** Say when a toy model simplifies reality, and what the real thing does differently.
8. **Minimal code.** No abstractions until they are needed twice. Prefer native platform features. Few dependencies.

## 6. The Learning Experience

### 6.1 Site map

```
/                         Landing: what this is, the 15-day path, start button
/path                     The full curriculum map (all days, tracks, lessons, prerequisites)
/learn/<track>/<lesson>   A lesson
/learn/<track>/<lesson>#<step-title>   Deep link to a step
/refresher                Index of all refresher cards (for researchers)
/glossary                 Every term, linked to the lesson that teaches it
/math                     Optional math primer (vectors, matrices, derivatives, probability)
/about                    About the project & author (portfolio link)
```

### 6.2 Anatomy of a lesson

Every lesson follows the same structure so learners always know where they are:

1. **Refresher card** (collapsed by default for beginners, can be opened for experts)
   TL;DR in 2–3 sentences · key equations · key figure · "you need to know first" links.
2. **Hook** — one question or surprising fact that makes you want the answer.
3. **Steps** — the core. Each step = short text + interactive widget state. All steps are stacked on one scrolling page (never paginated); each step title is an anchor link.
4. **Show the math** — collapsible blocks next to the relevant step, KaTeX-rendered.
5. **Show the code** — collapsible, tiny NumPy / PyTorch snippet that matches the widget.
6. **Common misconceptions** — "You might think X. Actually Y."
7. **Check yourself** — 3–5 quick interactive questions, answered in the browser, nothing saved to a server.
8. **Going deeper** — key papers and best external resources.
9. **Next lesson** link.

### 6.3 Features

| Feature | Description |
|---|---|
| **Step-through lessons** | Arrow keys / buttons; each step is linkable |
| **Sandbox mode** | On any widget, unlock all controls and play freely |
| **Presentation mode** | `?present` — big fonts, hidden chrome, for teaching on a projector |
| **Dark / light theme** | Follows the system setting by default, toggle available |
| **Progress (local only)** | Completed lessons remembered in `localStorage`. No accounts. Works without it. |
| **Search** | Full-text search across all lessons, fully in the browser (Pagefind) |
| **Glossary hover** | Hover a technical term to see its definition and jump to its lesson |
| **Reduced motion** | Respects `prefers-reduced-motion`: animations become instant transitions |
| **Print / PDF friendly** | Lessons print cleanly with widget snapshots |

## 7. Curriculum — The 15-Day Path

Each day is ~2–3 hours. Lessons are also standalone — anyone can jump straight to any lesson. ⭐ marks the **flagship lessons**: the hardest-to-learn concepts that get the most elaborate widgets and get built first.

The **Math Primer** (vectors, dot products, matrices, derivatives, the chain rule, probability basics) is an optional side track, linked wherever a lesson depends on it.

### Week 1 — How learning works

**Day 1 — What is learning?**
- What a model is: a function with knobs (parameters)
- Data, features and labels
- Loss functions: measuring "how wrong"
- Linear regression — drag a line, watch the loss
- Gradient descent — a ball rolling down the loss surface ⭐

**Day 2 — Classification & generalization**
- Logistic regression and the sigmoid
- Softmax and cross-entropy
- Decision boundaries
- Overfitting, underfitting and the bias–variance trade-off
- Train / validation / test splits
- Regularization (L1, L2)

**Day 3 — Classic machine learning**
- k-Nearest Neighbors
- Decision trees and how they split
- Ensembles: random forests and gradient boosting
- Support vector machines and the kernel trick
- Unsupervised: k-means clustering, PCA
- When to use classic ML vs deep learning

**Day 4 — Neural networks**
- The neuron: weighted sum + activation
- Activation functions (step, sigmoid, tanh, ReLU, GELU) and why non-linearity matters
- Multi-layer perceptrons: stacking neurons
- What hidden layers learn — watching space get folded
- Universal approximation, intuitively

**Day 5 — Backpropagation ⭐⭐ (flagship)**
- Computational graphs
- The chain rule, visually
- Forward pass, then backward pass — step by step, node by node
- Backprop through a full MLP with real numbers
- Automatic differentiation (how PyTorch does it)
- Vanishing and exploding gradients

**Day 6 — Training deep networks**
- Optimizers: SGD, momentum, RMSProp, Adam — racing on a loss landscape
- Learning rates and schedules (warmup, cosine decay)
- Weight initialization
- Normalization: BatchNorm, LayerNorm, RMSNorm
- Dropout and other regularizers
- Loss landscapes and why deep nets train at all

**Day 7 — Convolutional neural networks**
- Convolution as a sliding filter
- Channels, stride, padding, pooling
- Receptive fields
- Classic architectures → ResNets and skip connections
- What CNNs "see": feature visualization

### Week 2 — Modern AI

**Day 8 — Sequences & recurrent networks ⭐⭐ (flagship)**
- Why sequences need memory
- The RNN cell and hidden state
- Unrolling through time
- Backpropagation through time (BPTT)
- Why vanilla RNNs forget: vanishing gradients over time
- LSTMs and GRUs: gates as valves
- Seq2seq and the first attention mechanism

**Day 9 — Transformers ⭐ (flagship)**
- Tokenization (BPE) — a live tokenizer playground
- Embeddings: words as points in space
- Self-attention: queries, keys, values
- Multi-head attention
- Positional encodings (sinusoidal, RoPE)
- The transformer block: attention + MLP + residual + norm
- Causal masking

**Day 10 — Large language models**
- Pretraining: next-token prediction at scale
- Sampling: temperature, top-k, top-p
- Scaling laws
- Inference: KV cache, batching
- Emergent abilities and in-context learning
- Hallucination: why it happens

**Day 11 — Reinforcement learning** (5 lessons, 165 min)
- RL basics: agents, rewards, policies, MDPs (30 min)
- Q-learning: Q-values, Bellman backups, tabular learning and exploration (30 min)
- Policy gradients: REINFORCE, baselines and actor-critic (35 min)
- Deep RL: function approximation, DQN, replay buffers and target networks (35 min)
- PPO: advantages, clipped objectives and a few epochs per batch (35 min); reused by Day 12 RLHF

**Day 12 — Alignment** (split from Day 11 on 2026-10-03, owner decision; the course is 15 days)
- Supervised fine-tuning (SFT)
- RLHF: reward models + PPO
- DPO and preference optimization
- Constitutional AI / RLAIF
- RL with verifiable rewards (the engine behind reasoning models)
- Evaluations and benchmarks — and their pitfalls

**Day 13 — Graphs & generative models**
- Graph neural networks ⭐⭐ (flagship): message passing, GCN, GAT, graph-level readout
- Autoencoders and latent spaces
- Variational autoencoders
- GANs
- Diffusion models: noising and denoising ⭐
- Flow matching

**Day 14 — Applied AI**
- Embeddings and vector search
- Retrieval-augmented generation (RAG)
- Prompting techniques that actually work
- Tool use and function calling
- Agents: loops, planning, memory, multi-agent systems
- Multimodal models: ViT, CLIP, vision-language models
- Fine-tuning efficiently: LoRA and QLoRA

**Day 15 — The frontier (late 2026)**
- Reasoning models and test-time compute
- Mixture of Experts (MoE)
- State-space models (Mamba) and hybrid architectures
- Long context: how models handle millions of tokens
- Efficient inference: quantization, speculative decoding, distillation
- Mechanistic interpretability: probing, sparse autoencoders, circuits
- AI safety: alignment problems, evaluations, red-teaming
- What's new — a curated, dated list of recent breakthroughs

> ⚠️ **The frontier track must be researched from primary sources before writing.** Each frontier lesson carries a "last reviewed" date and is re-reviewed at least every 6 months.

## 8. Signature Interactive Widgets

These are the widgets that make the site worth visiting. All run real math on tiny models in the browser.

### 8.1 Backprop Visualizer ⭐ (Day 5)
- A small network (2 inputs → 3 hidden → 1 output) drawn as a graph.
- **Forward mode:** press step — values flow left-to-right, each node shows its number.
- **Backward mode:** gradients flow right-to-left. Each edge shows ∂L/∂w, colored by sign, thickness by magnitude.
- Click any node to see the chain rule written out **with the actual numbers** at that node.
- Toggle "update weights" to apply one gradient descent step and watch the loss drop.
- Also a "computational graph" mode for a single expression (e.g. `(a·b + c)²`), micrograd-style.

### 8.2 RNN Unroller ⭐ (Day 8)
- A single RNN cell, shown folded; press "unroll" to animate it into a chain across time steps.
- Type a short sequence; watch the hidden state vector (as a colored bar) update at each step.
- BPTT mode: gradients flow backward through time; watch them shrink (vanishing) or blow up (exploding) as you change the recurrent weight.
- Swap the cell for an LSTM: see the forget / input / output gates open and close as valves.

### 8.3 GNN Message Passing ⭐ (Day 13)
- An editable graph (add/remove nodes and edges, drag nodes).
- Each node has a feature vector, shown as a small color swatch.
- Press step: messages animate along edges, get aggregated, node features update.
- Show how after k steps each node "knows" about its k-hop neighborhood (highlight the receptive field).
- Toggle GCN vs GAT: GAT shows attention weights on edges.

### 8.4 Loss Landscape & Optimizers (Days 1, 6)
- A 2D loss surface (contour and optional 3D view).
- Drop balls for SGD, momentum, RMSProp and Adam; watch them race.
- Sliders for learning rate and momentum; see overshooting and divergence live.

### 8.5 Attention Explorer (Day 9)
- A real tiny transformer's attention weights on an input sentence.
- Hover a token to see what it attends to; switch heads and layers.
- Step-by-step Q·Kᵀ → scale → softmax → ·V with the real matrices.

### 8.6 Others
- **Neuron playground** (Day 4) — weights, bias, activation; watch the decision boundary move.
- **Space folding** (Day 4) — watch a hidden layer bend a 2D space until classes separate.
- **Convolution sliding window** (Day 7) — filter slides over an image; output builds up pixel by pixel.
- **Tokenizer playground** (Day 9) — type text, see BPE tokens and IDs.
- **Embedding space** (Day 9) — 2D projection of word embeddings; vector arithmetic.
- **Sampling lab** (Day 10) — adjust temperature/top-p on a real next-token distribution.
- **Diffusion denoiser** (Day 13) — scrub a slider from pure noise to image and back.
- **RAG pipeline** (Day 14) — a query flows through embed → search → retrieve → generate.
- **MoE router** (Day 15) — tokens get routed to experts; see load balancing.

## 9. Design Direction

**Clean and academic.** Think a well-typeset textbook that came alive, not a startup landing page.

### Typography
- Body: a readable serif (e.g. *Source Serif 4*)
- UI & headings: a clean sans (e.g. *Inter*)
- Code & numbers: a monospace (e.g. *JetBrains Mono*), tabular figures for numbers that change
- Fonts are **self-hosted and subset** — no external font requests.

### Color
- Neutral paper-like background, near-black text, one restrained accent.
- A **fixed semantic palette for visualizations**, used identically on every page:
  | Meaning | Usage |
  |---|---|
  | Forward pass / activations | one consistent hue |
  | Gradients / backward pass | a second, contrasting hue |
  | Positive vs negative values | diverging scale (e.g. blue ↔ red), colorblind-safe |
  | Attention / weights | sequential scale |
  | Highlight / focus | the accent color |
  | Categories (identity, no order) | `--viz-cat-1..8`, fixed order, always with a label or shape (added 2026-10-03) |
- All colors defined as CSS custom properties with light and dark variants.
- WCAG AA contrast minimum everywhere.

### Motion
- Motion explains, never decorates. Every animation shows causality (this caused that).
- Springs and easing via Motion (motion.dev, vanilla API).
- Durations short (150–400 ms for UI, longer only for explanatory sequences the user controls).
- Everything respects `prefers-reduced-motion`.

### UI references
- **motion.dev** — animation engine (framework-agnostic core, used directly).
- **Kokonut UI**, **Bklit** — React libraries used as **visual inspiration only**; the components we like are rebuilt in Svelte so we don't ship React.

## 10. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Framework | **Astro** | Every page is static pre-rendered HTML; JS only ships for interactive "islands" |
| Interactive widgets | **Svelte 5** (runes) | Compiles away, tiny runtime, fine-grained reactivity, built-in transitions |
| Language | **TypeScript** (strict) | Safety for the math and simulation code |
| Lesson content | **MDX** via Astro content collections | Prose with widgets embedded inline; frontmatter validated by schema |
| Math | **KaTeX, rendered at build time** (remark-math + rehype-katex) | Zero math JS shipped to the browser |
| Visuals | **SVG** (diagrams, ≤ few hundred elements), **Canvas 2D** (dense or fast animation), **WebGL / WebGPU** (only where needed) | Cheapest renderer that holds 60 fps |
| Data helpers | **D3 modules** (`d3-scale`, `d3-shape`, `d3-force`, …) | Math and layout only — Svelte owns the DOM |
| Animation | **Motion** (`motion`, vanilla API) + Svelte transitions | Smooth springs without a React dependency |
| Heavy compute | **Web Workers** (TypeScript) | Training loops never block the UI thread |
| Search | **Pagefind** | Static full-text search, no server |
| Styling | Plain CSS with custom properties (scoped in Svelte/Astro) | No CSS framework needed |
| Unit tests | **Vitest** | For the pure math / simulation modules |
| E2E & perf tests | **Playwright** + **Lighthouse CI** | Screenshots, smoke tests, performance gates |
| Hosting | **Vercel** (static) | Free, deploys on every push, preview URLs |

### Dependency policy
- Every new dependency must justify itself against "could we write this in < 50 lines?"
- No React. No UI kit. No CSS framework.
- Prefer native platform features (`<details>`, `<dialog>`, CSS `:has()`, View Transitions API).

### Tooling available to AI agents working on this repo
- **Svelte MCP** — official Svelte 5 docs + autofixer. Run the autofixer on every Svelte component.
- **Astro docs MCP** — Astro documentation.
- **context7** — current docs for other libraries (D3, Motion, KaTeX, Pagefind…).
- **Playwright MCP** — open pages, take screenshots, measure performance.

## 11. Performance Budget

These are **hard requirements**. A lesson that misses them is not done.

| Metric | Budget |
|---|---|
| Lighthouse Performance | ≥ 95 on every page (mobile profile) |
| Largest Contentful Paint | < 1.2 s |
| Cumulative Layout Shift | 0 (widgets reserve their space) |
| Interaction to Next Paint | < 100 ms |
| Animation frame rate | 60 fps on a mid-range laptop, no dropped frames during widget interaction |
| JS on a lesson page before any widget loads | < 15 KB gzipped |
| JS per widget island | < 40 KB gzipped (target), lazy-loaded with `client:visible` |
| Model weights | Fetched as data from `public/models/` when the widget is visible, cached across pages, outside the JS budget (owner decision 2026-10-03). Float16, int8 past 40 KB per file |
| Fonts | Self-hosted, subset, `font-display: swap`, preload only the body font |

### Rules to hit it
- Widgets load **only when scrolled into view** (`client:visible`).
- Simulations run in **Web Workers**; the main thread only renders.
- Use `requestAnimationFrame`; never animate layout properties (only `transform` and `opacity`).
- Prefer Canvas over SVG once element counts get large.
- Use typed arrays (`Float32Array`) for all numeric work.
- Pause animations when the widget is off-screen or the tab is hidden.
- No layout thrashing: batch DOM reads before writes.

## 12. Architecture & Project Structure

```
/
├── IDEA.md                    ← this file
├── CLAUDE.md                  ← guidance for AI agents
├── astro.config.mjs
├── package.json
├── public/
│   └── fonts/                 ← self-hosted, subset fonts
├── src/
│   ├── content/
│   │   ├── config.ts          ← content collection schemas
│   │   └── lessons/
│   │       └── <track>/<lesson>.mdx
│   ├── components/
│   │   ├── ui/                ← buttons, step, cards, theme toggle (Svelte / Astro)
│   │   └── widgets/
│   │       └── <WidgetName>/
│   │           ├── <WidgetName>.svelte   ← rendering + interaction only
│   │           ├── sim.ts                ← pure logic, no DOM (unit tested)
│   │           └── worker.ts             ← optional, for heavy compute
│   ├── lib/
│   │   ├── nn/                ← tiny autograd engine + layers (shared by widgets)
│   │   ├── math/              ← vectors, matrices, RNG with seed
│   │   └── render/            ← canvas helpers, color scales
│   ├── layouts/
│   │   ├── Base.astro
│   │   └── Lesson.astro
│   ├── pages/
│   └── styles/
│       └── tokens.css         ← all colors, spacing, type as CSS variables
└── tests/
    ├── unit/                  ← Vitest
    └── e2e/                   ← Playwright
```

### Key architectural decisions
- **Logic is separated from rendering.** Every widget's math lives in a pure `sim.ts` with no DOM access. This makes it testable and movable into a worker.
- **One shared tiny autograd engine** (`src/lib/nn`, micrograd-style, TypeScript) powers backprop, RNN, MLP and optimizer widgets, so the numbers shown are always consistent and real.
- **Seeded randomness** so every widget is reproducible and deep links show the same state.
- **Lesson frontmatter schema** (validated at build time):
  ```yaml
  title: Backpropagation
  day: 5
  track: neural-networks
  order: 3
  level: core          # intro | core | advanced | frontier
  duration: 35         # minutes
  prereqs: [neural-networks/mlp, math/chain-rule]
  summary: How gradients flow backward through a network.
  lastReviewed: 2026-09-29   # required for frontier lessons
  ```
- **Steps in MDX:** a `<Step>` component wraps each step; the lesson layout stacks them on one page; step titles are anchors.

## 13. Quality, Testing & Accessibility

### Testing
- **Unit tests (Vitest)** for all `sim.ts` and `src/lib` code — especially the autograd engine (gradient checks against numerical derivatives).
- **E2E (Playwright)**: every lesson page loads, has no console errors, every step is reachable.
- **Visual snapshots** of each widget in light and dark themes.
- **Lighthouse CI** gate on every PR against the §11 budget.

### Content quality
- Every lesson is technically reviewed for correctness.
- Every simplification is labeled ("In real models…").
- Papers are cited with links.

### Accessibility (WCAG 2.2 AA)
- All widgets operable by keyboard.
- Every visual state has a text equivalent (screen-reader live region describing what changed).
- Color is never the only signal (use shape, pattern, or labels too).
- Colorblind-safe palettes.
- `prefers-reduced-motion` respected.
- Focus states visible.

## 14. Roadmap

| Milestone | Scope | Done when |
|---|---|---|
| **M0 — Foundations** | This doc, CLAUDE.md, repo, Astro + Svelte scaffold, CI | Empty site builds and deploys |
| **M1 — Shell & design system** | Tokens, typography, layouts, theme toggle, lesson layout with steps, `/path` curriculum map with all lessons listed (stubs) | Clicking through the whole path works, on budget |
| **M2 — Flagship lessons** | Backprop (Day 5), RNNs (Day 8), GNNs (Day 13), Gradient descent (Day 1) + the shared autograd engine | The four hardest topics are fully built |
| **M3 — Week 1** | All remaining Day 1–7 lessons | Week 1 complete |
| **M4 — Week 2** | All remaining Day 8–13 lessons | Week 2 complete |
| **M5 — Frontier** | Day 15, researched from primary sources | Frontier complete with review dates |
| **M6 — Polish & launch** | Search, glossary, refresher index, presentation mode, a11y audit, perf audit, deploy to Vercel, portfolio write-up | Public launch |

Flagship lessons (M2) come before filling in easier lessons because they are the reason the site exists and they drive the shared engine design.

## 15. Open Questions

- **Domain.** The name is Inside the Model (2026-10-05). A custom domain is still needed; the site runs on `inside-model.vercel.app` until then.
- **License.** Suggested: code under MIT, lesson content under CC BY-NC-SA 4.0.
- **Analytics.** None, or a privacy-friendly option (e.g. Vercel Analytics without cookies)?
- **Languages.** English only at launch; i18n later?
- **Community contributions.** Open to outside PRs for lessons, or author-only?

## 16. How to Contribute

### Adding a lesson
1. Create `src/content/lessons/<track>/<lesson>.mdx` with valid frontmatter (§12).
2. Follow the lesson anatomy (§6.2): refresher card, hook, steps, math, code, misconceptions, check-yourself, going deeper.
3. Reuse existing widgets where possible.
4. Verify: page builds, all steps reachable, Lighthouse ≥ 95.

### Adding a widget
1. Create `src/components/widgets/<Name>/` with `<Name>.svelte` and `sim.ts`.
2. Put all math in `sim.ts` and unit-test it.
3. Use the semantic color tokens from `tokens.css` — never hard-code colors.
4. Make it keyboard-operable and add a text description for screen readers.
5. Measure: 60 fps while interacting, bundle under budget.
6. Run the Svelte MCP autofixer on the component.

### Standards
- TypeScript strict; no `any`.
- Svelte 5 runes syntax only (`$state`, `$derived`, `$effect`, `$props`).
- Keep it minimal: no abstraction until it's needed twice.
- Commit messages: short, imperative ("Add RNN unroller widget").
