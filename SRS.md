# Software Requirements Specification — Inside the Model

**Document status:** Draft, refreshed from current source  
**Version:** 0.4  
**Date:** 2026-10-05  
**Product owner:** Shafin Ahmed  
**Product name:** Inside the Model  
**System type:** Static interactive educational website  
**Baseline note:** Inventory and implementation states refreshed from the working tree on 2026-10-05. Earlier wave results, including Day 10 acceptance, are attributed to their recorded checkpoints; they are not fresh validation of subsequent edits. Counts and completion states must be refreshed before a release is declared.

---

## 1. Purpose

This Software Requirements Specification (SRS) defines the intended behavior, architecture, interfaces, content model, constraints, quality requirements, and acceptance criteria for Inside the Model.

It is intended to:

- Give human and AI contributors a single implementation-oriented reference.
- Separate implemented functionality from planned functionality.
- Prevent lesson stubs, placeholder pages, or aspirational documentation from being treated as finished.
- Provide stable requirement identifiers for implementation, review, testing, and release planning.
- Define what “done” means for lessons, widgets, pages, milestones, and the complete product.

### 1.1 Normative language

- **Shall** means a mandatory requirement.
- **Should** means an expected requirement unless a documented reason prevents it.
- **May** means an optional capability.
- **Implemented** means the behavior was present in the inspected source.
- **Partial** means some required behavior exists, but the complete requirement was not demonstrated.
- **Planned** means the behavior is specified but was not implemented in the inspected source.
- **Required** means an acceptance obligation whose complete implementation or verification has not been established.
- **Open** means an owner decision is required.

### 1.2 Source-of-truth hierarchy

When project documents disagree, contributors shall use this order:

1. Current owner instructions.
2. This SRS after it has been deliberately updated to reflect those instructions.
3. Tested behavior in the current source tree.
4. `IDEA.md` for product vision and curriculum.
5. `LESSONS.md` for lesson and widget authoring conventions.
6. `CLAUDE.md` for contributor workflow and engineering guidance.

An observed implementation shall not silently redefine a stated product requirement. A disagreement shall be recorded and resolved explicitly.

---

## 2. Product overview

Inside the Model is a free, no-login website that teaches how artificial intelligence works through interactive explanations of real mechanisms. It is intended to take a motivated learner from introductory ideas to modern AI topics through a fifteen-day path while also serving as a quick refresher for experienced readers.

The site uses small, deterministic models that run in the browser. It emphasizes direct manipulation, visible numerical state, technically honest simplifications, and a progression from intuition to visualization, mathematics, and code.

### 2.1 Product principles

The product shall prioritize:

1. Intuition before formalism.
2. Real mechanisms rather than decorative or misleading metaphors.
3. Direct manipulation and immediate visual feedback.
4. One new teaching idea per lesson step.
5. A consistent progression from picture to math to code.
6. Reproducible toy models that expose authentic mechanisms.
7. Static delivery with minimal client-side JavaScript.
8. Accessibility, privacy, and performance.
9. Honest descriptions of simplifications and limitations.
10. Standalone lessons suitable for search-driven reference use.

### 2.2 Goals

- **G-001 — Intuition:** Difficult concepts shall receive an interactive explanation before formal mathematical treatment.
- **G-002 — Breadth:** The curriculum shall span fundamentals, classic machine learning, deep learning, transformers, LLMs, alignment, generative models, applied AI, and frontier topics.
- **G-003 — Fifteen-day path:** The core curriculum shall be organized into fifteen days (Day 11 split into RL and alignment on 2026-10-03) of approximately two to three hours each.
- **G-004 — Refresher value:** An experienced reader should be able to obtain a lesson’s key idea, equations, visual, and references quickly.
- **G-005 — Speed:** Pages and interactive figures shall meet the performance budgets in this document.
- **G-006 — Quality:** Content, code, visuals, and citations shall be suitable for a public portfolio project.
- **G-007 — Privacy:** Normal use shall require no account and no transmission of learner progress to a server.

### 2.3 Non-goals

The initial product shall not include:

- User accounts or authentication.
- A backend application server or database.
- Server-side learner profiles or cloud progress synchronization.
- Payments, subscriptions, advertisements, or paywalls.
- Execution of production-scale models in the browser.
- A hosted programming exercise environment or notebook.
- A continuously updated AI news feed.
- Undisclosed analytics or tracking.
- React, CSS frameworks, or general-purpose UI kits.

---

## 3. Stakeholders and users

### 3.1 Product owner

The product owner controls:

- Product direction and scope.
- Brand and domain decisions.
- Curriculum scope and ordering.
- Content acceptance.
- Licensing and analytics policy.
- Production launch approval.

### 3.2 Learner personas

#### Curious beginner

- May have little machine-learning background.
- Needs plain language and visual explanations.
- Must be able to progress without opening advanced math sections.

#### Student

- Needs connections among diagrams, equations, and course terminology.
- Benefits from step-by-step explanations of difficult mechanisms.

#### Engineer

- Needs transformers, LLMs, inference, RAG, agents, and fine-tuning concepts.
- Expects technical accuracy without a full programming course.

#### Researcher

- Needs concise refreshers, equations, limitations, and primary references.
- Should be able to extract the central idea from a lesson quickly.

#### Instructor

- Needs presentation mode, readable projected visuals, deterministic examples, and stable deep links.

#### Contributor

- Needs explicit schemas, architectural constraints, testing rules, and definitions of done.

---

## 4. Current repository baseline

This section records the state observed during the 2026-10-05 source inspection of `G:\PROJECTS\RECOVERY\Ai learner website`. It is evidence, not a release claim. The working files remain untracked in Git, so this is a dated working-tree snapshot rather than a committed revision.

### 4.1 Inventory

| Area | Observed state |
|---|---|
| Curriculum | 98 MDX lesson entries |
| Lesson status | 67 draft, 31 stub, 0 done |
| Estimated lesson duration | 2,710 minutes including the 115-minute math primer; 2,595 minutes for the core path |
| Draft lesson steps | 605 `<Step>` instances |
| Hydrated widget placements | 379 hydration directives in lesson MDX |
| Widget/support directories | 68 excluding `_shared`: 63 component/simulation pairs, 4 empty widget directories, and `F1Shared` support helpers |
| Complete widget component/simulation pairs | 63 by file presence; this does not establish acceptance |
| Empty widget directories | `FlowField`, `GANDuel`, `GenCommon`, `LatentSpace` |
| Unit-test files | 66 matching `tests/unit/**/*.test.ts` |
| Worker files | 10: `FeatureViz`, `HallucinationLab`, `ICLLab`, `ModelChooser`, `ReceptiveField`, `ResidualTrainer`, `SamplingLab`, `Seq2SeqAttention`, `SpaceFolding`, `TrainingHero` |
| Backend/API/database | None |
| Authentication | None |
| Production CI | Not present in the inspected tree |
| E2E suite | Placeholder directory; standalone Playwright check scripts exist |
| Search | Not implemented |
| Learning progress | Not implemented |
| Glossary index | Placeholder page |
| Refresher index | Placeholder page |
| Presentation mode | Partial |
| Theme selection | Implemented |
| SEO foundation | Implemented |
| Production domain | Placeholder |

The refresh counted lesson metadata, draft `<Step>` tags, lesson hydration directives, widget files, workers, and unit-test files directly from source. It also inspected package/configuration files, layouts, authoring guidance, model loading/core files, and model artifacts. Saved-report descriptions below are retained historical evidence, not fresh browser verification. No build, unit test, browser test, accessibility audit, or performance benchmark was executed during this documentation refresh. Recorded checks below belong to earlier development checkpoints.

### 4.2 Curriculum maturity

| Track | Draft | Stub | Total |
|---|---:|---:|---:|
| Math primer | 6 | 0 | 6 |
| Foundations | 11 | 0 | 11 |
| Classic ML | 7 | 0 | 7 |
| Neural networks | 17 | 0 | 17 |
| CNNs | 5 | 0 | 5 |
| Sequences | 7 | 0 | 7 |
| Transformers | 7 | 0 | 7 |
| Large language models | 6 | 0 | 6 |
| RL | 0 | 5 | 5 |
| Alignment | 0 | 6 | 6 |
| Graphs and generative models | 1 | 5 | 6 |
| Applied AI | 0 | 7 | 7 |
| Frontier | 0 | 8 | 8 |
| **Total** | **67** | **31** | **98** |

`BUILDING.md` records coordinator acceptance for the math primer and Days 1–10 (including the Day 9 tokenizer lesson and all six Day 10 LLM lessons); Day 6 has a later coordinator clarity revision. Owner clarity review remains pending for Days 6–10. Days 11–15 remain planned/stubbed, with a Day 13 GNN draft exception. All authored lessons retain `status: draft`; coordinator acceptance does not set `status: done` or establish release readiness.

Frontmatter estimates exceed the two-to-three-hour promise on Days 5 (185 minutes), 8 (205), 9 (210), 13 (215), 14 (205), and 15 (250); Day 1 is 110 minutes. These estimates require a scope or workload review, not an assumption that the course already meets G-003. Day 10's six drafted lessons total 160 minutes.

### 4.3 Known discrepancies requiring resolution

1. `IDEA.md` describes the project as “Planning. No code yet,” although substantial code exists.
2. The current owner-directed sequence is one day at a time; Day 10 is coordinator-accepted and Day 11 is next, with Days 11–15 remaining. Owner clarity review for Days 6–10 is pending. Day 6's original acceptance and clarity revisions are historical; they do not require restarting the sequence.
3. Architecture guidance requires lesson steps to remain stacked on one scrolling page, while `scripts/check-curriculum.mjs` expects paginated controls such as `data-next`, `data-count`, and `.is-active`.
4. Product documentation mentions `/math`, while math lessons currently use `/learn/math/<slug>`.
5. Pagefind, Motion, D3, Lighthouse CI, and deployment infrastructure are described as planned technologies but are not installed in the inspected package manifest.
6. Authoring guidance calls for workers for heavy training, while architecture guidance makes them conditional on workload. Ten worker files exist; this does not establish that every heavy workload meets the performance budget.
7. The configured site URL is a placeholder and must be replaced with an owner-approved production domain.
8. All visible project files appeared untracked in Git during the inspection, so the inspected tree was not a stable historical baseline.
9. The schema accepts `duration` as an unrestricted number even though this SRS requires positive minute values; a source count is not duration validation.
10. Historical wave evidence in `BUILDING.md` predates later edits and must not be presented as current validation. The current inventory has 66 unit-test files; the current Day 6 lessons contain 34 hydrated widget placements, while saved reports covering earlier revisions are not acceptance evidence for the current source.
11. The week-plan calls for model caching across pages, but `src/lib/transformer/load.ts` memoizes promises in a module-local `Map`; it shares requests and results within the loaded module lifetime. Application-managed cross-page persistence is not implemented; browser HTTP caching is separate.

---

### 4.4 Historical Day 4 continuation — 2026-10-01

The five Day 4 lessons and their originally assigned widgets are now written and coordinator-reviewed: NeuronPlayground, ActivationLab, MLPForward, SpaceFolding and BumpBuilder. The interrupted MLP and hidden-layer stubs were completed using the partial widget/simulation files. All five lessons retain `status: draft` under the authoring workflow; acceptance in `BUILDING.md` is distinct from production `done` and release approval.

Historical wave evidence (not rerun for this refresh): 402 unit tests pass in 39 files, Astro check reports zero errors/warnings/hints, each Day 4 Svelte component compiles with zero warnings, and the static build produces 101 pages. Browser walkthroughs and multi-time screenshot review cover light/dark desktop and mobile; keyboard seeking, reduced-motion behavior, theme switching and expanded mathematics pass. A production check at 320 px caught a shared navigation overflow, then resolved by allowing the header to wrap.

At the Day 4 checkpoint, widget chunks measured 5.5–9.7 KB gzip and their dependency closures measured 26.0–35.0 KB, below 40 KB. SpaceFolding trains in a worker, provides accessible numerical captions and displays an explanatory error when worker loading fails. Approximation errors are identified as sampled measurements. Lighthouse/LCP/INP and a full WCAG audit remain separate release gates. This is historical wave evidence from `BUILDING.md`.

---

### 4.5 Historical Day 5 acceptance refresh — 2026-10-01

The six existing Day 5 lessons were audited and revalidated before Day 6 implementation started. Corrections include the high-learning-rate arithmetic example, the chain-rule product reveal, mobile keyboard scrubbing, exact numerical labels during reveals, recomputed loss at interpolated update weights, and qualified residual/normalization claims. Missing mechanism figures reuse the original widgets. Long inline equations and code no longer widen mobile pages.

The Day 5 checkpoint passed 405 tests in 39 files, zero Astro diagnostics, a 101-page static build, and desktop/mobile light/dark multi-time browser checks including keyboard, reduced-motion, autoplay/pause, theme switching and expanded mathematics. Widget dependency closures measured 24.2–32.8 KB gzip. Evidence is recorded in `BUILDING.md` and `.playwright-mcp/check/day5-acceptance/`. Lessons remain `draft`; this checkpoint is not a production release or full accessibility/performance approval.

---

### 4.6 Historical Day 6 implementation and acceptance — 2026-10-02

**Implemented and coordinator-reviewed:** the six Day 6 lessons on optimizers, learning-rate schedules, initialization, normalization, dropout/regularizers and loss landscapes now use six dedicated Svelte widgets with pure, unit-tested simulations. Optimizer/schedule comparisons reuse shared optimizer state and actual gradients. The remaining demonstrations compute their displayed statistics, mask expectations, decay updates, stopping history, objective slices and residual derivatives; toy-model limitations and coordinate-dependent sharpness are explicit. Day 5 also passed additional 320 px checks after narrow pager and autodiff-control fixes.

The continuation checkpoint passes **456 tests in 46 files**, zero Astro diagnostics, zero Svelte compiler warnings for the six new widgets, and a **101-page** static build. Browser evidence covers both themes at 1280/390/320 px, multiple timeline states, keyboard seeking, reduced motion, reset, autoplay/pause, theme changes, parameter controls and expanded mathematics. Additional layout checks cover 768/1024/1440 px, with numeric-input and early-stop endpoint regressions. Widget dependency closures measure approximately **24.2–28.2 KB gzip**, below the 40 KB budget. Evidence and reproducible commands are recorded in `BUILDING.md` and `.playwright-mcp/check/day6-acceptance/`.

**Still required for release:** Lighthouse/LCP/INP measurement and a full accessibility audit. These lessons retain `status: draft`; coordinator acceptance is not release approval. This initial acceptance was superseded by owner feedback and later teaching revisions.

### 4.7 Historical Day 6 teaching and clarity refresh — 2026-10-02

The owner found Day 6 confusing despite its technical checks. The first teaching revision added visible update calculations, guided interactions, signal/gradient bars, normalization calculations, dropout contributions, and loss-height cross sections. `BUILDING.md` records 462 passing tests in 47 files, zero Astro diagnostics, a 101-page build, six clean Svelte components, and widget dependency closures of 25.3–29.7 KB gzip at that checkpoint. It also records remaining Astro/Vite head-inject directive warnings. These figures were not rerun during this refresh.

Saved `.playwright-mcp/check/day6-teaching/results.json` and `control-results.json` both report `passed: true`, covering 36 lesson layouts and 32 additional control layouts respectively. Their six-lesson set contains 43 widget instances. Current MDX contains 34: optimizers 6, schedules 7, initialization 4, normalization 6, dropout/regularizers 6, and landscapes 5. `BUILDING.md` §5a describes a later coordinator revision removing redundant figures, clarifying prose, correcting defaults/readouts, and adding meaningful plot context and logarithmic scales. Source inspection confirms these changes in representative lessons/widgets, but the saved earlier reports do not establish acceptance of the later revision.

**Checkpoint status then: Partial.** Day 6 was authored and revised, with owner clarity review pending at that checkpoint. Later `BUILDING.md` entries document revisions through Day 9; current acceptance state is summarized in §4.2. The 464-test checkpoint, 45 test files, 101-page build, and browser reports are historical evidence only and were not rerun for this SRS refresh. Technical checks and screenshots alone do not establish learner comprehension.

### 4.8 Days 7–10 coordinator acceptance — 2026-10-05

`BUILDING.md` records Days 7–10 as accepted and wave-checked, with owner clarity review pending for Days 6–10. All six Day 10 lessons (`pretraining`, `sampling`, `scaling-laws`, `inference-kv-cache`, `emergent-icl`, `hallucination`) are drafts. Their widgets are PretrainLab, SamplingLab, ScalingLab, KVCacheLab, ICLLab, and HallucinationLab; SamplingLab, ICLLab, and HallucinationLab have workers. Day 11 reinforcement learning is next.

The recorded Day 9 checkpoint in `BUILDING.md` reports 628 tests in 59 files, zero Astro diagnostics, a 104-page build, and widget dependency closures of 27.0 to 39.3 KB gzip. These checks were not rerun here and do not validate subsequent Day 10 edits.

The recorded Day 10 checkpoint (2026-10-04/05) reports 715 passing tests, zero Astro diagnostics, a 104-page build without source maps, and widget dependency closures of 30.9–38.8 KB gzip. It records clean desktop/mobile checks on all six lessons and a Day 9 recheck with the new per-step stall gate; the worst measured stall was 67 ms on that machine. These are attributed development results, not checks rerun for this document or proof of the separate release accessibility/performance gates.

Day 10 demonstrations distinguish measured small-model results from published scaling fits. The measured sweep does not show an interior IsoFLOP optimum within its one-epoch data cap; the ICL model does not exhibit a clean induction head and its size sweep is non-monotone; the alternative facts model answering an unknown token is not shipped. `plans/day-10.md` §6 records these boundaries. The SRS does not generalize those experiments into production-model behavior.

Day 10 review also found multi-second main-thread freezes that the earlier first-step FPS check missed. Current `scripts/check-page.mjs --steps` measures animation-frame gaps during each step's hydration/autoplay window, records `stepMaxGapMs`, and reports gaps over 100 ms as errors. The documented fixes use raw model state, shared memoized computation, chunking, and workers. This bounded sampling is a runtime regression check, not an INP benchmark or a guarantee for every later interaction.

The shared transformer core exists under `src/lib/transformer/`: model layout and forward/training math, serialization/manifest parsing, and a lazy `loadModel(name)` loader. `public/models/` contains binary weights and JSON manifests for house, facts, ICL, and tiny-LM variants, plus logs/sweep artifacts. The loader starts no fetch at import time, fetches manifest and binary when called, validates model/layout/byte length and optional SHA-256, decodes float16 or per-row int8 weights, and memoizes promises in a module-local map. That map is not cross-page persistent caching. This implementation is inspected source evidence; no new test, build, performance, accessibility, or browser validation was run for this SRS update.

---

## 5. System context and architecture

The production system shall be a statically generated Astro site.

```text
MDX lessons + TypeScript simulations
                │
                ▼
          Astro static build
     ┌──────────┴──────────┐
     │                     │
Static HTML/CSS       Lazy Svelte islands
     │                     │
     └──────────┬──────────┘
                ▼
        Learner's browser
                │
       local preferences only
```

### 5.1 Technology requirements

| Layer | Required or intended technology |
|---|---|
| Framework | Astro, statically generated |
| Interactive islands | Svelte 5 |
| Language | TypeScript in strict mode |
| Content | MDX through Astro content collections |
| Math | Build-time KaTeX using remark/rehype plugins |
| Rendering | SVG, Canvas 2D, WebGL2, or WebGPU according to measured need |
| Styling | Plain CSS and custom properties |
| Fonts | Locally hosted variable fonts |
| Unit tests | Vitest |
| Browser checks | Playwright |
| Search | Static client-side search; Pagefind is the current intended option |
| Hosting | Static hosting; Vercel is the current intended option |

### 5.2 Architectural boundaries

- Astro pages and layouts shall own static document structure and metadata.
- MDX shall own lesson prose and composition.
- Svelte components shall own interactive rendering and user input.
- Pure `sim.ts` modules shall own mathematical and timeline behavior.
- Workers shall be used when measured computation would otherwise violate interaction or frame-rate budgets.
- `src/lib/nn` shall provide shared neural-network primitives.
- `src/lib/cnn` shall provide shared convolution and pooling operations; `src/lib/transformer` shall provide the small-transformer model/training/inference core; Day 10 LM-specific helpers belong in `src/lib/lm` and shall reuse that core.
- `src/lib/math` shall provide reusable deterministic mathematical helpers.
- `src/styles/tokens.css` shall define shared visual tokens.
- Curriculum ordering shall be centralized rather than duplicated across pages.

### 5.3 Dependency policy

- A dependency shall justify itself against a small native implementation.
- React shall not be introduced.
- UI kits and CSS frameworks shall not be introduced.
- Native browser features shall be preferred when they provide the required behavior.
- Svelte shall own DOM rendering; mathematical libraries shall not mutate component DOM.

---

## 6. Functional requirements

### 6.1 Site shell and navigation

- **FR-SHELL-001 — Implemented:** The system shall provide global navigation to Path, Refresher, Glossary, and About.
- **FR-SHELL-002 — Implemented:** Public pages shall use a shared header and footer.
- **FR-SHELL-003 — Implemented:** The active primary navigation item shall use `aria-current`.
- **FR-SHELL-004 — Implemented:** The system shall provide a custom 404 page with recovery links.
- **FR-SHELL-005 — Implemented:** The document shall expose a single primary content region.
- **FR-SHELL-006 — Required:** Every page shall contain exactly one `<h1>`.
- **FR-SHELL-007 — Required:** Navigation shall remain usable at mobile widths and with keyboard-only input.
- **FR-SHELL-008 — Planned:** Final author identity and portfolio links shall replace About-page placeholders.

### 6.2 Landing page

- **FR-HOME-001 — Implemented:** The landing page shall communicate the fifteen-day course proposition.
- **FR-HOME-002 — Implemented:** Its primary action shall link to the first curriculum lesson.
- **FR-HOME-003 — Implemented:** It shall display both curriculum weeks and all fifteen day headings.
- **FR-HOME-004 — Implemented:** It shall include an interactive training demonstration.
- **FR-HOME-005 — Implemented:** It shall include a loss-landscape demonstration.
- **FR-HOME-006 — Required:** Counts shall be derived from curriculum data rather than manually duplicated.
- **FR-HOME-007 — Required:** Demonstrations shall remain enhancements; core information shall exist in static HTML.

### 6.3 Curriculum map

- **FR-CUR-001 — Implemented:** Lessons shall be grouped into fifteen days plus an optional math primer.
- **FR-CUR-002 — Implemented:** Curriculum ordering shall be controlled centrally.
- **FR-CUR-003 — Implemented:** Each entry shall expose title, track, level, duration, and status.
- **FR-CUR-004 — Implemented:** Lesson IDs shall use `<track>/<slug>`.
- **FR-CUR-005 — Required:** Published lesson IDs shall remain stable.
- **FR-CUR-006 — Required:** Prerequisite references shall resolve to valid lesson IDs.
- **FR-CUR-007 — Required:** Day and track assignments shall agree with the canonical curriculum.
- **FR-CUR-008 — Required:** Only lessons satisfying the lesson definition of done may use `status: done`.
- **FR-CUR-009 — Implemented:** Stub lessons shall be visibly identified as incomplete.
- **FR-CUR-010 — Required:** Draft and stub content shall not be represented as production-complete.
- **FR-CUR-011 — Required:** The sum of lesson durations for each core day should fit the promised two-to-three-hour workload or document the exception.

### 6.4 Lesson content model

Each lesson shall be an MDX entry with validated frontmatter.

| Field | Type | Requirement |
|---|---|---|
| `title` | string | Required |
| `summary` | string | Required |
| `day` | integer 1–15 or `null` | Required |
| `track` | defined track enum | Required |
| `order` | number | Required |
| `level` | `intro`, `core`, `advanced`, or `frontier` | Required |
| `duration` | positive number of minutes | Required |
| `prereqs` | lesson-ID array | Defaults to empty |
| `status` | `stub`, `draft`, or `done` | Defaults to `stub` |
| `lastReviewed` | date | Required for frontier-level lessons |

- **FR-DATA-001 — Implemented:** Frontmatter shall be validated during Astro content loading.
- **FR-DATA-002 — Required:** Duplicate lesson IDs shall fail validation.
- **FR-DATA-003 — Required:** Broken prerequisite references shall fail an automated curriculum check.
- **FR-DATA-004 — Required:** Invalid track/day combinations shall be reported.
- **FR-DATA-005 — Required:** Frontier lessons shall carry visible review dates.
- **FR-DATA-006 — Required:** Frontier lessons shall be re-reviewed at least every six months.
- **FR-DATA-007 — Required:** Lesson durations shall be checked against the course workload promise.

### 6.5 Lesson rendering

- **FR-LESSON-001 — Implemented:** Every lesson shall receive a statically generated route.
- **FR-LESSON-002 — Implemented:** The layout shall display title, duration, level, track/day context, and review date where applicable.
- **FR-LESSON-003 — Implemented:** The layout shall provide previous and next lesson navigation.
- **FR-LESSON-004 — Implemented:** Stub lessons shall display an incomplete-content banner.
- **FR-LESSON-005 — Implemented:** Step titles shall produce stable hash anchors.
- **FR-LESSON-006 — Implemented:** All steps shall remain on one scrolling page unless the owner explicitly changes this requirement.
- **FR-LESSON-007 — Required:** A lesson shall remain understandable when an interactive island fails to hydrate.
- **FR-LESSON-008 — Required:** Every lesson shall contain at least one meaningful interactive figure before it can become `done`.
- **FR-LESSON-009 — Required:** Each mechanism-explanation step shall show the relevant widget state.
- **FR-LESSON-010 — Implemented:** Equations shall be rendered at build time using KaTeX.
- **FR-LESSON-011 — Required:** Code examples shall match the described mechanism and visible values.
- **FR-LESSON-012 — Required:** Simplifications shall be labeled explicitly.
- **FR-LESSON-013 — Required:** Technical terms shall be defined before being relied upon.
- **FR-LESSON-014 — Required:** Lesson deep links shall use step-title hash anchors. The stale `?step=N` pagination checks shall be replaced with checks of stacked steps and anchors; they shall not redefine the owner's scrolling-page requirement.

### 6.6 Completed lesson anatomy

A completed lesson shall contain:

1. A concise refresher.
2. Prerequisite links where applicable.
3. A motivating hook.
4. Focused teaching steps, normally five to twelve.
5. Interactive states tied to mechanism explanations.
6. Collapsible mathematics near the relevant explanation.
7. Short matching code where useful.
8. At least three misconception corrections.
9. Three to five check-yourself questions with answers.
10. Labeled real-world limitations.
11. Primary papers and carefully selected further resources.
12. A clear next-lesson path.

### 6.7 Interactive widgets

- **FR-WIDGET-001 — Partial:** Widget simulation logic shall be separated from rendering and input handling.
- **FR-WIDGET-002 — Partial:** The standard widget structure shall be `<Name>.svelte`, `sim.ts`, and optional `worker.ts`.
- **FR-WIDGET-003 — Implemented for current completed pairs:** Shared figure-card and timeline components shall provide consistent presentation and controls.
- **FR-WIDGET-004 — Required:** Each time-driven widget shall derive visible state from continuous time `t`.
- **FR-WIDGET-005 — Required:** The timeline shall support play, pause, scrubbing, speed selection, and labeled phases.
- **FR-WIDGET-006 — Required:** A discrete Step control shall remain secondary to the continuous timeline.
- **FR-WIDGET-007 — Partial:** Widgets shall autoplay only while visible.
- **FR-WIDGET-008 — Partial:** Widgets shall pause while offscreen or while the document is hidden.
- **FR-WIDGET-009 — Partial:** Reduced-motion mode shall begin paused and remove nonessential movement.
- **FR-WIDGET-010 — Required:** Scrubbing and numerical inspection shall work in reduced-motion mode.
- **FR-WIDGET-011 — Required:** The drawing, labels, readouts, and narration shall derive from the same simulation frame.
- **FR-WIDGET-012 — Required:** All meaningful controls shall be keyboard operable.
- **FR-WIDGET-013 — Partial:** Each widget shall provide an accessible name and polite live narration of meaningful phase changes.
- **FR-WIDGET-014 — Required:** Color shall not be the only way to communicate state.
- **FR-WIDGET-015 — Required:** Visual domains shall fit their content without clipping, excessive empty space, or overlapping labels.
- **FR-WIDGET-016 — Required:** Canvas rendering shall account for device pixel ratio and cap it when necessary.
- **FR-WIDGET-017 — Required:** Expensive computation shall move off the main thread when profiling shows it could violate interaction or frame-rate budgets.
- **FR-WIDGET-018 — Required:** Stochastic widgets shall use a deterministic seeded random-number generator.
- **FR-WIDGET-019 — Required:** Every complete `sim.ts` shall have unit tests for mathematics, determinism, and important timeline endpoints.
- **FR-WIDGET-020 — Required:** A widget shall not hard-code semantic colors that already exist as shared tokens.
- **FR-WIDGET-021 — Required:** Widget props shall allow lesson steps to configure the relevant state without duplicating the component.
- **FR-WIDGET-022 — Required:** Each core figure shall visibly connect a learner change to the mechanism and its consequence, with a concrete interaction to try and inspectable intermediate values where needed.
- **FR-WIDGET-023 — Required:** Defaults, plot context, axis scale and timeline labels shall make the lesson's claim visible. Effects across orders of magnitude shall use a suitable scale; advanced statistics shall follow the worked explanation.
- **FR-WIDGET-024 — Required:** Core steps shall configure distinct teaching states. Misconceptions, check-yourself and going-deeper sections shall not repeat the widget, following `LESSONS.md` and `BUILDING.md` §5a.

### 6.8 Shared neural-network engine

- **FR-NN-001 — Implemented:** The engine shall support scalar values and inspectable computational graphs.
- **FR-NN-002 — Implemented:** It shall support addition, subtraction, multiplication, division, negation, powers, exponentials, logarithms, tanh, ReLU, and sigmoid.
- **FR-NN-003 — Implemented:** It shall expose topological ordering and stepwise local backpropagation.
- **FR-NN-004 — Implemented:** It shall support neurons, layers, and multilayer perceptrons.
- **FR-NN-005 — Implemented:** It shall support mean-squared error and binary cross-entropy.
- **FR-NN-006 — Implemented:** It shall provide SGD, momentum, RMSProp, and Adam.
- **FR-NN-007 — Required:** Analytic gradients shall be verified against numerical gradients.
- **FR-NN-008 — Required:** Widgets using equivalent mathematical operations shall reuse the shared engine where practical.
- **FR-NN-009 — Required:** Simulation code shall avoid hidden nondeterministic state.

### 6.9 Themes and visual language

- **FR-THEME-001 — Implemented:** The default theme shall follow the operating-system preference.
- **FR-THEME-002 — Implemented:** Users shall be able to select light or dark mode.
- **FR-THEME-003 — Implemented:** The preference shall be stored locally and fail safely if storage is unavailable.
- **FR-THEME-004 — Required:** Widgets shall use semantic design tokens.
- **FR-THEME-005 — Required:** Forward values, gradients, signed values, sequential intensity, and focus shall retain consistent semantic colors.
- **FR-THEME-006 — Required:** Widgets shall respond correctly when the theme changes after hydration.
- **FR-THEME-007 — Required:** Decorative gradients, glassmorphism, glowing effects, generic AI imagery, emoji-as-interface, and hype-oriented visual conventions shall not be introduced.
- **FR-THEME-008 — Required:** Typography shall use the configured serif body, sans-serif interface, and monospace numerical/code roles.

### 6.10 Presentation and print

- **FR-PRESENT-001 — Partial:** A `?present` query parameter shall activate presentation styling.
- **FR-PRESENT-002 — Implemented:** Presentation mode shall hide normal site chrome and enlarge teaching content.
- **FR-PRESENT-003 — Required:** Presentation controls and keyboard behavior shall be defined and tested.
- **FR-PRINT-001 — Partial:** Print mode shall hide navigation and theme controls.
- **FR-PRINT-002 — Planned:** Widgets shall provide meaningful static print representations or snapshots.
- **FR-PRINT-003 — Required:** Printed lessons shall not contain clipped text, unusable controls, or unexplained empty widget regions.

### 6.11 Progress tracking

- **FR-PROGRESS-001 — Planned:** Learners shall be able to mark lessons complete.
- **FR-PROGRESS-002 — Planned:** Completion shall be stored only in browser-local storage.
- **FR-PROGRESS-003 — Required:** The site shall remain usable if local storage is blocked or cleared.
- **FR-PROGRESS-004 — Required:** Progress shall not be sent to a server.
- **FR-PROGRESS-005 — Required:** The Path page shall identify completed lessons without relying on color alone.
- **FR-PROGRESS-006 — Open:** Reset, export, and import behavior requires an owner decision.

### 6.12 Search

- **FR-SEARCH-001 — Planned:** The system shall provide full-text search across published lessons.
- **FR-SEARCH-002 — Planned:** Search shall operate without an application server.
- **FR-SEARCH-003 — Required:** Results shall include title, summary, track/day context, and matching text.
- **FR-SEARCH-004 — Required:** Search shall support keyboard navigation and screen-reader labels.
- **FR-SEARCH-005 — Required:** Stub content shall be excluded or clearly labeled.
- **FR-SEARCH-006 — Open:** Pagefind remains the intended implementation but was not installed in the inspected baseline.

### 6.13 Refresher index

- **FR-REFRESH-001 — Planned:** The Refresher page shall aggregate completed lesson refreshers.
- **FR-REFRESH-002 — Required:** Entries shall link to their corresponding lessons.
- **FR-REFRESH-003 — Required:** Entries shall expose summary, key equations, prerequisites, and review dates where applicable.
- **FR-REFRESH-004 — Required:** Stub refreshers shall be excluded or visibly labeled.

### 6.14 Glossary

- **FR-GLOSS-001 — Planned:** The Glossary page shall list canonical technical terms.
- **FR-GLOSS-002 — Required:** Every term shall include a plain-language definition.
- **FR-GLOSS-003 — Required:** Every term shall link to its canonical teaching lesson.
- **FR-GLOSS-004 — Planned:** Lesson content may provide glossary previews on hover or focus.
- **FR-GLOSS-005 — Required:** Hover behavior shall have equivalent keyboard and touch behavior.
- **FR-GLOSS-006 — Required:** Terms shall not have conflicting definitions across the site.

### 6.15 SEO and discovery

- **FR-SEO-001 — Implemented:** Every page shall have a title and meta description.
- **FR-SEO-002 — Implemented:** Public pages shall receive canonical URLs.
- **FR-SEO-003 — Implemented:** Open Graph and Twitter summary metadata shall be emitted.
- **FR-SEO-004 — Implemented:** The site shall provide a custom social preview image.
- **FR-SEO-005 — Implemented:** WebSite, Course, and LearningResource structured data shall be emitted where applicable.
- **FR-SEO-006 — Implemented:** Sitemap generation shall be enabled.
- **FR-SEO-007 — Implemented:** `robots.txt` shall permit normal search and named AI crawlers.
- **FR-SEO-008 — Implemented:** `llms.txt` shall enumerate the curriculum.
- **FR-SEO-009 — Required:** Production canonical URLs shall use the owner-approved domain.
- **FR-SEO-010 — Required:** Draft/stub publication and indexing policy shall be explicitly defined.
- **FR-SEO-011 — Required:** Every meaningful image shall have appropriate alternative text.

---

## 7. Curriculum requirements

### 7.1 Math primer

The optional primer shall cover:

- Vectors.
- Dot products.
- Matrices.
- Derivatives.
- The chain rule.
- Probability basics.

### 7.2 Fifteen-day core path

| Day | Theme | Required subject coverage |
|---:|---|---|
| 1 | What is learning? | Models, data, loss, linear regression, gradient descent |
| 2 | Classification and generalization | Logistic regression, softmax, boundaries, overfitting, splits, regularization |
| 3 | Classic machine learning | k-NN, trees, ensembles, SVMs, k-means, PCA, model-family selection |
| 4 | Neural networks | Neurons, activations, MLPs, hidden representations, universal approximation |
| 5 | Backpropagation | Computational graphs, chain rule, forward/backward passes, MLP backprop, autodiff, gradient failure |
| 6 | Training deep networks | Optimizers, schedules, initialization, normalization, regularization, landscapes |
| 7 | Convolutional networks | Convolution, channels, stride, padding, receptive fields, residual networks, feature visualization |
| 8 | Sequences | Memory, RNN cells, unrolling, BPTT, vanishing gradients, LSTM/GRU, seq2seq attention |
| 9 | Transformers | Tokenization, embeddings, self-attention, heads, positional encoding, blocks, masking |
| 10 | Large language models | Pretraining, sampling, scaling laws, KV cache, in-context learning, hallucination |
| 11 | Reinforcement learning | RL basics, Q-learning, policy gradients, deep RL, PPO |
| 12 | Alignment | SFT, RLHF, DPO, constitutional methods, RLVR, evaluation |
| 13 | Graphs and generative models | GNNs, autoencoders, VAEs, GANs, diffusion, flow matching |
| 14 | Applied AI | Vector search, RAG, prompting, tools, agents, multimodality, LoRA/QLoRA |
| 15 | Frontier | Reasoning, MoE, state-space models, long context, efficient inference, interpretability, safety, current developments |

### 7.3 Frontier content controls

- Frontier claims shall be researched from primary sources.
- Sources shall be linked and dated.
- The review date shall be visible.
- Claims shall not be written from model memory alone.
- A review older than six months shall be treated as stale until checked.
- Unsettled findings shall be described with appropriate uncertainty.

---

## 8. External interfaces

### 8.1 User interface

The user interface shall support:

- Desktop and mobile browsers.
- Mouse, touch, and keyboard input.
- System-selected and explicit color themes.
- Reduced-motion settings.
- Projector-friendly presentation.
- Printable lesson output.

### 8.2 Browser interfaces

The system may use:

- `localStorage` for theme and progress.
- `requestAnimationFrame` for animation.
- `IntersectionObserver` for visibility control.
- The Page Visibility API for background pausing.
- Web Workers for expensive computation.
- Canvas 2D, SVG, WebGL2, or WebGPU according to measured need.

Failure of an optional browser API shall degrade safely rather than making lesson prose unavailable.

### 8.3 Network interface

- Normal operation shall require only static asset delivery.
- No application API shall be required.
- Third-party runtime requests should be avoided.
- Fonts shall be self-hosted.
- External research links may navigate away from the site but shall not be required for core lesson rendering.

---

## 9. Nonfunctional requirements

### 9.1 Performance

These thresholds are release gates, not claims about the current build.

| Metric | Required threshold |
|---|---:|
| Mobile Lighthouse performance | At least 95 |
| Largest Contentful Paint | Under 1.2 seconds |
| Cumulative Layout Shift | 0 |
| Interaction to Next Paint | Under 100 ms |
| Animation rate | Approximately 60 FPS on the reference mid-range laptop |
| Pre-widget lesson JavaScript | Under 15 KB gzip |
| Individual widget island | Under 40 KB gzip target |

The budget script measures the static JavaScript dependency closure plus the largest lazily loaded mode for widgets that split modes. Worker assets are excluded by that script and require separate accounting.

The owner approved model-weight binaries under `public/models/` as data outside the 40 KB gzipped JavaScript budget. They shall load lazily when requested by a visible widget and stay compact (float16, int8 for files above 40 KB). Current source implements a lazy loader and module-local promise memoization; persistent cross-page caching is not implemented.

Additional requirements:

- Widgets shall reserve display space before hydration.
- Lesson widgets shall normally hydrate with `client:visible`.
- Animation loops shall stop while offscreen or hidden.
- Main-thread work shall remain bounded. Per-step browser checks shall sample hydration/autoplay stalls and reject animation-frame gaps above 100 ms; first-step FPS alone shall not establish responsiveness across a lesson. This check shall supplement, rather than replace, release INP measurement.
- DOM reads and writes shall be batched to avoid layout thrashing.
- Dense or rapidly changing visuals should use Canvas rather than excessive SVG nodes.
- Typed arrays should be used where they provide a measured benefit.
- Production source maps shall not be publicly deployed unless explicitly approved.
- Performance claims shall be supported by captured measurements.

### 9.2 Accessibility

The target is WCAG 2.2 AA.

The system shall provide:

- Keyboard access to every meaningful control.
- Visible focus indicators.
- Accessible names for figures and controls.
- Text narration of meaningful visual-state changes.
- Reduced-motion behavior.
- Sufficient light and dark contrast.
- Non-color indicators for state.
- Semantic headings and landmarks.
- Mobile and zoom usability.
- No keyboard traps.
- Announcement frequency that does not overwhelm screen-reader users.
- Alternative explanations for information otherwise available only through animation.

### 9.3 Privacy

- No account shall be required.
- No learner progress shall leave the browser during normal operation.
- Local preferences shall contain no sensitive personal data.
- Analytics shall remain disabled until the owner documents a decision.
- If analytics are added, the privacy statement and this SRS shall be updated before deployment.

### 9.4 Security

- MDX shall be trusted repository content rather than user-submitted markup.
- Dependencies shall be minimized and maintained.
- Secrets shall not be embedded in client code.
- External links shall be reviewed.
- Future user-controlled HTML or query input shall be escaped or validated.
- Production builds shall not expose environment secrets.
- Content security policy requirements shall be considered before launch.

### 9.5 Reliability and failure behavior

- Static prose shall remain readable if widget JavaScript fails.
- Missing local storage shall not break navigation or lessons.
- Invalid lesson metadata shall fail during validation.
- Widget inputs shall reject or constrain invalid values.
- Mathematical singularities and non-finite values shall be handled explicitly.
- A failed worker shall produce a recoverable state or explanatory message.
- Broken internal links shall fail release validation.
- Console errors shall fail browser acceptance checks.

### 9.6 Maintainability

- TypeScript strict mode shall remain enabled.
- Svelte components shall use Svelte 5 conventions.
- Simulation logic shall not depend on the DOM.
- Shared behavior shall be abstracted only after demonstrated reuse.
- Widget-specific helpers shall remain in their widget directory unless genuinely shared.
- Colors, spacing, typography, and semantic visualization roles shall use shared tokens.
- New dependencies shall justify themselves against a small native implementation.
- `any` shall not be introduced without documented necessity.
- Contributor documentation shall be updated when an architectural decision changes.

### 9.7 Reproducibility

- Randomized simulations shall accept deterministic seeds.
- Reloading the same lesson configuration shall reproduce the same initial data.
- Visible values, narration, and equations shall derive from the same state.
- Tests shall use stable seeds.
- Deep-linked states shall not depend on previous browsing history.

### 9.8 Compatibility

The exact supported-browser policy remains open. The launch baseline should include current stable versions of:

- Chromium-based browsers.
- Firefox.
- Safari desktop.
- Safari on iOS.

Advanced rendering APIs shall have a fallback when their absence would otherwise remove essential teaching content.

### 9.9 Content quality

- Explanations shall be technically reviewed.
- Equations, signs, dimensions, and numerical examples shall be checked.
- Numbers stated in prose shall match the associated widget state.
- Simplifications shall state how real systems differ.
- Primary papers shall be preferred for historical and frontier claims.
- Prose shall avoid hype language and undefined jargon.
- Lessons shall use a clear, precise, and warm tone.
- Each core step shall begin with intuition, define new terms, and give one concrete thing to look at or try. Control manuals and repeated caveats shall not replace teaching.
- Claims about a demonstration shall be checked against computed simulation values before they are written. Defaults shall agree with the prose.
- Coordinator review shall assess whether a newcomer can explain what changed, how the mechanism caused it, and why the result matters. Technical checks alone shall not establish this acceptance.

---

## 10. Verification and testing

### 10.1 Required automated checks

A stable validation workflow shall include:

1. Astro type and content validation.
2. Vitest unit tests.
3. Svelte compilation with zero warnings.
4. Curriculum-integrity validation.
5. Internal-link validation.
6. Browser checks in light and dark modes.
7. Mobile-width checks.
8. Reduced-motion checks.
9. Console-error detection.
10. Horizontal-overflow detection.
11. Keyboard and automated accessibility checks.
12. Screenshot review.
13. Performance and bundle-size measurement.
14. Production static build.

### 10.2 Lesson definition of done

A lesson may become `done` only when:

- Its frontmatter is valid.
- Its status is intentionally changed to `done`.
- It contains the required lesson anatomy.
- Its prerequisite and internal links resolve.
- Its technical content is reviewed.
- Its simplifications and limitations are identified.
- Its citations are appropriate.
- Its widget mathematics pass tests.
- Its figures work with keyboard input.
- Light, dark, mobile, and reduced-motion checks pass.
- It produces no browser console errors.
- It introduces no horizontal overflow.
- It satisfies applicable performance budgets.
- Screenshots have been visually reviewed rather than merely generated.
- A reviewer has checked newcomer clarity using `BUILDING.md` §5a, including intermediate states, distinct figure configurations, defaults, axes, and a visible mechanism and consequence.

### 10.3 Widget definition of done

A widget may be considered complete only when:

- Rendering and simulation logic are separated.
- Simulation results are deterministic.
- Mathematical behavior has unit tests.
- Important timeline endpoints and phase transitions have tests.
- Controls are labeled and keyboard accessible.
- Visible values and narration agree.
- Reduced-motion behavior is correct.
- Offscreen and hidden-tab pausing works.
- Light and dark themes are verified.
- Mobile layouts do not clip or overlap.
- Its bundle size and frame-rate measurements meet the budget.
- Svelte compiles it without warnings.
- Its worked interaction teaches the claimed mechanism before exposing advanced statistics, and displayed defaults/readouts agree with the lesson.

### 10.4 Page acceptance criteria

Every public page shall:

- Return the intended successful or error status.
- Have exactly one `<h1>`.
- Have a unique title and description.
- Have no unexpected console errors.
- Have no horizontal overflow at required viewport sizes.
- Remain usable with keyboard input.
- Remain readable in light and dark themes.
- Respect reduced-motion preferences.
- Contain static HTML for its essential information.

### 10.5 Release acceptance criteria

A public release shall require:

- A clean, reviewable Git baseline.
- Passing unit, type, content, and static-build checks.
- Passing required browser and accessibility checks.
- Measured performance results meeting the documented budgets or explicitly approved exceptions.
- An approved name and production domain.
- Resolved licensing and analytics decisions.
- No accidental exposure of secrets or development artifacts.
- Correct sitemap, robots, canonical, social, and structured metadata.
- A documented policy for draft and stub visibility.

---

## 11. Traceability matrix

| Requirement area | Primary implementation evidence | Primary validation |
|---|---|---|
| Content schema | `src/content.config.ts` | Astro content/type check |
| Curriculum ordering | `src/lib/curriculum.ts` | Curriculum integrity check |
| Lesson routes | `src/pages/learn/[...slug].astro` | Static build and browser checks |
| Shared lesson layout | `src/layouts/Lesson.astro` | Page acceptance checks |
| Global metadata and shell | `src/layouts/Base.astro` | SEO and page checks |
| Theme | `src/components/ui/ThemeToggle.astro`, `src/styles/tokens.css` | Theme persistence and visual checks |
| Lesson steps | `src/components/ui/Step.astro` | Anchor and structure checks |
| Shared widget presentation | `src/components/widgets/_shared/` | Widget/browser checks |
| Deterministic RNG | `src/lib/math/rng.ts` | Unit tests |
| Autograd and optimizers | `src/lib/nn/` | Numerical-gradient and optimizer tests |
| CNN shared operations | `src/lib/cnn/conv.ts` | Convolution and pooling gradient tests |
| Transformer core and serialized model assets | `src/lib/transformer/`, `public/models/` | Model-core, serialization, loader, and inference tests |
| Day 10 LM helpers and model training | `src/lib/lm/`, `scripts/train-models.ts`, `plans/day-10.md` | LM helper tests, deterministic training reports, and model-manifest checks |
| Lesson content | `src/content/lessons/` | Content review and link checks |
| SEO helpers | `src/lib/seo.ts` | Metadata and structured-data checks |
| Robots and AI discovery | `src/pages/robots.txt.ts`, `src/pages/llms.txt.ts` | Build-output checks |
| Visual/runtime inspection | `scripts/check-page.mjs`, `scripts/check-curriculum.mjs` | Playwright execution and screenshot review |
| Unit tests | `tests/unit/` | Vitest |
| Day 4–6 interaction checks | `scripts/verify-day4.mjs` (supports `--day 5` and `--day 6`) | Multi-time, theme, keyboard, reduced-motion and narrow-layout checks |
| Day 6 teaching and controls | `tests/unit/day6-teaching.test.ts`, `scripts/verify-day6-teaching.mjs`, `scripts/verify-day6-controls.mjs`, `scripts/verify-day6-layout.mjs` | Worked-explanation regressions and browser checks; regenerate evidence after later edits |
| Widget dependency budgets | `scripts/check-widget-budgets.mjs` | Gzip size of widget chunks and shared JavaScript dependency closures after a build |
| Newcomer clarity and day acceptance | `LESSONS.md`, `BUILDING.md` §§1, 5, 5a | Coordinator prose/screenshot review and owner feedback |

---

## 12. Delivery plan

Owner-directed sequence (from `BUILDING.md`):

1. Reconcile concurrent work for a reviewable baseline; the owner performs commits and pushes.
2. Replace stale paginated curriculum checks with checks of the required stacked lessons and anchors.
3. Update stale project-status documentation.
4. Continue one day at a time from Day 11 through Day 15 after the recorded Day 10 coordinator acceptance; finish and review each day before beginning the next.
5. Research Day 15 frontier content from current primary sources and keep its review dates current.
6. Implement progress, search, glossary, and refresher aggregation.
7. Complete presentation and print behavior.
8. Add CI, accessibility, browser, and performance gates.
9. Select the final name, domain, licenses, and analytics policy.
10. Perform a launch audit and deploy.

### 12.1 Milestones

| Milestone | Scope | Completion condition |
|---|---|---|
| M0 — Foundations | Product docs, repository, Astro/Svelte scaffold, basic tooling | Empty/static shell validates and builds |
| M1 — Shell and design system | Layouts, tokens, theme, curriculum map, lesson routing | Whole path is navigable and shell checks pass |
| M2 — Flagship lessons | Gradient descent, backpropagation, RNNs, GNNs, shared math engine | Flagship lessons satisfy the full definition of done |
| M3 — Week 1 | Days 1–7 | All Week 1 lessons are done |
| M4 — Week 2 | Days 8–14 | All non-frontier Week 2 lessons are done |
| M5 — Frontier | Day 15 based on primary sources | All frontier lessons are done and freshly reviewed |
| M6 — Polish and launch | Search, glossary, refreshers, progress, presentation, print, audits, deployment | Release acceptance criteria pass |

Milestone labels group scope; they do not authorize work out of curriculum order. M2 remains partial because GNN flagship content is not accepted. M3 has authored work through Day 7; M4 has coordinator-accepted work through Day 10, with Days 11–14 remaining. All entries remain `draft` or `stub` in frontmatter, so no lesson is marked `done`. Owner clarity review remains pending for Days 6–10. `CLAUDE.md`'s M0–M1 completion is a scaffold checkpoint, not evidence that CI, deployment or release gates have passed.

---

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Concurrent agents edit overlapping files | Lost or conflicting work | Assign bounded ownership; inspect diffs before integration |
| Planning docs become stale | Agents implement obsolete assumptions | Keep this SRS and status sections synchronized |
| Draft lessons are mistaken for finished content | Public misinformation or poor experience | Enforce status semantics and completion checks |
| Widget values disagree with narration | Incorrect teaching | Use one frame as the source of truth and test it |
| Heavy animation blocks the main thread | Poor interaction and accessibility | Profile, pause offscreen, and use workers where needed |
| Curriculum exceeds promised workload | Misleading product promise | Audit per-day duration and revise estimates or scope |
| Frontier content becomes outdated | Incorrect current claims | Require primary sources and six-month reviews |
| Static prose depends on hydration | Content disappears on JS failure | Render core explanation in HTML and treat widgets as enhancement |
| No stable Git baseline | Difficult review and recovery | Establish commits before large-scale parallel development |
| Placeholder domain enters production metadata | Broken canonical and sitemap URLs | Gate release on domain replacement |
| Performance targets remain unmeasured | Aspirational claims presented as facts | Require captured Lighthouse, bundle, and frame-rate evidence |
| Technical checks pass but teaching is confusing | Learners cannot explain the mechanism | Review prose and worked interactions with a newcomer perspective before acceptance |
| Saved reports predate current source | Old passes conceal regressions | Attribute reports to checkpoints and rerun checks after substantive edits |

---

## 14. Open decisions

The owner must decide:

1. Final product name and domain.
2. Source-code and lesson-content licenses.
3. Whether any analytics are permitted.
4. Exact supported-browser matrix.
5. Whether external lesson contributions are accepted.
6. Whether progress export/import is needed.
7. Whether stub pages should be publicly indexed.
8. Whether to provide optional query aliases for existing hash anchors; stacked scrolling lessons are already required and pagination is not an open choice.
9. Whether `/math` should be a dedicated index or redirect.
10. Whether all simulations must use workers or only measured heavy workloads.
11. How interactive figures will be represented in print.
12. What reference hardware defines the frame-rate target.

---

## 15. Glossary

| Term | Meaning in this project |
|---|---|
| Content collection | Astro-managed set of lesson entries with validated metadata |
| Draft | Substantial lesson content under development but not accepted as complete |
| Frontier lesson | Time-sensitive advanced content requiring a review date and primary-source research |
| Hydration | Activation of a static widget’s client-side Svelte behavior |
| Island | A localized interactive component embedded in otherwise static HTML |
| Lesson ID | Stable `<track>/<slug>` identifier used for routing and prerequisites |
| Refresher | Compact summary containing the central idea, equations, prerequisites, and references |
| Simulation frame | Complete visible and narrated widget state derived for a particular time and configuration |
| Stub | Metadata and outline placeholder without complete teaching content |
| Widget | Interactive explanatory figure backed by deterministic simulation logic |

---

## 16. Repository references

- [`IDEA.md`](./IDEA.md) — product vision, curriculum, design direction, and original roadmap.
- [`CLAUDE.md`](./CLAUDE.md) — contributor and engineering guidance.
- [`LESSONS.md`](./LESSONS.md) — lesson and widget authoring guide.
- [`BUILDING.md`](./BUILDING.md) — current day status, owner-directed order, coordinator review, clarity criteria, and recorded wave evidence.
- [`package.json`](./package.json) — scripts, runtime requirements, and installed dependencies.
- [`src/content.config.ts`](./src/content.config.ts) — lesson schema and track identifiers.
- [`src/lib/curriculum.ts`](./src/lib/curriculum.ts) — day, track, and lesson ordering.
- [`src/layouts/Base.astro`](./src/layouts/Base.astro) — global page shell and metadata.
- [`src/layouts/Lesson.astro`](./src/layouts/Lesson.astro) — lesson presentation and navigation.
- [`src/styles/tokens.css`](./src/styles/tokens.css) — shared visual and semantic tokens.
- [`src/components/widgets/_shared/README.md`](./src/components/widgets/_shared/README.md) — shared widget infrastructure.
- [`scripts/check-page.mjs`](./scripts/check-page.mjs) — per-page browser inspection.
- [`scripts/check-curriculum.mjs`](./scripts/check-curriculum.mjs) — curriculum browser inspection.
- [`scripts/verify-day4.mjs`](./scripts/verify-day4.mjs) — Day 4–6 multi-time browser verification.
- [`scripts/verify-day6-teaching.mjs`](./scripts/verify-day6-teaching.mjs) — Day 6 teaching layouts and timeline interactions.
- [`scripts/verify-day6-controls.mjs`](./scripts/verify-day6-controls.mjs) — Day 6 parameter and expanded-detail checks.
- [`scripts/verify-day6-layout.mjs`](./scripts/verify-day6-layout.mjs) — additional Day 6 viewport coverage.
- [`scripts/check-widget-budgets.mjs`](./scripts/check-widget-budgets.mjs) — widget gzip dependency-closure measurement.

---

## 17. Change-control rule

This SRS shall evolve with deliberate product decisions. A contributor changing a requirement, route contract, content schema, performance budget, lesson anatomy, or definition of done shall update this document in the same reviewed change or explicitly record why the update is deferred.

Observed implementation drift shall not be resolved by silently rewriting requirements. The owner or designated maintainer shall decide whether the implementation or the requirement is authoritative.

### 17.1 Revision history

| Version / date | Scope | Verification boundary |
|---|---|---|
| 0.1 / 2026-09-30 | Initial repository baseline; later Day 4–6 checkpoint additions | Source audit and separately recorded development checks |
| 0.2 / 2026-10-02 | Refreshed inventory, curriculum maturity and workload; Day 6 clarity requirements; checkpoint provenance; owner-directed delivery order; traceability | Source counts and saved-report inspection only; no application checks rerun |
| 0.3 / 2026-10-04 | Refreshed inventory and curriculum through Day 10; corrected 15-day schedule, historical checkpoint labels, lazy model-weight behavior, and current delivery order | Source inventory and selected implementation/docs inspection only; no tests, build, server, or app checks run |
| 0.4 / 2026-10-05 | Refreshed inventory after all six Day 10 lessons were drafted; recorded coordinator acceptance and pending owner clarity review; documented experiment boundaries, per-step stall checks, and Day 11 continuation | Source counts, selected implementation/docs inspection, and document integrity checks only; recorded wave results not rerun; no tests, build, server, or app checks run |
