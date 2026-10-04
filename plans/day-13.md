# Day 13 plan: graphs and generative models

Lessons in order: `graphs-generative/{gnn, autoencoders, vaes, gans, diffusion, flow-matching}`. The skeletons in each `.mdx` hold the per-step brief (idea, widget props, motion, the one thing to try, the numbers to quote). This file holds the cross-lesson decisions.

## 0. Audit

| Widget | State on disk | Verdict |
|---|---|---|
| `GNNMessagePassing` | `.svelte` 22.5 KB, `sim.ts`, `tests/unit/gnn-mp.test.ts`, used by `gnn.mdx` 10 times. Timeline + `createLoop` + autoplay in view + reduced-motion start paused; seeded `createRng`; tokens only. | **Reuse.** Fix list is in the review checklist below. |
| `DiffusionDenoiser` | `.svelte` 11.5 KB + `sim.ts` + `tests/unit/diffusion-denoiser.test.ts` (score vs numerical gradient, closed-form endpoints, ODE/SDE determinism). No lesson uses it. Honest: exact closed-form score of a Gaussian mixture under the VP schedule of Song et al. 2021, so there is no fake "network". | **Reuse with fixes:** (1) the `data` narration hard-codes "600 samples ... spiral" whatever the `n`/`dataset`; (2) in `mode="forward"/"reverse"` the Timeline still shows all markers over the full `T` while playback loops only `[lo, hi]`; Step jumps to 0 outside the loop; (3) `const u = phase ... ? data.u : data.u` is dead code; (4) the cursor readout is pointer-only, so add a keyboard or focus fallback (e.g. a fixed probe point when focused); (5) points are coloured by "position along the curve" on a `--viz-seq` ramp, which is fine, but the legend says "data position along the curve" while the reverse phase colours by the *source* point's u (wrong after re-sampling), so colour by final position or drop the ramp; (6) move `mixture`/`logp`/`score` into the shared core (below) and re-export from `sim.ts` so existing tests keep passing; (7) add modes `closed`, `field`, `train`, `image`, `guide` and prop `denoiser: 'exact' \| 'learned'`, default dataset `moons`. |
| `LatentSpace`, `GANDuel`, `FlowField`, `GenCommon` | **Empty folders** (no files, no tests, no lesson imports). | **Build** `LatentSpace`, `GANDuel`, `FlowField`. **Delete** `GenCommon`: the shared code goes to `src/lib/gen/`, following the `src/lib/cnn/` precedent (coordinator removes the empty folder). |

### gnn.mdx review checklist (do not rewrite; a reviewer checks and fixes)
1. **Probe every number** against `GNNMessagePassing/sim.ts` (seed 3): GAT into C = 0.55 / 0.14 / 0.17 / 0.15; spread 0.53 → 0.03 (gcn, 6 layers), 0.03 (sum), 0.06 (max); GCN D: c_DD = 0.5, c_DC = 0.35; hop distances D→C 1, A/B 2, E 3, F/G 4 (Check-yourself Q2 path D, C, B, E, F).
2. **Meta-notes LESSONS.md forbids:** "The unit tests compare it against the dense matrix formula" (GCN step) and "which the unit tests check" (GAT math block). Delete them.
3. **Hedging paragraph** in "Limits": the sentence about the other aggregators also ending small reads like a disclaimer. Cut it to one sentence or move it to the math block.
4. **Colour words in prose:** "blue up for positive, orange down for negative" and "the same purple scale" have to match `--viz-pos` (blue), `--viz-neg` (vermillion) and `--viz-seq-*` in **both** themes. In dark mode the seq ramp is dark indigo, so check screenshots or reword to "the attention scale".
5. **Keyboard:** the node buttons bind ArrowLeft/Right to nudge, but `_shared/README.md` says "never bind ←/→". Confirm this doesn't collide with page or Timeline keys while a node has focus. Otherwise keep it, since focus-scoped is acceptable, and note it in BUILDING.
6. **IDEA 8.3 coverage:** editable graph, drag, k-hop halo, GCN/GAT toggle and step-through are all present. Message passing runs on a timeline, which is correct per the motion contract.
7. **"Show the mechanism, real numbers":** W and **a** are random, not trained. The prose says so once, which is acceptable. *Owner question:* add a trained node-classification mode? It is not required by 8.3.
8. **Distinct states:** the 10 instances have distinct props (verified by reading). Re-check that step 1 (`layers=2`, no panel) and "Readout" (`layers=2 select=3`) actually *look* different.
9. Standard gates: autofixer, 0 compiler warnings, `check-page --steps` and `--mobile`, look at mid-animation screenshots (message particles vs the readout phase), and the island closure under 40 KB. The `.svelte` is the largest in the repo.

## 1. Shared core: `src/lib/gen/`

Every generative lesson learns **the same data**: the two-moons Gaussian mixture already in `DiffusionDenoiser/sim.ts` (52 components, σ = 0.07), 600 training points, `sampleData(MOONS, 600, 7)`. Two separate moons give each model the same visible test. AE: a gap in the code line. VAE: a bridge of samples between the moons. GAN: possible collapse onto one moon. Diffusion and flow: both moons covered. Because the density is a known mixture, every lesson can draw the **exact** answer next to the learned one.

- `toy.ts` (no DOM)
  - `type Mixture`, `mixture('moons' | 'spiral')`, `MOONS`, `sampleData(m, n, seed) → { x: Float32Array, k: Uint16Array /*component*/, moon: Uint8Array }`.
  - `posteriorMean(m, a, b, x, y, out, only?: 'upper' | 'lower')`: E[x₀ | x = a·x₀ + b·ε] for the mixture, optionally restricted to one moon. Score, ideal ε̂, ideal denoiser, FM marginal velocity and guidance all derive from it: `score = (a·E − x)/b²`, `v*(x,t) = (E[x₁|x] − x)/(1 − t)` with a = t, b = 1 − t.
  - `logp(m, a, b, x, y)` for the score tests.
  - `quality(pts, m) → { onData, upper, lower }`: on-data = nearest component mean within 3σ (0.21). These are **the** comparison numbers quoted across lessons.
- `mlp.ts` (no DOM): typed-array MLP for small nets.
  - `initMlp(sizes, rng) → Float64Array`, `forward(p, sizes, X, B, cache)`, `backward(p, sizes, cache, dY, grad)`, `adam(p, g, st, lr, step)`.
  - `tanh`/`exp` built from + − × ÷ (copy the Seq2SeqAttention trick) so Chrome and Node train **bit-identically** and prose numbers match the tests. Sigmoid uses `exp`.
  - `timeFeatures(t, out)`: 4 Fourier features plus t, for the diffusion and flow nets.
- **Why not `src/lib/nn`:** scalar `Value` autograd is about 100× too slow for thousands of batched steps. `src/lib/nn` stays untouched.
- **Training:** each widget's `sim.ts` owns its trainer (`trainAE`, `trainVAE`, `trainGAN`, `trainEps`, `trainFM`). Each returns `{ snaps: Float64Array[] /*params at K checkpoints*/, steps: number[], losses: Float32Array }`. The timeline scrubs snapshots (real models, never blended), and each widget runs its `worker.ts` once per page, posting progress first and the result at the end (Seq2SeqAttention pattern). Budget: hidden 32 (GAN, AE, VAE) or 64 (eps-net, FM), batch 64, 1.5k–4k steps, about 1–4 s in the worker. Main-thread init stays under 50 ms. Show "training in your browser… 40%" in the figure until the result arrives.
- **Determinism:** `createRng(seed)` for init, batches and noise. Fixed seeds per run are documented in each `sim.ts`, and every test asserts that two runs give identical snapshots.
- **Tests:** `tests/unit/gen-core.test.ts` covers MLP gradients vs finite differences, Adam on a quadratic, `posteriorMean`-derived score vs a numerical gradient of `logp`, the FM v* identity at t → 1, and `quality` on exact samples ≥ 0.95.

## 2. Widgets

All widgets use the shared plane domain [-2.2, 2.2]² (moons fit with padding), data drawn neutral (`--text-muted`), model output in `--viz-forward`, arrows and gradients in `--viz-gradient`, signed maps in `--viz-pos/--viz-neg`. FigureCard + Timeline + `createLoop`, canvas 2D, and readouts from the same frame as the drawing. Each `frameAt(t, run, props)` is pure.

**LatentSpace** (`autoencoders`, `vaes`). The sim trains AE (latent 1), AE (latent 2), DAE (input noise 0.25), and VAE at β ∈ {0.1, 1, 4}: one worker, six small nets. Layout: data plane plus a horizontal latent strip (code histogram, prior curve, marker). Props: `model: 'ae'|'dae'|'vae'`, `mode`, `focus?`, `latentDim?: 1|2`, `beta?`, `showPrior?`. Risk: medium, about 30 KB closure. The sim trains only the models that props need, so the AE page skips the VAEs.

| Lesson / step | props | frameAt(t) |
|---|---|---|
| ae 1 | `ae tour` | points → codes → reconstructions, loop |
| ae 2 | `ae focus focus=k` | pulse enc → z → dec for one point |
| ae 3 | `ae train` | snapshot scrub + log loss cursor |
| ae 4 | `ae tour latentDim=2` | round trip that changes nothing |
| ae 5 | `ae explore` | z marker sweep, decoded point traces curve |
| ae 6 | `ae sample` | uniform codes decode one by one |
| ae 7 | `dae denoise` | noise ramps 0 → 0.25, arrows onto moons |
| vae 1 | `ae sample showPrior` | prior codes on the AE strip |
| vae 2 | `vae encode focus=k` | samples stream from one bell |
| vae 3 | `vae reparam focus=k` | ε sweep, μ + σε mapping |
| vae 4 | `vae train beta=1` | snapshots, recon and KL curves |
| vae 5 | `vae beta beta=4` (selector) | tween between 3 real models, readout snaps |
| vae 6 | `vae sample beta=1` | prior samples decode |
| vae 7 | `vae explore beta=1` | z sweep −2.5 … 2.5 |

**GANDuel** (`gans`). Runs `stable`, `unstable` (×4 lr), `collapse`, with snapshots every 25 steps. Layout: plane with the D(x) map (64×64 offscreen canvas, upscaled), a noise panel in `generator` mode, and loss / distance / moon-share mini charts. Props: `mode: 'duel'|'generator'|'discriminator'|'objective'|'gradients'|'collapse'`, `run`. Risk: medium, about 30 KB. Training reliability is the main risk (§5).

| step | props | frameAt(t) |
|---|---|---|
| 1 | `duel run=stable` | scrub whole run |
| 2 | `generator` | noise grid flows through G |
| 3 | `discriminator` | D-only training, map sharpens |
| 4 | `objective run=stable` | loss curves + log 4 line, cursor |
| 5 | `gradients` | arrows grow, then one exaggerated update |
| 6 | `duel run=unstable` (lr selector) | oscillating run + distance curve |
| 7 | `collapse run=collapse` | collapsing run + moon-share bars |

**DiffusionDenoiser** (`diffusion`). Existing, with the fixes from §0. The learned eps-net is 2+5 → 64 → 64 → 2, trained in a new `worker.ts` only when `denoiser="learned"` or `mode="train"`. The image mode uses 10 hand-coded 8×8 glyph bitmaps in `sim.ts` and the exact posterior mean over them, drawn as three 8×8 canvases (`x_t`, guess, noise). Guidance uses the exact conditional score via `posteriorMean(..., only)`. Risk: low to medium, the existing island is small and the worker chunk is separate.

| step | props | frameAt(t) |
|---|---|---|
| 1 | `both moons exact showControls=false` | dissolve → hold → gather |
| 2 | `forward moons showScore=false` | blur + share plot |
| 3 | `closed focus=k` | two head-to-tail arrows for one point |
| 4 | `field exact` | τ 1 → 0.02, field sharpens, hover probe |
| 5 | `train learned` | snapshots, learned vs exact arrows at τ 0.3 |
| 6 | `reverse learned sampler=ode` (toggle) | 600 particles with trails |
| 7 | `image` | τ 0 → 1 → 0, three image panels |
| 8 | `guide guidance=3` (label and w selectors) | guided reverse from fixed noise |

**FlowField** (`flow-matching`). Trains v_θ (2+5 → 64 → 64 → 2) in a worker. The exact v* comes from the core. Euler sampler `integrate(v, x0, steps)` records every step. The `compare` mode imports `reversePath` from `DiffusionDenoiser/sim.ts` (pure TS import, as ensembles reuses TreeSplitter). Props: `mode: 'flow'|'pairs'|'marginal'|'train'|'sample'|'compare'`, `steps?`. Risk: low, about 28 KB.

| step | props | frameAt(t) |
|---|---|---|
| 1 | `flow` | 600 particles, learned field, 64 steps |
| 2 | `pairs` | 40 pairs slide along straight lines |
| 3 | `marginal` | exact field + particles, faint pair lines |
| 4 | `train` | snapshots, arrows vs exact, loss + floor |
| 5 | `sample steps=64` | Euler trail of a highlighted particle |
| 6 | `sample steps=4` (selector 1–64) | jumps + quality-vs-steps chart |
| 7 | `compare` (budget 4 / 64) | diffusion vs flow paths in sync |

## 3. Reuse
- `_shared` FigureCard, Timeline, `createLoop`, `readColors`, `advance`. `createRng` from `src/lib/math/rng.ts`.
- The moons mixture, VP schedule, `reversePath` and `pathAt` from DiffusionDenoiser (moved or re-exported via `src/lib/gen/toy.ts`).
- The worker pattern and the deterministic exp/tanh idea come from Seq2SeqAttention (copied into `src/lib/gen/mlp.ts`, not imported from a widget).
- Day-wide comparison numbers come from `quality()`. The AE, VAE, GAN, diffusion and flow lessons each quote one, so readers can compare.

## 4. Build order
1. **Coordinator first:** `src/lib/gen/{toy,mlp}.ts` + `tests/unit/gen-core.test.ts`. Move the mixture code out of DiffusionDenoiser with a re-export so its tests still pass. Delete the empty `GenCommon`. Nothing else starts before this.
2. **In parallel:** (a) gnn review (independent); (b) LatentSpace + autoencoders; (c) GANDuel + gans; (d) DiffusionDenoiser fixes + diffusion.
3. **After (b):** vaes (adds `model="vae"` to LatentSpace; same agent as (b) if possible, to avoid two agents editing one widget).
4. **After (d):** flow-matching (FlowField imports DiffusionDenoiser's sim for `compare`).
5. Wave checks per BUILDING §6, plus `check-widget-budgets.mjs LatentSpace GANDuel DiffusionDenoiser FlowField GNNMessagePassing`.

## 5. Risks and open questions
- **GAN collapse on two moons may not happen honestly** with tiny nets. The fallback, if probing finds no collapse config, is a ring of 8 Gaussians used *only* in that step and labelled as a different dataset. **Owner: OK to break the "one dataset" rule there?**
- **GAN instability** may not show at a sane lr either. The prose follows the probe, not the textbook.
- **AE/VAE on 2D data with a 1-number code** is honest (the moons are curves) but less vivid than images. *Owner: is 8×8 images for AE/VAE wanted later?* The plan says no (YAGNI). Diffusion step 7 covers "image".
- **Diffusion image mode** uses an ideal denoiser over 10 glyphs, so it can only reproduce those glyphs. One sentence says why real models differ. *Owner: acceptable as the IDEA 8.6 "noise to image" slider?*
- **Class colours:** guidance needs "upper / lower moon" colours, and GAN needs real vs generated. There is no categorical token set (BUILDING §7 open question). The plan uses neutral for data, `--viz-forward` for model output, and pos/neg only for signed D(x). Guidance labels borrow `--viz-pos`/`--viz-neg` unless the owner adds categorical tokens.
- **In-browser training time** across four widgets: each runs once per page in a worker. The AE page trains 3 nets, the VAE page 4. If a run exceeds about 4 s on a mid laptop, cut steps and say so in the plan, not the prose.
- **GNN:** add a trained mode (node classification) or keep random weights? See checklist item 7.
