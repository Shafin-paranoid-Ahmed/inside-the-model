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

## 6. Core results (coordinator, 2026-10-11)

- `src/lib/gen/mlp.ts`: deterministic `exp`/`tanh`/`sigmoid`/`log`/`sin`/`cos`/`normal(rng)` (only + − × ÷ and sqrt, so Chrome and Node train bit-identically), `timeFeatures(t, out, o)` (5 features: t, sin/cos πt, sin/cos 2πt; `TIME_FEATURES = 5`), `paramCount`, `initMlp(sizes, rng)`, `createCache(sizes, B)`, `forward(p, sizes, X, B, cache)` (tanh hidden, linear output; returns `cache.acts[L]`), `backward(p, sizes, cache, dY, grad, B?)` (ADDS into grad, writes dLoss/dInput into `cache.dIn`, which GAN and VAE need), `createAdam(n, b1 = 0.9, b2 = 0.999)`, `adam(p, g, st, lr)`. Use `normal(rng)` from here, never `rng.normal()` (Math.log/cos), in anything that feeds training.
- `src/lib/gen/toy.ts`: `Mixture` now has `upper` (26 for the moons: components 0..25 are the upper moon), `mixture()`, `MOONS`, `DOMAIN = 2.2`, `sampleData(m, n, seed) → { x, k, moon }` (moon 0 upper, 1 lower), `posteriorMean(m, a, b, x, y, out, only?)`, `score(m, a, b, x, y, out, only?)`, `logp(m, a, b, x, y)`, `quality(pts, m) → { onData, upper, lower }` (upper/lower are shares of the on-data points).
- `DiffusionDenoiser/sim.ts` keeps its VP-parameterised `logp(m, tau, …)`/`score(m, tau, …)` and `sampleData` (adds `u`) as thin wrappers over the core. `GenCommon` deleted.
- Measured training cost in Node (vitest, this machine), batch 64, forward + backward + Adam: **7→64→64→2 ≈ 2.2 ms/step** (4k steps ≈ 9 s, so the plan's 4k steps is too many: use ≤ 1.8k steps or hidden 48), **2→32→32→1 ≈ 0.55 ms/step**. Size each trainer so its worker finishes in about 4 s.
- Tests: `tests/unit/gen-core.test.ts` (13).

## 8. Diffusion learned-sampler probe (diffusion agent, 2026-10-12)

`quality().onData` of learned ODE samples from the same fixed noise, 1,800 updates (3.3–3.4 s training each). b² = 1 − ᾱ.

| Loss weighting | Plain sampler | Stop at τ 0.1 (x̂0 step) | 0.05 (x̂0) | 0.02 (x̂0) |
|---|---|---|---|---|
| uniform (shipped) | 0.667 | 0.370 (0.655) | 0.558 (0.677) | 0.628 (0.678) |
| 0.1/(b²+0.05) | 0.672 | | | |
| 0.3/(b²+0.2) | 0.720 | | | 0.715 (0.725) |
| (b²+0.01)^-½, capped | 0.693 | | | |

Training longer helps (4k steps 0.67, 8k steps 0.77) but breaks the ~4 s budget. The shortfall is mid-τ routing (x̂0 at τ 0.1 is already only 65% on a moon), not the last small-τ steps. Kept: uniform loss, 0.667.

## 7. GAN probe results

Trainer: `GANDuel/sim.ts` `trainGAN`. G 2-32-32-2, D 2-32-32-1 (sigmoid), batch 64, Adam beta1 0.5, D step then non-saturating G step, G output layer scaled by 0.1 at init (`gInit`; without it the random G already overlapped the moons at step 0 with 53% on-data, which hid the story). 1500 steps, checkpoint every 25 (61 snaps), about 3 s per run in Node. `onData` = `quality()` on 300 fixed noise points. Seeds are the `createRng` seed for init, batches and noise.

Findings while probing (2000 steps, before `gInit`): equal rates (0.0005 to 0.002) give a noisy generator that hovers at 20-70% on data with D at chance. A fast detective and slow forger is what converges: lrD 0.005, lrG 0.0003 reached 86-90% on data with both moons about 50/50 on seeds 1 and 2 (also 0.003/0.0003 at 75%, 0.012/0.0012 at 92%). Instability appears as lrG rises toward lrD (0.003-0.004 vs 0.005). On the two moons I found no honest collapse: lrG 0.005-0.01 with beta1 0.9 and lrD 0.0003 only throws the cloud off the data (onData near 0, shares 0/1 are noise), never a settled one-moon cloud. Per owner decision 4 the collapse step uses a ring of 8 Gaussians (radius 1.6, sigma 0.07, `RING`, labelled in the figure).

Final runs (`RUN_CFG`, all seed 1, gInit 0.1; stats read at the last checkpoint unless stated; D(real)/D(fake) = mean D output):

| run | lrD | lrG | data | result |
|---|---|---|---|---|
| stable | 0.005 | 0.0003 | moons | onData 0 -> 0.817; upper/lower 0.465/0.535; D(real) 0.578, D(fake) 0.477; D accuracy 0.598; second half (steps 750-1500) onData 0.640-0.817 (mean 0.752); losses (25-step means) D 1.278/G 0.885 at step 25, D 1.092/G 1.229 at step 100, D 1.264/G 0.838 averaged over steps 500-1500 |
| unstable | 0.005 | 0.004 | moons | second half onData 0.130-0.730 (mean 0.464); ends 0.353; losses D 1.389, G 0.704 (steps 500+), i.e. no warning in the loss |
| collapse | 0.005 | 0.0003 | ring of 8 | onData 0.643; shares of generated points by mode 1..8: 16 0 17 1 15 0 15 0 %, off 36%; D accuracy 0.765, losses D 0.867/G 1.597 (last 25 steps) |
| ringOk | 0.003 | 0.003 | ring of 8 | onData 0.890; shares 14 12 9 9 13 14 13 6 %, off 11% |
| disc (G frozen) | 0.005 | none | moons, 600 steps | D accuracy 0.483 at step 0, 1.000 by step 300; D(real) 0.999, D(fake) 0.001 at end |

Other ring runs at rates 0.005/0.0003 (seeds 1-5, no gInit) also ended on exactly 4 of 8 modes, so the collapse is not a seed fluke. Ring seeds are otherwise erratic (0.001/0.001 left several seeds at 4-6 modes), which is why ringOk uses 0.003/0.003 (8 modes on seeds 1 and 2).

Other numbers quoted in the prose: noise grid 15x15 on [-2, 2]^2 through the stable G: 192/225 = 85% on a moon (`quality` gives 86% on a slightly different grid test earlier; the widget count is the quoted one). Gradient step (stable run, checkpoint 8 = step 200, first 60 generated points, move scaled so the mean move is 0.25): mean D 0.459 -> 0.574, mean |grad log D| 0.826 -> 0.436, 19 of the 26 points more than 3 sigma off the moons are pushed toward their nearest centre, on-data share 0.567 -> 0.617. The direction claim is noisy: at neighbouring checkpoints the fraction was 50-75%, and checkpoint 8 was picked for the clearest effect.
