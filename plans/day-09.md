# Day 9 plan: Transformers

Lessons (curriculum order): tokenization, embeddings, self-attention, multi-head-attention, positional-encodings, transformer-block, causal-masking. Each skeleton's MDX comments are the per-step brief; this file is the shared design.

## 1. Shared core: `src/lib/transformer/`

**Decision: one tiny decoder-only transformer ("TinyLM"), trained offline by a deterministic Node script, weights shipped as a fetched data file; every forward pass the widgets show runs live in the browser on those weights.**

Why not train per page in a worker (the Seq2SeqAttention route): five of seven lessons need the same model, often in 4 to 6 islands per page. Training a 7k-parameter transformer costs roughly 10x the seq2seq RNN (several seconds per page), screenshots would catch half-trained states, and Chrome/Node bit-identity would need the custom `exp` everywhere. Offline training gives one model whose numbers are fixed, so prose can quote them, and Day 10 (sampling, KV cache, pretraining) can reuse it. The shared core uses the arithmetic-only exp/tanh from the Seq2SeqAttention precedent in both training and inference, as required by plans/README.md; RoPE rotations also use arithmetic-only sine/cosine. Node and Chrome were checked directly (see the verification below).

Why not bundle the weights as JS: about 7.3k params as float16 is ~15 KB, too much on top of a ~29 KB widget closure. A `fetch` of `/models/<name>.json` plus `/models/<name>.bin` is data, not island JS; it is cached across all Day 9/10 pages, and a module-level promise shares one fetch between islands on a page.

**Model (TinyLM).** Word-level vocab of 34 (`[start]`, `.`, determiners this/that/a (sg), these/those (pl), the (either); adjectives small/big/red/old; animals cat(s), dog(s), fox(es), bird(s), sheep (sg = pl); foods apple(s), seed(s), fish (sg = pl); verbs sees/see, chases/chase (animal object), eats/eat (food object), likes/like (any)). Sentence = `[start] DET ADJ0-2 ANIMAL VERB DET ADJ0-2 NOUN .`, with at most three adjectives total, max 10 tokens, context 10. Without that total cap the two 0-2 adjective ranges would allow 11 tokens. Number agreement (det, noun, verb) and verb-object selection give real, checkable long-range dependencies, and sheep/fish force the model to read the determiner. Architecture: d 16, 2 heads (d_k 8), 2 blocks, pre-LayerNorm, ReLU MLP width 64, learned absolute positions, tied unembedding, causal. Exactly 7,330 parameters, including projection/MLP biases and final LayerNorm. Config also supports other vocabularies, widths, layer/head counts, context lengths, attention-only blocks, untied unembedding, pre/post normalization, and learned/RoPE positions with pluggable position scaling.

**Files (implemented in Task 1)**
- `grammar.ts`: `VOCAB`, `WORD_CLASS` (kind, number, noun category, verb object class), `sampleSentence(rng)`, `allowedNext(prefix): Set<number>`, `encodeWords` / `decodeIds`, and fixed seeded `SAMPLE_200` (evaluation seed 92026). Unknown words and invalid prefixes raise errors. Terminal prefixes have no legal next tokens.
- `model.ts`: `Config`, `Weights` (one flat `Float64Array`), `parameterLayout` (named offsets, rows, cols, length), `initWeights`, `forward(w, ids, opts) => Trace`. Options: causal on/off, noPositions, positionScale 0..1, noScale, pre/post norm, ropeScale, per-head ablation, and replacement patches at named residual streams/absolute token positions. `Trace` includes embedding-plus-position `x0`; each layer's normalized inputs, q/k/v, all raw unscaled Q.K scores, masked/scaled softmax weights, individual head outputs, concatenation, attention projection, residuals, post-ReLU hidden activations and MLP outputs; final normalized residuals, logits and probabilities. Full-pass arrays are row-major tokens-by-width; each head's score/weight grid is T-by-T.
- `model.ts` also exports `prefill` and `decodeStep` with preallocated per-layer/head KV buffers, `logitLens`, `grammaticalMass`, and `generate(w, prompt, { sampler, logitMask, stopId, maxTokens })`. A mask supplies allowed ids; it never changes the grammatical-mass evaluation. Causal generation uses the cache; unmasked generation recomputes the full prefix. KV caching rejects bidirectional attention, context overflow, invalid ids and a cache from different weights; discard caches after changing weights. Ablated head outputs are zeroed; attention weights remain available to inspect. Patches replace the specified stream vector; training rejects patch interventions.
- `train.ts`: `backward` (manual Float64Array reverse pass), token-mean `lossAndGrad`, `makeAdam` / `adamStep` with running bias products and global gradient clipping. Supports tied/untied outputs, learned/RoPE positions, pre/post norm, attention-only blocks, skipped targets and optional soft targets. Both shipped jobs use sampled next-token cross-entropy, not oracle targets. Residual-gradient diagnostics expose the last example's per-layer gradients scaled by the batch token count.
- `math.ts`: deterministic exp/tanh matching `Seq2SeqAttention/sim.ts`, arithmetic RoPE rotations, stable softmax, cosine. `index.ts` is the public API.
- `load.ts`: `loadModel(name): Promise<Weights>` lazily fetches a binary plus manifest, shares one promise per name, checks its SHA-256 and validates layout/length/vocabulary. Float16 is the default; int8 uses manifest per-tensor-row scales above 40 KB. `encodeWeights`, `parseModel`, IEEE binary16 round-to-nearest/ties-to-even conversion also serve the offline pipeline and tests. Snapshot metadata records offsets into a separate float16 binary of embedding tables.
- `report.ts`: pure frozen-set evaluation, head summaries and pairwise attention-row total variation; embedding/context/position and residual-gradient probes.
- `scripts/train-models.ts`: the single seeded offline pipeline, bundled with installed esbuild and run in Node. Implemented entries are `tiny-lm`, `tiny-lm-nomask`, `day9` (both), and `report` (`--write` refreshes manifests from verified shipped binaries without retraining). Future Day 10/14/15 model entries belong here; no second transformer or training script. It prints loss/mass progress and the full measured report; fails acceptance before writing a causal model below the bar. Eight embedding checkpoints include initialization and the final update.
- Outputs: `public/models/tiny-lm.bin` + `.json`, `tiny-lm-nomask.bin` + `.json`, `tiny-lm-snapshots.bin` + `.json`. Model manifests contain config/vocab/offsets/dtype, SHA-256, optimizer/seed/training settings, loss curve, rounded-weight metrics, head report and probes. Embedding snapshots share the binary16 representation and have explicit checkpoint byte offsets. Each model binary is 14,660 bytes; eight embedding tables total 8,704 bytes. No weights in island JS, no worker needed for inference.
- `tests/unit/transformer-core.test.ts`: all-parameter numerical gradients in pre/post norm with learned/RoPE positions; attention-only/untied/soft-target gradients; causal invariance; permutation equivariance without positions/mask; cache equality for both norm/position modes; ablations/patches; deterministic Adam and constrained/unmasked generation; grammar validity; binary16 fixtures and exhaustive finite-code round trip; int8 row scales; lazy memoised fetching; shipped checksums, exact head reports, grammatical mass and eight snapshots.

Acceptance bar: token-weighted unmasked probability on legal next tokens must be at least 0.97 on `SAMPLE_200` after quantization, and two heads must have clearly different patterns. The generator operationalizes the second check as mean attention-row total variation of at least 0.2 for two heads **within the same layer**, excluding the trivial first query. This threshold was in the generator before training. Both checks passed; measured results follow below. No head role is inferred from its name or an untested textbook expectation.

## 2. Widgets (all: FigureCard + Timeline + createLoop, pure `frameAt(t, mode, ...)` in `sim.ts`, tokens only, reduced motion starts paused)

| Widget (folder) | sim.ts computes | Renderer | Budget risk |
|---|---|---|---|
| `TokenizerLab` | byte-level BPE trainer + encoder on a ~3 KB built-in English corpus (tie-break: lowest byte order), merge history, tokens-per-100-chars curve vs merges, quirk presets | SVG chips (under 150 elements) and an SVG path for the vocab curve | low (~3 KB gz corpus) |
| `EmbeddingSpace` | PCA basis of final embeddings (reuse `eigSym`, `covariance` from `PCAProjector/sim.ts`), snapshot projections, cosine/dot tables, offset arithmetic + 16-D nearest neighbours, PCA of block-2 residuals | SVG scatter (34 points) | low |
| `AttentionExplorer` (signature, IDEA §8.5) | per-mode views over `Trace`: qkv strips, score grid, scaled/softmax row, value mix, head split, concat + W_O, gallery, ablation bars | Canvas for matrices and strips, SVG overlay for arcs and labels | **medium**: most modes; keep shared drawing helpers in the widget, split the file if it passes ~30 KB source |
| `PositionLab` | permutation demo, position fade ablation (real forwards at the displayed scale), sinusoidal table, learned-position cosine matrix, RoPE rotations, relative score grids | Canvas heatmaps + SVG plane | low |
| `BlockFlow` | block diagram flow, residual deltas, MLP hidden activity, pre/post gradient norms (live `backward` on random-init stacks), logit lens, parameter counts | SVG diagram + Canvas strips | low to medium (pulls `train.ts`) |
| `CausalMask` | mask grid, -inf row, parallel vs generative passes, edit-the-future check, encoder vs decoder grids, leak comparison (greedy generation from both models) | Canvas grids + SVG labels | low |

Lesson to step to mode (S = "these small sheep chase a red fox"; L = layer, H = head; exact head picked from the head report):

| Lesson | Steps (in order) → widget mode |
|---|---|
| tokenization | granularity · merges · apply · playground · vocab · quirks (TokenizerLab) |
| embeddings | lookup · training · similarity · projection · arithmetic · contextual (EmbeddingSpace) |
| self-attention | arcs (L1, number head) · qkv · scores · softmax · mix · explorer · reach (AttentionExplorer; query "sheep" for scores/softmax/mix) |
| multi-head | compare · split · heads · concat · gallery · ablate (AttentionExplorer) |
| positional | shuffle · ablate · sinusoidal · learned · rope · relative (PositionLab) |
| transformer-block | overview · residual · mlp · norm · stream · count (BlockFlow) |
| causal-masking | leak · mask · neginf · parallel · edit · encdec (CausalMask) |

Motion over t, per widget: TokenizerLab, one segment per merge/preset with chips sliding together; EmbeddingSpace, points glide between real checkpoints with the readout snapped to the nearest real one; AttentionExplorer, a query cursor sweeps tokens while the stage (score, scale, softmax, mix) replays for it, with particles along arcs scaled by weight; PositionLab, words swap places, a position cursor, rotating arrows; BlockFlow, particles climb the residual line and cross between columns only at attention; CausalMask, a row sweep, bars dropping to -inf, a pass counter. Markers name segments, never points.

Every lesson step in the skeletons has a distinct mode or prop set; reference steps have no widget.

## 3. Reuse

- `src/lib/math/rng.ts` `createRng` (generator, random-init stacks, demo sentences).
- `PCAProjector/sim.ts` `eigSym`, `covariance` (embedding and residual projections). Precedent: ensembles imports TreeSplitter.
- `Seq2SeqAttention/sim.ts`: code idiom (typed-array `mvAdd`/`mvBack`, Adam with running bias products, snapshots). Its arithmetic-only exp/tanh are the determinism reference; the core keeps them in its own pure math module rather than importing a widget.
- `SoftmaxBars/sim.ts` `softmax` if its signature fits; otherwise the core's own, as only one is needed.
- `_shared`: FigureCard, Timeline, createLoop, readColors, advance.
- `src/lib/nn` (scalar autograd) is **not** used: a scalar graph over 7k params times 10 tokens is too slow; typed-array manual backprop like Seq2Seq instead, gradient-checked.
- Links back to Day 8 (`seq2seq-attention`, `vanishing-gradients-time`), Day 6 `normalization`, Day 7 `resnets`, `classic-ml/pca`, `math/dot-product`.

## 4. Build order

1. **Core first (one agent or the coordinator):** `src/lib/transformer/*`, `scripts/train-models.ts`, binary/manifest pairs (causal model, no-mask model, embedding checkpoints), tests, head report. Every lesson except tokenization waits for it. Probe before handing off: cats - cat + dog nearest word; sheep contextual split; learned-position similarity; pre/post-norm gradient pattern; leak model output. Write the results into this file under "Probe results" so lesson agents start from numbers.
2. **In parallel, any time:** tokenization (TokenizerLab has no core dependency).
3. **After the core:** embeddings, self-attention (builds AttentionExplorer with ALL self-attention and multi-head modes listed above), positional-encodings, transformer-block, causal-masking, in parallel.
4. **After self-attention:** multi-head-attention (only writes the lesson; adds or fixes AttentionExplorer modes if needed, so one agent at a time owns the widget).

## 5. Risks and open questions

- **Causal attention in the self-attention lesson.** TinyLM is a decoder, so its attention grids are triangular two lessons before causal masking explains why. Plan: show full Q K^T scores, hatch the future cells, one forward-link sentence. Alternative: ship a second, bidirectional model (masked-word training) for lessons 3 to 5, at the cost of a second core training run. Owner: is the forward reference acceptable?
- **Toy vocabulary vs BPE.** TinyLM is word-level (34 words) while lesson 1 teaches BPE; prose must say real models feed BPE tokens. The explorer only accepts the 34 words; the input needs a clear message and word suggestions.
- **Interpretability is not guaranteed.** Heads may not split cleanly; analogies may fail; post-norm may not show the Xiong pattern at this size. The skeletons say "probe first". If a probe fails, change the demo (seed, grammar) or the prose, never assert.
- **Categorical colours: resolved.** Use the approved `--viz-cat-1..8` tokens for word-class identity, with labels/shapes; signed number distinctions may use `--viz-pos`/`--viz-neg`. See plans/README.md.
- **Fetched weights: approved.** Binary/manifest data is outside the island JS budget (plans/README.md). Islands must reserve space and show a quiet loading state while loading. See actual byte sizes above.
- **Generator path:** `scripts/train-models.ts` plus owner-published outputs in `public/models/`. Retraining changes every quoted number, so tests pin the manifest metrics.
- Bengali in tokenization quirks is approved in plans/README.md; choose the exact sample when authoring that lesson.
- Day 10 reuse (sampling lab, KV cache) of TinyLM is assumed; Day 10's planner should read §1.

## 6. Task 1 head report and probe results (2026-10-03)

Source: the decoded `public/models/tiny-lm.bin` and its generated `.json` manifest, not pre-quantization training arrays. Both models use seed 9, 6,000 Adam updates, batch 16, peak learning rate 0.003, 100-update warmup and linear decay to 0.15 of the peak. Both use the same seeded sampled-sentence schedule and next-token objective; the attention mask is their only configuration difference.

Reproduce in PowerShell from the repository root (temporary bundle only):

```powershell
npx esbuild scripts/train-models.ts --bundle --platform=node --format=esm --outfile="$env:TEMP/ai-learner-train-models.mjs"
node "$env:TEMP/ai-learner-train-models.mjs" day9 --steps 6000 --seed 9 --batch 16 --lr 0.003
node "$env:TEMP/ai-learner-train-models.mjs" report
npx vitest run tests/unit/transformer-core.test.ts
npm run check
```

Rounded-weight acceptance:

- Grammar evaluation: 1442 next-token positions across 200 sentences; mean legal mass **0.997887330707**, above 0.97. Sampled next-token cross-entropy **1.667588186057** nats/token. No grammar mask is applied during evaluation. This is a fixed synthetic-grammar evaluation, not evidence about natural-language generalization.
- Distinct-head acceptance: layer 2, heads 1 and 2, mean attention-row total variation **0.526220157357**, above 0.2. Layer/head labels in this report are one-based; the API and manifests use zero-based indexes.
- Causal binary SHA-256: `c717b1ce1fddab371b6e299ad6436f4e3d5919a2f28d250959c9dfe1f665b756`. No-mask binary SHA-256: `ba455717de46533d1985f48c5d8645295bd724c0020dd9ad374b50343ae379ac`.

Mean attention weights (not exclusive categories: a previous token may also be a determiner or verb). Exclude the first query; determiner column uses token 1; verb column averages only queries at/after the subject verb. The other columns average all remaining queries.

| Layer | Head | Previous token | Subject determiner | Verb | [start] |
|---|---|---|---|---|---|
| 1 | 1 | 0.218817 | 0.278142 | 0.283158 | 0.211371 |
| 1 | 2 | 0.224982 | 0.271680 | 0.198074 | 0.132816 |
| 2 | 1 | 0.253765 | 0.325902 | 0.354449 | 0.156263 |
| 2 | 2 | 0.260529 | 0.230937 | 0.325316 | 0.211887 |

Measured ablations (one head output zeroed, otherwise unchanged):

| Condition | Grammatical mass |
|---|---|
| Full model | 0.997887 |
| No positions | 0.869001 |
| Remove layer 1 head 1 | 0.861865 |
| Remove layer 1 head 2 | 0.860681 |
| Remove layer 2 head 1 | 0.992661 |
| Remove layer 2 head 2 | 0.983285 |

For `[start] these small sheep chase a red fox`, query `sheep` is absolute token 3 (zero-based). Top target per head:

| Layer | Head | Top target | Weight |
|---|---|---|---|
| 1 | 1 | [start] | 0.526539 |
| 1 | 2 | these | 0.556897 |
| 2 | 1 | sheep | 0.714935 |
| 2 | 2 | these | 0.670027 |

Probe results for later lesson authors:

- `cats - cat + dog`, excluding those three input words: nearest **dogs**, cosine **0.985441**; `dogs` rank **1**. Cat-to-cats versus dog-to-dogs offset cosine **0.991701**. These particular analogy results are measured, not a promise for every word pair.
- Sheep under `this` versus `these` in `[start] this/these small sheep sees/see a red fox`: cosine at embedding-plus-position, block 1 and block 2 is **1.000000, 0.880533, 0.172188**. No claim about `the` lying between the two was tested.
- Learned position 0 versus positions 1 through 9: cosines **-0.489122, -0.168683, -0.309374, 0.116493, 0.131489, -0.216543, -0.073907, -0.048690, -0.076716**. Position 9 is never an input during these next-token training sequences (the longest sentence ends there), so its row remains at initialization; do not present that row as learned.
- Random-init 12-block stack, seed 91, same input and weights for the pre/post comparison: input residual gradient norm **0.886831** pre versus **0.163976** post; final block residual norm **0.024168** pre versus **0.123528** post. These are gradients before final LayerNorm for this specific initialization; they do not establish a universal pre/post-norm training claim. Full per-layer arrays are in the manifest.
- Greedy causal continuation from `[start] these small sheep`: `[start] these small sheep see a old old dog.`. The grammar treats `a` and adjectives syntactically and does not model the phonetic a/an rule.
- No-mask model with future tokens present: loss **0.000121372919**, legal mass **0.999917504149**. Restricting the same weights to causal/prefix-only attention gives loss **11.089744**, mass **0.357285**; its unmasked greedy prefix-only continuation is `[start] these small sheep.`. The high full-sequence mass is leakage, not success at generation.

Verification: 24 transformer tests passed (including all-parameter gradients, causal invariance, cache equivalence, binary16 round trips, shipped reports and snapshots); the repository suite passed 561 tests in 53 files. An isolated headless Chrome 154.0.8037.93 matched Node 22.15.0 exactly for 272 logits and all 7,330 parameters after five fixed-batch Adam updates (maximum differences 0 and 0). This was a direct in-memory check without a server; it did not rerun all 6,000 updates in Chrome. No lesson or widget was written in Task 1.
