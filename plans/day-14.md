# Day 14 plan: Applied AI

Lessons (order fixed): applied/vector-search, rag, prompting, tool-use, agents, multimodal, lora-qlora. Skeleton briefs live in each `.mdx` as MDX comments, one per step.

**Honesty rule for the whole day.** Every mechanism on screen runs for real: embeddings, search, indexes, prompt assembly, token budgets, tool validation, the agent loop, ViT patching, contrastive training, LoRA training and quantization. Where a large LLM would be needed, we use either (a) a tiny transformer trained for this page (PromptModel) whose effects are measured, or (b) a deterministic stand-in labelled *in the figure* ("Reader: copies the best-supported sentence (not a language model)", "Scripted policy (stands in for the model)"). Claims about large models are cited from primary sources, never demonstrated by the stand-ins.

## 1. Shared core: `src/lib/applied/`

| File | Exports | Notes |
|---|---|---|
| `corpus.ts` | `PASSAGES: {id, title, text, date?}[]` (~40), `BACKGROUND: string[]` (~150 sentences used only for co-occurrence statistics), `QA: {q, goldIds, answer}[]` (24 RAG questions), `TASKS` (30 agent tasks with checkers), `chunk(text, size, overlap)` | Made-up "Elm Street Tool Library" handbook, labelled as invented in the first figure. Authored so synonym pairs (late/overdue, cost/fee, borrow/loan) co-occur; includes an old + new opening-hours notice and an item code (PW-200) for the failure steps. |
| `embed.ts` | `tokenize`, `buildIndex(passages, background, k=16) → {vocab, idf, U, S, docVecs}`, `embedText(index, text) → Float64Array`, `cosine`, `dot`, `bm25(index, query)`, `rrf(rankings)` | LSA: TF-IDF then truncated SVD via symmetric eigen of the doc Gram matrix (reuse `eigSym` from `PCAProjector/sim.ts`). Closed form, deterministic, < 20 ms; computed at widget init, no worker. Unit tests pin the probe queries the lessons rely on (e.g. "bring the drill back late" → overdue passage rank 1 under LSA, not top 5 under BM25). |
| `ann.ts` | `clustered(seed, n, dim)`, `exactTopK`, `buildIVF(pts, nlist, seed)`, `searchIVF(ivf, q, k, nprobe)`, `buildHNSW(pts, M, seed)`, `searchHNSW(g, q, k, ef)` returning `{ids, comparisons, trace}`, `recallAtK` | 2-D sets reuse `KMeansLab` `kmeans` + `initCentroids('plusplus')`. `trace` records every hop/visit so `frameAt(t)` just indexes it. Counts are distance computations, not ms. The 32-D trade-off sweep (5000 pts, 200 queries) runs in a worker if init > 50 ms. |
| `loop.ts` | `Message`, `ToolDef` (name, description, JSON-schema subset), `validate(call, def)`, `parseCall(text)`, `runLoop({policy, tools, task, maxSteps, budget, memory, retries, gate}) → Trace` | One loop for tool-use and agents. `Policy = (ctx) → string`, implemented by PromptModel (tool-use) and by `scriptedPolicy(rules, rng, p)` (agents). `gate` = allowlist + confirmation rules. `Trace` is a list of events with token counts; frames index it. |

Determinism: `createRng` everywhere; sampling uses a seeded RNG; scripted mistakes use a seeded rate `p` and are labelled "simulated mistakes".

**Token counting:** use the Day 9 BPE tokenizer if it exists (`src/lib/transformer/` or the tokenization widget); otherwise count words and *label them words*.

## 2. Widgets

All follow the motion contract (autoplay on view, Play/Pause, scrubbable timeline, speed, pure `frameAt(t)`, reduced motion starts paused), `FigureCard` + `Timeline` from `_shared`, tokens only.

| Widget | sim.ts computes | Renderer | Budget risk |
|---|---|---|---|
| `EmbedSearch` | LSA vectors, PCA-2D projection, BM25 vs cosine rankings, dot vs cosine | SVG (40 points) | low, ~25 KB closure |
| `AnnIndex` | exact / IVF / HNSW traces, recall vs comparisons sweep | Canvas (2000 pts) + SVG chart | low-medium; worker for 32-D sweep |
| `RAGPipeline` | pipeline stages, chunking sweep, prompt assembly vs budget, hybrid RRF, eval over 24 QA, extractive reader | SVG + HTML text | medium (text layout); reader is ~40 lines |
| `PromptLab` | runs PromptModel on test sets: format, tag, shots, orders, scratchpad, constrained decoding, eval-set resampling | SVG/HTML cells + bars | medium-high: transformer forward code (~6 KB) + weights fetched as a binary asset (not JS) |
| `AgentTrace` | `runLoop` traces for tool-use (PromptModel policy) and agents (scripted policy) | HTML transcript + SVG lanes | medium: shares PromptModel code with PromptLab; scripted agent path must not import weights |
| `ClipLab` | shapes dataset, ViT patching, patch embed = conv check, contrastive training snapshots (worker), zero-shot, VLM token counts, order-blindness | Canvas (images) + SVG | low-medium; worker for training |
| `LoRALab` | memory formula, low-rank product, full vs LoRA training, SVD spectrum, merge, int4/NF4 blockwise quantization, QLoRA training, data-size sweep | SVG + Canvas heat maps | low |

### Modes per lesson step

| Lesson | Step → mode / key props |
|---|---|
| vector-search | 1 `EmbedSearch keyword` · 2 `embed` · 3 `similarity` · 4 `AnnIndex exact k=5` · 5 `ivf nlist=16 nprobe=1` · 6 `hnsw M=6 ef=8` · 7 `tradeoff dim=32` |
| rag | 1 `RAGPipeline closed-book` · 2 `pipeline k=3` · 3 `chunking chunk="sweep"` · 4 `prompt k=5 budget=256` · 5 `miss hybrid=false` · 6 `conflict k=4 budget=160` · 7 `eval sweep="k"` |
| prompting | 1 `PromptLab format` · 2 `specific` · 3 `shots task=shift` · 4 `order task=sort shots=4` · 5 `steps task=shift-reverse` · 6 `structured temperature=1.2` · 7 `eval` |
| tool-use | 1 `AgentTrace no-tool` · 2 `schema` · 3 `call` · 4 `result` · 5 `errors temperature=1 retries=2` · 6 `safety` |
| agents | 1 `AgentTrace loop` · 2 `plan` · 3 `short-memory budget=300` · 4 `long-memory` · 5 `multi p=0.1` · 6 `compound p=0.05` · 7 `eval runs=5` |
| multimodal | 1 `ClipLab patches` · 2 `embed` · 3 `contrastive` · 4 `zeroshot` · 5 `vlm` · 6 `limits` |
| lora-qlora | 1 `LoRALab memory model=7B` · 2 `lowrank r=2` · 3 `train` · 4 `spectrum` · 5 `merge r=4` · 6 `quantize scheme=nf4` · 7 `qlora r=4` · 8 `data r=4` |

Motion over `t` per step is specified in each step's comment. Common patterns: rankings' bars grow then glide to new order; traces (scan, hops, loop iterations, tokens) are revealed by index; training replays snapshots; sweeps move a cursor along a precomputed curve. Readouts come from the same frame as the drawing.

## 3. Reuse and dependencies on Days 9–10

Reuse: `PCAProjector` `eigSym` (LSA, 2-D projection); `KMeansLab` `kmeans`/`initCentroids` (IVF); `src/lib/cnn/conv.ts` `conv2d` (patch embedding = strided conv, tested); `createRng`; `_shared`. `DotProduct` geometry is linked, not repeated. `SoftmaxBars` style for next-token bars if its component fits.

**PromptModel (needed by prompting + tool-use).** From the Day 9–10 core `src/lib/transformer/` I need:
1. `forward(params, tokens) → logits` for a causal decoder (2 layers, d≈48, 4 heads, context ≈ 64, char-level vocab ≈ 40), and per-layer attention weights.
2. `generate(params, prompt, {temperature, seed, logitMask?, stop?})` with a `logitMask(position, prefix) → allowed[]` hook (constrained decoding).
3. Weight (de)serialization to a compact binary (float16 or int8 + scales).
4. A deterministic Node training script (seeded, `scripts/train-promptmodel.mjs` or reuse theirs) so weights are reproducible.

Training mixture (owned by Day 14, run by that script): 3-letter strings over a..h; tasks reverse, shift, sort, copy and shift-then-reverse; format `abc>cba;` with optional task tag `[rev]` (50%), 0–8 in-context examples, composition with and without the written intermediate (`abc>bcd>dcb;`), and tool data `37*48>CALL calc(37*48) RESULT 1776 >1776;` plus direct-answer arithmetic. Delimiters `=` and `|` never appear anywhere in training (format-sensitivity test). If Day 10 trains an ICL model, extend its mixture rather than train a second model.

**Weights are trained offline and shipped as a static asset** (est. 60–120 KB, fetched on view, not counted in the JS island). Training in the browser would take minutes. Open question below.

If `src/lib/transformer/` does not land: Day 14 writes its own minimal forward + generate (~150 lines) in `src/lib/applied/tinylm.ts` with the same API so swapping later is a one-line import change.

## 4. Build order inside the day

1. `src/lib/applied/{corpus,embed,ann}.ts` + tests (probe queries pinned). Blocks vector-search and rag.
2. vector-search (`EmbedSearch`, `AnnIndex`) and multimodal (`ClipLab`) and lora-qlora (`LoRALab`) in parallel: independent.
3. rag (`RAGPipeline`), after 1.
4. `loop.ts` + tests.
5. PromptModel weights (needs Day 9–10 core or the fallback) → prompting (`PromptLab`) → tool-use (`AgentTrace`, PromptModel policy).
6. agents (`AgentTrace` scripted modes) after tool-use, extending the same widget.

## 5. Risks and open questions for the owner

1. **Stand-ins (please confirm the labels):**
   - RAG generator = extractive reader, labelled "copies the best-supported sentence (not a language model)". Lost-in-the-middle is cited, not simulated.
   - Agent policy = scripted rules, labelled "Scripted policy (stands in for the model)"; mistakes injected at a seeded, labelled rate `p`. The loop, memory, budget, tools and evaluation are real.
   - Safety step in tool-use uses scripted calls (the gate logic is real).
   - VLM step shows real token plumbing only; no image-to-text generation unless a tiny captioner on the Day 9–10 core is wanted (extra work, more weights).
2. **Shipping trained weights as a static binary** (PromptModel ~60–120 KB) instead of training in the browser. OK? The alternative is a rule-based "model" for prompting too, which would turn prompting into claims-only.
3. **LSA instead of a neural sentence encoder** (16-D, from the corpus). Honest and fast, but synonymy depends on corpus authoring; the agent must iterate the corpus until the pinned probes pass. A small trained encoder would be more realistic but costs weights and time.
4. **Fictional corpus.** The handbook is invented so no model could know it; it is labelled as made up. Alternative: index this site's own lesson summaries (real, but changes as lessons are written).
5. **Budget:** `PromptLab` and `AgentTrace` both carry the transformer forward; keep it in a shared module so Vite splits it once. Watch the RAGPipeline text layout at 390 px.
6. **Determinism across browsers:** PromptModel inference must match Node for quoted numbers; use only + − × ÷ and a deterministic `exp` (as Seq2SeqAttention does) or quote rounded values.
7. **Citations marked VERIFY** in the skeletons must be checked before writing (arXiv ids, LLaVA-1.5 token count, ZeRO 16 bytes/param, NF4 levels, vendor doc URLs).

## 6. Core results (measured; these override the briefs)

Started 2026-10-12 with owner permission. The Day 10 `icl` model is not extended (its numbers are quoted by Day 10 lessons); `prompt-model` is a separate entry in `scripts/train-models.ts`. Defaults taken for §5 open questions: stand-ins and their figure labels as written in §5.1, shipped weights (owner decision 1), LSA, the invented handbook (owner decision 4).

### 6.1 PromptModel

(core agent: training command, config, size, every metric the lessons may quote)

### 6.2 Retrieval and loop core

(core agent: corpus stats, probe-query results, ANN numbers, loop and task-suite results)
