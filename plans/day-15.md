# Day 15 plan: the frontier (skeletons written 2026-10-03)

Lesson skeletons live in `src/content/lessons/frontier/*.mdx`. Each step's `{/* … */}` brief gives mode/props, motion, the one thing to try, the numbers to quote and research IDs (R1…). Every lesson has a top brief listing (a) textbook-stable mechanisms and (b) its research list. **Rule:** reader-facing facts about recent models, numbers or dates come only from primary sources checked when the lesson is written, cited with link + date. Widgets show mechanisms on tiny models and never claim facts about real ones.

## 1. Shared core

- **`src/lib/frontier/seqcost.ts`** (new, small, pure). Counted costs, not timings: `attentionPairs(L, window?, pattern?)`, `attentionFlops(L, d, layers)`, `kvCacheBytes({layers, kvHeads, headDim, L, bytes})`, `ssmFlopsPerToken(d, N)`, `ssmStateBytes(...)`, `hybridCost(ratio, layers, L, ...)`, `weightBytes(params, bits)`. Used by SelectiveScan, LongContextLab and InferenceLab. Test it against closed forms.
- Nothing else is shared. Each widget's sim stays in its own folder, because no second widget uses it (ponytail rule). Tiny MLPs use `src/lib/nn` (MLP, adam, seeded `createRng`), which must not be edited.

## 2. Widgets

All widgets follow the motion contract: autoplay on view, Play/Pause, a scrubbable Timeline with segment-named markers, speed control, a pure `frameAt(t)`, and paused start under reduced motion. They use FigureCard, Timeline, createLoop and readColors from `_shared`, semantic tokens only, and seeded RNG. Training runs once per page in a `worker.ts` and is stored as snapshots; `t` only interpolates the drawing, and readouts snap to real snapshots.

| Widget | sim.ts computes | Motion over t | Renderer | Budget risk |
|---|---|---|---|---|
| TestTimeCompute | DIRECT vs STEP MLPs on digit-sum mod 10; temperature sampling of chains; majority vote, exact/noisy verifier, pass@N, adaptive allocation, REINFORCE over chain length | n sweep; N samples (log); budget spend; RL iterations | Canvas 2D + SVG axes | Low (nn engine ~shared) |
| MoERouter | router softmax + top-k, 8 tiny experts, Switch aux loss, bias balancing, capacity/dropping, param counts; 3 seeded runs | token particles to experts; training snapshots; slot filling; E sweep | Canvas 2D | Medium (training in worker; keep snapshots small) |
| SelectiveScan | diagonal SSM (ZOH), LTI kernel, selective delta/B/C, selective-copy task, outer-product recall memory, hybrid stack costs | tokens streaming; kernel sliding; L sweep; pairs sweep; ratio sweep | Canvas 2D | Low |
| LongContextLab | score-pair counts, KV bytes, RoPE angles, PI/NTK/YaRN scaling, window/sparse masks, needle heatmap on tiny transformer | L sweep (log); position sweep on dials; mask build; heatmap fill | Canvas 2D | **Medium-high**: depends on Day 9 transformer core size |
| InferenceLab | absmax quantization (tensor/row/group), MLP re-eval; n-gram target/draft speculative sampling (exact rule, residual resample), TV distance; speed-up formula; distillation teacher/student | bits sweep; scale shrink; draft/check/accept rounds; trial count (log); gamma sweep; epochs | Canvas 2D | Medium (5 sub-sims; tree-shake per mode not possible, keep each small; corpus string under 8 KB) |
| InterpLab | linear probes + control task; toy superposition model; SAE (L1); lambda sweep; activation patching; head ablation on tiny attention-only transformer | layers walk; training snapshots; feature lighting; heatmap fill; ablation sequence | Canvas 2D | **High**: transformer core + SAE + toy; split into worker, keep main-thread bundle to drawing |
| SafetyLab | tabular Q-learning gridworlds (race, coin, inspector), optimisation sweep, Wilson intervals, logistic filter + greedy substitution attack, adversarial retrain | agent episodes; optimisation sweep; elicitation sweep; attack iterations; checkpoints vs thresholds | Canvas 2D (grid) + SVG charts | Low-medium |
| FrontierLog | load/validate `src/data/frontier-log.json`, filter by kind/topic/status, compute visibility at calendar time t, stale flags | calendar time reveals entries; status changes at statusDate; review cursor | SVG axis + HTML list | Low (data imported statically; keep JSON under ~20 KB) |

### Lesson → step → mode/props

| Lesson | Steps (core, in order) → widget props |
|---|---|
| reasoning-models | `mode="cot"` · `vote n=8 temperature=1` · `verify n=10 temperature=1.2 verifierNoise=0.1` · `budget budget=320` · `rlvr stepCost=0.01` · `cost` |
| moe | `dense` · `route k=2` · `train k=2 aux=0.01` · `collapse k=2 aux=0` · `balance k=2 balance="aux"` (toggle bias) · `capacity capacity=1.0` · `params k=2` |
| ssm-mamba | `lti N=8` · `conv N=8` · `copy variant="lti"` · `copy variant="selective"` · `cost` · `recall N=8` · `hybrid ratio=4` |
| long-context | `quadratic` · `kvcache kvHeads=8` · `extrapolate trainLen=64` · `interpolate method="pi"` · `window window=8 layers=4 sinks=0` · `sparse window=4 global=2 random=2` · `needle task="single"` (toggle multi-hop) |
| efficient-inference | `memory batch=1` · `quantize grouping="tensor"` · `outliers bits=4 outlier` · `speculate gamma=4` · `exact context="th"` · `speedup draftCost=0.1` · `distill T=4` · `tradeoffs` |
| mech-interp | `probe target="repeated"` · `probe target="control"` · `superposition n=6 m=2 sparsity=0.9` · `polysemantic n=12 m=4` · `sae dict=16 lambda=0.01` · `sae-sweep` · `patch` · `circuit` |
| ai-safety | `gaming` · `overoptimize` · `misgeneralize` · `oversight` · `evals trials=20` · `redteam` · `thresholds` |
| whats-new | `all` · `recent kinds=[model,paper] months=12` · `by-topic` · `status` · `review` |

### `src/data/frontier-log.json` schema (FrontierLog)

```ts
type Source = { url: string; title: string; type: 'paper'|'model-card'|'tech-report'|'lab-post'|'docs'|'report'|'policy'; published: string /* YYYY-MM-DD */ };
type Entry = {
  id: string;                       // kebab-case, stable
  date: string;                     // YYYY-MM-DD the thing was released/published
  title: string;                    // plain, no hype words
  kind: 'model'|'paper'|'method'|'benchmark'|'evaluation'|'policy'|'tool';
  topics: string[];                 // lesson ids, e.g. "frontier/moe"; "other" allowed
  org: string;                      // as named in the source
  claim: string;                    // the one checkable claim, ≤ 2 sentences, attributed ("X reports …")
  sources: Source[];                // ≥ 1 primary
  checked: string;                  // YYYY-MM-DD the writer verified the sources
  status: 'holding'|'contested'|'superseded'|'retracted';
  statusDate?: string; statusNote?: string; statusSources?: Source[];  // required unless 'holding'
};
type Log = { reviewed: string; nextReview: string; entries: Entry[] };
```
A unit test validates the schema: dates parse, at least one primary source per entry, non-holding entries have statusSources, and `nextReview` is at most `reviewed` + 6 months.

## 3. Reuse and dependencies on Days 9–11

- **Day 9 `src/lib/transformer/` (needed by LongContextLab steps 3, 4, 7 and InterpLab steps 1, 2, 7, 8).** The functions I need:
  1. A seeded config `{vocab, d, heads, layers, mlp: boolean, pos: 'rope'|'learned'|'none', ropeBase, ropeScale(pos)→pos}` where position scaling is a function, so PI, NTK and YaRN can be plugged in.
  2. A pure `forward(params, tokens, hooks?)` returning logits, the residual stream per layer and position, and attention weights per head. `hooks` can overwrite a residual vector or a head output (for patching), or zero it (for ablation).
  3. A worker-safe `train(config, taskSampler, steps, seed)` returning params and a loss history, with Float32Array params.
  4. Tasks: key–value retrieval at a given length, and repeated random sequences for induction heads.
  Without these, use the fallbacks written in each lesson brief (hand-built RoPE attention head, or a 2-layer MLP).
- **Day 11 RL (optional, SafetyLab and TestTimeCompute `rlvr`):** a tabular Q-learning gridworld and a REINFORCE-with-baseline helper. If none ships, each sim keeps a ≤ 40-line local version. Don't block on it.
- **Reuse inside Day 15:** `seqcost.ts` (section 1). InferenceLab's tradeoffs mode reuses its own quantize and distill results. Link to, but don't import from, existing widgets: KV cache → `llms/inference-kv-cache`, RoPE → `transformers/positional-encodings`.

## 4. Research protocol and sources

Protocol: the lesson writer answers every R-question in their lesson brief from primary sources (paper, official model card, tech report or lab post) via web search. Each reader-facing claim cites link + date. No numbers about real models come from memory. Mark anything unconfirmed and leave it out of the prose. Set `lastReviewed` when the lesson is written. Re-review at least every 6 months (IDEA.md §7).

Sources identified (URL checked 2026-10-03 by web search unless marked VERIFY):

| Lesson | Confirmed | VERIFY |
|---|---|---|
| reasoning-models | 2201.11903, 2203.11171, 2305.20050, 2407.21787, 2408.03314, 2501.12948, 2503.11926 | 2110.14168; OpenAI "learning to reason" post URL; current reasoning-model reports |
| moe | 1701.06538, 2101.03961, 2412.19437 | 2006.16668, 2401.04088, 2401.06066; current model cards |
| ssm-mamba | 2312.00752, 2405.21060, 2403.19887 | 2111.00396, 2402.19427, 2312.04927, 2402.01032 |
| long-context | 2104.09864, 2306.15595, 2309.00071, 2309.17453, 2307.03172, 2404.06654 | 2004.05150, 2007.14062, 2205.14135, 2305.13245, NTK-aware origin, needle repo |
| efficient-inference | 2211.17192, 2302.01318, 2210.17323, 1503.02531 | 2208.07339, 2306.00978, 2211.10438, 2309.06180; FP8/FP4 format docs |
| mech-interp | 1610.01644, 2209.10652 (+transformer-circuits 2022 toy_model), transformer-circuits 2023 monosemantic-features, 2309.08600, transformer-circuits 2024 scaling-monosemanticity, transformer-circuits 2025 attribution-graphs/methods | 1909.03368, 2406.04093, 2202.05262, 2211.00593, induction-heads URL, "Biology" URL, 2501.16496 |
| ai-safety | DeepMind spec-gaming post (2020-04-21), 2210.10760, 2503.11926, 2105.14111, 2401.05566, 2412.14093, 2305.15324, 2503.14499, International AI Safety Report 2026 (pub. 2026-02-03) | 1606.06565, 2210.01790, 2202.03286, 2209.07858, 2307.15043; lab frameworks (RSP, Preparedness, FSF); EU AI Act text |
| whats-new | International AI Safety Report 2026 | AI Index 2026 URL (found, pub. 2026-04-13), arXiv listing URL, Epoch AI; all entries |

Totals: **37 confirmed** primary sources; **about 35 VERIFY items**, mostly well-known arXiv IDs not re-checked, plus current model cards and framework pages, which must be fetched when the lessons are written.

## 5. Build order

1. `seqcost.ts` + tests (an hour; unblocks three widgets).
2. Independent of other days, can start in parallel: **MoERouter** (moe; IDEA.md flagship-style §8.6), **TestTimeCompute** (reasoning-models), **SelectiveScan** (ssm-mamba), **InferenceLab** (efficient-inference), **SafetyLab** (ai-safety), **FrontierLog** + data file (whats-new; research-heavy, small code).
3. After the Day 9 transformer core lands and passes its tests: **LongContextLab** (long-context), then **InterpLab** (mech-interp; largest).
4. Coordinator review per BUILDING.md §5/§5a; budget check `node scripts/check-widget-budgets.mjs …` with all eight.

Writing order within each lesson: research questions first, then probe the sim (BUILDING.md §10 stage 3), then prose.

## 6. Risks and open questions for the owner

- **Categorical colours.** MoE experts, sparse-mask parts and FrontierLog kinds need distinguishable categories. Today the skeletons use number labels, shapes and `--viz-seq` shades. Should we add categorical tokens to `tokens.css`? (This is the same open question as BUILDING.md §7.)
- **Day 9 dependency.** LongContextLab and InterpLab wait on `src/lib/transformer/`. The fallbacks are weaker (hand-set heads). Should those two lessons wait for Day 9, or ship with the fallbacks?
- **InterpLab budget.** A transformer, a toy model and an SAE could approach 40 KB gz. If they do, split the work into two widgets (`SuperpositionSAE` and `CircuitProbe`). Decide at build time from measured numbers.
- **whats-new upkeep.** Who re-reviews every 6 months, and should the review date show prominently on the page? Should entries older than ~18 months drop off or move to an archive?
- **Safety tone.** The oversight toy (step 4) demonstrates conditional behaviour, not intent. The lesson says so once and then cites the real studies. Is that framing acceptable?
- **Hardware numbers.** The efficient-inference step 1 roofline uses a labelled example bandwidth/compute ratio rather than a named GPU. Is that preferred, or should it cite one vendor datasheet?
- **Outline changes.** None dropped. Additions: "When an expert is full" (moe), "What a fixed-size state forgets" (ssm-mamba), "Why serving is slow" (efficient-inference), and "Behaving well only when watched" (ai-safety; covers the outline's "deception concerns" bullet).
