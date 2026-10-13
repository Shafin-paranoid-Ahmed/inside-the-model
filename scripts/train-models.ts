/** Offline model pipeline. Bundle with esbuild and run with Node; no browser/server required.
 * Day 9 (tiny-lm*), Day 10 (house, scaling, icl, icl-sweep, facts) and Day 14 (prompt-model) jobs; later days add theirs here.
 * npx esbuild scripts/train-models.ts --bundle --platform=node --format=esm --outfile="$env:TEMP/train-models.mjs"
 * node "$env:TEMP/train-models.mjs" tiny-lm [--steps 6000 --seed 9 --batch 16 --lr 0.003]
 * node "$env:TEMP/train-models.mjs" day9   # tiny-lm plus tiny-lm-nomask
 * node "$env:TEMP/train-models.mjs" report [--write] # recompute reports from shipped bytes
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { createRng, type Rng } from '../src/lib/math/rng';
import { VOCAB, sampleSentence, encodeWords, decodeIds } from '../src/lib/transformer/grammar';
import { TINY_CONFIG, initWeights, generate, forward, prefill, decodeStep, parameterLayout, type Config, type Weights } from '../src/lib/transformer/model';
import { sinCos } from '../src/lib/transformer/math';
import { lossAndGrad, makeAdam, adamStep, type Example } from '../src/lib/transformer/train';
import { CHARSET, VOCAB as LM_VOCAB, encode, decode } from '../src/lib/lm/charset';
import { generate as lmGenerate } from '../src/lib/lm/sample';
import { encodeWeights, parseModel, toFloat16, type DType, type Manifest } from '../src/lib/transformer/load';
import { evaluate, probes, residualGradientProbe } from '../src/lib/transformer/report';
import * as PM from '../src/lib/applied/promptmodel';

const args = process.argv.slice(2), job = args[0] ?? 'day9';
const option = (name: string, fallback: number) => {
  const index = args.indexOf(`--${name}`), value = index < 0 ? fallback : Number(args[index + 1]);
  if (!Number.isFinite(value) || (name === 'seed' ? value < 0 : value <= 0)) throw new Error(`Invalid --${name}`);
  return value;
};
const steps = option('steps', 6000), seed = option('seed', 9), batch = option('batch', 16), lr = option('lr', 0.003);
if (![steps, seed, batch].every(Number.isInteger)) throw new Error('Steps, seed and batch must be integers');
const output = resolve('public/models');
const sha256 = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

function report(w: Weights) {
  const full = evaluate(w), noPositions = evaluate(w, { noPositions: true }).grammaticalMass;
  const prefixOnly = evaluate(w, { causal: true });
  const ablations = full.heads.map(({ layer, head }) => ({ layer, head, grammaticalMass: evaluate(w, { ablate: [[layer, head]] }).grammaticalMass }));
  const prompt = encodeWords('[start] these small sheep');
  const generated = decodeIds(generate(w, prompt, { stopId: 1 }));
  const randomDeep = initWeights({ ...TINY_CONFIG, layers: 12 }, createRng(91));
  return { ...full, prefixOnlyGrammaticalMass: prefixOnly.grammaticalMass, prefixOnlyNextTokenLoss: prefixOnly.nextTokenLoss,
    noPositions, ablations, probes: probes(w), randomInit12BlockGradientNorms: residualGradientProbe(randomDeep), greedySample: generated };
}

function train(name: 'tiny-lm' | 'tiny-lm-nomask') {
  const rng = createRng(seed), w = initWeights({ ...TINY_CONFIG, causal: name === 'tiny-lm' }, rng), optimizer = makeAdam(w.params.length);
  const curve: { step: number; loss: number }[] = [], snapshots: { step: number; embedding: Float64Array }[] = [];
  const marks = new Set(Array.from({ length: 8 }, (_, i) => Math.round(i * steps / 7)));
  const snapshot = (step: number) => {
    if (name === 'tiny-lm' && marks.has(step)) {
      const e = w.offsets.embedding; snapshots.push({ step, embedding: w.params.slice(e.offset, e.offset + e.length) });
    }
  };
  console.log(`${name}: ${w.params.length} parameters, seed ${seed}, ${steps} updates, batch ${batch}`);
  snapshot(0);
  for (let step = 1; step <= steps; step++) {
    const examples = Array.from({ length: batch }, () => {
      const ids = sampleSentence(rng); return { ids: ids.slice(0, -1), targets: ids.slice(1) };
    });
    const result = lossAndGrad(w, examples);
    // Linear warmup then linear decay: only arithmetic, no schedule libm calls.
    const rate = lr * Math.min(1, step / 100) * Math.max(0.15, 1 - 0.85 * step / steps);
    adamStep(w.params, result.grad, optimizer, rate); snapshot(step);
    if (step === 1 || step % Math.max(1, Math.floor(steps / 60)) === 0 || step === steps) curve.push({ step, loss: result.loss });
    if (step % 250 === 0 || step === steps) console.log(`${name} ${step}/${steps}: loss ${result.loss.toFixed(4)}, grammatical mass ${evaluate(w).grammaticalMass.toFixed(6)}`);
  }
  const packed = encodeWeights(w);
  const manifest: Manifest = { version: 1, name, config: w.config, vocab: w.config.vocab, offsets: w.offsets,
    dtype: packed.dtype, scales: packed.scales, parameterCount: w.params.length, sha256: sha256(packed.bytes), metrics: {}, lossCurve: curve,
    training: { seed, steps, batch, lr, optimizer: 'Adam beta1=0.9 beta2=0.999 epsilon=1e-8; global gradient clip=1',
      schedule: '100-update warmup; linear decay to 0.15*lr', objective: 'sampled next-token cross-entropy, token-mean per batch',
      grammar: 'two adjectives per noun phrase, three total; 34 words, max 10 tokens', evaluationSeed: 92026,
      generator: 'scripts/train-models.ts', deterministicMath: 'arithmetic exp/tanh; Adam running bias products', node: process.version } };
  const rounded = parseModel(manifest, packed.bytes);
  manifest.metrics = report(rounded);
  console.log(JSON.stringify({ name, dtype: packed.dtype, bytes: packed.bytes.length, metrics: manifest.metrics }, null, 2));
  if (name === 'tiny-lm' && ((manifest.metrics.grammaticalMass as number) < 0.97 || (manifest.metrics.distinctHeads as { meanTotalVariation: number }).meanTotalVariation < 0.2)) {
    throw new Error('Acceptance failed after quantization: need mass >= 0.97 and two heads with mean attention-row TV >= 0.2; adjust seed/steps/size before shipping');
  }
  mkdirSync(output, { recursive: true });
  writeFileSync(resolve(output, `${name}.bin`), packed.bytes);
  writeFileSync(resolve(output, `${name}.json`), JSON.stringify(manifest, null, 2) + '\n');
  if (name === 'tiny-lm') {
    const columns = VOCAB.length * w.config.d, bytes = new Uint8Array(snapshots.length * columns * 2), view = new DataView(bytes.buffer);
    snapshots.forEach((s, index) => s.embedding.forEach((v, j) => view.setUint16((index * columns + j) * 2, toFloat16(v), true)));
    writeFileSync(resolve(output, 'tiny-lm-snapshots.bin'), bytes);
    writeFileSync(resolve(output, 'tiny-lm-snapshots.json'), JSON.stringify({ version: 1, name: 'tiny-lm-snapshots', dtype: 'float16', vocab: VOCAB,
      shape: [VOCAB.length, w.config.d], sha256: sha256(bytes), checkpoints: snapshots.map((s, i) => ({ step: s.step, byteOffset: i * columns * 2 })),
      training: manifest.training }, null, 2) + '\n');
  }
}


// ---------------------------------------------------------------------------------------------
// Day 10 (llms). Same core, same weight format, same arithmetic-exp determinism as Day 9.
// node train-models.mjs corpus <pg11.txt> <pg12.txt>  # rebuild scripts/data/alice.txt from Gutenberg files
// node train-models.mjs house | scaling [--widths 8,12] | icl | icl-sweep | facts | facts-idk | day10
// ---------------------------------------------------------------------------------------------
const opt = (name: string, fallback: number) => args.includes(`--${name}`) ? option(name, fallback) : fallback;
const r4 = (x: number) => Math.round(x * 1e4) / 1e4;
const writeJson = (name: string, value: unknown, pretty = false) => {
  const text = JSON.stringify(value, null, pretty ? 2 : undefined) + '\n';
  writeFileSync(resolve(output, `${name}.json`), text);
  console.log(`wrote public/models/${name}.json (${text.length} bytes)`);
};
/** Warmup then cosine decay to 10% of lr; cosine from the arithmetic sinCos, no libm. */
const cosineLr = (lr0: number, step: number, total: number, warmup: number) =>
  lr0 * Math.min(1, step / warmup) * (0.1 + 0.45 * (1 + sinCos(Math.PI * Math.min(1, step / total))[1]));
const nonEmbedding = (c: Config) => {
  const { offsets, size } = parameterLayout(c);
  return size - offsets.embedding.length - (offsets.position?.length ?? 0) - (offsets.unembedding?.length ?? 0) - offsets['logit.bias'].length;
};
/** Encode with the core; int8 row scales rounded to 6 significant digits to keep the manifest small.
 * Every metric is then measured on the decoded bytes, i.e. after quantisation. */
function ship(name: string, w: Weights, lossCurve: { step: number; loss: number }[], training: Record<string, unknown>,
  measure: (rounded: Weights) => Record<string, unknown>, accept: (metrics: Record<string, any>) => string | undefined, dtype?: DType): { rounded: Weights; manifest: Manifest; bytes: Uint8Array } {
  const packed = encodeWeights(w, dtype);
  const scales = packed.scales && Object.fromEntries(Object.entries(packed.scales).map(([k, v]) => [k, v.map((s) => Number(s.toPrecision(6)))]));
  const manifest: Manifest = { version: 1, name, config: w.config, vocab: w.config.vocab, offsets: w.offsets, dtype: packed.dtype, scales,
    parameterCount: w.params.length, sha256: sha256(packed.bytes), metrics: {}, lossCurve,
    training: { ...training, generator: 'scripts/train-models.ts', node: process.version,
      deterministicMath: 'arithmetic exp/tanh/sin/cos; Adam running bias products; mulberry32 createRng' } };
  const rounded = parseModel(manifest, packed.bytes);
  // Keep a recoverable candidate before the expensive lesson probes. A failed
  // teaching hypothesis must not discard training or publish unreviewed weights.
  if (name === 'prompt-model') {
    mkdirSync(resolve('tmp'), { recursive: true });
    writeFileSync(resolve('tmp/prompt-model-candidate.bin'), packed.bytes);
    writeFileSync(resolve('tmp/prompt-model-candidate.json'), JSON.stringify(manifest) + '\n');
  }
  manifest.metrics = measure(rounded);
  const problem = accept(manifest.metrics);
  if (problem) { console.log(JSON.stringify(manifest.metrics, null, 2)); throw new Error(name + ' acceptance failed (nothing written): ' + problem); }
  mkdirSync(output, { recursive: true });
  writeFileSync(resolve(output, `${name}.bin`), packed.bytes);
  writeFileSync(resolve(output, `${name}.json`), JSON.stringify(manifest) + '\n');
  console.log(`wrote public/models/${name}.bin (${packed.bytes.length} bytes, ${packed.dtype}, sha256 ${manifest.sha256})`);
  return { rounded, manifest, bytes: packed.bytes };
}
function adamTrain(w: Weights, total: number, lr0: number, warmup: number, batchAt: (step: number) => Example[],
  after?: (step: number, loss: number) => void): void {
  const state = makeAdam(w.params.length);
  for (let step = 1; step <= total; step++) {
    const result = lossAndGrad(w, batchAt(step));
    adamStep(w.params, result.grad, state, cosineLr(lr0, step, total, warmup));
    after?.(step, result.loss);
  }
}

// --- Corpus: Lewis Carroll's two Alice books (Project Gutenberg #11, #12; public domain) ---
const ALICE = resolve('scripts/data/alice.txt');
function prepareAlice(raw: string): string {
  const body = raw.replace(/\r/g, '').split(/\*\*\* START OF[^\n]*\n/)[1].split(/\*\*\* END OF/)[0];
  return body.replace(/\[Illustration[^\]]*\]/g, '').toLowerCase()
    .replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/[—–]/g, '-').replace(/æ/g, 'ae').replace(/_/g, '')
    .split(/\n\s*\n/)
    .map((paragraph) => [...paragraph.replace(/\s+/g, ' ')].filter((c) => c !== '\n' && CHARSET.includes(c)).join('').replace(/ +/g, ' ').trim())
    .filter(Boolean).join('\n');
}
/** 50 contiguous chunks; every 10th (indexes 9, 19, ...) is validation: a 90/10 split. */
function aliceSplit() {
  const ids = encode(readFileSync(ALICE, 'utf8')), size = Math.floor(ids.length / 50);
  const chunks = Array.from({ length: 50 }, (_, i) => ids.slice(i * size, i === 49 ? ids.length : (i + 1) * size));
  const train = chunks.filter((_, i) => i % 10 !== 9), valid = chunks.filter((_, i) => i % 10 === 9);
  /** Non-overlapping 65-token windows (64 inputs, 64 shifted targets) that never cross a chunk edge. */
  const windows = (parts: number[][]) => parts.flatMap((c) => Array.from({ length: Math.floor((c.length - 1) / 64) }, (_, k) => c.slice(k * 64, k * 64 + 65)));
  return { total: ids.length, trainChars: train.reduce((s, c) => s + c.length, 0), validChars: valid.reduce((s, c) => s + c.length, 0),
    train, valid, trainWindows: windows(train), validWindows: windows(valid) };
}
const toExample = (win: readonly number[]): Example => ({ ids: win.slice(0, -1), targets: win.slice(1) });
/** Token-mean next-character loss (nats) over windows; every position counts, contexts 1..64. */
function windowLoss(w: Weights, windows: readonly (readonly number[])[]): number {
  const V = w.config.vocab.length;
  let loss = 0, n = 0;
  for (const win of windows) {
    const f = forward(w, win.slice(0, -1));
    for (let t = 0; t < win.length - 1; t++) { loss -= Math.log(f.probs[t * V + win[t + 1]]); n++; }
  }
  return loss / n;
}
/** Unigram/bigram entropies of the training text, and their add-0.5 cross-entropies on validation. */
function ngramBaselines(train: number[][], valid: number[][], V: number) {
  const uni = new Float64Array(V), bi = new Float64Array(V * V);
  for (const c of train) c.forEach((id, i) => { uni[id]++; if (i) bi[c[i - 1] * V + id]++; });
  const n = uni.reduce((s, v) => s + v, 0), rowTotal = (a: number) => bi.subarray(a * V, a * V + V).reduce((s, v) => s + v, 0);
  let hUni = 0, hBi = 0, pairs = 0;
  for (let a = 0; a < V; a++) {
    if (uni[a]) hUni -= uni[a] / n * Math.log(uni[a] / n);
    const row = rowTotal(a);
    for (let b = 0; b < V; b++) if (bi[a * V + b]) { hBi -= bi[a * V + b] * Math.log(bi[a * V + b] / row); pairs += bi[a * V + b]; }
  }
  let xUni = 0, xBi = 0, m = 0;
  for (const c of valid) for (let i = 1; i < c.length; i++) {
    xUni -= Math.log((uni[c[i]] + 0.5) / (n + 0.5 * V));
    xBi -= Math.log((bi[c[i - 1] * V + c[i]] + 0.5) / (rowTotal(c[i - 1]) + 0.5 * V)); m++;
  }
  return { unigramEntropy: hUni, bigramEntropy: hBi / pairs, unigramValidCrossEntropy: xUni / m, bigramValidCrossEntropy: xBi / m, uniformLnV: Math.log(V) };
}

const HOUSE: Config = { vocab: LM_VOCAB, d: 36, heads: 2, layers: 2, context: 64, ff: 144, positions: 'learned', causal: true, norm: 'pre', tied: true };
const HOUSE_SAMPLE_PROMPT = 'alice was ';
const HOUSE_PROBES: [string, string][] = [['the q', 'u'], ['said alic', 'e'], ['the queen sh', 'o']];
const housePromptSample = (w: Weights, seed: number, n = 100) => {
  const r = createRng(seed), us = Array.from({ length: n }, () => r.next());
  return decode(lmGenerate(w, encode(HOUSE_SAMPLE_PROMPT), n, { temperature: 1 }, us).ids);
};
const nextProb = (w: Weights, context: string, next: string) => {
  const f = forward(w, encode(context)), V = w.config.vocab.length;
  return f.probs[(encode(context).length - 1) * V + encode(next)[0]];
};

function house() {
  const total = opt('steps', 6000), batchSize = opt('batch', 16), lr0 = opt('lr', 0.003), seed0 = opt('seed', 10);
  const data = aliceSplit(), rng = createRng(seed0), w = initWeights(HOUSE, rng);
  const baselines = ngramBaselines(data.train, data.valid, HOUSE.vocab.length);
  const N = w.params.length, Nne = nonEmbedding(HOUSE), tokensPerStep = batchSize * 64;
  console.log(`house: ${N} parameters (${Nne} non-embedding), ${data.trainChars} train / ${data.validChars} validation chars, ${data.trainWindows.length}/${data.validWindows.length} windows`);
  console.log('baselines (nats/char):', JSON.stringify(baselines));
  const curve: { step: number; loss: number }[] = [], checkpoints: Record<string, unknown>[] = [];
  const marks = [0, 25, 50, 100, 200, 400, 800, 1600, 3200, total];
  const record = (step: number) => {
    if (step % 50 === 0 || step === 10 || step === 25 || step === total) curve.push({ step, loss: r4(windowLoss(w, data.validWindows)) });
    if (marks.includes(step)) {
      checkpoints.push({ step, validLoss: curve.at(-1)!.loss, sample: housePromptSample(w, 7),
        probes: HOUSE_PROBES.map(([c, n]) => r4(nextProb(w, c, n))) });
      console.log(`house ${step}/${total}: valid ${curve.at(-1)!.loss}`, JSON.stringify(checkpoints.at(-1)));
    }
  };
  record(0);
  const t0 = Date.now();
  adamTrain(w, total, lr0, 200, () => Array.from({ length: batchSize }, () => {
    const c = data.train[Math.floor(rng.next() * data.train.length)], start = Math.floor(rng.next() * (c.length - 65));
    return toExample(c.slice(start, start + 65));
  }), (step, loss) => {
    record(step);
    if (step % 250 === 0) console.log(`house ${step}: train ${loss.toFixed(4)}, valid ${curve.at(-1)!.loss}, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  });
  const D = total * tokensPerStep;
  const training = { seed: seed0, steps: total, batch: batchSize, lr: lr0, tokensPerStep, tokens: D, epochs: r4(D / data.trainChars),
    schedule: '200-update warmup, cosine decay to 0.1*lr', optimizer: 'Adam beta1=0.9 beta2=0.999 eps=1e-8, global clip 1',
    objective: 'next-character cross-entropy on random 64-char windows of the training chunks',
    corpus: 'scripts/data/alice.txt; 50 contiguous chunks, every 10th (index 9, 19, ...) held out' };
  const { rounded, manifest } = ship('house', w, curve, training, (q) => {
    const validLoss = windowLoss(q, data.validWindows);
    // KV-cache decode against full forward on one validation window, every position.
    const win = data.validWindows[0].slice(0, 64), V = q.config.vocab.length, full = forward(q, win).logits;
    const pre = prefill(q, win.slice(0, 1));
    let maxDiff = 0;
    for (let i = 0; i < V; i++) maxDiff = Math.max(maxDiff, Math.abs(pre.logits[i] - full[i]));
    for (let t = 1; t < 64; t++) {
      const step = decodeStep(q, pre.cache, win[t]);
      for (let i = 0; i < V; i++) maxDiff = Math.max(maxDiff, Math.abs(step[i] - full[t * V + i]));
    }
    return { validLoss, validPerplexity: Math.exp(validLoss), floatValidLoss: windowLoss(w, data.validWindows), kvCacheMaxLogitDiff: maxDiff,
      parameters: N, nonEmbedding: Nne, tokens: D, compute: 6 * N * D, ...baselines };
  }, (m) => m.validLoss < m.bigramValidCrossEntropy - 0.3 && m.kvCacheMaxLogitDiff < 1e-9 ? undefined : 'need quantised validation loss clearly below the bigram baseline and cached decode equal to forward');
  const m = manifest.metrics as Record<string, number>;
  const prompts = ['alice was ', 'the queen said ', '"what is the use of ', 'q: who did alice follow down the hole? a:'];
  const continuations = prompts.map((p, i) => {
    const r = createRng(100 + i), us = Array.from({ length: 120 }, () => r.next());
    return { prompt: p, greedy: decode(lmGenerate(rounded, encode(p), 120, { temperature: 0 }, us).ids),
      sampled: decode(lmGenerate(rounded, encode(p), 120, { temperature: 1 }, us).ids),
      t08p095: decode(lmGenerate(rounded, encode(p), 120, { temperature: 0.8, topP: 0.95 }, us).ids) };
  });
  writeJson('house-log', { version: 1, model: 'house', every: 50, curve: curve.map((c) => [c.step, c.loss]), checkpoints,
    samplePrompt: HOUSE_SAMPLE_PROMPT, sampleSeed: 7, sampleTemperature: 1, probes: HOUSE_PROBES.map(([context, next]) => ({ context, next })),
    baselines: Object.fromEntries(Object.entries(baselines).map(([k, v]) => [k, r4(v)])),
    N, nonEmbedding: Nne, D, C: 6 * N * D, tokensPerStep, validLossQuantised: r4(m.validLoss) });
  console.log(JSON.stringify({ house: m, continuations }, null, 2));
}

// --- Scaling sweep: 7 widths × 6 token budgets, house architecture (ff = 4d), at most one epoch ---
const SCALING_WIDTHS = [8, 12, 16, 24, 36, 48, 64], SCALING_BATCH = opt('batch', 4);
const scalingDir = resolve(tmpdir(), 'day10-scaling');
const SCALING_LRS = [0.003, 0.006, 0.01];
interface ScalingRun { lr?: number; d: number; N: number; nonEmbedding: number; D: number; C: number; steps: number; validLoss: number; curve: [number, number][] }
/** Least squares y = m x + k with R². */
function linearFit(xs: number[], ys: number[]) {
  const n = xs.length, mx = xs.reduce((s, x) => s + x, 0) / n, my = ys.reduce((s, y) => s + y, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; syy += (ys[i] - my) ** 2; });
  const m = sxy / sxx, k = my - m * mx;
  return { m, k, r2: syy ? sxy * sxy / (sxx * syy) : 1 };
}
/** L(C) = E + a C^(-b): grid search E, log-log least squares for a and b; best E minimises squared error in L. */
function powerFit(Cs: number[], Ls: number[]) {
  let best = { E: 0, a: 0, b: 0, r2: 0, sse: Infinity };
  const lo = Math.min(...Ls);
  for (let E = 0; E < lo - 1e-3; E += 0.005) {
    const f = linearFit(Cs.map(Math.log), Ls.map((L) => Math.log(L - E)));
    const a = Math.exp(f.k), b = -f.m, sse = Cs.reduce((s, C, i) => s + (E + a * C ** -b - Ls[i]) ** 2, 0);
    if (sse < best.sse) best = { E: r4(E), a, b, r2: f.r2, sse };
  }
  return best;
}
function scalingReport(runs: ScalingRun[]) {
  const finals = runs.map((r) => ({ C: r.C, L: r.validLoss, d: r.d, D: r.D })).sort((a, b) => a.C - b.C);
  const frontier = finals.filter((p, i) => finals.slice(0, i).every((q) => q.L > p.L));
  const compute = powerFit(frontier.map((p) => p.C), frontier.map((p) => p.L));
  const pure = linearFit(frontier.map((p) => Math.log(p.C)), frontier.map((p) => Math.log(p.L)));
  // IsoFLOP: per width, interpolate log L against log C between its measured budgets; minimise over log N with a parabola.
  const widths = [...new Set(runs.map((r) => r.d))].sort((a, b) => a - b);
  const byWidth = widths.map((d) => runs.filter((r) => r.d === d).sort((a, b) => a.C - b.C));
  // Budgets span the full measured range; each uses only the widths whose six runs bracket it (at least 3).
  const lcMin = Math.log(Math.min(...byWidth.map((rs) => rs[0].C))), lcMax = Math.log(Math.max(...byWidth.map((rs) => rs.at(-1)!.C)));
  const isoflop = [];
  for (let i = 0; i < 9; i++) {
    const lc = lcMin + (lcMax - lcMin) * (i + 0.5) / 9, C = Math.exp(lc);
    const covering = byWidth.filter((rs) => Math.log(rs[0].C) <= lc && lc <= Math.log(rs.at(-1)!.C));
    if (covering.length < 3) continue;
    const points = covering.map((rs) => {
      const j = rs.findIndex((r) => Math.log(r.C) >= lc), lo = rs[Math.max(0, j - 1)], hi = rs[j];
      const f = hi === lo ? 0 : (lc - Math.log(lo.C)) / (Math.log(hi.C) - Math.log(lo.C));
      return { N: rs[0].N, logN: Math.log(rs[0].N), L: Math.exp(Math.log(lo.validLoss) + f * (Math.log(hi.validLoss) - Math.log(lo.validLoss))) };
    });
    let k = points.reduce((b, p, j) => p.L < points[b].L ? j : b, 0);
    let logNopt = points[k].logN;
    if (k > 0 && k < points.length - 1) {
      const [x0, x1, x2] = [points[k - 1].logN, points[k].logN, points[k + 1].logN], [y0, y1, y2] = [points[k - 1].L, points[k].L, points[k + 1].L];
      const den = (x0 - x1) * (x0 - x2) * (x1 - x2), A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den, B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den;
      if (A > 0) logNopt = Math.min(x2, Math.max(x0, -B / (2 * A)));
    } else k = -1; // minimum at the edge of the measured widths: not an interior optimum
    const Nopt = Math.exp(logNopt), Dopt = C / (6 * Nopt);
    isoflop.push({ C, Nopt, Dopt, tokensPerParam: Dopt / Nopt, interior: k >= 0, N: points.map((p) => p.N), losses: points.map((p) => r4(p.L)) });
  }
  const interior = isoflop.filter((p) => p.interior);
  const nFit = interior.length >= 2 ? linearFit(interior.map((p) => Math.log(p.C)), interior.map((p) => Math.log(p.Nopt))) : undefined;
  return { runs: runs.length, frontier: frontier.map((p) => ({ C: p.C, L: r4(p.L), d: p.d, D: p.D })),
    computeFit: { E: compute.E, a: compute.a, b: r4(compute.b), r2: r4(compute.r2) }, pureLogLog: { b: r4(-pure.m), r2: r4(pure.r2) },
    isoflop: isoflop.map((p) => ({ ...p, Nopt: Math.round(p.Nopt), Dopt: Math.round(p.Dopt), tokensPerParam: r4(p.tokensPerParam) })),
    nOptExponent: nFit && r4(nFit.m), dOptExponent: nFit && r4(1 - nFit.m), isoflopWidthsN: byWidth.map((rs) => rs[0].N) };
}
function scaling() {
  const data = aliceSplit(), order = data.trainWindows.slice(), shuffle = createRng(1010);
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(shuffle.next() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const W = order.length - order.length % SCALING_BATCH, budgets = [32, 16, 8, 4, 2, 1].map((k) => Math.floor(W / k / SCALING_BATCH) * SCALING_BATCH);
  const curveSet = data.validWindows.slice(0, 32), lrs = args.includes('--lr') ? [opt('lr', 0.01)] : SCALING_LRS, dir = (lr: number) => resolve(scalingDir, `lr${lr}`);
  const widthsArg = args.indexOf('--widths'), widths = args.includes('--assemble') ? [] : widthsArg < 0 ? SCALING_WIDTHS : args[widthsArg + 1].split(',').map(Number);
  for (const lr0 of lrs) for (const d of widths) {
    mkdirSync(dir(lr0), { recursive: true });
    const runs: ScalingRun[] = [];
    for (const windows of budgets) {
      const cfg: Config = { ...HOUSE, d, ff: 4 * d }, w = initWeights(cfg, createRng(2000 + d)), total = windows / SCALING_BATCH;
      const marks = new Set(Array.from({ length: 20 }, (_, i) => Math.max(1, Math.round(total ** (i / 19)))));
      const curve: [number, number][] = [], t0 = Date.now();
      adamTrain(w, total, lr0, Math.max(1, Math.min(50, Math.round(total / 10))),
        (step) => order.slice((step - 1) * SCALING_BATCH, step * SCALING_BATCH).map(toExample),
        (step) => { if (marks.has(step)) curve.push([step, r4(windowLoss(w, curveSet))]); });
      const N = w.params.length, D = windows * 64;
      runs.push({ d, N, nonEmbedding: nonEmbedding(cfg), D, C: 6 * N * D, steps: total, validLoss: r4(windowLoss(w, data.validWindows)), curve });
      console.log(`scaling lr=${lr0} d=${d} N=${N} D=${D}: valid ${runs.at(-1)!.validLoss} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
    }
    writeFileSync(resolve(dir(lr0), `d${d}.json`), JSON.stringify(runs));
  }
  // Each (width, budget) keeps the best of the learning rates in SCALING_LRS, as a per-run tuned sweep would.
  const parts = SCALING_LRS.flatMap((lr) => SCALING_WIDTHS.map((d) => resolve(dir(lr), `d${d}.json`)));
  if (!parts.every(existsSync)) { console.log('scaling: partial results in', scalingDir, '- run every width at --lr', SCALING_LRS.join(', '), 'to assemble'); return; }
  const candidates = SCALING_LRS.map((lr) => SCALING_WIDTHS.flatMap((d) => (JSON.parse(readFileSync(resolve(dir(lr), `d${d}.json`), 'utf8')) as ScalingRun[]).map((r) => ({ ...r, lr }))));
  const runs: ScalingRun[] = candidates[0].map((_, i) => candidates.map((c) => c[i]).reduce((a, b) => b.validLoss < a.validLoss ? b : a));
  // Ship about 14 of the 20 logged curve points per run (first and last kept) to stay inside the 15 KB budget.
  for (const r of runs) r.curve = r.curve.filter((_, i, all) => i % 3 !== 1 || i === all.length - 1);
  const report = scalingReport(runs);
  writeJson('scaling-sweep', { version: 1, architecture: 'house: 2 layers, 2 heads, ff = 4d, context 64, learned positions, pre-LN, tied; char vocab 38',
    widths: SCALING_WIDTHS, batchTokens: SCALING_BATCH * 64, lr: `best of ${SCALING_LRS.join(', ')} per run (field lr)`, schedule: 'warmup min(50, 10% of steps), cosine to 0.1*lr over each run',
    data: 'shuffled non-overlapping 64-char training windows, each budget a prefix of the same order; largest = one epoch', C: '6 * N(total) * D',
    curveEval: 'curve losses on the first 32 validation windows (2048 tokens); validLoss on all validation windows',
    runs, fit: report });
  console.log(JSON.stringify(report, null, 2));
}

// --- In-context learning: per-sequence shift cipher on a–h, pairs written "c>e" ---
const ICL_LETTERS = 'abcdefgh', ICL_VOCAB = [...ICL_LETTERS, '>', ' ', '\n', '<unk>'];
const iclEncode = (s: string) => [...s].map((c) => { const i = ICL_VOCAB.indexOf(c); return i < 0 || c === '<' ? 11 : i; });
const shiftWord = (word: string, s: number) => [...word].map((c) => ICL_LETTERS[(ICL_LETTERS.indexOf(c) + s) % 8]).join('');
const randomWord = (rng: Rng, n: number) => Array.from({ length: n }, () => ICL_LETTERS[Math.floor(rng.next() * 8)]).join('');
/** One training sequence: a hidden shift s in 0..7, then single-letter "src>dst " pairs until the context is full
 * (12 pairs). Loss only on answer letters: sources are random and carry nothing to learn. */
function iclSequence(rng: Rng, context: number): Example {
  const s = Math.floor(rng.next() * 8), ids: number[] = [], targets: number[] = [];
  while (ids.length + 4 <= context + 1) {
    const src = ICL_LETTERS[Math.floor(rng.next() * 8)];
    iclEncode(`${src}>${shiftWord(src, s)} `).forEach((id, j) => { ids.push(id); targets.push(j === 2 ? id : -1); });
  }
  // Inputs are ids[0..n-2]; the target at position t is ids[t+1] when that token is an answer letter.
  return { ids: ids.slice(0, -1), targets: targets.slice(1) };
}
interface IclTask { prompt: number[]; answer: number[]; answerAt: number[]; queryIndex: number }
/** k single-letter examples, then one query letter (or, with `word`, four query pairs that spell a 4-letter word, the
 * earlier answers written in as in teacher forcing). Options: random example answers, another separator or gap,
 * the last example wrong, or examples that avoid the query letter (novel). answerAt = positions predicting answers. */
function iclTask(rng: Rng, k: number, o: { word?: boolean; random?: boolean; sep?: string; gap?: string; wrong?: boolean; novel?: boolean } = {}): IclTask {
  const s = Math.floor(rng.next() * 8), query = randomWord(rng, o.word ? 4 : 1), sep = o.sep ?? '>', gap = o.gap ?? ' ';
  let text = '';
  for (let i = 0; i < k; i++) {
    let src = randomWord(rng, 1);
    while (o.novel && query.includes(src)) src = randomWord(rng, 1);
    const dst = o.random ? randomWord(rng, 1) : o.wrong && i === k - 1 ? shiftWord(src, (s + 1 + Math.floor(rng.next() * 7)) % 8) : shiftWord(src, s);
    text += `${src}${sep}${dst}${gap}`;
  }
  const answer = iclEncode(shiftWord(query, s)), answerAt: number[] = [];
  [...query].forEach((q, j) => { text += `${q}${sep}`; answerAt.push(text.length - 1); if (j < query.length - 1) text += `${shiftWord(q, s)}${gap}`; });
  return { prompt: iclEncode(text), answer, answerAt, queryIndex: answerAt[0] };
}
/** Teacher-forced: per-letter argmax hits, summed answer log-prob, exact = every letter right (then greedy agrees). */
function iclScore(w: Weights, task: IclTask) {
  const f = forward(w, task.prompt), V = ICL_VOCAB.length;
  let hits = 0, logp = 0;
  task.answer.forEach((a, j) => {
    const row = f.probs.subarray(task.answerAt[j] * V, (task.answerAt[j] + 1) * V);
    let best = 0; for (let i = 1; i < V; i++) if (row[i] > row[best]) best = i;
    hits += best === a ? 1 : 0; logp += Math.log(row[a]);
  });
  return { hits, letters: task.answer.length, logp, exact: hits === task.answer.length ? 1 : 0, p0: f.probs[task.queryIndex * V + task.answer[0]] };
}
function iclEval(w: Weights, k: number, o: Parameters<typeof iclTask>[2] = {}, n = 500, seed = 4242) {
  const rng = createRng(seed + 97 * k);
  let hits = 0, letters = 0, logp = 0, exact = 0, p0 = 0;
  for (let i = 0; i < n; i++) { const r = iclScore(w, iclTask(rng, k, o)); hits += r.hits; letters += r.letters; logp += r.logp; exact += r.exact; p0 += r.p0; }
  return { accuracy: r4(hits / letters), exact: r4(exact / n), meanLogp: r4(logp / n), loss: r4(-logp / letters), meanP: r4(p0 / n) };
}
/** Where each head looks from the query's '>' (the position that predicts the answer), averaged over 200 seeded
 * 4-example single-letter prompts. Mass by token type: example sources / '>' / answers / spaces, the query letter, and
 * '>' itself. match = attention on the answer of the one earlier example whose source equals the query letter (the
 * textbook induction target; prompts with exactly one such example). Positions: example i = source 4i, '>' 4i+1,
 * answer 4i+2, space 4i+3; then query letter 16 and '>' 17. Arrays are per head, index = layer * heads + head. */
const ATT_TYPES = ['sources', 'arrows', 'answers', 'spaces', 'query', 'self'] as const;
function attentionProfile(w: Weights, n = 200) {
  const rng = createRng(777), H = w.config.layers * w.config.heads;
  const mass = ATT_TYPES.map(() => new Float64Array(H)), match = new Float64Array(H);
  let matched = 0;
  for (let i = 0; i < n; i++) {
    const t = iclTask(rng, 4), q = t.prompt[t.queryIndex - 1], f = forward(w, t.prompt), T = t.prompt.length;
    const matches = [0, 1, 2, 3].filter((e) => t.prompt[e * 4] === q);
    if (matches.length === 1) matched++;
    f.layers.forEach((l, li) => l.weights.forEach((a, h) => {
      const row = a.subarray(t.queryIndex * T, (t.queryIndex + 1) * T), k = li * w.config.heads + h;
      for (let p = 0; p < 16; p++) mass[p % 4][k] += row[p];
      mass[4][k] += row[16]; mass[5][k] += row[17];
      if (matches.length === 1) match[k] += row[matches[0] * 4 + 2];
    }));
  }
  const avg = (a: Float64Array, m: number) => Array.from(a, (v) => r4(v / m));
  return { ...Object.fromEntries(ATT_TYPES.map((name, j) => [name, avg(mass[j], n)])), match: avg(match, matched) } as Record<typeof ATT_TYPES[number] | 'match', number[]>;
}
const ICL_DEFAULT_PROMPT = 'c>f g>b a>d e>h g>';
function iclTrain(d: number, ff: number, total: number, seed0: number, log?: (step: number, w: Weights) => void): Weights {
  const cfg: Config = { vocab: ICL_VOCAB, d, heads: 2, layers: 2, context: 50, ff, positions: 'rope', causal: true, norm: 'pre', tied: true };
  const rng = createRng(seed0), w = initWeights(cfg, rng), batchSize = 32;
  log?.(0, w);
  adamTrain(w, total, opt('lr', 0.003), 100, () => Array.from({ length: batchSize }, () => iclSequence(rng, cfg.context)), (step) => log?.(step, w));
  return w;
}
function icl() {
  const d = opt('d', 24), ff = opt('ff', 24), total = opt('steps', 4000), seed0 = opt('seed', 11), every = opt('every', 50);
  const log: Record<string, number>[] = [], profiles: ReturnType<typeof attentionProfile>[] = [], thumbs: { step: number; weights: number[][] }[] = [];
  const t0 = Date.now();
  const w = iclTrain(d, ff, total, seed0, (step, w) => {
    if (step % every) return;
    const zero = iclEval(w, 0, {}, 200), four = iclEval(w, 4, {}, 200), novel = iclEval(w, 4, { novel: true }, 200), word = iclEval(w, 4, { word: true }, 100);
    const att = attentionProfile(w, 100);
    log.push({ step, loss0: zero.loss, loss4: four.loss, acc0: zero.accuracy, acc4: four.accuracy, acc4novel: novel.accuracy, word4: word.accuracy, induction: Math.max(...att.match) });
    profiles.push(att);
    const f = forward(w, iclEncode(ICL_DEFAULT_PROMPT)), T = f.ids.length;
    // Lower triangle only (causal), percent.
    thumbs.push({ step, weights: f.layers.flatMap((l) => l.weights.map((a) => Array.from({ length: T * (T + 1) / 2 }, (_, i) => { const r = Math.floor((Math.sqrt(8 * i + 1) - 1) / 2), c = i - r * (r + 1) / 2; return Math.round(a[r * T + c] * 100); }))) });
    console.log(`icl ${step}/${total} (${((Date.now() - t0) / 1000).toFixed(0)} s)`, JSON.stringify(log.at(-1)), JSON.stringify(att));
  });
  const measure = (q: Weights) => ({
    shots: [0, 1, 2, 3, 4, 6].map((k) => ({ k, ...iclEval(q, k) })),
    novelShots: [1, 2, 4, 6].map((k) => ({ k, ...iclEval(q, k, { novel: true }) })),
    randomPairs4: iclEval(q, 4, { random: true }), oneWrong4: iclEval(q, 4, { wrong: true }),
    formats4: { trained: iclEval(q, 4), equalsSeparator: iclEval(q, 4, { sep: '=' }), newlineGap: iclEval(q, 4, { gap: '\n' }) },
    fourLetter4: iclEval(q, 4, { word: true }), attention: attentionProfile(q),
  });
  const { manifest } = ship('icl', w, log.map((l) => ({ step: l.step, loss: l.loss4 })),
    { seed: seed0, steps: total, batch: 32, d, ff, lr: opt('lr', 0.003), every,
      task: 'per-sequence shift s in 0..7 on letters a-h; "src>dst " pairs fill a 50-token context (12 single-letter pairs); loss on answer letters only',
      vocabNote: 'ids 0-7 letters a-h, 8 ">", 9 space, 10 newline, 11 <unk> (any other symbol, e.g. "=")', evaluation: 'iclEval: 500 seeded tasks per condition, k single-letter examples then a query' },
    measure, (m) => {
      const s = m.shots as { k: number; accuracy: number }[], novel = m.novelShots as { k: number; accuracy: number }[];
      return s[4].accuracy > 0.8 && s[0].accuracy < 0.25 && novel[2].accuracy > 0.8 ? undefined : 'need 4-shot accuracy > 0.8 (also on query letters absent from the examples) and 0-shot near chance';
    }, 'int8');
  // The head shown: the layer-2 head whose type profile moved most (total variation) between the first and last log.
  const tv = (a: typeof profiles[0], b: typeof profiles[0], k: number) => ATT_TYPES.reduce((s, t) => s + Math.abs(a[t][k] - b[t][k]), 0) / 2;
  const heads = w.config.heads, layer2 = Array.from({ length: heads }, (_, h) => heads + h), best = layer2.reduce((b, k) => tv(profiles[0], profiles.at(-1)!, k) > tv(profiles[0], profiles.at(-1)!, b) ? k : b);
  const cross = (a: number) => Math.max(0, log.findIndex((l) => l.acc4 >= a));
  writeJson('icl-log', { version: 1, every, defaultPrompt: ICL_DEFAULT_PROMPT, shownHead: { layer: Math.floor(best / heads), head: best % heads },
    // Every log until 4-shot accuracy first reaches 0.99, then every 5th (and the last), to stay near the 15 KB budget.
    log: log.map((l, i) => ({ ...l, profile: Object.fromEntries([...ATT_TYPES, 'match' as const].map((t) => [t, profiles[i][t][best]])) }))
      .filter((_, i) => i <= cross(0.99) || i % 5 === 0 || i === log.length - 1),
    // Six thumbnails: start, the last log before 4-shot accuracy reaches 0.25, the first logs at 0.25, 0.5 and 0.9, and the end.
    thumbnails: [...new Set([0, Math.max(0, cross(0.25) - 1), cross(0.25), cross(0.5), cross(0.9), log.length - 1])].map((i) => ({ step: thumbs[i].step, acc4: log[i].acc4, weights: thumbs[i].weights[best] })),
    thumbnailFormat: 'causal attention of the shown head on defaultPrompt; lower triangle row by row (row r has r+1 entries), percent' });
  console.log(JSON.stringify(manifest.metrics, null, 2));
}
/** About 6 sizes, same data/steps/seed; ff = d. Each size can run in its own process (--sizes 8,12); the last one assembles. */
function iclSweep() {
  const sizes = [8, 12, 16, 24, 32, 48], total = opt('steps', 8000), dir = resolve(tmpdir(), 'day10-icl-sweep');
  const i = args.indexOf('--sizes'), mine = args.includes('--assemble') ? [] : i < 0 ? sizes : args[i + 1].split(',').map(Number);
  mkdirSync(dir, { recursive: true });
  for (const d of mine) {
    const w = iclTrain(d, d, total, opt('seed', 11)), r = iclEval(w, 4, { word: true }), one = iclEval(w, 4);
    const row = { d, N: w.params.length, exact: r.exact, perLetter: r.accuracy, meanLogp: r.meanLogp, perLetterPow4: r4(r.accuracy ** 4), singleLetter4: one.accuracy };
    writeFileSync(resolve(dir, `d${d}.json`), JSON.stringify(row)); console.log('icl-sweep', JSON.stringify(row));
  }
  const files = sizes.map((d) => resolve(dir, `d${d}.json`));
  if (!files.every(existsSync)) { console.log('icl-sweep: partial results in', dir); return; }
  writeJson('icl-sweep', { version: 1, steps: total, ff: 'd', seed: opt('seed', 11), condition: '4 single-letter examples, then a 4-letter word asked one letter per pair (earlier answers written in); 500 seeded tasks. exact = all 4 letters right (argmax); perLetter = argmax accuracy; meanLogp = sum over the 4 answer letters (nats)',
    rows: files.map((f) => JSON.parse(readFileSync(f, 'utf8'))) });
}

// --- Facts: invented people and birth months; frequency in training decides what is learned ---
const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
const FACTS_VOCAB = [...'abcdefghijklmnopqrstuvwxyz .', ...MONTHS.map((m) => `<${m}>`), '<unknown>'];
const MONTH0 = 28, UNKNOWN_ID = 40;
const FACT_COUNTS: [number, number][] = [[20, 10], [10, 20], [5, 30], [2, 40], [1, 50], [0, 50]]; // [occurrences, people]
const factsEncode = (s: string) => [...s].map((c) => { const i = FACTS_VOCAB.indexOf(c); if (i < 0 || i >= MONTH0) throw new Error(`Not in facts charset: ${c}`); return i; });
/** Invented two-part names from seeded syllables (consonant-vowel, optional final consonant). */
function inventName(rng: Rng): string {
  const C = 'bdfgklmnprstvz', Vw = 'aeiou', pick = (s: string) => s[Math.floor(rng.next() * s.length)];
  const part = (n: number, end: number) => Array.from({ length: n }, () => pick(C) + pick(Vw)).join('') + (rng.next() < end ? pick('lnrsv') : '');
  return `${part(2, 0.3)} ${part(2, 0.6)}`;
}
function factsWorld() {
  const rng = createRng(1212), used = new Set<string>(), fresh = () => { let n; do n = inventName(rng); while (used.has(n)); used.add(n); return n; };
  const prior = MONTHS.map(() => 0.5 + rng.next() * 1.5), total = prior.reduce((s, v) => s + v, 0);
  const drawMonth = (r: Rng) => { let u = r.next() * total, m = 0; while (u >= prior[m] && m < 11) u -= prior[m++]; return m; };
  const people = FACT_COUNTS.flatMap(([count, n]) => Array.from({ length: n }, () => ({ name: fresh(), month: drawMonth(rng), count })));
  const copyNames = Array.from({ length: 3000 }, fresh); // context-copy lines only; never asked about
  const idkNames = Array.from({ length: 100 }, fresh); // facts-idk only: trained to answer <unknown>
  const probeNames = Array.from({ length: 50 }, fresh); // never in any training line
  return { people, copyNames, idkNames, probeNames, drawMonth };
}
const factLine = (name: string, month: number | 'unknown') => [...factsEncode(`${name} was born in `), month === 'unknown' ? UNKNOWN_ID : MONTH0 + month, factsEncode('.')[0]];
const askIds = (name: string, context?: [string, number]) => [...(context ? [...factLine(...context), factsEncode(' ')[0]] : []), ...factsEncode(`${name} was born in `)];
/** p over the 13 answer tokens (12 months + unknown) after "<name> was born in ", optionally after a stated line. */
function answerDist(w: Weights, name: string, context?: [string, number]) {
  const ids = askIds(name, context), f = forward(w, ids), V = FACTS_VOCAB.length, row = f.probs.subarray((ids.length - 1) * V, ids.length * V);
  const months = Array.from({ length: 12 }, (_, m) => row[MONTH0 + m]), mass = months.reduce((s, v) => s + v, 0);
  return { months: months.map((p) => p / mass), monthMass: mass, unknown: row[UNKNOWN_ID] };
}
function factsMeasure(w: Weights, world: ReturnType<typeof factsWorld>) {
  const argmax = (p: number[]) => p.reduce((b, v, i) => v > p[b] ? i : b, 0);
  const rows = world.people.map((p) => { const a = answerDist(w, p.name), pick = argmax(a.months); return { ...p, pick, conf: a.months[pick], correct: pick === p.month, unknown: a.unknown, monthMass: a.monthMass, dist: a.months }; });
  const byCount = FACT_COUNTS.map(([count]) => { const r = rows.filter((x) => x.count === count); return { count, people: r.length, accuracy: r4(r.filter((x) => x.correct).length / r.length), meanConfidence: r4(r.reduce((s, x) => s + x.conf, 0) / r.length) }; });
  const bins = Array.from({ length: 10 }, (_, b) => { const r = rows.filter((x) => Math.min(9, Math.floor(x.conf * 10)) === b); return { bin: b, n: r.length, accuracy: r.length ? r4(r.filter((x) => x.correct).length / r.length) : null, confidence: r.length ? r4(r.reduce((s, x) => s + x.conf, 0) / r.length) : null }; });
  const ece = bins.reduce((s, b) => s + (b.n ? b.n / rows.length * Math.abs(b.accuracy! - b.confidence!) : 0), 0);
  const unseen = rows.filter((x) => x.count === 0), probe = world.probeNames.map((n) => answerDist(w, n));
  const freq = MONTHS.map((_, m) => world.people.reduce((s, p) => s + (p.month === m ? p.count : 0), 0)), fTotal = freq.reduce((s, v) => s + v, 0);
  const avg = MONTHS.map((_, m) => unseen.reduce((s, x) => s + x.dist[m], 0) / unseen.length);
  const tv = avg.reduce((s, v, m) => s + Math.abs(v - freq[m] / fTotal), 0) / 2;
  const wrongMonth = (m: number) => (m + 5) % 12;
  const ctx = (wrong: boolean) => unseen.map((x) => { const a = answerDist(w, x.name, [x.name, wrong ? wrongMonth(x.month) : x.month]); return argmax(a.months); });
  const right = ctx(false), wrong = ctx(true);
  const coverage = [0, 0.3, 0.5, 0.7, 0.9].map((t) => { const ans = rows.filter((x) => x.conf >= t); return { threshold: t, coverage: r4(ans.length / rows.length), error: ans.length ? r4(ans.filter((x) => !x.correct).length / ans.length) : null }; });
  return {
    byCount, onceShare: r4(world.people.filter((p) => p.count === 1).length / world.people.filter((p) => p.count > 0).length),
    accuracy: r4(rows.filter((x) => x.correct).length / rows.length), ece: r4(ece), reliability: bins,
    unseen: { meanConfidence: r4(unseen.reduce((s, x) => s + x.conf, 0) / unseen.length), accuracy: r4(unseen.filter((x) => x.correct).length / unseen.length),
      meanUnknownProb: unseen.reduce((s, x) => s + x.unknown, 0) / unseen.length, averageDist: avg.map(r4), tvToTrainingFrequencies: r4(tv) },
    probeNames: { meanUnknownProb: probe.reduce((s, a) => s + a.unknown, 0) / probe.length, unknownTop: probe.filter((a) => a.unknown > Math.max(...a.months) * a.monthMass).length },
    answeredUnknown: { known: r4(rows.filter((x) => x.count > 0 && x.unknown > Math.max(...x.dist) * x.monthMass).length / rows.filter((x) => x.count > 0).length),
      unseen: r4(unseen.filter((x) => x.unknown > Math.max(...x.dist) * x.monthMass).length / unseen.length), probeNames: r4(probe.filter((a) => a.unknown > Math.max(...a.months) * a.monthMass).length / probe.length) },
    trainingMonthFrequencies: freq.map((f) => r4(f / fTotal)), coverage,
    retrieval: { noContext: r4(unseen.filter((x) => x.correct).length / unseen.length), correctContext: r4(right.filter((m, i) => m === unseen[i].month).length / unseen.length),
      wrongContext: r4(wrong.filter((m, i) => m === unseen[i].month).length / unseen.length), copiesWrong: r4(wrong.filter((m, i) => m === wrongMonth(unseen[i].month)).length / unseen.length) },
    rows,
  };
}
function trainFacts(world: ReturnType<typeof factsWorld>, epochs: number, idk: boolean, log?: (epoch: number, w: Weights) => void): Weights {
  const cfg: Config = { ...HOUSE, vocab: FACTS_VOCAB }, rng = createRng(opt('seed', 13)), w = initWeights(cfg, rng);
  const base = world.people.flatMap((p) => Array.from({ length: p.count }, () => factLine(p.name, p.month)));
  if (idk) for (const n of world.idkNames) for (let i = 0; i < 3; i++) base.push(factLine(n, 'unknown'));
  const copies = Math.round(base.length * 0.3 / 0.7), batchSize = 16;
  let copyIndex = 0;
  const epochLines = () => {
    const lines = [...base, ...Array.from({ length: copies }, () => {
      const name = world.copyNames[copyIndex++ % world.copyNames.length], m = world.drawMonth(rng);
      return [...factLine(name, m), factsEncode(' ')[0], ...factLine(name, m)];
    })];
    for (let i = lines.length - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); [lines[i], lines[j]] = [lines[j], lines[i]]; }
    return lines;
  };
  const perEpoch = Math.ceil((base.length + copies) / batchSize), total = epochs * perEpoch;
  let lines: number[][] = [];
  log?.(0, w);
  adamTrain(w, total, opt('lr', 0.003), 100, (step) => {
    const k = (step - 1) % perEpoch;
    if (k === 0) lines = epochLines();
    return lines.slice(k * batchSize, (k + 1) * batchSize).map(toExample);
  }, (step) => { if (step % perEpoch === 0) log?.(step / perEpoch, w); });
  return w;
}
function facts(idk = false) {
  const world = factsWorld(), epochs = opt('epochs', 25), every = opt('every', 5), curve: { step: number; loss: number }[] = [];
  const w = trainFacts(world, epochs, idk, (epoch, w) => {
    if (epoch % every) return;
    const m = factsMeasure(w, world);
    curve.push({ step: epoch, loss: m.accuracy });
    console.log(`facts${idk ? '-idk' : ''} epoch ${epoch}:`, JSON.stringify({ byCount: m.byCount.map((b) => b.accuracy), ece: m.ece, answeredUnknown: m.answeredUnknown, unseenConf: m.unseen.meanConfidence, retrieval: m.retrieval, probeUnknown: m.probeNames }));
  });
  const name = idk ? 'facts-idk' : 'facts';
  const { manifest } = ship(name, w, curve, { epochs, seed: opt('seed', 13), lossCurveMeaning: 'step = epoch, loss = overall accuracy (not a loss)',
    data: 'per epoch: each fact line repeated by its count, plus 30% context-copy lines ("x was born in m. x was born in m.") with fresh invented names' + (idk ? '; plus 100 held-out names x 3 lines answering <unknown>' : '') },
  (q) => { const { rows, ...rest } = factsMeasure(q, world); return rest; },
  (m) => {
    if (idk) return undefined;
    const acc = (m.byCount as { accuracy: number }[]).map((b) => b.accuracy);
    return acc[0] > acc[3] && acc[3] >= acc[4] && acc[0] >= 0.8 ? undefined : 'need accuracy rising with training frequency';
  });
  if (!idk) {
    const q = parseModel(manifest, new Uint8Array(readFileSync(resolve(output, 'facts.bin'))));
    const rows = factsMeasure(q, world).rows;
    writeJson('facts-data', { version: 1, months: MONTHS, people: rows.map((r) => ({ name: r.name, month: r.month, count: r.count, pick: r.pick, confidence: r4(r.conf) })),
      probeNames: world.probeNames, copyNamesNote: 'context-copy lines used 3000 other fresh invented names; probeNames never appear in training',
      trainingMonthFrequencies: manifest.metrics.trainingMonthFrequencies, wrongContextRule: 'wrong month = (true + 5) mod 12' });
  }
  console.log(JSON.stringify(manifest.metrics, null, 2));
}

// ---------------------------------------------------------------------------------------------
// Day 14 (applied/prompting, applied/tool-use): prompt-model. Language, prompts and every measured effect live in
// src/lib/applied/promptmodel.ts so the widgets and tests reproduce the numbers from the shipped bytes.
// node train-models.mjs prompt-model [--steps 9000 --seed 14 --batch 24 --lr 0.003 --every 1000]
// ---------------------------------------------------------------------------------------------
const PROMPT_TOOL_SHARE = 0.2;
/** One training sequence. Letter tasks (80%): task uniform, tag 50%, k = 0..8 examples (0..6 for sr scratchpads, to fit
 * 96 tokens), inputs any 3 letters over a..h; loss on every answer (examples and query) and its `;`. Tool lines (20%):
 * half `[tools] a*b>CALL calc(a*b) RESULT n >n;` (loss on the call and on `>n;`; RESULT n is written by the program),
 * half direct `a*b>n;` (loss on n;). `=` and `|` never occur. */
function promptSequence(rng: Rng): Example {
  const parts: [string, boolean][] = [];
  const word = () => Array.from({ length: 3 }, () => PM.LETTERS[Math.floor(rng.next() * 8)]).join('');
  if (rng.next() < PROMPT_TOOL_SHARE) {
    const a = 10 + Math.floor(rng.next() * 90), b = 10 + Math.floor(rng.next() * 90), n = a * b;
    if (rng.next() < 0.5) parts.push([`[tools] ${a}*${b}>`, false], [`CALL calc(${a}*${b})`, true], [`RESULT ${n}`, false], [`>${n};`, true]);
    else parts.push([`${a}*${b}>`, false], [`${n};`, true]);
  } else {
    const task = PM.TASKS[Math.floor(rng.next() * PM.TASKS.length)], tag = rng.next() < 0.5;
    const scratchpad = task === 'sr' && rng.next() < 0.5, k = Math.floor(rng.next() * ((scratchpad ? 6 : 8) + 1));
    if (tag) parts.push([PM.TAG[task], false]);
    for (let i = 0; i <= k; i++) { const x = word(); parts.push([`${x}>`, false], [PM.answerBody({ task, query: x, scratchpad }) + ';', true]); }
  }
  const ids: number[] = [], sup: boolean[] = [];
  for (const [text, s] of parts) for (const t of PM.encode(text)) { ids.push(t); sup.push(s); }
  return { ids: ids.slice(0, -1), targets: ids.slice(1).map((t, i) => (sup[i + 1] ? t : -1)) };
}
/** Fast progress probe on the 50 test strings (not the acceptance metrics). */
function promptProbe(w: Weights) {
  const ex = PM.EXAMPLES;
  return {
    rev4: PM.evalPrompt(w, { task: 'rev', examples: ex.slice(0, 4) }).accuracy,
    rev4unseen: PM.evalPrompt(w, { task: 'rev', examples: ex.slice(0, 4), delims: ['=', '|'] }).accuracy,
    shift: [0, 2, 8].map((k) => PM.evalPrompt(w, { task: 'shift', examples: ex.slice(0, k) }).accuracy),
    sort: [0, 2, 8].map((k) => PM.evalPrompt(w, { task: 'sort', examples: ex.slice(0, k) }).accuracy),
    srDirect4: PM.evalPrompt(w, { task: 'sr', examples: ex.slice(0, 4) }).accuracy,
    srPad4: PM.evalPrompt(w, { task: 'sr', examples: ex.slice(0, 4), scratchpad: true }).accuracy,
    tool: PM.accuracy(PM.QUESTIONS.slice(0, 20).map(([a, b]) => PM.toolRun(w, a, b))),
    direct: PM.accuracy(PM.QUESTIONS.slice(0, 20).map(([a, b]) => PM.directAnswer(w, a, b))),
  };
}
function promptMeasure(w: Weights) {
  const r = (x: number) => r4(x), t0 = Date.now();
  const format = PM.formatExperiment(w), specific = PM.specificExperiment(w);
  const shots = { shift: PM.shotsExperiment(w, 'shift'), sort: PM.shotsExperiment(w, 'sort') };
  const order = PM.orderExperiment(w), steps = PM.stepsExperiment(w), structured = PM.structuredExperiment(w), evalx = PM.evalExperiment(w);
  const tool = PM.toolExperiment(w);
  // Inference cost: one full forward on a 70-token prompt (8 untagged reverse examples + query = 68 tokens, padded with a tag).
  const prompt70 = PM.buildPrompt({ task: 'rev', tag: true, examples: [...PM.EXAMPLES, PM.TEST[0]], query: PM.TEST[1] }).slice(0, 70);
  const c0 = performance.now(); for (let i = 0; i < 20; i++) forward(w, prompt70); const forwardMs = (performance.now() - c0) / 20;
  return {
    format: { trained: format.trained.accuracy, unseen: format.unseen.accuracy },
    specific: { untaggedP: r(specific.untagged.meanP), taggedP: r(specific.tagged.meanP), untaggedAccuracy: specific.untagged.accuracy,
      taggedAccuracy: specific.tagged.accuracy, untaggedSpread: Object.fromEntries(Object.entries(specific.spread).map(([k, v]) => [k, r(v)])) },
    shots: { k: [...PM.SHOTS], shift: shots.shift.map((s) => s.accuracy), sort: shots.sort.map((s) => s.accuracy) },
    order: { set: PM.ORDER_SET, min: order.min, median: order.median, max: order.max, accuracies: order.orders.map((o) => o.accuracy),
      quirkySet: PM.QUIRKY_SET, quirkyMin: order.quirkyMin, quirkyMedian: order.quirkyMedian, quirkyMax: order.quirkyMax },
    steps: { direct: steps.direct.accuracy, scratchpad: steps.scratchpad.accuracy, directFirstP: r(steps.direct.firstP), scratchpadFirstP: r(steps.scratchpad.firstP),
      directFinalP: r(steps.direct.finalP), scratchpadFinalP: r(steps.scratchpad.finalP) },
    structured: { unconstrainedParseable: structured.unconstrained.parseable, unconstrainedCorrect: structured.unconstrained.correctAmongParseable,
      constrainedParseable: structured.constrained.parseable, constrainedCorrect: structured.constrained.correctAmongParseable },
    eval: { accA: r(evalx.accA), accB: r(evalx.accB), trueDiff: r(evalx.trueDiff), bySize: evalx.bySize.map((s) => ({ size: s.size, worse: s.worse, tie: s.tie })) },
    tool: { directAccuracy: tool.directAccuracy, pCallWithTools: r(tool.pCallWithTools), pCallWithoutTools: r(tool.pCallWithoutTools), toolAccuracy: tool.toolAccuracy,
      firstAttemptRejected: tool.firstAttemptRejected, successWithRetries: tool.successWith, resultAttentionPerHead: tool.resultAttentionPerHead.map(r),
      rejections: tool.sampled.flatMap((s) => s.attempts.filter((a) => !a.parse.ok).map((a) => PM.decode(a.output))) },
    forwardMs70: r(forwardMs), measureSeconds: Math.round((Date.now() - t0) / 1000),
  };
}
function promptAccept(m: Record<string, any>): string | undefined {
  const fails: string[] = [], rises = (xs: number[]) => xs.at(-1)! - xs[0] >= 0.3 && xs.every((x, i) => !i || x >= xs[i - 1] - 0.06);
  if (m.format.trained - m.format.unseen < 0.2) fails.push('format: trained delimiters not clearly better');
  if (m.specific.taggedP - m.specific.untaggedP < 0.3 || m.specific.untaggedP > 0.6) fails.push('specific: tag does not lift p(correct first letter)');
  if (!rises(m.shots.shift) || !rises(m.shots.sort)) fails.push('shots: accuracy does not rise with k');
  if (m.order.max - m.order.min < 0.1 || m.order.quirkyMedian >= m.order.median) fails.push('order: no spread or quirky set not worse');
  if (m.steps.scratchpad - m.steps.direct < 0.2) fails.push('steps: scratchpad not clearly better');
  if (m.structured.unconstrainedParseable >= 50 || m.structured.constrainedParseable !== 50) fails.push('structured: parse rates');
  if (m.eval.trueDiff <= 0) fails.push('eval: B (6-shot) not better');
  if (m.tool.directAccuracy > 0.2 || m.tool.pCallWithTools < 0.9 || m.tool.toolAccuracy < 0.9) fails.push('tool: direct/call/with-tool accuracy');
  const waived = (args.includes('--waive') ? args[args.indexOf('--waive') + 1].split(',') : []);
  const left = fails.filter((f) => !waived.some((w) => f.startsWith(w)));
  return left.length ? left.join('; ') : undefined;
}
function promptModel() {
  const total = opt('steps', 9000), batchSize = opt('batch', 24), lr0 = opt('lr', 0.003), seed0 = opt('seed', 14), every = opt('every', 1000);
  const cfg: Config = { vocab: PM.VOCAB, d: opt('d', 48), heads: 4, layers: 2, context: 96, ff: opt('ff', 128), positions: 'rope', causal: true, norm: 'pre', tied: true };
  const rng = createRng(seed0), w = initWeights(cfg, rng), curve: { step: number; loss: number }[] = [];
  console.log(`prompt-model: ${w.params.length} parameters, ${total} updates, batch ${batchSize}, lr ${lr0}, seed ${seed0}`);
  const t0 = Date.now();
  let avg = 0;
  adamTrain(w, total, lr0, 200, () => Array.from({ length: batchSize }, () => promptSequence(rng)), (step, loss) => {
    avg += loss;
    if (step % 50 === 0) { curve.push({ step, loss: r4(avg / 50) }); avg = 0; }
    if (step === 20) console.log(`prompt-model: ${((Date.now() - t0) / 20).toFixed(0)} ms/step`);
    if (step % every === 0 || step === total) console.log(`prompt-model ${step}/${total} (${((Date.now() - t0) / 1000).toFixed(0)} s) loss ${curve.at(-1)?.loss}`, JSON.stringify(promptProbe(w)));
  });
  const wall = Math.round((Date.now() - t0) / 1000);
  const { manifest } = ship('prompt-model', w, curve, { seed: seed0, steps: total, batch: batchSize, lr: lr0, wallSeconds: wall,
    schedule: '200-update warmup, cosine decay to 0.1*lr', optimizer: 'Adam beta1=0.9 beta2=0.999 eps=1e-8, global clip 1',
    data: `promptSequence: ${1 - PROMPT_TOOL_SHARE} letter tasks (rev/shift/sort/copy/sr uniform; tag 50%; 0-8 examples, 0-6 for sr scratchpads; sr scratchpad 50%), ${PROMPT_TOOL_SHARE} tool lines (half [tools] calls, half direct answers); loss on model-written tokens only`,
    vocabNote: '"=" and "|" have their own ids and never occur in training; other symbols map to <unk>', lossCurveMeaning: 'mean training loss over the previous 50 updates' },
  promptMeasure, (m) => args.includes('--dry') ? `dry run (acceptance would say: ${promptAccept(m) ?? 'pass'})` : promptAccept(m), 'int8');
  console.log(JSON.stringify(manifest.metrics, null, 2));
}

if (job === 'report') {
  for (const name of ['tiny-lm', 'tiny-lm-nomask']) {
    const manifest: Manifest = JSON.parse(readFileSync(resolve(output, `${name}.json`), 'utf8'));
    const bytes = new Uint8Array(readFileSync(resolve(output, `${name}.bin`)));
    if (sha256(bytes) !== manifest.sha256) throw new Error(`Checksum mismatch: ${name}`);
    manifest.metrics = report(parseModel(manifest, bytes));
    if (args.includes('--write')) writeFileSync(resolve(output, `${name}.json`), JSON.stringify(manifest, null, 2) + '\n');
    console.log(JSON.stringify({ name, metrics: manifest.metrics }, null, 2));
  }
} else if (job === 'corpus') {
  const text = [args[1], args[2]].map((p) => prepareAlice(readFileSync(p, 'utf8'))).join('\n') + '\n';
  writeFileSync(ALICE, text); console.log('wrote scripts/data/alice.txt', text.length, 'chars');
} else if (job === 'house') house();
else if (job === 'scaling') scaling();
else if (job === 'icl') icl();
else if (job === 'icl-sweep') iclSweep();
else if (job === 'facts') facts();
else if (job === 'facts-idk') facts(true);
else if (job === 'prompt-model') promptModel();
else if (job === 'day10') { house(); scaling(); icl(); iclSweep(); facts(); }
else if (job === 'day9') { train('tiny-lm'); train('tiny-lm-nomask'); }
else if (job === 'tiny-lm' || job === 'tiny-lm-nomask') train(job);
else throw new Error(`Unknown job ${job}. Implemented jobs: day9, tiny-lm, tiny-lm-nomask, report, corpus, house, scaling, icl, icl-sweep, facts, facts-idk, day10, prompt-model. Later days add their own model entries here.`);
