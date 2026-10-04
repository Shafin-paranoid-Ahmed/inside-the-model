import type { Rng } from '../math/rng';
import { allowedNext, VOCAB } from './grammar';
import { exp, sinCos, softmax } from './math';
export { softmax, cosine } from './math';

export interface Config {
  vocab: readonly string[];
  d: number;
  heads: number;
  layers: number;
  context: number;
  ff: number; // zero = attention-only
  positions: 'learned' | 'rope';
  causal: boolean;
  norm: 'pre' | 'post';
  tied: boolean;
}
export const TINY_CONFIG: Config = {
  vocab: VOCAB, d: 16, heads: 2, layers: 2, context: 10, ff: 64,
  positions: 'learned', causal: true, norm: 'pre', tied: true,
};
export interface Tensor { offset: number; rows: number; cols: number; length: number }
export type Offsets = Record<string, Tensor>;
export interface Weights { config: Config; params: Float64Array; offsets: Offsets }

export function parameterLayout(c: Config): { offsets: Offsets; size: number } {
  for (const key of ['d', 'heads', 'layers', 'context'] as const) {
    if (!Number.isInteger(c[key]) || c[key] < 1) throw new Error(`Invalid ${key}`);
  }
  if (c.d % c.heads || (c.positions === 'rope' && c.d / c.heads % 2)) throw new Error('Invalid head width');
  if (!Number.isInteger(c.ff) || c.ff < 0 || c.vocab.length < 2 || new Set(c.vocab).size !== c.vocab.length) throw new Error('Invalid ff/vocab');
  const offsets: Offsets = {};
  let size = 0;
  const add = (name: string, rows: number, cols = 1) => {
    offsets[name] = { offset: size, rows, cols, length: rows * cols }; size += rows * cols;
  };
  add('embedding', c.vocab.length, c.d);
  if (c.positions === 'learned') add('position', c.context, c.d);
  for (let l = 0; l < c.layers; l++) {
    for (const norm of ['ln1', 'ln2']) { add(`${l}.${norm}.gain`, c.d); add(`${l}.${norm}.bias`, c.d); }
    for (const projection of ['q', 'k', 'v', 'o']) { add(`${l}.${projection}.weight`, c.d, c.d); add(`${l}.${projection}.bias`, c.d); }
    if (c.ff) {
      add(`${l}.ff1.weight`, c.ff, c.d); add(`${l}.ff1.bias`, c.ff);
      add(`${l}.ff2.weight`, c.d, c.ff); add(`${l}.ff2.bias`, c.d);
    }
  }
  add('final.gain', c.d); add('final.bias', c.d);
  if (!c.tied) add('unembedding', c.vocab.length, c.d);
  add('logit.bias', c.vocab.length);
  return { offsets, size };
}

export function initWeights(config: Config, rng: Rng): Weights {
  const { offsets, size } = parameterLayout(config);
  const params = new Float64Array(size);
  for (const [name, tensor] of Object.entries(offsets)) {
    if (name.endsWith('.gain')) { params.fill(1, tensor.offset, tensor.offset + tensor.length); continue; }
    if (name.endsWith('.bias')) continue;
    const scale = name === 'embedding' || name === 'position' ? 0.15 : Math.sqrt(3 / tensor.cols);
    for (let i = 0; i < tensor.length; i++) params[tensor.offset + i] = rng.uniform(-scale, scale);
  }
  return { config, params, offsets };
}

export interface NormState { input: Float64Array; normalized: Float64Array; output: Float64Array; invStd: Float64Array }
export function layerNorm(input: Float64Array, d: number, gain: ArrayLike<number>, bias: ArrayLike<number>): NormState {
  const normalized = new Float64Array(input.length), output = new Float64Array(input.length);
  const invStd = new Float64Array(input.length / d);
  for (let t = 0; t < invStd.length; t++) {
    let mean = 0, variance = 0;
    for (let j = 0; j < d; j++) mean += input[t * d + j];
    mean /= d;
    for (let j = 0; j < d; j++) variance += (input[t * d + j] - mean) ** 2;
    const inv = 1 / Math.sqrt(variance / d + 1e-5); invStd[t] = inv;
    for (let j = 0; j < d; j++) {
      const i = t * d + j; normalized[i] = (input[i] - mean) * inv;
      output[i] = normalized[i] * gain[j] + bias[j];
    }
  }
  return { input, normalized, output, invStd };
}

export interface ResidualPatch {
  layer: number; // -1 for x0
  position: number; // absolute sequence position
  stream: 'x0' | 'attnOut' | 'resid1' | 'resid2';
  value: ArrayLike<number>;
}
export interface ForwardOptions {
  causal?: boolean;
  noPositions?: boolean;
  positionScale?: number;
  noScale?: boolean;
  norm?: 'pre' | 'post';
  ropeScale?: (position: number) => number;
  ablate?: readonly (readonly [number, number])[];
  patch?: readonly ResidualPatch[];
}
export interface LayerTrace {
  input: Float64Array;
  ln1: Float64Array; ln2: Float64Array;
  norm1: NormState; norm2: NormState;
  attentionInput: Float64Array; mlpInput: Float64Array;
  q: Float64Array[]; k: Float64Array[]; v: Float64Array[];
  scores: Float64Array[]; weights: Float64Array[]; headOut: Float64Array[];
  concat: Float64Array; attnOut: Float64Array; resid1: Float64Array;
  hidden: Float64Array; mlpOut: Float64Array; resid2: Float64Array;
}
export interface Trace {
  ids: readonly number[]; start: number; keyLength: number;
  options: ForwardOptions; norm: 'pre' | 'post';
  x0: Float64Array; layers: LayerTrace[];
  final: NormState; logits: Float64Array; probs: Float64Array;
}
export interface KVCache {
  length: number;
  keys: Float64Array[][]; values: Float64Array[][];
  weights: Weights; options: ForwardOptions;
}

export function project(w: Weights, name: string, input: Float64Array, tokens: number): Float64Array {
  const tensor = w.offsets[`${name}.weight`], bias = w.offsets[`${name}.bias`];
  const out = new Float64Array(tokens * tensor.rows);
  for (let t = 0; t < tokens; t++) for (let r = 0; r < tensor.rows; r++) {
    let sum = w.params[bias.offset + r];
    for (let j = 0; j < tensor.cols; j++) sum += w.params[tensor.offset + r * tensor.cols + j] * input[t * tensor.cols + j];
    out[t * tensor.rows + r] = sum;
  }
  return out;
}
const add = (a: Float64Array, b: Float64Array) => Float64Array.from(a, (v, i) => v + b[i]);
function normAt(w: Weights, name: string, input: Float64Array): NormState {
  const d = w.config.d;
  return layerNorm(input, d, w.params.subarray(w.offsets[`${name}.gain`].offset, w.offsets[`${name}.gain`].offset + d),
    w.params.subarray(w.offsets[`${name}.bias`].offset, w.offsets[`${name}.bias`].offset + d));
}
export function rotate(vec: Float64Array, width: number, start: number, opts: ForwardOptions, inverse = false): void {
  for (let t = 0; t < vec.length / width; t++) for (let j = 0; j < width; j += 2) {
    const position = (opts.ropeScale ?? ((p: number) => p))(start + t);
    const angle = position * (opts.positionScale ?? 1) * exp(-9.210340371976184 * j / width);
    const [s0, co] = sinCos(angle), si = inverse ? -s0 : s0;
    const i = t * width + j, a = vec[i], b = vec[i + 1];
    vec[i] = co * a - si * b; vec[i + 1] = si * a + co * b;
  }
}
function applyPatch(vec: Float64Array, opts: ForwardOptions, layer: number, stream: ResidualPatch['stream'], d: number, start: number): void {
  for (const patch of opts.patch ?? []) if (patch.layer === layer && patch.stream === stream) {
    if (patch.value.length !== d) throw new Error('Patch width differs from model width');
    const t = patch.position - start;
    if (t >= 0 && t < vec.length / d) vec.set(patch.value, t * d);
  }
}

/** Full and incremental passes share every block operation. Scores expose all raw Q.K pairs;
 * weights alone apply the scale and causal mask. Arrays are row-major (tokens × width).
 */
function run(w: Weights, ids: readonly number[], opts: ForwardOptions, cache?: KVCache): Trace {
  const c = w.config, d = c.d, T = ids.length, dk = d / c.heads, start = cache?.length ?? 0, N = start + T;
  if (!T || N > c.context) throw new Error('Sequence is empty or exceeds context');
  if (ids.some((id) => !Number.isInteger(id) || id < 0 || id >= c.vocab.length)) throw new Error('Invalid token id');
  if (opts.positionScale !== undefined && (!Number.isFinite(opts.positionScale) || opts.positionScale < 0 || opts.positionScale > 1)) throw new Error('Invalid position scale');
  const causal = opts.causal ?? c.causal, norm = opts.norm ?? c.norm;
  if (cache && !causal) throw new Error('KV caching requires causal attention');
  const x0 = new Float64Array(T * d), eo = w.offsets.embedding.offset, po = w.offsets.position?.offset;
  for (let t = 0; t < T; t++) for (let j = 0; j < d; j++) {
    x0[t * d + j] = w.params[eo + ids[t] * d + j] +
      (!opts.noPositions && po !== undefined ? (opts.positionScale ?? 1) * w.params[po + (start + t) * d + j] : 0);
  }
  applyPatch(x0, opts, -1, 'x0', d, start);
  const layers: LayerTrace[] = [];
  let x: Float64Array = x0;
  for (let l = 0; l < c.layers; l++) {
    const input = x;
    let norm1 = normAt(w, `${l}.ln1`, input);
    const attentionInput = norm === 'pre' ? norm1.output : input;
    const qAll = project(w, `${l}.q`, attentionInput, T), kAll = project(w, `${l}.k`, attentionInput, T), vAll = project(w, `${l}.v`, attentionInput, T);
    const q: Float64Array[] = [], k: Float64Array[] = [], v: Float64Array[] = [];
    const scores: Float64Array[] = [], weights: Float64Array[] = [], headOut: Float64Array[] = [];
    const concat = new Float64Array(T * d);
    for (let h = 0; h < c.heads; h++) {
      const qh = new Float64Array(T * dk), kh = new Float64Array(T * dk), vh = new Float64Array(T * dk);
      for (let t = 0; t < T; t++) for (let j = 0; j < dk; j++) {
        qh[t * dk + j] = qAll[t * d + h * dk + j];
        kh[t * dk + j] = kAll[t * d + h * dk + j]; vh[t * dk + j] = vAll[t * d + h * dk + j];
      }
      if (c.positions === 'rope' && !opts.noPositions) { rotate(qh, dk, start, opts); rotate(kh, dk, start, opts); }
      let keys: Float64Array = kh, values: Float64Array = vh;
      if (cache) {
        cache.keys[l][h].set(kh, start * dk); cache.values[l][h].set(vh, start * dk);
        keys = cache.keys[l][h].subarray(0, N * dk); values = cache.values[l][h].subarray(0, N * dk);
      }
      q.push(qh); k.push(keys.slice()); v.push(values.slice());
      const raw = new Float64Array(T * N), aw = new Float64Array(T * N), out = new Float64Array(T * dk);
      for (let t = 0; t < T; t++) {
        const scaled = new Float64Array(N);
        for (let s = 0; s < N; s++) {
          let dot = 0;
          for (let j = 0; j < dk; j++) dot += qh[t * dk + j] * keys[s * dk + j];
          raw[t * N + s] = dot;
          scaled[s] = causal && s > start + t ? -Infinity : dot / (opts.noScale ? 1 : Math.sqrt(dk));
        }
        const row = softmax(scaled); aw.set(row, t * N);
        for (let j = 0; j < dk; j++) {
          let sum = 0;
          for (let s = 0; s < N; s++) sum += row[s] * values[s * dk + j];
          out[t * dk + j] = sum;
        }
      }
      if (opts.ablate?.some(([ll, hh]) => ll === l && hh === h)) out.fill(0);
      for (let t = 0; t < T; t++) for (let j = 0; j < dk; j++) concat[t * d + h * dk + j] = out[t * dk + j];
      scores.push(raw); weights.push(aw); headOut.push(out);
    }
    const attnOut = project(w, `${l}.o`, concat, T);
    applyPatch(attnOut, opts, l, 'attnOut', d, start);
    const sum1 = add(input, attnOut);
    if (norm === 'post') norm1 = normAt(w, `${l}.ln1`, sum1);
    const resid1 = norm === 'post' ? norm1.output : sum1;
    applyPatch(resid1, opts, l, 'resid1', d, start);
    let norm2 = normAt(w, `${l}.ln2`, resid1);
    const mlpInput = norm === 'pre' ? norm2.output : resid1;
    const hidden = c.ff ? project(w, `${l}.ff1`, mlpInput, T) : new Float64Array(0);
    for (let i = 0; i < hidden.length; i++) hidden[i] = Math.max(0, hidden[i]);
    const mlpOut = c.ff ? project(w, `${l}.ff2`, hidden, T) : new Float64Array(T * d);
    const sum2 = add(resid1, mlpOut);
    if (norm === 'post') norm2 = normAt(w, `${l}.ln2`, sum2);
    const resid2 = norm === 'post' ? norm2.output : sum2;
    applyPatch(resid2, opts, l, 'resid2', d, start);
    layers.push({ input, ln1: norm1.output, ln2: norm2.output, norm1, norm2, attentionInput, mlpInput,
      q, k, v, scores, weights, headOut, concat, attnOut, resid1, hidden, mlpOut, resid2 });
    x = resid2;
  }
  const final = normAt(w, 'final', x), logits = new Float64Array(T * c.vocab.length), probs = new Float64Array(logits.length);
  const uo = w.offsets[c.tied ? 'embedding' : 'unembedding'].offset, bo = w.offsets['logit.bias'].offset;
  for (let t = 0; t < T; t++) {
    for (let id = 0; id < c.vocab.length; id++) {
      let sum = w.params[bo + id];
      for (let j = 0; j < d; j++) sum += final.output[t * d + j] * w.params[uo + id * d + j];
      logits[t * c.vocab.length + id] = sum;
    }
    probs.set(softmax(logits.subarray(t * c.vocab.length, (t + 1) * c.vocab.length)), t * c.vocab.length);
  }
  if (cache) cache.length = N;
  return { ids: [...ids], start, keyLength: N, options: opts, norm, x0, layers, final, logits, probs };
}

export const forward = (w: Weights, ids: readonly number[], opts: ForwardOptions = {}): Trace => run(w, ids, opts);
export function prefill(w: Weights, ids: readonly number[], opts: ForwardOptions = {}): { cache: KVCache; logits: Float64Array; trace: Trace } {
  const c = w.config, dk = c.d / c.heads;
  const cache: KVCache = { length: 0, weights: w, options: opts,
    keys: Array.from({ length: c.layers }, () => Array.from({ length: c.heads }, () => new Float64Array(c.context * dk))),
    values: Array.from({ length: c.layers }, () => Array.from({ length: c.heads }, () => new Float64Array(c.context * dk))) };
  const trace = run(w, ids, opts, cache);
  return { cache, logits: trace.logits, trace };
}
export function decodeStep(w: Weights, cache: KVCache, token: number): Float64Array {
  if (w !== cache.weights) throw new Error('Cache belongs to different weights');
  return run(w, [token], cache.options, cache).logits;
}

export function logitLens(w: Weights, vec: Float64Array): Float64Array {
  if (vec.length !== w.config.d) throw new Error('Logit lens expects one residual vector');
  const x = normAt(w, 'final', vec).output, out = new Float64Array(w.config.vocab.length);
  const o = w.offsets[w.config.tied ? 'embedding' : 'unembedding'].offset;
  for (let id = 0; id < out.length; id++) {
    let sum = w.params[w.offsets['logit.bias'].offset + id];
    for (let j = 0; j < x.length; j++) sum += x[j] * w.params[o + id * x.length + j];
    out[id] = sum;
  }
  return out;
}
export function grammaticalMass(trace: Trace, ids: readonly number[] = trace.ids): number {
  const V = trace.probs.length / trace.ids.length;
  let mass = 0, rows = 0;
  for (let t = 0; t < ids.length; t++) {
    const legal = allowedNext(ids.slice(0, t + 1));
    if (!legal.size) continue;
    for (const id of legal) mass += trace.probs[t * V + id];
    rows++;
  }
  return rows ? mass / rows : 1;
}

export interface GenerateOptions {
  maxTokens?: number;
  sampler?: (probabilities: Float64Array, step: number) => number;
  logitMask?: (ids: readonly number[]) => Iterable<number>; // allowed ids, not banned ids
  stopId?: number;
  forward?: ForwardOptions;
}
export function generate(w: Weights, prompt: readonly number[], opts: GenerateOptions = {}): number[] {
  if (!prompt.length) throw new Error('Generation needs a prompt');
  const ids = [...prompt], causal = opts.forward?.causal ?? w.config.causal;
  const pre = causal ? prefill(w, ids, opts.forward) : undefined;
  let logits: Float64Array = (pre?.logits ?? forward(w, ids, opts.forward).logits).slice(-w.config.vocab.length);
  const sampler = opts.sampler ?? ((p: Float64Array) => {
    let best = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i; return best;
  });
  const maxTokens = opts.maxTokens ?? w.config.context - ids.length;
  for (let step = 0; step < maxTokens; step++) {
    if (ids.length >= w.config.context || (opts.stopId !== undefined && ids.at(-1) === opts.stopId)) break;
    const masked = logits.slice();
    if (opts.logitMask) {
      const allowed = new Set(opts.logitMask(ids));
      if (!allowed.size) throw new Error('Logit mask permits no tokens');
      for (let id = 0; id < masked.length; id++) if (!allowed.has(id)) masked[id] = -Infinity;
    }
    const p = softmax(masked), next = sampler(p, step);
    if (!Number.isInteger(next) || next < 0 || next >= p.length || !p[next]) throw new Error('Sampler selected an invalid/masked token');
    ids.push(next);
    if (ids.length < w.config.context && next !== opts.stopId) {
      logits = pre ? decodeStep(w, pre.cache, next) : forward(w, ids, opts.forward).logits.slice(-w.config.vocab.length);
    }
  }
  return ids;
}
