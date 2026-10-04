// KVCacheLab: pure logic. A real greedy run of the house model with its KV cache and without it,
// plus analytic operation, memory and roofline counts (never wall-clock time, so every number is deterministic).
import { decodeStep, forward, prefill, type Weights } from '../../../lib/transformer/model';
import { decode, encode } from '../../../lib/lm/charset';
import { CONFIGS, kvBytes, macsForward, roofline, weightBytes, type PublishedAccelerator, type PublishedModel } from '../../../lib/lm/cost';

export type Mode = 'loop' | 'nocache' | 'cache' | 'memory' | 'phases' | 'batch';
export interface Marker { at: number; label: string }

export const PROMPT = 'alice was not a ';
export const CONTEXT = 64;
export const LOOP_STEPS = 12;
export const MAX_CTX_LOG2 = 17; // 131,072 tokens
export const MAX_BATCH_LOG2 = 9; // 512

const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
export const ease = (x: number) => { const c = clamp(x); return c * c * (3 - 2 * c); };
const argmax = (a: ArrayLike<number>) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };

/** One character for display: a space shows as a middle dot and a newline as a return mark. */
export const show = (ch: string) => (ch === ' ' ? '·' : ch === '\n' ? '↵' : ch);

// ---------- formatting (one set of strings for readouts, labels and prose) ----------
export const fmtInt = (n: number) => Math.round(n).toLocaleString('en-US');
export function fmtBytes(n: number): string {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let k = 0;
  while (k < 4 && n >= 1024 ** (k + 1) * 0.9995) k++;
  const v = n / 1024 ** k;
  return `${Number((v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)))} ${units[k]}`;
}
export function fmtCount(n: number): string {
  if (n >= 1e9) return `${Number((n / 1e9).toPrecision(3))} G`;
  if (n >= 1e6) return `${Number((n / 1e6).toPrecision(3))} M`;
  if (n >= 1e3) return `${Number((n / 1e3).toPrecision(3))} k`;
  return String(Math.round(n));
}
/** Decimal gigabytes, the unit of the GPU's advertised memory. */
export const fmtGB = (n: number) => (n >= 1e12 ? `${(n / 1e12).toFixed(1)} TB` : `${(n / 1e9).toFixed(n < 1e11 ? 1 : 0)} GB`);
export const fmtDiff = (x: number) => (x === 0 ? '0' : x.toExponential(1));

// ---------- the real run ----------
export interface Run {
  ids: number[]; P: number; N: number;
  /** Per generation step: the largest |logit difference| between the cached and the uncached pass. */
  diff: number[];
  stepNo: number[]; stepCache: number[];
  /** Per layer, rows = cached positions (N - 1), columns = d (heads side by side). */
  keys: Float64Array[]; values: Float64Array[];
  /** Per layer: attention weights of each position over earlier ones, averaged over heads, (N-1) x (N-1). */
  attn: Float64Array[];
  maxK: number[]; maxV: number[];
}
export const stepMacs = (cfg: Weights['config'], P: number, i: number) => (i === 0 ? macsForward(cfg, P, P) : macsForward(cfg, 1, P + i));

const runs = new WeakMap<Weights, Promise<Run>>();
/** One shared run per loaded model (all islands on a page reuse it). */
export const sharedRun = (w: Weights): Promise<Run> => runs.get(w) ?? (runs.set(w, buildRun(w)), runs.get(w)!);
/** Greedy generation with the real KV cache, checked step by step against a full uncached forward pass. */
export async function buildRun(w: Weights, prompt = PROMPT, N = CONTEXT): Promise<Run> {
  const c = w.config, V = c.vocab.length, d = c.d, dk = d / c.heads;
  const ids = encode(prompt), P = ids.length, G = N - P;
  const pre = prefill(w, ids);
  let logits: Float64Array = pre.logits.slice(-V);
  const diff: number[] = [], stepNo: number[] = [], stepCache: number[] = [];
  for (let i = 0; i < G; i++) {
    if (i % 3 === 2) await new Promise((r) => setTimeout(r)); // yield: no task longer than ~20 ms
    const L = P + i, plain = forward(w, ids).logits.slice(-V);
    let m = 0;
    for (let j = 0; j < V; j++) m = Math.max(m, Math.abs(logits[j] - plain[j]));
    diff.push(m); stepNo.push(macsForward(c, L, L)); stepCache.push(stepMacs(c, P, i));
    ids.push(argmax(logits));
    if (ids.length < N) logits = decodeStep(w, pre.cache, ids[ids.length - 1]);
  }
  const rows = N - 1, tr = forward(w, ids.slice(0, rows));
  const keys: Float64Array[] = [], values: Float64Array[] = [], attn: Float64Array[] = [];
  for (let l = 0; l < c.layers; l++) {
    const k = new Float64Array(rows * d), v = new Float64Array(rows * d), a = new Float64Array(rows * rows);
    for (let h = 0; h < c.heads; h++) {
      for (let r = 0; r < rows; r++) for (let j = 0; j < dk; j++) {
        k[r * d + h * dk + j] = pre.cache.keys[l][h][r * dk + j]; v[r * d + h * dk + j] = pre.cache.values[l][h][r * dk + j];
      }
      const wt = tr.layers[l].weights[h];
      for (let i = 0; i < a.length; i++) a[i] += wt[i] / c.heads;
    }
    keys.push(k); values.push(v); attn.push(a);
  }
  const mx = (a: Float64Array) => a.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
  return { ids, P, N, diff, stepNo, stepCache, keys, values, attn, maxK: keys.map(mx), maxV: values.map(mx) };
}
export const textOf = (run: Run, n = run.ids.length) => decode(run.ids.slice(0, n));
/** Display character at position k of the run. */
export const chr = (run: Run, k: number) => show(decode([run.ids[k]]));

// ---------- scenes: everything that does not depend on t ----------
const cum = (a: number[]) => a.reduce<number[]>((o, x) => (o.push((o.at(-1) ?? 0) + x), o), []);

export interface StepScene {
  kind: 'steps'; mode: 'loop' | 'nocache' | 'cache'; run: Run; P: number; G: number; N: number; beat: number; duration: number;
  markers: Marker[]; stepNo: number[]; stepCache: number[]; cumNo: number[]; cumCache: number[]; cumLen: number[]; diffMax: number[];
  totalNo: number; totalCache: number; speedup: number; repeated: number; compare: boolean;
}
/** `total` = prompt + generated characters (so the model runs total - P steps). */
export function stepScene(mode: StepScene['mode'], run: Run, total: number, compare = true): StepScene {
  const P = run.P, N = clamp(Math.round(total), P + 1, run.N), G = N - P, beat = mode === 'loop' ? 1 : 0.35;
  const stepNo = run.stepNo.slice(0, G), stepCache = run.stepCache.slice(0, G);
  const cumNo = cum(stepNo), cumCache = cum(stepCache), cumLen = cum(Array.from({ length: G }, (_, i) => P + i));
  let m = 0;
  const diffMax = run.diff.slice(0, G).map((x) => (m = Math.max(m, x)));
  const markers = {
    loop: [{ at: 0, label: 'One run per new character' }],
    nocache: [{ at: 0, label: 'Every step re-reads the text' }],
    cache: [{ at: 0, label: 'The cache grows each step' }],
  }[mode];
  const totalNo = cumNo[G - 1], totalCache = cumCache[G - 1];
  return { kind: 'steps', mode, run, P, G, N, beat, duration: G * beat, markers, stepNo, stepCache, cumNo, cumCache, cumLen, diffMax,
    totalNo, totalCache, speedup: totalNo / totalCache, repeated: 1 - totalCache / totalNo, compare };
}

export interface StepFrame {
  kind: 'steps'; s: number; f: number; L: number; newFrom: number; predicted: string; phase: string; status: string;
  /** loop: how much of each layer's row is lit (0 to 1); counters tween with them. */
  lit1: number; lit2: number; showPred: number; fly: number;
  positions: number; macsNo: number; macsCache: number; diffMax: number;
}
export function stepFrameAt(t: number, sc: StepScene): StepFrame {
  const u0 = clamp(t / sc.beat, 0, sc.G), u = u0 > sc.G - 1e-9 ? sc.G : u0, s = Math.min(sc.G - 1, Math.floor(u)), f = u >= sc.G ? 1 : u - s;
  const L = sc.P + s, predicted = show(decode([sc.run.ids[L]]));
  const sweep = clamp(f / 0.5), lit1 = clamp(sweep * 1.5), lit2 = clamp(sweep * 1.5 - 0.5);
  const showPred = clamp((f - 0.5) / 0.12), fly = clamp((f - 0.62) / 0.38);
  const before = (c: number[]) => (s ? c[s - 1] : 0);
  const positions = before(sc.cumLen) + Math.round(L * lit2);
  const diffMax = Math.max(s ? sc.diffMax[s - 1] : 0, f >= 0.5 ? sc.run.diff[s] : 0);
  const base = `Step ${s + 1} of ${sc.G}: `;
  let phase = '', status = '';
  if (sc.mode === 'loop') {
    phase = f < 0.5 ? 'reading' : 'predicting';
    status = f < 0.5 ? `${base}the model reads all ${L} characters so far.` : `${base}it reads all ${L} characters and predicts “${predicted}”, which joins the end of the text.`;
  } else if (sc.mode === 'nocache') {
    phase = s === 0 ? 'prompt' : 'repeat';
    status = s === 0 ? `${base}the ${L} prompt characters are all new work.` : `${base}re-reads all ${L} characters; only the last one is new, the other ${L - 1} repeat earlier work.`;
  } else {
    phase = s === 0 ? 'prefill' : 'append';
    status = s === 0 ? `${base}the ${L} prompt characters go through the model once and their keys and values are stored.`
      : `${base}one new character goes through the model, its keys and values are appended, and it reads all ${L} stored rows.`;
  }
  return { kind: 'steps', s, f, L, newFrom: s === 0 ? 0 : L - 1, predicted, phase, status, lit1, lit2, showPred, fly, positions,
    macsNo: f === 1 ? sc.cumNo[s] : before(sc.cumNo) + f * sc.stepNo[s], macsCache: f === 1 ? sc.cumCache[s] : before(sc.cumCache) + f * sc.stepCache[s], diffMax };
}

// ---------- memory ----------
export interface MemRow { key: string; name: string; layers: number; kvDim: number; perToken: number; color: string }
export const BYTES16 = 2;
export interface MemScene { kind: 'memory'; rows: MemRow[]; batch: number; mem: number; hwName: string; duration: number; markers: Marker[] }
export function memoryScene(w: Weights, batch: number): MemScene {
  const raw = [
    { key: 'house', name: 'House model (this page)', layers: w.config.layers, kvDim: w.config.d, color: '--viz-cat-1' },
    { key: 'llama2', name: CONFIGS.llama2_7b.name, layers: CONFIGS.llama2_7b.layers, kvDim: CONFIGS.llama2_7b.kvDim, color: '--viz-cat-2' },
    { key: 'llama3', name: CONFIGS.llama3_8b.name, layers: CONFIGS.llama3_8b.layers, kvDim: CONFIGS.llama3_8b.kvDim, color: '--viz-cat-3' },
  ];
  const rows = raw.map((r) => ({ ...r, perToken: kvBytes({ layers: r.layers, kvDim: r.kvDim, tokens: 1, bytes: BYTES16 }) }));
  const duration = 12, at = (c: number) => Math.log2(c) / MAX_CTX_LOG2 * duration;
  return { kind: 'memory', rows, batch, mem: CONFIGS.h100_sxm.memoryBytes, hwName: CONFIGS.h100_sxm.name, duration,
    markers: [{ at: 0, label: 'Up to 16k tokens' }, { at: at(16384), label: 'To 128k' }] };
}
export interface MemFrame { kind: 'memory'; ctx: number; bytes: number[]; full: boolean[]; status: string }
export function memoryFrameAt(t: number, sc: MemScene): MemFrame {
  const ctx = Math.max(1, Math.round(2 ** (clamp(t / sc.duration) * MAX_CTX_LOG2)));
  const bytes = sc.rows.map((r) => kvBytes({ layers: r.layers, kvDim: r.kvDim, tokens: ctx, batch: sc.batch, bytes: BYTES16 }));
  const full = bytes.map((b) => b > sc.mem);
  return { kind: 'memory', ctx, bytes, full, status: `${fmtInt(ctx)} tokens, batch ${sc.batch}: ${sc.rows.map((r, i) => `${r.name} ${fmtBytes(bytes[i])}`).join(', ')}.` };
}
/** Context length (tokens) at which a row's cache alone reaches the memory size; may exceed the sweep. */
export const crossing = (r: MemRow, sc: MemScene) => sc.mem / (r.perToken * sc.batch);

// ---------- prefill and decode ----------
export interface PhasesScene {
  kind: 'phases'; P: number; D: number; N: number; ops: number[]; bytes: number; intensity: number[]; prefillSec: number; beat: number;
  duration: number; markers: Marker[];
}
export function phasesScene(w: Weights, P: number, N = CONTEXT): PhasesScene {
  P = clamp(Math.round(P), 1, N - 2);
  const D = N - P - 1, bytes = weightBytes(w.config, 1), ops: number[] = [];
  for (let i = 0; i <= D; i++) ops.push(2 * stepMacs(w.config, P, i));
  const prefillSec = 3, beat = 0.12;
  return { kind: 'phases', P, D, N, ops, bytes, intensity: ops.map((o) => o / bytes), prefillSec, beat, duration: prefillSec + beat * D,
    markers: [{ at: 0, label: 'Prefill: all at once' }, { at: prefillSec, label: 'Decode: one at a time' }] };
}
export interface PhasesFrame { kind: 'phases'; stage: 'prefill' | 'decode'; j: number; g: number; cells: number; bars: number; status: string }
/** `bars`: how many step bars are visible (the last may be partial via `g`); `cells`: cache rows drawn (fractional last). */
export function phasesFrameAt(t: number, sc: PhasesScene): PhasesFrame {
  if (t < sc.prefillSec) {
    const g = ease(t / sc.prefillSec);
    return { kind: 'phases', stage: 'prefill', j: 0, g, cells: sc.P * g, bars: 1,
      status: `Prefill: all ${sc.P} prompt characters in one pass, ${fmtCount(sc.ops[0])} operations, ${fmtInt(sc.bytes)} weight bytes read once, ${sc.intensity[0].toFixed(1)} operations per byte.` };
  }
  const u = clamp((t - sc.prefillSec) / sc.beat, 0, sc.D), j = Math.min(sc.D, Math.floor(u) + 1), g = u >= sc.D ? 1 : u - Math.floor(u);
  return { kind: 'phases', stage: 'decode', j, g, cells: sc.P + j - 1 + g, bars: j + 1,
    status: `Decode step ${j} of ${sc.D}: one new character, ${fmtCount(sc.ops[j])} operations, the same ${fmtInt(sc.bytes)} weight bytes read again, ${sc.intensity[j].toFixed(1)} operations per byte.` };
}

// ---------- batching and the roofline ----------
export type ModelKey = 'llama2_7b' | 'llama3_8b';
export interface BatchScene {
  kind: 'batch'; model: PublishedModel; hw: PublishedAccelerator; ctx: number; weightB: number; kvPer: number; flopsPerToken: number;
  ridge: number; fitMax: number; duration: number; markers: Marker[];
}
export interface BatchFrame {
  kind: 'batch'; b: number; intensity: number; flops: number; tokens: number; perUser: number; used: number; kv: number; fits: boolean;
  bound: 'memory' | 'compute'; regime: string; status: string;
}
export function batchPoint(sc: BatchScene, b: number): BatchFrame {
  const intensity = b * sc.flopsPerToken / sc.weightB, r = roofline(sc.hw, intensity);
  const kv = b * sc.kvPer, used = sc.weightB + kv, fits = used <= sc.hw.memoryBytes, tokens = r.flops / sc.flopsPerToken;
  const regime = `${fits ? '' : 'Cache full, '}${r.bound === 'memory' ? 'memory-bound' : 'compute-bound'}`;
  return { kind: 'batch', b, intensity, flops: r.flops, tokens, perUser: tokens / b, used, kv, fits, bound: r.bound, regime,
    status: `Batch ${b}: ${Number(intensity.toPrecision(3))} operation${Number(intensity.toPrecision(3)) === 1 ? "" : "s"} per byte, about ${fmtInt(tokens)} tokens per second in total, ${fmtInt(tokens / b)} per user. ${fits ? 'The cache fits.' : 'The cache does not fit.'} ${r.bound === 'memory' ? 'Limited by memory bandwidth.' : 'Limited by arithmetic.'}` };
}
export const batchAt = (t: number) => (t >= MAX_BATCH_LOG2 ? 2 ** MAX_BATCH_LOG2 : Math.max(1, Math.round(2 ** Math.max(0, t))));
export function batchScene(model: ModelKey, ctx: number): BatchScene {
  const m = CONFIGS[model], hw = CONFIGS.h100_sxm;
  const weightB = weightBytes(m.params, BYTES16);
  const kvPer = kvBytes({ layers: m.layers, kvDim: m.kvDim, tokens: ctx, batch: 1, bytes: BYTES16 });
  const flopsPerToken = 2 * m.params, ridge = hw.peakFlops / hw.bandwidth;
  const sc: BatchScene = { kind: 'batch', model: m, hw, ctx, weightB, kvPer, flopsPerToken, ridge, duration: MAX_BATCH_LOG2 + 1,
    fitMax: Math.max(0, Math.floor((hw.memoryBytes - weightB) / kvPer)), markers: [] };
  const markers: Marker[] = [];
  for (let t = 0; t <= MAX_BATCH_LOG2; t += 0.02) {
    const p = batchPoint(sc, batchAt(t)), name = p.fits ? (p.bound === 'memory' ? 'Memory-bound' : 'Compute-bound') : p.bound === 'memory' ? 'Full' : p.fits ? 'Compute-bound' : 'Compute';
    if (!markers.length || markers[markers.length - 1].label !== name) markers.push({ at: Math.round(t * 100) / 100, label: name });
  }
  sc.markers = markers;
  return sc;
}
export const batchFrameAt = (t: number, sc: BatchScene): BatchFrame => batchPoint(sc, batchAt(t));
