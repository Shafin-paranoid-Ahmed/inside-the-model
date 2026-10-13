/**
 * LoRALab: every number is computed here, DOM-free.
 * Task: a linear layer y = W x (32 -> 32). The "pretrained" base is W0; task B asks for W* = W0 + Delta, with noisy labels.
 * Training uses sufficient statistics of the data (S = X X^T / n, C = Y X^T / n), so a step costs 32^3 multiplications and the
 * loss on fresh data is exact: ||W - W*||_F^2 / D + noise variance. Elementary functions come from gen/mlp.ts (Chrome = Node).
 */
import { createRng, type Rng } from '../../../lib/math/rng';
import { normal, adam, createAdam } from '../../../lib/gen/mlp';
import { eigSym } from '../../../lib/math/linalg';

export const D = 32;
export const NOISE = 0.3; // label noise std; the loss floor on fresh data is NOISE^2
export const FLOOR = NOISE * NOISE;
export const ALPHA_OVER_R = 2; // LoRA scale alpha / r (alpha = 2r)
export const RANKS = [1, 2, 4, 8] as const;
export type Rank = (typeof RANKS)[number];
export const STEPS = 300;
export const EVERY = 5;
export const NCK = STEPS / EVERY + 1; // checkpoints 0, 5, ..., 300
export const N_SWEEP = [4, 16, 64, 256] as const;
export const N_MAX = 256;
const LR_FULL = 0.01, LR_LORA = 0.02;

// ------------------------------------------------------------------ memory (ZeRO, arXiv:1910.02054, section 3.1)
/** Bytes per parameter for mixed-precision Adam, as the ZeRO paper states them: fp16 copy 2, fp16 gradient 2, optimizer K = 4 + 4 + 4. */
export const BYTES = { weights: 2, grads: 2, master: 4, momentum: 4, variance: 4 } as const;
export const OPT_BYTES = BYTES.master + BYTES.momentum + BYTES.variance; // K = 12
export const FULL_BYTES = BYTES.weights + BYTES.grads + OPT_BYTES; // 16
export const SCALE_BITS = 32 / 64; // one fp32 scale per 64-weight block
export const Q_BITS = 4 + SCALE_BITS; // 4.5 bits per stored weight
export const DQ_SAVING_BITS = 0.37; // QLoRA paper: double quantization saves about 0.37 bits per parameter

export interface Preset { id: string; label: string; params: number; layers: number; width: number }
/** Illustrative decoder shapes; adapters sit on the four attention projections (q, k, v, o) of every layer. */
export const PRESETS: Preset[] = [
  { id: '1B', label: '1B', params: 1e9, layers: 16, width: 2048 },
  { id: '7B', label: '7B', params: 7e9, layers: 32, width: 4096 },
  { id: '65B', label: '65B', params: 65e9, layers: 80, width: 8192 },
  { id: '70B', label: '70B', params: 70e9, layers: 80, width: 8192 },
];
export const LORA_RANK_BIG = 16;
export const GPU_GB = 80;
export const GB = 1e9;
export const adapterParams = (p: Preset, r = LORA_RANK_BIG): number => 4 * r * 2 * p.width * p.layers;

export type Setup = 'full' | 'lora' | 'qlora';
export interface MemParts { weights: number; grads: number; optimizer: number; total: number } // GB; activations are not counted
export function memory(setup: Setup, p: Preset, r = LORA_RANK_BIG): MemParts {
  const N = p.params, A = adapterParams(p, r);
  let weights: number, grads: number, optimizer: number;
  if (setup === 'full') { weights = N * BYTES.weights; grads = N * BYTES.grads; optimizer = N * OPT_BYTES; }
  else {
    // frozen base (no gradient, no optimizer state) plus a trained adapter that costs the full 16 bytes per parameter
    weights = setup === 'lora' ? N * BYTES.weights + A * BYTES.weights : (N * Q_BITS) / 8 + A * BYTES.weights;
    grads = A * BYTES.grads; optimizer = A * OPT_BYTES;
  }
  return { weights: weights / GB, grads: grads / GB, optimizer: optimizer / GB, total: (weights + grads + optimizer) / GB };
}

// ------------------------------------------------------------------ dense helpers (row-major Float64Array)
export function matmul(A: Float64Array, B: Float64Array, n: number, k: number, m: number): Float64Array {
  const out = new Float64Array(n * m);
  for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) { const a = A[i * k + p]; if (a !== 0) for (let j = 0; j < m; j++) out[i * m + j] += a * B[p * m + j]; }
  return out;
}
const transpose = (A: Float64Array, n: number, m: number): Float64Array => { const T = new Float64Array(n * m); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j * n + i] = A[i * m + j]; return T; };
const normSq = (A: ArrayLike<number>): number => { let s = 0; for (let i = 0; i < A.length; i++) s += A[i] * A[i]; return s; };

/** s * B A, with B (D x r) and A (r x D). */
export const lowRank = (B: Float64Array, A: Float64Array, r: number, scale = ALPHA_OVER_R): Float64Array => { const M = matmul(B, A, D, r, D); for (let i = 0; i < M.length; i++) M[i] *= scale; return M; };
/** Layer i of s B A: column i of B times row i of A (a rank-1 matrix). */
export function layer(B: Float64Array, A: Float64Array, r: number, i: number, scale = ALPHA_OVER_R): Float64Array {
  const M = new Float64Array(D * D);
  for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) M[a * D + b] = scale * B[a * r + i] * A[i * D + b];
  return M;
}
/** Singular values of a D x D matrix, descending (square roots of the eigenvalues of M^T M). */
export function singularValues(M: Float64Array): Float64Array {
  const G = matmul(transpose(M, D, D), M, D, D, D);
  const rows = Array.from({ length: D }, (_, i) => Array.from(G.subarray(i * D, (i + 1) * D)));
  return Float64Array.from(eigSym(rows).values.map((v) => Math.sqrt(Math.max(0, v))));
}
/** Share of the squared singular values held by the top k. */
export function energyTop(sv: ArrayLike<number>, k: number): number {
  let a = 0, b = 0;
  for (let i = 0; i < sv.length; i++) { b += sv[i] * sv[i]; if (i < k) a += sv[i] * sv[i]; }
  return a / b;
}

// ------------------------------------------------------------------ the tasks
export type TaskId = 'low' | 'full' | 'c';
export const base = (): Float64Array => { const r = createRng(11); return Float64Array.from({ length: D * D }, () => 0.3 * normal(r)); };
function unit(rng: Rng): Float64Array { const v = Float64Array.from({ length: D }, () => normal(rng)); const n = Math.sqrt(normSq(v)); return v.map((x) => x / n); }
/** The change task B wants: two rank-1 layers plus a little dense noise ('low', 'c'), or a dense random matrix of the same size ('full'). */
export function delta(task: TaskId): Float64Array {
  const M = new Float64Array(D * D), rng = createRng(task === 'low' ? 21 : task === 'full' ? 22 : 23);
  if (task === 'full') { for (let i = 0; i < M.length; i++) M[i] = Math.sqrt(32 / (D * D)) * normal(rng); return M; }
  const sig = task === 'low' ? [4.2, 3.4] : [3.8, 3.0];
  for (const s of sig) { const u = unit(rng), v = unit(rng); for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) M[a * D + b] += s * u[a] * v[b]; }
  for (let i = 0; i < M.length; i++) M[i] += 0.02 * normal(rng);
  return M;
}
export const target = (task: TaskId): Float64Array => { const W = base(), d = delta(task); for (let i = 0; i < W.length; i++) W[i] += d[i]; return W; };

export interface Stats { n: number; S: Float64Array; C: Float64Array; yy: number }
/** Statistics of the first n of 256 fixed examples of a task: x ~ N(0, I), y = W* x + noise. */
export function dataStats(task: TaskId, n: number): Stats {
  const Ws = target(task), rng = createRng(task === 'low' ? 31 : task === 'full' ? 32 : 33);
  const X = new Float64Array(D * N_MAX), E = new Float64Array(D * N_MAX);
  for (let i = 0; i < X.length; i++) X[i] = normal(rng);
  for (let i = 0; i < E.length; i++) E[i] = NOISE * normal(rng);
  const S = new Float64Array(D * D), C = new Float64Array(D * D);
  let yy = 0;
  for (let c = 0; c < n; c++) {
    const y = new Float64Array(D);
    for (let a = 0; a < D; a++) { let s = E[a * N_MAX + c]; for (let b = 0; b < D; b++) s += Ws[a * D + b] * X[b * N_MAX + c]; y[a] = s; yy += s * s; }
    for (let a = 0; a < D; a++) { const xa = X[a * N_MAX + c]; for (let b = 0; b < D; b++) { S[a * D + b] += xa * X[b * N_MAX + c]; C[a * D + b] += y[a] * X[b * N_MAX + c]; } }
  }
  for (let i = 0; i < S.length; i++) { S[i] /= n; C[i] /= n; }
  return { n, S, C, yy: yy / n };
}
/** Mean squared error per output on the n training examples, from the statistics. */
export function trainLoss(W: Float64Array, st: Stats): number {
  const WS = matmul(W, st.S, D, D, D); let a = 0, b = 0;
  for (let i = 0; i < W.length; i++) { a += W[i] * WS[i]; b += W[i] * st.C[i]; }
  return (a - 2 * b + st.yy) / D;
}
/** Mean squared error per output on fresh examples (exact): ||W - W*||^2 / D + noise variance. */
export function testLoss(W: Float64Array, Ws: Float64Array): number { let s = 0; for (let i = 0; i < W.length; i++) { const d = W[i] - Ws[i]; s += d * d; } return s / D + FLOOR; }

// ------------------------------------------------------------------ training
export interface Curve { train: Float64Array; test: Float64Array; W: Float64Array; B?: Float64Array; A?: Float64Array; r: number; params: number }
/**
 * Adam on either all of W (r = 0, "full") or only B, A of W = Wbase + s B A (LoRA, Wbase frozen).
 * B starts at zero and A at N(0, 1/D), so step 0 is exactly the base model.
 */
export function train(Wbase: Float64Array, Ws: Float64Array, st: Stats, r: number, seed = 5, steps = STEPS): Curve {
  const full = r === 0, rng = createRng(seed);
  const nB = full ? 0 : D * r, p = full ? Float64Array.from(Wbase) : new Float64Array(2 * D * r);
  if (!full) for (let i = 0; i < D * r; i++) p[nB + i] = normal(rng) / Math.sqrt(D);
  const g = new Float64Array(p.length), ad = createAdam(p.length), lr = full ? LR_FULL : LR_LORA;
  const trn = new Float64Array(steps / EVERY + 1), tst = new Float64Array(steps / EVERY + 1);
  const weights = (): Float64Array => {
    if (full) return p;
    const W = lowRank(p.subarray(0, nB), p.subarray(nB), r);
    for (let i = 0; i < W.length; i++) W[i] += Wbase[i];
    return W;
  };
  for (let s = 0; s <= steps; s++) {
    const W = weights();
    if (s % EVERY === 0) { trn[s / EVERY] = trainLoss(W, st); tst[s / EVERY] = testLoss(W, Ws); }
    if (s === steps) break;
    const WS = matmul(W, st.S, D, D, D), G = new Float64Array(D * D);
    for (let i = 0; i < G.length; i++) G[i] = (2 / D) * (WS[i] - st.C[i]);
    if (full) g.set(G);
    else {
      const B = p.subarray(0, nB), A = p.subarray(nB);
      const gB = matmul(G, transpose(A, r, D), D, D, r), gA = matmul(transpose(B, D, r), G, r, D, D);
      for (let i = 0; i < nB; i++) g[i] = ALPHA_OVER_R * gB[i];
      for (let i = 0; i < r * D; i++) g[nB + i] = ALPHA_OVER_R * gA[i];
    }
    adam(p, g, ad, lr);
  }
  const W = Float64Array.from(weights());
  return full
    ? { train: trn, test: tst, W, r, params: D * D }
    : { train: trn, test: tst, W, B: Float64Array.from(p.subarray(0, nB)), A: Float64Array.from(p.subarray(nB)), r, params: 2 * D * r };
}

// ------------------------------------------------------------------ quantization (QLoRA, arXiv:2305.14314)
/** Standard normal CDF from the erf series (accurate to ~1e-15 for |z| < 4). */
function cdf(z: number): number {
  const x = z / Math.SQRT2;
  let term = x, sum = x;
  for (let n = 1; n < 60; n++) { term *= -(x * x) / n; sum += term / (2 * n + 1); }
  return 0.5 + sum / Math.sqrt(Math.PI);
}
/** Inverse of the standard normal CDF by bisection. */
export function ppf(q: number): number {
  let lo = -6, hi = 6;
  for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (cdf(m) < q) lo = m; else hi = m; }
  return (lo + hi) / 2;
}
const linspace = (a: number, b: number, n: number): number[] => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
/**
 * NF4 levels from the definition: quantiles of N(0,1) with an asymmetric split (7 negative, a zero, 8 positive, so all 16 codes are
 * used and 0 is exact), scaled to [-1, 1]. offset = 0.9677083 is the outer quantile bitsandbytes uses, 1 - (1/32 + 1/30) / 2.
 */
export function nf4Levels(offset = 0.9677083): number[] {
  const pos = linspace(offset, 0.5, 9).slice(0, -1).map(ppf), neg = linspace(offset, 0.5, 8).slice(0, -1).map((q) => -ppf(q));
  const all = [...pos, 0, ...neg].sort((a, b) => a - b), m = Math.max(...all.map(Math.abs));
  return all.map((v) => v / m);
}
/** Absmax int4: the integers -7..7 divided by 7. */
export const INT4_LEVELS: number[] = Array.from({ length: 15 }, (_, i) => (i - 7) / 7);
export const NF4_LEVELS = nf4Levels();
export type Scheme = 'int4' | 'nf4';
export const levelsOf = (s: Scheme): number[] => (s === 'nf4' ? NF4_LEVELS : INT4_LEVELS);
export const BLOCK = 64;

export interface Quant { norm: Float64Array; snapped: Float64Array; deq: Float64Array; absmax: Float64Array; mse: number }
/** Blockwise absmax quantization: divide each block of 64 by its largest |w|, snap to the nearest level, multiply back. */
export function quantize(w: Float64Array, scheme: Scheme): Quant {
  const L = levelsOf(scheme), n = w.length, nb = Math.ceil(n / BLOCK);
  const norm = new Float64Array(n), snapped = new Float64Array(n), deq = new Float64Array(n), absmax = new Float64Array(nb);
  let se = 0;
  for (let b = 0; b < nb; b++) {
    let m = 0;
    for (let i = 0; i < Math.min(BLOCK, n - b * BLOCK); i++) m = Math.max(m, Math.abs(w[b * BLOCK + i]));
    absmax[b] = m;
    for (let i = 0; i < BLOCK; i++) {
      const j = b * BLOCK + i;
      if (j >= n) break;
      const x = m === 0 ? 0 : w[j] / m;
      let best = L[0];
      for (const l of L) if (Math.abs(l - x) < Math.abs(best - x)) best = l;
      norm[j] = x; snapped[j] = best; deq[j] = best * m; se += (w[j] - deq[j]) ** 2;
    }
  }
  return { norm, snapped, deq, absmax, mse: n ? se / n : 0 };
}
export const variance = (w: ArrayLike<number>): number => {
  let m = 0, s = 0;
  for (let i = 0; i < w.length; i++) m += w[i];
  m /= w.length;
  for (let i = 0; i < w.length; i++) s += (w[i] - m) ** 2;
  return s / w.length;
};

// ------------------------------------------------------------------ everything the worker computes
export interface TaskRes { ws: Float64Array; full: Curve; lora: Record<Rank, Curve>; sv: Float64Array }
export interface Run {
  low: TaskRes; fullrank: TaskRes; c: Curve;
  qlora: { fp: Curve; nf4: Curve; int4: Curve; baseNf4: number; baseInt4: number };
  data: { n: number; curve: Curve }[];
}
function taskRes(task: TaskId): TaskRes {
  const W0 = base(), Ws = target(task), st = dataStats(task, N_MAX);
  const full = train(W0, Ws, st, 0);
  const lora = {} as Record<Rank, Curve>;
  for (const r of RANKS) lora[r] = train(W0, Ws, st, r, 5 + r);
  return { ws: Ws, full, lora, sv: singularValues(full.W.map((v, i) => v - W0[i])) };
}
export function computeRun(progress: (f: number) => void = () => {}): Run {
  const W0 = base(), Ws = target('low'), st = dataStats('low', N_MAX);
  const low = taskRes('low'); progress(0.3);
  const fullrank = taskRes('full'); progress(0.55);
  const c = train(W0, target('c'), dataStats('c', N_MAX), 4, 9);
  const qn = quantize(W0, 'nf4').deq, qi = quantize(W0, 'int4').deq;
  const qlora = { fp: low.lora[4], nf4: train(qn, Ws, st, 4, 9), int4: train(qi, Ws, st, 4, 9), baseNf4: testLoss(qn, Ws), baseInt4: testLoss(qi, Ws) };
  progress(0.8);
  const data = N_SWEEP.map((n) => ({ n, curve: train(W0, Ws, dataStats('low', n), 4, 9) }));
  progress(1);
  return { low, fullrank, c, qlora, data };
}

// ------------------------------------------------------------------ merging
export interface MergeResult { x: Float64Array; unmerged: Float64Array; merged: Float64Array; maxDiff: number; madUnmerged: number; madMerged: number }
/** The layer's output for one input, two ways: W x + s B (A x), and (W + s B A) x. */
export function mergeCheck(W: Float64Array, B: Float64Array, A: Float64Array, r: number, seed = 3): MergeResult {
  const rng = createRng(seed), x = Float64Array.from({ length: D }, () => normal(rng));
  const Wm = lowRank(B, A, r);
  for (let i = 0; i < Wm.length; i++) Wm[i] += W[i];
  const Ax = new Float64Array(r), unmerged = new Float64Array(D), merged = new Float64Array(D);
  for (let i = 0; i < r; i++) for (let j = 0; j < D; j++) Ax[i] += A[i * D + j] * x[j];
  let maxDiff = 0;
  for (let a = 0; a < D; a++) {
    let s = 0, m = 0, u = 0;
    for (let b = 0; b < D; b++) { s += W[a * D + b] * x[b]; m += Wm[a * D + b] * x[b]; }
    for (let i = 0; i < r; i++) u += B[a * r + i] * Ax[i];
    unmerged[a] = s + ALPHA_OVER_R * u; merged[a] = m; maxDiff = Math.max(maxDiff, Math.abs(unmerged[a] - merged[a]));
  }
  return { x, unmerged, merged, maxDiff, madUnmerged: D * D + r * 2 * D, madMerged: D * D };
}
/** Swap adapters in a merged weight: subtract one adapter's product, add another's. */
export function swapAdapter(Wm: Float64Array, from: Curve, to: Curve): Float64Array {
  const a = lowRank(from.B!, from.A!, from.r), b = lowRank(to.B!, to.A!, to.r);
  return Wm.map((v, i) => v - a[i] + b[i]);
}

// ------------------------------------------------------------------ timelines: pure frames of t
export type Mode = 'memory' | 'lowrank' | 'train' | 'spectrum' | 'merge' | 'quantize' | 'qlora' | 'data';
export interface Marker { at: number; label: string }
const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);
export const seg = (t: number, a: number, b: number): number => clamp01((t - a) / (b - a));
export const ease = (x: number): number => x * x * (3 - 2 * x);

export const UPDATES_PER_S = 50;
export function modeTime(mode: Mode, r = 2): { dur: number; markers: Marker[] } {
  switch (mode) {
    case 'memory': return { dur: 7, markers: [{ at: 0, label: 'Weights' }, { at: 2, label: 'Gradients' }, { at: 4, label: 'Optimizer state' }, { at: 6, label: 'Total' }] };
    case 'lowrank': return { dur: r + 3, markers: [{ at: 0, label: 'Rank-1 layers' }, { at: r, label: 'Add to W' }] };
    case 'train': case 'qlora': return { dur: 7.5, markers: [{ at: 0, label: 'Updates 0 to 100' }, { at: 2, label: '100 to 200' }, { at: 4, label: '200 to 300' }, { at: 6, label: 'Final' }] };
    case 'spectrum': return { dur: 9, markers: [{ at: 0, label: 'Singular values' }, { at: 3, label: 'Energy in the top r' }, { at: 5, label: 'LoRA loss at each r' }] };
    case 'merge': return { dur: 9, markers: [{ at: 0, label: 'Two paths' }, { at: 3, label: 'Fold BA into W' }, { at: 6, label: 'One path' }] };
    case 'quantize': return { dur: 10, markers: [{ at: 0, label: 'One block' }, { at: 5, label: 'All 16 blocks' }, { at: 8, label: 'Final error' }] };
    case 'data': return { dur: 8, markers: N_SWEEP.map((n, i) => ({ at: 2 * i, label: `${n} examples` })) };
  }
}
/** Update number reached at time t (training modes): 50 per second, held at 300. */
export const updateAt = (t: number): number => Math.max(0, Math.min(STEPS, Math.floor(t * UPDATES_PER_S)));
export const checkpointAt = (t: number): number => Math.floor(updateAt(t) / EVERY);

export function memoryFrame(t: number) {
  const phase = t < 2 ? 0 : t < 4 ? 1 : t < 6 ? 2 : 3;
  const f = [ease(seg(t, 0, 2)), ease(seg(t, 2, 4)), ease(seg(t, 4, 6))];
  return { phase, weights: f[0], grads: f[1], optimizer: f[2], gpu: seg(t, 6, 7) };
}
/** Layer k fades in during second k; the sum is layers 0..k-1 plus fade times layer k. From t = r the sum is complete and BA is added onto W. */
export function lowrankFrame(t: number, r: number) {
  const k = Math.min(r - 1, Math.floor(t)), fade = t >= r ? 1 : ease(t - k);
  return { k, fade, add: ease(seg(t, r, r + 2)), phase: t < r ? 0 : 1 };
}
export function spectrumFrame(t: number) {
  const bars = seg(t, 0, 3) * D, line = ease(seg(t, 3, 5)), ri = t < 5 ? -1 : Math.min(3, Math.floor((t - 5) / 1));
  return { bars, line, ri, phase: t < 3 ? 0 : t < 5 ? 1 : 2 };
}
export function mergeFrame(t: number) {
  return { flow: t < 3 ? (t * 1.2) % 1 : 1, fill: ease(seg(t, 0.4, 2.6)), fold: ease(seg(t, 3, 6)), fillMerged: ease(seg(t, 6, 8)), phase: t < 3 ? 0 : t < 6 ? 1 : 2 };
}
export const QUANT_BLOCK_ORDER = [5, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
export function quantFrame(t: number) {
  const slide = ease(seg(t, 0.8, 3.5)), err = ease(seg(t, 3.5, 5)), blocks = t < 5 ? 1 : Math.min(16, 1 + Math.floor(seg(t, 5, 8) * 16));
  return { slide, err, blocks, phase: t < 5 ? 0 : t < 8 ? 1 : 2 };
}
/** Running mean squared weight error over the first k blocks of the display order. */
export function runningMse(w: Float64Array, q: Quant, k: number): number {
  let s = 0;
  for (let i = 0; i < k; i++) { const b = QUANT_BLOCK_ORDER[i]; for (let j = b * BLOCK; j < (b + 1) * BLOCK; j++) s += (w[j] - q.deq[j]) ** 2; }
  return s / (k * BLOCK);
}
export function dataFrame(t: number) {
  const i = Math.min(N_SWEEP.length - 1, Math.floor(t / 2)), local = t - 2 * i, u = Math.min(STEPS, Math.floor(seg(local, 0, 1.4) * STEPS / EVERY) * EVERY);
  return { i, update: u, done: u >= STEPS, revealed: i + (u >= STEPS ? 1 : 0) };
}
export function qloraFrame(t: number) {
  return { ck: checkpointAt(t), bars: ease(seg(t, 0, 3)) };
}


/** Visual sum of rank-one corrections; only its reveal fades, trained values stay fixed. */
export function correctionFrame(t: number, c: Curve): Float64Array {
  const f = lowrankFrame(Math.max(0,t),c.r), out = new Float64Array(D*D);
  for (let k=0;k<=f.k;k++) {
    const part=layer(c.B!,c.A!,c.r,k), opacity=k===f.k?f.fade:1;
    for(let i=0;i<out.length;i++) out[i]+=part[i]*opacity;
  }
  return out;
}
