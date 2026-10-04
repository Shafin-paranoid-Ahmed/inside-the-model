/**
 * Universal approximation by hand-set weights: bumps from pairs of units, sums of bumps, the cost of a sharp feature,
 * and the tent-map depth example. No training anywhere; every curve is an analytic 1 -> N -> 1 (or deeper) network. Pure logic, no DOM.
 */
import { createRng } from '../../../lib/math/rng';

export type Act = 'sigmoid' | 'relu';
export type Mode = 'bump' | 'sum' | 'limits' | 'depth';
export interface Marker { at: number; label: string }

/** Dyadic grid on [0,1] (2^11 + 1 points): tent-map kinks land exactly on grid points, and N up to 1024 bumps get two samples per cell. */
export const NG = 2049;
/** Error is measured on a finer grid (8 samples per cell at N = 1024) than the one that is drawn. */
const NE = 8193;
export const XS = Float64Array.from({ length: NG }, (_, i) => i / (NG - 1));

const sig = (z: number) => 1 / (1 + Math.exp(-z));
const relu = (z: number) => (z > 0 ? z : 0);
const smooth = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };
const mix = (a: number, b: number, f: number) => a + (b - a) * f;

/** 10% to 90% transition width of a sigmoid is 4.4 / k; the ReLU ramp is given the same width, so the two are comparable. */
export const RAMP_W = 4.4;
/**
 * A "step unit" rising from 0 to 1 around centre c with steepness k.
 * sigmoid: one unit, sigma(k (x - c)).  relu: a ramp built from two ReLUs, (relu(x - c + w/2) - relu(x - c - w/2)) / w with w = 4.4 / k.
 */
export function stepUnit(act: Act, x: number, c: number, k: number): number {
  if (act === 'sigmoid') return sig(k * (x - c));
  const w = RAMP_W / k;
  return (relu(x - c + w / 2) - relu(x - c - w / 2)) / w;
}
export const unitsPerStep = (act: Act) => (act === 'sigmoid' ? 1 : 2);
/** Hidden units in one bump (a rising step minus a shifted rising step). */
export const unitsPerBump = (act: Act) => 2 * unitsPerStep(act);

// ---------------------------------------------------------------- bump mode
export const T_BUMP = 12;
export const BUMP_A = 0.35, BUMP_B = 0.65, K0 = 4, K1 = 16, K2 = 80;
export const BUMP_MARKERS: Marker[] = [{ at: 0, label: 'One unit' }, { at: 3, label: 'Add a second' }, { at: 6.5, label: 'Sharpen' }];

export interface BumpFrame {
  kind: 'bump'; phase: string; say: string; act: Act;
  k: number; a: number; b: number; hasB: boolean; units: number; peak: number;
  ya: Float64Array; yb: Float64Array | null; diff: Float64Array;
}
export function bumpFrameAt(t: number, act: Act = 'sigmoid'): BumpFrame {
  const k = t < 3 ? mix(K0, K1, smooth(t / 3)) : t < 6.5 ? K1 : mix(K1, K2, smooth((t - 6.5) / 4.5)); // k eases; second unit slides in from the right
  const hasB = t >= 3, b = t < 3 ? 1.3 : t < 6.5 ? mix(1.3, BUMP_B, smooth((t - 3) / 3.5)) : BUMP_B;
  const ya = XS.map((x) => stepUnit(act, x, BUMP_A, k));
  const yb = hasB ? XS.map((x) => stepUnit(act, x, b, k)) : null;
  const diff = yb ? ya.map((v, i) => v - yb[i]) : ya;
  const phase = t < 3 ? 'One unit' : t < 6.5 ? 'Add a second' : 'Sharpen';
  const step = act === 'sigmoid' ? 'A sigmoid unit is a smooth step' : 'Two ReLUs make a ramp';
  const say = phase === 'One unit'
    ? `${step}: near 0 on the left, near 1 on the right, with its middle at x = ${BUMP_A}. The steepness k sets how abrupt the rise is.`
    : phase === 'Add a second'
      ? 'A second unit is subtracted. It is the same step, shifted right. Where both have risen they cancel, so what is left is a bump between the two steps.'
      : `Raising k steepens both edges. The bump approaches a flat-topped box. Its position comes from the two biases, its height from the output weight (1 here).`;
  return { kind: 'bump', phase, say, act, k, a: BUMP_A, b, hasB, units: unitsPerStep(act) * (hasB ? 2 : 1), peak: diff.reduce((m, v) => Math.max(m, v), 0), ya, yb, diff };
}

// ---------------------------------------------------------------- targets
const memo = new Map<string, unknown>();
const once = <T>(key: string, make: () => T): T => { if (!memo.has(key)) memo.set(key, make()); return memo.get(key) as T; };

/** A seeded smooth curve on [0,1] with values in [0.12, 0.88]. */
function makeSmooth(seed: number): (x: number) => number {
  const r = createRng(seed), p1 = r.uniform(0, 6.28), p2 = r.uniform(0, 6.28), raw = (x: number) => 0.9 * Math.sin(6.2832 * x + p1) + 0.45 * Math.sin(6.2832 * 2.6 * x + p2);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < 2001; i++) { const v = raw(i / 2000); lo = Math.min(lo, v); hi = Math.max(hi, v); }
  return (x) => 0.12 + (0.76 * (raw(x) - lo)) / (hi - lo);
}
export const SMOOTH = makeSmooth(7);
export const SPIKE_AT = 0.62, SPIKE_W = 0.004, SPIKE_H = 0.5;
/** The smooth curve plus one narrow Gaussian spike. */
export const SPIKY = (x: number) => SMOOTH(x) + SPIKE_H * Math.exp(-0.5 * ((x - SPIKE_AT) / SPIKE_W) ** 2);
export const TOL = 0.05;

// ---------------------------------------------------------------- bump sums
/** Steepness per bump count: edges are ~0.44 / N wide, a tenth of a cell. */
export const K_PER_N = 10;
export const SUM_NS = [1, 2, 4, 8, 16, 32];
export const LIM_NS = [1, 4, 16, 64, 256, 1024];
/** Bump i covers [i/N, (i+1)/N]. The outer two edges are parked off the domain so the ends stay flat (ponytail: still counted as 2 units per bump). */
const edge = (N: number, j: number) => (j === 0 ? -1 : j === N ? 2 : j / N);

export interface StageData {
  n: number; units: number; curve: Float64Array; maxErr: number; mse: number;
  /** Individual bumps (only kept for N up to 32). */
  bumps: Float64Array[];
}

function buildSum(fn: (x: number) => number, N: number, act: Act): StageData {
  const k = K_PER_N * N, h = Float64Array.from({ length: N }, (_, i) => fn((i + 0.5) / N));
  // sum_i h_i (u(x, e_i) - u(x, e_i+1)) regrouped per edge: (h_j - h_{j-1}) u(x, e_j). Edges two cells or more to the left have
  // fully risen (their weights telescope to h_{c-2}) and edges further right are still 0, so only a window of edges is evaluated.
  const d = (j: number) => (j < N ? h[j] : 0) - (j > 0 ? h[j - 1] : 0);
  const at = (x: number) => {
    const c = Math.min(N - 1, Math.floor(x * N));
    let y = c >= 2 ? h[c - 2] : 0;
    for (let j = Math.max(0, c - 1); j <= Math.min(N, c + 2); j++) y += d(j) * stepUnit(act, x, edge(N, j), k);
    return y;
  };
  const curve = XS.map(at);
  let maxErr = 0, se = 0;
  for (let g = 0; g < NE; g++) { const x = g / (NE - 1), er = Math.abs(at(x) - fn(x)); maxErr = Math.max(maxErr, er); se += er * er; }
  const bumps = N <= 32 ? Array.from(h, (hi, i) => XS.map((x) => hi * (stepUnit(act, x, edge(N, i), k) - stepUnit(act, x, edge(N, i + 1), k)))) : [];
  return { n: N, units: N * unitsPerBump(act), curve, maxErr, mse: se / NE, bumps };
}
export const sumStages = (which: 'sum' | 'limits' | 'limitsSmooth', act: Act): StageData[] =>
  once(`${which}:${act}`, () => {
    const fn = which === 'limits' ? SPIKY : SMOOTH;
    return (which === 'sum' ? SUM_NS : LIM_NS).map((N) => buildSum(fn, N, act));
  });
/** Smallest stage whose max error is at most `tol`, or null. */
export const firstWithin = (st: StageData[], tol = TOL) => st.find((s) => s.maxErr <= tol) ?? null;
export const targetCurve = (which: 'sum' | 'limits') => once(`tgt:${which}`, () => XS.map(which === 'sum' ? SMOOTH : SPIKY));

// ---------------------------------------------------------------- tent map / depth
/** One tent layer: two ReLUs. On [0,1], T(x) = 2 relu(x) - 4 relu(x - 1/2). */
export const tent = (x: number) => 2 * relu(x) - 4 * relu(x - 0.5);
/** Output of a network of k stacked tent layers (2k hidden ReLU units in total). */
export function deepAt(k: number, x: number): number { for (let i = 0; i < k; i++) x = tent(x); return x; }
export const DEPTHS = [1, 2, 3, 4, 5];
/** Number of linear pieces of a sampled piecewise-linear function (counts slope changes on the grid). */
export function countPieces(y: ArrayLike<number>): number {
  let changes = 0, prev = y[1] - y[0];
  for (let i = 1; i < y.length - 1; i++) { const d = y[i + 1] - y[i]; if (Math.abs(d - prev) > 1e-9) changes++; prev = d; }
  return changes + 1;
}
/** Hidden units a single hidden layer needs at minimum: m ReLUs give at most m + 1 pieces, so 2^k pieces need m >= 2^k - 1. */
export const shallowMin = (k: number) => 2 ** k - 1;
export const deepUnits = (k: number) => 2 * k;
export interface DepthStage { k: number; curve: Float64Array; pieces: number }
export const depthStages = (): DepthStage[] => once('depth', () => DEPTHS.map((k) => { const curve = XS.map((x) => deepAt(k, x)); return { k, curve, pieces: countPieces(curve) }; }));

// ---------------------------------------------------------------- staged timeline (sum, limits, depth)
const STAGE = { sum: { D: 2.4, M: 0.5, S: SUM_NS.length }, limits: { D: 2.4, M: 0.5, S: LIM_NS.length }, depth: { D: 2.4, M: 0.5, S: DEPTHS.length } } as const;
export const durationOf = (mode: Mode) => (mode === 'bump' ? T_BUMP : STAGE[mode].D * STAGE[mode].S);

/** A real network snapshot changes at each marker. Motion traces that snapshot instead of mixing different networks. */
export function stageClock(t: number, S: number, D: number, M: number) {
  const idx = Math.min(S - 1, Math.max(0, Math.floor(t / D)));
  return { a: idx, b: idx, w: 0, s: idx, reveal: smooth((t - idx * D) / M) };
}

export function markersOf(mode: Mode): Marker[] {
  if (mode === 'bump') return BUMP_MARKERS;
  const { D } = STAGE[mode];
  const labels = mode === 'sum' ? SUM_NS.map((n) => `N = ${n}`) : mode === 'limits' ? LIM_NS.map((n) => `N = ${n}`) : DEPTHS.map((k) => `k = ${k}`);
  return labels.map((label, i) => ({ at: i * D, label }));
}

export interface StagedFrame {
  kind: 'staged'; mode: 'sum' | 'limits' | 'depth'; phase: string; say: string; act: Act;
  /** Stage index shown in the readout and the side chart. */
  s: number; a: number; b: number; w: number;
  reveal: number;
  curve: Float64Array; target: Float64Array | null;
  bumpsA: Float64Array[]; bumpsB: Float64Array[];
  n: number; units: number; maxErr: number; mse: number;
  /** Depth only. */
  pieces: number; shallow: number; kinks: number[];
}
export function stagedFrameAt(mode: 'sum' | 'limits' | 'depth', t: number, act: Act = 'sigmoid'): StagedFrame {
  const { D, M, S } = STAGE[mode], { a, b, w, s, reveal } = stageClock(t, S, D, M);
  if (mode === 'depth') {
    const st = depthStages(), cur = st[s];
    const kinks = Array.from({ length: 2 ** cur.k - 1 }, (_, j) => (j + 1) / 2 ** cur.k);
    return {
      kind: 'staged', mode, phase: `k = ${cur.k}`, act, s, a, b, w, reveal, curve: cur.curve, target: null, bumpsA: [], bumpsB: [],
      n: cur.k, units: deepUnits(cur.k), maxErr: 0, mse: 0, pieces: cur.pieces, shallow: shallowMin(cur.k), kinks,
      say: cur.k === 1
        ? 'One tent layer (two ReLUs) turns the diagonal into a tent: it rises to 1 at x = 1/2, then falls back. Two linear pieces.'
        : `Feed the output back in as the input of another tent layer and every piece is folded again. After ${cur.k} layers: ${cur.pieces} pieces from ${deepUnits(cur.k)} units, because the pieces double with each layer while the units grow by only two.`,
    };
  }
  const key = mode === 'sum' ? 'sum' : 'limits', st = sumStages(key, act), cur = st[s], tgt = targetCurve(key);
  const width = (1 / cur.n).toFixed(cur.n >= 100 ? 4 : 3);
  const say = mode === 'sum'
    ? `${cur.n} bump${cur.n > 1 ? 's' : ''}, each of width ${width} and as tall as the target at its centre. The largest error on the measurement grid is ${cur.maxErr.toFixed(3)}.`
    : `${cur.n} bump${cur.n > 1 ? 's' : ''} of width ${width}. Max error ${cur.maxErr.toFixed(3)}${cur.maxErr <= TOL ? ', inside' : ', still outside'} the tolerance of ${TOL}. The spike is only captured once a bump is about as narrow as the spike.`;
  return {
    kind: 'staged', mode, phase: `N = ${cur.n}`, say, act, s, a, b, w, reveal, curve: cur.curve, target: tgt,
    bumpsA: st[a].bumps, bumpsB: a === b ? [] : st[b].bumps, n: cur.n, units: cur.units, maxErr: cur.maxErr, mse: cur.mse, pieces: 0, shallow: 0, kinks: [],
  };
}

export type Frame = BumpFrame | StagedFrame;
export const frameAt = (mode: Mode, t: number, act: Act = 'sigmoid'): Frame => (mode === 'bump' ? bumpFrameAt(t, act) : stagedFrameAt(mode, t, act));
