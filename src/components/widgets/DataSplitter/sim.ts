/** Train / validation / test splits: one dataset, three bins, a polynomial model trained by full-batch gradient descent. Pure logic, no DOM. */
import { createRng } from '../../../lib/math/rng';

export const N = 100, DEG = 14, EPOCHS = 1000, NOISE = 0.5;
export const NOISE_FLOOR = NOISE * NOISE;
/** Timeline: dots flow into bins for T_SPLIT seconds, then the model trains until T. */
export const T_SPLIT = 2.5, T = 12;
export const ROWS = 4;
export type Bin = 0 | 1 | 2; // train, validation, test

const truth = (x: number) => Math.sin(3 * x) + 0.4 * x;
/** Legendre features P0..P(DEG-1) on [-1, 1]: well conditioned, so gradient descent fits the smooth part first and the wiggles last. */
function features(x: number): Float64Array {
  const p = new Float64Array(DEG);
  p[0] = 1; p[1] = x;
  for (let k = 2; k < DEG; k++) p[k] = ((2 * k - 1) * x * p[k - 1] - (k - 1) * p[k - 2]) / k;
  return p;
}

interface Data { x: Float64Array; y: Float64Array }
function sample(n: number, seed: number): Data {
  const r = createRng(seed), x = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) { x[i] = r.uniform(-1, 1); y[i] = truth(x[i]) + r.normal(0, NOISE); }
  return { x, y };
}
const DATA = sample(N, 7);
const FRESH = sample(600, 99); // a large untouched sample stands in for "the real world"
const PERM = (() => {
  const r = createRng(3), a = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) { const j = Math.floor(r.next() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
})();
const POOL = (() => { const r = createRng(11), p = new Float32Array(2 * N); for (let i = 0; i < 2 * N; i++) p[i] = r.next(); return p; })();

export interface Split { nTrain: number; nVal: number; nTest: number }
/** Fractions of the dataset for train and validation (test = the rest); every bin keeps at least 5 points. */
export function sizes(trainFrac: number, valFrac: number): Split {
  const nTrain = Math.min(N - 10, Math.max(5, Math.round(N * trainFrac)));
  const nVal = Math.min(N - nTrain - 5, Math.max(5, Math.round(N * valFrac)));
  return { nTrain, nVal, nTest: N - nTrain - nVal };
}

/** Features of every point of a dataset, row-major (n × DEG), computed once: mse runs ~2,000 times per training run. */
const phiCache = new Map<Data, Float64Array>();
const phiOf = (d: Data): Float64Array => {
  let m = phiCache.get(d);
  if (!m) { m = new Float64Array(d.x.length * DEG); for (let i = 0; i < d.x.length; i++) m.set(features(d.x[i]), i * DEG); phiCache.set(d, m); }
  return m;
};

const mse = (w: Float64Array, d: Data, idx?: number[]): number => {
  const n = idx ? idx.length : d.x.length, phi = phiOf(d);
  let s = 0;
  for (let q = 0; q < n; q++) {
    const i = idx ? idx[q] : q, o = i * DEG;
    let p = 0;
    for (let k = 0; k < DEG; k++) p += w[k] * phi[o + k];
    s += (p - d.y[i]) ** 2;
  }
  return s / n;
};

export interface Run {
  split: Split;
  /** Bin of dot i (dots keep their identity when the split moves). */
  bin: Uint8Array;
  /** Pool and bin positions of every dot, in the unit square of the split strip (x right, y down). */
  from: Float32Array; to: Float32Array;
  train: Float64Array; val: Float64Array; test: Float64Array; fresh: Float64Array;
  /** Epoch with the lowest validation loss: the early-stopping point. */
  bestVal: number;
  ymax: number;
}

function layout(trainFrac: number, valFrac: number) {
  const split = sizes(trainFrac, valFrac), { nTrain, nVal } = split;
  const bin = new Uint8Array(N), idx: number[][] = [[], [], []];
  PERM.forEach((d, r) => { const b = (r < nTrain ? 0 : r < nTrain + nVal ? 1 : 2) as Bin; bin[d] = b; idx[b].push(d); });
  // bin layout: three segments along x, width proportional to size, dots in a ROWS-high grid
  const from = new Float32Array(2 * N), to = new Float32Array(2 * N), counts = [nTrain, nVal, split.nTest];
  let x0 = 0;
  for (let b = 0; b < 3; b++) {
    const w = counts[b] / N, cols = Math.ceil(counts[b] / ROWS);
    idx[b].forEach((d, q) => { to[2 * d] = x0 + w * ((Math.floor(q / ROWS) + 0.5) / cols); to[2 * d + 1] = ((q % ROWS) + 0.5) / ROWS; });
    x0 += w;
  }
  for (let d = 0; d < N; d++) { from[2 * d] = 0.04 + 0.92 * POOL[2 * d]; from[2 * d + 1] = 0.05 + 0.9 * POOL[2 * d + 1]; }
  return { split, bin, idx, from, to };
}

export function makeRun(trainFrac: number, valFrac: number): Run {
  const { split, bin, idx, from, to } = layout(trainFrac, valFrac);

  // gradient descent on train MSE; step = 1 / lam, where lam is the top eigenvalue of (1/n)ΦᵀΦ (power iteration)
  const F = idx[0].map((i) => features(DATA.x[i])), n = F.length;
  const v = new Float64Array(DEG).fill(1);
  let lam = 1;
  for (let it = 0; it < 40; it++) {
    const u = new Float64Array(DEG);
    for (const f of F) { let p = 0; for (let k = 0; k < DEG; k++) p += f[k] * v[k]; for (let k = 0; k < DEG; k++) u[k] += (f[k] * p) / n; }
    lam = Math.hypot(...u);
    for (let k = 0; k < DEG; k++) v[k] = u[k] / lam;
  }
  const lr = 0.5 / lam; // the MSE Hessian is 2·(1/n)ΦᵀΦ, so the stability limit is 1 / lam; half of it gives smooth, non-oscillating curves
  const w = new Float64Array(DEG), out = [0, 1, 2, 3].map(() => new Float64Array(EPOCHS + 1));
  for (let e = 0; e <= EPOCHS; e++) {
    out[0][e] = mse(w, DATA, idx[0]); out[1][e] = mse(w, DATA, idx[1]);
    out[2][e] = mse(w, DATA, idx[2]); out[3][e] = mse(w, FRESH);
    const g = new Float64Array(DEG);
    F.forEach((f, q) => {
      let p = 0;
      for (let k = 0; k < DEG; k++) p += w[k] * f[k];
      const r = p - DATA.y[idx[0][q]];
      for (let k = 0; k < DEG; k++) g[k] += (2 / n) * r * f[k];
    });
    for (let k = 0; k < DEG; k++) w[k] -= lr * g[k];
  }
  const arg = (a: Float64Array) => a.reduce((b, x, i) => (x < a[b] ? i : b), 0);
  const ymax = Math.min(1.6, Math.max(0.6, out[0][0] * 0.5, out[1][0] * 0.5));
  return { split, bin, from, to, train: out[0], val: out[1], test: out[2], fresh: out[3], bestVal: arg(out[1]), ymax };
}

/** Epochs are stretched in time (epoch = EPOCHS·u³): the interesting part, where validation bottoms out, happens early. */
export const epochAt = (t: number) => EPOCHS * Math.min(1, Math.max(0, (t - T_SPLIT) / (T - T_SPLIT))) ** 3;
/** Position 0..1 along the training axis (and the inverse of epochAt up to the T_SPLIT offset). */
export const uOfEpoch = (e: number) => Math.cbrt(e / EPOCHS);
export const timeOfEpoch = (e: number) => T_SPLIT + uOfEpoch(e) * (T - T_SPLIT);
export const markers = (r: Run) => [
  { at: 0, label: 'split' }, { at: T_SPLIT, label: 'train' }, { at: timeOfEpoch(r.bestVal), label: 'early stop' }, { at: T, label: 'end' },
];

export interface Frame {
  /** Dot positions in the split strip's unit square. */
  dots: Float32Array;
  epoch: number;
  phase: 'split' | 'train' | 'stop' | 'overfit';
}
const ease = (u: number) => u * u * (3 - 2 * u);
/** Everything to draw at time t. Dots flow with a small per-dot stagger, then the curves grow with the epoch. */
export function frameAt(t: number, r: Run): Frame {
  const dots = dotsAt(t, r);
  const epoch = epochAt(t);
  return { dots, epoch, phase: t < T_SPLIT ? 'split' : epoch < r.bestVal ? 'train' : epoch < EPOCHS - 1 ? 'stop' : 'overfit' };
}

/** The score you would report after early stopping on validation, and what the model really does on fresh data. */
export function scores(r: Run) {
  const e = r.bestVal;
  return { epoch: e, reported: r.test[e], real: r.fresh[e] };
}

/** x positions (0..1) of the points of a loss curve drawn up to epoch e: strictly increasing. Integer epochs while the curve moves fast, then even steps in time. */
export function curveEpochs(e: number, J = 160, EI = 30): number[] {
  const es: number[] = [];
  for (let i = 0; i <= Math.min(EI, e); i++) es.push(i);
  if (e > EI) { const u0 = uOfEpoch(EI), uc = uOfEpoch(e); for (let j = 1; j <= J; j++) es.push(EPOCHS * (u0 + ((uc - u0) * j) / J) ** 3); }
  else if (e > es[es.length - 1]) es.push(e);
  return es;
}

// ---- tuning many candidates (the "peek at test" demonstration) ----
export const DEGREES = [4, 5, 6, 7, 8, 9], RESAMPLES = 5, NCAND = DEGREES.length * RESAMPLES;
export const T_TUNE = 10;

/** Solve A x = b (n×n, Gaussian elimination with partial pivoting). */
function solve(A: Float64Array, b: Float64Array, n: number): Float64Array {
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r * n + c]) > Math.abs(A[p * n + c])) p = r;
    if (p !== c) { for (let k = 0; k < n; k++) [A[c * n + k], A[p * n + k]] = [A[p * n + k], A[c * n + k]]; [b[c], b[p]] = [b[p], b[c]]; }
    for (let r = c + 1; r < n; r++) {
      const f = A[r * n + c] / A[c * n + c];
      for (let k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
      b[r] -= f * b[c];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) { let s = b[r]; for (let k = r + 1; k < n; k++) s -= A[r * n + k] * x[k]; x[r] = s / A[r * n + r]; }
  return x;
}

export interface Tune {
  split: Split; bin: Uint8Array; from: Float32Array; to: Float32Array;
  /** Loss of each candidate on validation, test, and fresh data. */
  val: Float64Array; test: Float64Array; fresh: Float64Array;
  bestVal: number; bestTest: number;
  ymax: number;
}
/** 20 candidate models: ridge polynomials of degree 5 to 9, each fitted to a bootstrap resample of the training bin. Closed form, so instant. */
export function makeTune(trainFrac: number, valFrac: number): Tune {
  const { split, bin, idx, from, to } = layout(trainFrac, valFrac);
  const val = new Float64Array(NCAND), test = new Float64Array(NCAND), fresh = new Float64Array(NCAND);
  for (let c = 0; c < NCAND; c++) {
    const d = DEGREES[c % DEGREES.length], r = createRng(500 + c), n = idx[0].length;
    const A = new Float64Array(d * d), y = new Float64Array(d);
    for (let q = 0; q < n; q++) {
      const i = idx[0][Math.floor(r.next() * n)], f = features(DATA.x[i]);
      for (let a = 0; a < d; a++) { y[a] += f[a] * DATA.y[i]; for (let b = 0; b < d; b++) A[a * d + b] += f[a] * f[b]; }
    }
    for (let a = 0; a < d; a++) A[a * d + a] += 0.002 * n;
    const w = new Float64Array(DEG); w.set(solve(A, y, d));
    val[c] = mse(w, DATA, idx[1]); test[c] = mse(w, DATA, idx[2]); fresh[c] = mse(w, FRESH);
  }
  const arg = (a: Float64Array) => a.reduce((b, x, i) => (x < a[b] ? i : b), 0);
  return { split, bin, from, to, val, test, fresh, bestVal: arg(val), bestTest: arg(test), ymax: 0.8 };
}
export const tuneMarkers = () => [{ at: 0, label: 'split' }, { at: T_SPLIT, label: 'try settings' }, { at: T_TUNE, label: 'pick' }];
/** How many candidates have been tried by time t (fractional: the last one is still being evaluated). */
export const triedAt = (t: number) => NCAND * Math.min(1, Math.max(0, (t - T_SPLIT) / (T_TUNE - T_SPLIT)));
/** Dots for either mode (positions only). */
export function dotsAt(t: number, r: { from: Float32Array; to: Float32Array }): Float32Array {
  const dots = new Float32Array(2 * N);
  for (let d = 0; d < N; d++) {
    const u = ease(Math.min(1, Math.max(0, (t / T_SPLIT - 0.35 * (d / N)) / 0.65)));
    dots[2 * d] = r.from[2 * d] + (r.to[2 * d] - r.from[2 * d]) * u;
    dots[2 * d + 1] = r.from[2 * d + 1] + (r.to[2 * d + 1] - r.from[2 * d + 1]) * u;
  }
  return dots;
}
/** Reported test score and fresh-data score of the candidate picked by validation (honest) or by test (peeking). */
export function pick(r: Tune, peek: boolean) { const c = peek ? r.bestTest : r.bestVal; return { c, reported: r.test[c], real: r.fresh[c] }; }
