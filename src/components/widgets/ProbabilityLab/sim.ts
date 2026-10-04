/** Probability lab: samples falling into a histogram, and a Bayes population grid. Pure logic, no DOM. */
import { createRng } from '../../../lib/math/rng';

export type Mode = 'bernoulli' | 'gaussian' | 'bayes';
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const ease = (u: number) => { const c = clamp(u, 0, 1); return c * c * (3 - 2 * c); };

// ---------------------------------------------------------------- sampling modes
export const N = 400;
/** Samples arrive during [0, T_FILL]; the rest of the loop holds the finished picture. */
export const T_FILL = 13, T_SAMPLE = 16;
export const X_MIN = -8, X_MAX = 8, BINS = 32, BIN_W = (X_MAX - X_MIN) / BINS;
export const sampleMarkers = [{ at: 0, label: 'first samples' }, { at: 5, label: 'about 100' }, { at: T_FILL, label: 'all 400' }];

export interface Cfg { mode: 'bernoulli' | 'gaussian'; p: number; mu: number; sigma: number }

// Fixed underlying randomness: changing p / mu / sigma reshapes the *same* draws, so the demo is deterministic.
let base: { u: Float64Array; z: Float64Array } | undefined;
function draws() {
  if (!base) {
    const r = createRng(2024), u = new Float64Array(N), z = new Float64Array(N);
    for (let i = 0; i < N; i++) { u[i] = r.next(); z[i] = r.normal(); }
    base = { u, z };
  }
  return base;
}

export const normalPdf = (x: number, mu: number, sigma: number) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
export const trueMean = (c: Cfg) => (c.mode === 'bernoulli' ? c.p : c.mu);
export const trueVar = (c: Cfg) => (c.mode === 'bernoulli' ? c.p * (1 - c.p) : c.sigma * c.sigma);

export interface Data { cfg: Cfg; xs: Float64Array; cumMean: Float64Array; cumVar: Float64Array }
/** All N samples plus prefix statistics (mean and variance of the first k samples). */
export function makeData(cfg: Cfg): Data {
  const { u, z } = draws();
  const xs = new Float64Array(N), cumMean = new Float64Array(N + 1), cumVar = new Float64Array(N + 1);
  let s = 0, s2 = 0;
  for (let i = 0; i < N; i++) {
    xs[i] = cfg.mode === 'bernoulli' ? (u[i] < cfg.p ? 1 : 0) : cfg.mu + cfg.sigma * z[i];
    s += xs[i]; s2 += xs[i] * xs[i];
    const k = i + 1;
    cumMean[k] = s / k;
    cumVar[k] = Math.max(0, s2 / k - cumMean[k] ** 2);
  }
  return { cfg, xs, cumMean, cumVar };
}

export const binOf = (x: number) => (x < X_MIN || x >= X_MAX ? -1 : Math.floor((x - X_MIN) / BIN_W));

/** How many samples have landed at time t (fractional: the fractional part is the one still falling). */
export const countAt = (t: number) => N * Math.pow(clamp(t / T_FILL, 0, 1), 1.5);

export interface SampleFrame {
  /** Samples that have landed. */
  k: number;
  /** Fall progress 0..1 of sample k, or -1 when nothing is falling. */
  fall: number;
  /** Value of the falling sample. */
  fallX: number;
  mean: number;
  variance: number;
  /** Histogram counts: 2 bars (tails, heads) for a coin, BINS bars for the Gaussian. */
  counts: Float64Array;
}
export function frameAt(d: Data, t: number): SampleFrame {
  const n = countAt(t), k = Math.min(N, Math.floor(n + 1e-9));
  const counts = new Float64Array(d.cfg.mode === 'bernoulli' ? 2 : BINS);
  for (let i = 0; i < k; i++) {
    const b = d.cfg.mode === 'bernoulli' ? d.xs[i] : binOf(d.xs[i]);
    if (b >= 0) counts[b]++;
  }
  const falling = k < N && t > 0;
  return {
    k, fall: falling ? n - k : -1, fallX: falling ? d.xs[k] : 0,
    mean: d.cumMean[k], variance: d.cumVar[k], counts,
  };
}

// ---------------------------------------------------------------- Bayes population grid
export const POP = 1000, COLS = 50, ROWS = 20;
export const T_BAYES = 16;
export const bayesMarkers = [
  { at: 0, label: 'crowd' }, { at: 3.5, label: 'sick' }, { at: 6.5, label: 'test' },
  { at: 8.5, label: 'filter' }, { at: 10.5, label: 'fraction' },
];

export interface BayesCfg { prevalence: number; sensitivity: number; fpr: number }
export interface BayesCounts { sick: number; healthy: number; tp: number; fn: number; fp: number; tn: number; posterior: number }
/** Whole-person counts for a population of POP (rounded, so dots are never fractional). */
export function bayesCounts(c: BayesCfg): BayesCounts {
  const sick = Math.round(POP * c.prevalence), healthy = POP - sick;
  const tp = Math.round(sick * c.sensitivity), fp = Math.round(healthy * c.fpr);
  return { sick, healthy, tp, fn: sick - tp, fp, tn: healthy - fp, posterior: tp + fp > 0 ? tp / (tp + fp) : 0 };
}
/** Bayes' rule in probabilities: P(sick | positive). */
export const bayesFormula = (c: BayesCfg) =>
  (c.sensitivity * c.prevalence) / (c.sensitivity * c.prevalence + c.fpr * (1 - c.prevalence));

export interface BayesLayout {
  counts: BayesCounts;
  /** Cell coordinates of each dot in the population grid and in the compact "positives" cluster. */
  x0: Float32Array; y0: Float32Array; x1: Float32Array; y1: Float32Array;
  /** bit 0 = sick, bit 1 = tested positive. */
  flags: Uint8Array;
  /** Stagger for the appearance animation, 0..1. */
  r: Float32Array;
  /** Cluster cell size relative to the population cell. */
  scale: number;
}
export function bayesLayout(c: BayesCfg): BayesLayout {
  const counts = bayesCounts(c), rng = createRng(77);
  const perm = Array.from({ length: POP }, (_, i) => i);
  for (let i = POP - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  const x0 = new Float32Array(POP), y0 = new Float32Array(POP), x1 = new Float32Array(POP), y1 = new Float32Array(POP);
  const flags = new Uint8Array(POP), r = new Float32Array(POP);
  // dots 0..sick-1 are sick (the first tp of them test positive), the rest are healthy (the first fp test positive)
  for (let i = 0; i < POP; i++) {
    const sick = i < counts.sick;
    const pos = sick ? i < counts.tp : i - counts.sick < counts.fp;
    flags[i] = (sick ? 1 : 0) | (pos ? 2 : 0);
    x0[i] = (perm[i] % COLS) + 0.5; y0[i] = Math.floor(perm[i] / COLS) + 0.5; r[i] = rng.next();
  }
  // cluster: positives packed row by row, true positives first
  const npos = counts.tp + counts.fp, cr = Math.max(1, Math.ceil(Math.sqrt(npos / 2.5))), cc = Math.max(1, Math.ceil(npos / cr));
  const scale = Math.min(ROWS / cr, COLS / cc, 2.6);
  const order = [...Array(POP).keys()].filter((i) => flags[i] & 2).sort((a, b) => (flags[b] & 1) - (flags[a] & 1) || a - b);
  order.forEach((i, m) => {
    x1[i] = COLS / 2 + ((m % cc) + 0.5 - cc / 2) * scale;
    y1[i] = ROWS / 2 + (Math.floor(m / cc) + 0.5 - cr / 2) * scale;
  });
  for (let i = 0; i < POP; i++) if (!(flags[i] & 2)) { x1[i] = x0[i]; y1[i] = y0[i]; }
  return { counts, x0, y0, x1, y1, flags, r, scale };
}

export interface BayesFrame {
  /** 0..1 progress of each phase: dots appear, sick coloured, test result shown, negatives leave, positives regroup. */
  appear: number; sick: number; test: number; leave: number; pack: number;
}
export function bayesFrameAt(t: number): BayesFrame {
  return {
    appear: ease(t / 2.5), sick: ease((t - 3.5) / 1.5), test: ease((t - 6.5) / 1.5),
    leave: ease((t - 8.5) / 1.5), pack: ease((t - 10.5) / 2),
  };
}
