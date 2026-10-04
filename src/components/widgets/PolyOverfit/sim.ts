/** Polynomial fits of degree 1..12 to noisy samples of a smooth curve; bias-variance ensemble. Pure logic, no DOM. */
import { createRng } from '../../../lib/math/rng';

export const truth = (x: number) => 0.8 * Math.sin(3 * x);
export const SIGMA = 0.25, N = 14, D_MAX = 12, NG = 121, K_ENS = 40, N_GHOST = 20;
export const NOISE = SIGMA * SIGMA;
export const GX = Float64Array.from({ length: NG }, (_, g) => -1 + (2 * g) / (NG - 1));
export const GT = GX.map(truth);

/** Chebyshev basis T_0..T_d at x in [-1,1]: same polynomial space as 1,x,..,x^d but well conditioned. */
export function basis(x: number, d: number, out = new Float64Array(d + 1)): Float64Array {
  out[0] = 1; if (d >= 1) out[1] = x;
  for (let k = 2; k <= d; k++) out[k] = 2 * x * out[k - 1] - out[k - 2];
  return out;
}

/** Least squares by normal equations with partial pivoting. Returns Chebyshev coefficients. */
export function fitPoly(xs: ArrayLike<number>, ys: ArrayLike<number>, d: number): Float64Array {
  const m = d + 1, A = Array.from({ length: m }, () => new Float64Array(m + 1)), b = new Float64Array(m);
  for (let i = 0; i < xs.length; i++) {
    const phi = basis(xs[i], d);
    for (let r = 0; r < m; r++) { b[r] += phi[r] * ys[i]; for (let c = 0; c < m; c++) A[r][c] += phi[r] * phi[c]; }
  }
  for (let r = 0; r < m; r++) { A[r][r] += 1e-10; A[r][m] = b[r]; } // tiny ridge: keeps rank-deficient cases solvable
  for (let c = 0; c < m; c++) {
    let p = c; for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = c + 1; r < m; r++) { const f = A[r][c] / A[c][c]; for (let k = c; k <= m; k++) A[r][k] -= f * A[c][k]; }
  }
  const w = new Float64Array(m);
  for (let r = m - 1; r >= 0; r--) { let s = A[r][m]; for (let k = r + 1; k < m; k++) s -= A[r][k] * w[k]; w[r] = s / A[r][r]; }
  return w;
}

export const evalPoly = (w: Float64Array, x: number) => { const p = basis(x, w.length - 1); let s = 0; for (let k = 0; k < w.length; k++) s += w[k] * p[k]; return s; };
const onGrid = (w: Float64Array) => Float64Array.from(GX, (x) => evalPoly(w, x));

/** A fixed noisy test set (same distribution), 200 points. */
const tr = createRng(999);
export const TEST = Array.from({ length: 200 }, (_, i) => { const x = -1 + (2 * i) / 199; return { x, y: truth(x) + tr.normal(0, SIGMA) }; });
const mse = (w: Float64Array, pts: { x: number; y: number }[]) => pts.reduce((s, p) => s + (evalPoly(w, p.x) - p.y) ** 2, 0) / pts.length;

export function sample(seed: number) {
  const r = createRng(seed), xs = new Float64Array(N), ys = new Float64Array(N);
  for (let i = 0; i < N; i++) { xs[i] = -0.95 + (1.9 * i) / (N - 1) + r.uniform(-0.04, 0.04); ys[i] = truth(xs[i]) + r.normal(0, SIGMA); }
  return { xs, ys };
}

export interface Model {
  seed: number; xs: Float64Array; ys: Float64Array;
  /** Index d-1 for degree d. */
  curves: Float64Array[]; train: number[]; test: number[];
}
export function fitModel(seed: number): Model {
  const { xs, ys } = sample(seed), pts = Array.from(xs, (x, i) => ({ x, y: ys[i] }));
  const ws = Array.from({ length: D_MAX }, (_, i) => fitPoly(xs, ys, i + 1));
  return { seed, xs, ys, curves: ws.map(onGrid), train: ws.map((w) => mse(w, pts)), test: ws.map((w) => mse(w, TEST)) };
}
export const SEED0 = 3;
export const BASE = fitModel(SEED0);

/** Fixed ensemble of datasets: expected behavior over the randomness of the sample. */
export const ENSEMBLE = Array.from({ length: K_ENS }, (_, k) => fitModel(5000 + k).curves);
export const BIAS2: number[] = [], VAR: number[] = [];
for (let d = 0; d < D_MAX; d++) {
  let b = 0, v = 0;
  for (let g = 0; g < NG; g++) {
    let m = 0; for (const e of ENSEMBLE) m += e[d][g]; m /= K_ENS;
    let s = 0; for (const e of ENSEMBLE) s += (e[d][g] - m) ** 2;
    b += (m - GT[g]) ** 2; v += s / K_ENS;
  }
  BIAS2.push(b / NG); VAR.push(v / NG);
}
/** Degree with the lowest expected test error (bias² + variance). */
export const BEST_DEG = BIAS2.reduce((bi, _, i) => (BIAS2[i] + VAR[i] < BIAS2[bi] + VAR[bi] ? i : bi), 0) + 1;

// ---- timeline ----
export const T_HOLD = 1.5, T_SWEEP = 12, T_END = 2, T_TOTAL = T_HOLD + T_SWEEP + T_END;
export const timeOfDegree = (d: number) => T_HOLD + ((d - 1) / (D_MAX - 1)) * T_SWEEP;
export const degreeAt = (t: number) => 1 + (D_MAX - 1) * Math.min(1, Math.max(0, (t - T_HOLD) / T_SWEEP));

export type Phase = 'Underfit' | 'Good fit' | 'Overfit';
const mix = (a: number, b: number, f: number) => a + (b - a) * f;
/** Expected error (bias² + variance) at fractional degree d. */
const expected = (d: number) => { const i = Math.min(D_MAX - 2, Math.floor(d - 1)), f = d - 1 - i; return mix(BIAS2[i] + VAR[i], BIAS2[i + 1] + VAR[i + 1], f); };
/** Good fit: expected error within 1.5x of its minimum. */
export const phaseOf = (d: number): Phase => (expected(d) <= 1.5 * expected(BEST_DEG) ? 'Good fit' : d < BEST_DEG ? 'Underfit' : 'Overfit');
const firstDegree = (ph: Phase) => { for (let d = 1; d <= D_MAX; d += 0.01) if (phaseOf(d) === ph) return d; return D_MAX; };
export const MARKERS = [{ at: 0, label: 'Underfit' }, { at: timeOfDegree(firstDegree('Good fit')), label: 'Good fit' }, { at: timeOfDegree(firstDegree('Overfit')), label: 'Overfit' }];

function at<T>(d: number, get: (i: number) => T, lerp: (a: T, b: T, f: number) => T): T {
  const i = Math.min(D_MAX - 2, Math.floor(d - 1)), f = d - 1 - i;
  return lerp(get(i), get(i + 1), f);
}
const lerpArr = (a: Float64Array, b: Float64Array, f: number) => Float64Array.from(a, (v, g) => mix(v, b[g], f));

export interface Frame { d: number; curve: Float64Array; train: number; test: number; bias2: number; variance: number; phase: Phase }
/** Everything to draw at fractional degree d (tweens between the neighbouring fitted curves). */
export function frameAtDegree(d: number, m: Model = BASE): Frame {
  d = Math.min(D_MAX, Math.max(1, d));
  const n = (arr: number[]) => at(d, (i) => arr[i], mix);
  return { d, curve: at(d, (i) => m.curves[i], lerpArr), train: n(m.train), test: n(m.test), bias2: n(BIAS2), variance: n(VAR), phase: phaseOf(d) };
}
export const frameAt = (t: number, m: Model = BASE) => frameAtDegree(degreeAt(t), m);
/** Ensemble member k's fitted curve at fractional degree d. */
export const ghostAt = (k: number, d: number) => at(Math.min(D_MAX, Math.max(1, d)), (i) => ENSEMBLE[k][i], lerpArr);
