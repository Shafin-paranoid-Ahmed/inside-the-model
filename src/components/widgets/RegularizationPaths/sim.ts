/** L1 / L2 regularization paths. Quadratic loss written via its Gram form, so one solver serves
 *  the 2-weight picture and the 6-feature model. Pure logic, no DOM. */
import { createRng } from '../../../lib/math/rng';

export type Penalty = 'l1' | 'l2';
/** Loss(w) = 1/2 w'Gw - c'w + const;  penalty: L1 = lambda*|w|_1, L2 = lambda/2*|w|^2. */
export interface Gram { G: number[][]; c: number[] }

const soft = (z: number, l: number) => (z > l ? z - l : z < -l ? z + l : 0);

/** Lasso by cyclic coordinate descent (warm-startable). */
export function lasso({ G, c }: Gram, lam: number, w0?: number[]): number[] {
  const p = c.length, w = w0 ? w0.slice() : new Array(p).fill(0);
  for (let it = 0; it < 2000; it++) {
    let d = 0;
    for (let j = 0; j < p; j++) {
      let r = c[j];
      for (let k = 0; k < p; k++) if (k !== j) r -= G[j][k] * w[k];
      const nw = soft(r, lam) / G[j][j];
      d = Math.max(d, Math.abs(nw - w[j]));
      w[j] = nw;
    }
    if (d < 1e-13) break;
  }
  return w;
}

/** Ridge in closed form: (G + lambda I) w = c, by Gaussian elimination with partial pivoting. */
export function ridge({ G, c }: Gram, lam: number): number[] {
  const p = c.length, A = G.map((r, i) => [...r.map((v, j) => v + (i === j ? lam : 0)), c[i]]);
  for (let i = 0; i < p; i++) {
    let m = i;
    for (let r = i + 1; r < p; r++) if (Math.abs(A[r][i]) > Math.abs(A[m][i])) m = r;
    [A[i], A[m]] = [A[m], A[i]];
    for (let r = i + 1; r < p; r++) { const f = A[r][i] / A[i][i]; for (let k = i; k <= p; k++) A[r][k] -= f * A[i][k]; }
  }
  const w = new Array(p).fill(0);
  for (let i = p - 1; i >= 0; i--) { let s = A[i][p]; for (let k = i + 1; k < p; k++) s -= A[i][k] * w[k]; w[i] = s / A[i][i]; }
  return w;
}
export const solve = (g: Gram, pen: Penalty, lam: number, w0?: number[]) => (pen === 'l1' ? lasso(g, lam, w0) : ridge(g, lam));

// ---- the two-weight problem ----
export const G2: Gram = { G: [[1, 0.6], [0.6, 1]], c: [2.02, 1.66] };
/** Unconstrained optimum of the 2-weight problem: solves G w = c, i.e. (1.6, 0.7). */
export const W2_OLS = ridge(G2, 0);
/** Loss above the minimum: 1/2 (w - w*)' G (w - w*). */
export const excess2 = (w: number[]) => {
  const d0 = w[0] - W2_OLS[0], d1 = w[1] - W2_OLS[1];
  return 0.5 * (d0 * (G2.G[0][0] * d0 + G2.G[0][1] * d1) + d1 * (G2.G[1][0] * d0 + G2.G[1][1] * d1));
};
/** Contour of the 2-weight loss at excess `dl`, as an ellipse polyline [x0,y0,x1,y1,...]. */
export function ellipse(dl: number, n = 72): number[] {
  const [a, b, d] = [G2.G[0][0], G2.G[0][1], G2.G[1][1]], tr = a + d, det = a * d - b * b, s = Math.sqrt((tr * tr) / 4 - det);
  const l1 = tr / 2 + s, l2 = tr / 2 - s, th = 0.5 * Math.atan2(2 * b, a - d), ct = Math.cos(th), st = Math.sin(th);
  const out: number[] = [];
  for (let i = 0; i <= n; i++) {
    const q = (i / n) * 2 * Math.PI, u = Math.sqrt((2 * dl) / l1) * Math.cos(q), v = Math.sqrt((2 * dl) / l2) * Math.sin(q);
    out.push(W2_OLS[0] + u * ct - v * st, W2_OLS[1] + u * st + v * ct);
  }
  return out;
}

// ---- the six-feature model ----
export const P = 6, N_TRAIN = 30, N_VAL = 400;
export const NAMES = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'];
/** Only x1..x3 matter; x4 is a noisy copy of x1; x5, x6 are pure noise. */
export const TRUE_W = [2.5, -1.8, 1.2, 0, 0, 0];
const SEED = 3;
const rng = createRng(SEED);
function make(n: number) {
  const X: number[][] = [], y: number[] = [];
  for (let i = 0; i < n; i++) {
    const q = rng.normal(), z = Array.from({ length: P }, () => rng.normal());
    const x = z.map((v) => 0.5 * q + 0.87 * v);
    x[3] = 0.6 * x[0] + 0.8 * z[3];
    X.push(x); y.push(x.reduce((s, v, j) => s + v * TRUE_W[j], 0) + rng.normal(0, 2));
  }
  return { X, y };
}
const tr = make(N_TRAIN), va = make(N_VAL);
const mu = Array.from({ length: P }, (_, j) => tr.X.reduce((s, r) => s + r[j], 0) / N_TRAIN);
const sd = mu.map((m, j) => Math.sqrt(tr.X.reduce((s, r) => s + (r[j] - m) ** 2, 0) / N_TRAIN));
const my = tr.y.reduce((s, v) => s + v, 0) / N_TRAIN;
const prep = (d: { X: number[][]; y: number[] }) => ({ X: d.X.map((r) => r.map((v, j) => (v - mu[j]) / sd[j])), y: d.y.map((v) => v - my) });
const TR = prep(tr), VA = prep(va);
/** Standardized columns => G[j][j] = 1. */
export const G6: Gram = {
  G: Array.from({ length: P }, (_, j) => Array.from({ length: P }, (_, k) => TR.X.reduce((s, r) => s + r[j] * r[k], 0) / N_TRAIN)),
  c: Array.from({ length: P }, (_, j) => TR.X.reduce((s, r, i) => s + r[j] * TR.y[i], 0) / N_TRAIN),
};
const mse = (d: { X: number[][]; y: number[] }, w: number[]) =>
  d.X.reduce((s, r, i) => s + (r.reduce((a, v, j) => a + v * w[j], 0) - d.y[i]) ** 2, 0) / d.y.length;

// ---- precomputed paths ----
export const N = 241;
/** Largest lambda on the axis: the smallest one that zeroes every lasso coefficient. */
export const LMAX = Math.max(...G6.c.map(Math.abs));
export interface Path { lam: Float64Array; w6: Float64Array; w2: Float64Array; train: Float64Array; val: Float64Array; best: number }
function build(pen: Penalty): Path {
  const lam = new Float64Array(N), w6 = new Float64Array(N * P), w2 = new Float64Array(N * 2), train = new Float64Array(N), val = new Float64Array(N);
  let a: number[] | undefined, b: number[] | undefined;
  for (let i = N - 1; i >= 0; i--) { // warm-start from the sparse end
    const l = (i / (N - 1)) * LMAX;
    a = solve(G6, pen, l, a); b = solve(G2, pen, l, b);
    lam[i] = l; w6.set(a, i * P); w2.set(b, i * 2);
    train[i] = mse(TR, a); val[i] = mse(VA, a);
  }
  let best = 0;
  for (let i = 1; i < N; i++) if (val[i] < val[best]) best = i;
  return { lam, w6, w2, train, val, best };
}
export const PATHS: Record<Penalty, Path> = { l1: build('l1'), l2: build('l2') };

// ---- timeline ----
export const T = 14, T_HOLD = 2, T_END = 12;
export const MARKERS = [{ at: 0, label: 'Start' }, { at: T_HOLD, label: 'Raise λ' }, { at: T_END, label: 'Strong λ' }];
export type Phase = 'no penalty' | 'raise λ' | 'strong penalty';
/** Fraction of the lambda axis reached at time t (eased at both ends). */
export const uAt = (t: number) => { const s = Math.min(1, Math.max(0, (t - T_HOLD) / (T_END - T_HOLD))); return s * s * (3 - 2 * s); };

export interface Frame {
  lam: number; k: number; phase: Phase;
  w6: number[]; w2: [number, number];
  /** Size of the constraint region that gives the same 2-weight solution: |w|_1 or |w|_2. */
  radius: number;
  zeros: number; norm6: number; train: number; val: number;
}
export function frameAt(t: number, pen: Penalty): Frame {
  const p = PATHS[pen], k = uAt(t) * (N - 1), i = Math.min(N - 2, Math.floor(k)), f = k - i;
  const mix = (arr: Float64Array, m: number, j: number) => arr[i * m + j] * (1 - f) + arr[(i + 1) * m + j] * f;
  const w6 = Array.from({ length: P }, (_, j) => mix(p.w6, P, j)), w2: [number, number] = [mix(p.w2, 2, 0), mix(p.w2, 2, 1)];
  // interpolating two sparse solutions can leave a tiny non-zero; snap when the nearer sample is exactly zero
  const near = (arr: Float64Array, m: number, j: number) => (f < 0.5 ? arr[i * m + j] : arr[(i + 1) * m + j]);
  for (let j = 0; j < P; j++) if (near(p.w6, P, j) === 0) w6[j] = 0;
  for (let j = 0; j < 2; j++) if (near(p.w2, 2, j) === 0) w2[j] = 0;
  return {
    lam: mix(p.lam, 1, 0), k, phase: t < T_HOLD ? 'no penalty' : t < T_END ? 'raise λ' : 'strong penalty',
    w6, w2, radius: pen === 'l1' ? Math.abs(w2[0]) + Math.abs(w2[1]) : Math.hypot(w2[0], w2[1]),
    zeros: w6.filter((v) => v === 0).length, norm6: pen === 'l1' ? w6.reduce((s, v) => s + Math.abs(v), 0) : Math.hypot(...w6),
    train: mix(p.train, 1, 0), val: mix(p.val, 1, 0),
  };
}
