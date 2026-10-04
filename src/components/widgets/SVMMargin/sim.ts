/** Pure logic for SVMMargin: a linear SVM, a lifted 1-D SVM, an RBF-kernel SVM and the polynomial kernel identity. No DOM. */
import { createRng } from '../../../lib/math/rng';

export type Mode = 'margin' | 'soft' | 'lift' | 'kernel' | 'identity';
export type KData = 'rings' | 'xor';
export interface Pt { x: number; y: number; c: 0 | 1 }

/** The plane shows [-DOM, DOM]^2. */
export const DOM = 1.5;
/** The bias is folded in as a constant feature of value B, so b is only weakly penalised (penalty (b/B)^2 / 2). */
export const B = 3;
const B2 = B * B;
const sg = (c: 0 | 1) => (c ? 1 : -1);
const ease = (u: number) => 1 - (1 - u) * (1 - u);
const clamp01 = (u: number) => Math.min(1, Math.max(0, u));
const clip = (v: number) => Math.max(-DOM + 0.08, Math.min(DOM - 0.08, v));
const phaseOf = (t: number, ms: readonly { at: number; label: string }[]) => ms.reduce((s, m) => (m.at <= t + 1e-9 ? m.label : s), ms[0].label);

/* ------------------------------------------------------------------ data */

/** Two blobs. `overlap: false` keeps only points at least 0.18 from a common direction, so a separating line exists. */
export function blobs(overlap: boolean, seed = overlap ? 5 : 2): Pt[] {
  const rng = createRng(seed + (overlap ? 100 : 0));
  const pts: Pt[] = [];
  const ux = 0.78, uy = 0.625;
  for (const c of [0, 1] as const) {
    const n = overlap ? 14 : 10, s = sg(c), cx = overlap ? 0.42 : 0.55, cy = overlap ? 0.34 : 0.45, sd = overlap ? 0.38 : 0.27;
    let k = 0;
    while (k < n) {
      const x = clip(rng.normal(cx * s, sd)), y = clip(rng.normal(cy * s, sd));
      if (!overlap && s * (ux * x + uy * y) < 0.18) continue;
      pts.push({ x, y, c }); k++;
    }
  }
  return pts;
}

/** Ring data (class 1 inside, class 0 outside) or XOR corners, each with two deliberately mislabelled points. */
export function kernelData(kind: KData, seed = 9): Pt[] {
  const rng = createRng(seed + (kind === 'rings' ? 1 : 2));
  const pts: Pt[] = [];
  if (kind === 'rings') {
    for (let i = 0; i < 16; i++) { const r = rng.normal(0.42, 0.07), a = rng.uniform(0, 2 * Math.PI); pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), c: 1 }); }
    for (let i = 0; i < 24; i++) { const r = rng.normal(1.05, 0.08), a = rng.uniform(0, 2 * Math.PI); pts.push({ x: clip(r * Math.cos(a)), y: clip(r * Math.sin(a)), c: 0 }); }
    pts.push({ x: 0.22, y: -0.18, c: 0 }, { x: -0.8, y: 0.62, c: 1 });
  } else {
    for (let i = 0; i < 40; i++) {
      const qx = i % 2 ? 1 : -1, qy = (i >> 1) % 2 ? 1 : -1;
      pts.push({ x: clip(rng.normal(0.7 * qx, 0.24)), y: clip(rng.normal(0.7 * qy, 0.24)), c: qx * qy > 0 ? 1 : 0 });
    }
    pts.push({ x: 0.75, y: 0.7, c: 0 }, { x: -0.7, y: 0.68, c: 1 });
  }
  return pts;
}

/** 1-D data: class 1 near the origin, class 0 on both sides. No single threshold separates them. */
export function lineData(seed = 3): { x: number; c: 0 | 1 }[] {
  const rng = createRng(seed);
  const out: { x: number; c: 0 | 1 }[] = [];
  for (let i = 0; i < 12; i++) out.push({ x: rng.uniform(-0.45, 0.45), c: 1 });
  for (let i = 0; i < 16; i++) out.push({ x: (i % 2 ? 1 : -1) * rng.uniform(0.9, 1.4), c: 0 });
  return out;
}

/* ------------------------------------------------------------ dual solver */

/**
 * Dual coordinate descent for the L1-loss SVM (Hsieh et al. 2008): maximise sum(a) - 1/2 sum a_i a_j y_i y_j K_ij with 0 <= a <= C.
 * `K` must already include the +B^2 bias term. Returns the alpha after every sweep (row 0 = start).
 */
export function dualSweeps(K: Float64Array, y: ArrayLike<number>, C: number, sweeps: number, a0?: ArrayLike<number>): Float64Array[] {
  const n = y.length, a = Float64Array.from(a0 ?? new Float64Array(n));
  const f = new Float64Array(n); // f_i = sum_j a_j y_j K_ij
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) f[i] += a[j] * y[j] * K[i * n + j];
  const out = [Float64Array.from(a)];
  for (let s = 0; s < sweeps; s++) {
    for (let i = 0; i < n; i++) {
      const g = y[i] * f[i] - 1, q = K[i * n + i];
      const an = Math.min(C, Math.max(0, a[i] - g / q)), d = an - a[i];
      if (d !== 0) { a[i] = an; for (let j = 0; j < n; j++) f[j] += d * y[i] * K[j * n + i]; }
    }
    out.push(Float64Array.from(a));
  }
  return out;
}

/* ---------------------------------------------------------------- linear */

export interface LinRun {
  pts: Pt[]; C: number;
  /** Snapshots [wx, wy, b] for steps 0..K. */
  W: Float64Array; K: number;
  /** The exact optimum [wx, wy, b] from the dual. */
  star: Float64Array;
  /** Primal objective after every step. */
  obj: Float64Array;
}

/** Primal objective 1/2(|w|^2 + (b/B)^2) + C sum hinge. */
export function objective(pts: readonly Pt[], w: ArrayLike<number>, C: number) {
  let h = 0;
  for (const p of pts) h += Math.max(0, 1 - sg(p.c) * (w[0] * p.x + w[1] * p.y + w[2]));
  return 0.5 * (w[0] * w[0] + w[1] * w[1] + (w[2] * w[2]) / B2) + C * h;
}

/** Exact linear SVM by dual coordinate descent. */
export function solveLinear(pts: readonly Pt[], C: number, sweeps = 3000): Float64Array {
  const n = pts.length, K = new Float64Array(n * n), y = pts.map((p) => sg(p.c));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] = pts[i].x * pts[j].x + pts[i].y * pts[j].y + B2;
  const a = dualSweeps(K, y, C, sweeps).pop()!;
  const w = new Float64Array(3);
  for (let i = 0; i < n; i++) { w[0] += a[i] * y[i] * pts[i].x; w[1] += a[i] * y[i] * pts[i].y; w[2] += a[i] * y[i] * B2; }
  return w;
}

export const STEPS = 160;

/**
 * Subgradient descent on the hinge objective from `w0`, with the Polyak step (J - J*) / |g|^2 using the known optimum J*.
 * The last snapshot is the exact optimum, so the path ends where the dual says it should.
 */
export function trainLinear(pts: readonly Pt[], C: number, w0: ArrayLike<number> = [0, 0, 0], steps = STEPS): LinRun {
  const star = solveLinear(pts, C), Js = objective(pts, star, C);
  const w = Float64Array.from(w0), g = new Float64Array(3);
  const W = new Float64Array((steps + 1) * 3), obj = new Float64Array(steps + 1);
  for (let k = 0; k <= steps; k++) {
    // ponytail: Polyak steps creep near the optimum, so the last 30% of the path is blended onto the exact dual solution.
    const beta = clamp01((k / steps - 0.7) / 0.3);
    for (let j = 0; j < 3; j++) w[j] += beta * beta * (star[j] - w[j]);
    W.set(w, k * 3); obj[k] = objective(pts, w, C);
    g[0] = w[0]; g[1] = w[1]; g[2] = w[2] / B2;
    for (const p of pts) {
      const s = sg(p.c);
      if (1 - s * (w[0] * p.x + w[1] * p.y + w[2]) > 0) { g[0] -= C * s * p.x; g[1] -= C * s * p.y; g[2] -= C * s; }
    }
    const gg = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    if (gg > 1e-18) { const eta = (Math.max(0, obj[k] - Js) / gg) * 0.35; for (let j = 0; j < 3; j++) w[j] -= eta * g[j]; }
  }
  return { pts: pts.slice(), C, W, K: steps, star, obj };
}

/** Margin width 2/|w| of a line w.x + b (Infinity for w = 0). */
export const marginWidth = (w: ArrayLike<number>) => { const n = Math.hypot(w[0], w[1]); return n > 1e-12 ? 2 / n : Infinity; };
/** Functional margin y (w.x + b) of a point. */
export const fmargin = (w: ArrayLike<number>, p: Pt) => sg(p.c) * (w[0] * p.x + w[1] * p.y + w[2]);

/** Points on or inside the margin (y f(x) <= 1 + tol). For the optimum these are the support vectors. */
export function supportIdx(pts: readonly Pt[], w: ArrayLike<number>, tol = 0.03): number[] {
  const out: number[] = [];
  pts.forEach((p, i) => { if (fmargin(w, p) <= 1 + tol) out.push(i); });
  return out;
}

export interface Cand { w: Float64Array; width: number }
/**
 * Separating lines for the fan: for `n` directions spread across the range that still separates the classes,
 * the best offset (midway between the closest points). `w` is scaled so the closest point has y f = 1; width = 2/|w|.
 */
export function fanLines(pts: readonly Pt[], n = 9): Cand[] {
  const cand = (th: number): Cand | null => {
    const ux = Math.cos(th), uy = Math.sin(th);
    let lo1 = Infinity, hi0 = -Infinity;
    for (const p of pts) { const d = ux * p.x + uy * p.y; if (p.c) lo1 = Math.min(lo1, d); else hi0 = Math.max(hi0, d); }
    const gap = lo1 - hi0;
    if (!(gap > 1e-6)) return null;
    const s = 2 / gap, off = (lo1 + hi0) / 2;
    const w = Float64Array.of(s * ux, s * uy, -s * off);
    return { w, width: gap };
  };
  // Widest direction first, then extend both ways until the classes stop being separable.
  const N = 720, at = (i: number) => (i / N) * 2 * Math.PI;
  let best = -1;
  for (let i = 0; i < N; i++) { const c = cand(at(i)); if (c && (best < 0 || c.width > cand(at(best))!.width)) best = i; }
  if (best < 0) return [];
  let lo = best, hi = best;
  while (cand(at(lo - 1))) lo--;
  while (cand(at(hi + 1))) hi++;
  const a = at(lo), b = at(hi);
  const out: Cand[] = [];
  for (let i = 0; i < n; i++) { const c = cand(a + ((b - a) * (i + 0.5)) / n); if (c) out.push(c); }
  // The narrowest line goes last: it is where training starts.
  const m = out.reduce((bi, c, i) => (c.width < out[bi].width ? i : bi), 0);
  out.push(...out.splice(m, 1));
  return out;
}

/** Timeline constants for the two linear modes. */
export const T_FAN = 3;
export const MARKERS_MARGIN = [{ at: 0, label: 'many lines' }, { at: 3, label: 'widening' }, { at: 8.4, label: 'settled' }];
export const MARKERS_SOFT = [{ at: 0, label: 'start' }, { at: 1.2, label: 'fitting' }, { at: 5.6, label: 'settled' }];
export const T_MARGIN = 10;
export const T_SOFT = 7;
const T_TRAIN_MARGIN = 5;
const T_TRAIN_SOFT = 5;

export interface LinFrame {
  w: Float64Array; k: number; phase: string;
  /** Fan phase: index of the highlighted candidate (-1 otherwise) and how visible the fan is (0..1). */
  hi: number; fan: number;
  width: number; hinge: number; reg: number; viol: number; nsv: number; acc: number;
}

/** Everything needed to draw the linear SVM at time t. `fan` = start with the fan of separating lines (the `margin` mode). */
export function linearFrameAt(t: number, run: LinRun, cands: readonly Cand[] | null): LinFrame {
  const fan = !!cands && cands.length > 0;
  const ms = fan ? MARKERS_MARGIN : MARKERS_SOFT;
  const w = new Float64Array(3);
  let k = 0, hi = -1, fv = 0;
  if (fan && t < T_FAN) {
    hi = Math.min(cands!.length - 1, Math.floor((t / T_FAN) * cands!.length));
    w.set(cands![hi].w); fv = 1;
  } else {
    const u = clamp01((t - (fan ? T_FAN : 0)) / (fan ? T_TRAIN_MARGIN : T_TRAIN_SOFT));
    k = run.K * ease(u);
    const i = Math.min(run.K - 1, Math.floor(k)), f = k - i;
    for (let j = 0; j < 3; j++) w[j] = run.W[i * 3 + j] * (1 - f) + run.W[(i + 1) * 3 + j] * f;
    if (fan) fv = 1 - clamp01((t - T_FAN) / 1.2);
  }
  let hinge = 0, viol = 0, ok = 0;
  for (const p of run.pts) { const m = fmargin(w, p); hinge += Math.max(0, 1 - m); if (m < 0.98) viol++; if (m > 0) ok++; }
  return {
    w, k, phase: phaseOf(t, ms), hi, fan: fv, width: marginWidth(w), hinge, reg: 0.5 * (w[0] * w[0] + w[1] * w[1]),
    viol, nsv: supportIdx(run.pts, w).length, acc: ok / run.pts.length,
  };
}

/** The start line for the fan mode: the last candidate, which is the narrowest, so training visibly widens the margin. */
export const startLine = (cands: readonly Cand[]): Float64Array => cands[cands.length - 1].w;

/* ------------------------------------------------------------------ lift */

export const T_LIFT = 12;
export const MARKERS_LIFT = [{ at: 0, label: 'line' }, { at: 1.5, label: 'lift' }, { at: 5, label: 'flat cut' }, { at: 8, label: 'back down' }];
export const LIFT_Y = 2.4;

export interface LiftRun { data: { x: number; c: 0 | 1 }[]; pts: Pt[]; C: number; w: Float64Array; width: number; roots: number[] }

/** Fit a linear SVM in the lifted space phi(x) = (x, x^2). */
export function trainLift(data = lineData(), C = 100): LiftRun {
  const pts: Pt[] = data.map((d) => ({ x: d.x, y: d.x * d.x, c: d.c }));
  const w = solveLinear(pts, C);
  // roots of w0 x + w1 x^2 + b = 0
  const [a, b, c] = [w[1], w[0], w[2]], roots: number[] = [];
  if (Math.abs(a) > 1e-9) { const D = b * b - 4 * a * c; if (D >= 0) roots.push((-b - Math.sqrt(D)) / (2 * a), (-b + Math.sqrt(D)) / (2 * a)); roots.sort((p, q) => p - q); }
  return { data, pts, C, w, width: marginWidth(w), roots };
}

export interface LiftFrame {
  /** Lift amount 0..1: a point sits at (x, s x^2). */
  s: number;
  /** Visibility of the flat cut and margin lines (0..1). */
  cut: number;
  /** How far the cut has come back down to the number line (0..1). */
  back: number;
  phase: string;
  /** Points separated correctly by a single threshold on the line, and by the lifted plane. */
  sep: boolean;
}

export function liftFrameAt(t: number): LiftFrame {
  const s = clamp01((t - 1.5) / 3.5), cut = clamp01((t - 5) / 1.5), back = clamp01((t - 8) / 3);
  const e = (u: number) => u * u * (3 - 2 * u);
  return { s: e(s), cut: e(cut), back: e(back), phase: phaseOf(t, MARKERS_LIFT), sep: s >= 1 };
}

/** Best accuracy of any single threshold on the original line (either orientation). */
export function bestThreshold(data: readonly { x: number; c: 0 | 1 }[]): number {
  const xs = data.map((d) => d.x).sort((a, b) => a - b);
  let best = 0;
  for (let i = 0; i <= xs.length; i++) {
    const th = i === 0 ? xs[0] - 1 : i === xs.length ? xs[xs.length - 1] + 1 : (xs[i - 1] + xs[i]) / 2;
    const hit = data.filter((d) => (d.x > th ? 1 : 0) === d.c).length;
    best = Math.max(best, hit, data.length - hit);
  }
  return best / data.length;
}

/* ---------------------------------------------------------------- kernel */

export const SWEEPS = 40;
export const T_KERNEL = 9;
export const MARKERS_KERNEL = [{ at: 0, label: 'start' }, { at: 1.5, label: 'fitting' }, { at: 6, label: 'settled' }];
export const KC = 10;
export const GAMMAS = [0.2, 0.5, 1, 2, 5, 10, 30, 100];

export const rbf = (g: number, dx: number, dy: number) => Math.exp(-g * (dx * dx + dy * dy));

export interface KernelRun { pts: Pt[]; gamma: number; A: Float64Array[]; y: number[] }

/** Dual coordinate descent with the RBF kernel; keeps alpha after every sweep so the timeline only interpolates. */
export function trainKernel(pts: readonly Pt[], gamma: number, sweeps = SWEEPS, C = KC): KernelRun {
  const n = pts.length, K = new Float64Array(n * n), y = pts.map((p) => sg(p.c));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] = rbf(gamma, pts[i].x - pts[j].x, pts[i].y - pts[j].y) + 1;
  return { pts: pts.slice(), gamma, A: dualSweeps(K, y, C, sweeps), y };
}

/** Decision value f(x, y) = sum a_i y_i (k(x_i, x) + 1). */
export function kscore(run: KernelRun, a: ArrayLike<number>, x: number, y: number): number {
  let f = 0;
  for (let i = 0; i < run.pts.length; i++) if (a[i] > 1e-9) f += a[i] * run.y[i] * (rbf(run.gamma, run.pts[i].x - x, run.pts[i].y - y) + 1);
  return f;
}

export interface KernelFrame { a: Float64Array; k: number; phase: string; nsv: number; acc: number }
const T_TRAIN_K = 6;
export function kernelFrameAt(t: number, run: KernelRun): KernelFrame {
  const S = run.A.length - 1, k = S * ease(clamp01(t / T_TRAIN_K)), i = Math.min(S - 1, Math.floor(k)), f = k - i, n = run.pts.length;
  const a = new Float64Array(n);
  for (let j = 0; j < n; j++) a[j] = run.A[i][j] * (1 - f) + run.A[i + 1][j] * f;
  let nsv = 0, ok = 0;
  for (let j = 0; j < n; j++) if (a[j] > 1e-6) nsv++;
  for (const p of run.pts) if ((kscore(run, a, p.x, p.y) > 0 ? 1 : 0) === p.c) ok++;
  return { a, k, phase: phaseOf(t, MARKERS_KERNEL), nsv, acc: ok / n };
}

/** Decision values on an n x n grid of cell centres over the plane; row 0 is the bottom. */
export function kfield(run: KernelRun, a: ArrayLike<number>, n: number, out: Float32Array) {
  for (let j = 0; j < n; j++) {
    const y = -DOM + ((j + 0.5) / n) * 2 * DOM;
    for (let i = 0; i < n; i++) out[j * n + i] = kscore(run, a, -DOM + ((i + 0.5) / n) * 2 * DOM, y);
  }
}

/* -------------------------------------------------------------- identity */

export const T_ID = 8;
export const MARKERS_ID = [{ at: 0, label: 'aligned' }, { at: 2, label: 'orthogonal' }, { at: 4, label: 'opposite' }, { at: 6, label: 'orthogonal again' }];
export const ID_X: readonly [number, number] = [0.8, 0.45];
export const ID_R = 1;

/** Feature map of the homogeneous degree-2 polynomial kernel in 2-D. */
export const phi2 = (x: number, y: number): [number, number, number] => [x * x, Math.SQRT2 * x * y, y * y];
/** k(x, z) = (x . z)^2, computed without building phi. */
export const polyK = (x: readonly number[], z: readonly number[]) => (x[0] * z[0] + x[1] * z[1]) ** 2;

export interface IdFrame {
  x: [number, number]; z: [number, number]; angle: number;
  px: [number, number, number]; pz: [number, number, number]; prod: [number, number, number];
  direct: number; lifted: number; phase: string;
}

/** z circles x once per cycle (starting aligned); `zo` overrides z when the learner drags it. */
export function identityFrameAt(t: number, zo?: readonly [number, number] | null): IdFrame {
  const x: [number, number] = [ID_X[0], ID_X[1]], a0 = Math.atan2(x[1], x[0]);
  const angle = (2 * Math.PI * t) / T_ID;
  const z: [number, number] = zo ? [zo[0], zo[1]] : [ID_R * Math.cos(a0 + angle), ID_R * Math.sin(a0 + angle)];
  const px = phi2(x[0], x[1]), pz = phi2(z[0], z[1]);
  const prod: [number, number, number] = [px[0] * pz[0], px[1] * pz[1], px[2] * pz[2]];
  return { x, z, angle, px, pz, prod, direct: polyK(x, z), lifted: prod[0] + prod[1] + prod[2], phase: phaseOf(t, MARKERS_ID) };
}
