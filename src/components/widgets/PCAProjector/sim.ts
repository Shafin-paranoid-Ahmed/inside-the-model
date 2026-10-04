/** PCA on a tiny dataset: covariance, eigenvectors (Jacobi), projections and the timelines of the four lesson figures. Pure logic, no DOM. */
import { createRng } from '../../../lib/math/rng';
import { smooth, clamp } from '../F1Shared/plot';
import { covariance, eigSym } from '../../../lib/math/linalg';
export { covariance, eigSym };

export type Pt = [number, number];
export type Mode = 'rotate' | 'eigen' | 'project' | 'scree';
export const DEG = Math.PI / 180;

/** Matrix times vector. */
export const matVec = (M: number[][], v: number[]) => M.map((r) => r.reduce((s, x, j) => s + x * v[j], 0));

// ---------------------------------------------------------------- the 2D cloud

export interface Model2D {
  pts: Pt[];
  /** Covariance [[sxx, sxy], [sxy, syy]]. */
  cov: number[][];
  values: [number, number];
  /** Unit eigenvectors: vectors[0] is PC1. */
  vectors: [Pt, Pt];
  trace: number;
  /** Direction of PC1 in radians, in (-90°, 90°]. */
  th1: number;
  /** Index of the point that PC1 fits worst. */
  focus: number;
}

/** A correlated, elongated cloud of n points, centred on the origin and tilted by `tilt` degrees. Deterministic. */
export function makeModel(tilt = 28, seed = 7, n = 90): Model2D {
  const rng = createRng(seed), r = tilt * DEG, c = Math.cos(r), s = Math.sin(r);
  const raw = Array.from({ length: n }, () => { const a = rng.normal(0, 1.6), b = rng.normal(0, 0.5); return [a * c - b * s, a * s + b * c] as Pt; });
  const mx = raw.reduce((t, p) => t + p[0], 0) / n, my = raw.reduce((t, p) => t + p[1], 0) / n;
  const pts = raw.map((p): Pt => [p[0] - mx, p[1] - my]);
  const cov = covariance(pts), e = eigSym(cov);
  const v1 = e.vectors[0] as Pt, v2 = e.vectors[1] as Pt;
  const th1 = Math.atan2(v1[1], v1[0]);
  let focus = 0, worst = -1;
  pts.forEach((p, i) => { const d = Math.abs(p[0] * v1[1] - p[1] * v1[0]); if (d > worst) { worst = d; focus = i; } });
  return { pts, cov, values: [e.values[0], e.values[1]], vectors: [v1, v2], trace: cov[0][0] + cov[1][1], th1, focus };
}

/** Variance of the data projected on the unit direction at angle theta: u' C u. */
export const varAt = (m: Model2D, theta: number) => {
  const c = Math.cos(theta), s = Math.sin(theta);
  return m.cov[0][0] * c * c + 2 * m.cov[0][1] * c * s + m.cov[1][1] * s * s;
};
/** Mean squared perpendicular distance to that line. Variance plus this always equals the trace. */
export const residAt = (m: Model2D, theta: number) => m.trace - varAt(m, theta);
/** Coordinate of p along unit vector u, and the foot of the perpendicular. */
export const coord = (p: Pt, u: Pt) => p[0] * u[0] + p[1] * u[1];
/** One point through a component: its 1D coordinate z and where it lands again in 2D (the reconstruction). */
export const recon = (p: Pt, u: Pt): { z: number; hat: Pt } => { const z = coord(p, u); return { z, hat: [z * u[0], z * u[1]] }; };
/** Mean squared distance from the points to the line through the origin along u, computed directly from the points. */
export const meanSqDist = (pts: Pt[], u: Pt) => pts.reduce((t, p) => { const z = coord(p, u); return t + (p[0] - z * u[0]) ** 2 + (p[1] - z * u[1]) ** 2; }, 0) / pts.length;

// ---------------------------------------------------------------- mode 'rotate'

export const ROTATE = { hold: 1.8, sweepEnd: 7, settleEnd: 9, T: 10 };
export const ROTATE_MARKERS = [{ at: 0, label: 'worst' }, { at: ROTATE.hold, label: 'sweep' }, { at: ROTATE.sweepEnd, label: 'settle' }, { at: ROTATE.settleEnd, label: 'best axis' }];

export interface RotateFrame {
  theta: number;
  /** Angle travelled since the start, radians in [0, π]. The curves are drawn up to the furthest point reached. */
  rel: number;
  reach: number;
  variance: number;
  resid: number;
  phase: 'worst axis' | 'sweep' | 'settle' | 'best axis';
}

/** Start on PC2 (least variance), turn the line through 180°, then glide back to the peak at PC1. */
export function rotateFrame(t: number, m: Model2D, override: number | null = null): RotateFrame {
  const { hold, sweepEnd, settleEnd } = ROTATE, start = m.th1 - Math.PI / 2;
  const sweep = smooth((t - hold) / (sweepEnd - hold)), settle = smooth((t - sweepEnd) / (settleEnd - sweepEnd));
  let rel = Math.PI * sweep - (Math.PI / 2) * settle, reach = Math.PI * sweep;
  if (override !== null) { rel = clamp(override - start, 0, Math.PI); reach = Math.max(reach, rel); }
  const theta = start + rel;
  return {
    theta, rel, reach, variance: varAt(m, theta), resid: residAt(m, theta),
    phase: t < hold ? 'worst axis' : t < sweepEnd ? 'sweep' : t < settleEnd ? 'settle' : 'best axis',
  };
}

// ---------------------------------------------------------------- mode 'eigen'

export const EIGEN = { start: 1.8, at1: 4, to2: 5.4, at2: 7.8, T: 10 };
export const EIGEN_MARKERS = [{ at: 0, label: 'probe' }, { at: EIGEN.start, label: 'turn to PC1' }, { at: EIGEN.at1, label: 'PC1' }, { at: EIGEN.to2, label: 'turn to PC2' }, { at: EIGEN.at2, label: 'PC2' }];

export interface EigenFrame {
  theta: number;
  v: Pt;
  /** C·v */
  Cv: Pt;
  /** Rayleigh quotient v'Cv: the variance along v. */
  rayleigh: number;
  /** Angle between C·v and v in degrees (0 or 180 when C·v points along v). */
  off: number;
  phase: 'probe' | 'turn to PC1' | 'PC1' | 'turn to PC2' | 'PC2';
}

/** A unit probe vector v sweeps towards PC1, rests, then turns to PC2. Wherever C·v lines up with v, v is an eigenvector. */
export function eigenFrame(t: number, m: Model2D): EigenFrame {
  const { start, at1, to2, at2 } = EIGEN, a = m.th1 - 70 * DEG;
  const theta = t < start ? a : t < at1 ? a + 70 * DEG * smooth((t - start) / (at1 - start)) : t < to2 ? m.th1 : m.th1 + 90 * DEG * smooth((t - to2) / (at2 - to2));
  const v: Pt = [Math.cos(theta), Math.sin(theta)], Cv: Pt = [m.cov[0][0] * v[0] + m.cov[0][1] * v[1], m.cov[1][0] * v[0] + m.cov[1][1] * v[1]];
  const off = Math.abs(Math.atan2(v[0] * Cv[1] - v[1] * Cv[0], v[0] * Cv[0] + v[1] * Cv[1])) / DEG;
  return { theta, v, Cv, rayleigh: v[0] * Cv[0] + v[1] * Cv[1], off, phase: t < start ? 'probe' : t < at1 ? 'turn to PC1' : t < to2 ? 'PC1' : t < at2 ? 'turn to PC2' : 'PC2' };
}

// ---------------------------------------------------------------- mode 'project'

export const PROJECT = { slide: 1.2, slideLen: 2.6, compressed: 3.8, recon: 6, T: 9 };
export const PROJECT_MARKERS = [{ at: 0, label: 'cloud' }, { at: PROJECT.slide, label: 'project' }, { at: PROJECT.compressed, label: 'compressed' }, { at: PROJECT.recon, label: 'reconstruct' }];

export interface ProjectFrame {
  /** 0 = original positions, 1 = all points on the line. */
  s: number;
  /** 0..1 opacity of the error segments. */
  err: number;
  phase: 'cloud' | 'project' | 'compressed' | 'reconstruct';
}
export function projectFrame(t: number): ProjectFrame {
  const { slide, slideLen, compressed, recon } = PROJECT;
  return { s: smooth((t - slide) / slideLen), err: smooth((t - recon) / 1.2), phase: t < slide ? 'cloud' : t < compressed ? 'project' : t < recon ? 'compressed' : 'reconstruct' };
}
/** Numbers for keeping one component: variance kept, mean squared error lost (exactly the other eigenvalue). */
export function keepOne(m: Model2D, comp: 0 | 1) {
  return { kept: m.values[comp] / m.trace, mse: m.values[1 - comp], u: m.vectors[comp] };
}

// ---------------------------------------------------------------- mode 'scree'

export const DIMS = 6;
export interface ScreeModel {
  /** Scores (coordinates along each PC), n × DIMS. */
  scores: number[][];
  values: number[];
  ratios: number[];
  cum: number[];
  total: number;
  vectors: number[][];
}
const LOAD = [[0.9, 0.1], [0.8, -0.2], [0.7, 0.3], [0.1, 0.9], [-0.2, 0.8], [0.3, 0.7]];
/** 150 samples with 6 correlated measurements: two hidden factors plus noise, so two components carry most of the variance. */
export function makeScree(seed = 11, n = 150): ScreeModel {
  const rng = createRng(seed);
  const X = Array.from({ length: n }, () => { const f1 = rng.normal(0, 2.2), f2 = rng.normal(0, 1.3); return LOAD.map((l) => l[0] * f1 + l[1] * f2 + rng.normal(0, 0.45)); });
  const mu = Array.from({ length: DIMS }, (_, j) => X.reduce((t, r) => t + r[j], 0) / n);
  const Xc = X.map((r) => r.map((x, j) => x - mu[j]));
  const e = eigSym(covariance(Xc)), total = e.values.reduce((a, b) => a + b, 0);
  const ratios = e.values.map((v) => v / total);
  let acc = 0;
  return { scores: Xc.map((r) => e.vectors.map((v) => v.reduce((t, x, j) => t + x * r[j], 0))), values: e.values, ratios, cum: ratios.map((r) => (acc += r)), total, vectors: e.vectors };
}

export const SCREE = { hold: 1.5, dwell: 1, tween: 0.35, T: 8.5 };
export const SCREE_MARKERS = [{ at: 0, label: 'all six' }, { at: SCREE.hold, label: 'drop one at a time' }, { at: SCREE.hold + 5 * SCREE.dwell, label: 'only PC1' }];

export interface ScreeFrame {
  /** Continuous number of kept components, for tweening. */
  kf: number;
  /** Components kept, an integer 1..DIMS (what the readouts quote). */
  k: number;
  /** Per-component weight 0..1 (1 = fully kept). */
  w: number[];
  varKept: number;
  /** Mean squared reconstruction error = sum of the dropped eigenvalues. */
  mse: number;
}
export function screeFrame(t: number, s: ScreeModel, override: number | null = null): ScreeFrame {
  const { hold, dwell, tween } = SCREE;
  let kf: number;
  if (override !== null) kf = override;
  else if (t < hold) kf = DIMS;
  else {
    const i = Math.min(DIMS - 2, Math.floor((t - hold) / dwell)), u = (t - hold) / dwell - i;
    kf = t >= hold + (DIMS - 1) * dwell ? 1 : DIMS - i - smooth((u - (1 - tween / dwell)) / (tween / dwell));
  }
  const k = clamp(Math.round(kf), 1, DIMS);
  const w = Array.from({ length: DIMS }, (_, j) => smooth(clamp(kf - j, 0, 1)));
  return { kf, k, w, varKept: s.cum[k - 1], mse: s.values.slice(k).reduce((a, b) => a + b, 0) };
}

// ---------------------------------------------------------------- timelines

export const duration = (mode: Mode) => (mode === 'rotate' ? ROTATE.T : mode === 'eigen' ? EIGEN.T : mode === 'project' ? PROJECT.T : SCREE.T);
export const markers = (mode: Mode) => (mode === 'rotate' ? ROTATE_MARKERS : mode === 'eigen' ? EIGEN_MARKERS : mode === 'project' ? PROJECT_MARKERS : SCREE_MARKERS);
/** How long to hold the last frame before the loop restarts. */
export const HOLD = 1;
