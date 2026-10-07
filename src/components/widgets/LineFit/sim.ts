/** Linear regression y = w x + b: loss, gradient, closed form, gradient-descent path. Pure logic, no DOM. */
import { createRng } from '../../../lib/math/rng';

export interface Pt { x: number; y: number }

const rng = createRng(7);
/** Twelve noisy points around y = 0.9x + 0.6. */
export const POINTS: Pt[] = Array.from({ length: 12 }, (_, i) => {
  const x = -3 + (6 * i) / 11 + rng.uniform(-0.15, 0.15);
  return { x, y: 0.9 * x + 0.6 + rng.normal(0, 0.7) };
});
export const W_DOM: [number, number] = [-2, 3.2], B_DOM: [number, number] = [-3.2, 3.6];
export const START = { w: -1.4, b: -2.2 };
export const LR = 0.08, STEPS = 40;

export const loss = (w: number, b: number, pts: Pt[] = POINTS) =>
  pts.reduce((s, p) => s + (w * p.x + b - p.y) ** 2, 0) / pts.length;

/** Analytic gradient of the MSE: [dL/dw, dL/db]. */
export function grad(w: number, b: number, pts: Pt[] = POINTS): [number, number] {
  let gw = 0, gb = 0;
  for (const p of pts) { const r = w * p.x + b - p.y; gw += 2 * r * p.x; gb += 2 * r; }
  return [gw / pts.length, gb / pts.length];
}

/** Least squares in closed form (Legendre 1805, Gauss): w = cov(x,y)/var(x), b = mean(y) - w mean(x). */
export function closedForm(pts: Pt[] = POINTS): { w: number; b: number } {
  const n = pts.length, mx = pts.reduce((s, p) => s + p.x, 0) / n, my = pts.reduce((s, p) => s + p.y, 0) / n;
  let sxy = 0, sxx = 0;
  for (const p of pts) { sxy += (p.x - mx) * (p.y - my); sxx += (p.x - mx) ** 2; }
  const w = sxy / sxx;
  return { w, b: my - w * mx };
}
export const BEST = closedForm();

export interface PathPt { w: number; b: number; loss: number }
/** Full-batch gradient descent from `start`; element 0 is the start itself. */
export function gdPath(start = START, lr = LR, steps = STEPS, pts: Pt[] = POINTS): PathPt[] {
  let { w, b } = start;
  const out: PathPt[] = [{ w, b, loss: loss(w, b, pts) }];
  for (let i = 0; i < steps; i++) {
    const [gw, gb] = grad(w, b, pts);
    w -= lr * gw; b -= lr * gb;
    out.push({ w, b, loss: loss(w, b, pts) });
  }
  return out;
}
export const PATH = gdPath();

/** Expand the same MSE used by the plot into per-example arithmetic. */
export function calculation(w: number, b: number, pts: Pt[] = POINTS) {
  const rows = pts.map((p) => {
    const prediction = w * p.x + b, error = prediction - p.y;
    return { ...p, prediction, error, squared: error ** 2 };
  });
  const sum = rows.reduce((s, p) => s + p.squared, 0);
  const sumError = rows.reduce((s, p) => s + p.error, 0);
  const sumErrorX = rows.reduce((s, p) => s + p.error * p.x, 0);
  const [gw, gb] = grad(w, b, pts);
  const next = { w: w - LR * gw, b: b - LR * gb };
  return { rows, sum, sumError, sumErrorX, mse: sum / pts.length, gw, gb,
    next: { ...next, loss: loss(next.w, next.b, pts) } };
}

// ---- timeline ----
/** iterative: gradient descent walks to the answer. closed: the formula jumps there. tour: the dot visits lines all over the map. */
export type Mode = 'iterative' | 'closed' | 'tour' | 'prediction' | 'errors' | 'update';
export const DT = 0.18, T_GUESS = 2;
/** Tour stops on the (w, b) map: a bad line, lines in other corners, then the best one. 2 s per leg, then a 2 s hold. */
export const TOUR = [START, { w: 2.6, b: -2.4 }, { w: 2.4, b: 3.0 }, { w: -1.3, b: 3.0 }, { w: 0.2, b: 1.6 }, { w: BEST.w, b: BEST.b }];
const T_LEG = 2;
export const duration = (m: Mode) => (m === 'update' ? 12 : m === 'prediction' || m === 'errors' ? 10 : m === 'iterative' ? T_GUESS + STEPS * DT + 2.5 : m === 'tour' ? (TOUR.length - 1) * T_LEG + 2 : 7);
export const markers = (m: Mode) =>
  m === 'prediction' || m === 'errors' ? [{ at: 0, label: 'start' }, { at: 3, label: 'raise b' }, { at: 7, label: 'new line' }]
    : m === 'update' ? [{ at: 0, label: 'read slopes' }, { at: 4, label: 'move' }, { at: 8, label: 'compare MSE' }]
    : m === 'iterative'
    ? [{ at: 0, label: 'a guess' }, { at: T_GUESS, label: 'descend' }, { at: T_GUESS + STEPS * DT, label: 'settled' }]
    : m === 'tour' ? [{ at: 0, label: 'a bad line' }, { at: 2 * T_LEG, label: 'other lines' }, { at: (TOUR.length - 2) * T_LEG, label: 'approach best' }]
    : [{ at: 0, label: 'a guess' }, { at: T_GUESS, label: 'solve' }, { at: 4.5, label: 'solved' }];
const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };

export interface Frame {
  w: number; b: number; loss: number;
  /** Fractional index into PATH (iterative) or 0..1 progress (closed). */
  k: number;
  phase: 'a guess' | 'descend' | 'solve' | 'settled' | 'tour';
}
/** `path` lets a widget replay descent from a start point the user chose (its first element is the start). */
export function frameAt(t: number, mode: Mode, path: PathPt[] = PATH, best = BEST): Frame {
  const S = path[0];
  if (mode === 'prediction' || mode === 'errors') {
    const u = ease((t - 3) / 4), w = S.w, b = S.b + u * Math.min(1, B_DOM[1] - S.b);
    return { w, b, loss: loss(w, b), k: u, phase: 'tour' };
  }
  if (mode === 'update') {
    const u = ease((t - 4) / 4), z = calculation(S.w, S.b).next;
    const w = S.w + (z.w - S.w) * u, b = S.b + (z.b - S.b) * u;
    return { w, b, loss: loss(w, b), k: u, phase: t < 4 ? 'a guess' : t < 8 ? 'descend' : 'settled' };
  }
  if (mode === 'tour') {
    const u = Math.max(0, t) / T_LEG, i = Math.min(TOUR.length - 2, Math.floor(u)), f = ease(u - i);
    const a = i === 0 ? S : TOUR[i], z = TOUR[i + 1];
    const w = a.w + (z.w - a.w) * f, b = a.b + (z.b - a.b) * f;
    return { w, b, loss: loss(w, b), k: 0, phase: u >= TOUR.length - 1 ? 'settled' : 'tour' };
  }
  if (mode === 'closed') {
    const u = ease((t - T_GUESS) / 2.5), w = S.w + (best.w - S.w) * u, b = S.b + (best.b - S.b) * u;
    return { w, b, loss: loss(w, b), k: u, phase: t < T_GUESS ? 'a guess' : u < 1 ? 'solve' : 'settled' };
  }
  const k = Math.min(STEPS, Math.max(0, (t - T_GUESS) / DT)), i = Math.min(STEPS - 1, Math.floor(k)), f = k - i;
  const a = path[i], z = path[i + 1];
  const w = a.w + (z.w - a.w) * f, b = a.b + (z.b - a.b) * f;
  return { w, b, loss: loss(w, b), k, phase: t < T_GUESS ? 'a guess' : k < STEPS ? 'descend' : 'settled' };
}
