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

// ---- timeline ----
export type Mode = 'iterative' | 'closed';
export const DT = 0.18, T_GUESS = 2;
export const duration = (m: Mode) => (m === 'iterative' ? T_GUESS + STEPS * DT + 2.5 : 7);
export const markers = (m: Mode) =>
  m === 'iterative'
    ? [{ at: 0, label: 'a guess' }, { at: T_GUESS, label: 'descend' }, { at: T_GUESS + STEPS * DT, label: 'settled' }]
    : [{ at: 0, label: 'a guess' }, { at: T_GUESS, label: 'solve' }, { at: 4.5, label: 'solved' }];
const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };

export interface Frame {
  w: number; b: number; loss: number;
  /** Fractional index into PATH (iterative) or 0..1 progress (closed). */
  k: number;
  phase: 'a guess' | 'descend' | 'solve' | 'settled';
}
/** `path` lets a widget replay descent from a start point the user chose (its first element is the start). */
export function frameAt(t: number, mode: Mode, path: PathPt[] = PATH, best = BEST): Frame {
  const S = path[0];
  if (mode === 'closed') {
    const u = ease((t - T_GUESS) / 2.5), w = S.w + (best.w - S.w) * u, b = S.b + (best.b - S.b) * u;
    return { w, b, loss: loss(w, b), k: u, phase: t < T_GUESS ? 'a guess' : u < 1 ? 'solve' : 'settled' };
  }
  const k = Math.min(STEPS, Math.max(0, (t - T_GUESS) / DT)), i = Math.min(STEPS - 1, Math.floor(k)), f = k - i;
  const a = path[i], z = path[i + 1];
  const w = a.w + (z.w - a.w) * f, b = a.b + (z.b - a.b) * f;
  return { w, b, loss: loss(w, b), k, phase: t < T_GUESS ? 'a guess' : k < STEPS ? 'descend' : 'settled' };
}
