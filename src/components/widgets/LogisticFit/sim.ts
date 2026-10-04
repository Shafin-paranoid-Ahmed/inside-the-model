/** Logistic regression p = sigmoid(w x + b) in 1D: log-loss, gradient, fit, descent path, timeline frames. Pure logic, no DOM. */
import { createRng } from '../../../lib/math/rng';

export interface Pt { x: number; y: 0 | 1 }

const rng = createRng(11);
/** Sixteen points: class 0 centred at x = -1, class 1 at x = +1, overlapping in the middle. */
export const POINTS: Pt[] = Array.from({ length: 16 }, (_, i) => {
  const y = (i % 2) as 0 | 1;
  // Two conflicting labels at the same feature value guarantee overlap and a finite optimum.
  return { x: i >= 14 ? 0.25 : (y ? 1 : -1) + rng.normal(0, 1.5), y };
});
export const X_DOM: [number, number] = [-4, 4];
export const W_DOM: [number, number] = [-1, 4], B_DOM: [number, number] = [-3, 3];
export const START = { w: 0.1, b: -2.2 };
export const LR = 0.5, STEPS = 40;

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
const softplus = (z: number) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)));

/** Mean log-loss (binary cross-entropy): -mean[ y ln p + (1-y) ln(1-p) ], computed stably from the score. */
export const loss = (w: number, b: number, pts: Pt[] = POINTS) =>
  pts.reduce((s, p) => { const z = w * p.x + b; return s + softplus(z) - p.y * z; }, 0) / pts.length;

/** Analytic gradient: dL/dw = mean[(p - y) x], dL/db = mean[p - y]. */
export function grad(w: number, b: number, pts: Pt[] = POINTS): [number, number] {
  let gw = 0, gb = 0;
  for (const p of pts) { const e = sigmoid(w * p.x + b) - p.y; gw += e * p.x; gb += e; }
  return [gw / pts.length, gb / pts.length];
}

export const accuracy = (w: number, b: number, pts: Pt[] = POINTS) =>
  pts.reduce((s, p) => s + ((w * p.x + b > 0 ? 1 : 0) === p.y ? 1 : 0), 0) / pts.length;

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
/** Numerical approximation to the finite optimum on this deliberately overlapping dataset. */
export const BEST = (() => { const p = gdPath(START, 2, 20000); return p[p.length - 1]; })();
export const PATH = gdPath();

// ---- timeline ----
export type Mode = 'train' | 'sweep' | 'steepen';
export const DT = 0.18, T_GUESS = 2;
const T_SWEEP = 9, T_STEEP = 9, S_LO = 0.1, S_HI = 3;
export const duration = (m: Mode) => (m === 'train' ? T_GUESS + STEPS * DT + 2.5 : m === 'sweep' ? T_SWEEP : T_STEEP);
const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };
/** Steepen: the fitted (w, b) are both multiplied by s, so the boundary stays put while the curve sharpens. */
const steepU = (t: number) => ease((t - 1) / 7);
const steepS = (u: number) => S_LO * Math.exp(u * Math.log(S_HI / S_LO));
const uOne = Math.log(1 / S_LO) / Math.log(S_HI / S_LO);
const tOne = (() => { let a = 1, b = 8; for (let i = 0; i < 50; i++) { const m = (a + b) / 2; if (steepU(m) < uOne) a = m; else b = m; } return a; })();
export const X_BOUND = -BEST.b / BEST.w;
const tBound = ((X_BOUND - X_DOM[0]) / (X_DOM[1] - X_DOM[0])) * 8 + 0.5;

export const markers = (m: Mode) =>
  m === 'train'
    ? [{ at: 0, label: 'a guess' }, { at: T_GUESS, label: 'descend' }, { at: T_GUESS + STEPS * DT, label: 'settled' }]
    : m === 'sweep'
      ? [{ at: 0, label: 'far left' }, { at: tBound - 0.7, label: 'near the boundary' }, { at: tBound + 0.7, label: 'far right' }]
      : [{ at: 0, label: 'too flat' }, { at: tOne, label: 'best fit' }, { at: T_STEEP - 1.4, label: 'overconfident' }];

export interface Frame {
  w: number; b: number; loss: number;
  /** Fractional index into PATH (train) else 0..1 progress. */
  k: number;
  phase: string;
  /** Sweep only: the probe position with its score and probability. */
  probe: { x: number; z: number; p: number } | null;
}
/** `path` lets a widget replay descent from a start point the user chose (its first element is the start). */
export function frameAt(t: number, mode: Mode, path: PathPt[] = PATH): Frame {
  if (mode === 'sweep') {
    const x = X_DOM[0] + (X_DOM[1] - X_DOM[0]) * Math.min(1, Math.max(0, (t - 0.5) / 8)), z = BEST.w * x + BEST.b;
    const near = Math.abs(x - X_BOUND) < 0.7;
    return { w: BEST.w, b: BEST.b, loss: BEST.loss, k: 0, phase: near ? 'near the boundary' : x < X_BOUND ? 'far left' : 'far right', probe: { x, z, p: sigmoid(z) } };
  }
  if (mode === 'steepen') {
    const u = steepU(t), s = steepS(u), w = BEST.w * s, b = BEST.b * s;
    return { w, b, loss: loss(w, b), k: u, phase: s < 0.9 ? 'too flat' : s < 1.15 ? 'best fit' : s < 2 ? 'sharpening' : 'overconfident', probe: null };
  }
  const k = Math.min(STEPS, Math.max(0, (t - T_GUESS) / DT)), i = Math.min(STEPS - 1, Math.floor(k)), f = k - i;
  const a = path[i], z = path[i + 1], w = a.w + (z.w - a.w) * f, b = a.b + (z.b - a.b) * f;
  return { w, b, loss: loss(w, b), k, phase: t < T_GUESS ? 'a guess' : k < STEPS ? 'descend' : 'settled', probe: null };
}

/** Render-ready probabilities and per-example losses; threshold changes decisions, never the fitted probabilities. */
export function inspect(w: number, b: number, threshold = 0.5) {
  const samples = POINTS.map((point) => {
    const z = w * point.x + b, p = sigmoid(z);
    return { ...point, p, error: p - point.y, loss: softplus(z) - point.y * z, predicted: p >= threshold ? 1 : 0 };
  });
  return {
    curve: Array.from({ length: 161 }, (_, i) => { const x = -4 + i / 20; return { x, p: sigmoid(w * x + b) }; }),
    samples,
    accuracy: samples.filter((p) => p.predicted === p.y).length / samples.length,
    boundary: Math.abs(w) < 1e-10 ? null : (Math.log(threshold / (1 - threshold)) - b) / w,
  };
}
