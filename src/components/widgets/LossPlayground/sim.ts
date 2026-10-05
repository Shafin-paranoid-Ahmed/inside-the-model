/** Loss playground: residuals, squared vs absolute error, log loss. Pure logic, no DOM. */

export interface Pt { x: number; y: number }
export type Task = 'regression' | 'classification';
export type Loss = 'mse' | 'mae' | 'logloss';

const NOISE = [0.3, -0.5, 0.4, -0.2, 0.5, -0.4, 0.2, -0.3];
/** Eight points near y = 0.6x + 1. */
export const CLEAN: Pt[] = NOISE.map((e, i) => ({ x: i - 3.5, y: 0.6 * (i - 3.5) + 1 + e }));
/** The outlier rests on the trend at amt = 0 and slides down by OUT_SLIDE at amt = 1. */
export const OUT_X = 3, OUT_SLIDE = -5.5;
export const outlier = (amt: number): Pt => ({ x: OUT_X, y: 0.6 * OUT_X + 1 + OUT_SLIDE * amt });
export const points = (amt: number): Pt[] => (amt > 0 ? [...CLEAN, outlier(amt)] : CLEAN);

/** Classification: one feature, labels 0/1, model p = sigmoid(w x + b). */
const LABELS: (0 | 1)[] = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1];
export const CLASS_PTS: { x: number; label: 0 | 1 }[] = [-3, -2.4, -1.8, -1.1, -0.5, 0.3, 0.7, 1.4, 2, 2.8].map((x, i) => ({ x, label: LABELS[i] }));

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const EPS = 1e-9;
export const logLoss1 = (label: number, p: number) => -Math.log(Math.max(label ? p : 1 - p, EPS));

export const residuals = (w: number, b: number, pts: Pt[]) => pts.map((p) => p.y - (w * p.x + b));
export const mse = (w: number, b: number, pts: Pt[]) => residuals(w, b, pts).reduce((s, r) => s + r * r, 0) / pts.length;
export const mae = (w: number, b: number, pts: Pt[]) => residuals(w, b, pts).reduce((s, r) => s + Math.abs(r), 0) / pts.length;
export const logloss = (w: number, b: number) => CLASS_PTS.reduce((s, p) => s + logLoss1(p.label, sigmoid(w * p.x + b)), 0) / CLASS_PTS.length;

export function lossValue(kind: Loss, w: number, b: number, amt: number): number {
  if (kind === 'logloss') return logloss(w, b);
  const pts = points(amt);
  return kind === 'mse' ? mse(w, b, pts) : mae(w, b, pts);
}

export const W_RANGE: Record<Task, [number, number]> = { regression: [-0.5, 2.2], classification: [-1, 6] };
export const B_RANGE: Record<Task, [number, number]> = { regression: [-2, 4], classification: [-6, 4] };

/** Loss along the w axis with b held fixed, sampled at n points; `best` is the w with the lowest sampled loss. */
export function curve(kind: Loss, b: number, amt: number, n = 121): { w: number[]; v: number[]; best: number } {
  const [a, z] = W_RANGE[kind === 'logloss' ? 'classification' : 'regression'];
  const w = Array.from({ length: n }, (_, i) => a + ((z - a) * i) / (n - 1));
  const v = w.map((x) => lossValue(kind, x, b, amt));
  let bi = 0;
  v.forEach((x, i) => { if (x < v[bi]) bi = i; });
  return { w, v, best: w[bi] };
}

/** Intercept that minimises squared error for a fixed slope (the mean residual). */
export const bestB = (w: number, pts: Pt[]) => pts.reduce((s, p) => s + p.y - w * p.x, 0) / pts.length;

/** Probability the model gives to the correct label of a classification point. */
export const pCorrect = (label: number, p: number) => (label ? p : 1 - p);

// ---- views: each figure in the lesson has one job, so no two figures repeat each other ----
/**
 * residuals: the gaps only. squares: squared vs absolute error drawn on the points. curve: adds the loss-vs-slope chart.
 * outlier: the loss chart while one point slides away. classify: a probability curve and each point's log loss.
 */
export type View = 'residuals' | 'squares' | 'curve' | 'outlier' | 'classify';
export const taskOf = (v: View): Task => (v === 'classify' ? 'classification' : 'regression');

// ---- timeline ----
export const T_TILT = 6, T_OUT = 9, T_CLS = 8;
export const duration = (v: View) => (v === 'classify' ? T_CLS : v === 'outlier' ? T_OUT : T_TILT);
export const markers = (v: View) =>
  v === 'classify' ? [{ at: 0, label: 'sharpen the curve' }]
    : v === 'outlier' ? [{ at: 0, label: 'add an outlier' }, { at: 6, label: 'remove it' }]
    : [{ at: 0, label: 'tilt the line' }];
const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };
const wrap = (t: number, T: number) => ((t % T) + T) % T;

export interface Frame { w: number; b: number; amt: number }
/**
 * Auto demo. Tilt views: the line swings to both sides of the best fit and back (the intercept follows its best value).
 * Outlier: the line stays on the best fit while one point slides away (1 s to 4 s), holds, then returns (6 s to 9 s).
 * Classify: the sigmoid sharpens and relaxes.
 */
export function frameAt(v: View, t: number): Frame {
  if (v === 'classify') {
    const w = 0.3 + 3.2 * (0.5 - 0.5 * Math.cos((2 * Math.PI * wrap(t, T_CLS)) / T_CLS));
    return { w, b: -0.2 * w, amt: 0 };
  }
  if (v === 'outlier') {
    const s = wrap(t, T_OUT);
    return { w: 0.6, b: bestB(0.6, CLEAN), amt: s < 6 ? ease((s - 1) / 3) : 1 - ease((s - 6) / 3) };
  }
  const w = 0.6 + 0.6 * Math.sin((2 * Math.PI * wrap(t, T_TILT)) / T_TILT);
  return { w, b: bestB(w, CLEAN), amt: 0 };
}
