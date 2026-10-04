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

// ---- timeline ----
export const T_REG = 14, T_CLS = 8;
export const duration = (task: Task) => (task === 'regression' ? T_REG : T_CLS);
export const markers = (task: Task) =>
  task === 'regression'
    ? [{ at: 0, label: 'tilt the line' }, { at: 6, label: 'add an outlier' }, { at: 11, label: 'remove it' }]
    : [{ at: 0, label: 'sharpen the curve' }];
const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };

export interface Frame { w: number; b: number; amt: number }
/**
 * Auto demo. Regression: tilt the line to both sides of the best fit and back (the intercept follows its best value),
 * then slide an outlier away while the line stays put, then bring it back. Classification: sharpen the sigmoid.
 */
export function frameAt(task: Task, t: number): Frame {
  if (task === 'classification') {
    const w = 0.3 + 3.2 * (0.5 - 0.5 * Math.cos((2 * Math.PI * (((t % T_CLS) + T_CLS) % T_CLS)) / T_CLS));
    return { w, b: -0.2 * w, amt: 0 };
  }
  const s = ((t % T_REG) + T_REG) % T_REG;
  if (s < 6) {
    const w = 0.6 + 0.6 * Math.sin((2 * Math.PI * s) / 6);
    return { w, b: bestB(w, CLEAN), amt: 0 };
  }
  return { w: 0.6, b: bestB(0.6, CLEAN), amt: s < 11 ? ease((s - 6) / 4) : 1 - ease((s - 11) / 3) };
}
