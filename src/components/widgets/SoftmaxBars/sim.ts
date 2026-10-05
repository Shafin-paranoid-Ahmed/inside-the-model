/** Softmax bars: logits -> probabilities -> cross-entropy -> gradient (p - y). Pure logic, no DOM. */

export const CLASSES = ['A', 'B', 'C', 'D'] as const;
export const N = CLASSES.length;
export const Z0 = [1.5, 0.8, -0.4, 0.3];
export const Z_RANGE: [number, number] = [-4, 5];
export const TAU_RANGE: [number, number] = [0.2, 4];
/** 'swing' and 'temp' are the two halves of 'softmax', for steps that teach one idea each. */
export type Mode = 'softmax' | 'swing' | 'temp' | 'loss' | 'grad';

/** softmax(z / tau), computed stably. */
export function softmax(z: ArrayLike<number>, tau = 1): number[] {
  if (!(tau > 0) || !Number.isFinite(tau)) throw new RangeError('Temperature must be positive and finite');
  let m = -Infinity;
  for (let i = 0; i < z.length; i++) m = Math.max(m, z[i] / tau);
  const e = Array.from(z, (v) => Math.exp(v / tau - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
/** Cross-entropy of a one-hot target: -ln p[k]. */
export const crossEntropy = (p: ArrayLike<number>, k: number) => -Math.log(p[k]);
/** Stable negative log-softmax, including probabilities that underflow to zero. */
export function lossFromLogits(z: number[], k: number, tau = 1): number {
  const a = z.map(v => v / tau), m = Math.max(...a);
  return m - a[k] + Math.log(a.reduce((sum, v) => sum + Math.exp(v - m), 0));
}
/** dL/d(original logits), including the temperature chain-rule factor. */
export const gradLogits = (p: ArrayLike<number>, k: number, tau = 1) => Array.from(p, (v, i) => (v - (i === k ? 1 : 0)) / tau);

const STEPS = 48, LR = 0.35;
const cache = new Map<number, number[][]>();
/** Gradient-descent path of the logits (tau = 1) from Z0 toward class k. Index 0 is the start. */
export function trajectory(k: number): number[][] {
  let tr = cache.get(k);
  if (!tr) {
    tr = [Z0.slice()];
    for (let s = 0; s < STEPS; s++) {
      const z = tr[s], g = gradLogits(softmax(z), k);
      tr.push(z.map((v, i) => v - LR * g[i]));
    }
    cache.set(k, tr);
  }
  return tr;
}

export const duration = (mode: Mode) => (mode === 'softmax' ? 10 : mode === 'swing' || mode === 'temp' ? 5 : 8);
export const markers = (mode: Mode) =>
  mode === 'softmax' ? [{ at: 0, label: 'swing one score' }, { at: 5, label: 'temperature' }]
  : mode === 'swing' ? [{ at: 0, label: 'swing one score' }]
  : mode === 'temp' ? [{ at: 0, label: 'temperature' }]
  :[{ at: 0, label: 'train logits' }, { at: 6.5, label: 'replay reset' }];

export interface Frame { z: number[]; tau: number; p: number[]; loss: number; grad: number[] }
export function derive(z: number[], tau: number, truth: number): Frame {
  const p = softmax(z, tau);
  return { z, tau, p, loss: lossFromLogits(z, truth, tau), grad: gradLogits(p, truth, tau) };
}

/** Everything to draw at time t (loops with the duration). Scrubbing is free. */
export function frameAt(mode: Mode, t: number, truth: number): Frame {
  const T = duration(mode), s = ((t % T) + T) % T;
  if (mode === 'swing' || mode === 'temp') return frameAt('softmax', mode === 'swing' ? s : s + 5, truth);
  if (mode === 'softmax') {
    if (s < 5) { const z = Z0.slice(); z[0] = Z0[0] + 2.5 * Math.sin((2 * Math.PI * s) / 5); return derive(z, 1, truth); }
    return derive(Z0.slice(), Math.exp(-1.2 * Math.sin((2 * Math.PI * (s - 5)) / 5)), truth);
  }
  const tr = trajectory(truth);
  if (s >= 6.5) {
    const u = (s - 6.5) / 1.5, e = u * u * (3 - 2 * u);
    return derive(tr[STEPS].map((v, j) => v + (Z0[j] - v) * e), 1, truth);
  }
  const x = Math.min(1, s / 6) * STEPS, i = Math.min(STEPS - 1, Math.floor(x)), f = x - i;
  return derive(tr[i].map((v, j) => v + (tr[i + 1][j] - v) * f), 1, truth);
}
