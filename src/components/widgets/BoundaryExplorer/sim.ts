/** Pure logic for BoundaryExplorer: a logistic classifier trained by full-batch gradient descent on 2-D points. No DOM. */
import { createRng } from '../../../lib/math/rng';

export type Dataset = 'linear' | 'xor' | 'circles';
export type Model = 'linear' | 'poly';
export interface Pt { x: number; y: number; c: 0 | 1 }

/** The plane shows [-DOM, DOM]^2. */
export const DOM = 1.5;
export const STEPS = 200;
export const LR = 0.8;
/** Timeline: 0..T seconds. Training plays over 0..T_TRAIN, then holds. */
export const T = 8;
export const T_TRAIN = 6;
export const MARKERS = [{ at: 0, label: 'start' }, { at: 0.8, label: 'learning' }, { at: 5, label: 'settled' }];

export const nFeatures = (m: Model) => (m === 'linear' ? 2 : 5);
/** Weight vector layout: [bias, one weight per feature]. Feature order: x, y, x², xy, y². */
export const nWeights = (m: Model) => nFeatures(m) + 1;

export function makeDataset(name: Dataset, seed = 7, n = 30): Pt[] {
  const rng = createRng(seed + (name === 'linear' ? 1 : name === 'xor' ? 2 : 3));
  const pts: Pt[] = [];
  const clip = (v: number) => Math.max(-DOM + 0.08, Math.min(DOM - 0.08, v));
  for (const c of [0, 1] as const) {
    for (let i = 0; i < n; i++) {
      let x: number, y: number;
      if (name === 'linear') {
        const s = c ? 1 : -1;
        x = rng.normal(0.55 * s, 0.3); y = rng.normal(0.45 * s, 0.3);
      } else if (name === 'xor') {
        const q = rng.next() < 0.5 ? 1 : -1;
        x = rng.normal(0.7 * q, 0.24); y = rng.normal(0.7 * (c ? q : -q), 0.24);
      } else {
        const r = rng.normal(c ? 1.05 : 0.38, 0.1), a = rng.uniform(0, 2 * Math.PI);
        x = r * Math.cos(a); y = r * Math.sin(a);
      }
      pts.push({ x: clip(x), y: clip(y), c });
    }
  }
  return pts;
}

export function features(m: Model, x: number, y: number, out: Float64Array | number[]) {
  out[0] = x; out[1] = y;
  if (m === 'poly') { out[2] = x * x; out[3] = x * y; out[4] = y * y; }
}
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** Score z = b + w·φ(x, y). */
export function score(m: Model, w: ArrayLike<number>, x: number, y: number): number {
  let z = w[0] + w[1] * x + w[2] * y;
  if (m === 'poly') z += w[3] * x * x + w[4] * x * y + w[5] * y * y;
  return z;
}
export const prob = (m: Model, w: ArrayLike<number>, x: number, y: number) => sigmoid(score(m, w, x, y));

/** Mean cross-entropy and accuracy of `w` on `pts`. */
export function evaluate(m: Model, w: ArrayLike<number>, pts: readonly Pt[]) {
  let loss = 0, hit = 0;
  for (const p of pts) {
    const z = score(m, w, p.x, p.y);
    loss += Math.log1p(Math.exp(-Math.abs(z))) + Math.max(0, p.c ? -z : z); // stable BCE with logits
    if ((z > 0 ? 1 : 0) === p.c) hit++;
  }
  return { loss: loss / pts.length, acc: hit / pts.length };
}

/** Carry weights over to another model (feature order is a prefix, extra weights start at 0). */
export function embed(w: ArrayLike<number>, m: Model): Float64Array {
  const o = new Float64Array(nWeights(m));
  for (let i = 0; i < o.length && i < w.length; i++) o[i] = w[i];
  return o;
}

export interface Run { model: Model; W: Float64Array; loss: Float64Array; acc: Float64Array; nw: number }

/** Precompute every training snapshot once; the timeline only interpolates them. */
export function train(pts: readonly Pt[], model: Model, w0?: ArrayLike<number>, steps = STEPS, lr = LR): Run {
  const nw = nWeights(model), F = nw - 1;
  const w = embed(w0 ?? [], model), g = new Float64Array(nw), phi = new Float64Array(F);
  const W = new Float64Array((steps + 1) * nw), loss = new Float64Array(steps + 1), acc = new Float64Array(steps + 1);
  for (let k = 0; k <= steps; k++) {
    W.set(w, k * nw);
    const e = evaluate(model, w, pts); loss[k] = e.loss; acc[k] = e.acc;
    g.fill(0);
    for (const p of pts) {
      const r = prob(model, w, p.x, p.y) - p.c;
      features(model, p.x, p.y, phi);
      g[0] += r;
      for (let j = 0; j < F; j++) g[j + 1] += r * phi[j];
    }
    for (let j = 0; j < nw; j++) w[j] -= (lr * g[j]) / pts.length;
  }
  return { model, W, loss, acc, nw };
}

const ease = (u: number) => 1 - (1 - u) * (1 - u);
/** Fractional training step shown at time t (fast at first, then slowing). */
export const stepAt = (t: number, steps = STEPS) => steps * ease(Math.min(1, Math.max(0, t / T_TRAIN)));

export interface Frame { k: number; w: Float64Array; loss: number; acc: number; phase: string }
/** Everything needed to draw at time t: interpolated weights, loss, accuracy. */
export function frameAt(t: number, run: Run): Frame {
  const steps = run.loss.length - 1, k = stepAt(t, steps), i = Math.min(steps - 1, Math.floor(k)), f = k - i, nw = run.nw;
  const w = new Float64Array(nw);
  for (let j = 0; j < nw; j++) w[j] = run.W[i * nw + j] * (1 - f) + run.W[(i + 1) * nw + j] * f;
  const lerp = (a: Float64Array) => a[i] * (1 - f) + a[i + 1] * f;
  const phase = MARKERS.reduce((s, m) => (m.at <= t + 1e-9 ? m.label : s), MARKERS[0].label);
  return { k, w, loss: lerp(run.loss), acc: lerp(run.acc), phase };
}

/** P(class 1) on an n×n grid of cell centres over the plane; row j = 0 is the bottom. */
export function field(m: Model, w: ArrayLike<number>, n: number, out: Float32Array) {
  for (let j = 0; j < n; j++) {
    const y = -DOM + ((j + 0.5) / n) * 2 * DOM;
    for (let i = 0; i < n; i++) out[j * n + i] = prob(m, w, -DOM + ((i + 0.5) / n) * 2 * DOM, y);
  }
}

/** Signed distance from a point to the boundary of a linear model: z / ‖(w_x, w_y)‖. */
export function signedDistance(w: ArrayLike<number>, x: number, y: number) {
  const n = Math.hypot(w[1], w[2]);
  return n > 1e-9 ? (w[0] + w[1] * x + w[2] * y) / n : 0;
}
