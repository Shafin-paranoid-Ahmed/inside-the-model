/**
 * The Day 13 toy data: a 2D Gaussian mixture every generative lesson learns, pure and DOM-free.
 * Two moons = 52 components (26 per moon, sigma 0.07). Because the density is a known mixture, every
 * lesson can draw the exact answer next to the learned one.
 *
 * Noised data is x = a x0 + b eps (x0 ~ data, eps ~ N(0, I)). Each component k then has marginal
 * N(a mu_k, (a^2 sigma^2 + b^2) I), and everything follows from the posterior mean E[x0 | x]:
 *   score            grad log p(x) = (a E[x0|x] - x) / b^2          (Tweedie)
 *   ideal eps-hat    (x - a E[x0|x]) / b
 *   flow matching    a = t, b = 1 - t:  v*(x, t) = (E[x1|x] - x) / (1 - t)
 */
import { createRng } from '../math/rng';
import { exp, normal } from './mlp';

export type Dataset = 'spiral' | 'moons';
/** `upper` = number of leading components on the upper moon (K for the spiral, which has no moons). */
export interface Mixture { K: number; mu: Float64Array; sigma: number; upper: number }

export function mixture(ds: Dataset): Mixture {
  const pts: number[] = [];
  if (ds === 'spiral') {
    const K = 64;
    for (let k = 0; k < K; k++) {
      const u = k / (K - 1), a = 0.3 * Math.PI + u * 4 * Math.PI, r = 0.22 + 1.1 * u;
      pts.push(r * Math.cos(a), r * Math.sin(a));
    }
    return { K, mu: Float64Array.from(pts), sigma: 0.075, upper: K };
  }
  const M = 26;
  for (let k = 0; k < M; k++) { const a = (k / (M - 1)) * Math.PI; pts.push(1.25 * (Math.cos(a) - 0.5), 1.25 * (Math.sin(a) - 0.25)); }
  for (let k = 0; k < M; k++) { const a = (k / (M - 1)) * Math.PI; pts.push(1.25 * (1 - Math.cos(a) - 0.5), 1.25 * (0.5 - Math.sin(a) - 0.25)); }
  return { K: 2 * M, mu: Float64Array.from(pts), sigma: 0.07, upper: M };
}
export const MOONS = mixture('moons');
/** The shared plane every Day 13 widget draws: [-DOMAIN, DOMAIN]^2. */
export const DOMAIN = 2.2;

/** Exact samples: positions (2n), the component k of each, and its moon (0 upper, 1 lower). */
export function sampleData(m: Mixture, n: number, seed: number) {
  const r = createRng(seed);
  const x = new Float32Array(2 * n), k = new Uint16Array(n), moon = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const c = Math.min(m.K - 1, Math.floor(r.next() * m.K));
    x[2 * i] = m.mu[2 * c] + m.sigma * normal(r); x[2 * i + 1] = m.mu[2 * c + 1] + m.sigma * normal(r);
    k[i] = c; moon[i] = c < m.upper ? 0 : 1;
  }
  return { x, k, moon };
}

let buf = new Float64Array(0);
/**
 * E[x0 | x = a x0 + b eps] for the mixture, into out[0..1]. `only` restricts the data to one moon
 * (the class-conditional posterior used by guidance).
 */
export function posteriorMean(m: Mixture, a: number, b: number, x: number, y: number, out: number[] | Float64Array, only?: 'upper' | 'lower'): void {
  const s2 = m.sigma * m.sigma, v = a * a * s2 + b * b, g = (a * s2) / v; // per-component shrink toward the noisy point
  const k0 = only === 'lower' ? m.upper : 0, k1 = only === 'upper' ? m.upper : m.K;
  if (buf.length < m.K) buf = new Float64Array(m.K);
  let mx = -Infinity;
  for (let k = k0; k < k1; k++) {
    const dx = x - a * m.mu[2 * k], dy = y - a * m.mu[2 * k + 1];
    buf[k] = -(dx * dx + dy * dy) / (2 * v);
    if (buf[k] > mx) mx = buf[k];
  }
  let z = 0, ex = 0, ey = 0;
  for (let k = k0; k < k1; k++) {
    const w = exp(buf[k] - mx), mux = m.mu[2 * k], muy = m.mu[2 * k + 1];
    z += w; ex += w * (mux + g * (x - a * mux)); ey += w * (muy + g * (y - a * muy));
  }
  out[0] = ex / z; out[1] = ey / z;
}

/** Score grad_x log p(x) of the noised mixture, into out[0..1]. */
export function score(m: Mixture, a: number, b: number, x: number, y: number, out: number[] | Float64Array, only?: 'upper' | 'lower'): void {
  posteriorMean(m, a, b, x, y, out, only);
  const b2 = b * b;
  out[0] = (a * out[0] - x) / b2; out[1] = (a * out[1] - y) / b2;
}

/** log p(x) of the noised mixture (for tests). */
export function logp(m: Mixture, a: number, b: number, x: number, y: number): number {
  const v = a * a * m.sigma * m.sigma + b * b;
  let mx = -Infinity; const l: number[] = [];
  for (let k = 0; k < m.K; k++) {
    const dx = x - a * m.mu[2 * k], dy = y - a * m.mu[2 * k + 1], e = -(dx * dx + dy * dy) / (2 * v);
    l.push(e); if (e > mx) mx = e;
  }
  let sum = 0; for (const e of l) sum += Math.exp(e - mx);
  return mx + Math.log(sum / m.K) - Math.log(2 * Math.PI * v);
}

/**
 * The comparison numbers quoted across Day 13. A point is on the data if its nearest component mean is within
 * 3 sigma (0.21 for the moons). onData = share of all points on the data; upper/lower = share of the on-data
 * points on each moon (0.5 / 0.5 for a model that covers both equally; 0 / 0 if nothing lands).
 */
export function quality(pts: ArrayLike<number>, m: Mixture): { onData: number; upper: number; lower: number } {
  const n = pts.length / 2, r2 = 9 * m.sigma * m.sigma;
  let on = 0, up = 0;
  for (let i = 0; i < n; i++) {
    const x = pts[2 * i], y = pts[2 * i + 1];
    let best = Infinity, bk = 0;
    for (let k = 0; k < m.K; k++) {
      const dx = x - m.mu[2 * k], dy = y - m.mu[2 * k + 1], d = dx * dx + dy * dy;
      if (d < best) { best = d; bk = k; }
    }
    if (best <= r2) { on++; if (bk < m.upper) up++; }
  }
  return { onData: n ? on / n : 0, upper: on ? up / on : 0, lower: on ? (on - up) / on : 0 };
}
