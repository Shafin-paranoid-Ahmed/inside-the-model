/**
 * The log-linear candidate policy pi_theta(y|x) = softmax_y(theta . phi(x, y) + c(x, y)) over the K = 8 candidates of
 * bank.ts, with exact expectations, KL and gradients (everything is a sum over 8 candidates). Pure TS.
 * Reward tables are flat Float64Array(P * K) indexed x * K + y (see `scores`).
 */
import type { Rng } from '../math/rng';
import { C, D, K, P, PHI, TRAIN, W_GOLD } from './bank';

export type Theta = ArrayLike<number>;

/** pi(.|x) into out[off .. off + K). Returns out. */
export function probsInto(theta: Theta, x: number, out: Float64Array = new Float64Array(K), off = 0): Float64Array {
  let m = -Infinity;
  for (let y = 0; y < K; y++) {
    const o = (x * K + y) * D;
    let z = C[x * K + y];
    for (let j = 0; j < D; j++) z += theta[j] * PHI[o + j];
    out[off + y] = z;
    if (z > m) m = z;
  }
  let s = 0;
  for (let y = 0; y < K; y++) { out[off + y] = Math.exp(out[off + y] - m); s += out[off + y]; }
  for (let y = 0; y < K; y++) out[off + y] /= s;
  return out;
}
/** pi(y|x) for every prompt, flat P * K. */
export function probs(theta: Theta): Float64Array {
  const out = new Float64Array(P * K);
  for (let x = 0; x < P; x++) probsInto(theta, x, out, x * K);
  return out;
}
export const logp = (theta: Theta, x: number, y: number): number => Math.log(probsInto(theta, x)[y]);

/** w . phi(x, y) for every prompt and candidate (a reward table). */
export function scores(w: Theta): Float64Array {
  const out = new Float64Array(P * K);
  for (let i = 0; i < P * K; i++) { let s = 0; for (let j = 0; j < D; j++) s += w[j] * PHI[i * D + j]; out[i] = s; }
  return out;
}
/** The gold utility of every candidate. */
export const GOLD = scores(W_GOLD);

/** Mean over `prompts` of E_pi[r(x, y)]. */
export function expected(theta: Theta, r: ArrayLike<number>, prompts: number[] = TRAIN): number {
  const p = new Float64Array(K);
  let s = 0;
  for (const x of prompts) { probsInto(theta, x, p); for (let y = 0; y < K; y++) s += p[y] * r[x * K + y]; }
  return s / prompts.length;
}
/** Mean over `prompts` of E_pi[phi]. */
export function expectedFeatures(theta: Theta, prompts: number[] = TRAIN): Float64Array {
  const p = new Float64Array(K), out = new Float64Array(D);
  for (const x of prompts) { probsInto(theta, x, p); for (let y = 0; y < K; y++) for (let j = 0; j < D; j++) out[j] += p[y] * PHI[(x * K + y) * D + j]; }
  for (let j = 0; j < D; j++) out[j] /= prompts.length;
  return out;
}
/** Probability of candidate y averaged over prompts. */
export function meanProb(theta: Theta, y: number, prompts: number[] = TRAIN): number {
  const p = new Float64Array(K);
  let s = 0;
  for (const x of prompts) s += probsInto(theta, x, p)[y];
  return s / prompts.length;
}
/** Exact KL(pi_theta || pi_ref), summed over the 8 candidates and averaged over prompts. */
export function kl(theta: Theta, ref: Theta, prompts: number[] = TRAIN): number {
  const p = new Float64Array(K), q = new Float64Array(K);
  let s = 0;
  for (const x of prompts) {
    probsInto(theta, x, p); probsInto(ref, x, q);
    for (let y = 0; y < K; y++) if (p[y] > 0) s += p[y] * Math.log(p[y] / q[y]);
  }
  return Math.max(0, s / prompts.length);
}
/** Exact gradient of mean_x E_pi[r(x, y)]: mean_x E_pi[(r - E_pi r) phi]. */
export function rewardGrad(theta: Theta, r: ArrayLike<number>, prompts: number[] = TRAIN, out = new Float64Array(D)): Float64Array {
  const p = new Float64Array(K);
  out.fill(0);
  for (const x of prompts) {
    probsInto(theta, x, p);
    let er = 0;
    for (let y = 0; y < K; y++) er += p[y] * r[x * K + y];
    for (let y = 0; y < K; y++) { const a = p[y] * (r[x * K + y] - er), o = (x * K + y) * D; for (let j = 0; j < D; j++) out[j] += a * PHI[o + j]; }
  }
  for (let j = 0; j < D; j++) out[j] /= prompts.length;
  return out;
}
/** grad_theta log pi(y|x) = phi(x, y) - E_pi(.|x)[phi], added into out with weight w. */
export function addGradLogp(theta: Theta, x: number, y: number, w: number, out: Float64Array, p: Float64Array = probsInto(theta, x)): void {
  for (let j = 0; j < D; j++) {
    let e = 0;
    for (let z = 0; z < K; z++) e += p[z] * PHI[(x * K + z) * D + j];
    out[j] += w * (PHI[(x * K + y) * D + j] - e);
  }
}
/** Sample a candidate from a probability row. */
export function sampleFrom(p: ArrayLike<number>, u: number, off = 0, n = K): number {
  let c = 0;
  for (let y = 0; y < n; y++) { c += p[off + y]; if (u < c) return y; }
  return n - 1;
}
export const sample = (theta: Theta, x: number, rng: Rng): number => sampleFrom(probsInto(theta, x), rng.next());
