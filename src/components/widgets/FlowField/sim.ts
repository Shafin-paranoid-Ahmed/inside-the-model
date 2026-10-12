/**
 * Flow matching on the two moons, pure and DOM-free.
 * Path: x_t = t x1 + (1 - t) x0, with x0 noise at t = 0 and x1 data at t = 1 (the opposite direction of the diffusion
 * lesson's tau). One pair's velocity is the constant x1 - x0. The exact marginal velocity of the moons mixture is
 *   v*(x, t) = (E[x1 | x_t = x] - x) / (1 - t)
 * from the shared core's posterior mean (a = t, b = 1 - t). The learned field is the same 2+5 -> 64 -> 64 -> 2 MLP,
 * trained for the same budget as the diffusion lesson's noise network (1,800 updates, batch 64).
 */
import { createRng } from '../../../lib/math/rng';
import { MOONS, posteriorMean, quality } from '../../../lib/gen/toy';
import { initMlp, createCache, forward, backward, createAdam, adam, timeFeatures, TIME_FEATURES, normal, type Cache } from '../../../lib/gen/mlp';
import { N_DATA, TRAIN_BATCH, TRAIN_STEPS, SNAP_STEPS, moonData, startNoise, reversePath, type Path } from '../DiffusionDenoiser/sim';

export { MOONS, quality, startNoise, moonData, reversePath, N_DATA, TRAIN_STEPS };
export type { Path };
export type Mode = 'flow' | 'pairs' | 'marginal' | 'train' | 'sample' | 'compare';

export const FLOW_SIZES = [2 + TIME_FEATURES, 64, 64, 2];
export const TRAIN_SEED = 12, LR = 1e-2;
/** Euler step counts whose samples the worker records. */
export const STEP_COUNTS = [1, 2, 4, 8, 64];
/** Times at which learned and exact velocities are compared. */
export const ANGLE_TS = [0.5, 0.9];
export const T_MAX = 0.995;

/** Batched velocity field: velocities of n points (2n floats) at time t, into out. */
export type VelFn = (t: number, cur: Float64Array, n: number, out: Float64Array) => void;

const e2 = [0, 0];
/** The exact marginal velocity of the moons mixture. t is clamped below 1, where the field is singular. */
export const exactVelFn: VelFn = (t, cur, n, out) => {
  const tt = Math.min(t, T_MAX);
  for (let i = 0; i < n; i++) {
    posteriorMean(MOONS, tt, 1 - tt, cur[2 * i], cur[2 * i + 1], e2);
    out[2 * i] = (e2[0] - cur[2 * i]) / (1 - tt); out[2 * i + 1] = (e2[1] - cur[2 * i + 1]) / (1 - tt);
  }
};

let netCache: Cache | null = null;
export const netVelFn = (p: Float64Array): VelFn => (t, cur, n, out) => {
  if (!netCache || netCache.B !== n) netCache = createCache(FLOW_SIZES, n);
  const f = new Float64Array(TIME_FEATURES), X = new Float64Array(n * 7);
  timeFeatures(t, f);
  for (let i = 0; i < n; i++) { X[i * 7] = cur[2 * i]; X[i * 7 + 1] = cur[2 * i + 1]; for (let k = 0; k < TIME_FEATURES; k++) X[i * 7 + 2 + k] = f[k]; }
  const y = forward(p, FLOW_SIZES, X, n, netCache);
  for (let i = 0; i < 2 * n; i++) out[i] = y[i];
};

/** Euler steps x += h v(x, t) from x0 (2n floats), recording every state: (S + 1) x 2n. A run you can advance a few steps at a time. */
export function eulerBuilder(vf: VelFn, x0: ArrayLike<number>, S: number) {
  const n = x0.length / 2, xs = new Float32Array((S + 1) * 2 * n), cur = Float64Array.from(x0), v = new Float64Array(2 * n);
  xs.set(cur, 0);
  let s = 0;
  return {
    path: { S, xs, n } as Path,
    get done() { return s >= S; },
    step() {
      if (s >= S) return;
      vf(s / S, cur, n, v);
      for (let i = 0; i < 2 * n; i++) cur[i] += v[i] / S;
      s++; xs.set(cur, s * 2 * n);
    },
  };
}
export function eulerPath(vf: VelFn, x0: ArrayLike<number>, S: number): Path {
  const b = eulerBuilder(vf, x0, S);
  while (!b.done) b.step();
  return b.path;
}
/** Positions at progress u in [0, 1] along a recorded path (straight chords between recorded states). */
export function pathAtU(p: Path, u: number, out: Float32Array): void {
  const f = Math.min(p.S, Math.max(0, u * p.S)), s = Math.min(p.S - 1, Math.floor(f)), w = f - s, L = 2 * p.n;
  for (let i = 0; i < L; i++) out[i] = p.xs[s * L + i] * (1 - w) + p.xs[(s + 1) * L + i] * w;
}
/** Total path length over total straight start-to-end distance, summed over the first `count` points (a mean of per-point ratios would be blown up by points that end near where they started). */
export function pathStats(p: Path, count = p.n): { ratio: number } {
  let len = 0, chord = 0;
  for (let i = 0; i < count; i++) {
    for (let s = 0; s < p.S; s++) len += Math.hypot(p.xs[(s + 1) * 2 * p.n + 2 * i] - p.xs[s * 2 * p.n + 2 * i], p.xs[(s + 1) * 2 * p.n + 2 * i + 1] - p.xs[s * 2 * p.n + 2 * i + 1]);
    chord += Math.hypot(p.xs[p.S * 2 * p.n + 2 * i] - p.xs[2 * i], p.xs[p.S * 2 * p.n + 2 * i + 1] - p.xs[2 * i + 1]);
  }
  return { ratio: len / Math.max(chord, 1e-9) };
}

export interface FlowRun {
  snaps: Float64Array[]; steps: number[]; losses: Float32Array;
  /** angles[i][k]: mean angle in degrees between learned and exact velocity at t = ANGLE_TS[i], over the 600 pairs' points, for snapshot k. */
  angles: number[][];
  /** Learned Euler samples from the fixed noise for each step count (final snapshot). Filled by the worker. */
  paths?: Record<number, Path>;
}

/** Points of the 600 pairs at time t: x_t = t x1 + (1 - t) x0, with the page's fixed noise as x0. */
export function pairPoints(t: number, n = N_DATA): Float64Array {
  const d = moonData().x, z = startNoise(n), out = new Float64Array(2 * n);
  for (let i = 0; i < 2 * n; i++) out[i] = t * d[i] + (1 - t) * z[i];
  return out;
}
export function meanAngle(p: Float64Array, t: number): number {
  const pts = pairPoints(t), n = pts.length / 2, a = new Float64Array(2 * n), e = new Float64Array(2 * n);
  netVelFn(p)(t, pts, n, a); exactVelFn(t, pts, n, e);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = (a[2 * i] * e[2 * i] + a[2 * i + 1] * e[2 * i + 1]) / (Math.hypot(a[2 * i], a[2 * i + 1]) * Math.hypot(e[2 * i], e[2 * i + 1]) + 1e-12);
    s += Math.acos(Math.max(-1, Math.min(1, d)));
  }
  return (s / n) * (180 / Math.PI);
}

/** Conditional flow matching: random pair (x0 noise, x1 data), random t, regress v(x_t, t) onto x1 - x0. */
export function trainFlow(steps = TRAIN_STEPS, onProgress?: (f: number) => void, seed = TRAIN_SEED): FlowRun {
  const data = moonData().x, r = createRng(seed), p = initMlp(FLOW_SIZES, r), g = new Float64Array(p.length), st = createAdam(p.length);
  const B = TRAIN_BATCH, cache = createCache(FLOW_SIZES, B), X = new Float64Array(B * 7), Y = new Float64Array(B * 2), dY = new Float64Array(B * 2), f = new Float64Array(TIME_FEATURES);
  const losses = new Float32Array(steps), snaps: Float64Array[] = [], taken: number[] = [], want = SNAP_STEPS.filter((s) => s < steps).concat(steps);
  const angles: number[][] = ANGLE_TS.map(() => []);
  let w = 0;
  const snap = (s: number) => { snaps.push(p.slice()); taken.push(s); ANGLE_TS.forEach((tt, i) => angles[i].push(meanAngle(p, tt))); w++; };
  for (let s = 0; s <= steps; s++) {
    if (s === want[w]) snap(s);
    if (s === steps) break;
    for (let j = 0; j < B; j++) {
      const i = Math.floor(r.next() * N_DATA), t = r.next(), x0 = normal(r), y0 = normal(r), x1 = data[2 * i], y1 = data[2 * i + 1];
      X[j * 7] = t * x1 + (1 - t) * x0; X[j * 7 + 1] = t * y1 + (1 - t) * y0;
      timeFeatures(t, f); for (let k = 0; k < TIME_FEATURES; k++) X[j * 7 + 2 + k] = f[k];
      Y[2 * j] = x1 - x0; Y[2 * j + 1] = y1 - y0;
    }
    const out = forward(p, FLOW_SIZES, X, B, cache);
    let L = 0;
    for (let i = 0; i < 2 * B; i++) { const d = out[i] - Y[i]; L += d * d; dY[i] = (2 * d) / (2 * B); }
    losses[s] = L / (2 * B);
    g.fill(0); backward(p, FLOW_SIZES, cache, dY, g);
    adam(p, g, st, LR * (1 - 0.9 * (s / steps)));
    if (onProgress && s % 50 === 0) onProgress(s / steps);
  }
  return { snaps, steps: taken, losses, angles };
}
export function lossAt(run: FlowRun, step: number): number {
  const lo = Math.max(0, step - 25), hi = Math.min(run.losses.length, step + 26);
  let s = 0; for (let i = lo; i < hi; i++) s += run.losses[i];
  return s / (hi - lo);
}
/** The loss of the exact marginal field on the training distribution: the floor, because each target is one random pairing. */
export function lossFloor(n = 3000, seed = 22): number {
  const d = moonData().x, r = createRng(seed), cur = new Float64Array(2), v = new Float64Array(2);
  let L = 0;
  for (let j = 0; j < n; j++) {
    const i = Math.floor(r.next() * N_DATA), t = r.next(), x0 = normal(r), y0 = normal(r);
    cur[0] = t * d[2 * i] + (1 - t) * x0; cur[1] = t * d[2 * i + 1] + (1 - t) * y0;
    exactVelFn(t, cur, 1, v);
    L += ((v[0] - (d[2 * i] - x0)) ** 2 + (v[1] - (d[2 * i + 1] - y0)) ** 2) / 2;
  }
  return L / n;
}

/** What the worker returns for the final network: samples for each step count. */
export function samplePaths(p: Float64Array): Record<number, Path> {
  const z = startNoise(), vf = netVelFn(p), out: Record<number, Path> = {};
  for (const S of STEP_COUNTS) out[S] = eulerPath(vf, z, S);
  return out;
}
export const endPoints = (p: Path) => p.xs.subarray(p.S * 2 * p.n);

/** The spot where the "one velocity per place" figure looks, at time T_SPOT. */
export const T_SPOT = 0.5, SPOT: [number, number] = [0.5, 0], SPOT_K = 24;
/**
 * K paths through the spot at T_SPOT: each comes from a data point x1 drawn from the exact posterior p(x1 | x_t = spot),
 * and its velocity is (x1 - spot) / (1 - t). Returns those velocities (2K floats), and the marginal field's arrow v* (their average).
 */
export function spotVelocities(seed = 3, K = SPOT_K): { vs: number[]; exact: [number, number] } {
  const m = MOONS, a = T_SPOT, b = 1 - T_SPOT, s2 = m.sigma * m.sigma, v = a * a * s2 + b * b, g = (a * s2) / v, sd = Math.sqrt(s2 * (1 - a * g));
  const r = createRng(seed), w = new Float64Array(m.K), vs: number[] = [];
  let z = 0;
  for (let k = 0; k < m.K; k++) { const dx = SPOT[0] - a * m.mu[2 * k], dy = SPOT[1] - a * m.mu[2 * k + 1]; w[k] = Math.exp(-(dx * dx + dy * dy) / (2 * v)); z += w[k]; }
  for (let j = 0; j < K; j++) {
    let u = r.next() * z, k = 0;
    while (k < m.K - 1 && u > w[k]) { u -= w[k]; k++; }
    const x1 = m.mu[2 * k] + g * (SPOT[0] - a * m.mu[2 * k]) + sd * normal(r), y1 = m.mu[2 * k + 1] + g * (SPOT[1] - a * m.mu[2 * k + 1]) + sd * normal(r);
    vs.push((x1 - SPOT[0]) / b, (y1 - SPOT[1]) / b);
  }
  const cur = Float64Array.of(SPOT[0], SPOT[1]), ve = new Float64Array(2);
  exactVelFn(T_SPOT, cur, 1, ve);
  return { vs, exact: [ve[0], ve[1]] };
}

// ---- timelines ----
export interface Seg { at: number; label: string }
export interface ModeTime { dur: number; markers: Seg[] }
export const T_RUN = 7, T_HOLD = 1.2;
export const T_SNAP = 1.1;
/** In `marginal`, the path time t sweeps 0 to T_SPOT, holds there to show the crossing, then sweeps on to 1. */
export const M_A = 3, M_B = 4.6, M_END = 8.6;
export function modeTime(mode: Mode, nSnap = SNAP_STEPS.length): ModeTime {
  switch (mode) {
    case 'flow': return { dur: T_RUN + T_HOLD, markers: [{ at: 0, label: 'noise' }, { at: T_RUN, label: 'data' }] };
    case 'pairs': return { dur: T_RUN + T_HOLD, markers: [{ at: 0, label: 'noise' }, { at: T_RUN, label: 'data' }] };
    case 'marginal': return { dur: M_END, markers: [{ at: 0, label: 'approach' }, { at: M_A, label: 'one spot' }, { at: M_B, label: 'on to data' }] };
    case 'train': return { dur: nSnap * T_SNAP, markers: [{ at: 0, label: 'start' }, { at: 2 * T_SNAP, label: 'early' }, { at: 4 * T_SNAP, label: 'middle' }, { at: 6 * T_SNAP, label: 'late' }] };
    default: return { dur: T_RUN + T_HOLD, markers: [{ at: 0, label: 'noise' }, { at: T_RUN, label: 'data' }] };
  }
}
/** Path time t in [0, 1] at local timeline time s. */
export function pathTime(mode: Mode, s: number): number {
  if (mode === 'marginal') return s < M_A ? (s / M_A) * T_SPOT : s < M_B ? T_SPOT : T_SPOT + ((Math.min(s, M_END) - M_B) / (M_END - M_B)) * (1 - T_SPOT);
  return Math.min(1, Math.max(0, s / T_RUN));
}
