/**
 * LatentSpace: autoencoders (and, later, variational autoencoders) on the Day 13 two-moons data. Pure, no DOM.
 * A model is an encoder [2, 32, 32, E] and a decoder [L, 32, 32, 2] stored in one parameter vector (encoder first).
 * E = L for an autoencoder; a VAE encoder outputs a mean and a log-variance per code number (E = 2L).
 * Each model trains in its own worker from a fixed seed, so every reload (and Node) gives the same weights.
 */
import { createRng } from '../../../lib/math/rng';
import {
  initMlp, createCache, forward, backward, createAdam, adam, exp, normal, paramCount, type Cache,
} from '../../../lib/gen/mlp';
import { MOONS, sampleData, quality, DOMAIN } from '../../../lib/gen/toy';

export { DOMAIN, MOONS, quality };
export const N = 600, SEED = 7;
export const DATA = sampleData(MOONS, N, SEED);

export type ModelId = 'ae1' | 'ae2' | 'dae' | 'vae0.03' | 'vae0.1' | 'vae1' | 'vae4';
export interface Spec { id: ModelId; latent: 1 | 2; noise: number; vae: boolean; beta: number }
export const SPECS: Record<ModelId, Spec> = {
  ae1: { id: 'ae1', latent: 1, noise: 0, vae: false, beta: 0 },
  ae2: { id: 'ae2', latent: 2, noise: 0, vae: false, beta: 0 },
  dae: { id: 'dae', latent: 1, noise: 0.25, vae: false, beta: 0 },
  'vae0.03': { id: 'vae0.03', latent: 1, noise: 0, vae: true, beta: 0.03 },
  'vae0.1': { id: 'vae0.1', latent: 1, noise: 0, vae: true, beta: 0.1 },
  vae1: { id: 'vae1', latent: 1, noise: 0, vae: true, beta: 1 },
  vae4: { id: 'vae4', latent: 1, noise: 0, vae: true, beta: 4 },
};
export const HIDDEN = 32, STEPS = 1500, BATCH = 64, LR = 0.003, SNAPS = 30;

export type Kind = 'ae' | 'dae' | 'vae';
/** The KL weights the VAE figures can show, smallest first. */
export const VAE_BETAS = [0.03, 0.1, 1, 4] as const;
/** Which trained model a figure needs. */
export function specFor(model: Kind, latentDim: 1 | 2 = 1, beta = 1): Spec {
  if (model === 'dae') return SPECS.dae;
  if (model === 'vae') return SPECS[`vae${beta}` as ModelId] ?? SPECS['vae0.1'];
  return latentDim === 2 ? SPECS.ae2 : SPECS.ae1;
}

const encSizes = (s: Spec) => [2, HIDDEN, HIDDEN, s.vae ? 2 * s.latent : s.latent];
const decSizes = (s: Spec) => [s.latent, HIDDEN, HIDDEN, 2];
const split = (p: Float64Array, s: Spec) => { const ne = paramCount(encSizes(s)); return { pe: p.subarray(0, ne), pd: p.subarray(ne), ne }; };

export interface Run {
  id: ModelId;
  /** parameters at SNAPS + 1 checkpoints (index 0 = untrained) */
  snaps: Float64Array[];
  steps: number[];
  /** training loss of every step (reconstruction, plus beta times the KL for a VAE) */
  losses: Float32Array;
  /** average squared reconstruction distance on the 600 training points at each checkpoint */
  snapLoss: Float32Array;
  /** the two loss terms at each checkpoint, on the 600 points with fixed noise: reconstruction (squared distance) and KL (nats; 0 for an autoencoder) */
  snapRec: Float32Array;
  snapKl: Float32Array;
  /** a VAE's mean code of each training point at each checkpoint (empty for other models) */
  snapMu: Float32Array[];
}
export const finalParams = (r: Run) => r.snaps[r.snaps.length - 1];

const caches = new Map<string, Cache>();
const cacheFor = (key: string, sizes: number[], B: number) => {
  const k = key + ':' + B;
  let c = caches.get(k);
  if (!c) caches.set(k, (c = createCache(sizes, B)));
  return c;
};

/** Codes of n points (n x 2) -> n x latent (for a VAE: the mean). */
export function encodeAll(p: Float64Array, s: Spec, X: ArrayLike<number>, n: number): Float64Array {
  const { pe } = split(p, s), E = encSizes(s)[3], out = new Float64Array(n * s.latent);
  const y = forward(pe, encSizes(s), X, n, cacheFor('e' + s.id, encSizes(s), n));
  for (let i = 0; i < n; i++) for (let j = 0; j < s.latent; j++) out[i * s.latent + j] = y[i * E + j];
  return out;
}
/** A VAE's encoder output for n points: the mean and the standard deviation of each point's bell curve. */
export function encodeDist(p: Float64Array, s: Spec, X: ArrayLike<number>, n: number): { mu: Float64Array; sigma: Float64Array } {
  const { pe } = split(p, s), E = encSizes(s)[3], L = s.latent, mu = new Float64Array(n * L), sigma = new Float64Array(n * L);
  const y = forward(pe, encSizes(s), X, n, cacheFor('e' + s.id, encSizes(s), n));
  for (let i = 0; i < n; i++) for (let j = 0; j < L; j++) { mu[i * L + j] = y[i * E + j]; sigma[i * L + j] = s.vae ? exp(0.5 * y[i * E + L + j]) : 0; }
  return { mu, sigma };
}
export function decodeAll(p: Float64Array, s: Spec, Z: ArrayLike<number>, n: number): Float64Array {
  const { pd } = split(p, s);
  return Float64Array.from(forward(pd, decSizes(s), Z, n, cacheFor('d' + s.id, decSizes(s), n)).subarray(0, 2 * n));
}
export const reconstruct = (p: Float64Array, s: Spec, X: ArrayLike<number>, n: number) => decodeAll(p, s, encodeAll(p, s, X, n), n);

/** Average squared distance between n point pairs. */
export function sqErr(a: ArrayLike<number>, b: ArrayLike<number>, n: number): number {
  let e = 0;
  for (let i = 0; i < 2 * n; i++) { const d = a[i] - b[i]; e += d * d; }
  return e / n;
}

/**
 * One loss evaluation (and, if `g` is given, its gradient ADDED into g) on a batch: X inputs, T targets (both B x 2) and the
 * fixed noise eps (B x latent, used by a VAE only). Returns the mean reconstruction term (squared distance) and the mean KL.
 * The loss is rec + beta * kl. Exposed so tests can check the gradient against finite differences.
 */
export function makeStepper(s: Spec, B: number) {
  const es = encSizes(s), ds = decSizes(s), E = es[3], L = s.latent;
  const ce = createCache(es, B), cd = createCache(ds, B);
  const Z = new Float64Array(B * L), dY = new Float64Array(B * 2), dZ = new Float64Array(B * E);
  return (p: Float64Array, X: ArrayLike<number>, T: ArrayLike<number>, eps: ArrayLike<number>, g: Float64Array | null): { rec: number; kl: number } => {
    const { pe, pd } = split(p, s);
    const H = forward(pe, es, X, B, ce);
    let kl = 0;
    for (let b = 0; b < B; b++) for (let j = 0; j < L; j++) {
      if (!s.vae) { Z[b * L + j] = H[b * E + j]; continue; }
      const mu = H[b * E + j], lv = H[b * E + L + j];
      Z[b * L + j] = mu + exp(0.5 * lv) * eps[b * L + j];
      kl += 0.5 * (mu * mu + exp(lv) - lv - 1);
    }
    const Y = forward(pd, ds, Z, B, cd);
    let rec = 0;
    for (let i = 0; i < 2 * B; i++) { const d = Y[i] - T[i]; rec += d * d; dY[i] = (2 * d) / B; }
    if (g) {
      backward(pd, ds, cd, dY, g.subarray(paramCount(es)), B);
      for (let b = 0; b < B; b++) for (let j = 0; j < L; j++) {
        const dz = cd.dIn[b * L + j];
        if (!s.vae) { dZ[b * E + j] = dz; continue; }
        const mu = H[b * E + j], lv = H[b * E + L + j];
        dZ[b * E + j] = dz + (s.beta * mu) / B;
        dZ[b * E + L + j] = dz * 0.5 * exp(0.5 * lv) * eps[b * L + j] + (s.beta * 0.5 * (exp(lv) - 1)) / B;
      }
      backward(pe, es, ce, dZ, g.subarray(0, paramCount(es)), B);
    }
    return { rec: rec / B, kl: kl / B };
  };
}

/** Fixed noise for reporting a VAE's two loss terms at a checkpoint. */
const reportEps = (() => { const r = createRng(99), e = new Float64Array(N); for (let i = 0; i < N; i++) e[i] = normal(r); return e; })();

export function train(id: ModelId, steps = STEPS, onProgress?: (f: number) => void): Run {
  const s = SPECS[id], L = s.latent;
  const rng = createRng(11), batchRng = createRng(12);
  const p = Float64Array.from([...initMlp(encSizes(s), rng), ...initMlp(decSizes(s), rng)]);
  const g = new Float64Array(p.length), st = createAdam(p.length), step = makeStepper(s, BATCH), report = makeStepper(s, N);
  const X = new Float64Array(BATCH * 2), T = new Float64Array(BATCH * 2), eps = new Float64Array(BATCH * L);
  const snaps = [p.slice()], stepAt = [0], losses = new Float32Array(steps), snapLoss: number[] = [], snapRec: number[] = [], snapKl: number[] = [], snapMu: Float32Array[] = [];
  const checkpoint = () => {
    const Zc = encodeAll(p, s, DATA.x, N);
    snapLoss.push(sqErr(decodeAll(p, s, Zc, N), DATA.x, N));
    if (s.vae) snapMu.push(Float32Array.from(Zc));
    const { rec, kl } = report(p, DATA.x, DATA.x, reportEps, null);
    snapRec.push(rec); snapKl.push(kl);
  };
  checkpoint();
  for (let t = 1; t <= steps; t++) {
    for (let b = 0; b < BATCH; b++) {
      const i = Math.min(N - 1, Math.floor(batchRng.next() * N));
      T[2 * b] = DATA.x[2 * i]; T[2 * b + 1] = DATA.x[2 * i + 1];
      X[2 * b] = T[2 * b]; X[2 * b + 1] = T[2 * b + 1];
      if (s.noise) { X[2 * b] += s.noise * normal(batchRng); X[2 * b + 1] += s.noise * normal(batchRng); }
    }
    if (s.vae) for (let i = 0; i < BATCH * L; i++) eps[i] = normal(batchRng);
    g.fill(0);
    const { rec, kl } = step(p, X, T, eps, g);
    adam(p, g, st, LR);
    losses[t - 1] = rec + s.beta * kl;
    if (t % (steps / SNAPS) === 0) { snaps.push(p.slice()); stepAt.push(t); checkpoint(); }
    if (onProgress && t % 25 === 0) onProgress(t / steps);
  }
  return { id, snaps, steps: stepAt, losses, snapLoss: Float32Array.from(snapLoss), snapRec: Float32Array.from(snapRec), snapKl: Float32Array.from(snapKl), snapMu };
}

// ---------------------------------------------------------------- what each figure shows (pure: everything drawn at time t)
export type Mode = 'tour' | 'focus' | 'train' | 'explore' | 'sample' | 'denoise' | 'encode' | 'reparam' | 'beta';
export interface Marker { at: number; label: string }
export const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
export const ease = (v: number) => { const x = clamp01(v); return x * x * (3 - 2 * x); };

/** Points shown off the data in the "tour" figure when `probes` is on: far from both moons. */
export const PROBES: [number, number][] = [[0, 0], [1.4, 0.8], [-1.2, 0]];
/** The training point the "focus" figure follows: near the left tip of the upper moon. */
export const FOCUS_DEFAULT = 59;
export const CURVE_N = 160, SAMPLES = 300, WIDE = 4, DENOISE_SIGMA = 0.25, DENOISE_POINTS = 150;
const ON_R2 = 9 * MOONS.sigma * MOONS.sigma;

/** Distance to the nearest mixture mean, and the moon (0 upper, 1 lower) it belongs to. */
export function nearest(x: number, y: number): { d: number; moon: number } {
  let b = Infinity, k = 0;
  for (let i = 0; i < MOONS.K; i++) { const dx = x - MOONS.mu[2 * i], dy = y - MOONS.mu[2 * i + 1], d = dx * dx + dy * dy; if (d < b) { b = d; k = i; } }
  return { d: Math.sqrt(b), moon: k < MOONS.upper ? 0 : 1 };
}
const onData = (x: number, y: number) => nearest(x, y).d ** 2 <= ON_R2;

export interface Scene {
  spec: Spec; run: Run; p: Float64Array; L: number;
  /** points shown: the 600 training points, then the probes if asked for */
  n: number; X: Float64Array; Z: Float64Array; R: Float64Array; moon: Uint8Array; jit: Float32Array;
  /** a VAE's standard deviation per point (zeros for an autoencoder) */
  sigma: Float64Array;
  mse: number;
  /** range of the training codes (a VAE: the bulk of the prior, -2.5 to 2.5); the strip shows a padded view */
  zlo: number; zhi: number; vlo: number; vhi: number; zm: number;
  curve: Float64Array; curves: Float64Array[];
  /** code histogram over [a, b], by moon. Counts for an autoencoder; densities (area 1) when the strip also shows the prior. `top` is the value at full height. */
  hist: { up: Float64Array; lo: Float64Array; a: number; b: number; top: number };
  prior: boolean;
  /** a VAE's code histogram at every checkpoint, same bins as `hist` (train figure) */
  snapHist: { up: Float64Array; lo: Float64Array }[];
  samples: { z: Float64Array; pt: Float64Array; on: Uint8Array; cum: Int32Array; q: { onData: number; upper: number; lower: number } }[];
  noise: Float64Array;
  /** the followed point's bell curve and BELL_K codes drawn from it (VAE) */
  bell: { focus: number; mu: number; sigma: number; z: Float64Array; pt: Float64Array; spread: number } | null;
  /** final values of the two loss terms */
  rec: number; kl: number;
}
export const HIST_BINS = 40, BELL_K = 40, PRIOR_VIEW = 3.3, PRIOR_RANGE = 2.5;
export const priorPdf = (z: number) => Math.exp(-0.5 * z * z) / 2.5066282746310002;
/** Probability that a standard normal falls in [a, b] (Simpson's rule on the density). */
export function priorMass(a: number, b: number): number {
  const n = 400, h = (b - a) / n;
  let sum = priorPdf(a) + priorPdf(b);
  for (let i = 1; i < n; i++) sum += (i % 2 ? 4 : 2) * priorPdf(a + i * h);
  return (sum * h) / 3;
}

export function buildScene(run: Run, mode: Mode, probes = false, showPrior = false, focus = FOCUS_DEFAULT): Scene {
  const spec = SPECS[run.id], L = spec.latent, p = finalParams(run), prior = spec.vae || showPrior;
  const m = mode === 'tour' && probes ? PROBES.length : 0, n = N + m;
  const X = new Float64Array(2 * n); X.set(DATA.x);
  PROBES.slice(0, m).forEach(([a, b], i) => { X[2 * (N + i)] = a; X[2 * (N + i) + 1] = b; });
  const { mu: Z, sigma } = encodeDist(p, spec, X, n), R = decodeAll(p, spec, Z, n);
  const moon = new Uint8Array(n); moon.set(DATA.moon);
  const jr = createRng(3), jit = new Float32Array(n); for (let i = 0; i < n; i++) jit[i] = jr.uniform(-1, 1);
  let zlo = Infinity, zhi = -Infinity, zm = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < L; j++) { const v = Z[i * L + j]; if (L === 1) { zlo = Math.min(zlo, v); zhi = Math.max(zhi, v); } zm = Math.max(zm, Math.abs(v)); }
  if (L === 2) { zlo = -zm; zhi = zm; }
  if (spec.vae) { zlo = -PRIOR_RANGE; zhi = PRIOR_RANGE; }
  const pad = 0.06 * (zhi - zlo), vlo = prior ? -PRIOR_VIEW : zlo - pad, vhi = prior ? PRIOR_VIEW : zhi + pad;
  const sweep = (q: Float64Array) => { const z = new Float64Array(CURVE_N); for (let i = 0; i < CURVE_N; i++) z[i] = zlo + ((zhi - zlo) * i) / (CURVE_N - 1); return decodeAll(q, spec, z, CURVE_N); };
  const curve = L === 1 ? sweep(p) : new Float64Array(0);
  const curves = mode === 'train' && L === 1 ? run.snaps.map(sweep) : [];
  const [ha, hb] = prior ? [vlo, vhi] : [zlo, zhi], bw = (hb - ha) / HIST_BINS;
  const hist = { up: new Float64Array(HIST_BINS), lo: new Float64Array(HIST_BINS), a: ha, b: hb, top: 1 };
  if (L === 1) {
    for (let i = 0; i < N; i++) {
      const k = Math.min(HIST_BINS - 1, Math.floor((Z[i] - ha) / bw));
      if (k >= 0) (moon[i] ? hist.lo : hist.up)[k] += prior ? 1 / (N * bw) : 1;
    }
    let mx = 0; for (let k = 0; k < HIST_BINS; k++) mx = Math.max(mx, hist.up[k] + hist.lo[k]);
    hist.top = prior ? Math.max(mx, priorPdf(0)) * 1.1 : mx;
  }
  const snapHist: Scene['snapHist'] = [];
  if (mode === 'train' && spec.vae) {
    for (const zz of run.snapMu) {
      const up = new Float64Array(HIST_BINS), lo = new Float64Array(HIST_BINS);
      for (let i = 0; i < N; i++) { const k = Math.min(HIST_BINS - 1, Math.floor((zz[i] - ha) / bw)); if (k >= 0) (moon[i] ? lo : up)[k] += 1 / (N * bw); }
      snapHist.push({ up, lo });
    }
  }
  const samples: Scene['samples'] = [];
  if (mode === 'sample' || mode === 'beta') {
    const r = createRng(21);
    const draw = (gen: () => number) => {
      const z = new Float64Array(SAMPLES); for (let i = 0; i < SAMPLES; i++) z[i] = gen();
      const pt = decodeAll(p, spec, z, SAMPLES), on = new Uint8Array(SAMPLES), cum = new Int32Array(SAMPLES + 1);
      for (let i = 0; i < SAMPLES; i++) { on[i] = onData(pt[2 * i], pt[2 * i + 1]) ? 1 : 0; cum[i + 1] = cum[i] + on[i]; }
      samples.push({ z, pt, on, cum, q: quality(pt, MOONS) });
    };
    if (prior) draw(() => normal(r));
    else { draw(() => r.uniform(zlo, zhi)); draw(() => r.uniform(-WIDE, WIDE)); }
  }
  const noise = new Float64Array(2 * DENOISE_POINTS);
  if (mode === 'denoise') { const r = createRng(5); for (let i = 0; i < noise.length; i++) noise[i] = normal(r); }
  let bell: Scene['bell'] = null;
  if (mode === 'encode' && spec.vae) {
    const r = createRng(33), z = new Float64Array(BELL_K);
    for (let k = 0; k < BELL_K; k++) z[k] = Z[focus] + sigma[focus] * normal(r);
    const pt = decodeAll(p, spec, z, BELL_K); let spread = 0;
    for (let k = 0; k < BELL_K; k++) spread += Math.hypot(pt[2 * k] - R[2 * focus], pt[2 * k + 1] - R[2 * focus + 1]);
    bell = { focus, mu: Z[focus], sigma: sigma[focus], z, pt, spread: spread / BELL_K };
  }
  const last = run.snapRec.length - 1;
  return { spec, run, p, L, n, X, Z, R, moon, jit, sigma, mse: sqErr(R, X, N), zlo, zhi, vlo, vhi, zm, curve, curves, hist, prior, snapHist, samples, noise, bell, rec: run.snapRec[last], kl: run.snapKl[last] };
}

// ---- timelines
const RT = {
  tour: { cut: [1.2, 3.4, 4.6, 6.8, 10], labels: ['Inputs', 'Encode', 'Codes', 'Decode', 'Rebuilt'] },
  focus: { cut: [1.2, 3.2, 4.6, 6.6, 9], labels: ['Input', 'Encoder', 'Code', 'Decoder', 'Output'] },
};
export const DWELL = 0.35, BETA_SEG = 4, BETA_FADE = 0.8, REPARAM_T = 12, ENCODE_T = 10;
/** Where a paused (reduced-motion) figure rests: the finished state of each mode. */
export const restAt = (mode: Mode, T: number): number => ({ tour: 9.9, focus: 8.9, train: T - 0.01, explore: 3.5, sample: 5.5, denoise: 5, encode: 9, reparam: 3.6, beta: BETA_SEG + 3 })[mode];
export function timing(mode: Mode, sc?: Scene): { T: number; markers: Marker[] } {
  if (mode === 'tour' || mode === 'focus') {
    const { cut, labels } = RT[mode];
    return { T: cut[4], markers: labels.map((label, i) => ({ at: i ? cut[i - 1] : 0, label })) };
  }
  if (mode === 'train') {
    const K = sc ? sc.run.snaps.length : SNAPS + 1;
    return { T: K * DWELL, markers: ([[0, 'Start'], [8, 'Middle'], [20, 'Late']] as [number, string][]).map(([k, label]) => ({ at: k * DWELL, label })) };
  }
  if (mode === 'explore') return { T: 14, markers: [{ at: 0, label: 'Left to right' }, { at: 7, label: 'Right to left' }] };
  if (mode === 'sample') return sc?.samples.length === 1 ? { T: 10, markers: [{ at: 0, label: 'Prior codes' }] } : { T: 14, markers: [{ at: 0, label: 'Training range' }, { at: 6.5, label: 'Wider range' }] };
  if (mode === 'encode') return { T: ENCODE_T, markers: [{ at: 0, label: 'Point' }, { at: 1.5, label: 'Draws' }, { at: 7.5, label: 'Spread' }] };
  if (mode === 'reparam') return { T: REPARAM_T, markers: [{ at: 0, label: 'Epsilon rising' }, { at: REPARAM_T / 2, label: 'Epsilon falling' }] };
  if (mode === 'beta') return { T: BETA_SEG * VAE_BETAS.length, markers: VAE_BETAS.map((b, k) => ({ at: k * BETA_SEG, label: `${b}` })) };
  return { T: 10, markers: [{ at: 0, label: 'Noise rising' }, { at: 4, label: 'Full noise' }, { at: 6, label: 'Noise falling' }] };
}

export function roundTrip(t: number, mode: 'tour' | 'focus') {
  const { cut, labels } = RT[mode], [a, b, c, d] = cut;
  const phase = t < a ? 0 : t < b ? 1 : t < c ? 2 : t < d ? 3 : 4;
  const mix = t < a ? 0 : t < b ? ease((t - a) / (b - a)) : t < c ? 1 : t < d ? 1 + ease((t - c) / (d - c)) : 2;
  return { phase, label: labels[phase], mix };
}

export function trainFrame(t: number, sc: Scene) {
  const k = Math.min(sc.run.snaps.length - 1, Math.floor(t / DWELL));
  return { k, step: sc.run.steps[k], loss: sc.run.snapLoss[k], rec: sc.run.snapRec[k], kl: sc.run.snapKl[k] };
}

export function exploreFrame(t: number, sc: Scene, zOver: number | null = null) {
  const q = (t / timing('explore').T) % 1, u = q < 0.5 ? ease(2 * q) : ease(2 - 2 * q);
  const z = zOver ?? sc.zlo + (sc.zhi - sc.zlo) * u, pt = decodeAll(sc.p, sc.spec, [z], 1);
  const nr = nearest(pt[0], pt[1]);
  return { z, x: pt[0], y: pt[1], near: nr.d <= 0.21, moon: nr.moon };
}

export function sampleFrame(t: number, sc: Scene) {
  const one = sc.samples.length === 1, batch = one || t < 6.5 ? 0 : 1, s = sc.samples[batch], n = SAMPLES * ease((t - (batch ? 6.5 : 0)) / (one ? 5 : 4.2));
  const whole = Math.floor(n);
  return { batch, n, whole, on: s.cum[whole], off: whole - s.cum[whole] };
}

/** The noise level rises to DENOISE_SIGMA, holds, and falls; each noisy input and the model's answer are real. */
export function denoiseFrame(t: number, sc: Scene) {
  const sigma = DENOISE_SIGMA * (t < 4 ? ease(t / 4) : t < 6 ? 1 : ease((10 - t) / 4));
  const m = DENOISE_POINTS, inp = new Float64Array(2 * m);
  for (let i = 0; i < m; i++) { inp[2 * i] = DATA.x[2 * i] + sigma * sc.noise[2 * i]; inp[2 * i + 1] = DATA.x[2 * i + 1] + sigma * sc.noise[2 * i + 1]; }
  const out = reconstruct(sc.p, sc.spec, inp, m);
  let before = 0, after = 0;
  for (let i = 0; i < m; i++) { before += nearest(inp[2 * i], inp[2 * i + 1]).d; after += nearest(out[2 * i], out[2 * i + 1]).d; }
  return { sigma, inp, out, before: before / m, after: after / m };
}

/** A VAE point's bell curve (encode figure): the point first, then codes drawn from the bell one by one. */
export function encodeFrame(t: number) {
  const n = BELL_K * ease((t - 1.5) / 6);
  return { phase: t < 1.5 ? 0 : t < 7.5 ? 1 : 2, n, whole: Math.floor(n) };
}

/** z = mu + sigma * eps with eps sweeping -2 to 2 and back; the decoded point is real. */
export function reparamFrame(t: number, sc: Scene, i = FOCUS_DEFAULT) {
  const q = (t / REPARAM_T) % 1, u = q < 0.5 ? ease(2 * q) : ease(2 - 2 * q), eps = -2 + 4 * u;
  const mu = sc.Z[i], sigma = sc.sigma[i], z = mu + sigma * eps, pt = decodeAll(sc.p, sc.spec, [z], 1);
  return { eps, z, mu, sigma, x: pt[0], y: pt[1] };
}

/** Which KL weight the beta figure shows: segment k, fading in from the segment before it. */
export function betaFrame(t: number, n = VAE_BETAS.length) {
  const k = Math.min(n - 1, Math.floor(t / BETA_SEG));
  return { k, fade: k === 0 ? 1 : ease((t - k * BETA_SEG) / BETA_FADE) };
}
