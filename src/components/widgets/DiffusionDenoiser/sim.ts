/**
 * Diffusion on a 2D Gaussian-mixture target with the EXACT score (no network), pure and DOM-free.
 * Forward: VP-SDE of Song et al. 2021, x_t = sqrt(abar) x_0 + sqrt(1-abar) eps, beta(t) = 0.1 + 19.9 t.
 * The marginal p_t of a Gaussian mixture is again a Gaussian mixture, so its score is closed form.
 */
import { createRng } from '../../../lib/math/rng';
import { mixture, MOONS, sampleData as sampleCore, logp as logpAB, score as scoreAB, quality, type Mixture, type Dataset } from '../../../lib/gen/toy';
import { exp, normal, initMlp, createCache, forward, backward, createAdam, adam, timeFeatures, TIME_FEATURES, type Cache } from '../../../lib/gen/mlp';

export { mixture, MOONS, quality };
export type { Mixture, Dataset };
export type Sampler = 'ode' | 'sde';

const B_MIN = 0.1, B_MAX = 20;
export const TAU_MIN = 0.002;
export const beta = (tau: number) => B_MIN + tau * (B_MAX - B_MIN);
/** abar(tau) = exp(-int_0^tau beta). */
export const alphaBar = (tau: number) => exp(-(B_MIN * tau + 0.5 * (B_MAX - B_MIN) * tau * tau));

/** Exact samples of the target plus a 0..1 curve position (used only for coloring). */
export function sampleData(m: Mixture, n: number, seed: number) {
  const d = sampleCore(m, n, seed);
  return { ...d, u: Float32Array.from(d.k, (k) => k / (m.K - 1)) };
}

/** log p_tau(x) of the noised mixture. */
export const logp = (m: Mixture, tau: number, x: number, y: number) => logpAB(m, Math.sqrt(alphaBar(tau)), Math.sqrt(1 - alphaBar(tau)), x, y);
/** Exact score grad_x log p_tau(x) into out[0..1]. */
export const score = (m: Mixture, tau: number, x: number, y: number, out: number[] | Float64Array) =>
  scoreAB(m, Math.sqrt(alphaBar(tau)), Math.sqrt(1 - alphaBar(tau)), x, y, out);

/** The noise the ideal denoiser predicts: eps_hat = -sqrt(1-abar) * score. */
export const epsHat = (tau: number, sc: number) => -Math.sqrt(1 - alphaBar(tau)) * sc;

/** Forward noising in closed form with a fixed eps per point. */
export function forwardAt(tau: number, x0: Float32Array, eps: Float32Array, out: Float32Array) {
  const a = Math.sqrt(alphaBar(tau)), b = Math.sqrt(1 - alphaBar(tau));
  for (let i = 0; i < out.length; i++) out[i] = a * x0[i] + b * eps[i];
}

export interface Path { S: number; xs: Float32Array; n: number }
/** Batched score: scores of the n points in cur (2n floats) at diffusion time tau, into out (2n floats). */
export type ScoreFn = (tau: number, cur: Float64Array, n: number, out: Float64Array) => void;
/** The exact score of a mixture, optionally restricted to one moon. */
export const exactScoreFn = (m: Mixture, only?: 'upper' | 'lower'): ScoreFn => (tau, cur, n, out) => {
  const a = Math.sqrt(alphaBar(tau)), b = Math.sqrt(1 - alphaBar(tau)), sc = [0, 0];
  for (let i = 0; i < n; i++) { scoreAB(m, a, b, cur[2 * i], cur[2 * i + 1], sc, only); out[2 * i] = sc[0]; out[2 * i + 1] = sc[1]; }
};
/** Classifier-free guidance on the exact scores: (1 + w) s(x | c) - w s(x). w = 0 is plain conditional sampling. */
export const guidedScoreFn = (m: Mixture, only: 'upper' | 'lower', w: number): ScoreFn => (tau, cur, n, out) => {
  const a = Math.sqrt(alphaBar(tau)), b = Math.sqrt(1 - alphaBar(tau)), c = [0, 0], u = [0, 0];
  for (let i = 0; i < n; i++) {
    scoreAB(m, a, b, cur[2 * i], cur[2 * i + 1], c, only); scoreAB(m, a, b, cur[2 * i], cur[2 * i + 1], u);
    out[2 * i] = (1 + w) * c[0] - w * u[0]; out[2 * i + 1] = (1 + w) * c[1] - w * u[1];
  }
};
/** A reverse run you can advance a few steps at a time (so the main thread never stalls). */
export function reverseBuilder(sf: ScoreFn, xT: Float32Array, kind: Sampler, S: number, seed: number) {
  const n = xT.length / 2, xs = new Float32Array((S + 1) * 2 * n), r = createRng(seed), sc = new Float64Array(2 * n), cur = Float64Array.from(xT);
  xs.set(xT, 0);
  let s = 0;
  return {
    path: { S, xs, n } as Path,
    get done() { return s >= S; },
    get frac() { return s / S; },
    step() {
      if (s >= S) return;
      const tau = 1 - (s / S) * (1 - TAU_MIN), h = (1 - TAU_MIN) / S, b = beta(tau);
      sf(tau, cur, n, sc);
      for (let i = 0; i < n; i++) {
        const x = cur[2 * i], y = cur[2 * i + 1];
        if (kind === 'ode') {
          cur[2 * i] = x + h * 0.5 * b * (x + sc[2 * i]); cur[2 * i + 1] = y + h * 0.5 * b * (y + sc[2 * i + 1]);
        } else {
          const q = Math.sqrt(b * h);
          cur[2 * i] = x + h * b * (0.5 * x + sc[2 * i]) + q * normal(r); cur[2 * i + 1] = y + h * b * (0.5 * y + sc[2 * i + 1]) + q * normal(r);
        }
      }
      s++;
      xs.set(cur, s * 2 * n);
    },
  };
}
/** Reverse trajectories from x_T (n points, 2n floats) down to TAU_MIN. ODE = probability flow (DDIM-like), SDE = ancestral (DDPM-like). Exact score unless `sf` is given. */
export function reversePath(m: Mixture, xT: Float32Array, kind: Sampler, S = 240, seed = 1, sf: ScoreFn = exactScoreFn(m)): Path {
  const b = reverseBuilder(sf, xT, kind, S, seed);
  while (!b.done) b.step();
  return b.path;
}
/** Path positions at diffusion time tau (linear interpolation between stored steps). */
export function pathAt(p: Path, tau: number, out: Float32Array) {
  const f = Math.min(p.S, Math.max(0, ((1 - tau) / (1 - TAU_MIN)) * p.S)), s = Math.min(p.S - 1, Math.floor(f)), w = f - s, L = 2 * p.n;
  for (let i = 0; i < L; i++) out[i] = p.xs[s * L + i] * (1 - w) + p.xs[(s + 1) * L + i] * w;
}

// ---- timeline ----
export const T_FWD = 6, T_HOLD = 0.7, T_REV = 6, T_END = 0.9;
export const T = T_FWD + T_HOLD + T_REV + T_END;
export type Phase = 'data' | 'forward' | 'noise' | 'reverse' | 'clean';
export const MARKERS = [{ at: 0, label: 'noising' }, { at: T_FWD, label: 'denoising' }, { at: T - T_END, label: 'clean' }];
export function tauAt(t: number): { tau: number; phase: Phase } {
  const c = Math.min(Math.max(t, 0), T);
  if (c < T_FWD) return { tau: c / T_FWD, phase: c < 0.35 ? 'data' : 'forward' };
  if (c < T_FWD + T_HOLD) return { tau: 1, phase: 'noise' };
  const u = (c - T_FWD - T_HOLD) / T_REV;
  return u < 1 ? { tau: 1 - u * (1 - TAU_MIN), phase: 'reverse' } : { tau: TAU_MIN, phase: 'clean' };
}

// ---- the learned noise-prediction network ----
export const EPS_SIZES = [2 + TIME_FEATURES, 64, 64, 2];
export const N_DATA = 600, DATA_SEED = 7, TRAIN_SEED = 11, TRAIN_STEPS = 1800, TRAIN_BATCH = 64;
/** Updates completed when each snapshot was taken (the last is TRAIN_STEPS). */
export const SNAP_STEPS = [0, 25, 75, 200, 400, 650, 950, 1300, TRAIN_STEPS];
/** Noise levels at which learned and exact scores are compared (the first is the default view). */
export const ANGLE_TAUS = [0.3, 0.1];
export const moonData = () => sampleCore(MOONS, N_DATA, DATA_SEED);

export interface TrainRun {
  snaps: Float64Array[]; steps: number[]; losses: Float32Array;
  /** angles[i][k]: mean angle in degrees between the learned and the exact score at tau = ANGLE_TAUS[i], over the 600 noised data points, for snapshot k. */
  angles: number[][];
  /** Learned reverse samples from fixed noise (final snapshot). Filled by the worker. */
  paths?: Partial<Record<Sampler, Path>>;
}

/** Noise levels are drawn as tau = u^2 (more low-noise examples: the last steps of sampling decide whether points land on a moon); Adam lr 1e-2 decaying to 10%. */
const LR = 1e-2;
let netCache: Cache | null = null;
/** Learned score -eps_hat(x, tau) / sqrt(1 - abar) for n points. */
export const netScoreFn = (p: Float64Array): ScoreFn => (tau, cur, n, out) => {
  if (!netCache || netCache.B !== n) netCache = createCache(EPS_SIZES, n);
  const f = new Float64Array(TIME_FEATURES), b = Math.sqrt(1 - alphaBar(tau));
  timeFeatures(tau, f);
  const tmp = new Float64Array(n * 7);
  for (let i = 0; i < n; i++) { tmp[i * 7] = cur[2 * i]; tmp[i * 7 + 1] = cur[2 * i + 1]; for (let k = 0; k < TIME_FEATURES; k++) tmp[i * 7 + 2 + k] = f[k]; }
  const y = forward(p, EPS_SIZES, tmp, n, netCache);
  for (let i = 0; i < 2 * n; i++) out[i] = -y[i] / b;
};

/** Mean angle (degrees) between the network's score and the exact score at the given points. */
export function meanAngle(p: Float64Array, pts: Float64Array, tau: number): number {
  const n = pts.length / 2, a = new Float64Array(2 * n), e = new Float64Array(2 * n);
  netScoreFn(p)(tau, pts, n, a); exactScoreFn(MOONS)(tau, pts, n, e);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = (a[2 * i] * e[2 * i] + a[2 * i + 1] * e[2 * i + 1]) / (Math.hypot(a[2 * i], a[2 * i + 1]) * Math.hypot(e[2 * i], e[2 * i + 1]) + 1e-12);
    s += Math.acos(Math.max(-1, Math.min(1, d)));
  }
  return (s / n) * (180 / Math.PI);
}

/** The 600 data points noised to tau in closed form with the page's fixed noise. */
export function noisedData(tau: number): Float64Array {
  const d = moonData().x, r = createRng(8), out = new Float64Array(d.length), a = Math.sqrt(alphaBar(tau)), b = Math.sqrt(1 - alphaBar(tau));
  for (let i = 0; i < d.length; i++) out[i] = a * d[i] + b * normal(r);
  return out;
}

/** DDPM noise-prediction training on the 600 moons points: tau ~ U(TAU_MIN, 1), x_t in closed form, squared error to eps. */
export function trainEps(steps = TRAIN_STEPS, onProgress?: (f: number) => void, seed = TRAIN_SEED): TrainRun {
  const data = moonData().x, r = createRng(seed), p = initMlp(EPS_SIZES, r), g = new Float64Array(p.length), st = createAdam(p.length);
  const B = TRAIN_BATCH, cache = createCache(EPS_SIZES, B), X = new Float64Array(B * 7), Y = new Float64Array(B * 2), dY = new Float64Array(B * 2), f = new Float64Array(TIME_FEATURES);
  const losses = new Float32Array(steps), snaps: Float64Array[] = [], taken: number[] = [], want = SNAP_STEPS.filter((s) => s < steps).concat(steps);
  const pts = ANGLE_TAUS.map(noisedData), angles: number[][] = ANGLE_TAUS.map(() => []);
  let w = 0;
  const snap = (s: number) => { snaps.push(p.slice()); taken.push(s); ANGLE_TAUS.forEach((tt, i) => angles[i].push(meanAngle(p, pts[i], tt))); w++; };
  for (let s = 0; s <= steps; s++) {
    if (s === want[w]) snap(s);
    if (s === steps) break;
    for (let j = 0; j < B; j++) {
      const i = Math.floor(r.next() * N_DATA), u = r.next(), tau = TAU_MIN + (1 - TAU_MIN) * u * u, a = Math.sqrt(alphaBar(tau)), b = Math.sqrt(1 - alphaBar(tau));
      const e0 = normal(r), e1 = normal(r);
      X[j * 7] = a * data[2 * i] + b * e0; X[j * 7 + 1] = a * data[2 * i + 1] + b * e1;
      timeFeatures(tau, f); for (let k = 0; k < TIME_FEATURES; k++) X[j * 7 + 2 + k] = f[k];
      Y[2 * j] = e0; Y[2 * j + 1] = e1;
    }
    const out = forward(p, EPS_SIZES, X, B, cache);
    let L = 0;
    for (let i = 0; i < 2 * B; i++) { const d = out[i] - Y[i]; L += d * d; dY[i] = (2 * d) / (2 * B); }
    losses[s] = L / (2 * B);
    g.fill(0); backward(p, EPS_SIZES, cache, dY, g);
    adam(p, g, st, LR * (1 - 0.9 * (s / steps)));
    if (onProgress && s % 50 === 0) onProgress(s / steps);
  }
  return { snaps, steps: taken, losses, angles };
}
/** Loss at an update count: the mean over a 51-update window, so the curve is readable. */
export function lossAt(run: TrainRun, step: number): number {
  const lo = Math.max(0, step - 25), hi = Math.min(run.losses.length, step + 26);
  let s = 0; for (let i = lo; i < hi; i++) s += run.losses[i];
  return s / (hi - lo);
}

/** The fixed starting noise for every reverse run on the page: the 600 data points pushed to tau = 1 with the page's noise. */
export function startNoise(n = N_DATA): Float32Array {
  const d = moonData().x, r = createRng(8), out = new Float32Array(2 * n);
  forwardAt(1, d.subarray(0, 2 * n), Float32Array.from({ length: 2 * n }, () => normal(r)), out);
  return out;
}
export const REVERSE_STEPS = 100;
/** Steps of the (cheaper) guided runs. */
export const GUIDE_STEPS = 80;

/** Which moon (0 upper, 1 lower; for the spiral a 0..3 bin along the curve) each point is nearest to. */
export function groupOf(m: Mixture, pts: ArrayLike<number>, out: Uint8Array): void {
  for (let i = 0; i < out.length; i++) {
    let best = Infinity, bk = 0;
    for (let k = 0; k < m.K; k++) { const dx = pts[2 * i] - m.mu[2 * k], dy = pts[2 * i + 1] - m.mu[2 * k + 1], d = dx * dx + dy * dy; if (d < best) { best = d; bk = k; } }
    out[i] = m.upper < m.K ? (bk < m.upper ? 0 : 1) : Math.min(3, Math.floor((bk / m.K) * 4));
  }
}

// ---- 8x8 images ----
const G = [
  ['..####..', '.##..##.', '##....##', '##....##', '##....##', '##....##', '.##..##.', '..####..'],
  ['...##...', '..###...', '...##...', '...##...', '...##...', '...##...', '...##...', '.######.'],
  ['.#####..', '##...##.', '.....##.', '....##..', '...##...', '..##....', '.##.....', '#######.'],
  ['.#####..', '##...##.', '.....##.', '..####..', '.....##.', '.....##.', '##...##.', '.#####..'],
  ['....##..', '...###..', '..####..', '.##.##..', '##..##..', '#######.', '....##..', '....##..'],
  ['#######.', '##......', '######..', '.....##.', '.....##.', '.....##.', '##...##.', '.#####..'],
  ['..####..', '.##.....', '##......', '######..', '##...##.', '##...##.', '.##.##..', '..###...'],
  ['#######.', '.....##.', '....##..', '...##...', '..##....', '..##....', '..##....', '..##....'],
  ['.#####..', '##...##.', '##...##.', '.#####..', '##...##.', '##...##.', '##...##.', '.#####..'],
  ['.#####..', '##...##.', '##...##.', '.######.', '.....##.', '.....##.', '##..##..', '.####...'],
];
/** Ten digit glyphs, 64 pixels each, ink = +1 and paper = -1. */
export const GLYPHS: Float64Array[] = G.map((rows) => Float64Array.from(rows.join(''), (c) => (c === '#' ? 1 : -1)));
export const NPIX = 64;

/** Exact posterior over the glyphs for x = a g + b eps: weights w (10) and the mean E[g | x] (64). */
export function imagePosterior(x: ArrayLike<number>, tau: number, w: Float64Array, guess: Float64Array): void {
  const a = Math.sqrt(alphaBar(tau)), b2 = 1 - alphaBar(tau);
  let mx = -Infinity;
  for (let k = 0; k < GLYPHS.length; k++) {
    let d = 0; for (let i = 0; i < NPIX; i++) { const e = x[i] - a * GLYPHS[k][i]; d += e * e; }
    w[k] = -d / (2 * b2); if (w[k] > mx) mx = w[k];
  }
  let z = 0; for (let k = 0; k < GLYPHS.length; k++) { w[k] = exp(w[k] - mx); z += w[k]; }
  guess.fill(0);
  for (let k = 0; k < GLYPHS.length; k++) { w[k] /= z; for (let i = 0; i < NPIX; i++) guess[i] += w[k] * GLYPHS[k][i]; }
}
export interface ImageRun { glyph: number; eps: Float64Array; eps2: Float64Array; path: Float64Array /* (S+1) x 64: noise to image by the probability-flow ODE */; S: number }
/** One glyph noised with fixed noise `eps`; and, separately, the exact-denoiser ODE run from fresh noise `eps2` (generation never sees the glyph). */
export function imageRun(glyph: number, seed = 5, S = 100): ImageRun {
  const r = createRng(seed), eps = Float64Array.from({ length: NPIX }, () => normal(r)), eps2 = Float64Array.from({ length: NPIX }, () => normal(r));
  const path = new Float64Array((S + 1) * NPIX), cur = new Float64Array(NPIX), w = new Float64Array(10), gs = new Float64Array(NPIX);
  const b1 = Math.sqrt(1 - alphaBar(1));
  for (let i = 0; i < NPIX; i++) cur[i] = b1 * eps2[i];
  path.set(cur, 0);
  for (let s = 0; s < S; s++) {
    const tau = 1 - (s / S) * (1 - TAU_MIN), h = (1 - TAU_MIN) / S, b = beta(tau), a = Math.sqrt(alphaBar(tau)), b2 = 1 - alphaBar(tau);
    imagePosterior(cur, tau, w, gs);
    for (let i = 0; i < NPIX; i++) cur[i] += h * 0.5 * b * (cur[i] + (a * gs[i] - cur[i]) / b2);
    path.set(cur, (s + 1) * NPIX);
  }
  return { glyph, eps, eps2, path, S };
}
/** The three image panels at diffusion time tau: x_t, the guess E[x0 | x_t], and the noise implied by it. `up` = noising (closed form), else generating (ODE path). */
export function imageAt(run: ImageRun, tau: number, up: boolean, xt: Float64Array, guess: Float64Array, noise: Float64Array, w: Float64Array): void {
  const a = Math.sqrt(alphaBar(tau)), b = Math.sqrt(1 - alphaBar(tau));
  if (up) for (let i = 0; i < NPIX; i++) xt[i] = a * GLYPHS[run.glyph][i] + b * run.eps[i];
  else {
    const f = Math.min(run.S, Math.max(0, ((1 - tau) / (1 - TAU_MIN)) * run.S)), s = Math.min(run.S - 1, Math.floor(f)), u = f - s;
    for (let i = 0; i < NPIX; i++) xt[i] = run.path[s * NPIX + i] * (1 - u) + run.path[(s + 1) * NPIX + i] * u;
  }
  imagePosterior(xt, tau, w, guess);
  for (let i = 0; i < NPIX; i++) noise[i] = (xt[i] - a * guess[i]) / b;
}

// ---- timelines per mode ----
export type Mode = 'both' | 'forward' | 'reverse' | 'closed' | 'field' | 'train' | 'image' | 'guide';
export interface Seg { at: number; label: string }
export const T_CLOSED = 7.5, T_CLOSED_SWEEP = 6.5;
export const T_FIELD = 9.5, T_FIELD_SWEEP = 8.5, FIELD_TAU_MIN = 0.02;
export const T_SNAP = 1.1;
export const T_GUIDE = 7, T_GUIDE_RUN = 6;
export const T_IMG = 5 + 0.6 + 5 + 0.8;

export interface ModeTime { dur: number; off: number; markers: Seg[] }
/** Local timeline of a mode: duration, offset into the shared both-timeline (forward / reverse only), and segment markers. */
export function modeTime(mode: Mode, nSnap = SNAP_STEPS.length): ModeTime {
  switch (mode) {
    case 'forward': return { dur: T_FWD + T_HOLD, off: 0, markers: [{ at: 0, label: 'noising' }, { at: T_FWD, label: 'static' }] };
    case 'reverse': return { dur: T_REV + T_END, off: T_FWD + T_HOLD, markers: [{ at: 0, label: 'denoising' }, { at: T_REV, label: 'clean' }] };
    case 'closed': return { dur: T_CLOSED, off: 0, markers: [{ at: 0, label: 'first half' }, { at: T_CLOSED_SWEEP / 2, label: 'second half' }, { at: T_CLOSED_SWEEP, label: 'pure noise' }] };
    case 'field': return { dur: T_FIELD, off: 0, markers: [{ at: 0, label: 'noisy' }, { at: 3.4, label: 'mixed' }, { at: 6.8, label: 'sharp' }] };
    case 'train': return { dur: nSnap * T_SNAP, off: 0, markers: [{ at: 0, label: 'start' }, { at: 2 * T_SNAP, label: 'early' }, { at: 4 * T_SNAP, label: 'middle' }, { at: 6 * T_SNAP, label: 'late' }] };
    case 'image': return { dur: T_IMG, off: 0, markers: [{ at: 0, label: 'noising' }, { at: 5, label: 'generating' }, { at: 10.6, label: 'done' }] };
    case 'guide': return { dur: T_GUIDE, off: 0, markers: [{ at: 0, label: 'denoising' }, { at: T_GUIDE_RUN, label: 'sampled' }] };
    default: return { dur: T, off: 0, markers: MARKERS };
  }
}
/** Diffusion time for the local time of the single-sweep modes. */
export function sweepTau(mode: Mode, t: number): number {
  if (mode === 'closed') return Math.min(1, Math.max(0, t / T_CLOSED_SWEEP));
  if (mode === 'field') return 1 - (1 - FIELD_TAU_MIN) * Math.min(1, Math.max(0, t / T_FIELD_SWEEP));
  if (mode === 'guide') return 1 - (1 - TAU_MIN) * Math.min(1, Math.max(0, t / T_GUIDE_RUN));
  if (mode === 'image') return t < 5 ? t / 5 : t < 5.6 ? 1 : 1 - (1 - TAU_MIN) * Math.min(1, (t - 5.6) / 5);
  return 0;
}
/** The tau at which signal and noise shares are equal (abar = 1/2). */
export function tauHalf(): number {
  let lo = 0, hi = 1;
  for (let i = 0; i < 40; i++) { const m = (lo + hi) / 2; if (alphaBar(m) > 0.5) lo = m; else hi = m; }
  return lo;
}

/** Loss of the ideal denoiser (the exact eps-hat of the moons mixture) on the training distribution: the floor no network can beat. */
export function lossFloor(n = 3000, seed = 21): number {
  const d = moonData().x, r = createRng(seed), sc = [0, 0];
  let L = 0;
  for (let j = 0; j < n; j++) {
    const i = Math.floor(r.next() * N_DATA), u = r.next(), tau = TAU_MIN + (1 - TAU_MIN) * u * u, a = Math.sqrt(alphaBar(tau)), b = Math.sqrt(1 - alphaBar(tau));
    const e0 = normal(r), e1 = normal(r);
    scoreAB(MOONS, a, b, a * d[2 * i] + b * e0, a * d[2 * i + 1] + b * e1, sc);
    L += ((-b * sc[0] - e0) ** 2 + (-b * sc[1] - e1) ** 2) / 2;
  }
  return L / n;
}

/** Where points sit relative to one moon: share within 3 sigma of the data (anywhere), share on the chosen moon, and spread (std in components, out of 26) along it. */
export function moonStats(m: Mixture, pts: ArrayLike<number>, only: 'upper' | 'lower'): { onData: number; onMoon: number; spread: number; mean: number } {
  const n = pts.length / 2, r2 = 9 * m.sigma * m.sigma, k0 = only === 'upper' ? 0 : m.upper, k1 = only === 'upper' ? m.upper : m.K;
  let on = 0, chosen = 0, s = 0, s2 = 0;
  for (let i = 0; i < n; i++) {
    let best = Infinity, bk = 0;
    for (let k = 0; k < m.K; k++) { const dx = pts[2 * i] - m.mu[2 * k], dy = pts[2 * i + 1] - m.mu[2 * k + 1], d = dx * dx + dy * dy; if (d < best) { best = d; bk = k; } }
    if (best > r2) continue;
    on++;
    if (bk >= k0 && bk < k1) { chosen++; const q = bk - k0; s += q; s2 += q * q; }
  }
  const mean = chosen ? s / chosen : 0;
  return { onData: n ? on / n : 0, onMoon: n ? chosen / n : 0, spread: chosen ? Math.sqrt(Math.max(0, s2 / chosen - mean * mean)) : 0, mean };
}
