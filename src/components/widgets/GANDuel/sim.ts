/**
 * GANDuel sim: a tiny GAN on the Day 13 two moons, pure and DOM-free.
 * G: 2 -> 32 -> 32 -> 2 (tanh, linear out) from standard-normal noise. D: 2 -> 32 -> 32 -> 1, sigmoid on the logit.
 * Each step: one D update (real = 1, fake = 0, cross-entropy), then one G update with the non-saturating loss
 * -log D(G(z)); the gradient reaches G through D's input (cache.dIn). Adam both.
 */
import { createRng } from '../../../lib/math/rng';
import { MOONS, DOMAIN, sampleData, quality, type Mixture } from '../../../lib/gen/toy';
import { initMlp, createCache, forward, backward, createAdam, adam, normal, sigmoid, log, paramCount } from '../../../lib/gen/mlp';

export const G_SIZES = [2, 32, 32, 2];
export const D_SIZES = [2, 32, 32, 1];
export const NG = paramCount(G_SIZES);
export const ND = paramCount(D_SIZES);
export const DATA = sampleData(MOONS, 600, 7);
/** Eight Gaussians on a ring of radius 1.6: used only by the mode-collapse step, as a different dataset. */
export const RING: Mixture = {
  K: 8, sigma: 0.07, upper: 8,
  mu: Float64Array.from({ length: 16 }, (_, i) => 1.6 * (i % 2 ? Math.sin((Math.PI / 4) * (i >> 1)) : Math.cos((Math.PI / 4) * (i >> 1)))),
};
export const DATA_RING = sampleData(RING, 600, 7);
/** Which of the 8 ring modes a point belongs to (nearest centre within 3 sigma), or -1. */
export function ringMode(x: number, y: number): number {
  let best = Infinity, bk = -1;
  for (let k = 0; k < 8; k++) { const dx = x - RING.mu[2 * k], dy = y - RING.mu[2 * k + 1], d = dx * dx + dy * dy; if (d < best) { best = d; bk = k; } }
  return best <= 9 * RING.sigma ** 2 ? bk : -1;
}
export const BATCH = 64;
export const NOISE_N = 300; // fixed noise points used for every reported quality number

export interface GanCfg { seed: number; steps: number; lrD: number; lrG: number; snapEvery: number; freezeG?: boolean; b1?: number; ring?: boolean; gInit?: number }
export interface GanRun {
  cfg: GanCfg;
  /** params at each checkpoint: G (NG numbers) then D (ND numbers) */
  snaps: Float64Array[];
  steps: number[];
  /** per step: D loss, G loss */
  losses: Float32Array;
  /** per checkpoint (STAT numbers): onData, upper, lower of G(fixed noise); mean D on real data; mean D on generated; D accuracy */
  stats: Float32Array;
}
export const STAT = 6;

/** The noise every checkpoint is judged on. */
export function fixedNoise(n = NOISE_N, seed = 99): Float64Array {
  const r = createRng(seed), z = new Float64Array(2 * n);
  for (let i = 0; i < z.length; i++) z[i] = normal(r);
  return z;
}
const Z = fixedNoise();
const gc = createCache(G_SIZES, NOISE_N), dcN = createCache(D_SIZES, NOISE_N), dcR = createCache(D_SIZES, 600);

export function generate(snap: ArrayLike<number>, z: Float64Array = Z): Float64Array {
  const n = z.length / 2;
  const c = n === NOISE_N ? gc : createCache(G_SIZES, n);
  return Float64Array.from(forward((snap as Float64Array).subarray(0, NG), G_SIZES, z, n, c));
}
/** D(x) = sigmoid(logit) for a batch of points. */
export function discriminate(snap: ArrayLike<number>, pts: ArrayLike<number>): Float64Array {
  const n = pts.length / 2, c = createCache(D_SIZES, n);
  const o = forward((snap as Float64Array).subarray(NG, NG + ND), D_SIZES, pts, n, c);
  return Float64Array.from(o, sigmoid);
}
function stat(snap: Float64Array, out: Float32Array, at: number, ring: boolean) {
  const g = generate(snap), data = ring ? DATA_RING : DATA;
  let q = quality(g, ring ? RING : MOONS);
  if (ring) { // slot 1 = share of modes holding at least 5% of the on-data points, slot 2 unused
    const cnt = new Array(8).fill(0); let on = 0;
    for (let i = 0; i < NOISE_N; i++) { const m = ringMode(g[2 * i], g[2 * i + 1]); if (m >= 0) { cnt[m]++; on++; } }
    q = { onData: q.onData, upper: cnt.filter((c) => c >= 0.05 * on && on > 0).length / 8, lower: 0 };
  }
  const dp = (snap as Float64Array).subarray(NG, NG + ND);
  const dr = forward(dp, D_SIZES, data.x, 600, dcR), df = forward(dp, D_SIZES, g, NOISE_N, dcN);
  let sr = 0, sf = 0, hr = 0, hf = 0;
  for (let i = 0; i < 600; i++) { sr += sigmoid(dr[i]); if (dr[i] > 0) hr++; } // logit > 0 means D > 0.5
  for (let i = 0; i < NOISE_N; i++) { sf += sigmoid(df[i]); if (df[i] < 0) hf++; }
  out.set([q.onData, q.upper, q.lower, sr / 600, sf / NOISE_N, (hr / 600 + hf / NOISE_N) / 2], at);
}

export function trainGAN(cfg: GanCfg, onProgress?: (frac: number) => void): GanRun {
  const data = cfg.ring ? DATA_RING : DATA, rng = createRng(cfg.seed), B = BATCH, b1 = cfg.b1 ?? 0.5;
  const g = initMlp(G_SIZES, rng), d = initMlp(D_SIZES, rng);
  for (let i = NG - 2 * 33; i < NG; i++) g[i] *= cfg.gInit ?? 1; // small output layer: the cloud starts as a tight blob
  const gA = createAdam(NG, b1), dA = createAdam(ND, b1), gG = new Float64Array(NG), gD = new Float64Array(ND);
  const cG = createCache(G_SIZES, B), cDf = createCache(D_SIZES, B), cDr = createCache(D_SIZES, B);
  const zb = new Float64Array(2 * B), xr = new Float64Array(2 * B), dY = new Float64Array(B);
  const snaps: Float64Array[] = [], steps: number[] = [], losses = new Float32Array(2 * cfg.steps);
  const nSnap = Math.floor(cfg.steps / cfg.snapEvery) + 1, stats = new Float32Array(nSnap * STAT);
  const take = (s: number) => {
    const sn = new Float64Array(NG + ND); sn.set(g); sn.set(d, NG);
    stat(sn, stats, snaps.length * STAT, !!cfg.ring); snaps.push(sn); steps.push(s);
  };
  take(0);
  for (let s = 1; s <= cfg.steps; s++) {
    for (let i = 0; i < 2 * B; i++) zb[i] = normal(rng);
    for (let i = 0; i < B; i++) { const j = Math.floor(rng.next() * 600); xr[2 * i] = data.x[2 * j]; xr[2 * i + 1] = data.x[2 * j + 1]; }
    const fake = forward(g, G_SIZES, zb, B, cG);
    // D step: real -> 1, fake -> 0. Loss = mean of -log D(real) - log(1 - D(fake)). dLoss/dlogit = (D - label)/B.
    gD.fill(0);
    let lD = 0;
    let o = forward(d, D_SIZES, xr, B, cDr);
    for (let i = 0; i < B; i++) { const p = sigmoid(o[i]); lD -= log(Math.max(p, 1e-12)); dY[i] = (p - 1) / B; }
    backward(d, D_SIZES, cDr, dY, gD);
    o = forward(d, D_SIZES, fake, B, cDf);
    for (let i = 0; i < B; i++) { const p = sigmoid(o[i]); lD -= log(Math.max(1 - p, 1e-12)); dY[i] = p / B; }
    backward(d, D_SIZES, cDf, dY, gD);
    adam(d, gD, dA, cfg.lrD);
    // G step (non-saturating): loss = mean -log D(G(z)); dLoss/dlogit = (D - 1)/B, then dIn into G.
    let lG = 0;
    if (!cfg.freezeG) {
      o = forward(d, D_SIZES, fake, B, cDf);
      for (let i = 0; i < B; i++) { const p = sigmoid(o[i]); lG -= log(Math.max(p, 1e-12)); dY[i] = (p - 1) / B; }
      gD.fill(0);
      backward(d, D_SIZES, cDf, dY, gD); // gD discarded; we want cDf.dIn
      gG.fill(0);
      backward(g, G_SIZES, cG, cDf.dIn, gG);
      adam(g, gG, gA, cfg.lrG);
    }
    losses[2 * (s - 1)] = lD / B; losses[2 * (s - 1) + 1] = lG / B;
    if (s % cfg.snapEvery === 0) take(s);
    if (onProgress && s % 100 === 0) onProgress(s / cfg.steps);
  }
  return { cfg, snaps, steps, losses, stats };
}

// ------------------------------------------------------------------ the runs the lesson uses
export type RunName = 'stable' | 'unstable' | 'collapse' | 'ringOk' | 'disc';
/** All runs: batch 64, Adam (beta1 0.5). Rates and seeds were chosen by probing (plans/day-13.md section 7). */
export const RUN_CFG: Record<RunName, GanCfg> = {
  stable: { seed: 1, steps: 1500, lrD: 0.005, lrG: 0.0003, snapEvery: 25, gInit: 0.1 },
  unstable: { seed: 1, steps: 1500, lrD: 0.005, lrG: 0.004, snapEvery: 25, gInit: 0.1 },
  collapse: { seed: 1, steps: 1500, lrD: 0.005, lrG: 0.0003, snapEvery: 25, ring: true, gInit: 0.1 },
  ringOk: { seed: 1, steps: 1500, lrD: 0.003, lrG: 0.003, snapEvery: 25, ring: true, gInit: 0.1 },
  disc: { seed: 1, steps: 600, lrD: 0.005, lrG: 0, snapEvery: 10, freezeG: true, gInit: 0.1 },
};
export const train = (name: RunName, onProgress?: (f: number) => void): GanRun => trainGAN(RUN_CFG[name], onProgress);

// ------------------------------------------------------------------ pure helpers for drawing
const _gen = new WeakMap<Float64Array, Float64Array>();
/** G(fixed noise) for a checkpoint, memoised. */
export function points(snap: Float64Array): Float64Array {
  let g = _gen.get(snap);
  if (!g) _gen.set(snap, (g = generate(snap)));
  return g;
}
const _map = new WeakMap<Float64Array, Float32Array>();
export const MAP_N = 40;
/** D(x) on a MAP_N x MAP_N grid over the plane (row 0 = top, y = +DOMAIN), memoised. */
export function dMap(snap: Float64Array): Float32Array {
  let m = _map.get(snap);
  if (!m) {
    const pts = new Float64Array(2 * MAP_N * MAP_N);
    for (let r = 0; r < MAP_N; r++) for (let c = 0; c < MAP_N; c++) {
      pts[2 * (r * MAP_N + c)] = ((c + 0.5) / MAP_N * 2 - 1) * DOMAIN;
      pts[2 * (r * MAP_N + c) + 1] = (1 - (r + 0.5) / MAP_N * 2) * DOMAIN;
    }
    _map.set(snap, (m = Float32Array.from(discriminate(snap, pts))));
  }
  return m;
}
/** Checkpoint index at fraction u in [0, 1] of a run. */
export const snapAt = (run: GanRun, u: number): number => Math.min(run.snaps.length - 1, Math.max(0, Math.round(u * (run.snaps.length - 1))));
/** D's accuracy: real points called real (D > 0.5) and generated points called fake (D < 0.5), the two weighted equally. */
export function accuracy(snap: Float64Array, ring = false): number {
  const dr = discriminate(snap, (ring ? DATA_RING : DATA).x), df = discriminate(snap, points(snap));
  let a = 0, b = 0;
  for (const v of dr) if (v > 0.5) a++;
  for (const v of df) if (v < 0.5) b++;
  return (a / dr.length + b / df.length) / 2;
}
/** Moving average of one loss channel (0 = D, 1 = G) over `w` steps. */
export function smoothLoss(losses: Float32Array, ch: 0 | 1, w = 25): Float32Array {
  const n = losses.length / 2, out = new Float32Array(n);
  let s = 0;
  for (let i = 0; i < n; i++) { s += losses[2 * i + ch]; if (i >= w) s -= losses[2 * (i - w) + ch]; out[i] = s / Math.min(i + 1, w); }
  return out;
}
/** Share of generated points in each ring mode (8 numbers) and the share off the data. */
export function ringShares(g: ArrayLike<number>): { modes: number[]; off: number } {
  const cnt = new Array(8).fill(0); let off = 0; const n = g.length / 2;
  for (let i = 0; i < n; i++) { const m = ringMode(g[2 * i], g[2 * i + 1]); if (m >= 0) cnt[m]++; else off++; }
  return { modes: cnt.map((c) => c / n), off: off / n };
}

/** D(x) and the gradient of log D(x) with respect to x at each point (the direction G is pushed). */
export function pushDirections(snap: Float64Array, pts: ArrayLike<number>): { d: Float64Array; g: Float64Array } {
  const n = pts.length / 2, c = createCache(D_SIZES, n), dp = snap.subarray(NG, NG + ND);
  const o = forward(dp, D_SIZES, pts, n, c), d = new Float64Array(n), dY = new Float64Array(n), grad = new Float64Array(ND);
  for (let i = 0; i < n; i++) { d[i] = sigmoid(o[i]); dY[i] = 1 - d[i]; } // d log D / d logit = 1 - D
  backward(dp, D_SIZES, c, dY, grad);
  return { d, g: Float64Array.from(c.dIn) };
}

/** The noise grid of the generator step: GRID_N x GRID_N points on [-GRID_R, GRID_R]^2. */
export const GRID_N = 15, GRID_R = 2;
export const noiseGrid = (): Float64Array => {
  const z = new Float64Array(2 * GRID_N * GRID_N);
  for (let r = 0; r < GRID_N; r++) for (let c = 0; c < GRID_N; c++) {
    z[2 * (r * GRID_N + c)] = -GRID_R + (2 * GRID_R * c) / (GRID_N - 1);
    z[2 * (r * GRID_N + c) + 1] = GRID_R - (2 * GRID_R * r) / (GRID_N - 1);
  }
  return z;
};

// ------------------------------------------------------------------ the gradients step
export const GRAD_SNAP = 8; // stable run, step 200
export const GRAD_N = 60, GRAD_ETA = 0.25;
export interface GradDemo {
  from: Float64Array; arrow: Float64Array; to: Float64Array;
  /** mean D and mean |grad log D| over the points, before and after the move */
  dBefore: number; dAfter: number; lenBefore: number; lenAfter: number;
  /** points whose arrow turns them toward the nearest data centre / points farther than 3 sigma from the data */
  towardData: number; offData: number; onBefore: number; onAfter: number;
}
/** One exaggerated update: every point moves along its gradient, scaled so the mean move is GRAD_ETA plane units. */
export function gradDemo(snap: Float64Array): GradDemo {
  const from = points(snap).slice(0, 2 * GRAD_N), b = pushDirections(snap, from);
  let len = 0;
  for (let i = 0; i < GRAD_N; i++) len += Math.hypot(b.g[2 * i], b.g[2 * i + 1]);
  len /= GRAD_N;
  const arrow = new Float64Array(2 * GRAD_N), to = new Float64Array(2 * GRAD_N);
  for (let i = 0; i < 2 * GRAD_N; i++) { arrow[i] = (GRAD_ETA * b.g[i]) / len; to[i] = from[i] + arrow[i]; }
  const a = pushDirections(snap, to);
  let la = 0, db = 0, da = 0, toward = 0, off = 0;
  for (let i = 0; i < GRAD_N; i++) {
    la += Math.hypot(a.g[2 * i], a.g[2 * i + 1]); db += b.d[i]; da += a.d[i];
    let best = Infinity, bk = 0;
    for (let k = 0; k < MOONS.K; k++) { const dx = from[2 * i] - MOONS.mu[2 * k], dy = from[2 * i + 1] - MOONS.mu[2 * k + 1], d = dx * dx + dy * dy; if (d < best) { best = d; bk = k; } }
    if (best > 9 * MOONS.sigma ** 2) { off++; if (arrow[2 * i] * (MOONS.mu[2 * bk] - from[2 * i]) + arrow[2 * i + 1] * (MOONS.mu[2 * bk + 1] - from[2 * i + 1]) > 0) toward++; }
  }
  return { from, arrow, to, dBefore: db / GRAD_N, dAfter: da / GRAD_N, lenBefore: len, lenAfter: la / GRAD_N, towardData: toward, offData: off, onBefore: quality(from, MOONS).onData, onAfter: quality(to, MOONS).onData };
}

// ------------------------------------------------------------------ timelines (pure frames)
export type Mode = 'duel' | 'generator' | 'discriminator' | 'objective' | 'gradients' | 'collapse';
export const T_RUN = 12, T_GEN = 9, T_GRAD = 9, HOLD = 1.5;
export const durationOf = (m: Mode): number => (m === 'generator' ? T_GEN : m === 'gradients' ? T_GRAD : T_RUN);
const smooth = (x: number) => { const c = Math.min(1, Math.max(0, x)); return c * c * (3 - 2 * c); };

/** Training-run modes: which checkpoint is on screen at time t (floor, so a segment starts exactly at its marker). */
export function runFrame(run: GanRun, t: number): { idx: number; step: number; seg: number } {
  const n = run.snaps.length, u = Math.min(1, Math.max(0, t / T_RUN));
  const idx = Math.min(n - 1, Math.floor(u * (n - 1) + 1e-9));
  return { idx, step: run.steps[idx], seg: Math.min(2, Math.floor((run.steps[idx] / run.cfg.steps) * 3 + 1e-9)) };
}
/** Three segments of equal length, marker times at their first checkpoint. */
export function runMarkers(run: GanRun): { at: number; label: string }[] {
  const n = run.snaps.length, k = run.cfg.steps / 3;
  return [0, 1, 2].map((s) => ({ at: ((s * k) / run.cfg.snapEvery / (n - 1)) * T_RUN, label: ['Early', 'Middle', 'Late'][s] }));
}
export const GEN_MARKERS = [{ at: 0, label: 'Noise grid' }, { at: 2.5, label: 'Through G' }, { at: 6, label: 'Output' }];
/** Generator step: phase 0 noise grid, 1 flowing through G (s eases 0..1), 2 output. */
export function genFrame(t: number): { phase: 0 | 1 | 2; s: number } {
  return t < 2.5 ? { phase: 0, s: 0 } : t < 6 ? { phase: 1, s: smooth((t - 2.5) / 3.5) } : { phase: 2, s: 1 };
}
export const GRAD_MARKERS = [{ at: 0, label: 'Fakes' }, { at: 2, label: 'Arrows' }, { at: 4.5, label: 'One update' }, { at: 7, label: 'After' }];
export function gradFrame(t: number): { phase: 0 | 1 | 2 | 3; grow: number; move: number } {
  return { phase: t < 2 ? 0 : t < 4.5 ? 1 : t < 7 ? 2 : 3, grow: smooth((t - 2) / 2.5), move: smooth((t - 4.5) / 2.5) };
}
