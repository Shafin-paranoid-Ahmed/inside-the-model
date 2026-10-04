/**
 * Diffusion on a 2D Gaussian-mixture target with the EXACT score (no network), pure and DOM-free.
 * Forward: VP-SDE of Song et al. 2021, x_t = sqrt(abar) x_0 + sqrt(1-abar) eps, beta(t) = 0.1 + 19.9 t.
 * The marginal p_t of a Gaussian mixture is again a Gaussian mixture, so its score is closed form.
 */
import { createRng } from '../../../lib/math/rng';

export type Dataset = 'spiral' | 'moons';
export type Sampler = 'ode' | 'sde';
export interface Mixture { K: number; mu: Float64Array; sigma: number }

const B_MIN = 0.1, B_MAX = 20;
export const TAU_MIN = 0.002;
export const beta = (tau: number) => B_MIN + tau * (B_MAX - B_MIN);
/** abar(tau) = exp(-int_0^tau beta). */
export const alphaBar = (tau: number) => Math.exp(-(B_MIN * tau + 0.5 * (B_MAX - B_MIN) * tau * tau));

export function mixture(ds: Dataset): Mixture {
  const pts: number[] = [];
  if (ds === 'spiral') {
    const K = 64;
    for (let k = 0; k < K; k++) {
      const u = k / (K - 1), a = 0.3 * Math.PI + u * 4 * Math.PI, r = 0.22 + 1.1 * u;
      pts.push(r * Math.cos(a), r * Math.sin(a));
    }
    return { K, mu: Float64Array.from(pts), sigma: 0.075 };
  }
  const M = 26;
  for (let k = 0; k < M; k++) { const a = (k / (M - 1)) * Math.PI; pts.push(1.25 * (Math.cos(a) - 0.5), 1.25 * (Math.sin(a) - 0.25)); }
  for (let k = 0; k < M; k++) { const a = (k / (M - 1)) * Math.PI; pts.push(1.25 * (1 - Math.cos(a) - 0.5), 1.25 * (0.5 - Math.sin(a) - 0.25)); }
  return { K: 2 * M, mu: Float64Array.from(pts), sigma: 0.07 };
}

/** Exact samples of the target plus a 0..1 curve position (used only for coloring). */
export function sampleData(m: Mixture, n: number, seed: number) {
  const r = createRng(seed);
  const x = new Float32Array(2 * n), u = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const k = Math.min(m.K - 1, Math.floor(r.next() * m.K));
    x[2 * i] = m.mu[2 * k] + m.sigma * r.normal(); x[2 * i + 1] = m.mu[2 * k + 1] + m.sigma * r.normal();
    u[i] = k / (m.K - 1);
  }
  return { x, u };
}

/** Marginal variance of each mixture component at diffusion time tau. */
const varAt = (m: Mixture, ab: number) => ab * m.sigma * m.sigma + (1 - ab);

/** log p_tau(x) of the noised mixture. */
export function logp(m: Mixture, tau: number, x: number, y: number): number {
  const ab = alphaBar(tau), v = varAt(m, ab), s = Math.sqrt(ab);
  let mx = -Infinity; const l: number[] = [];
  for (let k = 0; k < m.K; k++) {
    const dx = x - s * m.mu[2 * k], dy = y - s * m.mu[2 * k + 1], e = -(dx * dx + dy * dy) / (2 * v);
    l.push(e); if (e > mx) mx = e;
  }
  let sum = 0; for (const e of l) sum += Math.exp(e - mx);
  return mx + Math.log(sum / m.K) - Math.log(2 * Math.PI * v);
}

let buf = new Float64Array(0);
/** Exact score grad_x log p_tau(x) into out[0..1]. */
export function score(m: Mixture, tau: number, x: number, y: number, out: number[] | Float64Array): void {
  const ab = alphaBar(tau), v = varAt(m, ab), s = Math.sqrt(ab), K = m.K;
  if (buf.length < K) buf = new Float64Array(K);
  let mx = -Infinity;
  for (let k = 0; k < K; k++) {
    const dx = x - s * m.mu[2 * k], dy = y - s * m.mu[2 * k + 1];
    buf[k] = -(dx * dx + dy * dy) / (2 * v);
    if (buf[k] > mx) mx = buf[k];
  }
  let z = 0, ax = 0, ay = 0;
  for (let k = 0; k < K; k++) { const w = Math.exp(buf[k] - mx); z += w; ax += w * s * m.mu[2 * k]; ay += w * s * m.mu[2 * k + 1]; }
  out[0] = (ax / z - x) / v; out[1] = (ay / z - y) / v;
}

/** The noise the ideal denoiser predicts: eps_hat = -sqrt(1-abar) * score. */
export const epsHat = (tau: number, sc: number) => -Math.sqrt(1 - alphaBar(tau)) * sc;

/** Forward noising in closed form with a fixed eps per point. */
export function forwardAt(tau: number, x0: Float32Array, eps: Float32Array, out: Float32Array) {
  const a = Math.sqrt(alphaBar(tau)), b = Math.sqrt(1 - alphaBar(tau));
  for (let i = 0; i < out.length; i++) out[i] = a * x0[i] + b * eps[i];
}

export interface Path { S: number; xs: Float32Array; n: number }
/** Reverse trajectories from x_T (n points, 2n floats) down to TAU_MIN. ODE = probability flow (DDIM-like), SDE = ancestral (DDPM-like). */
export function reversePath(m: Mixture, xT: Float32Array, kind: Sampler, S = 240, seed = 1): Path {
  const n = xT.length / 2, xs = new Float32Array((S + 1) * 2 * n);
  const r = createRng(seed), sc = [0, 0], cur = Float64Array.from(xT);
  xs.set(xT, 0);
  for (let s = 0; s < S; s++) {
    const tau = 1 - (s / S) * (1 - TAU_MIN), h = (1 - TAU_MIN) / S, b = beta(tau);
    for (let i = 0; i < n; i++) {
      const x = cur[2 * i], y = cur[2 * i + 1];
      score(m, tau, x, y, sc);
      if (kind === 'ode') {
        cur[2 * i] = x + h * 0.5 * b * (x + sc[0]); cur[2 * i + 1] = y + h * 0.5 * b * (y + sc[1]);
      } else {
        const q = Math.sqrt(b * h);
        cur[2 * i] = x + h * b * (0.5 * x + sc[0]) + q * r.normal(); cur[2 * i + 1] = y + h * b * (0.5 * y + sc[1]) + q * r.normal();
      }
    }
    xs.set(cur, (s + 1) * 2 * n);
  }
  return { S, xs, n };
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
export const MARKERS = [
  { at: 0, label: 'data' }, { at: 0.35, label: 'forward' }, { at: T_FWD, label: 'noise' },
  { at: T_FWD + T_HOLD, label: 'reverse' }, { at: T - T_END, label: 'clean' },
];
export function tauAt(t: number): { tau: number; phase: Phase } {
  const c = Math.min(Math.max(t, 0), T);
  if (c < T_FWD) return { tau: c / T_FWD, phase: c < 0.35 ? 'data' : 'forward' };
  if (c < T_FWD + T_HOLD) return { tau: 1, phase: 'noise' };
  const u = (c - T_FWD - T_HOLD) / T_REV;
  return u < 1 ? { tau: 1 - u * (1 - TAU_MIN), phase: 'reverse' } : { tau: TAU_MIN, phase: 'clean' };
}
