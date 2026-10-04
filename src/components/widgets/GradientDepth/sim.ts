/**
 * Pure logic for GradientDepth: a deep, bias-free MLP h_l = phi(W_l h_{l-1}) with W_l = (scale / sqrt(width)) * Z_l,
 * Z_l fixed standard-normal matrices. The loss is L = c . h_D with c a fixed unit vector, so the backward pass starts
 * from a gradient of norm 1 and each layer multiplies it by W_l^T diag(phi'(z_l)): a product of Jacobians.
 */
import { createRng } from '../../../lib/math/rng';

export type Act = 'sigmoid' | 'tanh' | 'relu';
export const ACTS: Act[] = ['sigmoid', 'tanh', 'relu'];
/** skip: residual connections, h_l = h_{l-1} + phi(z_l). */
export interface Cfg { act: Act; scale: number; skip?: boolean }
export interface Base { depth: number; width: number; Z: Float64Array[]; x: Float64Array; c: Float64Array }

/** Fixed random ingredients (seeded): reused for every activation and scale, so sliders move smoothly. */
export function makeBase(depth = 30, width = 24, seed = 7): Base {
  const rng = createRng(seed);
  const Z = Array.from({ length: depth }, () => Float64Array.from({ length: width * width }, () => rng.normal()));
  const x = Float64Array.from({ length: width }, () => rng.normal());
  const c = new Float64Array(width).fill(1 / Math.sqrt(width));
  return { depth, width, Z, x, c };
}

export const phi = (a: Act, z: number): number => (a === 'sigmoid' ? 1 / (1 + Math.exp(-z)) : a === 'tanh' ? Math.tanh(z) : z > 0 ? z : 0);
/** Derivative in terms of pre-activation z and output h. */
export const dphi = (a: Act, z: number, h: number): number => (a === 'sigmoid' ? h * (1 - h) : a === 'tanh' ? 1 - h * h : z > 0 ? 1 : 0);

/** Forward pass: pre-activations z[l] and activations h[l] (h[0] = input), l = 1..depth. */
export function forward(b: Base, cfg: Cfg, from = 0, h0: Float64Array = b.x) {
  const n = b.width, k = cfg.scale / Math.sqrt(n);
  const z: Float64Array[] = [new Float64Array(n)], h: Float64Array[] = [h0];
  for (let l = from + 1; l <= b.depth; l++) {
    const prev = h[h.length - 1], zl = new Float64Array(n), hl = new Float64Array(n), W = b.Z[l - 1];
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += W[i * n + j] * prev[j];
      zl[i] = k * s; hl[i] = phi(cfg.act, zl[i]) + (cfg.skip ? prev[i] : 0);
    }
    z.push(zl); h.push(hl);
  }
  return { z, h };
}

export const loss = (b: Base, cfg: Cfg, from = 0, h0: Float64Array = b.x): number => {
  const out = forward(b, cfg, from, h0).h.at(-1)!;
  let s = 0;
  for (let i = 0; i < b.width; i++) s += b.c[i] * out[i];
  return s;
};

export interface Profile {
  /** g[l] = dL/dh_l for l = 0..depth. */
  g: Float64Array[];
  /** norms[l] = ||g[l]||. */
  norms: number[];
  /** log10 of norms. */
  log: number[];
  /** Mean of phi'(z_l) over units, l = 1..depth (index l-1). */
  meanDeriv: number[];
  /** Geometric-mean gain per layer: (||g_0|| / ||g_D||)^(1/D). */
  gain: number;
}

export function profile(b: Base, cfg: Cfg): Profile {
  const n = b.width, D = b.depth, k = cfg.scale / Math.sqrt(n);
  const { z } = forward(b, cfg);
  const g: Float64Array[] = new Array(D + 1);
  g[D] = Float64Array.from(b.c);
  const meanDeriv: number[] = new Array(D);
  for (let l = D; l >= 1; l--) {
    const d = new Float64Array(n);
    let m = 0;
    for (let i = 0; i < n; i++) {
      const dp = dphi(cfg.act, z[l][i], phi(cfg.act, z[l][i])); // phi's own output, not h (which includes the skip)
      d[i] = dp * g[l][i]; m += dp;
    }
    meanDeriv[l - 1] = m / n;
    const W = b.Z[l - 1], out = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += W[i * n + j] * d[i];
      out[j] = k * s + (cfg.skip ? g[l][j] : 0);
    }
    g[l - 1] = out;
  }
  const norms = g.map((v) => Math.hypot(...v));
  const log = norms.map((v) => Math.log10(Math.max(v, 1e-300)));
  return { g, norms, log, meanDeriv, gain: 10 ** ((log[0] - log[D]) / D) };
}

/** Timeline: forward wave, then the backward wave down the depth. */
export const T = 9, FWD = [0, 2.2] as const, BWD = [2.8, 7.6] as const;
export const MARKERS = [{ at: 0, label: 'forward' }, { at: BWD[0], label: 'backward' }, { at: BWD[1], label: 'result' }];
const ease = (p: number) => { const q = Math.min(1, Math.max(0, p)); return q * q * (3 - 2 * q); };

export interface Frame {
  /** Forward front in layers (0..depth), or depth+1 when done. */
  fwd: number;
  /** Backward front measured from the output: layer l is reached when bwd > depth - l. */
  bwd: number;
  /** Reveal 0..1 of the bar at layer l = 0..depth. */
  reveal: number[];
  phase: 'forward' | 'backward' | 'result';
}

export function frameAt(t: number, depth: number): Frame {
  const p = (t - FWD[0]) / (FWD[1] - FWD[0]);
  const fwd = Math.min(1, Math.max(0, p)) * (depth + 1);
  const bwd = Math.min(1, Math.max(0, (t - BWD[0]) / (BWD[1] - BWD[0]))) * (depth + 1.5);
  const reveal = Array.from({ length: depth + 1 }, (_, l) => ease(bwd - (depth - l)));
  return { fwd, bwd, reveal, phase: t < BWD[0] ? 'forward' : t < BWD[1] ? 'backward' : 'result' };
}
