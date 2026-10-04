/**
 * Residual networks: what depth does to training, and why y = x + F(x) fixes it. Pure logic, no DOM.
 * The trainer is a small hand-written typed-array MLP (tanh layers, full-batch Adam), checked against numerical gradients.
 * Layers are narrow MLP layers, not convolutions: the optimization story (depth, skip, identity) is the same.
 */
import { createRng, type Rng } from '../../../lib/math/rng';

// ---------------------------------------------------------------- network
export type Act = 'tanh' | 'relu';
export interface NetSpec { depth: number; width: number; skip: boolean; act?: Act }
/** Parameter layout: stem (2 to width), `depth` hidden width-to-width layers, head (width to 1 logit). */
export interface Layout { offW: number[]; offB: number[]; inD: number[]; outD: number[]; size: number }

export function layout(s: NetSpec): Layout {
  const dims: [number, number][] = [[2, s.width]];
  for (let i = 0; i < s.depth; i++) dims.push([s.width, s.width]);
  dims.push([s.width, 1]);
  const offW: number[] = [], offB: number[] = [], inD: number[] = [], outD: number[] = [];
  let o = 0;
  for (const [i, j] of dims) { offW.push(o); o += i * j; offB.push(o); o += j; inD.push(i); outD.push(j); }
  return { offW, offB, inD, outD, size: o };
}

/** Standard init (Glorot for tanh, He for ReLU), zero biases. Nothing special for the skip net. */
export function initParams(s: NetSpec, rng: Rng): Float64Array {
  const L = layout(s), p = new Float64Array(L.size);
  for (let l = 0; l < L.inD.length; l++) {
    const sd = s.act === 'relu' ? Math.sqrt(2 / L.inD[l]) : Math.sqrt(2 / (L.inD[l] + L.outD[l]));
    for (let k = 0; k < L.inD[l] * L.outD[l]; k++) p[L.offW[l] + k] = rng.normal(0, sd);
  }
  return p;
}

export interface Data { X: Float64Array; y: Float64Array; n: number }

/** Two interleaved spirals, `turns` full turns, in roughly [-1, 1]^2. */
export function makeSpirals(n: number, turns: number, noise: number, rng: Rng): Data {
  const X = new Float64Array(2 * n), y = new Float64Array(n), m = n / 2;
  for (let i = 0; i < n; i++) {
    const c = i < m ? 0 : 1, k = (c ? i - m : i) / m, r = 0.12 + 0.88 * k, a = 2 * Math.PI * turns * k + c * Math.PI;
    X[2 * i] = r * Math.cos(a) + rng.normal(0, noise); X[2 * i + 1] = r * Math.sin(a) + rng.normal(0, noise); y[i] = c;
  }
  return { X, y, n };
}

interface Cache { H: Float64Array[]; U: (Float64Array | null)[]; logit: Float64Array }

/** Hidden layer l (1-based) is the first half of a block when skip is on and l is odd. */
const isA = (s: NetSpec, l: number) => s.skip && l % 2 === 1;
const isB = (s: NetSpec, l: number) => s.skip && l % 2 === 0;

export function forward(s: NetSpec, p: Float64Array, X: Float64Array, n: number): Cache {
  const L = layout(s), W = s.width, H: Float64Array[] = [], U: (Float64Array | null)[] = [], relu = s.act === 'relu';
  const lin = (l: number, inp: Float64Array, dst: Float64Array, act: boolean) => {
    const i = L.inD[l], o = L.outD[l], wo = L.offW[l], bo = L.offB[l];
    for (let r = 0; r < n; r++) for (let j = 0; j < o; j++) {
      let z = p[bo + j]; const wj = wo + j * i;
      for (let k = 0; k < i; k++) z += p[wj + k] * inp[r * i + k];
      dst[r * o + j] = act ? (relu ? (z > 0 ? z : 0) : Math.tanh(z)) : z;
    }
  };
  const h0 = new Float64Array(n * W); lin(0, X, h0, true); H.push(h0); U.push(null);
  let cur = h0, hin = h0;
  for (let l = 1; l <= s.depth; l++) {
    const out = new Float64Array(n * W);
    if (isA(s, l)) { hin = cur; lin(l, cur, out, true); U.push(out); H.push(out); cur = out; }
    else if (isB(s, l)) { lin(l, cur, out, false); for (let k = 0; k < n * W; k++) out[k] += hin[k]; U.push(null); H.push(out); cur = out; }
    else { lin(l, cur, out, true); U.push(null); H.push(out); cur = out; }
  }
  const logit = new Float64Array(n), hl = L.inD.length - 1;
  for (let r = 0; r < n; r++) { let z = p[L.offB[hl]]; for (let k = 0; k < W; k++) z += p[L.offW[hl] + k] * cur[r * W + k]; logit[r] = z; }
  return { H, U, logit };
}

const softplus = (z: number) => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));

/** Mean binary cross-entropy. */
export function lossOf(s: NetSpec, p: Float64Array, d: Data): number {
  const { logit } = forward(s, p, d.X, d.n); let t = 0;
  for (let r = 0; r < d.n; r++) t += softplus(logit[r]) - d.y[r] * logit[r];
  return t / d.n;
}

/** Loss and gradient by backpropagation. `hiddenNorms` (optional) receives the gradient norm at each hidden layer output. */
export function lossGrad(s: NetSpec, p: Float64Array, d: Data, g: Float64Array, hiddenNorms?: number[]): number {
  const L = layout(s), W = s.width, n = d.n, { H, U, logit } = forward(s, p, d.X, n);
  g.fill(0);
  let loss = 0; const dl = new Float64Array(n);
  for (let r = 0; r < n; r++) { loss += softplus(logit[r]) - d.y[r] * logit[r]; dl[r] = (1 / (1 + Math.exp(-logit[r])) - d.y[r]) / n; }
  loss /= n;
  const hl = L.inD.length - 1; let dh: Float64Array = new Float64Array(n * W);
  const last = H[H.length - 1];
  for (let r = 0; r < n; r++) {
    g[L.offB[hl]] += dl[r];
    for (let k = 0; k < W; k++) { g[L.offW[hl] + k] += dl[r] * last[r * W + k]; dh[r * W + k] = dl[r] * p[L.offW[hl] + k]; }
  }
  /** dz (n x o) to parameter grads of layer l with input `inp`, and the gradient wrt that input. */
  const back = (l: number, dz: Float64Array, inp: Float64Array): Float64Array => {
    const i = L.inD[l], o = L.outD[l], wo = L.offW[l], bo = L.offB[l], din = new Float64Array(n * i);
    for (let r = 0; r < n; r++) for (let j = 0; j < o; j++) {
      const v = dz[r * o + j]; if (v === 0) continue;
      g[bo + j] += v; const wj = wo + j * i;
      for (let k = 0; k < i; k++) { g[wj + k] += v * inp[r * i + k]; din[r * i + k] += v * p[wj + k]; }
    }
    return din;
  };
  const dact = s.act === 'relu' ? (h: number) => (h > 0 ? 1 : 0) : (h: number) => 1 - h * h; // derivative from the activated output
  const norm = (a: Float64Array) => { let t = 0; for (let k = 0; k < a.length; k++) t += a[k] * a[k]; return Math.sqrt(t); };
  for (let l = s.depth; l >= 1; l--) {
    if (hiddenNorms) hiddenNorms[l] = norm(dh);
    if (isB(s, l)) {
      const u = U[l - 1]!, hin = H[l - 2]; // H[k] is the output of layer k, so the block input is H[l-2]
      const du = back(l, dh, u);
      const dzu = new Float64Array(n * W); for (let k = 0; k < n * W; k++) dzu[k] = du[k] * dact(u[k]);
      const dhin = back(l - 1, dzu, hin);
      for (let k = 0; k < n * W; k++) dhin[k] += dh[k];
      dh = dhin; l--; if (hiddenNorms) hiddenNorms[l] = norm(dh);
    } else {
      const h = H[l], dz = new Float64Array(n * W); for (let k = 0; k < n * W; k++) dz[k] = dh[k] * dact(h[k]);
      dh = back(l, dz, H[l - 1]);
    }
  }
  if (hiddenNorms) hiddenNorms[0] = norm(dh);
  const h0 = H[0], dz0 = new Float64Array(n * W); for (let k = 0; k < n * W; k++) dz0[k] = dh[k] * dact(h0[k]);
  back(0, dz0, d.X);
  return loss;
}

export interface Adam { m: Float64Array; v: Float64Array; t: number }
export const newAdam = (size: number): Adam => ({ m: new Float64Array(size), v: new Float64Array(size), t: 0 });
export function adamStep(p: Float64Array, g: Float64Array, a: Adam, lr: number, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  a.t++; const c1 = 1 - b1 ** a.t, c2 = 1 - b2 ** a.t;
  for (let k = 0; k < p.length; k++) {
    a.m[k] = b1 * a.m[k] + (1 - b1) * g[k]; a.v[k] = b2 * a.v[k] + (1 - b2) * g[k] * g[k];
    p[k] -= (lr * (a.m[k] / c1)) / (Math.sqrt(a.v[k] / c2) + eps);
  }
}

/** Train from a fresh seeded init; returns the training loss at steps 0..steps. */
export function train(s: NetSpec, d: Data, steps: number, lr: number, seed: number): { loss: Float64Array; p: Float64Array } {
  const p = initParams(s, createRng(seed)), g = new Float64Array(p.length), a = newAdam(p.length), loss = new Float64Array(steps + 1);
  for (let k = 0; k <= steps; k++) { loss[k] = lossGrad(s, p, d, g); if (k < steps) adamStep(p, g, a, lr); }
  return { loss, p };
}

// ---------------------------------------------------------------- degradation experiment
export type NetId = 'shallow' | 'deep' | 'resnet';
export const NETS: { id: NetId; spec: NetSpec; label: string; short: string; color: string }[] = [
  { id: 'shallow', spec: { depth: 4, width: 10, skip: false }, label: '4 hidden layers, plain', short: '4 layers, plain', color: '--viz-forward' },
  { id: 'deep', spec: { depth: 20, width: 10, skip: false }, label: '20 hidden layers, plain', short: '20 layers, plain', color: '--viz-neg' },
  { id: 'resnet', spec: { depth: 20, width: 10, skip: true }, label: '20 hidden layers, with skips', short: '20 layers, skips', color: '--accent' },
];
export const STEPS = 800, SNAP = 8, GRID = 24, LR = 0.01, N_POINTS = 120, DATA_SEED = 7, DEFAULT_SEED = 2, EXTENT = 1.1;
export const LOSS_FLOOR = 1e-5;
export const dataset = () => makeSpirals(N_POINTS, 1.5, 0.02, createRng(DATA_SEED));

export interface NetRun {
  /** Training loss after 0..STEPS updates. */
  loss: Float32Array;
  /** Probability of class 1 on a GRID x GRID lattice (row-major, y up) every SNAP steps. */
  snaps: Float32Array;
  /** Gradient size at the first and last hidden layer on step 0. */
  gradFirst: number;
  gradLast: number;
}

/** Train one network from a seeded init with full-batch Adam, recording the loss every step and the decision map every SNAP steps. */
export function runNet(id: NetId, seed: number): NetRun {
  const spec = NETS.find((n) => n.id === id)!.spec, d = dataset();
  const p = initParams(spec, createRng(seed)), g = new Float64Array(p.length), a = newAdam(p.length);
  const loss = new Float32Array(STEPS + 1), snaps = new Float32Array((STEPS / SNAP + 1) * GRID * GRID);
  const gx = new Float64Array(2 * GRID * GRID);
  for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) {
    gx[2 * (j * GRID + i)] = -EXTENT + (2 * EXTENT * (i + 0.5)) / GRID; gx[2 * (j * GRID + i) + 1] = -EXTENT + (2 * EXTENT * (j + 0.5)) / GRID;
  }
  const norms: number[] = [];
  for (let k = 0; k <= STEPS; k++) {
    loss[k] = lossGrad(spec, p, d, g, k === 0 ? norms : undefined);
    if (k % SNAP === 0) {
      const { logit } = forward(spec, p, gx, GRID * GRID), o = (k / SNAP) * GRID * GRID;
      for (let c = 0; c < GRID * GRID; c++) snaps[o + c] = 1 / (1 + Math.exp(-logit[c]));
    }
    if (k < STEPS) adamStep(p, g, a, LR);
  }
  return { loss, snaps, gradFirst: norms[0], gradLast: norms[spec.depth] };
}

export const DEG_T = 16;
export const DEG_MARKERS = [{ at: 0, label: '0–200' }, { at: 4, label: '200–400' }, { at: 8, label: '400–600' }, { at: 12, label: '600–800' }];
export interface DegFrame { step: number; loss: number[]; snap: number; f: number }
/** Everything the degradation figure draws at time t (the timeline is training steps). */
export function degradationFrame(t: number, runs: Partial<Record<NetId, NetRun>>): DegFrame {
  const step = Math.round(Math.min(1, Math.max(0, t / DEG_T)) * STEPS), x = step / SNAP, snap = Math.min(STEPS / SNAP - 1, Math.floor(x));
  return { step, loss: NETS.map((n) => runs[n.id]?.loss[step] ?? NaN), snap, f: x - snap };
}

// ---------------------------------------------------------------- one residual block
export const BLOCK_D = 8, BLOCK_T = 12;
export const BLOCK_MARKERS = [{ at: 0, label: 'Flow' }, { at: 3, label: 'Shrink' }, { at: 8, label: 'Zero' }];
export interface Block { x: number[]; W1: number[]; W2: number[] }
export function makeBlock(seed = 5): Block {
  const r = createRng(seed), sd = Math.sqrt(2 / BLOCK_D);
  return {
    x: Array.from({ length: BLOCK_D }, () => r.uniform(-1, 1)),
    W1: Array.from({ length: BLOCK_D * BLOCK_D }, () => r.normal(0, sd)),
    W2: Array.from({ length: BLOCK_D * BLOCK_D }, () => r.normal(0, sd)),
  };
}
const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };
/** Factor applied to both weight matrices. */
export const blockScale = (t: number) => 1 - ease((t - 3) / 5);
export interface BlockFrame { w: number; x: number[]; branch: number[]; plain: number[]; resid: number[]; plainGap: number; residGap: number; flow: number }
/** F(x) = (wW2) relu((wW1) x); plain block outputs F(x), residual block outputs x + F(x). */
export function blockAt(t: number, b: Block): BlockFrame {
  const w = blockScale(t), D = BLOCK_D;
  const u = Array.from({ length: D }, (_, j) => { let z = 0; for (let k = 0; k < D; k++) z += w * b.W1[j * D + k] * b.x[k]; return Math.max(0, z); });
  const branch = Array.from({ length: D }, (_, j) => { let z = 0; for (let k = 0; k < D; k++) z += w * b.W2[j * D + k] * u[k]; return z; });
  const resid = branch.map((v, j) => v + b.x[j]);
  const dist = (y: number[]) => Math.hypot(...y.map((v, j) => v - b.x[j]));
  return { w, x: b.x, branch, plain: branch, resid, plainGap: dist(branch), residGap: dist(resid), flow: (t * 0.4) % 1 };
}

// ---------------------------------------------------------------- history
export interface Arch { name: string; year: number; layers: number; err: number | null; note: string }
/** Layer counts and ILSVRC top-5 test errors as stated in the original papers (see the lesson's references). */
export const ARCHS: Arch[] = [
  { name: 'LeNet-5', year: 1998, layers: 7, err: null, note: 'digits, not ImageNet' },
  { name: 'AlexNet', year: 2012, layers: 8, err: 15.3, note: 'ILSVRC 2012 winner' },
  { name: 'VGG-19', year: 2014, layers: 19, err: 7.3, note: 'ILSVRC 2014 runner-up (VGG-16 and VGG-19 ensemble)' },
  { name: 'ResNet-152', year: 2015, layers: 152, err: 3.57, note: 'ILSVRC 2015 winner (ensemble)' },
];
export const HIST_T = 12;
export const HIST_MARKERS = ARCHS.map((a, i) => ({ at: i * 3, label: String(a.year) }));
/** Reveal progress 0..1 of each network at time t. */
export const historyAt = (t: number) => ARCHS.map((_, i) => ease((t - i * 3) / 2));
