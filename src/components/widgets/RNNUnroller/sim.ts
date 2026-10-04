/**
 * RNNUnroller math: a tiny seeded RNN / LSTM / GRU, its forward pass, real BPTT (via the shared
 * scalar autograd), and the pure timeline `frameAt`. No DOM.
 *
 * Notation (same in every Day 8 lesson):
 *   RNN   h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)
 *   LSTM  f,i,o = sigmoid(W_x x_t + W_h h_{t-1} + b),  g = tanh(...),  c_t = f*c_{t-1} + i*g,  h_t = o*tanh(c_t)
 *   GRU   z,r = sigmoid(...),  n = tanh(W_x x_t + b + r*(W_h h_{t-1})),  h_t = (1-z)*h_{t-1} + z*n
 * Readout o_t = v . h_t and loss L = 1/2 * sum over loss steps (o_t - y_t)^2.
 */
import { Value, prepareBackward } from '../../../lib/nn';
import { createRng, type Rng } from '../../../lib/math/rng';

export type Cell = 'rnn' | 'lstm' | 'gru';
export type LossAt = 'last' | 'all';
export type Mat = number[][];
export const H = 4; // hidden size
export const D = 4; // input size: [random1, random2, isDigit, isSeparator]

export const GATES: Record<Cell, string[]> = { rnn: ['h'], lstm: ['f', 'i', 'o', 'g'], gru: ['z', 'r', 'n'] };

export interface Weights {
  cell: Cell;
  /** Every recurrent matrix is `scale * Q` with Q orthogonal, so all its singular values equal `scale`. */
  scale: number;
  Wx: Record<string, Mat>;
  Wh: Record<string, Mat>;
  b: Record<string, number[]>;
  v: number[];
}

// ---------- inputs ----------
const hashStr = (s: string): number => {
  let h = 2166136261;
  for (const ch of s) h = Math.imul(h ^ (ch.codePointAt(0) ?? 0), 16777619);
  return h >>> 0;
};
/** Token (a character or a word) to a fixed seeded input vector. Digits and separators get flag features. */
export function embed(tok: string): number[] {
  const r = createRng(hashStr(tok.toLowerCase()));
  r.next();
  const u = () => r.uniform(-1, 1);
  return [u(), u(), /^[0-9]$/.test(tok) ? 1 : 0, /^[\s.,;!?]$/.test(tok) ? 1 : 0];
}
export const tokens = (s: string, max: number): string[] => Array.from(s.toLowerCase()).slice(0, max);
export const showTok = (c: string): string => (c === ' ' ? '␣' : c);

// ---------- weights ----------
function orth(rng: Rng, n: number): Mat {
  const q: Mat = [];
  while (q.length < n) {
    let v = Array.from({ length: n }, () => rng.normal());
    for (const u of q) {
      const d = u.reduce((a, x, i) => a + x * v[i], 0);
      v = v.map((x, i) => x - d * u[i]);
    }
    const nv = Math.hypot(...v);
    if (nv > 1e-6) q.push(v.map((x) => x / nv));
  }
  return q;
}
const randMat = (rng: Rng, r: number, c: number, sd: number): Mat => Array.from({ length: r }, () => Array.from({ length: c }, () => rng.normal(0, sd)));

/** `forgetBias`: LSTM forget-gate bias (default 1.5, so f is about 0.82 on ordinary characters). */
export function makeWeights(cell: Cell, scale: number, seed = 11, forgetBias = 1.5): Weights {
  const rng = createRng(seed + { rnn: 0, lstm: 100, gru: 200 }[cell]);
  const w: Weights = { cell, scale, Wx: {}, Wh: {}, b: {}, v: [] };
  for (const k of GATES[cell]) {
    w.Wx[k] = randMat(rng, H, D, cell === 'rnn' ? 0.5 : 0.6);
    w.Wh[k] = orth(rng, H).map((row) => row.map((x) => x * scale));
    w.b[k] = Array(H).fill(0);
  }
  w.v = Array.from({ length: H }, () => rng.normal(0, 0.7));
  if (cell !== 'rnn') {
    // Hand-set input wiring so the gates visibly react to digits and separators (real nets learn their own).
    // Unit j has sensitivity a_j in [0.8, 1.2].
    const a = Array.from({ length: H }, () => rng.uniform(0.8, 1.2));
    const wire = (k: string, dig: number, sep: number, bias: number) => {
      w.Wx[k] = w.Wx[k].map((row, j) => [row[0] * 0.5, row[1] * 0.5, dig * a[j], sep * a[j]]);
      w.b[k] = Array(H).fill(bias);
    };
    if (cell === 'lstm') {
      wire('f', 0, -3, forgetBias); // forget closes on a separator: reset
      wire('i', 3, 0, -1.5); // input opens on digits: write
      wire('o', 2.5, 0, -1); // output opens on digits: read
    } else {
      wire('z', 3, 3, -1.5); // update opens on digits and separators
      wire('r', 0, -3, 1.5); // reset closes on a separator
    }
  }
  return w;
}

// ---------- forward + BPTT ----------
type VV = Value[];
const leaf = (m: Mat): VV[] => m.map((r) => r.map((x) => new Value(x)));
const mv = (M: VV[], x: VV): VV =>
  M.map((row) => row.reduce((s, m, i) => (s ? s.add(m.mul(x[i])) : m.mul(x[i])), null as Value | null)!);
const addv = (...vs: VV[]): VV => vs[0].map((_, j) => vs.reduce((s, v, i) => (i ? s.add(v[j]) : v[j]), null as unknown as Value));

export interface Grads { Wx: Record<string, Mat>; Wh: Record<string, Mat>; b: Record<string, number[]>; v: number[] }
export interface Trace {
  cell: Cell; n: number; toks: string[]; xs: number[][];
  h: number[][]; c: number[][];
  /** per-gate activations, n x H (gate name to rows) */
  gates: Record<string, number[][]>;
  /** RNN only: W_xh x_t and W_hh h_{t-1} (n x H each) and their sum z_t before tanh (bias added) */
  wx: number[][]; wh: number[][]; z: number[][];
  o: number[]; y: number[]; lossMask: boolean[]; loss: number;
  dh: number[][]; dc: number[][]; dhN: number[]; dcN: number[];
  grads: Grads;
  /** RNN only: norm of step t's contribution delta_t h_{t-1}^T to dL/dW_hh (they sum to the full gradient) */
  contrib: number[];
  /** RNN only: contribution matrices (n of H x H) */
  contribMat: Mat[];
  /** RNN only: spectral norm of dh_{n-1}/dh_t (product of Jacobians) */
  jac: number[];
  /** geometric-mean change of the gradient norm per step going backward (under 1 shrinks, over 1 explodes); NaN if undefined */
  factor: number;
}

/** Fixed made-up regression target for step t (the lesson is about gradients, not about this task). */
export const target = (_tok: string, t: number): number => 0.8 * Math.cos(0.7 * t);
const norm = (v: number[]) => Math.hypot(...v);
const outer = (a: number[], b: number[]): Mat => a.map((x) => b.map((y) => x * y));

/** Largest singular value of a small matrix by power iteration on M^T M. */
export function specNorm(M: Mat): number {
  const n = M[0].length;
  let v = Array.from({ length: n }, (_, i) => 1 + 0.1 * i);
  let s = 0;
  for (let it = 0; it < 60; it++) {
    const u = M.map((row) => row.reduce((a, x, i) => a + x * v[i], 0));
    const w = Array.from({ length: n }, (_, j) => M.reduce((a, row, i) => a + row[j] * u[i], 0));
    const nw = norm(w);
    if (nw < 1e-300) return 0;
    v = w.map((x) => x / nw);
    s = Math.sqrt(nw);
  }
  return s;
}
const matmul = (A: Mat, B: Mat): Mat => A.map((r) => B[0].map((_, j) => r.reduce((s, a, k) => s + a * B[k][j], 0)));

/** `truncate` k > 0: truncated BPTT, the state is detached every k steps so gradients stop at chunk boundaries. */
export function run(w: Weights, toks: string[], lossAt: LossAt = 'last', truncate = 0): Trace {
  const n = toks.length, cell = w.cell;
  const xs = toks.map(embed);
  const P = {
    Wx: Object.fromEntries(Object.entries(w.Wx).map(([k, m]) => [k, leaf(m)])) as Record<string, VV[]>,
    Wh: Object.fromEntries(Object.entries(w.Wh).map(([k, m]) => [k, leaf(m)])) as Record<string, VV[]>,
    b: Object.fromEntries(Object.entries(w.b).map(([k, m]) => [k, m.map((x) => new Value(x))])) as Record<string, VV>,
    v: w.v.map((x) => new Value(x)),
  };
  const zero = () => Array.from({ length: H }, () => new Value(0));
  let hp = zero(), cp = zero();
  const hs: VV[] = [], cs: VV[] = [], g: Record<string, VV[]> = {}, wxs: VV[] = [], whs: VV[] = [], zs: VV[] = [];
  for (const k of GATES[cell]) g[k] = [];
  for (let t = 0; t < n; t++) {
    if (truncate > 0 && t > 0 && t % truncate === 0) { hp = hp.map((q) => new Value(q.data)); cp = cp.map((q) => new Value(q.data)); }
    const x = xs[t].map((q) => new Value(q));
    let hn: VV, cn: VV = cp;
    const pre = (k: string) => addv(mv(P.Wx[k], x), mv(P.Wh[k], hp), P.b[k]);
    if (cell === 'rnn') {
      const a = mv(P.Wx.h, x), bb = mv(P.Wh.h, hp), z = addv(a, bb, P.b.h);
      wxs.push(a); whs.push(bb); zs.push(z);
      hn = z.map((q) => q.tanh());
    } else if (cell === 'lstm') {
      const f = pre('f').map((q) => q.sigmoid()), i = pre('i').map((q) => q.sigmoid());
      const o = pre('o').map((q) => q.sigmoid()), gg = pre('g').map((q) => q.tanh());
      cn = f.map((q, j) => q.mul(cp[j]).add(i[j].mul(gg[j])));
      hn = o.map((q, j) => q.mul(cn[j].tanh()));
      g.f.push(f); g.i.push(i); g.o.push(o); g.g.push(gg);
    } else {
      const z = pre('z').map((q) => q.sigmoid()), r = pre('r').map((q) => q.sigmoid());
      const un = mv(P.Wh.n, hp), wxn = mv(P.Wx.n, x);
      const nn = wxn.map((q, j) => q.add(P.b.n[j]).add(r[j].mul(un[j]))).map((q) => q.tanh());
      hn = z.map((q, j) => q.neg().add(1).mul(hp[j]).add(q.mul(nn[j])));
      g.z.push(z); g.r.push(r); g.n.push(nn);
    }
    hs.push(hn); cs.push(cn); hp = hn; cp = cn;
  }
  const o = hs.map((h) => h.reduce((s, q, j) => (s ? s.add(q.mul(P.v[j])) : q.mul(P.v[j])), null as Value | null)!);
  const y = toks.map((tk, t) => target(tk, t));
  const lossMask = toks.map((_, t) => lossAt === 'all' || t === n - 1);
  let L: Value | null = null;
  o.forEach((q, t) => {
    if (!lossMask[t]) return;
    const term = q.sub(y[t]).pow(2).mul(0.5);
    L = L ? L.add(term) : term;
  });
  const loss = L as unknown as Value;
  for (const nd of prepareBackward(loss)) nd.localBackward();

  const d = (m: VV[]) => m.map((r) => r.map((q) => q.data));
  const gr = (m: VV[]) => m.map((r) => r.map((q) => q.grad));
  const grads: Grads = {
    Wx: Object.fromEntries(Object.entries(P.Wx).map(([k, m]) => [k, gr(m)])),
    Wh: Object.fromEntries(Object.entries(P.Wh).map(([k, m]) => [k, gr(m)])),
    b: Object.fromEntries(Object.entries(P.b).map(([k, m]) => [k, m.map((q) => q.grad)])),
    v: P.v.map((q) => q.grad),
  };
  const h = d(hs), dh = hs.map((r) => r.map((q) => q.grad));
  const dc = cs.map((r) => r.map((q) => q.grad));
  const tr: Trace = {
    cell, n, toks, xs, h, c: d(cs),
    gates: Object.fromEntries(Object.entries(g).map(([k, m]) => [k, d(m)])),
    wx: d(wxs), wh: d(whs), z: d(zs),
    o: o.map((q) => q.data), y, lossMask, loss: loss.data,
    dh, dc, dhN: dh.map(norm), dcN: dc.map(norm), grads,
    contrib: [], contribMat: [], jac: [], factor: NaN,
  };
  if (cell === 'rnn') {
    // delta_t = dL/dz_t = dL/dh_t * (1 - h_t^2); step t adds delta_t h_{t-1}^T to dL/dW_hh
    for (let t = 0; t < n; t++) {
      const delta = dh[t].map((q, j) => q * (1 - h[t][j] ** 2));
      const m = outer(delta, t ? h[t - 1] : Array(H).fill(0));
      tr.contribMat.push(m); tr.contrib.push(Math.hypot(...m.flat()));
    }
    // dh_t/dh_{t-1} = diag(1 - h_t^2) W_hh; dh_{n-1}/dh_t = J_{n-1} ... J_{t+1}
    const J = (t: number): Mat => w.Wh.h.map((row, i) => row.map((q) => q * (1 - h[t][i] ** 2)));
    let acc: Mat = Array.from({ length: H }, (_, i) => Array.from({ length: H }, (_, j) => (i === j ? 1 : 0)));
    tr.jac = Array(n).fill(1);
    for (let t = n - 2; t >= 0; t--) { acc = matmul(acc, J(t + 1)); tr.jac[t] = specNorm(acc); }
  }
  if (lossAt === 'last' && n > 1) {
    const s = cell === 'lstm' ? tr.dcN : tr.dhN;
    if (s[0] > 0 && s[n - 1] > 0) tr.factor = (s[0] / s[n - 1]) ** (1 / (n - 1));
  }
  return tr;
}

// ---------- timeline ----------
export interface Timing { S: number; Sb: number; unrollAt: number; runAt: number; backAt: number; T: number; folded: boolean; bptt: boolean }
export function timing(n: number, folded: boolean, bptt: boolean): Timing {
  const S = 1.1, Sb = 0.9;
  const unrollAt = folded ? 1.4 : 0, runAt = folded ? 3.3 : 0.4;
  const backAt = runAt + n * S + 0.5;
  return { S, Sb, unrollAt, runAt, backAt, T: bptt ? backAt + n * Sb + 1.4 : runAt + n * S + 1.4, folded, bptt };
}
export type Phase = 'fold' | 'unroll' | 'forward' | 'backward';
export const markersOf = (tm: Timing) => [
  ...(tm.folded ? [{ at: 0, label: 'fold' }, { at: tm.unrollAt, label: 'unroll' }] : []),
  { at: tm.runAt, label: 'forward' },
  ...(tm.bptt ? [{ at: tm.backAt, label: 'backward' }] : []),
];
export const phaseStart = (tm: Timing, p: Phase): number => ({ fold: 0, unroll: tm.unrollAt, forward: tm.runAt, backward: tm.backAt }[p]);

export const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
export const ease = (v: number) => { const x = clamp01(v); return x * x * (3 - 2 * x); };

export interface StepFrame {
  /** 0..1: cell copy present and slid into its column */
  present: number;
  x: number; rec: number; hP: number; outP: number;
  /** backward: how much of this step's gradient has arrived; flow: particle along the edge into step k-1 */
  gradP: number; flow: number;
}
export interface Frame { phase: Phase; unroll: number; active: number; steps: StepFrame[] }

export function frameAt(t: number, n: number, tm: Timing): Frame {
  const un = tm.folded ? ease((t - tm.unrollAt) / (tm.runAt - 0.2 - tm.unrollAt)) : 1;
  const steps: StepFrame[] = [];
  let active = -1;
  for (let k = 0; k < n; k++) {
    const p = (t - (tm.runAt + k * tm.S)) / tm.S;
    const q = (t - (tm.backAt + (n - 1 - k) * tm.Sb)) / tm.Sb;
    if (p >= 0 && p < 1) active = k;
    if (tm.bptt && q >= 0 && q < 1) active = k;
    steps.push({
      present: k === 0 ? 1 : un,
      x: clamp01(p / 0.35), rec: clamp01(p / 0.5), hP: ease((p - 0.4) / 0.5), outP: clamp01((p - 0.8) / 0.2),
      gradP: tm.bptt ? ease(q / 0.5) : 0, flow: tm.bptt ? clamp01((q - 0.3) / 0.7) : 0,
    });
  }
  const phase: Phase = t < tm.unrollAt ? 'fold' : t < tm.runAt ? 'unroll' : tm.bptt && t >= tm.backAt ? 'backward' : 'forward';
  return { phase, unroll: un, active, steps };
}

// ---------- plan ----------
export interface Plan { w: Weights; tr: Trace; tm: Timing; n: number; series: number[] }
export function makePlan(cell: Cell, scale: number, seq: string, lossAt: LossAt, o: { folded: boolean; bptt: boolean; maxLen: number; truncate?: number; forgetBias?: number }): Plan {
  const toks = tokens(seq, o.maxLen);
  const w = makeWeights(cell, scale, 11, o.forgetBias);
  const tr = run(w, toks.length ? toks : ['a'], lossAt, o.truncate ?? 0);
  return { w, tr, tm: timing(tr.n, o.folded, o.bptt), n: tr.n, series: cell === 'lstm' ? tr.dcN : tr.dhN };
}

/** Log-scale bar height: 1e-4 maps to 0 and 1e2 to 1. */
export const gFrac = (g: number): number => clamp01((Math.log10(g + 1e-12) + 4) / 6);
export const fmt = (v: number): string => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
export const fmtG = (g: number): string => (g >= 0.01 ? g.toFixed(2) : g === 0 ? '0' : g.toExponential(0).replace('e-', 'e−'));
