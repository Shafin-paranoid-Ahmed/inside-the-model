/** Pure GNN math: GCN / GAT / sum / mean / max message passing on a small editable graph. No DOM. */
import { createRng } from '../../../lib/math/rng';

export type Agg = 'gcn' | 'sum' | 'mean' | 'max';
export type Model = 'gcn' | 'gat';
export const F = 3; // feature dimension shown as three bars

export interface Graph {
  x: number[]; y: number[];        // node positions in [0,1]²
  h: number[][];                   // input features, n × F
  edges: [number, number][];       // undirected, stored once
}
export interface Cfg { model: Model; agg: Agg; layers: number; seed: number }
export interface Msg { from: number; to: number; coef: number; vec: number[] }
export interface LayerTrace { msgs: Msg[]; pre: number[][]; H: number[][] }
export interface Trace { W: number[][][]; aL: number[][]; aR: number[][]; H: number[][][]; layers: LayerTrace[] }

export const letter = (i: number) => String.fromCharCode(65 + i);

export function nodeFeatures(seed: number, i: number): number[] {
  const r = createRng(seed * 7919 + i * 104729 + 11);
  return Array.from({ length: F }, () => Math.round(r.uniform(-1, 1) * 100) / 100);
}
export function defaultGraph(seed = 3): Graph {
  const pos = [[0.14, 0.22], [0.4, 0.14], [0.36, 0.58], [0.14, 0.8], [0.66, 0.38], [0.88, 0.16], [0.78, 0.82]];
  return {
    x: pos.map((p) => p[0]), y: pos.map((p) => p[1]),
    h: pos.map((_, i) => nodeFeatures(seed, i)),
    edges: [[0, 1], [0, 2], [1, 2], [2, 3], [1, 4], [4, 5], [4, 6], [5, 6]],
  };
}

export const neighbors = (g: Graph): number[][] => {
  const nb: number[][] = g.x.map(() => []);
  for (const [a, b] of g.edges) { nb[a].push(b); nb[b].push(a); }
  nb.forEach((l) => l.sort((p, q) => p - q));
  return nb;
};

/** Â = D̃^{-1/2}(A+I)D̃^{-1/2}, dense n × n (Kipf & Welling 2017). */
export function gcnNorm(g: Graph): number[][] {
  const n = g.x.length, nb = neighbors(g), d = nb.map((l) => l.length + 1);
  const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (const [a, b] of g.edges) { A[a][b] = 1; A[b][a] = 1; }
  return A.map((row, i) => row.map((v, j) => v / Math.sqrt(d[i] * d[j])));
}

/** Numerically stable softmax. */
export function softmax(z: number[]): number[] {
  const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
export const leaky = (v: number, s = 0.2) => (v > 0 ? v : s * v);

export function makeWeights(seed: number, layers: number) {
  // one RNG stream per layer, so layer l's weights do not change when the number of layers does
  const per = (l: number) => {
    const r = createRng(seed * 31 + 5 + l * 977);
    const mat = () => Array.from({ length: F }, () => Array.from({ length: F }, () => r.normal(0, 0.85)));
    const vec = () => Array.from({ length: F }, () => r.normal(0, 1.6));
    return { W: mat(), aL: vec(), aR: vec() };
  };
  const ls = Array.from({ length: layers }, (_, l) => per(l));
  return { W: ls.map((x) => x.W), aL: ls.map((x) => x.aL), aR: ls.map((x) => x.aR) };
}
const mulRow = (h: number[], W: number[][]) => W[0].map((_, c) => h.reduce((s, v, r) => s + v * W[r][c], 0));
const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);

/** One layer's messages for every directed edge j→i (self loop first). */
export function layerMessages(g: Graph, H: number[][], W: number[][], cfg: Cfg, aL: number[], aR: number[]): Msg[] {
  const nb = neighbors(g), Z = H.map((h) => mulRow(h, W)), out: Msg[] = [];
  for (let i = 0; i < H.length; i++) {
    const inc = [i, ...nb[i]];
    let coefs: number[];
    if (cfg.model === 'gat') coefs = softmax(inc.map((j) => leaky(dot(aL, Z[i]) + dot(aR, Z[j]))));
    else if (cfg.agg === 'gcn') coefs = inc.map((j) => 1 / Math.sqrt((nb[i].length + 1) * (nb[j].length + 1)));
    else if (cfg.agg === 'mean') coefs = inc.map(() => 1 / inc.length);
    else coefs = inc.map(() => 1);
    inc.forEach((j, k) => out.push({ from: j, to: i, coef: coefs[k], vec: Z[j].map((v) => v * coefs[k]) }));
  }
  return out;
}

export function trace(g: Graph, cfg: Cfg): Trace {
  const w = makeWeights(cfg.seed, cfg.layers), n = g.x.length;
  const H: number[][][] = [g.h.map((r) => [...r])], layers: LayerTrace[] = [];
  const useMax = cfg.model === 'gcn' && cfg.agg === 'max';
  for (let l = 0; l < cfg.layers; l++) {
    const msgs = layerMessages(g, H[l], w.W[l], cfg, w.aL[l], w.aR[l]);
    const pre = Array.from({ length: n }, () => new Array(F).fill(useMax ? -Infinity : 0));
    for (const m of msgs) for (let c = 0; c < F; c++) pre[m.to][c] = useMax ? Math.max(pre[m.to][c], m.vec[c]) : pre[m.to][c] + m.vec[c];
    layers.push({ msgs, pre, H: pre.map((r) => r.map(Math.tanh)) });
    H.push(layers[l].H);
  }
  return { ...w, H, layers };
}

/** Hop distance from `src` to every node (Infinity if unreachable). */
export function hops(g: Graph, src: number): number[] {
  const nb = neighbors(g), d = g.x.map(() => Infinity);
  d[src] = 0;
  for (const q = [src]; q.length; ) { const u = q.shift()!; for (const v of nb[u]) if (d[v] === Infinity) { d[v] = d[u] + 1; q.push(v); } }
  return d;
}

/** Mean per-feature standard deviation across nodes (shrinks as features over-smooth). */
export function spread(H: number[][]): number {
  const n = H.length; let s = 0;
  for (let c = 0; c < F; c++) {
    const mu = H.reduce((a, r) => a + r[c], 0) / n;
    s += Math.sqrt(H.reduce((a, r) => a + (r[c] - mu) ** 2, 0) / n);
  }
  return s / F;
}
export const readoutVec = (H: number[][]): number[] => Array.from({ length: F }, (_, c) => H.reduce((a, r) => a + r[c], 0) / H.length);

// ---- timeline ----
export const TL = { send: 1.7, agg: 0.6, upd: 1.3, readout: 2.4, hold: 0.9 };
export const LAYER_T = TL.send + TL.agg + TL.upd;
export const duration = (layers: number, readout: boolean) => layers * LAYER_T + (readout ? TL.readout : 0) + TL.hold;
export type Phase = 'send' | 'aggregate' | 'update' | 'readout' | 'hold';
export const ease = (p: number) => p * p * (3 - 2 * p);
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

export interface Frame {
  layer: number; phase: Phase; p: number;
  feat: number[][];       // displayed features, n × F
  k: number;              // completed layers, for the receptive field
  sendP: number;          // 0..1 message travel (only in send phase)
  aggP: number;           // 0..1 aggregate pulse
  readP: number; pooled: number[];
}
export function frameAt(t: number, tr: Trace, layers: number, readout: boolean): Frame {
  const total = duration(layers, readout);
  t = Math.min(Math.max(t, 0), total - 1e-6);
  const lt = layers * LAYER_T;
  if (t >= lt) {
    const u = t - lt, H = tr.H[layers];
    const rp = readout ? clamp01(u / TL.readout) : 0;
    return { layer: layers - 1, phase: readout && u < TL.readout ? 'readout' : 'hold', p: rp, feat: H, k: layers, sendP: 0, aggP: 0, readP: rp, pooled: readoutVec(H) };
  }
  const l = Math.floor(t / LAYER_T), u = t - l * LAYER_T, Hin = tr.H[l], Hout = tr.H[l + 1];
  let phase: Phase, p: number, feat = Hin, sendP = 0, aggP = 0;
  if (u < TL.send) { phase = 'send'; p = u / TL.send; sendP = ease(p); }
  else if (u < TL.send + TL.agg) { phase = 'aggregate'; p = (u - TL.send) / TL.agg; aggP = p; }
  else {
    phase = 'update'; p = (u - TL.send - TL.agg) / TL.upd; const e = ease(p);
    feat = Hin.map((r, i) => r.map((v, c) => v + (Hout[i][c] - v) * e));
  }
  return { layer: l, phase, p, feat, k: l + (phase === 'update' && p > 0.5 ? 1 : 0), sendP, aggP, readP: 0, pooled: readoutVec(Hin) };
}
export const markers = (layers: number, readout: boolean) => [
  ...Array.from({ length: layers }, (_, l) => ({ at: l * LAYER_T, label: `layer ${l + 1}` })),
  ...(readout ? [{ at: layers * LAYER_T, label: 'readout' }] : []),
];
