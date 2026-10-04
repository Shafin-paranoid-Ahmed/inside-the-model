/**
 * MLP forward-pass widget logic. No DOM. Networks come from the shared `MLP` (src/lib/nn), so the numbers
 * are the ones the backprop lessons use; the widget only reads them out and lays them over a timeline.
 *
 * Four modes, each a pure `frameAt(t, plan)`:
 *  - 'matmul'      one 2->3 layer as a neuron diagram AND the matrix row  z = W x + b, h = f(z)
 *  - 'forward'     a 2->2->2->1 network, a value pulse flows layer by layer
 *  - 'params'      architecture sliders, weights + biases counted layer by layer
 *  - 'depth-width' three architectures, each with a decision boundary from hardcoded trained weights
 */
import { MLP, createRng, type Activation } from '../../../lib/nn';
import { MLP_INIT } from '../BackpropViz/sim';

export type Mode = 'matmul' | 'depth-width' | 'params' | 'forward';

// ---------- small helpers ----------
export const clamp01 = (u: number) => Math.min(1, Math.max(0, u));
export const ease = (u: number) => { const c = clamp01(u); return c * c * (3 - 2 * c); };
/** Progress of t through [a, b], clamped to 0..1. */
export const seg = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
/** 2 decimals, true minus sign, fixed width friendly. */
export const f2 = (x: number): string => {
  const s = (Math.abs(x) < 0.005 ? 0 : x).toFixed(2);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
};

// ---------- networks ----------
export interface Net { sizes: number[]; act: Activation; W: number[][][]; b: number[][] }
export interface NetParams { W: number[][][]; b: number[][] }

export const actFn = (a: Activation): ((z: number) => number) =>
  a === 'tanh' ? Math.tanh : a === 'relu' ? (z) => (z > 0 ? z : 0) : a === 'sigmoid' ? (z) => 1 / (1 + Math.exp(-z)) : (z) => z;

export const paramCount = (sizes: number[]): number =>
  sizes.slice(1).reduce((s, n, l) => s + sizes[l] * n + n, 0);

export interface LayerCount { nIn: number; nOut: number; w: number; b: number; total: number }
/** Weights n_in * n_out and biases n_out for each weight layer. */
export const breakdown = (sizes: number[]): LayerCount[] =>
  sizes.slice(1).map((n, l) => ({ nIn: sizes[l], nOut: n, w: sizes[l] * n, b: n, total: sizes[l] * n + n }));

/** Shared-engine network (tanh hidden, linear output unless `outAct`). `set` overwrites the seeded init (Value.data). */
export function buildNet(sizes: number[], seed: number, act: Activation = 'tanh', set?: NetParams, outAct: Activation = 'linear'): { net: Net; mlp: MLP } {
  const mlp = new MLP(sizes[0], sizes.slice(1), { rng: createRng(seed), act, outAct });
  if (set) mlp.layers.forEach((L, l) => L.neurons.forEach((n, j) => { n.w.forEach((w, i) => { w.data = set.W[l][j][i]; }); n.b.data = set.b[l][j]; }));
  return { net: { sizes, act, W: mlp.layers.map((L) => L.neurons.map((n) => n.w.map((w) => w.data))), b: mlp.layers.map((L) => L.neurons.map((n) => n.b.data)) }, mlp };
}

/** Fast plain-number forward pass (hidden = net.act, last layer linear). Used for the boundary grids. */
export function forward(net: Net, x: number[]): number[] {
  const f = actFn(net.act);
  let h = x;
  for (let l = 0; l < net.W.length; l++) {
    const last = l === net.W.length - 1;
    h = net.W[l].map((row, j) => { let z = net.b[l][j]; for (let i = 0; i < row.length; i++) z += row[i] * h[i]; return last ? z : f(z); });
  }
  return h;
}

export interface Trace { z: number[][]; h: number[][] }
/**
 * Every pre-activation z and activation h, layer by layer (h[0] = x; z[l-1], h[l] belong to layer l).
 * h comes from the engine's `Layer.call`, z from the same weights, so they agree by construction.
 */
export function trace(mlp: MLP, x: number[]): Trace {
  const z: number[][] = [], h: number[][] = [x];
  mlp.layers.forEach((L, l) => {
    const prev = h[l];
    z.push(L.neurons.map((n) => n.w.reduce((s, w, i) => s + w.data * prev[i], n.b.data)));
    h.push(L.call(prev).map((v) => v.data));
  });
  return { z, h };
}

// ---------- layout (pure geometry, in logical canvas units) ----------
export interface Pt { x: number; y: number }
export interface Rect { x: number; y: number; w: number; h: number }
export const WIDE = { w: 720, h: 400 };
export const NARROW = { w: 360, h: 620 };
export const size = (narrow: boolean) => (narrow ? NARROW : WIDE);

/** Node centres for each layer: layers spread across the rect (or down it when `vertical`). */
export function columns(sizes: number[], r: Rect, vertical = false, maxGap = 34): Pt[][] {
  const L = sizes.length;
  return sizes.map((n, l) => {
    const along = L > 1 ? l / (L - 1) : 0.5;
    const span = vertical ? r.w : r.h;
    const gap = n > 1 ? Math.min(maxGap, (span - 24) / (n - 1)) : 0;
    return Array.from({ length: n }, (_, j) => {
      const o = (j - (n - 1) / 2) * gap;
      return vertical ? { x: r.x + r.w / 2 + o, y: r.y + along * r.h } : { x: r.x + along * r.w, y: r.y + r.h / 2 + o };
    });
  });
}

// ---------- per-mode geometry (logical units; wide = 720x400, narrow = 360x620) ----------
export interface MMLayout {
  inP: Pt[]; hidP: Pt[]; rowY: number[]; xY: number[]; headY: number; titleY: number;
  X: Record<'W1' | 'W2' | 'mul' | 'x' | 'plus' | 'b' | 'eq' | 'z' | 'arr' | 'h', number>;
}
export const mmLayout = (narrow: boolean): MMLayout => narrow
  ? {
    inP: [{ x: 56, y: 112 }, { x: 56, y: 198 }], hidP: [{ x: 280, y: 70 }, { x: 280, y: 155 }, { x: 280, y: 240 }],
    rowY: [372, 438, 504], xY: [405, 471], headY: 324, titleY: 292,
    X: { W1: 34, W2: 78, mul: 106, x: 130, plus: 154, b: 182, eq: 210, z: 240, arr: 273, h: 318 },
  }
  : {
    inP: [{ x: 44, y: 142 }, { x: 44, y: 258 }], hidP: [{ x: 210, y: 88 }, { x: 210, y: 200 }, { x: 210, y: 312 }],
    rowY: [88, 200, 312], xY: [144, 256], headY: 52, titleY: 20,
    X: { W1: 340, W2: 392, mul: 428, x: 462, plus: 496, b: 530, eq: 564, z: 598, arr: 640, h: 684 },
  };

export interface FWLayout { pos: Pt[][]; hw: number[]; hh: number; vertical: boolean }
export const fwLayout = (narrow: boolean): FWLayout => {
  const xs = narrow ? [105, 255] : [135, 265], one = narrow ? 180 : 200;
  const col = (l: number, n: number) => (narrow
    ? (n === 1 ? [{ x: one, y: 62 + l * 148 }] : xs.map((x) => ({ x, y: 62 + l * 148 })))
    : (n === 1 ? [{ x: 75 + l * 190, y: one }] : xs.map((y) => ({ x: 75 + l * 190, y }))));
  return { pos: [col(0, 2), col(1, 2), col(2, 2), col(3, 1)], hw: [40, 49, 49, 49], hh: 26, vertical: narrow };
};

export interface PALayout { net: Rect; list: { x: number; y: number; cLabel: number; cW: number; cB: number; cS: number; rowH: number } }
export const paLayout = (narrow: boolean): PALayout => narrow
  ? { net: { x: 30, y: 36, w: 300, h: 220 }, list: { x: 18, y: 334, cLabel: 18, cW: 170, cB: 252, cS: 342, rowH: 40 } }
  : { net: { x: 34, y: 40, w: 360, h: 320 }, list: { x: 440, y: 70, cLabel: 440, cW: 566, cB: 630, cS: 704, rowH: 48 } };

export interface DWLayout { net: Rect; thumbs: Rect[]; capX: number[]; titleY: number }
export const dwLayout = (narrow: boolean): DWLayout => narrow
  ? { net: { x: 30, y: 62, w: 300, h: 150 }, thumbs: [0, 1, 2].map((i) => ({ x: 20, y: 262 + i * 118, w: 128, h: 96 })), capX: [166], titleY: 28 }
  : { net: { x: 34, y: 70, w: 320, h: 270 }, thumbs: [0, 1, 2].map((i) => ({ x: 410, y: 26 + i * 124, w: 128, h: 96 })), capX: [556], titleY: 30 };

// ---------- matmul mode: one layer, two views ----------
export const MM_SEG = 3;
export const MM_T = MM_SEG * 3 + 3;
export const MM_X = [1, 0.5];
const MM_B = [0.1, -0.2, 0.2];
export interface MatmulPlan {
  mode: 'matmul'; T: number; x: number[]; W: number[][]; b: number[]; z: number[]; h: number[]; terms: number[][]; mlp: MLP;
}
export function matmulPlan(): MatmulPlan {
  // first-layer weights of the backprop lessons' 2-3-1 network (MLP_INIT, row j = hidden unit j)
  const W = [0, 1, 2].map((j) => [MLP_INIT[j * 2], MLP_INIT[j * 2 + 1]]);
  const { mlp } = buildNet([2, 3], 1, 'tanh', { W: [W], b: [MM_B] }, 'tanh');
  const t = trace(mlp, MM_X);
  return { mode: 'matmul', T: MM_T, x: MM_X, W, b: MM_B, z: t.z[0], h: t.h[1], terms: W.map((r) => r.map((w, i) => w * MM_X[i])), mlp };
}
export const MM_MARKERS = [
  { at: 0, label: 'Neuron 1' }, { at: MM_SEG, label: 'Neuron 2' }, { at: MM_SEG * 2, label: 'Neuron 3' }, { at: MM_SEG * 3, label: 'Whole layer' },
];
export type MMSub = 'dot' | 'bias' | 'act' | 'all';
export interface MatmulFrame {
  active: number; all: boolean; sub: MMSub;
  rows: { u: number; zShown: number; zA: number; bA: number; hShown: number; hA: number; tA: [number, number]; pulse: [number, number] }[];
}
export function matmulFrame(t: number, p: MatmulPlan): MatmulFrame {
  const all = t >= MM_SEG * 3;
  const active = all ? -1 : Math.min(2, Math.floor(t / MM_SEG));
  const rows = p.z.map((_, j) => {
    const u = all ? 1 : seg(t, j * MM_SEG, (j + 1) * MM_SEG);
    const s0 = ease(seg(u, 0.05, 0.3)), s1 = ease(seg(u, 0.3, 0.55)), sb = ease(seg(u, 0.55, 0.72)), sh = ease(seg(u, 0.74, 0.95));
    const [A, B] = p.terms[j];
    return {
      u,
      zShown: A * s0 + B * s1 + p.b[j] * sb,
      zA: u > 0.05 ? 1 : 0,
      bA: sb > 0 ? 1 : 0,
      hShown: p.h[j] * sh,
      hA: sh > 0 ? 1 : 0,
      tA: [s0, s1] as [number, number],
      // a value pulse on each input->neuron edge while that term is being multiplied
      pulse: [seg(u, 0.05, 0.3), seg(u, 0.3, 0.55)] as [number, number],
    };
  });
  const u = active >= 0 ? rows[active].u : 1;
  return { active, all, sub: all ? 'all' : u < 0.55 ? 'dot' : u < 0.74 ? 'bias' : 'act', rows };
}

// ---------- forward mode: 2-2-2-1 with real numbers ----------
export const FW_LEAD = 2, FW_LAYER = 3, FW_T = FW_LEAD + FW_LAYER * 3 + 1.8;
export const FW_X = [1, 0.5];
const FW_SET: NetParams = {
  W: [[[0.5, -0.6], [-0.7, 0.9]], [[0.8, -0.5], [0.4, 0.6]], [[1, -0.7]]],
  b: [[0.1, -0.2], [0.2, 0.1], [0.05]],
};
export interface ForwardPlan { mode: 'forward'; T: number; net: Net; tr: Trace; maxW: number; mlp: MLP }
export function forwardPlan(): ForwardPlan {
  const { net, mlp } = buildNet([2, 2, 2, 1], 1, 'tanh', FW_SET);
  return { mode: 'forward', T: FW_T, net, tr: trace(mlp, FW_X), mlp, maxW: Math.max(...net.W.flat(2).map(Math.abs)) };
}
export const FW_MARKERS = [
  { at: 0, label: 'Input' }, { at: FW_LEAD, label: 'Layer 1' }, { at: FW_LEAD + FW_LAYER, label: 'Layer 2' },
  { at: FW_LEAD + FW_LAYER * 2, label: 'Output' }, { at: FW_LEAD + FW_LAYER * 3, label: 'Result' },
];
export interface ForwardFrame {
  /** 0 = inputs only, 1..3 = that weight layer is running, 4 = finished. */
  layer: number;
  /** Per weight layer l (index l-1), per edge [j][i]: pulse position 0..1 along i->j, -1 when not travelling. */
  pulse: number[][][];
  labelA: number[];
  /** Per layer 0..3, per node: z (null for inputs) and h as shown, with 0..1 alphas. */
  nodes: { z: number; h: number; zA: number; hA: number }[][];
}
export function forwardFrame(t: number, p: ForwardPlan): ForwardFrame {
  const L = p.net.W.length;
  const layer = t < FW_LEAD ? 0 : t >= FW_LEAD + FW_LAYER * L ? L + 1 : 1 + Math.floor((t - FW_LEAD) / FW_LAYER);
  const prog = (l: number) => seg(t, FW_LEAD + (l - 1) * FW_LAYER, FW_LEAD + l * FW_LAYER);
  const nodes = [p.tr.h[0].map((h) => ({ z: 0, h, zA: 0, hA: 1 }))];
  const pulse: number[][][] = [], labelA: number[] = [];
  for (let l = 1; l <= L; l++) {
    const u = prog(l);
    const pf = ease(seg(u, 0.02, 0.5));
    pulse.push(p.net.W[l - 1].map((row) => row.map(() => (u > 0 && pf < 1 ? pf : -1))));
    labelA.push(layer === l ? Math.min(1, seg(u, 0, 0.1) * 5) * (1 - seg(u, 0.94, 1)) : 0);
    const zs = ease(seg(u, 0.45, 0.68)), hs = ease(seg(u, 0.66, 0.88));
    nodes.push(p.tr.z[l - 1].map((z, j) => ({ z: z * zs, h: p.tr.h[l][j] * hs, zA: zs > 0 ? 1 : 0, hA: hs > 0 ? 1 : 0 })));
  }
  return { layer, pulse, labelA, nodes };
}

// ---------- params mode ----------
export const PA_SEG = 2.2, PA_TAIL = 2;
export interface ParamsPlan { mode: 'params'; T: number; sizes: number[]; layers: LayerCount[]; total: number }
export function paramsPlan(hidden: number, width: number): ParamsPlan {
  const sizes = [2, ...Array<number>(hidden).fill(width), 1];
  const layers = breakdown(sizes);
  return { mode: 'params', T: PA_SEG * layers.length + PA_TAIL, sizes, layers, total: paramCount(sizes) };
}
export const paramsMarkers = (p: ParamsPlan) => [
  ...p.layers.map((_, l) => ({ at: l * PA_SEG, label: `L${l + 1}` })),
  { at: p.layers.length * PA_SEG, label: 'Total' },
];
export interface ParamsFrame {
  /** Weight layer being counted (-1 = all done). */
  active: number;
  /** Per weight layer: share of weights / biases lit (0..1) and the counts shown so far. */
  layers: { wP: number; bP: number; w: number; b: number }[];
  total: number;
}
export function paramsFrame(t: number, p: ParamsPlan): ParamsFrame {
  const L = p.layers.length;
  const active = t >= L * PA_SEG ? -1 : Math.floor(t / PA_SEG);
  const layers = p.layers.map((c, l) => {
    const u = seg(t, l * PA_SEG, (l + 1) * PA_SEG);
    const wP = ease(seg(u, 0.04, 0.55)), bP = ease(seg(u, 0.58, 0.86));
    return { wP, bP, w: Math.round(c.w * wP), b: Math.round(c.b * bP) };
  });
  return { active, layers, total: layers.reduce((s, l) => s + l.w + l.b, 0) };
}

// ---------- depth-width mode ----------
export const DW_SEG = 4;
export const DW_ARCHS: { sizes: number[]; name: string }[] = [
  { sizes: [2, 8, 1], name: 'One wide layer' },
  { sizes: [2, 4, 4, 1], name: 'Two layers of 4' },
  { sizes: [2, 3, 3, 3, 1], name: 'Three layers of 3' },
];
export const DW_T = DW_SEG * DW_ARCHS.length;
export const DW_MARKERS = DW_ARCHS.map((_, i) => ({ at: i * DW_SEG, label: ['Wide', '2 layers', '3 layers'][i] }));
export const GX = 48, GY = 36, DOMX = 1.8, DOMY = 1.35;
export const archKey = (sizes: number[]) => sizes.join('-');

/** Two interleaved moons, centred, 64 points, seeded. Class 1 = upper moon. */
export function makeMoons(n = 64, noise = 0.2, seed = 7): { x: number[][]; y: number[] } {
  const r = createRng(seed), x: number[][] = [], y: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2, a = Math.PI * r.next();
    const px = c ? Math.cos(a) : 1 - Math.cos(a), py = c ? Math.sin(a) : 0.5 - Math.sin(a);
    x.push([px - 0.5 + r.normal(0, noise), py - 0.25 + r.normal(0, noise)]);
    y.push(c);
  }
  return { x, y };
}
export const DATA = makeMoons();

/**
 * Trained weights (Adam, 1500 full-batch epochs, seed 3, logistic output) produced by `trainNet` in train.ts
 * and verified by tests/unit/mlp-forward.test.ts (which retrains and compares). Hardcoded so mounting costs nothing.
 */
export const TRAINED: Record<string, NetParams> = {
  '2-8-1': {W: [[[4.9226,-0.9959],[-4.1338,-5.9884],[5.2814,-2.3983],[-3.9415,-5.1383],[-7.0862,3.5333],[-3.6604,-3.5025],[5.9587,-0.3365],[2.8062,0.1163]],[[-2.7606,-3.4737,-4.439,-3.0409,4.1729,-3.1903,-5.774,-1.9536]]],b: [[-3.9545,1.039,4.5418,0.7968,-6.0465,0.3035,-5.1203,2.3491],[1.0056]]},
  '2-4-4-1': {W: [[[2.0582,-1.3407],[-3.3688,-2.9977],[2.6746,-0.8032],[2.2301,-0.0805]],[[-1.0664,-2.6006,-2.3765,-1.9394],[1.1578,2.7559,2.4053,2.155],[-1.2045,-1.9536,-1.8035,-1.1182],[-0.8426,-1.937,-1.262,-1.9284]],[[2.644,-2.4353,4.8334,3.6076]]],b: [[2.0151,0.2533,2.2539,-2.0391],[0.6725,-0.5882,0.9905,-0.1029],[0.1736]]},
  '2-3-3-3-1': {W: [[[3.4174,2.6761],[0.0363,-1.915],[0.9273,-0.1533]],[[0.4218,-0.88,-1.36],[2.2399,-0.0149,-2.6074],[0.6956,-1.6993,5.4724]],[[1.5772,1.7939,-1.0715],[-1.4351,-3.0156,1.8829],[1.8227,3.7488,-2.3443]],[[4.0181,-4.9028,3.4095]]],b: [[0.0007,-1.8841,0.0832],[0.2839,-0.2347,1.799],[-0.3153,-0.4154,0.492],[-0.1109]]},
};

export const probability = (net: Net, x: number, y: number) => 1 / (1 + Math.exp(-forward(net, [x, y])[0]));
export const accuracy = (net: Net, d = DATA) => d.x.filter((p, i) => (forward(net, p)[0] > 0 ? 1 : 0) === d.y[i]).length / d.y.length;

export interface DWArch { sizes: number[]; name: string; net: Net; params: number; acc: number; grid: Float32Array }
export interface DWPlan { mode: 'depth-width'; T: number; archs: DWArch[] }
let dwCache: DWPlan | undefined;
/** Three trained nets and their probability grids (rows top to bottom). ~5k forward passes of 37 weights: a few ms. */
export function dwPlan(): DWPlan {
  return (dwCache ??= {
    mode: 'depth-width', T: DW_T,
    archs: DW_ARCHS.map((a) => {
      const { net } = buildNet(a.sizes, 3, 'tanh', TRAINED[archKey(a.sizes)]);
      const grid = new Float32Array(GX * GY);
      for (let r = 0; r < GY; r++) for (let c = 0; c < GX; c++) {
        grid[r * GX + c] = probability(net, ((c + 0.5) / GX * 2 - 1) * DOMX, (1 - (r + 0.5) / GY * 2) * DOMY);
      }
      return { ...a, net, params: paramCount(a.sizes), acc: accuracy(net), grid };
    }),
  });
}
export interface DWFrame {
  idx: number; u: number;
  /** Fade-in of the active architecture's diagram (0..1). */
  fade: number;
  /** Exact parameter count for the architecture currently drawn. */
  paramsShown: number;
  /** Thumbnail opacities. */
  thumbA: number[];
}
export function dwFrame(t: number, p: DWPlan): DWFrame {
  const idx = Math.min(p.archs.length - 1, Math.floor(t / DW_SEG)), u = seg(t, idx * DW_SEG, (idx + 1) * DW_SEG);
  const fade = ease(seg(u, 0, 0.22));
  return {
    idx, u, fade,
    paramsShown: p.archs[idx].params,
    thumbA: p.archs.map((_, i) => (i < idx ? 1 : i === idx ? 0.2 + 0.8 * ease(seg(u, 0.1, 0.5)) : 0.22)),
  };
}

// ---------- the one entry point ----------
export type Plan = MatmulPlan | ForwardPlan | ParamsPlan | DWPlan;
export type Frame = MatmulFrame | ForwardFrame | ParamsFrame | DWFrame;
export function frameAt(t: number, p: Plan): Frame {
  switch (p.mode) {
    case 'matmul': return matmulFrame(t, p);
    case 'forward': return forwardFrame(t, p);
    case 'params': return paramsFrame(t, p);
    default: return dwFrame(t, p);
  }
}
export const markersOf = (p: Plan) =>
  p.mode === 'matmul' ? MM_MARKERS : p.mode === 'forward' ? FW_MARKERS : p.mode === 'params' ? paramsMarkers(p) : DW_MARKERS;

/** Where reduced-motion users start: a finished, fully labelled state. */
export const restAt = (p: Plan): number =>
  p.mode === 'matmul' ? MM_SEG * 3 + 0.5 : p.mode === 'forward' ? p.T - 0.5 : p.mode === 'params' ? p.T - 0.5 : DW_SEG * 0.6;
