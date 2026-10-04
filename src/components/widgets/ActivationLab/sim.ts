/** Activation functions, their derivatives, and the "stacked linear layers collapse" demo. Pure logic, no DOM. */
import { smooth } from '../F1Shared/plot';

export type ActId = 'step' | 'sigmoid' | 'tanh' | 'relu' | 'leaky' | 'gelu' | 'silu';
export const LEAK = 0.1;

export const step = (z: number) => (z > 0 ? 1 : 0);
export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
export const relu = (z: number) => (z > 0 ? z : 0);
export const leaky = (z: number, a = LEAK) => (z > 0 ? z : a * z);
const K = Math.sqrt(2 / Math.PI);
/** GELU, tanh approximation: 0.5 z (1 + tanh(sqrt(2/pi) (z + 0.044715 z^3))). */
export const gelu = (z: number) => 0.5 * z * (1 + Math.tanh(K * (z + 0.044715 * z ** 3)));
/** SiLU / Swish: z times sigmoid(z). */
export const silu = (z: number) => z * sigmoid(z);

/** Derivatives. Step is undefined at its jump; 0 there is only a display convention. ReLU uses slope 0 at z = 0. */
export const dStep = () => 0;
export const dSigmoid = (z: number) => { const s = sigmoid(z); return s * (1 - s); };
export const dTanh = (z: number) => 1 - Math.tanh(z) ** 2;
export const dRelu = (z: number) => (z > 0 ? 1 : 0);
export const dLeaky = (z: number, a = LEAK) => (z > 0 ? 1 : a);
export const dGelu = (z: number) => {
  const u = K * (z + 0.044715 * z ** 3), th = Math.tanh(u);
  return 0.5 * (1 + th) + 0.5 * z * (1 - th * th) * K * (1 + 3 * 0.044715 * z * z);
};
export const dSilu = (z: number) => { const s = sigmoid(z); return s * (1 + z * (1 - s)); };

export interface Act { name: string; f: (z: number, a?: number) => number; d: (z: number, a?: number) => number }
export const ACT: Record<ActId, Act> = {
  step: { name: 'Step', f: step, d: dStep },
  sigmoid: { name: 'Sigmoid', f: sigmoid, d: dSigmoid },
  tanh: { name: 'Tanh', f: Math.tanh, d: dTanh },
  relu: { name: 'ReLU', f: relu, d: dRelu },
  leaky: { name: 'Leaky ReLU', f: leaky, d: dLeaky },
  gelu: { name: 'GELU', f: gelu, d: dGelu },
  silu: { name: 'SiLU', f: silu, d: dSilu },
};

/** "Saturated / dead": the slope magnitude is below this. */
export const SAT = 0.05;

/** Intervals of z in [-zr, zr] where |f'| < SAT, ignoring slivers narrower than 0.25 (GELU's slope crosses 0 at its dip). */
export function flatZones(id: ActId, zr: number, a = LEAK): [number, number][] {
  const out: [number, number][] = [], n = Math.round(zr * 2 * 100), dz = (2 * zr) / n;
  let start: number | null = null;
  for (let i = 0; i <= n; i++) {
    const z = -zr + i * dz, flat = Math.abs(ACT[id].d(z, a)) < SAT;
    if (flat && start === null) start = z;
    if (start !== null && (!flat || i === n)) {
      const end = flat ? z : z - dz;
      if (end - start >= 0.25) out.push([start, end]);
      start = null;
    }
  }
  return out;
}

/** Minimum of GELU (the "dip") by golden-section search on [-3, 0]. */
export function geluDip(): { z: number; v: number } {
  let a = -3, b = 0;
  const g = (Math.sqrt(5) - 1) / 2;
  for (let i = 0; i < 80; i++) {
    const c = b - g * (b - a), d = a + g * (b - a);
    if (gelu(c) < gelu(d)) b = d; else a = c;
  }
  const z = (a + b) / 2;
  return { z, v: gelu(z) };
}

// ------------------------------------------------------------------ the plot modes

/** The largest slope on [-zr, zr] (0.25 for sigmoid, 1 for tanh and ReLU, about 1.13 for GELU). */
export function peakSlope(id: ActId, zr = 6, a = LEAK): number {
  let m = 0;
  for (let i = 0; i <= 1200; i++) m = Math.max(m, ACT[id].d(-zr + (i / 1200) * 2 * zr, a));
  return m;
}

export type PlotMode = 'saturation' | 'relu-family' | 'gelu' | 'derivatives';
export type Mode = PlotMode | 'collapse';
export interface PanelSpec {
  main: ActId;
  /** Other curves drawn in the same plot (GELU mode: ReLU as a reference, SiLU as a smooth sibling). */
  refs?: ActId[];
  /** Draw f' under the curve. */
  deriv?: boolean;
  yDom: [number, number];
  /** y domain of the derivative plot. */
  dDom?: [number, number];
}
export const PANELS: Record<PlotMode, PanelSpec[]> = {
  saturation: [
    { main: 'step', yDom: [-0.25, 1.25] },
    { main: 'sigmoid', yDom: [-0.25, 1.25] },
    { main: 'tanh', yDom: [-1.4, 1.4] },
  ],
  'relu-family': [
    { main: 'relu', yDom: [-2.1, 4.5] },
    { main: 'leaky', yDom: [-2.1, 4.5] },
  ],
  gelu: [{ main: 'gelu', refs: ['relu', 'silu'], yDom: [-0.6, 4.5] }],
  derivatives: [
    { main: 'sigmoid', deriv: true, yDom: [-0.15, 1.15], dDom: [-0.05, 1.1] },
    { main: 'tanh', deriv: true, yDom: [-1.3, 1.3], dDom: [-0.05, 1.1] },
    { main: 'relu', deriv: true, yDom: [-1, 6.5], dDom: [-0.05, 1.1] },
    { main: 'gelu', deriv: true, yDom: [-1, 6.5], dDom: [-0.3, 1.3] },
  ],
};
/** Half-width of the z range the probe sweeps. */
export const ZR: Record<PlotMode, number> = { saturation: 6, 'relu-family': 4, gelu: 4, derivatives: 6 };
export const HOLD = 1;

export interface Marker { at: number; label: string }
/** One narration line per marker, true for the whole segment. */
export const SEGMENTS: Record<PlotMode, { at: number; label: string; say: string }[]> = {
  saturation: [
    { at: 0, label: 'Flat left', say: 'Far to the left (z below −3) all three outputs have stopped moving: the slope is close to zero.' },
    { at: 3, label: 'Active middle', say: 'Near zero the sigmoid and tanh outputs change fast, and the step jumps at z = 0. Tanh levels off a little sooner than sigmoid.' },
    { at: 9, label: 'Flat right', say: 'Far to the right (z above 3) the curves are flat again: pushing z further changes almost nothing.' },
  ],
  'relu-family': [
    { at: 0, label: 'Dead half', say: 'For z below 0 ReLU outputs exactly 0 and its slope is exactly 0. Leaky ReLU keeps a small slope here.' },
    { at: 4, label: 'Live half', say: 'At zero the curves meet; for positive z both functions pass the input straight through, with slope 1.' },
  ],
  gelu: [
    { at: 0, label: 'Left tail', say: 'For very negative z, GELU like ReLU outputs almost nothing.' },
    { at: 2.5, label: 'The dip', say: 'Just left of zero GELU dips slightly below zero, then climbs back, where ReLU stays flat at 0.' },
    { at: 4, label: 'Right side', say: 'At zero the curves meet. On the positive side GELU approaches the ReLU line, and the two are nearly the same by z = 3.' },
  ],
  derivatives: [
    { at: 0, label: 'Flat left', say: 'On the far left sigmoid and tanh have slope near 0, and ReLU has slope exactly 0.' },
    { at: 3, label: 'Active middle', say: 'Near zero the slopes are largest: sigmoid peaks at 0.25, tanh at 1.' },
    { at: 9, label: 'Flat right', say: 'On the far right sigmoid and tanh are nearly flat, while ReLU has slope 1 and GELU approaches 1.' },
  ],
};

export interface PanelFrame { f: number; d: number; refs: number[] }
export interface PlotFrame { z: number; phase: number; panels: PanelFrame[] }
/** Everything needed to draw a plot mode at time t. The probe z sweeps from -zr to +zr. */
export function plotFrameAt(t: number, mode: PlotMode, a = LEAK): PlotFrame {
  const zr = ZR[mode], z = Math.min(zr, Math.max(-zr, t - zr));
  let phase = 0;
  SEGMENTS[mode].forEach((s, i) => { if (t >= s.at) phase = i; });
  return {
    z, phase,
    panels: PANELS[mode].map((p) => ({ f: ACT[p.main].f(z, a), d: ACT[p.main].d(z, a), refs: (p.refs ?? []).map((r) => ACT[r].f(z, a)) })),
  };
}

// ------------------------------------------------------------------ collapse: stacked layers on a grid

export type Mat2 = [number, number, number, number];
export type Collapse = 'none' | 'tanh' | 'relu';
export const mul2 = (A: Mat2, B: Mat2): Mat2 => [
  A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3],
  A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3],
];
export const apply2 = (M: Mat2, p: [number, number]): [number, number] => [M[0] * p[0] + M[1] * p[1], M[2] * p[0] + M[3] * p[1]];

/** Three fixed 2×2 weight matrices (no biases), picked by hand so the three pictures are easy to tell apart. */
export const W1: Mat2 = [1.1, -0.8, 0.7, 0.9];
export const W2: Mat2 = [0.8, 0.6, -0.5, 1.0];
export const W3: Mat2 = [1.0, -0.4, 0.6, 0.8];
export const WS = [W1, W2, W3] as const;
/** The single matrix that equals all three layers when there is no activation: W3·W2·W1. */
export const PRODUCT: Mat2 = mul2(W3, mul2(W2, W1));
/** The point the demo follows. */
export const PROBE: [number, number] = [1, -1];

const ACTS: Record<Collapse, (v: number) => number> = { none: (v) => v, tanh: Math.tanh, relu };
export const COLLAPSE_NAME: Record<Collapse, string> = { none: 'no activation', tanh: 'tanh', relu: 'ReLU' };

/** Timeline: a hold, then three layers (matrix morph, then activation morph), then a hold. */
const START = 1, LIN = 1.4, ACTV = 1.1, LAYER = LIN + ACTV, END = START + 3 * LAYER;
export const COLLAPSE_T = END + 1.6;
export const COLLAPSE_MARKERS: Marker[] = [
  { at: 0, label: 'Input' },
  { at: START, label: 'Layer 1' },
  { at: START + LAYER, label: 'Layer 2' },
  { at: START + 2 * LAYER, label: 'Layer 3' },
  { at: END, label: 'Output' },
];

export const duration = (mode: Mode) => (mode === 'collapse' ? COLLAPSE_T : 2 * ZR[mode]);
export const markers = (mode: Mode): Marker[] => (mode === 'collapse' ? COLLAPSE_MARKERS : SEGMENTS[mode].map(({ at, label }) => ({ at, label })));

/** The grid: 5 horizontal then 5 vertical lines at -2..2, each sampled at GRID_N+1 points so bends show. Layout: [x, y] pairs. */
export const GRID_N = 24, GRID_R = 2, GRID_LINES = 2 * (2 * GRID_R + 1);
export function makeGrid(): Float64Array {
  const pts: number[] = [];
  for (const dir of [0, 1]) for (let k = -GRID_R; k <= GRID_R; k++) for (let i = 0; i <= GRID_N; i++) {
    const u = -GRID_R + (2 * GRID_R * i) / GRID_N;
    pts.push(dir === 0 ? u : k, dir === 0 ? k : u);
  }
  return Float64Array.from(pts);
}
const linear = (P: Float64Array, W: Mat2) => {
  const o = new Float64Array(P.length);
  for (let i = 0; i < P.length; i += 2) { o[i] = W[0] * P[i] + W[1] * P[i + 1]; o[i + 1] = W[2] * P[i] + W[3] * P[i + 1]; }
  return o;
};

export interface Stages { pts: Float64Array[]; probe: [number, number][] }
const cache = new Map<Collapse, Stages>();
/** stages 0..6: the input, then per layer the result of the matrix and the result of the activation. The probe point follows the same path. */
export function stagesOf(a: Collapse): Stages {
  let s = cache.get(a);
  if (s) return s;
  const act = ACTS[a];
  let P = makeGrid(), q = PROBE;
  const pts = [P], probe: [number, number][] = [q];
  for (const W of WS) {
    P = linear(P, W); q = apply2(W, q); pts.push(P); probe.push(q);
    P = P.map(act); q = [act(q[0]), act(q[1])]; pts.push(P); probe.push(q);
  }
  s = { pts, probe };
  cache.set(a, s);
  return s;
}
/** Half-width of the square view that holds every stage of one run, so the camera never rescales mid-animation. */
export function viewOf(a: Collapse): number {
  let m = 0;
  for (const P of stagesOf(a).pts) for (const v of P) m = Math.max(m, Math.abs(v));
  // The comparison outline is the linear stack even when the actual stack is nonlinear.
  for (const v of stagesOf('none').pts[6]) m = Math.max(m, Math.abs(v));
  return Math.max(2.5, Math.ceil(m * 1.08 * 2) / 2);
}

export interface CollapseFrame {
  pts: Float64Array;
  probe: [number, number];
  phase: number;
  /** Fade-in (0..1) of the dashed outline where the single matrix would put the border. */
  ghost: number;
}
/** The grid at time t, interpolated between consecutive stages. */
export function collapseFrameAt(t: number, a: Collapse): CollapseFrame {
  const S = stagesOf(a), tt = t - START;
  let seg = 0, u = 0;
  if (tt >= 3 * LAYER) { seg = 5; u = 1; }
  else if (tt > 0) {
    const L = Math.floor(tt / LAYER), r = tt - L * LAYER;
    if (r < LIN) { seg = 2 * L; u = r / LIN; } else { seg = 2 * L + 1; u = (r - LIN) / ACTV; }
  }
  const s = smooth(u), A = S.pts[seg], B = S.pts[seg + 1], P = new Float64Array(A.length);
  for (let i = 0; i < A.length; i++) P[i] = A[i] + (B[i] - A[i]) * s;
  const qa = S.probe[seg], qb = S.probe[seg + 1];
  let phase = 0;
  COLLAPSE_MARKERS.forEach((m, i) => { if (t >= m.at) phase = i; });
  return {
    pts: P, probe: [qa[0] + (qb[0] - qa[0]) * s, qa[1] + (qb[1] - qa[1]) * s], phase,
    ghost: smooth((t - (END - 0.4)) / 0.8),
  };
}
/** Where the probe ends up layer by layer, versus where the single matrix W3·W2·W1 would put it. */
export function collapseResult(a: Collapse) {
  return { actual: stagesOf(a).probe[6], single: apply2(PRODUCT, PROBE) };
}
export function collapseNarration(a: Collapse, phase: number): string {
  const lin = a === 'none';
  if (phase === 0) return `Input: a square grid of points. We follow it through three layers, each a 2×2 weight matrix${lin ? '.' : `, then ${COLLAPSE_NAME[a]}.`}`;
  if (phase === 4) return lin
    ? 'Output: still straight, parallel lines. The grid sits exactly where the single matrix W3·W2·W1 puts it.'
    : `Output: the grid is bent. The dashed outline is where one matrix W3·W2·W1 would put the border, and ${COLLAPSE_NAME[a]} has left it behind.`;
  return lin
    ? `Layer ${phase}: a matrix turns and stretches the grid, and it stays straight and parallel.`
    : `Layer ${phase}: a matrix, then ${COLLAPSE_NAME[a]} bends the grid lines.`;
}
