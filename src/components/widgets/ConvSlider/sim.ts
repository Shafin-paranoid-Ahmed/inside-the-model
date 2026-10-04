/** ConvSlider: a 3x3 filter sliding over a small image. Pure math, no DOM. */
import { conv2d, conv2dBackward, convLayer, map3, outSize, type Map3 } from '../../../lib/cnn/conv';
import { createRng } from '../../../lib/math/rng';

export type Mode = 'slide' | 'filters' | 'sharing' | 'learn';

export const N = 12;
export const K = 3;
export const O = outSize(N, K); // 10
export const NPOS = O * O; // 100

export const MAXV = 9;
/** A 12x12 picture with integer brightness 0 to 9: a bright square, two dimmer bars and a little grain. */
export function makeImage(): Float64Array<ArrayBuffer> {
  const a = new Float64Array(N * N);
  const fill = (x0: number, x1: number, y0: number, y1: number, v: number) => {
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) a[y * N + x] = v;
  };
  fill(1, 5, 1, 5, 8); // square
  fill(8, 9, 1, 7, 4); // vertical bar
  fill(2, 10, 9, 10, 4); // horizontal bar
  const rng = createRng(11);
  for (let i = 0; i < a.length; i++) a[i] = Math.min(MAXV, a[i] + Math.floor(rng.next() * 3));
  return a;
}
export const IMAGE = makeImage();
export const asMap = (img: Float64Array<ArrayBuffer>): Map3 => map3(1, N, N, img);

export interface Filter { id: string; name: string; w: number[]; finds: string }
export const FILTERS: Filter[] = [
  { id: 'vertical', name: 'Vertical edge', w: [-1, 0, 1, -1, 0, 1, -1, 0, 1], finds: 'dark-to-bright steps going left to right' },
  { id: 'horizontal', name: 'Horizontal edge', w: [-1, -1, -1, 0, 0, 0, 1, 1, 1], finds: 'dark-to-bright steps going top to bottom' },
  { id: 'blur', name: 'Blur', w: Array(9).fill(1 / 9), finds: 'the local average, so sharp steps get softer' },
  { id: 'sharpen', name: 'Sharpen', w: [0, -1, 0, -1, 5, -1, 0, -1, 0], finds: 'the pixel minus its neighbours, so steps get stronger' },
];

/** Run a 3x3 filter over the image with the shared conv2d. */
export function applyFilter(w: ArrayLike<number>, bias = 0, img: Float64Array<ArrayBuffer> = IMAGE): Float64Array {
  const L = convLayer(1, 1, K);
  for (let i = 0; i < 9; i++) L.weight[i] = w[i];
  L.bias[0] = bias;
  return conv2d(asMap(img), L).data;
}

export interface Breakdown { patch: number[]; prods: number[]; sum: number }
/** The 3x3 patch under output cell `idx`, its nine products with `w`, and their sum. */
export function breakdownAt(idx: number, w: ArrayLike<number>, img: Float64Array<ArrayBuffer> = IMAGE): Breakdown {
  const ox = idx % O, oy = Math.floor(idx / O);
  const patch: number[] = [], prods: number[] = [];
  let sum = 0;
  for (let ky = 0; ky < K; ky++)
    for (let kx = 0; kx < K; kx++) {
      const v = img[(oy + ky) * N + ox + kx], p = v * w[ky * K + kx];
      patch.push(v); prods.push(p); sum += p;
    }
  return { patch, prods, sum };
}

const smooth = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };

export interface Sweep { idx: number; ox: number; oy: number; alpha: number; done: boolean }
/** Window position for u = positions visited so far. Cells before idx are filled, idx fades in. */
export function sweepAt(u: number): Sweep {
  const done = u >= NPOS;
  const idx = done ? NPOS - 1 : Math.max(0, Math.floor(u));
  return { idx, ox: idx % O, oy: Math.floor(idx / O), alpha: done ? 1 : smooth((u - idx) / 0.35), done };
}

// ---- timelines ----
export const RATE = 8; // positions per second in 'slide' and 'sharing'
export const HOLD = 1.5;
export const T_SLIDE = NPOS / RATE + HOLD;
export const SLIDE_MARKERS = [
  { at: 0, label: 'Top rows' }, { at: 30 / RATE, label: 'Middle' }, { at: 70 / RATE, label: 'Bottom rows' },
];
export const slideU = (t: number) => Math.min(NPOS, t * RATE);

export const FRATE = 40;
export const F_SEG = NPOS / FRATE + 1.6; // sweep, then hold on the finished map
export const T_FILTERS = F_SEG * FILTERS.length;
export const FILTER_MARKERS = FILTERS.map((f, i) => ({ at: i * F_SEG, label: f.name.replace(' edge', '') }));
export function filtersAt(t: number) {
  const fi = Math.min(FILTERS.length - 1, Math.floor(t / F_SEG));
  return { fi, u: Math.min(NPOS, (t - fi * F_SEG) * FRATE) };
}
export const filterStart = (fi: number) => fi * F_SEG;

/** Max |output| of a filter on the image: the fixed colour scale for its map. */
export const outScale = (out: Float64Array) => Math.max(1e-9, ...Array.from(out, Math.abs));

// ---- parameter counts ----
export function paramCounts(n = N, k = K) {
  const o = outSize(n, k);
  const conv = k * k + 1;
  const dense = n * n * o * o + o * o;
  return { inputs: n * n, outputs: o * o, convWeights: k * k, conv, denseWeights: n * n * o * o, dense, multiplies: o * o * k * k };
}

// ---- learning a filter ----
export const STEPS = 200;
export const LRATE = 25; // training steps per second
export const T_LEARN = STEPS / LRATE + HOLD;
export const LEARN_MARKERS = [
  { at: 0, label: 'Fast' }, { at: 20 / LRATE, label: 'Settling' }, { at: 120 / LRATE, label: 'Settled' },
];
export const LR = 0.005; // stable below about 0.0055 on this image
export const TARGET = FILTERS[0];

export interface Training { w: Float64Array[]; b: number[]; loss: number[]; out: Float64Array[]; target: Float64Array }
/** Gradient descent on one 3x3 filter and a bias so that conv(image) matches the vertical-edge map. Mean squared error. */
export function train(seed = 3, steps = STEPS, lr = LR): Training {
  const rng = createRng(seed), x = asMap(IMAGE), target = applyFilter(TARGET.w);
  const L = convLayer(1, 1, K);
  for (let i = 0; i < 9; i++) L.weight[i] = rng.normal(0, 0.5);
  const w: Float64Array[] = [], b: number[] = [], loss: number[] = [], out: Float64Array[] = [];
  for (let s = 0; s <= steps; s++) {
    const y = conv2d(x, L);
    let l = 0;
    const gy = map3(1, O, O);
    for (let i = 0; i < NPOS; i++) { const e = y.data[i] - target[i]; l += (e * e) / NPOS; gy.data[i] = (2 * e) / NPOS; }
    w.push(Float64Array.from(L.weight)); b.push(L.bias[0]); loss.push(l); out.push(Float64Array.from(y.data));
    if (s === steps) break;
    const gW = new Float64Array(9), gB = new Float64Array(1);
    conv2dBackward(x, L, gy, gW, gB);
    for (let i = 0; i < 9; i++) L.weight[i] -= lr * gW[i];
    L.bias[0] -= lr * gB[0];
  }
  return { w, b, loss, out, target };
}
export const learnStep = (t: number) => Math.min(STEPS, Math.floor(t * LRATE));
