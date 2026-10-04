/**
 * FeatureViz: train a tiny real CNN, then look inside it four ways (filters, activation
 * maximization, top-activating examples, saliency). Pure math, no DOM: runs in a worker and in tests.
 *
 * Net: 1x16x16 -> conv3x3 (6) + ReLU + maxpool2 -> conv3x3 (8) + ReLU + global max pool -> dense (4) -> softmax.
 * Built on the shared CNN core (src/lib/cnn/conv.ts); only the dense layer, softmax and Adam live here.
 */
import {
  conv2d, conv2dBackward, convLayer, map3, pool2d, pool2dBackward, relu, reluBackward,
  type ConvLayer, type Map3,
} from '../../../lib/cnn/conv';
import { createRng, type Rng } from '../../../lib/math/rng';

export const SIZE = 16;
export const N = SIZE * SIZE;
export const C1 = 6;
export const C2 = 8;
export const NCLS = 4;
export const CLASSES = ['vertical bar', 'horizontal bar', 'diagonal bar', 'ring'] as const;
/** Receptive-field side (input pixels) of a layer-1 and a layer-2 unit. */
export const RF1 = 3;
export const RF2 = 8;

// ---------------------------------------------------------------- data

/** Anti-aliased brightness at distance d from a stroke of half-width w. */
const ink = (d: number, w: number) => Math.min(1, Math.max(0, w + 0.5 - d));

/** One clean image of class `cls`: a stroke at a random place, orientation fixed by the class. */
export function drawClean(cls: number, rng: Rng): Float64Array {
  const im = new Float64Array(N);
  const w = rng.uniform(0.7, 1.1);
  const gain = rng.uniform(0.8, 1);
  let dist: (x: number, y: number) => number;
  if (cls === 0) {
    const x0 = rng.uniform(3, 12), a = rng.uniform(0, 3), b = rng.uniform(12, 16);
    dist = (x, y) => Math.hypot(x - x0, Math.max(a - y, 0, y - b));
  } else if (cls === 1) {
    const y0 = rng.uniform(3, 12), a = rng.uniform(0, 3), b = rng.uniform(12, 16);
    dist = (x, y) => Math.hypot(y - y0, Math.max(a - x, 0, x - b));
  } else if (cls === 2) {
    const flip = rng.next() < 0.5, o = rng.uniform(-3.5, 3.5);
    dist = (x, y) => {
      const xx = flip ? SIZE - 1 - x : x;
      return Math.abs(xx - y + o) / Math.SQRT2;
    };
  } else {
    const cx = rng.uniform(6.5, 9.5), cy = rng.uniform(6.5, 9.5), r = rng.uniform(3, 4.5);
    dist = (x, y) => Math.abs(Math.hypot(x - cx, y - cy) - r);
  }
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) im[y * SIZE + x] = gain * ink(dist(x, y), w);
  return im;
}

export interface Dataset { x: Float64Array[]; y: Uint8Array; clean: Float64Array[] }

/** n images, classes cycled so the set is balanced; Gaussian pixel noise on top. */
export function makeSet(n: number, rng: Rng, noise = 0.12): Dataset {
  const x: Float64Array[] = [], clean: Float64Array[] = [], y = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    y[i] = i % NCLS;
    const c = drawClean(y[i], rng);
    clean.push(c);
    x.push(c.map((v) => v + rng.normal(0, noise)));
  }
  return { x, y, clean };
}

// ---------------------------------------------------------------- network

export interface Net { c1: ConvLayer; c2: ConvLayer; W: Float64Array; b: Float64Array }
const FLAT = C2;
export const INIT1 = 22;

export function initNet(rng: Rng): Net {
  const net: Net = { c1: convLayer(1, C1, 3, 1, 1), c2: convLayer(C1, C2, 3, 1, 1), W: new Float64Array(NCLS * FLAT), b: new Float64Array(NCLS) };
  const fill = (a: Float64Array, fanIn: number) => a.forEach((_, i) => (a[i] = rng.normal(0, Math.sqrt(2 / fanIn))));
  fill(net.c1.weight, 9 * INIT1); fill(net.c2.weight, 9 * C1); fill(net.W, FLAT);
  return net;
}

const cloneNet = (n: Net): Net => ({
  c1: { ...n.c1, weight: n.c1.weight.slice(), bias: n.c1.bias.slice() },
  c2: { ...n.c2, weight: n.c2.weight.slice(), bias: n.c2.bias.slice() },
  W: n.W.slice(), b: n.b.slice(),
});

export interface Cache { x: Map3; a1: Map3; r1: Map3; p1: Map3; a2: Map3; r2: Map3; p2: Map3; z: Float64Array }

export function forward(net: Net, img: ArrayLike<number>): Cache {
  const x = map3(1, SIZE, SIZE, Float64Array.from(img));
  const a1 = conv2d(x, net.c1), r1 = relu(a1), p1 = pool2d(r1, 2, 'max');
  const a2 = conv2d(p1, net.c2), r2 = relu(a2), p2 = pool2d(r2, SIZE / 2, 'max');
  const z = new Float64Array(NCLS);
  for (let k = 0; k < NCLS; k++) {
    let s = net.b[k];
    for (let i = 0; i < FLAT; i++) s += net.W[k * FLAT + i] * p2.data[i];
    z[k] = s;
  }
  return { x, a1, r1, p1, a2, r2, p2, z };
}

export const softmax = (z: ArrayLike<number>) => {
  let m = -Infinity;
  for (let i = 0; i < z.length; i++) m = Math.max(m, z[i]);
  const e = Array.from(z, (v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};
export const argmax = (z: ArrayLike<number>) => { let b = 0; for (let i = 1; i < z.length; i++) if (z[i] > z[b]) b = i; return b; };

export interface Grads { c1W: Float64Array; c1B: Float64Array; c2W: Float64Array; c2B: Float64Array; W: Float64Array; b: Float64Array }
export const newGrads = (): Grads => ({
  c1W: new Float64Array(C1 * 9), c1B: new Float64Array(C1), c2W: new Float64Array(C2 * C1 * 9), c2B: new Float64Array(C2),
  W: new Float64Array(NCLS * FLAT), b: new Float64Array(NCLS),
});

/** Backward from the gradient at the layer-1 ReLU output; returns dL/dinput. */
function backFromR1(net: Net, c: Cache, gr1: Map3, g: Grads): Map3 {
  return conv2dBackward(c.x, net.c1, reluBackward(c.a1, gr1), g.c1W, g.c1B);
}
/** Backward from the gradient at the layer-2 ReLU output. */
function backFromR2(net: Net, c: Cache, gr2: Map3, g: Grads): Map3 {
  const gp1 = conv2dBackward(c.p1, net.c2, reluBackward(c.a2, gr2), g.c2W, g.c2B);
  return backFromR1(net, c, pool2dBackward(c.r1, 2, 'max', gp1), g);
}
/** Backward from dL/dlogits; returns dL/dinput. */
export function backward(net: Net, c: Cache, dz: ArrayLike<number>, g: Grads): Map3 {
  const gp2 = map3(C2, 1, 1);
  for (let k = 0; k < NCLS; k++) {
    g.b[k] += dz[k];
    for (let i = 0; i < FLAT; i++) { g.W[k * FLAT + i] += dz[k] * c.p2.data[i]; gp2.data[i] += dz[k] * net.W[k * FLAT + i]; }
  }
  return backFromR2(net, c, pool2dBackward(c.r2, SIZE / 2, 'max', gp2), g);
}

// ---------------------------------------------------------------- training

export interface Cfg { noise: number; seed: number; nTrain: number; nTest: number; epochs: number; batch: number; lr: number; snaps: number }
export const DEFAULT_CFG: Cfg = { noise: 0.12, seed: 3, nTrain: 240, nTest: 200, epochs: 10, batch: 12, lr: 0.01, snaps: 40 };

export const evaluate = (net: Net, set: Dataset, n = set.x.length) => {
  let right = 0, loss = 0;
  for (let i = 0; i < n; i++) {
    const z = forward(net, set.x[i]).z;
    if (argmax(z) === set.y[i]) right++;
    loss -= Math.log(Math.max(1e-12, softmax(z)[set.y[i]]));
  }
  return { acc: right / n, loss: loss / n };
};

export interface Snapshot { iter: number; filters: Float32Array; acc: number; loss: number }

export function train(cfg: Cfg, onEpoch?: (frac: number) => void) {
  const rng = createRng(cfg.seed);
  const trainSet = makeSet(cfg.nTrain, createRng(cfg.seed * 7 + 1), cfg.noise);
  const testSet = makeSet(cfg.nTest, createRng(cfg.seed * 7 + 2), cfg.noise);
  const net = initNet(rng);
  const params = () => [net.c1.weight, net.c1.bias, net.c2.weight, net.c2.bias, net.W, net.b];
  const P = params();
  const m = P.map((a) => new Float64Array(a.length)), v = P.map((a) => new Float64Array(a.length));
  const perEpoch = Math.floor(cfg.nTrain / cfg.batch), total = perEpoch * cfg.epochs;
  // snapshot iterations: dense early (filters change fastest), always including 0 and the end
  const at = new Set<number>();
  for (let k = 0; k <= cfg.snaps; k++) at.add(Math.round(total * (k / cfg.snaps) ** 2));
  const snaps: Snapshot[] = [];
  const snap = (iter: number) => {
    const e = evaluate(net, testSet, Math.min(cfg.nTest, 120));
    snaps.push({ iter, filters: Float32Array.from(net.c1.weight), acc: e.acc, loss: e.loss });
  };
  if (at.has(0)) snap(0);
  let iter = 0;
  for (let ep = 0; ep < cfg.epochs; ep++) {
    const order = Array.from({ length: cfg.nTrain }, (_, i) => i);
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (let bi = 0; bi < perEpoch; bi++) {
      const g = newGrads();
      for (let s = 0; s < cfg.batch; s++) {
        const idx = order[bi * cfg.batch + s], c = forward(net, trainSet.x[idx]), p = softmax(c.z);
        p[trainSet.y[idx]] -= 1;
        backward(net, c, p.map((q) => q / cfg.batch), g);
      }
      const G = [g.c1W, g.c1B, g.c2W, g.c2B, g.W, g.b];
      iter++;
      const b1 = 0.9, b2 = 0.999;
      P.forEach((w, pi) => {
        for (let i = 0; i < w.length; i++) {
          const gi = G[pi][i];
          m[pi][i] = b1 * m[pi][i] + (1 - b1) * gi; v[pi][i] = b2 * v[pi][i] + (1 - b2) * gi * gi;
          w[i] -= cfg.lr * (m[pi][i] / (1 - b1 ** iter)) / (Math.sqrt(v[pi][i] / (1 - b2 ** iter)) + 1e-8);
        }
      });
      if (at.has(iter)) snap(iter);
    }
    onEpoch?.((ep + 1) / cfg.epochs);
  }
  return { net, snaps, trainSet, testSet, rng, total };
}

// ---------------------------------------------------------------- activation maximization

export type TargetKind = 'class' | 'l2';
export interface Target { kind: TargetKind; index: number }
export const targetKey = (t: Target) => `${t.kind}${t.index}`;
export const targetName = (t: Target) => (t.kind === 'class' ? `the "${CLASSES[t.index]}" score` : `layer 2, channel ${t.index + 1}`);
export const ALL_TARGETS: Target[] = [
  ...Array.from({ length: NCLS }, (_, i): Target => ({ kind: 'class', index: i })),
  ...Array.from({ length: C2 }, (_, i): Target => ({ kind: 'l2', index: i })),
];

/** The number we try to raise: a class score (logit), or a layer-2 channel's average ReLU output over the image. */
export function score(c: Cache, t: Target): number {
  if (t.kind === 'class') return c.z[t.index];
  const r = c.r2, n = r.h * r.w;
  let s = 0;
  for (let i = 0; i < n; i++) s += r.data[t.index * n + i];
  return s / n;
}

/** d score / d input pixels. */
export function inputGrad(net: Net, img: ArrayLike<number>, t: Target): { score: number; g: Float64Array } {
  const c = forward(net, img), g = newGrads();
  let gx: Map3;
  if (t.kind === 'class') gx = backward(net, c, Array.from({ length: NCLS }, (_, k) => (k === t.index ? 1 : 0)), g);
  else {
    const gr = map3(c.r2.c, c.r2.h, c.r2.w), n = c.r2.h * c.r2.w;
    for (let i = 0; i < n; i++) gr.data[t.index * n + i] = 1 / n;
    gx = backFromR2(net, c, gr, g);
  }
  return { score: score(c, t), g: gx.data };
}

export const STEPS = 48;
/** Start brightness (a nearly blank page, like the training images), step size, pull toward blank, blur blend. */
const BASE = 0.1, LR = 0.06, LAMBDA = 0.01, BETA = 0.3;
export type AscentKind = 'plain' | 'reg';
export interface Ascent { frames: Float32Array; act: Float32Array }

const roll = (a: Float64Array, dx: number, dy: number) => {
  const o = new Float64Array(N);
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) o[((y + dy + SIZE) % SIZE) * SIZE + ((x + dx + SIZE) % SIZE)] = a[y * SIZE + x];
  return o;
};
export function blur(a: Float64Array): Float64Array {
  const o = new Float64Array(N), k = [0.25, 0.5, 0.25];
  const tmp = new Float64Array(N);
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    let s = 0; for (let d = -1; d <= 1; d++) s += k[d + 1] * a[y * SIZE + Math.min(SIZE - 1, Math.max(0, x + d))];
    tmp[y * SIZE + x] = s;
  }
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    let s = 0; for (let d = -1; d <= 1; d++) s += k[d + 1] * tmp[Math.min(SIZE - 1, Math.max(0, y + d)) * SIZE + x];
    o[y * SIZE + x] = s;
  }
  return o;
}

/**
 * Gradient ascent on the input image. 'plain' only follows the gradient; 'reg' also pulls the image
 * toward grey (L2 penalty), blurs it a little each step and shakes it by up to one pixel.
 */
export function ascend(net: Net, t: Target, kind: AscentKind, seed = 11, steps = STEPS, lambda = LAMBDA, beta = BETA): Ascent {
  const rng = createRng(seed);
  let x = new Float64Array(N).map(() => BASE + Math.abs(rng.normal(0, 0.04)));
  const frames = new Float32Array((steps + 1) * N), act = new Float32Array(steps + 1);
  const lr = LR;
  for (let s = 0; s <= steps; s++) {
    frames.set(x, s * N);
    const { score: sc, g } = inputGrad(net, x, t);
    act[s] = sc;
    if (s === steps) break;
    if (kind === 'plain') {
      let rms = 0; for (let i = 0; i < N; i++) rms += g[i] * g[i];
      rms = Math.sqrt(rms / N) + 1e-9;
      x = x.map((v, i) => Math.min(1, Math.max(0, v + lr * g[i] / rms)));
    } else {
      const dx = Math.floor(rng.next() * 3) - 1, dy = Math.floor(rng.next() * 3) - 1;
      const xs = roll(x, dx, dy), gs = inputGrad(net, xs, t).g;
      let rms = 0; for (let i = 0; i < N; i++) rms += gs[i] * gs[i];
      rms = Math.sqrt(rms / N) + 1e-9;
      const step = xs.map((v, i) => v + lr * gs[i] / rms - lambda * (v - BASE));
      const bl = blur(step); x = roll(step.map((v, i) => (1 - beta) * v + beta * bl[i]), -dx, -dy).map((v) => Math.min(1, Math.max(0, v)));
    }
  }
  return { frames, act };
}

/** Mean absolute difference between neighbouring pixels: how jagged an image is. */
export function roughness(a: ArrayLike<number>): number {
  let s = 0, n = 0;
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    if (x + 1 < SIZE) { s += Math.abs(a[y * SIZE + x] - a[y * SIZE + x + 1]); n++; }
    if (y + 1 < SIZE) { s += Math.abs(a[y * SIZE + x] - a[(y + 1) * SIZE + x]); n++; }
  }
  return s / n;
}

// ---------------------------------------------------------------- top-activating examples

export interface Example { img: Float32Array; cls: number; y: number; x: number; act: number }
/** n of the `of` best examples share the most common class `cls`. */
export interface Stat { n: number; of: number; cls: number }
export interface Gallery { l1: Example[][]; l2: Example[][]; stat1: Stat[]; stat2: Stat[]; top: number }
export const GALLERY_TOP = 6;
export const PURITY_TOP = 12;

/** Top-activating test images per channel; (y, x) is the centre of the receptive-field box in input pixels. */
export function gallery(net: Net, set: Dataset): Gallery {
  const best: Example[][][] = [Array.from({ length: C1 }, () => []), Array.from({ length: C2 }, () => [])];
  set.x.forEach((img, i) => {
    const c = forward(net, img);
    ([[c.r1, 0], [c.r2, 1]] as const).forEach(([r, L]) => {
      const n = r.h * r.w;
      for (let ch = 0; ch < r.c; ch++) {
        let a = -Infinity, p = 0;
        for (let q = 0; q < n; q++) if (r.data[ch * n + q] > a) { a = r.data[ch * n + q]; p = q; }
        const py = Math.floor(p / r.w), px = p % r.w;
        // layer 2 sits on the 8x8 grid, each cell 2 pixels: its 8x8 box is [2y-3, 2y+4]
        best[L][ch].push({ img: Float32Array.from(img), cls: set.y[i], y: L ? 2 * py : py, x: L ? 2 * px : px, act: a });
      }
    });
  });
  const sort = (arr: Example[]) => arr.sort((p, q) => q.act - p.act);
  const stat = (arr: Example[]): Stat => {
    const cnt = new Array(NCLS).fill(0);
    arr.slice(0, PURITY_TOP).forEach((e) => cnt[e.cls]++);
    const n = Math.max(...cnt);
    return { n, of: Math.min(PURITY_TOP, arr.length), cls: cnt.indexOf(n) };
  };
  const out: Gallery = { l1: [], l2: [], stat1: [], stat2: [], top: GALLERY_TOP };
  best[0].forEach((a) => { sort(a); out.stat1.push(stat(a)); out.l1.push(a.slice(0, GALLERY_TOP)); });
  best[1].forEach((a) => { sort(a); out.stat2.push(stat(a)); out.l2.push(a.slice(0, GALLERY_TOP)); });
  return out;
}

// ---------------------------------------------------------------- saliency

export const STAGES = ['trained', 'last layer randomised', 'last 2 layers randomised', 'all 3 layers randomised'] as const;

/** Net with the top `stage` layers swapped for the same layers of an untrained net (cascading randomization). */
export function randomizeUpTo(net: Net, fresh: Net, stage: number): Net {
  const n = cloneNet(net);
  if (stage >= 1) { n.W = fresh.W.slice(); n.b = fresh.b.slice(); }
  if (stage >= 2) { n.c2 = { ...fresh.c2, weight: fresh.c2.weight.slice(), bias: fresh.c2.bias.slice() }; }
  if (stage >= 3) { n.c1 = { ...fresh.c1, weight: fresh.c1.weight.slice(), bias: fresh.c1.bias.slice() }; }
  return n;
}

export function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
}

/** Gradient magnitude of an image: a plain edge map. */
export function edgeMap(a: ArrayLike<number>): Float64Array {
  const o = new Float64Array(N), at = (y: number, x: number) => a[Math.min(SIZE - 1, Math.max(0, y)) * SIZE + Math.min(SIZE - 1, Math.max(0, x))];
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) o[y * SIZE + x] = Math.hypot(at(y, x + 1) - at(y, x - 1), at(y + 1, x) - at(y - 1, x)) / 2;
  return o;
}

export interface SaliencyCase {
  cls: number; pred: number; logits: number[]; img: Float32Array; edges: Float32Array;
  /** per stage: signed d score / d pixel, and agreement numbers */
  maps: Float32Array[]; byClass: Float32Array[]; corrTrained: number[]; corrEdge: number[]; predStage: number[];
}
export interface Saliency { cases: SaliencyCase[]; meanCorrTrained: number[]; meanCorrEdge: number[]; nAvg: number }

export function saliency(net: Net, fresh: Net, set: Dataset, nAvg = 40): Saliency {
  const nets = STAGES.map((_, s) => randomizeUpTo(net, fresh, s));
  const absOf = (g: ArrayLike<number>) => Float64Array.from(g as ArrayLike<number>, Math.abs);
  const one = (i: number): SaliencyCase => {
    const img = set.x[i], pred = argmax(forward(net, img).z), edges = edgeMap(set.clean[i]);
    const maps = nets.map((n) => inputGrad(n, img, { kind: 'class', index: pred }).g);
    const trained = absOf(maps[0]);
    return {
      cls: set.y[i], pred, logits: Array.from(forward(net, img).z), img: Float32Array.from(img), edges: Float32Array.from(edges),
      maps: maps.map((m) => Float32Array.from(m)),
      byClass: Array.from({ length: NCLS }, (_, k) => Float32Array.from(inputGrad(net, img, { kind: 'class', index: k }).g)),
      corrTrained: maps.map((m) => pearson(absOf(m), trained)),
      corrEdge: maps.map((m) => pearson(absOf(m), edges)),
      predStage: nets.map((n) => argmax(forward(n, img).z)),
    };
  };
  const all = Array.from({ length: Math.min(nAvg, set.x.length) }, (_, i) => one(i));
  const mean = (f: (c: SaliencyCase) => number[]) => STAGES.map((_, s) => all.reduce((a, c) => a + f(c)[s], 0) / all.length);
  // shown cases: the first correctly classified test image of each class
  const cases = Array.from({ length: NCLS }, (_, k) => all.find((c) => c.cls === k && c.pred === k) ?? all.find((c) => c.cls === k)!);
  return { cases, meanCorrTrained: mean((c) => c.corrTrained), meanCorrEdge: mean((c) => c.corrEdge), nAvg: all.length };
}

// ---------------------------------------------------------------- everything the widget needs

export interface Trained {
  cfg: Cfg; snaps: Snapshot[]; finalAcc: number; finalLoss: number; total: number;
  ascents: Record<string, Record<AscentKind, Ascent>>;
  gallery: Gallery; saliency: Saliency;
}

export function trainAll(cfg: Cfg = DEFAULT_CFG, targets: Target[] = ALL_TARGETS, onProgress?: (frac: number) => void): Trained {
  const { net, snaps, testSet, total } = train(cfg, (f) => onProgress?.(0.7 * f));
  const fin = evaluate(net, testSet);
  const ascents: Trained['ascents'] = {};
  targets.forEach((t, i) => {
    ascents[targetKey(t)] = { plain: ascend(net, t, 'plain'), reg: ascend(net, t, 'reg') };
    onProgress?.(0.7 + 0.3 * (i + 1) / targets.length);
  });
  const fresh = initNet(createRng(cfg.seed + 1000));
  return { cfg, snaps, finalAcc: fin.acc, finalLoss: fin.loss, total, ascents, gallery: gallery(net, testSet), saliency: saliency(net, fresh, testSet) };
}

// ---------------------------------------------------------------- timeline frames

export const filterDuration = 10;
export const ascentDuration = 8;
export const galleryDuration = 8;
export const saliencyDuration = 8;
export type Mode = 'filters' | 'maximize' | 'hierarchy' | 'saliency';
export const duration = (m: Mode) => (m === 'filters' ? filterDuration : m === 'maximize' ? ascentDuration : m === 'hierarchy' ? galleryDuration : saliencyDuration);

export const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** Filter snapshot shown at time t (training progress is linear in snapshot index). */
export function filterFrameAt(t: number, tr: Trained) {
  const n = tr.snaps.length, i = Math.min(n - 1, Math.round(clamp01(t / filterDuration) * (n - 1)));
  return { i, snap: tr.snaps[i] };
}
export const filterMarkers = (tr: Trained) => {
  const n = tr.snaps.length, at = (a: number) => (a / (n - 1)) * filterDuration;
  const first = (acc: number) => Math.max(1, tr.snaps.findIndex((s) => s.acc >= acc));
  return [{ at: 0, label: 'guessing' }, { at: at(first(0.4)), label: 'learning' }, { at: at(first(0.9)), label: 'above 90%' }];
};

/** Ascent step shown at time t. */
export function ascentFrameAt(t: number) {
  const step = Math.min(STEPS, Math.round(clamp01(t / ascentDuration) * STEPS));
  return { step };
}
export const ascentMarkers = [{ at: 0, label: 'start' }, { at: ascentDuration * 0.1, label: 'rising' }, { at: ascentDuration * 0.5, label: 'slowing' }];

/** Gallery reveal: layer 1 examples appear one by one over the first half, layer 2 over the second. */
export function galleryFrameAt(t: number, top = GALLERY_TOP) {
  const u = clamp01(t / galleryDuration);
  const n1 = clamp01(u / 0.4) * top, n2 = clamp01((u - 0.45) / 0.45) * top;
  return { n1, n2, layer: u < 0.5 ? 1 : 2 };
}
export const galleryMarkers = [{ at: 0, label: 'layer 1' }, { at: galleryDuration * 0.5, label: 'layer 2' }];

/** The timeline is cut into equal segments, one per option; `i` is the segment t falls in. */
export const segmentAt = (t: number, total: number, n: number) => Math.min(n - 1, Math.floor(clamp01(t / total) * n));
const segMarkers = (total: number, labels: string[]) => labels.map((label, i) => ({ at: (i / labels.length) * total, label }));
export const classMarkers = segMarkers(saliencyDuration, ['vertical', 'horizontal', 'diagonal', 'ring']);
export const saliencyMarkers = segMarkers(saliencyDuration, ['trained', '1 layer', '2 layers', '3 layers']);
