/** Pure logic for KNNVoter: k-nearest-neighbour classification on 2-D points, plus distance concentration in d dimensions. No DOM. */
import { createRng } from '../../../lib/math/rng';

export type Mode = 'vote' | 'metric' | 'k' | 'scale' | 'dims';
export interface Pt { x: number; y: number; c: 0 | 1 }
export interface Marker { at: number; label: string }

/** The plane shows [-DOM, DOM]^2. */
export const DOM = 1.5;
/** Neighbour counts visited by the k mode. */
export const KS = [1, 5, 15, 31, 51, 75];
export const DIMS_MAX = 200;
const NQ = 30, NP = 200, BINS = 36, HMAX = 3;

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const smooth = (u: number) => { const c = clamp(u, 0, 1); return c * c * (3 - 2 * c); };
const clip = (v: number) => clamp(v, -DOM + 0.08, DOM - 0.08);

/** Minkowski distance with p = 1 (Manhattan), 2 (Euclidean) or in between. */
export function dist(p: number, dx: number, dy: number): number {
  const ax = Math.abs(dx), ay = Math.abs(dy);
  if (p === 2) return Math.sqrt(ax * ax + ay * ay);
  if (p === 1) return ax + ay;
  return Math.pow(Math.pow(ax, p) + Math.pow(ay, p), 1 / p);
}

/** Two interleaved half-moons (n points per class), centred on the origin. */
export function moons(seed: number, n: number, noise = 0.3): Pt[] {
  const rng = createRng(seed), pts: Pt[] = [];
  for (const c of [0, 1] as const) {
    for (let i = 0; i < n; i++) {
      const a = rng.uniform(0, Math.PI);
      const bx = c ? 1 - Math.cos(a) : Math.cos(a), by = c ? 0.5 - Math.sin(a) : Math.sin(a);
      pts.push({ x: clip((bx - 0.5) * 0.85 + rng.normal(0, noise)), y: clip((by - 0.25) * 0.85 + rng.normal(0, noise)), c });
    }
  }
  return pts;
}

/** Class depends on x only; y is noise. */
export function stripes(seed: number, n: number): Pt[] {
  const rng = createRng(seed), pts: Pt[] = [];
  for (const c of [0, 1] as const) {
    for (let i = 0; i < n; i++) pts.push({ x: clip(rng.normal(c ? 0.55 : -0.55, 0.3)), y: rng.uniform(-1.2, 1.2), c });
  }
  return pts;
}

// --- neighbours -----------------------------------------------------------------------------------------------

export interface Nearest { idx: number[]; d: number[] }
/** The m points closest to (qx, qy), nearest first. The y axis is multiplied by wy before measuring. */
export function nearest(pts: readonly Pt[], qx: number, qy: number, m: number, p = 2, wy = 1): Nearest {
  const idx: number[] = [], d: number[] = [];
  m = Math.min(m, pts.length);
  for (let i = 0; i < pts.length; i++) {
    const dd = dist(p, pts[i].x - qx, (pts[i].y - qy) * wy);
    let j: number;
    if (idx.length < m) { idx.push(i); d.push(dd); j = idx.length - 1; }
    else if (dd < d[m - 1]) { idx[m - 1] = i; d[m - 1] = dd; j = m - 1; }
    else continue;
    while (j > 0 && d[j - 1] > d[j]) { [d[j - 1], d[j]] = [d[j], d[j - 1]]; [idx[j - 1], idx[j]] = [idx[j], idx[j - 1]]; j--; }
  }
  return { idx, d };
}

const sd = new Float64Array(256), sc = new Uint8Array(256);
/** Fraction of the k nearest training points that are class 1 (allocation-free; k <= 255). */
export function knnFrac(pts: readonly Pt[], qx: number, qy: number, k: number, p = 2, wy = 1): number {
  let n = 0;
  for (let i = 0; i < pts.length; i++) {
    const P = pts[i], d = dist(p, P.x - qx, (P.y - qy) * wy);
    if (n < k) n++; else if (d >= sd[k - 1]) continue;
    let j = n - 1;
    while (j > 0 && sd[j - 1] > d) { sd[j] = sd[j - 1]; sc[j] = sc[j - 1]; j--; }
    sd[j] = d; sc[j] = P.c;
  }
  let s = 0;
  for (let j = 0; j < n; j++) s += sc[j];
  return s / n;
}

/** Share of `test` points the k-NN vote over `train` gets right. Passing the same array twice counts each point as its own neighbour. */
export function accuracy(train: readonly Pt[], test: readonly Pt[], k: number, p = 2, wy = 1): number {
  let hit = 0;
  for (const q of test) if ((knnFrac(train, q.x, q.y, k, p, wy) > 0.5 ? 1 : 0) === q.c) hit++;
  return hit / test.length;
}

/** P(class 1) = vote share on an n×n grid of cell centres; row j = 0 is the bottom. */
export function field(pts: readonly Pt[], n: number, k: number, p: number, wy: number, out: Float32Array) {
  for (let j = 0; j < n; j++) {
    const y = -DOM + ((j + 0.5) / n) * 2 * DOM;
    for (let i = 0; i < n; i++) out[j * n + i] = knnFrac(pts, -DOM + ((i + 0.5) / n) * 2 * DOM, y, k, p, wy);
  }
}

/** Train and validation accuracy for every k in `ks`, from one sorted neighbour list per evaluated point. */
export function accByK(train: readonly Pt[], val: readonly Pt[], ks: readonly number[]) {
  const run = (test: readonly Pt[]) => {
    const hit = new Float64Array(ks.length);
    for (const q of test) {
      const { idx } = nearest(train, q.x, q.y, Math.max(...ks));
      let ones = 0, j = 0;
      ks.forEach((k, a) => {
        for (; j < k; j++) ones += train[idx[j]].c;
        if ((ones / k > 0.5 ? 1 : 0) === q.c) hit[a]++;
      });
    }
    return Array.from(hit, (h) => h / test.length);
  };
  return { train: run(train), val: run(val) };
}

// --- the scene: everything precomputed for one mode ---------------------------------------------------------------

export const GRID = 72;
export const ODD_KS = Array.from({ length: 38 }, (_, i) => 2 * i + 1); // 1..75, the k-mode chart
export const RHOS = Array.from({ length: 61 }, (_, i) => Math.pow(100, i / 60)); // 1..100, the scale-mode chart

const T_OF: Record<Mode, number> = { vote: 12, metric: 12, k: 11, scale: 12, dims: 14 };
const MARKERS_OF: Record<Mode, Marker[]> = {
  vote: [{ at: 0, label: 'Orange' }, { at: 2.5, label: 'Border' }, { at: 5, label: 'Blue' }, { at: 8, label: 'Return' }],
  metric: [{ at: 0, label: 'Grow' }, { at: 3, label: 'Circle' }, { at: 4.5, label: 'Morph' }, { at: 8, label: 'Diamond' }],
  k: KS.map((k, i) => ({ at: (i * T_OF.k) / KS.length, label: `${k}-NN` })),
  scale: [{ at: 0, label: 'Equal' }, { at: 1.5, label: 'Stretch' }, { at: 6.5, label: 'Standardise' }],
  dims: [{ at: 0, label: '1 dimension' }, { at: 7, label: '200 dimensions' }],
};

export interface DimsData {
  /** Per dimension d = 1..DIMS_MAX (index d-1): mean nearest/farthest ratio over NQ queries, and for query 0 the nearest, farthest and mean distance. */
  ratio: Float64Array; near: Float64Array; far: Float64Array; mean: Float64Array;
  /** Histogram of query 0's distances divided by their mean, BINS bins over [0, HMAX], as shares; BINS * d + bin. */
  hist: Float32Array;
}

export interface Scene {
  mode: Mode; T: number; markers: Marker[];
  train: Pt[]; val: Pt[];
  k: number;
  /** vote/metric/scale: the query's resting place. */
  q0: { x: number; y: number };
  field?: Float32Array; fields?: Float32Array[];
  accK?: { train: number[]; val: number[] };
  accRho?: number[]; rhoStd?: number;
  dims?: DimsData;
}

function buildDims(): DimsData {
  const rng = createRng(11), rows = NQ + NP, X = new Float32Array(rows * DIMS_MAX);
  for (let i = 0; i < X.length; i++) X[i] = rng.next();
  const d2 = new Float64Array(NQ * NP);
  const ratio = new Float64Array(DIMS_MAX), near = new Float64Array(DIMS_MAX), far = new Float64Array(DIMS_MAX), mean = new Float64Array(DIMS_MAX);
  const hist = new Float32Array(BINS * DIMS_MAX);
  for (let d = 0; d < DIMS_MAX; d++) {
    let r = 0;
    for (let q = 0; q < NQ; q++) {
      let lo = Infinity, hi = 0, sum = 0;
      for (let p = 0; p < NP; p++) {
        const v = X[q * DIMS_MAX + d] - X[(NQ + p) * DIMS_MAX + d];
        const s = (d2[q * NP + p] += v * v), e = Math.sqrt(s);
        if (e < lo) lo = e;
        if (e > hi) hi = e;
        sum += e;
      }
      r += lo / hi;
      if (q === 0) {
        near[d] = lo; far[d] = hi; mean[d] = sum / NP;
        for (let p = 0; p < NP; p++) {
          const b = Math.min(BINS - 1, Math.floor((Math.sqrt(d2[p]) / mean[d] / HMAX) * BINS));
          hist[d * BINS + b] += 1 / NP;
        }
      }
    }
    ratio[d] = r / NQ;
  }
  return { ratio, near, far, mean, hist };
}

export function makeScene(mode: Mode, k = 5): Scene {
  const scene: Scene = { mode, T: T_OF[mode], markers: MARKERS_OF[mode], train: [], val: [], k, q0: { x: 0, y: 0 } };
  if (mode === 'dims') { scene.dims = buildDims(); return scene; }
  if (mode === 'scale') {
    scene.train = stripes(31, 50); scene.val = stripes(32, 50);
    scene.q0 = SCALE_Q;
    const sdv = (f: (p: Pt) => number) => { const m = scene.train.reduce((s, p) => s + f(p), 0) / scene.train.length; return Math.sqrt(scene.train.reduce((s, p) => s + (f(p) - m) ** 2, 0) / scene.train.length); };
    scene.rhoStd = sdv((p) => p.x) / sdv((p) => p.y);
    scene.accRho = RHOS.map((r) => accuracy(scene.train, scene.val, scene.k, 2, r));
    return scene;
  }
  scene.train = moons(21, 50); scene.val = moons(22, 50);
  if (mode === 'vote') { scene.q0 = VOTE_A; scene.field = new Float32Array(GRID * GRID); field(scene.train, GRID, scene.k, 2, 1, scene.field); }
  if (mode === 'metric') scene.q0 = METRIC_Q;
  if (mode === 'k') {
    scene.fields = KS.map((k) => { const f = new Float32Array(GRID * GRID); field(scene.train, GRID, k, 2, 1, f); return f; });
    scene.accK = accByK(scene.train, scene.val, ODD_KS);
  }
  return scene;
}

// query positions (chosen so the demos show their effect)
export const VOTE_A = { x: -0.9, y: 0.55 };
export const VOTE_B = { x: 0.9, y: -0.45 };
export const VOTE_BEND = 0.0;
export const METRIC_Q = { x: -0.2, y: -0.2 };
export const SCALE_Q = { x: 0.3, y: 0.2 };

export const votePath = (t: number, T = T_OF.vote) => {
  const s = 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / T);
  return { x: VOTE_A.x + (VOTE_B.x - VOTE_A.x) * s, y: VOTE_A.y + (VOTE_B.y - VOTE_A.y) * s + VOTE_BEND * Math.sin(Math.PI * s) };
};

// --- frames -----------------------------------------------------------------------------------------------------

export interface Neighbour { i: number; d: number; /** 0..1, how firmly it is inside the ball */ a: number; inside: boolean }
export interface Frame {
  mode: Mode; phase: string;
  q: { x: number; y: number } | null;
  /** Minkowski exponent, and the y-axis stretch applied before measuring. */
  p: number; rho: number;
  k: number; /** the ball radius in distance units */ r: number;
  nn: Neighbour[]; votes: [number, number]; pred: 0 | 1 | null;
  /** k mode: fractional index into KS, and the two accuracies at the current k. */
  ki: number; trainAcc: number; valAcc: number;
  /** dims mode */
  d: number; near: number; far: number; ratio: number; hist: Float32Array | null;
}

const blank = (mode: Mode): Frame => ({
  mode, phase: '', q: null, p: 2, rho: 1, k: 0, r: 0, nn: [], votes: [0, 0], pred: null,
  ki: 0, trainAcc: 0, valAcc: 0, d: 0, near: 0, far: 0, ratio: 0, hist: null,
});

const phaseAt = (t: number, m: Marker[]) => m.reduce((s, x) => (x.at <= t + 1e-9 ? x.label : s), m[0].label);

/** The ball around q that holds exactly the k nearest (g = 1), or a growing fraction g of it. */
function ball(f: Frame, pts: readonly Pt[], q: { x: number; y: number }, k: number, g: number) {
  const n = nearest(pts, q.x, q.y, k + 2, f.p, f.rho);
  const rb = (n.d[k - 1] + n.d[k]) / 2, r = g * rb, w = 0.15 * rb;
  f.q = q; f.k = k; f.r = r;
  f.nn = n.idx.map((i, j) => ({ i, d: n.d[j], a: clamp((r - n.d[j]) / w + 0.5, 0, 1), inside: n.d[j] <= r }));
  for (const e of f.nn) if (e.inside) f.votes[pts[e.i].c]++;
  f.pred = f.votes[0] === f.votes[1] ? null : f.votes[1] > f.votes[0] ? 1 : 0;
}

/** Everything needed to draw the scene at time t (seconds, looping). Scrubbing is free. `q` overrides the query position. */
export function frameAt(t: number, sc: Scene, opt: { q?: { x: number; y: number } } = {}): Frame {
  const f = blank(sc.mode), T = sc.T;
  t = clamp(t, 0, T);
  f.phase = phaseAt(t, sc.markers);
  if (sc.mode === 'vote') ball(f, sc.train, opt.q ?? votePath(t, T), sc.k, 1);
  else if (sc.mode === 'metric') {
    // grow (0-3), hold, morph 2 -> 1 (4.5-8), hold, morph back (9.5-11.4), shrink (11.4-12)
    const grow = smooth(t / 3), shrink = 1 - smooth((t - 11.4) / 0.6);
    f.p = 2 - smooth((t - 4.5) / 3.5) + smooth((t - 9.5) / 1.9);
    ball(f, sc.train, opt.q ?? sc.q0, sc.k, Math.min(grow, shrink));
  } else if (sc.mode === 'scale') {
    // stretch y by up to 100x (1-5), hold, standardise (6.5-10), hold
    const top = Math.log(100), std = Math.log(sc.rhoStd!);
    f.rho = Math.exp(top * smooth((t - 1.5) / 4) + (std - top) * smooth((t - 6.5) / 3.5));
    ball(f, sc.train, opt.q ?? sc.q0, sc.k, 1);
    f.valAcc = accuracy(sc.train, sc.val, sc.k, 2, f.rho);
  } else if (sc.mode === 'k') {
    const seg = T / KS.length, i = Math.floor(Math.min(t, T - 1e-9) / seg), u = (t - i * seg) / seg;
    f.ki = (i > 0 ? i - 1 + smooth(u / 0.3) : 0) + (i === KS.length - 1 ? smooth((u - 0.7) / 0.3) : 0);
    const a = ODD_KS.indexOf(KS[i]);
    // the shown k is the active timeline marker's k; only the shading eases between them
    f.k = KS[i]; f.trainAcc = sc.accK!.train[a]; f.valAcc = sc.accK!.val[a];
  } else {
    const D = sc.dims!, s = 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / T), df = Math.exp(Math.log(DIMS_MAX) * s);
    const lo = clamp(Math.floor(df) - 1, 0, DIMS_MAX - 2), fr = df - 1 - lo, L = (a: Float64Array) => a[lo] * (1 - fr) + a[lo + 1] * fr;
    f.d = df; f.near = L(D.near) / L(D.mean); f.far = L(D.far) / L(D.mean); f.ratio = L(D.ratio);
    f.hist = new Float32Array(BINS);
    for (let b = 0; b < BINS; b++) f.hist[b] = D.hist[lo * BINS + b] * (1 - fr) + D.hist[(lo + 1) * BINS + b] * fr;
  }
  return f;
}

export const BINS_N = BINS;
export const HIST_MAX = HMAX;
