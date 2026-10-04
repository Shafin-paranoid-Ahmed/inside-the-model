/**
 * Boosted trees vs a small neural net: two datasets, learning curves over training-set size,
 * a readable tree vs MLP weights, and a decision checklist. Pure logic, no DOM.
 * The trainers are tiny hand-written versions (Newton-boosted depth-2 trees; a 1-hidden-layer tanh MLP with Adam),
 * so the numbers come from real training runs, not a script.
 */
import { createRng, type Rng } from '../../../lib/math/rng';

// ---------- datasets ----------
export type DatasetId = 'tabular' | 'pixels';
export interface Data { X: Float64Array; y: Uint8Array; n: number; F: number }

export const DATASETS: Record<DatasetId, { F: number; noise: number; names: string[] }> = {
  // 2 informative columns + 2 irrelevant ones, like a spreadsheet with a few useful fields
  tabular: { F: 4, noise: 0.06, names: ['age', 'income', 'col C', 'col D'] },
  // a rotated, curved boundary that no single column explains, like pixels that only mean something together
  pixels: { F: 2, noise: 0.06, names: ['u', 'v'] },
};

/** The noise-free rule. `x` are features in [0, 1]. */
export function truth(id: DatasetId, x: ArrayLike<number>, o = 0): 0 | 1 {
  if (id === 'tabular') return (x[o] > 0.6 && x[o + 1] < 0.7) || (x[o] < 0.25 && x[o + 1] > 0.45) ? 1 : 0;
  return Math.sin(2 * Math.PI * 1.25 * (x[o] + x[o + 1])) > 0 ? 1 : 0; // diagonal stripes: no single column, and no single cut, explains them
}

export function makeData(id: DatasetId, n: number, rng: Rng): Data {
  const { F, noise } = DATASETS[id];
  const X = new Float64Array(n * F), y = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < F; j++) X[i * F + j] = rng.next();
    const t = truth(id, X, i * F);
    y[i] = rng.next() < noise ? 1 - t : t;
  }
  return { X, y, n, F };
}

// ---------- gradient-boosted trees ----------
export interface Tree { f: number; th: number; l: Tree | null; r: Tree | null; v: number; n: number; pos: number }
export interface Work { ops: number }

function grow(d: Data, idx: number[], g: Float64Array, h: Float64Array, depth: number, maxD: number, w: Work): Tree {
  let G = 0, H = 0, pos = 0;
  for (const i of idx) { G += g[i]; H += h[i]; pos += d.y[i]; }
  const leaf: Tree = { f: -1, th: 0, l: null, r: null, v: -G / (H + 1), n: idx.length, pos };
  if (depth >= maxD || idx.length < 4) return leaf;
  let best = 1e-9, bf = -1, bth = 0, bl: number[] = [], br: number[] = [];
  const base = (G * G) / (H + 1);
  for (let f = 0; f < d.F; f++) {
    const o = idx.slice().sort((a, b) => d.X[a * d.F + f] - d.X[b * d.F + f]);
    w.ops += idx.length * (Math.log2(idx.length) + 2);
    let gl = 0, hl = 0;
    for (let k = 0; k < o.length - 1; k++) {
      gl += g[o[k]]; hl += h[o[k]];
      const a = d.X[o[k] * d.F + f], b = d.X[o[k + 1] * d.F + f];
      if (a === b) continue;
      const gain = (gl * gl) / (hl + 1) + ((G - gl) * (G - gl)) / (H - hl + 1) - base;
      if (gain > best) { best = gain; bf = f; bth = (a + b) / 2; bl = o.slice(0, k + 1); br = o.slice(k + 1); }
    }
  }
  if (bf < 0) return leaf;
  return { f: bf, th: bth, l: grow(d, bl, g, h, depth + 1, maxD, w), r: grow(d, br, g, h, depth + 1, maxD, w), v: leaf.v, n: idx.length, pos };
}
const walk = (t: Tree, x: ArrayLike<number>, o: number): Tree => (t.f < 0 ? t : walk(x[o + t.f] <= t.th ? t.l! : t.r!, x, o));
const sig = (z: number) => 1 / (1 + Math.exp(-z));

export interface Boost { trees: Tree[]; lr: number; b0: number }
export const countLeaves = (t: Tree): number => (t.f < 0 ? 1 : countLeaves(t.l!) + countLeaves(t.r!));

export function fitBoost(d: Data, rounds = 40, depth = 2, lr = 0.3): { model: Boost; work: Work; params: number } {
  const w: Work = { ops: 0 }, g = new Float64Array(d.n), h = new Float64Array(d.n), s = new Float64Array(d.n);
  let m = 0; for (let i = 0; i < d.n; i++) m += d.y[i];
  m = Math.min(0.99, Math.max(0.01, m / d.n));
  const b0 = Math.log(m / (1 - m)), trees: Tree[] = [], all = Array.from({ length: d.n }, (_, i) => i);
  s.fill(b0);
  for (let r = 0; r < rounds; r++) {
    for (let i = 0; i < d.n; i++) { const p = sig(s[i]); g[i] = p - d.y[i]; h[i] = Math.max(1e-6, p * (1 - p)); }
    const t = grow(d, all, g, h, 0, depth, w);
    trees.push(t);
    for (let i = 0; i < d.n; i++) s[i] += lr * walk(t, d.X, i * d.F).v;
    w.ops += d.n;
  }
  const params = trees.reduce((a, t) => a + 2 * countLeaves(t) - 1, 0) * 2; // a threshold + feature per split, a value per leaf
  return { model: { trees, lr, b0 }, work: w, params };
}
export function predictBoost(m: Boost, x: ArrayLike<number>, o = 0): number {
  let s = m.b0; for (const t of m.trees) s += m.lr * walk(t, x, o).v;
  return sig(s);
}

/** One depth-limited tree fit on the labels, for reading. Leaves carry counts. */
export function fitTree(d: Data, depth = 3): Tree {
  let pos = 0; for (let i = 0; i < d.n; i++) pos += d.y[i];
  const g = new Float64Array(d.n), h = new Float64Array(d.n).fill(0.25), p = pos / d.n;
  for (let i = 0; i < d.n; i++) g[i] = p - d.y[i];
  return grow(d, Array.from({ length: d.n }, (_, i) => i), g, h, 0, depth, { ops: 0 });
}
export const treeClass = (t: Tree) => (t.pos / t.n >= 0.5 ? 1 : 0);

// ---------- tiny MLP, manual backprop ----------
export interface Mlp { F: number; H: number; W1: Float64Array; b1: Float64Array; W2: Float64Array; b2: number }
export const mlpParams = (F: number, H: number) => (F + 1) * H + H + 1;
const fwd = (m: Mlp, x: ArrayLike<number>, o: number, a: Float64Array): number => {
  let z = m.b2;
  for (let k = 0; k < m.H; k++) {
    let s = m.b1[k];
    for (let j = 0; j < m.F; j++) s += m.W1[k * m.F + j] * (2 * x[o + j] - 1);
    a[k] = Math.tanh(s); z += m.W2[k] * a[k];
  }
  return sig(z);
};
export const predictMlp = (m: Mlp, x: ArrayLike<number>, o = 0) => fwd(m, x, o, new Float64Array(m.H));

export function initMlp(F: number, H: number, rng: Rng): Mlp {
  const W1 = new Float64Array(H * F), W2 = new Float64Array(H);
  for (let i = 0; i < W1.length; i++) W1[i] = rng.normal(0, 1.2);
  for (let i = 0; i < H; i++) W2[i] = rng.normal(0, 0.5);
  return { F, H, W1, b1: new Float64Array(H), W2, b2: 0 };
}

/** Full-batch gradient of the mean log-loss. Returns the loss and a flat gradient [W1, b1, W2, b2]. */
export function lossGrad(m: Mlp, d: Data): { loss: number; grad: Float64Array } {
  const { F, H } = m, gr = new Float64Array(mlpParams(F, H)), a = new Float64Array(H);
  let loss = 0;
  const oB1 = H * F, oW2 = oB1 + H, oB2 = oW2 + H;
  for (let i = 0; i < d.n; i++) {
    const p = fwd(m, d.X, i * F, a), e = p - d.y[i];
    loss -= d.y[i] ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12);
    gr[oB2] += e;
    for (let k = 0; k < H; k++) {
      gr[oW2 + k] += e * a[k];
      const da = e * m.W2[k] * (1 - a[k] * a[k]);
      gr[oB1 + k] += da;
      for (let j = 0; j < F; j++) gr[k * F + j] += da * (2 * d.X[i * F + j] - 1);
    }
  }
  for (let q = 0; q < gr.length; q++) gr[q] /= d.n;
  return { loss: loss / d.n, grad: gr };
}
const flat = (m: Mlp) => { const v = new Float64Array(mlpParams(m.F, m.H)); v.set(m.W1); v.set(m.b1, m.H * m.F); v.set(m.W2, m.H * m.F + m.H); v[v.length - 1] = m.b2; return v; };
const unflat = (m: Mlp, v: Float64Array) => { m.W1.set(v.subarray(0, m.H * m.F)); m.b1.set(v.subarray(m.H * m.F, m.H * m.F + m.H)); m.W2.set(v.subarray(m.H * m.F + m.H, m.H * m.F + 2 * m.H)); m.b2 = v[v.length - 1]; };

export const EPOCHS = 250, HIDDEN = 16;
/** Adam, full batch. `snap(epoch, model)` is called after the epochs listed in `snaps`. */
export function fitMlp(d: Data, rng: Rng, epochs = EPOCHS, H = HIDDEN, lr = 0.05, snaps: number[] = [], snap?: (e: number, m: Mlp, loss: number) => void) {
  const m = initMlp(d.F, H, rng), P = mlpParams(d.F, H), v = flat(m), mo = new Float64Array(P), ve = new Float64Array(P);
  const w: Work = { ops: 0 };
  let last = 0;
  if (snap && snaps.includes(0)) snap(0, m, lossGrad(m, d).loss);
  for (let e = 1; e <= epochs; e++) {
    const { loss, grad } = lossGrad(m, d);
    last = loss;
    w.ops += 3 * d.n * (H * d.F + H);
    for (let q = 0; q < P; q++) {
      mo[q] = 0.9 * mo[q] + 0.1 * grad[q]; ve[q] = 0.999 * ve[q] + 0.001 * grad[q] * grad[q];
      v[q] -= (lr * (mo[q] / (1 - 0.9 ** e))) / (Math.sqrt(ve[q] / (1 - 0.999 ** e)) + 1e-8);
    }
    unflat(m, v);
    if (snap && snaps.includes(e)) snap(e, m, loss);
  }
  return { model: m, work: w, params: P, loss: last };
}

// ---------- learning curves ----------
export const SIZES = [16, 32, 64, 128, 256, 512, 1024];
export const REPEATS = 3;
export const GRID = 24;
const TEST_N = 600;

export interface CurvePoint {
  n: number; accTree: number; accNet: number; opsTree: number; opsNet: number;
  /** P(class 1) on a GRID x GRID slice (first two columns; the others fixed at 0.5), row-major, y up. */
  gridTree: Float32Array; gridNet: Float32Array;
}
export interface Curves { id: DatasetId; points: CurvePoint[]; ceiling: number; paramsTree: number[]; paramsNet: number }

export function computeCurves(id: DatasetId): Curves {
  const { F } = DATASETS[id];
  const test = makeData(id, TEST_N, createRng(9001));
  let hit = 0; for (let i = 0; i < TEST_N; i++) hit += truth(id, test.X, i * F) === test.y[i] ? 1 : 0;
  const acc = (p: (o: number) => number) => { let c = 0; for (let i = 0; i < TEST_N; i++) c += (p(i * F) >= 0.5 ? 1 : 0) === test.y[i] ? 1 : 0; return c / TEST_N; };
  const points: CurvePoint[] = [], paramsTree: number[] = [];
  for (const n of SIZES) {
    let at = 0, an = 0, ot = 0, on = 0, pt = 0, gt: Float32Array = new Float32Array(0), gn: Float32Array = new Float32Array(0);
    for (let r = 0; r < REPEATS; r++) {
      const d = makeData(id, n, createRng(100 * n + r));
      const b = fitBoost(d), m = fitMlp(d, createRng(7 + r));
      at += acc((o) => predictBoost(b.model, test.X, o)); an += acc((o) => predictMlp(m.model, test.X, o));
      ot += b.work.ops; on += m.work.ops; pt += b.params;
      if (r === 0) { gt = gridOf((x) => predictBoost(b.model, x), F); gn = gridOf((x) => predictMlp(m.model, x), F); }
    }
    points.push({ n, accTree: at / REPEATS, accNet: an / REPEATS, opsTree: ot / REPEATS, opsNet: on / REPEATS, gridTree: gt, gridNet: gn });
    paramsTree.push(Math.round(pt / REPEATS));
  }
  return { id, points, ceiling: hit / TEST_N, paramsTree, paramsNet: mlpParams(F, HIDDEN) };
}
function gridOf(p: (x: Float64Array) => number, F = 4): Float32Array {
  const g = new Float32Array(GRID * GRID), x = new Float64Array(F).fill(0.5);
  for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) {
    x[0] = (i + 0.5) / GRID; x[1] = (j + 0.5) / GRID; g[j * GRID + i] = p(x);
  }
  return g;
}

// ---------- reading the models (interpretability) ----------
export const INSPECT_N = 400, INSPECT_SNAPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 250];
export interface Inspect {
  tree: Tree; treeAcc: number; netAcc: number; data: Data; opsTree: number; opsNet: number;
  /** W1 (H x F) at each snapshot epoch, plus the loss. */
  snaps: { epoch: number; W1: Float64Array; loss: number }[]; H: number; F: number; paramsNet: number;
}
export function computeInspect(): Inspect {
  const id: DatasetId = 'tabular', { F } = DATASETS[id];
  const d = makeData(id, INSPECT_N, createRng(314)), test = makeData(id, TEST_N, createRng(9001));
  const tree = fitTree(d, 3), snaps: Inspect['snaps'] = [];
  const net = fitMlp(d, createRng(5), EPOCHS, HIDDEN, 0.05, INSPECT_SNAPS, (epoch, m, loss) => snaps.push({ epoch, W1: m.W1.slice(), loss }));
  let ta = 0, na = 0;
  for (let i = 0; i < TEST_N; i++) {
    ta += treeClass(walk(tree, test.X, i * F)) === test.y[i] ? 1 : 0;
    na += (predictMlp(net.model, test.X, i * F) >= 0.5 ? 1 : 0) === test.y[i] ? 1 : 0;
  }
  return { tree, treeAcc: ta / TEST_N, netAcc: na / TEST_N, data: d, opsTree: treeOps(d, 3), opsNet: net.work.ops, snaps, H: HIDDEN, F, paramsNet: net.params };
}
function treeOps(d: Data, depth: number): number {
  const w: Work = { ops: 0 }, g = new Float64Array(d.n), h = new Float64Array(d.n).fill(0.25);
  for (let i = 0; i < d.n; i++) g[i] = 0.5 - d.y[i];
  grow(d, Array.from({ length: d.n }, (_, i) => i), g, h, 0, depth, w);
  return w.ops;
}
/** Readable rule strings for every leaf of a tree. */
export function rules(t: Tree, names: string[], pre: string[] = []): { text: string; cls: number; n: number; pos: number }[] {
  if (t.f < 0) return [{ text: pre.length ? pre.join(' and ') : 'always', cls: treeClass(t), n: t.n, pos: t.pos }];
  const th = t.th.toFixed(2);
  return [...rules(t.l!, names, [...pre, `${names[t.f]} ≤ ${th}`]), ...rules(t.r!, names, [...pre, `${names[t.f]} > ${th}`])];
}

// ---------- checklist ----------
export interface Choice { data: 'table' | 'media'; rows: 'few' | 'many'; explain: 'yes' | 'no'; latency: 'tight' | 'relaxed' }
export interface Verdict { pick: string; why: string; path: string[] }
export const CHOICE_DEFAULT: Choice = { data: 'table', rows: 'few', explain: 'yes', latency: 'tight' };

/** An honest rule of thumb, not a law: where the usual evidence points. `path` is the subsequence of QUESTIONS that matters. */
export function recommend(c: Choice): Verdict {
  const end = (path: string[], pick: string, why: string): Verdict => ({ pick, why, path });
  if (c.data === 'media') {
    if (c.rows === 'few') return end(['data', 'rows'], 'A pretrained network, fine-tuned', 'Images, text and audio need learned features, and a few thousand labelled rows are too few to learn them from scratch. Start from a pretrained model and adapt it.');
    return c.explain === 'yes'
      ? end(['data', 'rows', 'explain'], 'A neural network, plus explanation tools', 'Raw media is where deep learning wins. The network itself is opaque, so add saliency maps or example-based checks and expect weaker guarantees than a tree gives.')
      : end(['data', 'rows', 'explain'], 'A neural network', 'Raw media is where deep learning wins, and with plenty of labelled data it can learn its own features.');
  }
  if (c.explain === 'yes') return end(['data', 'explain'], 'A linear model or a small decision tree', 'A table you must be able to explain: a few readable rules or weights. You give up a little accuracy to be able to audit every prediction.');
  const all = ['data', 'rows', 'explain', 'latency'];
  if (c.latency === 'tight') return end(all, 'Gradient-boosted trees', 'A table, no need for readable rules, and milliseconds to spare: boosted trees are the usual first pick. They train in seconds and predict fast on a CPU.');
  if (c.rows === 'many') return end(all, 'Boosted trees first, then try a neural net', 'With many rows and a GPU to spare, a neural net can get close to boosted trees on a table and sometimes pass them. Benchmark both, and keep the trees as the baseline.');
  return end(all, 'Gradient-boosted trees', 'A table with few rows: trees are the stronger default, and far cheaper to tune than a network.');
}
/** The nodes the recommendation walks through, in order: each is a question with the chosen answer. */
export function questions(c: Choice) {
  return [
    { id: 'data', q: 'What is the data?', a: c.data === 'table' ? 'A table of columns' : 'Images, text or audio' },
    { id: 'rows', q: 'How many labelled rows?', a: c.rows === 'few' ? 'Thousands or fewer' : 'Hundreds of thousands' },
    { id: 'explain', q: 'Must you explain each prediction?', a: c.explain === 'yes' ? 'Yes' : 'No' },
    { id: 'latency', q: 'Latency budget?', a: c.latency === 'tight' ? 'Tight (CPU, milliseconds)' : 'Relaxed (GPU available)' },
  ];
}

// ---------- timeline frames ----------
export type Mode = 'race' | 'inspect' | 'checklist';
export const T_RACE = 12;
export const duration = (m: Mode) => (m === 'inspect' ? 12 : m === 'checklist' ? 10 : T_RACE);
const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };

export interface RaceFrame { k: number; i: number; f: number; n: number }
const T_LO = 0.4, T_HI = T_RACE - 1, S = SIZES.length;
export const knotT = (j: number) => T_LO + (j * (T_HI - T_LO)) / (S - 1);
export function raceAt(t: number): RaceFrame {
  // Snap to the last finished run: there is no training run between two sizes, so nothing in between is shown.
  const lin = Math.min(1, Math.max(0, (t - T_LO) / (T_HI - T_LO))) * (S - 1) + 1e-9, i = Math.min(S - 2, Math.floor(lin)), f = lin >= S - 1 ? 1 : 0;
  return { k: i + f, i, f, n: SIZES[i + f] };
}
export const raceMarkers = [{ at: 0, label: 'small data' }, { at: knotT(2), label: 'medium data' }, { at: knotT(4), label: 'large data' }];
export const interp = (a: number, b: number, f: number) => a + (b - a) * f;

export interface InspectFrame { snap: number; f: number; depth: number; phase: string }
export const inspectMarkers = [{ at: 0, label: 'the tree splits' }, { at: 4, label: 'the net trains' }, { at: 10, label: 'compare' }];
export function inspectAt(t: number, nSnaps: number): InspectFrame {
  const depth = Math.min(3, Math.max(0, (t - 0.3) / 1.2)); // 0..3 levels revealed
  const u = ease((t - 4) / 5.5), k = u * (nSnaps - 1), snap = Math.min(nSnaps - 2, Math.floor(k));
  return { snap, f: k - snap, depth, phase: t < 4 ? 'the tree splits' : t < 10 ? 'the net trains' : 'compare' };
}

export const QUESTION_IDS = ['data', 'rows', 'explain', 'latency'] as const;
export const checklistMarkers = [{ at: 0, label: 'data' }, { at: 2, label: 'rows' }, { at: 4, label: 'explain' }, { at: 6, label: 'latency' }, { at: 8, label: 'verdict' }];
/** Question j is examined during [2j, 2j+2); the verdict arrives during [8, 9.5]. `head` is the position of the moving packet in node units (0..4). */
export function checklistAt(t: number) {
  return { head: Math.min(4, Math.max(0, t / 2)), verdict: ease((t - 8) / 1.5), phase: checklistMarkers.filter((m) => m.at <= t).at(-1)!.label };
}
