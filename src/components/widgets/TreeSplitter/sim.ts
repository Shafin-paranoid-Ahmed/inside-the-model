/**
 * Pure logic for TreeSplitter: a small CART decision tree (axis-aligned splits, Gini or entropy, max depth, min leaf),
 * the scripted "story" of a tree growing over time, and the depth / impurity studies. No DOM.
 * `fitTree`, `predict` and `accuracy` are reusable (e.g. by an ensembles lesson).
 */
import { createRng } from '../../../lib/math/rng';

export type Criterion = 'gini' | 'entropy';
export type Mode = 'split' | 'gain' | 'grow' | 'impurity' | 'depth';

const clamp01 = (u: number) => Math.min(1, Math.max(0, u));
const ease = (u: number) => { const c = clamp01(u); return c * c * (3 - 2 * c); };

// ---------------------------------------------------------------- CART

/** Impurity of a node from its class counts. Gini = 1 - sum p^2; entropy = -sum p log2 p (bits). */
export function impurity(counts: readonly number[], crit: Criterion = 'gini'): number {
  let n = 0;
  for (const c of counts) n += c;
  if (n === 0) return 0;
  let s = crit === 'gini' ? 1 : 0;
  for (const c of counts) {
    const p = c / n;
    if (crit === 'gini') s -= p * p;
    else if (p > 0) s -= p * Math.log2(p);
  }
  return s;
}
/** Two-class impurity as a function of the class-1 fraction p. */
export const impurityP = (p: number, crit: Criterion = 'gini') => impurity([1 - p, p], crit);

export interface SplitEval {
  nL: number; nR: number; cL: number[]; cR: number[];
  iL: number; iR: number;
  /** Parent impurity. */
  parent: number;
  /** Size-weighted average impurity of the two children. */
  weighted: number;
  /** Information gain = parent - weighted. */
  gain: number;
}
/** Evaluate the split "X[i][feature] <= thr" on the samples in `idx`. */
export function evalSplit(X: readonly number[][], y: readonly number[], idx: readonly number[], feature: number, thr: number, crit: Criterion = 'gini', K = 2): SplitEval {
  const cL = new Array<number>(K).fill(0), cR = new Array<number>(K).fill(0);
  for (const i of idx) (X[i][feature] <= thr ? cL : cR)[y[i]]++;
  const nL = cL.reduce((a, b) => a + b, 0), nR = cR.reduce((a, b) => a + b, 0), n = nL + nR;
  const iL = impurity(cL, crit), iR = impurity(cR, crit);
  const parent = impurity(cL.map((c, k) => c + cR[k]), crit);
  const weighted = n ? (nL * iL + nR * iR) / n : 0;
  return { nL, nR, cL, cR, iL, iR, parent, weighted, gain: parent - weighted };
}

export interface BestSplit { feature: number; threshold: number; gain: number; weighted: number }
/** Exhaustive search: every feature, every midpoint between neighbouring distinct values. Ties go to the first found. */
export function bestSplit(X: readonly number[][], y: readonly number[], idx: readonly number[], crit: Criterion = 'gini', minLeaf = 1, features?: readonly number[], K = 2): BestSplit | null {
  const n = idx.length, total = new Array<number>(K).fill(0);
  for (const i of idx) total[y[i]]++;
  const parent = impurity(total, crit);
  let best: BestSplit | null = null;
  for (const f of features ?? X[0].map((_, j) => j)) {
    const s = idx.slice().sort((a, b) => X[a][f] - X[b][f]);
    const left = new Array<number>(K).fill(0);
    for (let k = 0; k < n - 1; k++) {
      left[y[s[k]]]++;
      const a = X[s[k]][f], b = X[s[k + 1]][f];
      if (a === b || k + 1 < minLeaf || n - k - 1 < minLeaf) continue;
      const right = total.map((c, j) => c - left[j]);
      const weighted = ((k + 1) * impurity(left, crit) + (n - k - 1) * impurity(right, crit)) / n;
      const gain = parent - weighted;
      if (!best || gain > best.gain + 1e-12) best = { feature: f, threshold: (a + b) / 2, gain, weighted };
    }
  }
  return best;
}

export interface TreeNode {
  id: number; depth: number; n: number; counts: number[]; impurity: number;
  /** Majority class (ties go to the lower class). */
  pred: number;
  /** Indices of the training samples that reach this node. */
  idx: number[];
  /** Region of feature space this node covers. */
  lo: number[]; hi: number[];
  feature?: number; threshold?: number; gain?: number;
  left?: TreeNode; right?: TreeNode;
}
export interface FitOptions {
  maxDepth: number; criterion?: Criterion; minLeaf?: number;
  /** Restrict which features may be split on. */
  features?: readonly number[];
  /** Per-feature [lo, hi] of the root region (default: the data range). */
  bounds?: readonly (readonly [number, number])[];
  nClasses?: number;
}

/** Grow a CART tree greedily: split the node on the best (feature, threshold), recurse until pure, too deep, or no gain. Ids are pre-order. */
export function fitTree(X: readonly number[][], y: readonly number[], opts: FitOptions): TreeNode {
  const crit = opts.criterion ?? 'gini', minLeaf = opts.minLeaf ?? 1, K = opts.nClasses ?? 2, d = X[0].length;
  const lo = Array.from({ length: d }, (_, j) => opts.bounds?.[j][0] ?? Math.min(...X.map((r) => r[j])));
  const hi = Array.from({ length: d }, (_, j) => opts.bounds?.[j][1] ?? Math.max(...X.map((r) => r[j])));
  let next = 0;
  const grow = (idx: number[], depth: number, lo: number[], hi: number[]): TreeNode => {
    const counts = new Array<number>(K).fill(0);
    for (const i of idx) counts[y[i]]++;
    const imp = impurity(counts, crit);
    const node: TreeNode = { id: next++, depth, n: idx.length, counts, impurity: imp, pred: counts.indexOf(Math.max(...counts)), idx, lo, hi };
    if (depth >= opts.maxDepth || imp === 0) return node;
    const s = bestSplit(X, y, idx, crit, minLeaf, opts.features, K);
    if (!s || s.gain <= 1e-12) return node;
    node.feature = s.feature; node.threshold = s.threshold; node.gain = s.gain;
    const L = idx.filter((i) => X[i][s.feature] <= s.threshold), R = idx.filter((i) => X[i][s.feature] > s.threshold);
    const hiL = hi.slice(), loR = lo.slice();
    hiL[s.feature] = s.threshold; loR[s.feature] = s.threshold;
    node.left = grow(L, depth + 1, lo, hiL);
    node.right = grow(R, depth + 1, loR, hi);
    return node;
  };
  return grow(X.map((_, i) => i), 0, lo, hi);
}

export function predictOne(tree: TreeNode, x: readonly number[]): number {
  let n = tree;
  while (n.left && n.right) n = x[n.feature!] <= n.threshold! ? n.left : n.right;
  return n.pred;
}
export const predict = (tree: TreeNode, X: readonly number[][]) => X.map((x) => predictOne(tree, x));
export function accuracy(tree: TreeNode, X: readonly number[][], y: readonly number[]): number {
  let hit = 0;
  for (let i = 0; i < X.length; i++) if (predictOne(tree, X[i]) === y[i]) hit++;
  return hit / X.length;
}
/** Every node, pre-order. */
export function nodesOf(tree: TreeNode): TreeNode[] {
  const out: TreeNode[] = [];
  const walk = (n: TreeNode) => { out.push(n); if (n.left) walk(n.left); if (n.right) walk(n.right); };
  walk(tree);
  return out;
}
export const leavesOf = (tree: TreeNode) => nodesOf(tree).filter((n) => !n.left);

// ---------------------------------------------------------------- data

export interface Data { X: number[][]; y: number[] }
/** Two classes on the unit square. Truth: class 1 iff x > 0.5 and y < 0.6; then a fraction `noise` of labels is flipped. */
export function makeData(seed: number, n: number, noise = 0.12): Data {
  const r = createRng(seed), X: number[][] = [], y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = r.uniform(0.02, 0.98), v = r.uniform(0.02, 0.98), truth = x > 0.5 && v < 0.6 ? 1 : 0;
    X.push([x, v]); y.push(r.next() < noise ? 1 - truth : truth);
  }
  return { X, y };
}
export const TRAIN = makeData(21, 70);
export const VAL = makeData(22, 600);
export const BOUNDS: [number, number][] = [[0, 1], [0, 1]];
export const FEATURES = ['x', 'y'];

// ---------------------------------------------------------------- the growing-tree story (modes split, gain, grow)

/** One feature's candidate sweep at one node. */
export interface Sweep {
  feature: number; lo: number; hi: number;
  /** Weighted child impurity at G evenly spaced thresholds from lo to hi. */
  curve: Float64Array;
  best: { thr: number; weighted: number; gain: number };
}
export interface StoryEvent { node: number; sweeps: Sweep[]; win: number }
export interface Timing { intro: number; sweep: number; snap: number; reveal: number; outro: number }
export interface Story {
  mode: 'split' | 'gain' | 'grow';
  crit: Criterion;
  tree: TreeNode;
  /** Nodes by id. */
  nodes: TreeNode[];
  events: StoryEvent[];
  /** Event index at which a node is split (-1 for leaves) and its parent's event index (-1 for the root). */
  eventOf: number[]; parentEvent: number[];
  /** Flowchart layout: row (may be fractional) of each node, number of leaf rows, deepest level. */
  row: number[]; nRows: number; maxDepth: number;
  timing: Timing; T: number;
  markers: { at: number; label: string }[];
}
export const G = 161;
/** Smallest leaf the story allows, so it does not spend a split on one stray point. */
export const MIN_LEAF = 3;

export function buildStory(mode: Story['mode'], crit: Criterion = 'gini', maxDepth = 3, data: Data = TRAIN, minLeaf = MIN_LEAF): Story {
  const features = mode === 'split' ? [0] : [0, 1];
  const tree = fitTree(data.X, data.y, { maxDepth: mode === 'grow' ? maxDepth : 1, criterion: crit, minLeaf, features, bounds: BOUNDS });
  const nodes = nodesOf(tree).sort((a, b) => a.id - b.id);
  const order = nodes.filter((n) => n.left).sort((a, b) => a.depth - b.depth || a.id - b.id);
  const eventOf = nodes.map(() => -1), parentEvent = nodes.map(() => -1);
  order.forEach((n, e) => { eventOf[n.id] = e; parentEvent[n.left!.id] = e; parentEvent[n.right!.id] = e; });
  const events: StoryEvent[] = order.map((n) => {
    const sweeps = features.map((f): Sweep => {
      const lo = n.lo[f], hi = n.hi[f], curve = new Float64Array(G);
      for (let i = 0; i < G; i++) curve[i] = evalSplit(data.X, data.y, n.idx, f, lo + ((hi - lo) * i) / (G - 1), crit).weighted;
      return { feature: f, lo, hi, curve, best: { thr: 0, weighted: n.impurity, gain: 0 } };
    });
    const win = sweeps.findIndex((s) => s.feature === n.feature);
    const w = sweeps[win];
    w.best = { thr: n.threshold!, weighted: n.impurity - n.gain!, gain: n.gain! };
    // the losing features: their own best split, for comparison
    sweeps.forEach((s, i) => {
      if (i === win) return;
      const b = bestSplit(data.X, data.y, n.idx, crit, minLeaf, [s.feature]);
      if (b) s.best = { thr: b.threshold, weighted: b.weighted, gain: b.gain };
    });
    return { node: n.id, sweeps, win };
  });
  // flowchart rows: leaves get consecutive rows, parents sit between their children
  const row = nodes.map(() => 0);
  let r = 0;
  const place = (n: TreeNode): number => { row[n.id] = n.left ? (place(n.left) + place(n.right!)) / 2 : r++; return row[n.id]; };
  place(tree);
  const timing: Timing = mode === 'split' ? { intro: 1, sweep: 5, snap: 1.2, reveal: 2.8, outro: 0 }
    : mode === 'gain' ? { intro: 1, sweep: 5.5, snap: 1.3, reveal: 2.7, outro: 0 }
      : { intro: 0.8, sweep: 1.3, snap: 0.5, reveal: 0.7, outro: 1.5 };
  const per = timing.sweep + timing.snap + timing.reveal, T = timing.intro + events.length * per + timing.outro;
  const markers = mode === 'grow'
    ? [...order.flatMap((n, e) => (e === 0 || order[e - 1].depth !== n.depth ? [{ at: e === 0 ? 0 : timing.intro + e * per, label: n.depth === 0 ? 'root split' : `depth ${n.depth} splits` }] : [])), { at: T - timing.outro, label: 'done' }]
    : [{ at: 0, label: 'sweep' }, { at: timing.intro + timing.sweep, label: 'settle' }, { at: timing.intro + timing.sweep + timing.snap, label: 'result' }];
  return { mode, crit, tree, nodes, events, eventOf, parentEvent, row, nRows: r, maxDepth: Math.max(...nodes.map((n) => n.depth)), timing, T, markers };
}

export interface SweepFrame {
  feature: number;
  /** Candidate threshold position. */
  thr: number;
  /** Fraction of the impurity curve drawn so far (0..1). */
  traced: number;
  /** Visibility of this candidate line (losing features fade out when the best split snaps in). */
  alpha: number;
  /** The split evaluated exactly at `thr`. */
  ev: SplitEval;
}
export interface Frame {
  stage: 'intro' | 'sweep' | 'snap' | 'reveal' | 'done';
  event: number;
  node: number;
  sweeps: SweepFrame[];
  /** Number of events whose split is now in place (the last one may still be fading in). */
  splitCount: number;
  /** Fade-in of the children created by event splitCount - 1. */
  reveal: number;
}

/** Everything needed to draw the story at time t. Pure: scrubbing is free. */
export function frameAt(t: number, s: Story, data: Data = TRAIN): Frame {
  const { intro, sweep, snap, reveal } = s.timing, per = sweep + snap + reveal, m = s.events.length;
  t = Math.min(s.T, Math.max(0, t));
  let stage: Frame['stage'] = 'intro', event = 0, e = 0, u = 0, a = 0;
  if (t >= intro) {
    const local = t - intro;
    event = Math.floor(local / per);
    if (event >= m) { event = m - 1; stage = 'done'; a = 1; } else {
      const r = local - event * per;
      if (r < sweep) { stage = 'sweep'; e = r / sweep; } else if (r < sweep + snap) { stage = 'snap'; u = (r - sweep) / snap; } else { stage = 'reveal'; a = (r - sweep - snap) / reveal; }
    }
  }
  const E = s.events[event], node = s.nodes[E.node];
  const sweeps = E.sweeps.map((sw, i): SweepFrame => {
    const span = sw.hi - sw.lo, win = i === E.win;
    let thr = sw.lo, traced = 0, alpha = 0;
    if (stage === 'sweep') { thr = sw.lo + span * ease(e); traced = ease(e); alpha = 1; }
    else if (stage === 'snap') { traced = 1; thr = win ? sw.hi + (sw.best.thr - sw.hi) * ease(u) : sw.hi; alpha = win ? 1 : 1 - ease(u); }
    else if (stage !== 'intro') { traced = 1; thr = win ? sw.best.thr : sw.hi; alpha = win ? 1 : 0; }
    return { feature: sw.feature, thr, traced, alpha, ev: evalSplit(data.X, data.y, node.idx, sw.feature, thr, s.crit) };
  });
  const done = stage === 'reveal' || stage === 'done';
  return { stage, event, node: E.node, sweeps, splitCount: event + (done ? 1 : 0), reveal: stage === 'done' ? 1 : stage === 'reveal' ? ease(a) : 0 };
}

/** Opacity of a node in the flowchart / partition at this frame. */
export function nodeAlpha(s: Story, f: Frame, id: number): number {
  const pe = s.parentEvent[id];
  return pe < 0 || pe < f.splitCount - 1 ? 1 : pe === f.splitCount - 1 ? f.reveal : 0;
}
/** Has this node been split by this frame? */
export const isSplit = (s: Story, f: Frame, id: number) => s.eventOf[id] >= 0 && s.eventOf[id] < f.splitCount;

// ---------------------------------------------------------------- depth study (mode depth)

export const DMAX = 8, T_DEPTH = 12;
export interface Study { trees: TreeNode[]; train: number[]; val: number[]; leaves: number[]; markers: { at: number; label: string }[] }
/** Fit one tree per max depth 0..DMAX and record training and validation accuracy. */
export function depthStudy(minLeaf = 1, crit: Criterion = 'gini', train: Data = TRAIN, val: Data = VAL): Study {
  const trees = Array.from({ length: DMAX + 1 }, (_, d) => fitTree(train.X, train.y, { maxDepth: d, criterion: crit, minLeaf, bounds: BOUNDS }));
  const tr = trees.map((t) => accuracy(t, train.X, train.y)), va = trees.map((t) => accuracy(t, val.X, val.y));
  const best = va.indexOf(Math.max(...va)), mem = tr.findIndex((a) => a >= 1);
  const at = (d: number) => (d * T_DEPTH) / DMAX;
  const markers = [{ at: 0, label: 'one leaf' }, ...(best > 0 ? [{ at: at(best), label: 'best on validation' }] : []), ...(mem > best ? [{ at: at(mem), label: 'memorised' }] : [])];
  return { trees, train: tr, val: va, leaves: trees.map((t) => leavesOf(t).length), markers };
}
/** Continuous depth marker v (sits on integers, hops in the last 40% of each interval) and the depth shown, round(v). */
export function depthAt(t: number): { v: number; d: number } {
  const u = clamp01(t / T_DEPTH) * DMAX, d = Math.min(DMAX - 1, Math.floor(u));
  const v = d + ease((u - d - 0.6) / 0.4);
  return { v, d: Math.round(v) };
}

// ---------------------------------------------------------------- impurity study (mode impurity)

export const T_IMP = 12, BAG = 100;
export const IMP_MARKERS = [{ at: 0, label: 'pure' }, { at: 2.4, label: 'mixing' }, { at: 6, label: 'most mixed' }, { at: 9.6, label: 'unmixing' }];
/** Seeded rank of each of the 100 dots: dot i is class 1 iff rank[i] < k, so dots flip one at a time and never flip back. */
export const BAG_RANK: number[] = (() => {
  const r = createRng(5), order = Array.from({ length: BAG }, (_, i) => i);
  for (let i = BAG - 1; i > 0; i--) { const j = Math.floor(r.next() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const rank = new Array<number>(BAG);
  order.forEach((dot, k) => { rank[dot] = k; });
  return rank;
})();
/** The class-1 fraction is counted in steps of 1/100 so the dots, p and both impurities always agree. */
export function impurityAt(t: number) {
  const k = Math.round(clamp01(t / T_IMP) * BAG), p = k / BAG;
  return { k, p, gini: impurityP(p, 'gini'), entropy: impurityP(p, 'entropy') };
}
