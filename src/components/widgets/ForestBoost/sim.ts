/**
 * Pure logic for ForestBoost: bagging and random forests on a 2D classification set (CART trees from TreeSplitter),
 * gradient boosting and a random forest on a 1D regression curve (a tiny regression tree), and the timelines that play them.
 * No DOM. Everything is fitted once at construction; `*Frame` functions only pick and interpolate, so scrubbing is free.
 */
import { createRng } from '../../../lib/math/rng';
import { fitTree, bestSplit, predictOne, impurity, type TreeNode } from '../TreeSplitter/sim';

export type Mode = 'bag' | 'forest' | 'boost' | 'gradient' | 'compare';
const clamp01 = (u: number) => Math.min(1, Math.max(0, u));
export const ease = (u: number) => { const c = clamp01(u); return c * c * (3 - 2 * c); };

// ================================================================ classification: bagging and random forests

export interface ClsData { X: number[][]; y: number[] }
/** Two classes on the unit square, separated by a wavy diagonal; a fraction `noise` of labels is flipped. */
export function makeCls(seed: number, n: number, noise: number): ClsData {
  const r = createRng(seed), X: number[][] = [], y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = r.uniform(0.02, 0.98), v = r.uniform(0.02, 0.98), truth = x + v + 0.08 * Math.sin(6 * x) > 1 ? 1 : 0;
    X.push([x, v]); y.push(r.next() < noise ? 1 - truth : truth);
  }
  return { X, y };
}
export const N_TRAIN = 150;
export const CLS_TRAIN = makeCls(41, N_TRAIN, 0.1);
export const CLS_VAL = makeCls(42, 1000, 0.1);
export const CB: [number, number][] = [[0, 1], [0, 1]];
/** Resolution of the shaded decision map. */
export const GC = 44;
const GRID_PTS: number[][] = (() => { const o: number[][] = []; for (let j = 0; j < GC; j++) for (let i = 0; i < GC; i++) o.push([(i + 0.5) / GC, 1 - (j + 0.5) / GC]); return o; })();

/** CART tree that may look at only `mtry` randomly chosen features per split (the random-forest recipe). Reuses `bestSplit`. */
export function growRandom(X: readonly number[][], y: readonly number[], pick: () => number[], maxDepth: number, bounds: readonly (readonly [number, number])[] = CB): TreeNode {
  let next = 0;
  const grow = (idx: number[], depth: number, lo: number[], hi: number[]): TreeNode => {
    const counts = [0, 0];
    for (const i of idx) counts[y[i]]++;
    const imp = impurity(counts);
    const node: TreeNode = { id: next++, depth, n: idx.length, counts, impurity: imp, pred: counts[1] > counts[0] ? 1 : 0, idx, lo, hi };
    if (depth >= maxDepth || imp === 0) return node;
    const s = bestSplit(X, y, idx, 'gini', 1, pick()) ?? bestSplit(X, y, idx, 'gini', 1);
    if (!s || s.gain <= 1e-12) return node;
    node.feature = s.feature; node.threshold = s.threshold; node.gain = s.gain;
    const hiL = hi.slice(), loR = lo.slice();
    hiL[s.feature] = s.threshold; loR[s.feature] = s.threshold;
    node.left = grow(idx.filter((i) => X[i][s.feature] <= s.threshold), depth + 1, lo, hiL);
    node.right = grow(idx.filter((i) => X[i][s.feature] > s.threshold), depth + 1, loR, hi);
    return node;
  };
  return grow(X.map((_, i) => i), 0, bounds.map((b) => b[0]), bounds.map((b) => b[1]));
}

export interface ClsStats { trainAcc: number; valAcc: number; singleTrain: number; singleVal: number; disagree: number }
export interface ClsEnsemble {
  N: number; mtry: number;
  trees: TreeNode[];
  /** Bootstrap draws per tree, in the order they were drawn (indices into the training set). */
  draws: Uint8Array[];
  /** Number of times each training point appears in each tree's bootstrap sample. */
  count: Uint8Array[];
  /** Votes for class 1 among the first k trees (k = 0..N) on the decision-map grid. */
  gridVotes: Uint8Array[];
  /** Hard prediction of tree i on the grid. */
  treeGrid: Uint8Array[];
  /** stats[k]: accuracy of the vote of the first k trees (k >= 1), mean single-tree accuracy, mean pairwise disagreement. */
  stats: ClsStats[];
}

/**
 * Fit N trees, each on its own bootstrap sample. `mtry` = number of features each split may look at (2 = plain bagging, 1 = random forest).
 * Bootstrap draws depend only on `seed`, so a bagged and a random ensemble with the same seed see identical samples.
 */
export function buildCls(mtry: number, N: number, seed = 7, data: ClsData = CLS_TRAIN, val: ClsData = CLS_VAL): ClsEnsemble {
  const n = data.X.length, d = 2, dr = createRng(seed), fr = createRng(seed + 1000);
  const trees: TreeNode[] = [], draws: Uint8Array[] = [], count: Uint8Array[] = [];
  const pick = () => { if (mtry >= d) return [0, 1]; const a: number[] = [0, 1]; return [a[Math.floor(fr.next() * d)]]; };
  for (let b = 0; b < N; b++) {
    const dw = new Uint8Array(n), cn = new Uint8Array(n);
    for (let j = 0; j < n; j++) { dw[j] = Math.floor(dr.next() * n); cn[dw[j]]++; }
    const Xb = Array.from(dw, (i) => data.X[i]), yb = Array.from(dw, (i) => data.y[i]);
    trees.push(mtry >= d ? fitTree(Xb, yb, { maxDepth: 12, bounds: CB }) : growRandom(Xb, yb, pick, 12));
    draws.push(dw); count.push(cn);
  }
  const preds = (pts: readonly number[][]) => trees.map((t) => Uint8Array.from(pts, (x) => predictOne(t, x)));
  const treeGrid = preds(GRID_PTS), pv = preds(val.X), pt = preds(data.X);
  const cum = (per: Uint8Array[]) => {
    const out = [new Uint8Array(per[0].length)];
    for (let k = 0; k < per.length; k++) { const a = out[k].slice(); for (let i = 0; i < a.length; i++) a[i] += per[k][i]; out.push(a); }
    return out;
  };
  const gridVotes = cum(treeGrid), vv = cum(pv), tv = cum(pt);
  const acc = (p: Uint8Array, y: readonly number[]) => { let h = 0; for (let i = 0; i < y.length; i++) if (p[i] === y[i]) h++; return h / y.length; };
  const ens = (votes: Uint8Array, k: number, y: readonly number[]) => { let h = 0; for (let i = 0; i < y.length; i++) if ((votes[i] * 2 > k ? 1 : 0) === y[i]) h++; return h / y.length; };
  const aT = pt.map((p) => acc(p, data.y)), aV = pv.map((p) => acc(p, val.y));
  const stats: ClsStats[] = [{ trainAcc: NaN, valAcc: NaN, singleTrain: NaN, singleVal: NaN, disagree: NaN }];
  let pairSum = 0;
  for (let k = 1; k <= N; k++) {
    for (let i = 0; i < k - 1; i++) { let diff = 0; for (let q = 0; q < val.y.length; q++) if (pv[i][q] !== pv[k - 1][q]) diff++; pairSum += diff / val.y.length; }
    const mean = (a: number[]) => a.slice(0, k).reduce((s, v) => s + v, 0) / k;
    stats.push({ trainAcc: ens(tv[k], k, data.y), valAcc: ens(vv[k], k, val.y), singleTrain: mean(aT), singleVal: mean(aV), disagree: k > 1 ? pairSum / ((k * (k - 1)) / 2) : NaN });
  }
  return { N, mtry, trees, draws, count, gridVotes, treeGrid, stats };
}

export interface ClsTiming { per: number; draw: number; fit: number }
/** Bagging mode shows the bootstrap draw and the tree appearing; forest mode just adds trees. */
export const BAG_TIMING: ClsTiming = { per: 2.6, draw: 1.0, fit: 0.7 };
export const FOREST_TIMING: ClsTiming = { per: 0.9, draw: 0, fit: 0 };
export const BAG_N = 8, FOREST_N = 16;

export interface ClsFrame {
  /** Tree in progress (0-based) and its stage. */
  m: number; stage: 'draw' | 'fit' | 'merge' | 'done';
  /** Progress through the bootstrap draw (0..1), through the fit stage (0..1), and the eased merge into the ensemble (0..1). */
  draw: number; fit: number; merge: number;
  /** Trees counted in the readouts: those merged, or more than halfway through merging. */
  k: number;
  /** Draws revealed so far and how many times each training point has been picked. */
  picked: number;
}
export function clsFrame(t: number, n: number, tm: ClsTiming, N: number): ClsFrame {
  const T = N * tm.per;
  t = Math.min(T, Math.max(0, t));
  if (t >= T) return { m: N - 1, stage: 'done', draw: 1, fit: 1, merge: 1, k: N, picked: n };
  const m = Math.floor(t / tm.per), r = t - m * tm.per, mergeAt = tm.draw + tm.fit, mu = (r - mergeAt) / (tm.per - mergeAt);
  const stage = r < tm.draw ? 'draw' : r < mergeAt ? 'fit' : 'merge';
  const draw = tm.draw ? clamp01(r / tm.draw) : 1;
  return { m, stage, draw, fit: tm.fit ? clamp01((r - tm.draw) / tm.fit) : 1, merge: ease(mu), k: m + (stage === 'merge' && mu >= 0.5 ? 1 : 0), picked: Math.floor(draw * n + 1e-9) };
}
/** How many times each training point has been drawn among the first `picked` draws of tree m. */
export function picksAt(e: ClsEnsemble, m: number, picked: number, out: Uint8Array): Uint8Array {
  out.fill(0);
  const dw = e.draws[m];
  for (let j = 0; j < picked; j++) out[dw[j]]++;
  return out;
}
export function clsMarkers(tm: ClsTiming, N: number, first: string, last: string): { at: number; label: string }[] {
  return [{ at: 0, label: first }, { at: (N - 1) * tm.per, label: last }];
}

// ================================================================ regression: a tiny tree, boosting, forest

export interface RegData { x: number[]; y: number[] }
/** The curve the models try to recover. */
export const truth = (x: number) => Math.sin(2 * Math.PI * x) + 0.5 * x;
export const NOISE = 0.3;
export function makeReg(seed: number, n: number, noise = NOISE): RegData {
  const r = createRng(seed), x: number[] = [], y: number[] = [];
  for (let i = 0; i < n; i++) { const xi = r.uniform(0.02, 0.98); x.push(xi); y.push(truth(xi) + r.normal(0, noise)); }
  return { x, y };
}
export const REG_TRAIN = makeReg(31, 40);
export const REG_VAL = makeReg(32, 300);
/** Vertical range of the regression plots (covers every training point). */
export const Y_LO = -1.6, Y_HI = 1.6;
export const NG = 121;
export const GRID_X: number[] = Array.from({ length: NG }, (_, i) => i / (NG - 1));

export interface RegNode { value: number; n: number; thr?: number; left?: RegNode; right?: RegNode }
/** Regression tree on one feature: each split minimises the squared error of the two child means. */
export function fitReg(x: readonly number[], r: ArrayLike<number>, maxDepth: number, minLeaf = 1): RegNode {
  const ord = x.map((_, i) => i).sort((a, b) => x[a] - x[b]), xs = ord.map((i) => x[i]), rs = ord.map((i) => r[i]);
  const grow = (lo: number, hi: number, depth: number): RegNode => {
    const n = hi - lo;
    let S = 0;
    for (let k = lo; k < hi; k++) S += rs[k];
    const node: RegNode = { value: S / n, n };
    if (depth >= maxDepth || n < 2 * minLeaf) return node;
    let best = -1, bestGain = 1e-12, run = 0;
    for (let k = lo; k < hi - 1; k++) {
      run += rs[k];
      const nl = k - lo + 1, nr = n - nl;
      if (xs[k] === xs[k + 1] || nl < minLeaf || nr < minLeaf) continue;
      const gain = (run * run) / nl + ((S - run) * (S - run)) / nr - (S * S) / n;
      if (gain > bestGain) { bestGain = gain; best = k; }
    }
    if (best < 0) return node;
    node.thr = (xs[best] + xs[best + 1]) / 2;
    node.left = grow(lo, best + 1, depth + 1); node.right = grow(best + 1, hi, depth + 1);
    return node;
  };
  return grow(0, x.length, 0);
}
export function predictReg(t: RegNode, x: number): number {
  let n = t;
  while (n.left && n.right) n = x <= n.thr! ? n.left : n.right;
  return n.value;
}
/** The pieces of the step function a regression tree defines on [lo, hi]. */
export function leavesReg(t: RegNode, lo = 0, hi = 1): { lo: number; hi: number; v: number }[] {
  if (!t.left || !t.right) return [{ lo, hi, v: t.value }];
  return [...leavesReg(t.left, lo, t.thr!), ...leavesReg(t.right, t.thr!, hi)];
}

export type Loss = 'sq' | 'abs';
/** Loss of predictions p against targets y: mean of (y - p)^2 / 2 or of |y - p|. The squared loss drops the 1/2 in the readout (MSE). */
export function lossOf(y: readonly number[], p: ArrayLike<number>, loss: Loss): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) { const e = y[i] - p[i]; s += loss === 'sq' ? e * e : Math.abs(e); }
  return s / y.length;
}
/** Negative gradient of the loss with respect to the prediction F_i. Squared loss L = (y - F)^2 / 2 gives y - F; absolute loss gives sign(y - F). */
export const pseudoResidual = (y: number, F: number, loss: Loss) => (loss === 'sq' ? y - F : Math.sign(y - F));

export interface BoostOpts { lr: number; depth: number; rounds: number; loss: Loss; minLeaf?: number }
export interface BoostRun {
  opts: BoostOpts; data: RegData; R: number;
  /** F[m]: prediction after m rounds at the training points (m = 0..R). H[m]: the update added in round m + 1 (learning rate times the tree). */
  F: Float64Array[]; H: Float64Array[];
  /** Pseudo-residuals at F[m] (m = 0..R): G[m] is what round m + 1 fits, G[R] is what is left at the end. */
  G: Float64Array[];
  /** Same on the grid (Fg) and the tree's raw output on the grid (hg, before the learning rate). */
  Fg: Float64Array[]; Hg: Float64Array[]; hg: Float64Array[];
  trees: RegNode[];
  lossTrain: number[]; lossVal: number[];
}
/** Gradient boosting: F_0 = mean (or median) of y; each round fits a depth-`depth` tree to the pseudo-residuals and adds lr times it. */
export function boostRun(o: BoostOpts, data: RegData = REG_TRAIN, val: RegData = REG_VAL, withGrid = true): BoostRun {
  const n = data.x.length, R = o.rounds, minLeaf = o.minLeaf ?? 1;
  const med = [...data.y].sort((a, b) => a - b)[Math.floor(n / 2)];
  const f0 = o.loss === 'sq' ? data.y.reduce((s, v) => s + v, 0) / n : med;
  let F = new Float64Array(n).fill(f0), Fv = new Float64Array(val.x.length).fill(f0), Fg = new Float64Array(withGrid ? NG : 0).fill(f0);
  const run: BoostRun = { opts: o, data, R, F: [F], H: [], G: [], Fg: [Fg], Hg: [], hg: [], trees: [], lossTrain: [lossOf(data.y, F, o.loss)], lossVal: [lossOf(val.y, Fv, o.loss)] };
  for (let m = 0; m < R; m++) {
    const g = Float64Array.from(data.y, (yi, i) => pseudoResidual(yi, F[i], o.loss));
    const tree = fitReg(data.x, g, o.depth, minLeaf);
    const H = Float64Array.from(data.x, (xi) => o.lr * predictReg(tree, xi));
    const Hv = Float64Array.from(val.x, (xi) => o.lr * predictReg(tree, xi));
    F = F.map((v, i) => v + H[i]); Fv = Fv.map((v, i) => v + Hv[i]);
    run.G.push(g); run.H.push(H); run.trees.push(tree); run.F.push(F);
    run.lossTrain.push(lossOf(data.y, F, o.loss)); run.lossVal.push(lossOf(val.y, Fv, o.loss));
    if (withGrid) {
      const hg = Float64Array.from(GRID_X, (xi) => predictReg(tree, xi));
      run.hg.push(hg); run.Hg.push(hg.map((v) => v * o.lr)); Fg = Fg.map((v, i) => v + o.lr * hg[i]); run.Fg.push(Fg);
    }
  }
  run.G.push(Float64Array.from(data.y, (yi, i) => pseudoResidual(yi, F[i], o.loss)));
  return run;
}

/** Timeline of a boosting run: the first three rounds are slow so each stage can be followed, then the rounds speed up. */
export const SLOW_ROUNDS = 3, P_SLOW = 1.8, P_FAST = 0.7;
export const roundStart = (m: number) => (m <= SLOW_ROUNDS ? m * P_SLOW : SLOW_ROUNDS * P_SLOW + (m - SLOW_ROUNDS) * P_FAST);
export const boostT = (R: number) => roundStart(R);
export interface BoostFrame {
  /** Round in progress (0-based; R when finished). */
  m: number; stage: 'residual' | 'fit' | 'update' | 'done';
  /** Progress through the stage (0..1) and the eased update tween (0 before the update stage, 1 after). */
  a: number; w: number;
}
export function boostFrame(t: number, R: number): BoostFrame {
  t = Math.min(boostT(R), Math.max(0, t));
  if (t >= boostT(R)) return { m: R, stage: 'done', a: 1, w: 0 };
  let m = 0;
  while (m + 1 < R && roundStart(m + 1) <= t) m++;
  const u = (t - roundStart(m)) / (roundStart(m + 1) - roundStart(m));
  if (u < 0.3) return { m, stage: 'residual', a: u / 0.3, w: 0 };
  if (u < 0.6) return { m, stage: 'fit', a: (u - 0.3) / 0.3, w: 0 };
  return { m, stage: 'update', a: (u - 0.6) / 0.4, w: ease((u - 0.6) / 0.4) };
}
const mix = (a: Float64Array, b: Float64Array | undefined, w: number, out: Float64Array) => { for (let i = 0; i < a.length; i++) out[i] = b ? a[i] + w * b[i] : a[i]; return out; };
/** Prediction at the training points / on the grid at this frame (the update stage slides F_m toward F_{m+1}). */
export const predTrain = (r: BoostRun, f: BoostFrame, out: Float64Array) => mix(r.F[f.m], r.H[f.m], f.w, out);
export const predGrid = (r: BoostRun, f: BoostFrame, out: Float64Array) => mix(r.Fg[f.m], r.Hg[f.m], f.w, out);
/** Loss now: interpolated between the two rounds during the update stage. */
export function lossNow(r: BoostRun, f: BoostFrame): { train: number; val: number } {
  const m2 = Math.min(f.m + 1, r.R);
  return { train: r.lossTrain[f.m] + f.w * (r.lossTrain[m2] - r.lossTrain[f.m]), val: r.lossVal[f.m] + f.w * (r.lossVal[m2] - r.lossVal[f.m]) };
}
export function boostMarkers(R: number): { at: number; label: string }[] {
  return [{ at: 0, label: `Rounds 1 to ${SLOW_ROUNDS}` }, { at: roundStart(SLOW_ROUNDS), label: `Rounds ${SLOW_ROUNDS + 1} to ${R}` }];
}

// ================================================================ random forest vs boosting on the regression curve

export const CMP_N = 100, T_CMP = 16;
export const LR_HIGH = 0.5, LR_LOW = 0.05, CMP_DEPTH = 3;
export interface Compare {
  N: number;
  rfVal: number[]; rfTrain: number[]; hiVal: number[]; hiTrain: number[]; loVal: number[]; loTrain: number[];
  /** Index (number of trees) of the lowest validation error per method. */
  rfBest: number; hiBest: number; loBest: number;
  markers: { at: number; label: string }[];
}
/** Random forest of deep trees on bootstrap samples, against boosting with a high and a low learning rate. Entry k = error with k trees (k = 0 is the constant model). */
export function buildCompare(seed = 5, N = CMP_N, data: RegData = REG_TRAIN, val: RegData = REG_VAL): Compare {
  const rng = createRng(seed), n = data.x.length;
  const sumT = new Float64Array(n), sumV = new Float64Array(val.x.length);
  const f0 = data.y.reduce((s, v) => s + v, 0) / n;
  const rfVal = [lossOf(val.y, new Float64Array(val.x.length).fill(f0), 'sq')], rfTrain = [lossOf(data.y, new Float64Array(n).fill(f0), 'sq')];
  for (let b = 1; b <= N; b++) {
    const ix = Array.from({ length: n }, () => Math.floor(rng.next() * n));
    const tree = fitReg(ix.map((i) => data.x[i]), ix.map((i) => data.y[i]), 12, 1);
    for (let i = 0; i < n; i++) sumT[i] += predictReg(tree, data.x[i]);
    for (let i = 0; i < val.x.length; i++) sumV[i] += predictReg(tree, val.x[i]);
    rfTrain.push(lossOf(data.y, sumT.map((v) => v / b), 'sq')); rfVal.push(lossOf(val.y, sumV.map((v) => v / b), 'sq'));
  }
  const hi = boostRun({ lr: LR_HIGH, depth: CMP_DEPTH, rounds: N, loss: 'sq' }, data, val, false);
  const lo = boostRun({ lr: LR_LOW, depth: CMP_DEPTH, rounds: N, loss: 'sq' }, data, val, false);
  const argmin = (a: number[]) => a.indexOf(Math.min(...a));
  const rfBest = argmin(rfVal.slice(1)) + 1, hiBest = argmin(hi.lossVal), loBest = argmin(lo.lossVal);
  const at = (k: number) => (k * T_CMP) / N;
  const markers = [{ at: at(hiBest), label: 'Fast rate gets worse' }, { at: at(loBest), label: 'Slow rate has caught up' }].filter((m, i, a) => a.findIndex((q) => q.at === m.at) === i);
  return { N, rfVal, rfTrain, hiVal: hi.lossVal, hiTrain: hi.lossTrain, loVal: lo.lossVal, loTrain: lo.lossTrain, rfBest, hiBest, loBest, markers };
}
/** Number of trees shown at time t (0..N). */
export const cmpCount = (t: number, N = CMP_N) => Math.round(clamp01(t / T_CMP) * N);
