/** Space folding: a hidden layer bends the plane. Pure logic, no DOM. Training uses the shared autograd engine. */
import { MLP, binaryCrossEntropy, backward, adam, stepParams, createRng } from '../../../lib/nn';

export interface Data { xs: Float64Array; ys: Float64Array; n: number }
export type DataKind = 'spirals' | 'xor';

/**
 * Two classes, `perClass` points each, inside [-1.2, 1.2]².
 * spirals: two interleaved arms of `turns` turns, offset by half a turn. xor: four blobs; class 1 on the diagonal y = x, class 0 on y = -x.
 */
export function makeData(kind: DataKind, perClass: number, seed: number, turns = 0.85, noise = 0.03): Data {
  const rng = createRng(seed), n = 2 * perClass, xs = new Float64Array(2 * n), ys = new Float64Array(n);
  for (let k = 0; k < 2; k++) {
    for (let i = 0; i < perClass; i++) {
      const u = (i + 0.5) / perClass, j = k * perClass + i;
      let x: number, y: number, sd = noise;
      if (kind === 'spirals') {
        const r = 0.12 + 0.88 * u, a = turns * 2 * Math.PI * u + k * Math.PI;
        x = r * Math.cos(a); y = r * Math.sin(a);
      } else {
        x = 0.55 * (i % 2 ? 1 : -1); y = k ? x : -x; sd = 0.17;
      }
      xs[2 * j] = x + rng.normal(0, sd); xs[2 * j + 1] = y + rng.normal(0, sd); ys[j] = k;
    }
  }
  return { xs, ys, n };
}

/** Layer sizes including input (2) and output (1), e.g. [2, 2, 1]. */
export type Sizes = readonly number[];
export const paramCount = (s: Sizes) => s.slice(1).reduce((a, n, i) => a + n * (s[i] + 1), 0);

export interface Run { sizes: Sizes; epochs: number[]; weights: Float64Array[] }

/** Snapshot epochs: dense early (where the action is), sparse late. Always starts at 0 and ends at `epochs`. */
export function snapEpochs(epochs: number, count: number): number[] {
  const out = new Set<number>();
  for (let k = 0; k < count; k++) out.add(Math.round(epochs * (k / (count - 1)) ** 2));
  return [...out].sort((a, b) => a - b);
}

/** Full-batch Adam on binary cross-entropy with the shared engine. Deterministic from the seed. Weight layout matches `MLP.parameters()`. */
export function trainRun(sizes: Sizes, data: Data, o: { seed: number; epochs: number; lr?: number; snaps?: number }): Run {
  const net = new MLP(sizes[0], sizes.slice(1) as number[], { rng: createRng(o.seed), act: 'tanh', outAct: 'sigmoid' });
  const params = net.parameters(), opt = adam(o.lr ?? 0.03), eps = snapEpochs(o.epochs, o.snaps ?? 48), weights: Float64Array[] = [];
  let next = 0;
  for (let e = 0; e <= o.epochs; e++) {
    if (e === eps[next]) { weights.push(Float64Array.from(params, (p) => p.data)); next++; }
    if (e === o.epochs) break;
    const preds = [];
    for (let i = 0; i < data.n; i++) preds.push(net.call([data.xs[2 * i], data.xs[2 * i + 1]])[0]);
    backward(binaryCrossEntropy(preds, Array.from(data.ys)));
    stepParams(opt, params);
  }
  return { sizes, epochs: eps, weights };
}

/** Plain forward pass. Returns the activation vector of every layer, input first. `pre[l-1]` is Wx+b of layer l (before tanh / sigmoid). */
export function forward(sizes: Sizes, w: ArrayLike<number>, x: number, y: number): { pre: number[][]; act: number[][] } {
  const act: number[][] = [[x, y]], pre: number[][] = [];
  let o = 0;
  for (let l = 1; l < sizes.length; l++) {
    const a = act[l - 1], p: number[] = [], out: number[] = [], last = l === sizes.length - 1;
    for (let j = 0; j < sizes[l]; j++) {
      let s = 0;
      for (let i = 0; i < sizes[l - 1]; i++) s += w[o++] * a[i];
      s += w[o++];
      p.push(s); out.push(last ? 1 / (1 + Math.exp(-s)) : Math.tanh(s));
    }
    pre.push(p); act.push(out);
  }
  return { pre, act };
}

export function evaluate(sizes: Sizes, w: ArrayLike<number>, d: Data): { loss: number; acc: number; correct: number } {
  let loss = 0, ok = 0;
  for (let i = 0; i < d.n; i++) {
    const p = forward(sizes, w, d.xs[2 * i], d.xs[2 * i + 1]).act.at(-1)![0];
    loss -= d.ys[i] ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12);
    ok += (p >= 0.5 ? 1 : 0) === d.ys[i] ? 1 : 0;
  }
  return { loss: loss / d.n, acc: ok / d.n, correct: ok };
}

// ---------------------------------------------------------------------------------------------------
// The runs the widget shows. Seeds are fixed, so every reload draws the same pictures.
// ---------------------------------------------------------------------------------------------------
export type RunId = 'line' | 'xor' | 'features' | 'one' | 'deep';
export type Mode = 'spirals' | 'fold' | 'features' | 'depth';
interface Cfg { data: { kind: DataKind; perClass: number; seed: number }; sizes: Sizes; seed: number; epochs: number; lr: number }
const SPIRALS = { kind: 'spirals', perClass: 40, seed: 7 } as const;
export const CFG: Record<RunId, Cfg> = {
  line: { data: SPIRALS, sizes: [2, 1], seed: 1, epochs: 400, lr: 0.04 },
  xor: { data: { kind: 'xor', perClass: 32, seed: 3 }, sizes: [2, 2, 1], seed: 2, epochs: 800, lr: 0.05 },
  features: { data: SPIRALS, sizes: [2, 6, 1], seed: 1, epochs: 1200, lr: 0.04 },
  one: { data: SPIRALS, sizes: [2, 2, 1], seed: 1, epochs: 1500, lr: 0.04 },
  deep: { data: SPIRALS, sizes: [2, 2, 2, 2, 1], seed: 1, epochs: 1500, lr: 0.04 },
};
export const RUNS_FOR: Record<Mode, RunId[]> = { spirals: ['line'], fold: ['xor'], features: ['features'], depth: ['one', 'deep'] };
export const dataOf = (id: RunId) => makeData(CFG[id].data.kind, CFG[id].data.perClass, CFG[id].data.seed);
/** Train one configured run (runs in the worker). */
export const compute = (id: RunId, epochs = CFG[id].epochs): Run =>
  trainRun(CFG[id].sizes, dataOf(id), { seed: CFG[id].seed, epochs, lr: CFG[id].lr, snaps: id === 'line' ? 40 : 48 });

export const smooth = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };
export const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

// ---- training progress (modes spirals and features): t picks a point between two snapshots --------
export const FIT_T = 8, FIT_HOLD = 1.2;
export interface FitFrame { w: Float64Array; epoch: number; loss: number; acc: number; correct: number }
/** Weights at fractional snapshot index `s`: a straight blend of the two neighbours. */
export function weightsAt(run: Run, s: number): { w: Float64Array; epoch: number } {
  if (run.weights.length === 1) return { w: run.weights[0].slice(), epoch: run.epochs[0] };
  const last = run.weights.length - 1, c = clamp(s, 0, last), i = Math.min(last - 1, Math.floor(c)), f = c - i;
  const a = run.weights[i], b = run.weights[i + 1], w = new Float64Array(a.length);
  for (let k = 0; k < w.length; k++) w[k] = a[k] + (b[k] - a[k]) * f;
  return { w, epoch: Math.round(run.epochs[i] + (run.epochs[i + 1] - run.epochs[i]) * f) };
}
export function fitFrame(run: Run, data: Data, t: number): FitFrame {
  const s = (run.weights.length - 1) * clamp(t / (FIT_T - FIT_HOLD), 0, 1), { w, epoch } = weightsAt(run, s), e = evaluate(run.sizes, w, data);
  return { w, epoch, loss: e.loss, acc: e.acc, correct: e.correct };
}
/** Accuracy at every snapshot, for the plateau curve. */
export const accCurve = (run: Run, data: Data) => run.weights.map((w, i) => ({ epoch: run.epochs[i], acc: evaluate(run.sizes, w, data).acc }));

export const LINE_MARKERS = [{ at: 0, label: 'Random line' }, { at: 1.4, label: 'Line swings into place' }, { at: 3.6, label: 'Stuck' }];
export const FEATURE_MARKERS = [{ at: 0, label: 'Random weights' }, { at: 1, label: 'Ramps form' }, { at: 3, label: 'Ramps combine' }, { at: 5.5, label: 'Trained' }];

// ---- fields over the input plane ----------------------------------------------------------------------
export const LIM = 1.25;
export interface Field { res: number; units: Float32Array[]; logit: Float32Array; prob: Float32Array }
/** First-hidden-layer activations and the output over a res×res grid of [-LIM, LIM]². Row j is y, column i is x, j = 0 at the bottom. */
export function field(sizes: Sizes, w: ArrayLike<number>, res: number): Field {
  const units = Array.from({ length: sizes.length > 2 ? sizes[1] : 0 }, () => new Float32Array(res * res)), logit = new Float32Array(res * res), prob = new Float32Array(res * res);
  for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
    const f = forward(sizes, w, -LIM + (2 * LIM * i) / (res - 1), -LIM + (2 * LIM * j) / (res - 1)), k = j * res + i;
    units.forEach((u, q) => { u[k] = f.act[1][q]; });
    logit[k] = f.pre.at(-1)![0]; prob[k] = f.act.at(-1)![0];
  }
  return { res, units, logit, prob };
}

// ---- folding: the data and a grid pushed through the layers, one stage at a time ----------------------
export interface Scene { P: Float64Array; n: number; lines: { start: number; count: number; axis: boolean }[] }
/** Data points first, then grid polylines (x = const and y = const lines, spacing 0.25). */
export function makeScene(data: Data, seg = 24): Scene {
  const pts: number[] = Array.from(data.xs), lines: Scene['lines'] = [];
  for (const vertical of [true, false]) for (let g = -5; g <= 5; g++) {
    const c = g * 0.25, start = pts.length / 2;
    for (let q = 0; q <= seg; q++) { const v = -LIM + (2 * LIM * q) / seg; pts.push(vertical ? c : v, vertical ? v : c); }
    lines.push({ start, count: seg + 1, axis: g === 0 });
  }
  return { P: Float64Array.from(pts), n: data.n, lines };
}
/** Positions after each step of the hidden layers: [input, Wx+b of layer 1, tanh of layer 1, Wx+b of layer 2, ...]. Hidden layers must be 2 wide. */
export function buildStages(sizes: Sizes, w: ArrayLike<number>, P: Float64Array): Float64Array[] {
  const out = [P];
  let cur = P, o = 0;
  for (let l = 1; l < sizes.length - 1; l++) {
    const pre = new Float64Array(cur.length), act = new Float64Array(cur.length);
    for (let q = 0; q < cur.length; q += 2) {
      for (let j = 0; j < 2; j++) {
        const v = w[o + 3 * j] * cur[q] + w[o + 3 * j + 1] * cur[q + 1] + w[o + 3 * j + 2];
        pre[q + j] = v; act[q + j] = Math.tanh(v);
      }
    }
    o += 6; out.push(pre, act); cur = act;
  }
  return out;
}
export interface View { cx: number; cy: number; h: number }
/** Square window that fits the data points of a stage. */
export function viewOf(S: Float64Array, n: number): View {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) { x0 = Math.min(x0, S[2 * i]); x1 = Math.max(x1, S[2 * i]); y0 = Math.min(y0, S[2 * i + 1]); y1 = Math.max(y1, S[2 * i + 1]); }
  return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, h: (Math.max(x1 - x0, y1 - y0, 0.5) / 2) * 1.2 };
}
export interface Fold { sizes: Sizes; scene: Scene; stages: Float64Array[]; views: View[]; cut: [number, number, number]; labels: Uint8Array }
export function makeFold(sizes: Sizes, w: ArrayLike<number>, data: Data): Fold {
  const scene = makeScene(data), stages = buildStages(sizes, w, scene.P), k = w.length;
  return { sizes, scene, stages, views: stages.map((S) => viewOf(S, S.length / 2)), cut: [w[k - 3], w[k - 2], w[k - 1]], labels: Uint8Array.from(data.ys) };
}
export interface FoldTiming { t0: number; affine: number; squash: number }
export const FOLD_TIMING: FoldTiming = { t0: 1.5, affine: 3, squash: 3 };
export const DEPTH_TIMING: FoldTiming = { t0: 1.5, affine: 1.6, squash: 1.4 };
export const FOLD_T = 10, DEPTH_T = 12;
export const FOLD_MARKERS = [{ at: 0, label: 'Input' }, { at: 1.5, label: 'Affine' }, { at: 4.5, label: 'Tanh' }, { at: 7.5, label: 'Cut' }];
export const DEPTH_MARKERS = [{ at: 0, label: 'Input' }, { at: 1.5, label: 'Layer 1' }, { at: 4.5, label: 'Layer 2' }, { at: 7.5, label: 'Layer 3' }, { at: 10.5, label: 'Cut' }];

/** Which pair of stages the space is between at time t, and how far along (eased). m = 2*layer + part; part 0 is Wx+b, part 1 is tanh. */
export function stagePos(t: number, layers: number, tm: FoldTiming): { m: number; s: number } {
  let at = tm.t0;
  for (let m = 0; m < 2 * layers; m++) {
    const len = m % 2 ? tm.squash : tm.affine;
    if (t < at + len) return { m, s: t <= at ? 0 : smooth((t - at) / len) };
    at += len;
  }
  return { m: 2 * layers, s: 0 };
}
export interface FoldFrame { pos: Float64Array; view: View; m: number; s: number; right: number; wrong: Uint8Array; nWrong: number }
/** Everything to draw one morphing panel at time t. `wrong[i]` marks data points on the wrong side of the final cut as drawn. */
export function foldFrame(f: Fold, t: number, tm: FoldTiming): FoldFrame {
  const layers = f.stages.length >> 1, { m, s } = stagePos(t, layers, tm), m1 = Math.min(m + 1, f.stages.length - 1), A = f.stages[m], B = f.stages[m1];
  const pos = new Float64Array(A.length);
  for (let i = 0; i < pos.length; i++) pos[i] = A[i] + (B[i] - A[i]) * s;
  const va = f.views[m], vb = f.views[m1];
  const view = { cx: va.cx + (vb.cx - va.cx) * s, cy: va.cy + (vb.cy - va.cy) * s, h: va.h + (vb.h - va.h) * s };
  const wrong = new Uint8Array(f.scene.n), [a, b, c] = f.cut;
  let nWrong = 0;
  for (let i = 0; i < f.scene.n; i++) { const z = a * pos[2 * i] + b * pos[2 * i + 1] + c; wrong[i] = (z >= 0 ? 1 : 0) === f.labels[i] ? 0 : 1; nWrong += wrong[i]; }
  return { pos, view, m, s, right: f.scene.n - nWrong, wrong, nWrong };
}
