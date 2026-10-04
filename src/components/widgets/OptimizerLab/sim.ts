import { adam, momentum, rmsprop, sgd, grad2D } from '../../../lib/nn/optim';
import { Value } from '../../../lib/nn/value';
import { createRng } from '../../../lib/math/rng';

/** A small deterministic optimizer sandbox. All optimizer math is kept pure here. */
export const OPTIMIZERS = ['SGD', 'Momentum', 'RMSProp', 'Adam'] as const;
export type Optimizer = (typeof OPTIMIZERS)[number];
export type Landscape = 'ravine' | 'saddle';
export type Mode = 'update' | 'noise' | 'momentum' | 'scaling' | 'bias' | 'race';

export const DATA_SEED = 24061;
export const EXAMPLE_COUNT = 16;
export const STEPS = 120;
export const DURATION = 12;
export const DOMAIN = 3;
export const DEFAULTS = { lr: 0.08, batchSize: 4, noise: 1, beta: 0.9, alpha: 0.9, beta1: 0.9, beta2: 0.999, eps: 1e-8 };

export interface Point { x: number; y: number }
export interface Example { cx: number; cy: number }
export interface Config {
  landscape: Landscape;
  lr: number;
  batchSize: number;
  noise: number;
  seed?: number;
}
export interface OptState {
  point: Point;
  momentum: Point;
  square: Point;
  first: Point;
  second: Point;
  step: number;
  gradient: Point;
  samples: number[];
  diverged: boolean;
}
export interface Run { optimizer: Optimizer; states: OptState[] }
export interface Frame {
  t: number;
  step: number;
  point: Point;
  gradient: Point;
  loss: number;
  momentum: Point;
  square: Point;
  first: Point;
  second: Point;
  samples: number[];
  diverged: boolean;
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/** Zero-mean paired offsets make a full batch equal the exact objective gradient. */
export function makeDataset(seed = DATA_SEED, count = EXAMPLE_COUNT): Example[] {
  if (count < 2 || count % 2 !== 0) throw new Error('Dataset size must be a positive even number.');
  const random = createRng(seed);
  const data: Example[] = [];
  for (let i = 0; i < count / 2; i++) {
    const angle = random.uniform(0, Math.PI * 2);
    const radius = random.uniform(0.25, 1);
    const cx = Math.cos(angle) * radius, cy = Math.sin(angle) * radius;
    data.push({ cx, cy }, { cx: -cx, cy: -cy });
  }
  return data;
}

function eigensystem(landscape: Landscape): { lo: number; hi: number; angle: number } {
  return landscape === 'ravine' ? { lo: 0.22, hi: 3.4, angle: Math.PI / 5 } : { lo: -0.6, hi: 1.8, angle: Math.PI / 6 };
}

/** H·v for the rotated quadratic. Saddle has one negative-curvature direction. */
export function hessianVector(point: Point, landscape: Landscape): Point {
  const { lo, hi, angle } = eigensystem(landscape);
  const c = Math.cos(angle), s = Math.sin(angle);
  const u = c * point.x + s * point.y;
  const v = -s * point.x + c * point.y;
  return { x: c * lo * u - s * hi * v, y: s * lo * u + c * hi * v };
}

export function objective(point: Point, landscape: Landscape): number {
  const hv = hessianVector(point, landscape);
  return 1 + 0.5 * (point.x * hv.x + point.y * hv.y);
}

function exampleObjective(x: Value, y: Value, landscape: Landscape, cx: number, cy: number): Value {
  const { lo, hi, angle } = eigensystem(landscape);
  const c = Math.cos(angle), s = Math.sin(angle);
  const dx = x.sub(cx), dy = y.sub(cy);
  const u = dx.mul(c).add(dy.mul(s)), v = dy.mul(c).sub(dx.mul(s));
  return u.pow(2).mul(lo).add(v.pow(2).mul(hi)).mul(0.5).add(1);
}

function batchObjective(landscape: Landscape, examples: Example[], indices: number[], noise: number) {
  return (x: Value, y: Value): Value => {
    let total: Value | null = null;
    for (const index of indices) {
      const ex = examples[index];
      const value = exampleObjective(x, y, landscape, ex.cx * noise, ex.cy * noise);
      total = total ? total.add(value) : value;
    }
    return total!.div(indices.length);
  };
}

export function gradient(point: Point, landscape: Landscape, examples = makeDataset(), indices?: number[], noise = 1): Point {
  const exact = hessianVector(point, landscape);
  const picked = indices ?? examples.map((_, i) => i);
  let nx = 0, ny = 0;
  for (const index of picked) {
    const ex = examples[index];
    const sample = hessianVector({ x: point.x - ex.cx, y: point.y - ex.cy }, landscape);
    nx += sample.x;
    ny += sample.y;
  }
  return { x: exact.x + noise * (nx / picked.length - exact.x), y: exact.y + noise * (ny / picked.length - exact.y) };
}

function sampleIndices(batchSize: number, count: number, random: () => number): number[] {
  if (batchSize >= count) return Array.from({ length: count }, (_, i) => i);
  const pool = Array.from({ length: count }, (_, i) => i);
  // A seeded partial Fisher-Yates shuffle gives a fresh without-replacement batch each step.
  for (let i = 0; i < batchSize; i++) {
    const j = i + Math.floor(random() * (count - i));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, batchSize);
}

export function createOptimizerState(point: Point): OptState {
  const zero = (): Point => ({ x: 0, y: 0 });
  return { point: { ...point }, momentum: zero(), square: zero(), first: zero(), second: zero(), step: 0, gradient: zero(), samples: [], diverged: false };
}

export function startPoint(landscape: Landscape): Point {
  return landscape === 'ravine' ? { x: -2.1, y: 2.1 } : { x: 0.25, y: 0.7 };
}

export function stepOptimizerState(
  state: OptState,
  optimizer: Optimizer,
  g: Point,
  lr: number,
  samples: number[],
): OptState {
  const next = { ...state, point: { ...state.point }, momentum: { ...state.momentum }, square: { ...state.square }, first: { ...state.first }, second: { ...state.second }, step: state.step + 1, gradient: g, samples, diverged: false };
  const b = DEFAULTS;
  if (optimizer === 'Momentum') {
    next.momentum = { x: b.beta * state.momentum.x + g.x, y: b.beta * state.momentum.y + g.y };
  } else if (optimizer === 'RMSProp') {
    next.square = { x: b.alpha * state.square.x + (1 - b.alpha) * g.x * g.x, y: b.alpha * state.square.y + (1 - b.alpha) * g.y * g.y };
  } else {
    if (optimizer === 'Adam') {
      next.first = { x: b.beta1 * state.first.x + (1 - b.beta1) * g.x, y: b.beta1 * state.first.y + (1 - b.beta1) * g.y };
      next.second = { x: b.beta2 * state.second.x + (1 - b.beta2) * g.x * g.x, y: b.beta2 * state.second.y + (1 - b.beta2) * g.y * g.y };
    }
  }
  let dx: number, dy: number;
  if (optimizer === 'SGD') {
    dx = lr * g.x; dy = lr * g.y;
  } else if (optimizer === 'Momentum') {
    dx = lr * next.momentum.x; dy = lr * next.momentum.y;
  } else if (optimizer === 'RMSProp') {
    dx = lr * g.x / (Math.sqrt(next.square.x) + DEFAULTS.eps); dy = lr * g.y / (Math.sqrt(next.square.y) + DEFAULTS.eps);
  } else {
    const mHatX = next.first.x / (1 - DEFAULTS.beta1 ** next.step), mHatY = next.first.y / (1 - DEFAULTS.beta1 ** next.step);
    const vHatX = next.second.x / (1 - DEFAULTS.beta2 ** next.step), vHatY = next.second.y / (1 - DEFAULTS.beta2 ** next.step);
    dx = lr * mHatX / (Math.sqrt(vHatX) + DEFAULTS.eps); dy = lr * mHatY / (Math.sqrt(vHatY) + DEFAULTS.eps);
  }
  next.point = { x: state.point.x - dx, y: state.point.y - dy };
  next.diverged = !Number.isFinite(next.point.x) || !Number.isFinite(next.point.y);
  return next;
}

export function makeRun(optimizer: Optimizer, config: Config): Run {
  const examples = makeDataset(config.seed ?? DATA_SEED);
  const random = createRng((config.seed ?? DATA_SEED) + 17);
  const engineOptimizer = createEngineOptimizer(optimizer, config.lr);
  const start = startPoint(config.landscape);
  const states = [createOptimizerState(start)];
  for (let i = 0; i < STEPS; i++) {
    const prior = states[i];
    if (prior.diverged) { states.push({ ...prior, step: prior.step + 1, samples: [] }); continue; }
    const samples = sampleIndices(clamp(Math.round(config.batchSize), 1, examples.length), examples.length, random.next);
    const [, gx, gy] = grad2D(batchObjective(config.landscape, examples, samples, config.noise), prior.point.x, prior.point.y);
    const g = { x: gx, y: gy };
    const recorded = stepOptimizerState(prior, optimizer, g, config.lr, samples);
    const coordinates: [number, number] = [prior.point.x, prior.point.y];
    engineOptimizer.step(coordinates, [gx, gy]);
    if (Math.abs(recorded.point.x - coordinates[0]) > 1e-9 || Math.abs(recorded.point.y - coordinates[1]) > 1e-9) throw new Error(`${optimizer} simulation disagrees with the shared optimizer engine.`);
    states.push({ ...recorded, point: { x: coordinates[0], y: coordinates[1] } });
  }
  return { optimizer, states };
}

/** Constructs the shared engine's stateful optimizer for one recorded run. */
export function createEngineOptimizer(optimizer: Optimizer, lr = DEFAULTS.lr): ReturnType<typeof sgd> {
  if (optimizer === 'Momentum') return momentum(lr, DEFAULTS.beta);
  if (optimizer === 'RMSProp') return rmsprop(lr, DEFAULTS.alpha, DEFAULTS.eps);
  if (optimizer === 'Adam') return adam(lr, DEFAULTS.beta1, DEFAULTS.beta2, DEFAULTS.eps);
  return sgd(lr);
}

export function makeRace(config: Config): Record<Optimizer, Run> {
  return Object.fromEntries(OPTIMIZERS.map((name) => [name, makeRun(name, config)])) as Record<Optimizer, Run>;
}

export interface Box { x0: number; x1: number; y0: number; y1: number }
export interface Contour { level: number; points: [number, number, number, number] }

/** Marching-squares line segments of f over a box, one list for all levels. */
export function contourSegments(f: (x: number, y: number) => number, box: Box, levels: number[], count = 42): Contour[] {
  const out: Contour[] = [];
  for (let j = 0; j < count; j++) for (let i = 0; i < count; i++) {
    const xa = box.x0 + (box.x1 - box.x0) * i / count, xb = box.x0 + (box.x1 - box.x0) * (i + 1) / count;
    const ya = box.y0 + (box.y1 - box.y0) * j / count, yb = box.y0 + (box.y1 - box.y0) * (j + 1) / count;
    const corners = [[xa, ya, f(xa, ya)], [xb, ya, f(xb, ya)], [xb, yb, f(xb, yb)], [xa, yb, f(xa, yb)]] as const;
    for (const level of levels) {
      const hits: [number, number][] = [];
      for (let edge = 0; edge < 4; edge++) {
        const a = corners[edge], b = corners[(edge + 1) % 4];
        if ((a[2] < level) !== (b[2] < level)) {
          const p = (level - a[2]) / (b[2] - a[2]);
          hits.push([a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p]);
        }
      }
      for (let k = 0; k + 1 < hits.length; k += 2) out.push({ level, points: [hits[k][0], hits[k][1], hits[k + 1][0], hits[k + 1][1]] });
    }
  }
  return out;
}

/** Evenly spaced loss levels over a box, plus the level through `through` so its curve can be highlighted. */
export function localLevels(landscape: Landscape, box: Box, through: Point, count = 8): number[] {
  let lo = Infinity, hi = -Infinity;
  for (let j = 0; j <= 20; j++) for (let i = 0; i <= 20; i++) {
    const v = objective({ x: box.x0 + (box.x1 - box.x0) * i / 20, y: box.y0 + (box.y1 - box.y0) * j / 20 }, landscape);
    lo = Math.min(lo, v); hi = Math.max(hi, v);
  }
  const levels = Array.from({ length: count }, (_, k) => lo + (hi - lo) * (k + 0.5) / count);
  return [...levels, objective(through, landscape)];
}

function lerp(a: number, b: number, p: number) { return a + (b - a) * p; }
const mixPoint = (a: Point, b: Point, p: number): Point => ({ x: lerp(a.x, b.x, p), y: lerp(a.y, b.y, p) });

/** Read recorded optimizer states; evaluate the objective at the exact displayed point. */
export function frameAt(t: number, run: Run, landscape: Landscape): Frame {
  const time = clamp(t, 0, DURATION);
  const position = (time / DURATION) * STEPS;
  const index = Math.min(STEPS, Math.floor(position));
  const blend = position >= STEPS ? 0 : position - index;
  const a = run.states[index], b = run.states[Math.min(STEPS, index + 1)];
  const point = mixPoint(a.point, b.point, blend);
  return {
    t: time,
    step: a.step,
    point,
    gradient: { ...a.gradient },
    loss: objective(point, landscape),
    momentum: { ...a.momentum },
    square: { ...a.square },
    first: { ...a.first },
    second: { ...a.second },
    samples: [...a.samples],
    diverged: a.diverged || (blend > 0 && b.diverged),
  };
}

export function firstAdamUpdate(g: Point, lr = DEFAULTS.lr): { step: Point; first: Point; second: Point; correctedFirst: Point; correctedSecond: Point } {
  const first = { x: (1 - DEFAULTS.beta1) * g.x, y: (1 - DEFAULTS.beta1) * g.y };
  const second = { x: (1 - DEFAULTS.beta2) * g.x * g.x, y: (1 - DEFAULTS.beta2) * g.y * g.y };
  const correctedFirst = { x: first.x / (1 - DEFAULTS.beta1), y: first.y / (1 - DEFAULTS.beta1) };
  const correctedSecond = { x: second.x / (1 - DEFAULTS.beta2), y: second.y / (1 - DEFAULTS.beta2) };
  return {
    step: { x: lr * correctedFirst.x / (Math.sqrt(correctedSecond.x) + DEFAULTS.eps), y: lr * correctedFirst.y / (Math.sqrt(correctedSecond.y) + DEFAULTS.eps) },
    first, second, correctedFirst, correctedSecond,
  };
}

/** One recorded update, stretched into a read / transform / move explanation.
 * Memory is a completed calculation; only the position is interpolated. */
export function teachingFrameAt(t: number, run: Run, config: Config, update = 1) {
  const time = clamp(t, 0, DURATION);
  const index = clamp(Math.round(update), 1, run.states.length - 1);
  const before = run.states[index - 1], after = run.states[index];
  const progress = clamp((time - 7) / 3, 0, 1);
  const point = mixPoint(before.point, after.point, progress);
  const g = after.gradient;
  const correction1 = 1 - DEFAULTS.beta1 ** index, correction2 = 1 - DEFAULTS.beta2 ** index;
  const numerator = run.optimizer === 'Momentum' ? after.momentum
    : run.optimizer === 'Adam' ? { x: after.first.x / correction1, y: after.first.y / correction1 } : g;
  const denominator = run.optimizer === 'RMSProp'
    ? { x: Math.sqrt(after.square.x) + DEFAULTS.eps, y: Math.sqrt(after.square.y) + DEFAULTS.eps }
    : run.optimizer === 'Adam'
      ? { x: Math.sqrt(after.second.x / correction2) + DEFAULTS.eps, y: Math.sqrt(after.second.y / correction2) + DEFAULTS.eps }
      : { x: 1, y: 1 };
  const data = makeDataset(config.seed ?? DATA_SEED);
  return {
    time, index, before, after, point, progress, numerator, denominator, correction1, correction2,
    phase: time < 3 ? 'Read the slope' : time < 7 ? 'Build the update' : 'Move the parameters',
    loss: objective(point, config.landscape),
    beforeLoss: objective(before.point, config.landscape), afterLoss: objective(after.point, config.landscape),
    plainMove: { x: -config.lr * g.x, y: -config.lr * g.y },
    move: { x: after.point.x - before.point.x, y: after.point.y - before.point.y },
    exactMove: (() => { const full = hessianVector(before.point, config.landscape); return { x: -config.lr * full.x, y: -config.lr * full.y }; })(),
    exampleMoves: after.samples.map((i) => {
      const sample = gradient(before.point, config.landscape, data, [i], config.noise);
      return { x: -config.lr * sample.x, y: -config.lr * sample.y };
    }),
  };
}
