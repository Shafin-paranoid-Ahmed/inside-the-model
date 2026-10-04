import { createRng } from '../../../lib/math/rng';
import { createOptimizerState, gradient, makeDataset, objective, startPoint, stepOptimizerState, type OptState, type Point } from '../OptimizerLab/sim';

export const STEPS = 80;
export const DURATION = 12;
export const DATA_SEED = 24062;
export const EXAMPLE_COUNT = 16;
export const COLORS = ['--viz-forward', '--viz-gradient', '--viz-pos', '--viz-neg'] as const;

export type Mode = 'rate' | 'warmup' | 'decay' | 'batch';
export type Decay = 'step' | 'cosine' | 'linear';
export type BatchComparison = 'equal-updates' | 'equal-examples';

export interface ScheduleRun {
  id: string;
  label: string;
  color: string;
  states: OptState[];
  rates: number[];
  batchSize: number;
  examplesSeenPerUpdate: number;
  decay?: Decay;
}

export interface ScheduleFrame {
  t: number;
  step: number;
  point: Point;
  loss: number;
  learningRate: number;
  examplesSeen: number;
  batchSize: number;
  state: OptState;
  diverged: boolean;
}

export interface SimConfig {
  mode: Mode;
  batchComparison?: BatchComparison;
  batchScaling?: boolean;
  seed?: number;
}

const clamp = (x: number, low: number, high: number) => Math.min(high, Math.max(low, x));
const lerp = (a: number, b: number, p: number) => a + (b - a) * p;
const mixPoint = (a: Point, b: Point, p: number): Point => ({ x: lerp(a.x, b.x, p), y: lerp(a.y, b.y, p) });

export function learningRateAt(schedule: Decay, step: number, totalSteps = STEPS, peak = 0.12, floor = 0.01): number {
  const progress = clamp(step / Math.max(1, totalSteps - 1), 0, 1);
  if (schedule === 'step') {
    const drops = Number(progress >= 2 / 3) + Number(progress >= 1 / 3);
    return Math.max(floor, peak * 0.25 ** drops);
  }
  if (schedule === 'linear') return peak + (floor - peak) * progress;
  return floor + (peak - floor) * (1 + Math.cos(Math.PI * progress)) / 2;
}

export function warmupRateAt(step: number, warmupSteps = 20, peak = 0.24): number {
  return peak * clamp((step + 1) / warmupSteps, 0, 1);
}

function sampleStream(seed: number, length: number, count: number): number[] {
  const rng = createRng(seed);
  const stream: number[] = [];
  while (stream.length < length) {
    const epoch = Array.from({ length: count }, (_, i) => i);
    for (let i = epoch.length - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      [epoch[i], epoch[j]] = [epoch[j], epoch[i]];
    }
    stream.push(...epoch);
  }
  return stream;
}

function makeRun(options: {
  id: string;
  label: string;
  color: string;
  optimizer: 'SGD' | 'Adam';
  batchSize: number;
  steps: number;
  examples: ReturnType<typeof makeDataset>;
  samples: number[];
  rateAt: (step: number) => number;
  decay?: Decay;
}): ScheduleRun {
  const states = [createOptimizerState(startPoint('ravine'))];
  const rates: number[] = [];
  for (let step = 0; step < options.steps; step++) {
    const prior = states[step];
    const rate = options.rateAt(step);
    rates.push(rate);
    const offset = step * options.batchSize;
    const batch = options.samples.slice(offset, offset + options.batchSize);
    const g = gradient(prior.point, 'ravine', options.examples, batch, 0.8);
    states.push(stepOptimizerState(prior, options.optimizer, g, rate, batch));
  }
  return { id: options.id, label: options.label, color: options.color, states, rates, batchSize: options.batchSize, examplesSeenPerUpdate: options.batchSize, decay: options.decay };
}

export function makeScheduleRuns({ mode, batchComparison = 'equal-updates', batchScaling = false, seed = DATA_SEED }: SimConfig): ScheduleRun[] {
  const examples = makeDataset(seed, EXAMPLE_COUNT);
  const sampleSequence = sampleStream(seed + 17, 8192, examples.length);
  if (mode === 'rate') {
    return [
      { id: 'small', label: 'Small · 0.015', color: COLORS[1], lr: 0.015 },
      { id: 'workable', label: 'Workable · 0.12', color: COLORS[0], lr: 0.12 },
      { id: 'unstable', label: 'Too large · 0.7', color: COLORS[3], lr: 0.7 },
    ].map((item) => makeRun({ ...item, optimizer: 'SGD', batchSize: EXAMPLE_COUNT, steps: STEPS, examples, samples: sampleSequence, rateAt: () => item.lr }));
  }
  if (mode === 'warmup') {
    return [
      { id: 'warmup', label: 'Adam · 20-step warmup', color: COLORS[0], rateAt: (step: number) => warmupRateAt(step) },
      { id: 'fixed', label: 'Adam · fixed 0.24', color: COLORS[2], rateAt: () => 0.24 },
    ].map((item) => makeRun({ ...item, optimizer: 'Adam', batchSize: 4, steps: STEPS, examples, samples: sampleSequence, decay: undefined }));
  }
  if (mode === 'decay') {
    const decays = (['step', 'cosine', 'linear'] as const).map((decay, index) => makeRun({
      id: decay, label: `${decay[0].toUpperCase()}${decay.slice(1)} decay`, color: COLORS[index], optimizer: 'SGD',
      batchSize: 4, steps: STEPS, examples, samples: sampleSequence, rateAt: (step) => learningRateAt(decay, step), decay,
    }));
    // Baseline: the peak rate held constant, so the decay's effect on the final noise is visible.
    const constant = makeRun({ id: 'constant', label: 'Constant 0.12', color: COLORS[3], optimizer: 'SGD', batchSize: 4, steps: STEPS, examples, samples: sampleSequence, rateAt: () => 0.12 });
    return [constant, ...decays];
  }
  const smallSteps = STEPS;
  const largeSteps = batchComparison === 'equal-updates' ? STEPS : Math.floor(STEPS * 2 / 8);
  return [
    { id: 'small', label: `Batch 2 · ${smallSteps} updates`, color: COLORS[0], batchSize: 2, steps: smallSteps },
    { id: 'large', label: `Batch 8 · ${largeSteps} updates${batchScaling ? ' · scaled rate' : ''}`, color: COLORS[2], batchSize: 8, steps: largeSteps },
  ].map((item) => makeRun({ ...item, optimizer: 'SGD', examples, samples: sampleSequence, rateAt: () => item.batchSize === 8 && batchScaling ? 0.32 : 0.08 }));
}

/** Continuously interpolate only the recorded parameter path; metrics use that exact point.
 * Optimizer memory is snapped to the nearest committed update to avoid fictional moments.
 */
export function frameAt(t: number, run: ScheduleRun): ScheduleFrame {
  const time = clamp(t, 0, DURATION);
  const progress = time / DURATION;
  const steps = run.states.length - 1;
  const position = progress * steps;
  const index = Math.min(steps, Math.floor(position));
  const blend = index === steps ? 0 : position - index;
  const next = Math.min(steps, index + 1);
  const point = mixPoint(run.states[index].point, run.states[next].point, blend);
  const committed = run.states[index];
  const updateIndex = Math.min(run.rates.length - 1, index);
  return {
    t: time,
    step: committed.step,
    point,
    loss: objective(point, 'ravine'),
    learningRate: run.rates[Math.max(0, updateIndex)] ?? 0,
    examplesSeen: committed.step * run.examplesSeenPerUpdate,
    batchSize: run.batchSize,
    state: committed,
    diverged: run.states[index].diverged || (blend > 0 && run.states[next].diverged),
  };
}

/** The pending (or final, at the endpoint) recorded update behind a schedule. */
export function updateAt(t: number, run: ScheduleRun) {
  const f = frameAt(t, run);
  const index = Math.min(run.rates.length - 1, f.step);
  const before = run.states[index], after = run.states[index + 1];
  return { index: index + 1, rate: run.rates[index], before, after,
    change: { x: after.point.x - before.point.x, y: after.point.y - before.point.y } };
}
