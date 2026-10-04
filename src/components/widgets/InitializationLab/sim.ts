/** Pure initialization experiments. The deep-network profile is the same seeded
 * forward/backward calculation used by GradientDepth; this file adds only the
 * measured summaries and the small symmetry-breaking example. */
import { makeBase, forward, profile, type Act as DeepAct, type Base } from '../GradientDepth/sim';

export type Mode = 'symmetry' | 'signals' | 'xavier' | 'he';
export type Act = Extract<DeepAct, 'tanh' | 'relu'>;
export type Preset = 'zero' | 'tiny' | 'xavier' | 'he' | 'large' | 'custom';
export type SymmetryCase = 'matched' | 'perturbed' | 'zero';

export const DEPTH = 12;
export const WIDTH = 16;
export const SEED = 41;
export const BASE: Base = makeBase(DEPTH, WIDTH, SEED);
export const T = 6;
export const SIGNAL_MARKERS = [
  { at: 0, label: 'forward' },
  { at: 3, label: 'backward' },
];
export const SYMMETRY_MARKERS = [
  { at: 0, label: 'forward' },
  { at: 2, label: 'backward' },
  { at: 4, label: 'update' },
];

export interface SignalStats {
  activationVariance: number[];
  activationMean: number[];
  gradientNorm: number[];
  measuredWeightVariance: number[];
  theoreticalWeightVariance: number;
}

const mean = (a: Float64Array): number => a.reduce((s, x) => s + x, 0) / a.length;
const variance = (a: Float64Array): number => {
  const m = mean(a);
  return a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length;
};

/** Scale is a multiplier on 1/sqrt(fan_in); hence Xavier=1 and He=sqrt(2)
 * for this equal-width stack. Every displayed number is measured from BASE. */
export function signalStats(base: Base, act: Act, scale: number): SignalStats {
  const cfg = { act, scale };
  const pass = forward(base, cfg);
  const back = profile(base, cfg);
  const n = base.width;
  return {
    activationVariance: pass.h.map(variance),
    activationMean: pass.h.map(mean),
    gradientNorm: back.norms,
    measuredWeightVariance: base.Z.map((layer) => {
      const weights = Float64Array.from(layer, (w) => (scale * w) / Math.sqrt(n));
      return variance(weights);
    }),
    theoreticalWeightVariance: (scale * scale) / n,
  };
}

/** Relative sizes keep the picture tied to the signal entering each pass. */
export function signalSizes(stats: SignalStats) {
  const inputSpread = Math.sqrt(stats.activationVariance[0]);
  const outputGradient = stats.gradientNorm.at(-1)!;
  return {
    forward: stats.activationVariance.map((v) => inputSpread > 0 ? Math.sqrt(v) / inputSpread : 0),
    backward: stats.gradientNorm.map((v) => outputGradient > 0 ? v / outputGradient : 0),
  };
}

export function presetScale(preset: Preset): number {
  switch (preset) {
    case 'zero': return 0;
    case 'tiny': return 0.1;
    case 'xavier': return 1;
    case 'he': return Math.SQRT2;
    case 'large': return 3;
    case 'custom': return 1;
  }
}

export function initialPreset(mode: Mode): Preset {
  return mode === 'he' ? 'he' : mode === 'xavier' ? 'xavier' : 'tiny';
}

export interface ToyState {
  hidden: number[];
  output: number;
  prediction: number;
  loss: number;
  hiddenGradient: number[];
  outputGradient: number[];
}

export interface SymmetryExperiment {
  before: ToyState;
  after: ToyState;
  shown: ToyState;
  initialWeights: number[];
  updatedWeights: number[];
  updateProgress: number;
}

const X = [1, 0.5] as const;
const TARGET = 0.8;
const LEARNING_RATE = 0.2;

function evaluate(weights: number[]): ToyState {
  const hidden: number[] = [];
  for (let j = 0; j < 3; j++) {
    const z = weights[j * 2] * X[0] + weights[j * 2 + 1] * X[1];
    hidden.push(Math.tanh(z));
  }
  const outputWeights = weights.slice(6, 9);
  const prediction = outputWeights.reduce((sum, w, j) => sum + w * hidden[j], 0);
  const error = prediction - TARGET;
  const hiddenGradient = hidden.map((h, j) => error * outputWeights[j] * (1 - h * h));
  const outputGradient = hidden.map((h) => error * h);
  const inputGradients = hiddenGradient.flatMap((g) => [g * X[0], g * X[1]]);
  return {
    hidden,
    output: prediction,
    prediction,
    loss: 0.5 * error * error,
    hiddenGradient: inputGradients,
    outputGradient,
  };
}

/** A real one-example SGD update on a 2 -> 3 -> 1 tanh MLP, no biases.
 * matched: identical nonzero hidden rows and equal output weights. zero:
 * every weight in both layers is zero. */
export function symmetryExperiment(which: SymmetryCase, updateProgress = 1): SymmetryExperiment {
  const initialWeights = which === 'matched'
    ? [0.3, -0.2, 0.3, -0.2, 0.3, -0.2, 0.45, 0.45, 0.45]
    : which === 'perturbed'
      ? [0.3, -0.2, 0.34, -0.18, 0.3, -0.2, 0.45, 0.45, 0.45]
      : new Array(9).fill(0);
  const before = evaluate(initialWeights);
  const updatedWeights = initialWeights.map((w, i) => w - LEARNING_RATE * (i < 6
    ? before.hiddenGradient[i]
    : before.outputGradient[i - 6]));
  const after = evaluate(updatedWeights);
  const p = Math.max(0, Math.min(1, updateProgress));
  const shownWeights = initialWeights.map((w, i) => w * (1 - p) + updatedWeights[i] * p);
  return {
    before,
    after,
    shown: evaluate(shownWeights),
    initialWeights,
    updatedWeights,
    updateProgress: p,
  };
}

export interface TimelineFrame {
  t: number;
  phase: 'forward' | 'backward' | 'update';
  forwardReveal: number;
  backwardReveal: number;
  updateProgress: number;
}

const clamp01 = (x: number): number => Math.max(0, Math.min(1, x));

/** Pure timing map. The stats are actual completed pass results; reveal only
 * stages the drawing and never invents intermediate layer measurements. */
export function frameAt(t: number, mode: Mode): TimelineFrame {
  const time = Math.max(0, Math.min(T, t));
  if (mode === 'symmetry') {
    const forwardReveal = clamp01(time / 1.7);
    const backwardReveal = clamp01((time - 2) / 1.7);
    const updateProgress = clamp01((time - 4) / 1.5);
    const phase = time < 2 ? 'forward' : time < 4 ? 'backward' : 'update';
    return { t: time, phase, forwardReveal, backwardReveal, updateProgress };
  }
  const forwardReveal = clamp01(time / 2.8);
  const backwardReveal = clamp01((time - 3) / 2.8);
  const phase = time < 3 ? 'forward' : 'backward';
  return { t: time, phase, forwardReveal, backwardReveal, updateProgress: 0 };
}
