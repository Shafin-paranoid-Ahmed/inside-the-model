import { createRng } from '../../../lib/math/rng';

/** Deterministic, DOM-free demonstrations for dropout and regularization. */
export type Mode = 'mask' | 'scaling' | 'ensemble' | 'decay' | 'early-stop' | 'augment';
export const T = 8;
export const KEEP_MIN = 0.1;
export const KEEP_MAX = 1;
export const INPUT = [0.8, 0.5] as const;
export const WEIGHTS = [
  [0.8, 0.2],
  [0.2, 0.8],
  [0.6, -0.1],
  [-0.2, 0.7],
] as const;
export const BIASES = [0.1, -0.05, 0.02, 0.08] as const;
export const READOUT = [0.5, -0.35, 0.4, 0.25] as const;
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export interface ForwardResult { hidden: number[]; output: number }
export interface MaskResult extends ForwardResult { mask: number[]; probability: number; index: number }

/** One 2 -> 4 -> 1 tanh network. A missing mask means the full network. */
export function forward(input: readonly number[] = INPUT, mask?: readonly number[], keep = 1, inverted = true): ForwardResult {
  if (!(keep > 0 && keep <= 1)) throw new RangeError('keep must be greater than 0 and at most 1');
  if (mask && mask.length !== 4) throw new RangeError('mask must contain four entries');
  const hidden = WEIGHTS.map((row, i) => {
    const raw = Math.tanh(row[0] * input[0] + row[1] * input[1] + BIASES[i]);
    if (!mask) return raw;
    return raw * mask[i] * (inverted ? 1 / keep : 1);
  });
  return { hidden, output: hidden.reduce((sum, value, i) => sum + value * READOUT[i], 0) };
}

/** Enumerate all 2^4 masks in stable bit order and attach their Bernoulli probabilities. */
export function enumerateMasks(keep = 0.5): MaskResult[] {
  if (!(keep > 0 && keep <= 1)) throw new RangeError('keep must be greater than 0 and at most 1');
  return Array.from({ length: 16 }, (_, index) => {
    const mask = Array.from({ length: 4 }, (_, unit) => (index >> unit) & 1);
    const active = mask.filter(Boolean).length;
    const probability = keep ** active * (1 - keep) ** (4 - active);
    return { ...forward(INPUT, mask, keep, true), mask, probability, index };
  });
}

/** Repeatable Bernoulli draws for the training animation, separate from exact enumeration. */
export function sampleMasks(keep = 0.5, count = 16): MaskResult[] {
  if (!(keep > 0 && keep <= 1)) throw new RangeError('keep must be greater than 0 and at most 1');
  const rng = createRng(32);
  return Array.from({ length: count }, (_, index) => {
    const mask = Array.from({ length: 4 }, () => (rng.next() < keep ? 1 : 0));
    const active = mask.filter(Boolean).length;
    const probability = keep ** active * (1 - keep) ** (4 - active);
    return { ...forward(INPUT, mask, keep, true), mask, probability, index };
  });
}

export function expectation(keep = 0.5) {
  const masks = enumerateMasks(keep);
  const linear = masks.reduce((sum, result) => sum + result.probability * result.output, 0);
  const nonlinear = masks.reduce((sum, result) => sum + result.probability * sigmoid(result.output), 0);
  return { masks, linear, full: forward().output, nonlinear, sigmoidFull: sigmoid(forward().output), probabilityTotal: masks.reduce((s, x) => s + x.probability, 0) };
}

/** First-step SGD, coupled-L2 SGD, Adam with coupled L2, and decoupled AdamW. */
export function decayUpdates(theta = 1, gradient = 0.4, learningRate = 0.1, lambda = 0.2) {
  if (learningRate < 0 || lambda < 0) throw new RangeError('learningRate and lambda must be nonnegative');
  const sgd = theta - learningRate * gradient;
  const sgdL2 = theta - learningRate * (gradient + lambda * theta);
  const epsilon = 1e-8;
  const adamCoupled = theta - learningRate * (gradient + lambda * theta) / (Math.abs(gradient + lambda * theta) + epsilon);
  const adamDataStep = theta - learningRate * gradient / (Math.abs(gradient) + epsilon);
  const adamW = adamDataStep - learningRate * lambda * theta;
  return { theta, gradient, learningRate, lambda, epsilon, sgd, sgdL2, adamCoupled, adamDataStep, adamW };
}

export interface TrainingPoint { epoch: number; theta: number; train: number; validation: number }
export interface EarlyStopRun { history: TrainingPoint[]; bestEpoch: number; stopEpoch: number; selected: TrainingPoint }
/** Run real full-batch gradient descent on the training objective and evaluate a separate validation objective. */
export function earlyStopRun(): EarlyStopRun {
  const trainTarget = 2, validationTarget = 0.7, learningRate = 0.1, patience = 3, maxEpochs = 18;
  let theta = -1;
  const point = (epoch: number): TrainingPoint => ({ epoch, theta, train: (theta - trainTarget) ** 2, validation: (theta - validationTarget) ** 2 + 0.04 });
  const history = [point(0)];
  for (let epoch = 1; epoch <= maxEpochs; epoch++) {
    const gradient = 2 * (theta - trainTarget);
    theta -= learningRate * gradient;
    history.push(point(epoch));
  }
  let bestIndex = 0, stale = 0, stopEpoch = maxEpochs;
  for (let i = 1; i < history.length; i++) {
    if (history[i].validation < history[bestIndex].validation) { bestIndex = i; stale = 0; }
    else if (++stale >= patience) { stopEpoch = i; break; }
  }
  return { history, bestEpoch: history[bestIndex].epoch, stopEpoch, selected: history[bestIndex] };
}

export type AugmentTask = 'presence' | 'side';
export function augmentation(task: AugmentTask = 'presence') {
  const image = [[1, 0], [0, 0]];
  const flipped = image.map((row) => [...row].reverse());
  const present = (pixels: number[][]) => pixels.some((row) => row.some(Boolean));
  const labelBefore = task === 'presence' ? (present(image) ? 'bright pixel present' : 'no bright pixel') : 'bright pixel on left';
  const labelAfter = task === 'presence' ? (present(flipped) ? 'bright pixel present' : 'no bright pixel') : 'bright pixel on right';
  const wrongLabel = task === 'presence' ? labelBefore : 'bright pixel on left';
  return { image, flipped, task, labelBefore, labelAfter, wrongLabel, labelPreserved: labelBefore === labelAfter };
}

const STOP_RUN = earlyStopRun();
/** Each epoch gets one slot; the stop epoch keeps two, so the final state stays on screen. */
const EPOCH_SLOT = T / (STOP_RUN.stopEpoch + 2);
export const DECAY_SLOT = 2;
export const MARKERS: Record<Mode, { at: number; label: string }[]> = {
  mask: [{ at: 0, label: 'Training' }, { at: 4, label: 'Inference' }],
  scaling: [{ at: 0, label: 'Unscaled' }, { at: 2.6, label: 'Inverted' }, { at: 5.3, label: 'Full network' }],
  ensemble: [{ at: 0, label: 'Masks' }, { at: 4, label: 'Mean' }, { at: 6, label: 'Sigmoid' }],
  decay: [{ at: 0, label: 'SGD' }, { at: DECAY_SLOT, label: 'SGD+L2' }, { at: 2 * DECAY_SLOT, label: 'Adam+L2' }, { at: 3 * DECAY_SLOT, label: 'AdamW' }],
  'early-stop': [{ at: 0, label: 'Improving' }, { at: (STOP_RUN.bestEpoch + 1) * EPOCH_SLOT, label: 'Waiting' }, { at: STOP_RUN.stopEpoch * EPOCH_SLOT, label: 'Stop' }],
  augment: [{ at: 0, label: 'Original' }, { at: 3, label: 'Flip' }, { at: 5, label: 'Check label' }],
};

/** How many of n items are on screen after `span` seconds of a phase that started at `from`. */
const revealCount = (time: number, from: number, span: number, n = 16) => time < from ? 0 : Math.min(n, Math.floor(((time - from) / span) * n) + 1);

export function frameAt(t: number, mode: Mode, keep = 0.5, task: AugmentTask = 'presence') {
  const time = clamp(t, 0, T);
  const expectationData = expectation(keep);
  const trainingDraws = sampleMasks(keep);
  const run = STOP_RUN;
  const visibleEpoch = Math.min(run.stopEpoch, Math.floor(time / EPOCH_SLOT + 1e-9));
  const phase = MARKERS[mode].reduce((last, marker) => time >= marker.at ? marker.label : last, MARKERS[mode][0].label);
  const maskIndex = Math.min(15, Math.floor((time / T) * 16));
  const raw = forward();
  const inference = time >= 4;
  const network = inference ? raw : trainingDraws[maskIndex];
  const checkpoint = run.history.slice(0, visibleEpoch + 1).reduce((best, point) => point.validation < best.validation ? point : best);
  const decayRows = Math.min(4, Math.floor(time / DECAY_SLOT) + 1);
  const unscaledShown = revealCount(time, 0, 2), invertedShown = revealCount(time, 2.6, 2);
  return {
    time, phase,
    maskIndex,
    inference,
    network,
    units: network.hidden.map((activation, i) => ({ raw: raw.hidden[i], activation, weight: READOUT[i], contribution: activation * READOUT[i], kept: inference || trainingDraws[maskIndex].mask[i] === 1 })),
    checkpoint,
    staleChecks: visibleEpoch - checkpoint.epoch,
    /** Scaling view: dots and mean lines revealed so far. */
    unscaledShown, invertedShown,
    showUnscaledMean: unscaledShown === 16, showInvertedMean: invertedShown === 16, showFull: time >= 5.3,
    /** Ensemble view. */
    ensembleShown: revealCount(time, 0, 3.5), showMean: time >= 4, showSigmoid: time >= 6,
    masks: expectationData.masks,
    sampledMasks: trainingDraws,
    full: expectationData.full,
    weightedMean: expectationData.linear,
    probabilityTotal: expectationData.probabilityTotal,
    nonlinearMean: expectationData.nonlinear,
    sigmoidFull: expectationData.sigmoidFull,
    unscaledMean: keep * expectationData.full,
    run,
    visibleEpoch,
    augmentation: augmentation(task),
    /** 0 = original position, 1 = fully flipped. */
    flip: clamp((time - 3) / 1.2, 0, 1),
    showAfterLabel: time >= 5,
    decay: decayUpdates(),
    decayRows,
    /** 0..1 growth of the newest decay arrow. */
    decayGrow: clamp((time - (decayRows - 1) * DECAY_SLOT) / 0.6, 0, 1),
  };
}
