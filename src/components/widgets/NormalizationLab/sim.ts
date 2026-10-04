export type Mode = 'standardize' | 'batch' | 'layer' | 'rms' | 'affine' | 'inference';
export type Norm = 'batch' | 'layer' | 'rms';
export type Phase = 'group' | 'center' | 'scale' | 'affine';

export const T = 8;
export const PHASES: { at: number; label: string }[] = [
  { at: 0, label: 'Group' },
  { at: 2, label: 'Center' },
  { at: 4, label: 'Scale' },
  { at: 6, label: 'Gain' },
];
export const INPUT: number[][] = [[1, 2, 1], [3, 0, 2], [5, 4, 3]];
export type Coord = [number, number];
export type Group = Coord[];
const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
const populationVariance = (xs: number[], m: number): number => mean(xs.map((x) => (x - m) ** 2));
export const TRAINING_BATCHES = [
  [[0, 1, 2], [2, 3, 4], [4, 5, 6]],
  [[1, 2, 3], [3, 4, 5], [5, 6, 7]],
];

export function accumulateRunningStats(
  batches: number[][][],
  momentum = 0.2,
  initial = { mean: [0, 0, 0], variance: [1, 1, 1] },
) {
  const running = { mean: [...initial.mean], variance: [...initial.variance] };
  for (const batch of batches) for (let j = 0; j < running.mean.length; j++) {
    const column = batch.map((row) => row[j]);
    const batchMean = mean(column);
    const batchVariance = populationVariance(column, batchMean);
    running.mean[j] = (1 - momentum) * running.mean[j] + momentum * batchMean;
    running.variance[j] = (1 - momentum) * running.variance[j] + momentum * batchVariance;
  }
  return running;
}

// Illustrative EMA: start from mean 0 / variance 1; update with two synthetic batches.
export const RUNNING = accumulateRunningStats(TRAINING_BATCHES);

export interface Config {
  mode: Mode;
  norm: Norm;
  epsilon: number;
  gamma: number;
  beta: number;
  companionScale: number;
  companionOffset: number;
}

export interface Statistics { mean: number[]; variance: number[]; scale: number[] }
export interface Result {
  input: number[][];
  centered: number[][];
  scaled: number[][];
  affine: number[][];
  output: number[][];
  statistics: Statistics[];
  reductionGroups: Group[];
}
export interface Frame {
  t: number;
  phase: Phase;
  phaseProgress: number;
  input: number[][];
  centered: number[][];
  scaled: number[][];
  affine: number[][];
  output: number[][];
  training: Result;
  inference: Result;
  result: Result;
  reductionGroups: Group[];
  statistics: Statistics[];
  visible: number[][];
  revealCount: number;
  stageLabel: string;
  normalizedMean: number[];
  normalizedVariance: number[];
}

export const defaultConfig = (mode: Mode = 'standardize'): Config => ({
  mode,
  norm: mode === 'batch' || mode === 'inference' ? 'batch' : mode === 'rms' ? 'rms' : 'layer',
  epsilon: 0.01,
  gamma: mode === 'affine' ? 1.2 : 1,
  beta: mode === 'affine' ? 0.3 : 0,
  companionScale: 1,
  companionOffset: 0,
});

const copy = (x: number[][]): number[][] => x.map((row) => [...row]);
const groupsFor = (x: number[][], norm: Norm): Group[] =>
  norm === 'batch'
    ? x[0].map((_, j) => x.map((_, i) => [i, j] as Coord))
    : x.map((row, i) => row.map((_, j) => [i, j] as Coord));

/* Each statistics entry corresponds to one group. Groups hold [row, column] coordinates. */
function reduce(input: number[][], norm: Norm, epsilon: number): { stats: Statistics[]; groups: Group[] } {
  const groups = groupsFor(input, norm);
  const stats = groups.map((group) => {
    const xs = group.map(([i, j]) => input[i][j]);
    if (norm === 'rms') {
      const rms = Math.sqrt(mean(xs.map((x) => x * x)) + epsilon);
      return { mean: [mean(xs)], variance: [populationVariance(xs, mean(xs))], scale: [rms] };
    }
    const m = mean(xs);
    const variance = populationVariance(xs, m);
    return { mean: [m], variance: [variance], scale: [Math.sqrt(variance + epsilon)] };
  });
  return { stats, groups };
}

function transform(input: number[][], norm: Norm, epsilon: number, gamma: number, beta: number, fixed?: Pick<Statistics, 'mean' | 'variance'>): Result {
  const { stats, groups } = fixed
    ? { stats: input[0].map((_, j) => ({ mean: [fixed.mean[j]], variance: [fixed.variance[j]], scale: [Math.sqrt(fixed.variance[j] + epsilon)] })), groups: groupsFor(input, 'batch') }
    : reduce(input, norm, epsilon);
  const centered = copy(input);
  if (norm !== 'rms') {
    for (const [g, group] of groups.entries()) for (const [i, j] of group) centered[i][j] -= stats[g].mean[0];
  }
  const scaled = copy(centered);
  for (const [g, group] of groups.entries()) for (const [i, j] of group) scaled[i][j] /= stats[g].scale[0];
  const affine = scaled.map((row) => row.map((v) => gamma * v + beta));
  return { input: copy(input), centered, scaled, affine, output: affine, statistics: stats, reductionGroups: groups };
}

export function inputMatrix(config: Pick<Config, 'companionScale' | 'companionOffset'>): number[][] {
  return INPUT.map((row, i) => i === 1
    ? row.map((v) => v * config.companionScale + config.companionOffset)
    : [...row]);
}

const phaseAt = (t: number): { phase: Phase; phaseProgress: number; index: number } => {
  const index = t < 2 ? 0 : t < 4 ? 1 : t < 6 ? 2 : 3;
  const start = index * 2;
  return { phase: (['group', 'center', 'scale', 'affine'] as Phase[])[index], phaseProgress: Math.max(0, Math.min(1, (t - start) / 2)), index };
};

const getVisible = (result: Result, index: number): number[][] =>
  index === 0 ? result.input : index === 1 ? result.centered : index === 2 ? result.scaled : result.affine;

export function frameAt(time: number, config: Config): Frame {
  const t = Math.max(0, Math.min(T, time));
  const { phase, phaseProgress, index } = phaseAt(t);
  const input = inputMatrix(config);
  const training = transform(input, 'batch', config.epsilon, config.gamma, config.beta);
  const inference = transform(input, 'batch', config.epsilon, config.gamma, config.beta, RUNNING);
  const norm = config.mode === 'inference' ? 'batch' : config.norm;
  const result = transform(input, norm, config.epsilon, config.gamma, config.beta);
  const active = config.mode === 'inference' ? training : result;
  const target = getVisible(active, index);
  const prior = index === 0 ? target : getVisible(active, index - 1);
  const revealCount = index === 0 ? 0 : Math.min(input.length * input[0].length, Math.floor(phaseProgress * input.length * input[0].length));
  const visible = target.map((row, i) => row.map((value, j) =>
    index > 0 && i * row.length + j >= revealCount ? prior[i][j] : value,
  ));
  const resultMean = norm === 'batch'
    ? input[0].map((_, j) => mean(result.output.map((row) => row[j])))
    : result.output.map((row) => mean(row));
  const resultVariance = norm === 'batch'
    ? input[0].map((_, j) => populationVariance(result.output.map((row) => row[j]), resultMean[j]))
    : result.output.map((row, i) => populationVariance(row, resultMean[i]));
  const stages: Record<Phase, string> = { group: 'Output (not computed yet)', center: norm === 'rms' ? 'No centering step in RMSNorm' : 'Subtract each group mean', scale: norm === 'rms' ? 'Divide by each group RMS' : 'Divide by standard deviation', affine: 'Apply learned gain and shift' };
  return {
    t, phase, phaseProgress, input,
    centered: result.centered, scaled: result.scaled, affine: result.affine, output: result.output,
    training, inference, result, reductionGroups: result.reductionGroups, statistics: result.statistics,
    visible, revealCount, stageLabel: stages[phase], normalizedMean: resultMean, normalizedVariance: resultVariance,
  };
}

/** Exact worked example for the first cell in the highlighted group. */
export function workedExample(frame: Frame, config: Pick<Config, 'gamma' | 'beta'>, groupIndex: number) {
  const index = Math.max(0, Math.min(frame.reductionGroups.length - 1, Math.round(groupIndex)));
  const group = frame.reductionGroups[index], [row, column] = group[0];
  const stats = frame.statistics[index];
  return { row, column, values: group.map(([i, j]) => frame.input[i][j]),
    mean: stats.mean[0], variance: stats.variance[0], divisor: stats.scale[0],
    input: frame.input[row][column], centered: frame.centered[row][column],
    scaled: frame.scaled[row][column], output: frame.output[row][column],
    gamma: config.gamma, beta: config.beta };
}
