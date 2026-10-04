import { describe, expect, it } from 'vitest';
import { DEFAULTS, gradient, makeDataset, objective, stepOptimizerState, startPoint, type Point } from '../../src/components/widgets/OptimizerLab/sim';
import { DATA_SEED, DURATION, EXAMPLE_COUNT, STEPS, frameAt, learningRateAt, makeScheduleRuns, warmupRateAt } from '../../src/components/widgets/ScheduleLab/sim';

const close = (actual: number, expected: number, tolerance = 1e-9) => expect(Math.abs(actual - expected)).toBeLessThan(tolerance);

describe('ScheduleLab simulation', () => {
  it('keeps decay schedules at their start and floor on actual endpoint updates', () => {
    for (const schedule of ['step', 'cosine', 'linear'] as const) {
      close(learningRateAt(schedule, 0), 0.12);
      close(learningRateAt(schedule, STEPS - 1), 0.01);
    }
    expect(warmupRateAt(0, 20, 0.24)).toBeCloseTo(0.012);
    expect(warmupRateAt(19, 20, 0.24)).toBeCloseTo(0.24);
  });

  it('computes loss at the interpolated point and snaps optimizer memory to a committed update', () => {
    const run = makeScheduleRuns({ mode: 'warmup' })[0];
    const time = DURATION * 3.4 / STEPS;
    const frame = frameAt(time, run);
    expect(frame.step).toBe(3);
    expect(frame.state).toEqual(run.states[3]);
    expect(frame.point).not.toEqual(frame.state.point);
    close(frame.loss, objective(frame.point, 'ravine'));
    expect(frame.learningRate).toBe(run.rates[3]);
    expect(frameAt(0, run).point).toEqual(run.states[0].point);
    expect(frameAt(DURATION, run).point).toEqual(run.states[STEPS].point);
    expect(frameAt(DURATION, run).learningRate).toBe(run.rates[STEPS - 1]);
  });

  it('carries Adam moments across variable-rate warmup steps', () => {
    const run = makeScheduleRuns({ mode: 'warmup' })[0];
    const prior = run.states[1], sample = run.states[2].samples;
    const g: Point = gradient(prior.point, 'ravine', makeDataset(DATA_SEED, EXAMPLE_COUNT), sample, 0.8);
    const expected = stepOptimizerState(prior, 'Adam', g, run.rates[1], sample);
    expect(run.states[2].point).toEqual(expected.point);
    expect(run.states[2].first).toEqual(expected.first);
    expect(run.states[2].second).toEqual(expected.second);
    expect(DEFAULTS.beta1).toBe(0.9);
  });

  it('reuses identical sampled examples across schedule comparisons', () => {
    const warmup = makeScheduleRuns({ mode: 'warmup' });
    for (let step = 1; step <= STEPS; step++) expect(warmup[0].states[step].samples).toEqual(warmup[1].states[step].samples);
    const decay = makeScheduleRuns({ mode: 'decay' });
    for (let step = 1; step <= STEPS; step++) expect(decay[0].states[step].samples).toEqual(decay[2].states[step].samples);
    expect(makeScheduleRuns({ mode: 'rate' })[0].batchSize).toBe(EXAMPLE_COUNT);
  });

  it('pairs batch examples exactly for both equal-updates and equal-examples comparisons', () => {
    const equalUpdates = makeScheduleRuns({ mode: 'batch', batchComparison: 'equal-updates' });
    expect(equalUpdates[0].states).toHaveLength(STEPS + 1);
    expect(equalUpdates[1].states).toHaveLength(STEPS + 1);
    expect(equalUpdates[0].states[1].samples).toEqual(equalUpdates[1].states[1].samples.slice(0, 2));
    expect(equalUpdates[1].states[STEPS].step * 8).toBe(4 * equalUpdates[0].states[STEPS].step * 2);

    const equalExamples = makeScheduleRuns({ mode: 'batch', batchComparison: 'equal-examples' });
    expect(equalExamples[0].states.at(-1)?.step).toBe(80);
    expect(equalExamples[1].states.at(-1)?.step).toBe(20);
    const smallSeen = equalExamples[0].states.slice(1).flatMap((state) => state.samples);
    const largeSeen = equalExamples[1].states.slice(1).flatMap((state) => state.samples);
    expect(smallSeen).toEqual(largeSeen);
  });

  it('applies the optional linear batch scaling rule to the actual large-batch trajectory', () => {
    const runs = makeScheduleRuns({ mode: 'batch', batchScaling: true });
    expect(runs[0].rates[0]).toBe(0.08);
    expect(runs[1].rates[0]).toBe(0.32);
    expect(runs[0].states[1].point).not.toEqual(runs[1].states[1].point);
    expect(runs[1].states[0].point).toEqual(startPoint('ravine'));
  });
});
