import { describe, expect, it } from 'vitest';
import {
  DATA_SEED, DEFAULTS, DOMAIN, DURATION, EXAMPLE_COUNT, STEPS, contourSegments, createOptimizerState, firstAdamUpdate, frameAt, localLevels,
  gradient, hessianVector, makeDataset, makeRun, makeRace, objective, startPoint, stepOptimizerState,
  type Config, type Optimizer, type Point,
} from '../../src/components/widgets/OptimizerLab/sim';
import { adam, momentum, rmsprop, sgd } from '../../src/lib/nn/optim';

const config: Config = { landscape: 'ravine', lr: 0.04, batchSize: EXAMPLE_COUNT, noise: 1 };
const close = (a: number, b: number, tolerance = 1e-5) => expect(Math.abs(a - b)).toBeLessThan(tolerance);

describe('OptimizerLab simulation', () => {
  it('matches the analytic objective gradient with finite differences', () => {
    const point = { x: -0.73, y: 1.17 };
    const h = 1e-6;
    for (const landscape of ['ravine', 'saddle'] as const) {
      const exact = hessianVector(point, landscape);
      const dx = (objective({ x: point.x + h, y: point.y }, landscape) - objective({ x: point.x - h, y: point.y }, landscape)) / (2 * h);
      const dy = (objective({ x: point.x, y: point.y + h }, landscape) - objective({ x: point.x, y: point.y - h }, landscape)) / (2 * h);
      close(exact.x, dx); close(exact.y, dy);
    }
  });

  it('uses a reproducible, finite dataset whose complete batch is the exact mean gradient', () => {
    const examples = makeDataset(DATA_SEED);
    expect(examples).toEqual(makeDataset(DATA_SEED));
    expect(examples).toHaveLength(EXAMPLE_COUNT);
    const point = { x: 0.4, y: -0.8 };
    for (const landscape of ['ravine', 'saddle'] as const) {
      const full = gradient(point, landscape, examples, examples.map((_, i) => i));
      const exact = hessianVector(point, landscape);
      close(full.x, exact.x); close(full.y, exact.y);
    }
  });

  it('changes stochastic gradient sampling with batch size and noise scale', () => {
    const small = makeRun('SGD', { ...config, batchSize: 1 });
    const full = makeRun('SGD', { ...config, batchSize: EXAMPLE_COUNT });
    expect(small.states[1].samples).toHaveLength(1);
    expect(full.states[1].samples).toHaveLength(EXAMPLE_COUNT);
    expect(small.states[1].point).not.toEqual(full.states[1].point);
    expect(makeRun('SGD', { ...config, batchSize: 1, noise: 0 }).states[1].point)
      .toEqual(makeRun('SGD', { ...config, batchSize: EXAMPLE_COUNT, noise: 0 }).states[1].point);
  });

  it('precomputes deterministic runs for all four update rules', () => {
    const race = makeRace(config);
    expect(Object.keys(race)).toEqual(['SGD', 'Momentum', 'RMSProp', 'Adam']);
    for (const run of Object.values(race)) {
      expect(run.states).toHaveLength(STEPS + 1);
      expect(run.states).toEqual(makeRun(run.optimizer, config).states);
    }
    for (let step = 1; step <= STEPS; step++) {
      for (const name of ['Momentum', 'RMSProp', 'Adam'] as const) {
        expect(race[name].states[step].samples).toEqual(race.SGD.states[step].samples);
      }
    }
  });

  it('uses the shared site optimizer implementations for its parameter updates', () => {
    const rules: Record<Optimizer, ReturnType<typeof sgd>> = { SGD: sgd(config.lr), Momentum: momentum(config.lr), RMSProp: rmsprop(config.lr, DEFAULTS.alpha), Adam: adam(config.lr) };
    const examples = makeDataset(DATA_SEED);
    for (const name of ['SGD', 'Momentum', 'RMSProp', 'Adam'] as const) {
      const rule = rules[name];
      const run = makeRun(name, config), start = startPoint('ravine'), state = run.states[1];
      const g = gradient(start, 'ravine', examples, state.samples);
      const point = [start.x, start.y];
      rule.step(point, [g.x, g.y]);
      close(state.point.x, point[0]); close(state.point.y, point[1]);
      const pure = stepOptimizerState(createOptimizerState(start), name, g, config.lr, state.samples);
      close(pure.point.x, point[0]); close(pure.point.y, point[1]);
    }
  });

  it('applies Adam bias correction on the first update', () => {
    const g: Point = { x: 2, y: -0.5 };
    const corrected = firstAdamUpdate(g);
    close(corrected.first.x, (1 - DEFAULTS.beta1) * g.x);
    close(corrected.correctedFirst.x, g.x);
    close(corrected.correctedSecond.y, g.y * g.y);
    close(corrected.step.x, DEFAULTS.lr * Math.sign(g.x), 1e-6);
    close(corrected.step.y, DEFAULTS.lr * Math.sign(g.y), 1e-6);
    const run = makeRun('Adam', config), initial = startPoint('ravine');
    const firstGradient = gradient(initial, 'ravine', makeDataset(DATA_SEED), run.states[1].samples);
    close(run.states[1].point.x, initial.x - firstAdamUpdate(firstGradient, config.lr).step.x);
    close(run.states[1].point.y, initial.y - firstAdamUpdate(firstGradient, config.lr).step.y);
  });

  it('reports loss at the displayed interpolated point and has stable endpoints', () => {
    const run = makeRun('Momentum', config);
    expect(frameAt(0, run, 'ravine').point).toEqual(run.states[0].point);
    const middle = frameAt(4.25, run, 'ravine');
    close(middle.loss, objective(middle.point, 'ravine'));
    const end = frameAt(DURATION, run, 'ravine');
    expect(end.point).toEqual(run.states[STEPS].point);
    expect(end.step).toBe(STEPS);
  });

  it('does not call a valid saddle trajectory divergence just because it leaves the plot', () => {
    const run = makeRun('SGD', { landscape: 'saddle', lr: 0.08, batchSize: EXAMPLE_COUNT, noise: 0 });
    const end = frameAt(DURATION, run, 'saddle');
    expect(Number.isFinite(end.point.x + end.point.y)).toBe(true);
    expect(end.diverged).toBe(false);
    expect(Math.max(Math.abs(end.point.x), Math.abs(end.point.y))).toBeGreaterThan(DOMAIN);
  });

  it('places every contour segment endpoint on its loss level', () => {
    const box = { x0: -3, x1: 3, y0: 3, y1: -3 };
    const levels = localLevels('ravine', box, { x: -2.1, y: 2.1 });
    expect(levels.at(-1)).toBeCloseTo(objective({ x: -2.1, y: 2.1 }, 'ravine'), 12);
    const segments = contourSegments((x, y) => objective({ x, y }, 'ravine'), box, levels);
    expect(segments.length).toBeGreaterThan(100);
    for (const { level, points } of segments) {
      // Linear interpolation on a quadratic is close but not exact at grid scale.
      expect(Math.abs(objective({ x: points[0], y: points[1] }, 'ravine') - level)).toBeLessThan(0.05 * level);
    }
  });
});
