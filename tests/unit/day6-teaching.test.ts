import { describe, expect, it } from 'vitest';
import { DEFAULTS, DURATION, makeRun, hessianVector, objective, teachingFrameAt } from '../../src/components/widgets/OptimizerLab/sim';
import { makeScheduleRuns, updateAt } from '../../src/components/widgets/ScheduleLab/sim';
import { defaultConfig, frameAt as normalizationFrame, workedExample } from '../../src/components/widgets/NormalizationLab/sim';
import { frameAt as dropoutFrame, T as DROPOUT_TIME } from '../../src/components/widgets/DropoutLab/sim';
import { frameAt as landscapeFrame, nonconvexLoss, nonconvexGradient, tinyNetStats } from '../../src/components/widgets/LandscapeLab/sim';

const config = { landscape: 'ravine' as const, lr: DEFAULTS.lr, batchSize: 16, noise: 1 };

describe('Day 6 worked explanations', () => {
  it('stretches one exact SGD update without mixing in another gradient', () => {
    const run = makeRun('SGD', config), before = run.states[0].point;
    const g = hessianVector(before, 'ravine');
    for (const t of [0, 3, 7, 8.5, DURATION]) {
      const f = teachingFrameAt(t, run, config, 1);
      expect(f.after.gradient.x).toBeCloseTo(g.x, 12);
      expect(f.after.point.x).toBeCloseTo(before.x - config.lr * g.x, 12);
      expect(f.loss).toBeCloseTo(objective(f.point, 'ravine'), 12);
      expect(f.plainMove.x).toBeCloseTo(f.exactMove.x, 12);
    }
    expect(teachingFrameAt(0, run, config).point).toEqual(before);
    expect(teachingFrameAt(DURATION, run, config).point).toEqual(run.states[1].point);
  });

  it('shows momentum history and coordinate divisors from the selected recorded update', () => {
    const momentum = makeRun('Momentum', config);
    const first = teachingFrameAt(4, momentum, config, 1), later = teachingFrameAt(4, momentum, config, 8);
    expect(first.move.x).toBeCloseTo(first.plainMove.x, 12);
    expect(later.numerator.x).toBeCloseTo(0.9 * later.before.momentum.x + later.after.gradient.x, 12);
    expect(Math.abs(later.move.x - later.plainMove.x)).toBeGreaterThan(0.001);
    for (const optimizer of ['RMSProp', 'Adam'] as const) {
      const f = teachingFrameAt(6, makeRun(optimizer, config), config, 8);
      for (const axis of ['x', 'y'] as const) {
        expect(f.before.point[axis] - config.lr * f.numerator[axis] / f.denominator[axis]).toBeCloseTo(f.after.point[axis], 12);
      }
    }
    const adam = teachingFrameAt(4, makeRun('Adam', config), config, 1);
    expect(adam.numerator.x).toBeCloseTo(adam.after.gradient.x, 12);
  });

  it('links the warmup schedule to its actual first weight update and final endpoint', () => {
    const [warmup, fixed] = makeScheduleRuns({ mode: 'warmup' });
    const a = updateAt(0, warmup), b = updateAt(0, fixed);
    expect(a.rate).toBeCloseTo(0.012, 12);
    expect(b.rate).toBeCloseTo(0.24, 12);
    expect(Math.abs(a.change.x)).toBeCloseTo(Math.abs(b.change.x) / 20, 12);
    const final = updateAt(12, warmup);
    expect(final.index).toBe(80);
    expect(final.after).toEqual(warmup.states.at(-1));
  });

  it('works the highlighted row and column with the correct statistics', () => {
    const config = defaultConfig('layer');
    const layer = workedExample(normalizationFrame(8, config), config, 0);
    expect(layer.values).toEqual([1, 2, 1]);
    expect(layer.mean).toBeCloseTo(4 / 3, 12);
    expect(layer.variance).toBeCloseTo(2 / 9, 12);
    expect(layer.scaled).toBeCloseTo((-1 / 3) / Math.sqrt(2 / 9 + 0.01), 12);
    const batchConfig = { ...config, norm: 'batch' as const, companionOffset: 3 };
    const batch = workedExample(normalizationFrame(8, batchConfig), batchConfig, 0);
    expect(batch.values).toEqual([1, 6, 5]);
    expect(batch.output).not.toBeCloseTo(layer.output, 3);
    const unchanged = workedExample(normalizationFrame(8, { ...config, companionOffset: 3 }), config, 0);
    expect(unchanged.output).toBe(layer.output);
  });

  it('adds the same dropout contributions shown in the network and keeps future checkpoints hidden', () => {
    for (const time of [0, 2.5, 4, 8]) {
      const f = dropoutFrame(time, 'mask', 0.5);
      expect(f.units.reduce((sum, unit) => sum + unit.contribution, 0)).toBeCloseTo(f.network.output, 12);
      for (const unit of f.units) if (!unit.kept) expect(unit.activation).toBe(0);
    }
    const first = dropoutFrame(0, 'early-stop'), end = dropoutFrame(DROPOUT_TIME, 'early-stop');
    expect(first.checkpoint.epoch).toBe(0);
    expect(first.staleChecks).toBe(0);
    expect(end.checkpoint.epoch).toBe(end.run.bestEpoch);
    expect(end.staleChecks).toBe(3);
  });

  it('mirrors a loss path without changing objective values and explains the residual chain rule', () => {
    for (const t of [0, 5, 10]) {
      const right = landscapeFrame(t, 'nonconvex'), left = landscapeFrame(t, 'nonconvex', { side: 'left' });
      expect(left.point.x).toBe(-right.point.x);
      expect(left.loss).toBeCloseTo(nonconvexLoss(left.point), 12);
      expect(left.loss).toBe(right.loss);
      expect(nonconvexGradient(left.point).x).toBe(-nonconvexGradient(right.point).x);
    }
    const stats = tinyNetStats(0.5);
    expect(stats.blocks[1].input).toBe(stats.blocks[0].output);
    expect(0.5 * stats.blocks[0].derivative * stats.blocks[1].derivative).toBeCloseTo(stats.residualJacobian.at(-1)!, 12);
  });
});
