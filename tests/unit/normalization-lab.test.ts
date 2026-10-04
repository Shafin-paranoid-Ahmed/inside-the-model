import { describe, expect, test } from 'vitest';
import {
  accumulateRunningStats,
  defaultConfig,
  frameAt,
  INPUT,
  inputMatrix,
  RUNNING,
  T,
  TRAINING_BATCHES,
} from '../../src/components/widgets/NormalizationLab/sim';

const closeMatrix = (a: number[][], b: number[][], digits = 10) => {
  expect(a).toHaveLength(b.length);
  a.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(b[i][j], digits)));
};

describe('normalization reductions', () => {
  test('BatchNorm reduces each feature across examples', () => {
    const f = frameAt(8, { ...defaultConfig('batch'), gamma: 1, beta: 0 });
    expect(f.result.reductionGroups).toEqual([
      [[0, 0], [1, 0], [2, 0]], [[0, 1], [1, 1], [2, 1]], [[0, 2], [1, 2], [2, 2]],
    ]);
    expect(f.result.statistics.map((s) => s.mean[0])).toEqual([3, 2, 2]);
    closeMatrix(f.result.centered, [[-2, 0, -1], [0, -2, 0], [2, 2, 1]]);
    f.result.scaled.forEach((_row, j) => {
      const values = f.result.output.map((r) => r[j]);
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      expect(mean).toBeCloseTo(0, 10);
    });
  });

  test('LayerNorm reduces across each row and ignores a companion-row edit', () => {
    const cfg = { ...defaultConfig('layer'), gamma: 1, beta: 0 };
    const before = frameAt(8, cfg).result;
    cfg.companionOffset = 2;
    const after = frameAt(8, cfg).result;
    expect(after.reductionGroups).toEqual([[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]], [[2, 0], [2, 1], [2, 2]]]);
    expect(after.input[0]).toEqual(before.input[0]);
    expect(after.output[0]).toEqual(before.output[0]);
    expect(after.output[1]).not.toEqual(before.output[1]);
  });

  test('BatchNorm output for a row changes when a companion row changes', () => {
    const cfg = { ...defaultConfig('batch'), gamma: 1, beta: 0 };
    const before = frameAt(8, cfg).result.output[0];
    cfg.companionOffset = 2;
    const after = frameAt(8, cfg).result.output[0];
    expect(after).not.toEqual(before);
  });

  test('RMSNorm scales by RMS but leaves a generally nonzero row mean', () => {
    const cfg = { ...defaultConfig('rms'), gamma: 1, beta: 0 };
    const f = frameAt(8, cfg);
    const first = f.result.scaled[0];
    expect(first.reduce((a, b) => a + b, 0) / first.length).not.toBeCloseTo(0, 5);
    const rms = Math.sqrt((1 ** 2 + 2 ** 2 + 1 ** 2) / 3 + cfg.epsilon);
    expect(f.result.statistics[0].scale[0]).toBeCloseTo(rms, 10);
  });

  test('epsilon enters variance under the square root and prevents zero division', () => {
    const cfg = { ...defaultConfig('layer'), epsilon: 0.25, gamma: 1, beta: 0 };
    const f = frameAt(8, cfg);
    const variance = f.result.statistics[0].variance[0];
    expect(f.result.statistics[0].scale[0]).toBeCloseTo(Math.sqrt(variance + cfg.epsilon), 12);
    const constant = frameAt(8, { ...cfg, companionScale: 0, companionOffset: 2 });
    expect(constant.result.scaled.flat().every(Number.isFinite)).toBe(true);
  });

  test('affine gain and bias use the same normalized values, including zero and negative gain', () => {
    const zero = frameAt(8, { ...defaultConfig('affine'), gamma: 0, beta: 2 });
    expect(zero.result.affine.flat()).toEqual(Array(9).fill(2));
    const negative = frameAt(8, { ...defaultConfig('affine'), gamma: -2, beta: 0.5 });
    negative.result.scaled.forEach((row, i) => row.forEach((x, j) => {
      expect(negative.result.affine[i][j]).toBeCloseTo(-2 * x + 0.5, 12);
    }));
  });
});

describe('BatchNorm evaluation statistics', () => {
  test('EMA recurrence matches the two explicit population-stat batches', () => {
    const stats = accumulateRunningStats(TRAINING_BATCHES);
    stats.mean.forEach((x, i) => expect(x).toBeCloseTo([0.92, 1.28, 1.64][i], 12));
    stats.variance.forEach((x) => expect(x).toBeCloseTo(1.6, 12));
    expect(RUNNING).toEqual(stats);
  });

  test('evaluation output is fixed when the companion row changes', () => {
    const cfg = defaultConfig('inference');
    const first = frameAt(8, cfg).inference.output[0];
    cfg.companionOffset = 3;
    const second = frameAt(8, cfg).inference.output[0];
    expect(second).toEqual(first);
    expect(frameAt(8, cfg).training.output[0]).not.toEqual(first);
  });

  test('training and evaluation use the same current input but distinct statistics', () => {
    const f = frameAt(8, defaultConfig('inference'));
    expect(f.training.input).toEqual(f.input);
    expect(f.inference.input).toEqual(f.input);
    expect(f.training.statistics.map((s) => s.mean[0])).toEqual([3, 2, 2]);
    expect(f.inference.statistics.map((s) => s.mean[0])).toEqual(RUNNING.mean);
  });
});

describe('timeline frames', () => {
  test('all stages reveal exact values from the same input and statistics', () => {
    const cfg = defaultConfig('layer');
    const f0 = frameAt(0, cfg);
    const f1 = frameAt(4, cfg);
    const f2 = frameAt(6, cfg);
    const f3 = frameAt(8, cfg);
    const f4 = frameAt(T, cfg);
    expect(f0.visible).toEqual(f0.input);
    expect(f1.visible).toEqual(f1.centered);
    expect(f2.visible).toEqual(f2.scaled);
    expect(f3.visible).toEqual(f3.affine);
    expect(f4.visible).toEqual(f4.affine);
    expect(f4.phase).toBe('affine');
    expect(f4.revealCount).toBe(9);
    const mid = frameAt(3, cfg);
    expect(mid.revealCount).toBe(4);
    expect(mid.visible[0]).toEqual(mid.centered[0]);
    expect(mid.visible[1][0]).toBe(mid.centered[1][0]);
    expect(mid.visible[1][1]).toBe(mid.input[1][1]);
    const scaling = frameAt(5, cfg);
    expect(scaling.revealCount).toBe(4);
    expect(scaling.visible[0]).toEqual(scaling.scaled[0]);
    expect(scaling.visible[1][0]).toBe(scaling.scaled[1][0]);
    expect(scaling.visible[1][1]).toBe(scaling.centered[1][1]);
  });

  test('input editing changes only the companion row and frames are deterministic and clamped', () => {
    const cfg = { ...defaultConfig('batch'), companionScale: 2, companionOffset: 1 };
    const input = inputMatrix(cfg);
    expect(input[0]).toEqual(INPUT[0]);
    expect(input[2]).toEqual(INPUT[2]);
    expect(input[1]).toEqual([7, 1, 5]);
    expect(frameAt(3.25, cfg)).toEqual(frameAt(3.25, cfg));
    expect(frameAt(-1, cfg).t).toBe(0);
    expect(frameAt(99, cfg).t).toBe(T);
  });
});
