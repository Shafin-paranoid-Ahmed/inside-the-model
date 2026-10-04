import { describe, it, expect } from 'vitest';
import { makeData, paramCount, trainRun, forward, snapEpochs, weightsAt, evaluate, makeFold, foldFrame, FOLD_TIMING, stagePos, compute, dataOf } from '../../src/components/widgets/SpaceFolding/sim';

describe('space folding', () => {
  it('keeps the seed and class counts reproducible', () => {
    for (const kind of ['xor', 'spirals'] as const) {
      const d = makeData(kind, 12, 7);
      expect(d).toEqual(makeData(kind, 12, 7));
      expect(Array.from(d.ys).reduce((a, b) => a + b, 0)).toBe(12);
    }
  });
  it('matches a closed-form two-neuron forward pass and parameter count', () => {
    const w = [2, -1, .3, -1, 3, -.2, 4, -2, .1];
    const f = forward([2, 2, 1], w, .5, -.5);
    expect(paramCount([2, 2, 1])).toBe(9);
    expect(f.pre[0]).toEqual([1.8, -2.2]);
    expect(f.act[2][0]).toBeCloseTo(1 / (1 + Math.exp(-(4 * Math.tanh(1.8) - 2 * Math.tanh(-2.2) + .1))), 12);
  });
  it('snapshots include both endpoints and interpolate exact weights', () => {
    expect(snapEpochs(10, 5)).toEqual([0, 1, 3, 6, 10]);
    const r = { sizes: [2, 1], epochs: [0, 10], weights: [new Float64Array([0, 2, 4]), new Float64Array([2, 4, 6])] };
    expect(Array.from(weightsAt(r, .5).w)).toEqual([1, 3, 5]);
    expect(weightsAt(r, .5).epoch).toBe(5);
    expect(weightsAt({ ...r, weights: [r.weights[0]], epochs: [0] }, 3).w).toEqual(r.weights[0]);
  });
  it('affine maps preserve midpoints, then tanh squashes the real coordinates', () => {
    const d = { n: 3, xs: new Float64Array([-1, 0, 0, 0, 1, 0]), ys: new Float64Array([0, 0, 1]) };
    const w = [2, 1, .3, -1, 3, .2, 1, -1, 0], f = makeFold([2, 2, 1], w, d);
    for (let j = 0; j < 2; j++) {
      expect(f.stages[1][2 + j]).toBeCloseTo((f.stages[1][j] + f.stages[1][4 + j]) / 2);
      expect(f.stages[2][j]).toBeCloseTo(Math.tanh(f.stages[1][j]));
    }
    expect(foldFrame(f, 0, FOLD_TIMING).pos).toEqual(f.scene.P);
    expect(foldFrame(f, 10, FOLD_TIMING).pos).toEqual(f.stages[2]);
    expect(stagePos(4.5, 1, FOLD_TIMING)).toEqual({ m: 1, s: 0 });
    expect(foldFrame(f, 10, FOLD_TIMING).right).toBe(evaluate([2, 2, 1], w, d).correct);
  });
  it('training follows the shared engine deterministically and lowers the loss', () => {
    const d = makeData('xor', 8, 3), options = { seed: 2, epochs: 30, snaps: 4 };
    const a = trainRun([2, 2, 1], d, options);
    expect(a).toEqual(trainRun([2, 2, 1], d, options));
    expect(evaluate(a.sizes, a.weights.at(-1)!, d).loss).toBeLessThan(evaluate(a.sizes, a.weights[0], d).loss);
  });
  it('configured runs are measured honestly', () => {
    const expected = { line: 47, xor: 63, features: 80, one: 61, deep: 80 };
    for (const id of ['line', 'xor', 'features', 'one', 'deep'] as const) {
      const r = compute(id), e = evaluate(r.sizes, r.weights.at(-1)!, dataOf(id));
      expect(e.loss).toBeGreaterThanOrEqual(0);
      expect(e.correct).toBe(expected[id]);
      if (id === 'one' || id === 'deep' || id === 'xor') {
        const f = makeFold(r.sizes, r.weights.at(-1)!, dataOf(id));
        expect(foldFrame(f, 100, FOLD_TIMING).right).toBe(e.correct);
      }
    }
  }, 120000);
});
