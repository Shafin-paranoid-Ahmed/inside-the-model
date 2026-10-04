import { describe, expect, test } from 'vitest';
import * as S from '../../src/components/widgets/ConvSlider/sim';

describe('ConvSlider sim', () => {
  test('image is 12x12 integers 0..9, deterministic', () => {
    expect(S.IMAGE.length).toBe(144);
    expect(S.IMAGE.every((v) => Number.isInteger(v) && v >= 0 && v <= S.MAXV)).toBe(true);
    expect(Array.from(S.makeImage())).toEqual(Array.from(S.IMAGE));
  });

  test('breakdown sum equals the shared conv2d output at every position', () => {
    for (const f of S.FILTERS) {
      const out = S.applyFilter(f.w);
      expect(out.length).toBe(S.NPOS);
      for (let i = 0; i < S.NPOS; i++) {
        const b = S.breakdownAt(i, f.w);
        expect(b.sum).toBeCloseTo(out[i], 9);
        expect(b.prods.reduce((a, v) => a + v, 0)).toBeCloseTo(b.sum, 9);
      }
    }
  });

  test('vertical-edge filter: positive at the left edge of the square, negative at the right, zero inside', () => {
    const out = S.applyFilter(S.FILTERS[0].w);
    expect(out[1 * S.O + 0]).toBeGreaterThan(15);
    expect(out[1 * S.O + 4]).toBeLessThan(-15);
    expect(Math.abs(out[2 * S.O + 2])).toBeLessThanOrEqual(3);
  });

  test('sweepAt covers every position once and ends done', () => {
    expect(S.sweepAt(0).idx).toBe(0);
    expect(S.sweepAt(37.5).idx).toBe(37);
    expect(S.sweepAt(37.5)).toMatchObject({ ox: 7, oy: 3 });
    const end = S.sweepAt(S.slideU(S.T_SLIDE));
    expect(end).toMatchObject({ idx: 99, done: true, alpha: 1 });
  });

  test('filter timeline picks the right filter and reaches every position', () => {
    expect(S.filtersAt(0).fi).toBe(0);
    expect(S.filtersAt(S.filterStart(2) + 0.1).fi).toBe(2);
    expect(S.filtersAt(S.T_FILTERS).fi).toBe(S.FILTERS.length - 1);
    expect(S.filtersAt(S.filterStart(1) + S.F_SEG - 0.01).u).toBe(S.NPOS);
  });

  test('parameter counts follow from the sizes', () => {
    expect(S.paramCounts()).toMatchObject({ inputs: 144, outputs: 100, conv: 10, denseWeights: 14400, dense: 14500, multiplies: 900 });
  });

  test('training: deterministic, loss falls over decades, weights approach the target filter', () => {
    const a = S.train(3), b = S.train(3);
    expect(Array.from(a.w[S.STEPS])).toEqual(Array.from(b.w[S.STEPS]));
    expect(a.loss.length).toBe(S.STEPS + 1);
    expect(a.loss[S.STEPS]).toBeLessThan(a.loss[0] / 1000);
    for (let k = 1; k < 20; k++) expect(a.loss[k]).toBeLessThan(a.loss[k - 1]);
    a.w[S.STEPS].forEach((v, i) => expect(Math.abs(v - S.TARGET.w[i])).toBeLessThan(0.2));
    expect(S.train(4).w[0]).not.toEqual(a.w[0]);
  });

  test('stored loss matches a fresh evaluation at the stored weights', () => {
    const r = S.train(2), k = 50;
    const y = S.applyFilter(r.w[k], r.b[k]);
    const l = y.reduce((s, v, i) => s + (v - r.target[i]) ** 2, 0) / S.NPOS;
    expect(l).toBeCloseTo(r.loss[k], 8);
  });

  test('learnStep clamps', () => {
    expect(S.learnStep(0)).toBe(0);
    expect(S.learnStep(1e6)).toBe(S.STEPS);
  });
});
