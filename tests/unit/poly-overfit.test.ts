import { describe, it, expect } from 'vitest';
import { fitPoly, evalPoly, fitModel, frameAt, frameAtDegree, ghostAt, BASE, BEST_DEG, BIAS2, VAR, D_MAX, T_TOTAL, degreeAt, timeOfDegree } from '../../src/components/widgets/PolyOverfit/sim';

describe('PolyOverfit sim', () => {
  it('recovers an exact polynomial', () => {
    const xs = Array.from({ length: 20 }, (_, i) => -1 + i / 9.5), ys = xs.map((x) => 1 - 2 * x + 3 * x ** 3);
    const w = fitPoly(xs, ys, 3);
    for (const x of xs) expect(evalPoly(w, x)).toBeCloseTo(1 - 2 * x + 3 * x ** 3, 8);
  });
  it('matches the closed-form line fit at degree 1', () => {
    const xs = [-1, -0.3, 0.2, 0.9], ys = [0.1, 0.5, 0.4, 1.3], n = 4, mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
    let sxy = 0, sxx = 0; xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; });
    expect(evalPoly(fitPoly(xs, ys, 1), 0.5)).toBeCloseTo(my + (sxy / sxx) * (0.5 - mx), 9);
  });
  it('train error never rises with degree', () => {
    for (let i = 1; i < D_MAX; i++) expect(BASE.train[i]).toBeLessThanOrEqual(BASE.train[i - 1] + 1e-9);
    expect(BASE.train[D_MAX - 1]).toBeLessThan(BASE.train[0] / 5);
  });
  it('test error is U-shaped', () => {
    const t = BASE.test, lo = t.indexOf(Math.min(...t));
    expect(lo).toBeGreaterThan(0); expect(lo).toBeLessThan(D_MAX - 1);
    expect(t[D_MAX - 1]).toBeGreaterThan(2 * t[lo]); expect(t[0]).toBeGreaterThan(2 * t[lo]);
  });
  it('bias falls and variance rises with degree', () => {
    expect(BIAS2[0]).toBeGreaterThan(BIAS2[BEST_DEG - 1] * 5);
    expect(VAR[D_MAX - 1]).toBeGreaterThan(VAR[BEST_DEG - 1] * 5);
    expect(BEST_DEG).toBeGreaterThan(1); expect(BEST_DEG).toBeLessThan(D_MAX);
  });
  it('frameAt endpoints and tween', () => {
    expect(frameAt(0).d).toBe(1); expect(frameAt(T_TOTAL).d).toBe(D_MAX);
    expect(frameAt(0).curve).toEqual(BASE.curves[0]);
    const mid = frameAtDegree(3.5), a = BASE.curves[2], b = BASE.curves[3];
    expect(mid.curve[10]).toBeCloseTo((a[10] + b[10]) / 2, 12);
    expect(degreeAt(timeOfDegree(5))).toBeCloseTo(5, 12);
    expect(ghostAt(0, 4)).toEqual(ghostAt(0, 4));
  });
  it('is deterministic per seed and differs across seeds', () => {
    expect(fitModel(3).curves[5]).toEqual(fitModel(3).curves[5]);
    expect(fitModel(4).curves[5]).not.toEqual(fitModel(3).curves[5]);
  });
});
