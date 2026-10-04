import { describe, it, expect } from 'vitest';
import { FNS, hAt, timeForH, secantAt, slopeCurve, SECANT_T, bowl, bowlGrad, ballPath, gradAt, GRAD_T, GRAD_START, N_STEPS } from '../../src/components/widgets/SlopeExplorer/sim';

describe('secant mode', () => {
  it('derivatives match central differences', () => {
    for (const fn of Object.values(FNS)) for (const x of [-1.3, 0, 0.7, 2]) {
      const e = 1e-6;
      expect(fn.df(x)).toBeCloseTo((fn.f(x + e) - fn.f(x - e)) / (2 * e), 5);
    }
  });
  it('h shrinks from 2 to 0.001 and timeForH inverts hAt', () => {
    expect(hAt(0)).toBeCloseTo(2, 10);
    expect(hAt(SECANT_T)).toBeCloseTo(0.001, 10);
    expect(hAt(timeForH(0.1))).toBeCloseTo(0.1, 10);
  });
  it('secant slope converges to the tangent slope', () => {
    for (const fn of Object.values(FNS)) {
      const a = secantAt(0, fn, fn.x0), b = secantAt(SECANT_T, fn, fn.x0);
      expect(b.gap).toBeLessThan(a.gap);
      expect(b.gap).toBeLessThan(0.01);
    }
  });
  it('x squared: secant slope is exactly 2x + h', () => {
    const f = secantAt(3, FNS.square, 1);
    expect(f.secant).toBeCloseTo(2 + f.h, 10);
  });
  it('slopeCurve is deterministic and ends at the derivative', () => {
    const c = slopeCurve(FNS.exp, 0.5);
    expect(c).toEqual(slopeCurve(FNS.exp, 0.5));
    expect(c[c.length - 1].s).toBeCloseTo(Math.exp(0.5), 2);
  });
});

describe('gradient mode', () => {
  it('bowlGrad matches numerical partials', () => {
    const [x, y, e] = [1.2, -0.8, 1e-6], [gx, gy] = bowlGrad(x, y);
    expect(gx).toBeCloseTo((bowl(x + e, y) - bowl(x - e, y)) / (2 * e), 5);
    expect(gy).toBeCloseTo((bowl(x, y + e) - bowl(x, y - e)) / (2 * e), 5);
  });
  it('the ball descends monotonically', () => {
    const p = ballPath(GRAD_START);
    expect(p).toHaveLength(N_STEPS + 1);
    for (let i = 1; i < p.length; i++) expect(bowl(...p[i])).toBeLessThan(bowl(...p[i - 1]));
  });
  it('frameAt endpoints', () => {
    const a = gradAt(0, GRAD_START), b = gradAt(GRAD_T, GRAD_START), p = ballPath(GRAD_START);
    expect([a.ex, a.ey, a.eg]).toEqual([0, 0, 0]);
    expect(a.ball.alpha).toBe(0);
    expect(b.ex).toBe(1); expect(b.eg).toBe(1);
    expect(b.ball.x).toBeCloseTo(p[N_STEPS][0], 10);
    expect(b.ball.y).toBeCloseTo(p[N_STEPS][1], 10);
  });
});
