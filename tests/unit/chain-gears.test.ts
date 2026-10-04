import { describe, it, expect } from 'vitest';
import { SCENARIOS, evalScenario, numericDeriv, pathProducts, frameAt, duration, extentOf, summary } from '../../src/components/widgets/ChainGears/sim';

const ids = ['two', 'long', 'paths'] as const;

describe('ChainGears sim', () => {
  for (const id of ids) {
    const s = SCENARIOS[id];
    it(`${id}: dy/dx and every node gain match numerical derivatives`, () => {
      for (const x of [s.xRange[0], 0.3, s.x0, s.xRange[1]]) {
        const e = evalScenario(s, x);
        expect(e.dydx).toBeCloseTo(numericDeriv(s, x), 4);
        s.nodes.forEach((_, i) => {
          const sub = { ...s, nodes: s.nodes.slice(0, i + 1) };
          expect(e.gain[i]).toBeCloseTo(numericDeriv(sub, x), 4);
        });
        const total = pathProducts(s, x).reduce((a, p) => a + p.reduce((m, r) => m * r, 1), 0);
        expect(total).toBeCloseTo(e.dydx, 10);
      }
    });
    it(`${id}: frameAt endpoints and determinism`, () => {
      const dx = 0.02, T = duration(s);
      const a = frameAt(0, s, s.x0, dx), b = frameAt(T, s, s.x0, dx);
      expect(a.amp.slice(1).every((v) => v === 0)).toBe(true);
      expect(a.amp[0]).toBe(0);
      const e = evalScenario(s, s.x0);
      b.amp.forEach((v, i) => expect(v).toBeCloseTo(e.gain[i] * dx, 12));
      expect(b.total).toBe(1);
      expect(JSON.stringify(frameAt(3.3, s, s.x0, dx))).toBe(JSON.stringify(frameAt(3.3, s, s.x0, dx)));
      expect(extentOf(s, s.x0, dx)).toBeGreaterThan(0);
    });
    it(`${id}: a small real nudge changes y by about dy/dx times dx`, () => {
      const sm = summary(s, s.x0, 1e-4);
      expect(sm.actualSlope).toBeCloseTo(sm.dydx, 2);
    });
  }
  it('lesson numbers', () => {
    const two = evalScenario(SCENARIOS.two, 1);
    expect(two.vals).toEqual([1, 3, 9]);
    expect(two.rates[1][0]).toBe(3);
    expect(two.rates[2][0]).toBe(6);
    expect(two.dydx).toBe(18);
    const long = evalScenario(SCENARIOS.long, 0.5);
    expect(long.gain[4]).toBeCloseTo(-4, 12);
    const paths = pathProducts(SCENARIOS.paths, 2).map((p) => p.reduce((m, r) => m * r, 1));
    expect(paths).toEqual([24, 12]);
    expect(evalScenario(SCENARIOS.paths, 2).dydx).toBe(36);
  });
  it('paths: contributions stack to the total nudge at y', () => {
    const s = SCENARIOS.paths, f = frameAt(duration(s), s, 2, 0.02);
    expect(f.cum[3]).toEqual([0, expect.closeTo(0.48, 12), expect.closeTo(0.72, 12)]);
  });
});
