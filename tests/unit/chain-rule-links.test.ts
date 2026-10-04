import { describe, it, expect } from 'vitest';
import { CHAINS, evalChain, numericDeriv, frameAt, linkView, T, REVEAL_AT } from '../../src/components/widgets/ChainRuleLinks/sim';

describe('ChainRuleLinks sim', () => {
  for (const id of ['wave', 'saturate'] as const) {
    it(`${id}: product of local slopes equals the numerical derivative`, () => {
      const c = CHAINS[id];
      for (const x of [-1.5, -0.7, 0, 0.3, 1, 1.6]) {
        const e = evalChain(c, x);
        expect(e.dLdx).toBeCloseTo(numericDeriv(c, x), 5);
        expect(e.prods[3]).toBeCloseTo(e.slopes[0] * e.slopes[1] * e.slopes[2], 12);
      }
    });
  }
  it('known numbers at x=1 on the wave chain', () => {
    const e = evalChain(CHAINS.wave, 1);
    expect(e.vals[1]).toBe(1);
    expect(e.vals[2]).toBeCloseTo(Math.sin(1), 12);
    expect(e.dLdx).toBeCloseTo(2 * Math.cos(1) * 2 * Math.sin(1), 12); // 1.8186
  });
  it('frameAt reveals in order, is deterministic, and endpoints are clean', () => {
    const c = CHAINS.wave;
    const a = frameAt(0, c, 1), b = frameAt(T, c, 1);
    expect(a.reveal).toHaveLength(4);
    expect(a.reveal.every((r) => r === 0)).toBe(true);
    expect(b.reveal.every((r) => r === 1)).toBe(true);
    expect(frameAt(REVEAL_AT[1] + 0.6, c, 1).reveal[0]).toBe(1);
    const productMidpoint = frameAt(REVEAL_AT[3] + 0.6, c, 1).reveal;
    expect(productMidpoint.slice(0, 3)).toEqual([1, 1, 1]);
    expect(productMidpoint[3]).toBeCloseTo(0.5, 12);
    expect(frameAt(3, c, 1)).toEqual(frameAt(3, c, 1));
    expect(a.pts[0]).toBeCloseTo(1, 12);
    for (const [start, end] of a.pts.map((value, i) => [value, b.pts[i]])) {
      expect(end).toBeCloseTo(start, 12);
    }
  });
  it('linkView domain contains every input the link sees', () => {
    const c = CHAINS.saturate;
    const v = linkView(c, 1);
    expect(v.d0).toBeLessThan(Math.tanh(0.3 * 2) * 0 + 2 * c.xRange[0] + 1e-9);
    expect(v.d1).toBeGreaterThan(2 * c.xRange[1] - 1e-9);
  });
});
