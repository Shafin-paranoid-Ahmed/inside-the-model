import { describe, it, expect } from 'vitest';
import { stepUnit, bumpFrameAt, sumStages, firstWithin, depthStages, deepAt, tent, countPieces, shallowMin, deepUnits, stagedFrameAt, stageClock, durationOf, markersOf, frameAt, XS, NG, SMOOTH, SPIKY, TOL, SUM_NS, LIM_NS, DEPTHS, T_BUMP, unitsPerBump } from '../../src/components/widgets/BumpBuilder/sim';

describe('BumpBuilder sim', () => {
  it('step units go 0 -> 1 around their centre', () => {
    for (const act of ['sigmoid', 'relu'] as const) {
      expect(stepUnit(act, -1, 0.5, 20)).toBeLessThan(0.001);
      expect(stepUnit(act, 2, 0.5, 20)).toBeGreaterThan(0.999);
      expect(stepUnit(act, 0.5, 0.5, 20)).toBeCloseTo(0.5, 12);
    }
    expect(stepUnit('sigmoid', 0.6, 0.5, 8)).toBeCloseTo(1 / (1 + Math.exp(-0.8)), 12);
  });
  it('a sigmoid pair forms a bump that sharpens into a box', () => {
    const lo = bumpFrameAt(7, 'sigmoid'), hi = bumpFrameAt(T_BUMP, 'sigmoid');
    const mid = (NG - 1) / 2;
    expect(hi.diff[mid]).toBeGreaterThan(0.999);
    expect(hi.diff[0]).toBeLessThan(1e-4); expect(hi.diff[NG - 1]).toBeLessThan(1e-4);
    expect(lo.peak).toBeLessThan(hi.peak);
    expect(bumpFrameAt(0).hasB).toBe(false); expect(bumpFrameAt(6.5).hasB).toBe(true);
    expect(bumpFrameAt(T_BUMP).b).toBeCloseTo(0.65, 12);
  });
  it('ReLU pairs form a trapezoid (flat top, linear sides) and units are counted', () => {
    const f = bumpFrameAt(T_BUMP, 'relu'), at = (x: number) => f.diff[Math.round(x * (NG - 1))];
    expect(at(0.5)).toBeCloseTo(1, 12); expect(at(0.2)).toBeCloseTo(0, 12); expect(at(0.8)).toBeCloseTo(0, 12);
    expect(f.units).toBe(4); expect(bumpFrameAt(T_BUMP, 'sigmoid').units).toBe(2); expect(unitsPerBump('relu')).toBe(4);
    const w = 4.4 / f.k; // the ramp is linear across [a - w/2, a + w/2]
    expect(at(0.35)).toBeCloseTo(0.5, 2); expect(at(0.35 + w / 4)).toBeCloseTo(0.75, 2);
  });
  it('more bumps never use more error than the coarsest, and error falls overall', () => {
    for (const act of ['sigmoid', 'relu'] as const) {
      const st = sumStages('sum', act);
      expect(st.map((s) => s.n)).toEqual(SUM_NS);
      expect(st[st.length - 1].maxErr).toBeLessThan(st[0].maxErr / 5);
      expect(st[st.length - 1].mse).toBeLessThan(st[0].mse / 100);
      expect(st[2].units).toBe(4 * unitsPerBump(act));
      for (const s of st) expect(s.bumps.length).toBe(s.n);
    }
  });
  it('the regrouped window sum equals the plain sum of N bumps', () => {
    const st = sumStages('sum', 'sigmoid')[3]; // N = 8
    for (const g of [0, 100, 777, 1500, NG - 1]) {
      const plain = st.bumps.reduce((a, b) => a + b[g], 0);
      expect(st.curve[g]).toBeCloseTo(plain, 3);
    }
  });
  it('a one-bump-per-cell sum is the staircase of the target at cell centres', () => {
    const st = sumStages('sum', 'sigmoid')[2]; // N = 4, away from edges the value is f(centre)
    expect(st.curve[Math.round(0.125 * (NG - 1))]).toBeCloseTo(SMOOTH(0.125), 1);
    expect(st.curve[Math.round(0.625 * (NG - 1))]).toBeCloseTo(SMOOTH(0.625), 1);
  });
  it('a narrow spike needs far more bumps for the same tolerance', () => {
    for (const act of ['sigmoid', 'relu'] as const) {
      const smooth = firstWithin(sumStages('limitsSmooth', act), TOL)!, spike = firstWithin(sumStages('limits', act), TOL)!;
      expect(smooth.n).toBe(64); expect(spike.n).toBe(1024);
      expect(spike.n / smooth.n).toBeGreaterThanOrEqual(16);
    }
    expect(SPIKY(0.62) - SMOOTH(0.62)).toBeGreaterThan(0.49);
    expect(LIM_NS[LIM_NS.length - 1]).toBe(1024);
  });
  it('tent layers: pieces double with depth, units grow by two', () => {
    expect(tent(0)).toBe(0); expect(tent(0.5)).toBe(1); expect(tent(1)).toBe(0); expect(tent(0.25)).toBe(0.5);
    expect(depthStages().map((d) => d.pieces)).toEqual(DEPTHS.map((k) => 2 ** k));
    expect(DEPTHS.map(deepUnits)).toEqual([2, 4, 6, 8, 10]);
    expect(DEPTHS.map(shallowMin)).toEqual([1, 3, 7, 15, 31]);
    expect(deepAt(3, 0.1)).toBeCloseTo(0.8, 12); // 0.1 -> 0.2 -> 0.4 -> 0.8
  });
  it('countPieces matches known shapes', () => {
    expect(countPieces([0, 1, 2, 3])).toBe(1);
    expect(countPieces([0, 1, 0, 1, 0])).toBe(4);
    const xs = Array.from(XS);
    expect(countPieces(xs.map((x) => Math.abs(x - 0.5)))).toBe(2);
  });
  it('frameAt endpoints, phases and determinism', () => {
    for (const mode of ['bump', 'sum', 'limits', 'depth'] as const) {
      const T = durationOf(mode), ms = markersOf(mode);
      expect(ms[0].at).toBe(0);
      expect(frameAt(mode, 0).phase).toBe(ms[0].label);
      expect(frameAt(mode, T).phase).toBe(ms[ms.length - 1].label);
      for (const m of ms) expect(frameAt(mode, m.at + 0.01).phase).toBe(m.label);
      const f = frameAt(mode, T / 2), repeated = frameAt(mode, T / 2);
      expect(f.kind === 'bump' ? f.diff : f.curve).toEqual(repeated.kind === 'bump' ? repeated.diff : repeated.curve);
    }
    expect(stagedFrameAt('depth', 0).pieces).toBe(2); expect(stagedFrameAt('depth', durationOf('depth')).pieces).toBe(32);
    expect(stagedFrameAt('sum', durationOf('sum')).n).toBe(32);
    expect(stagedFrameAt('limits', durationOf('limits')).n).toBe(1024);
  });
  it('at transitions the curve, metrics and unit counts describe the same real network', () => {
    const D = 2.4, a = stagedFrameAt('depth', D - 0.2), b = stagedFrameAt('depth', D + 0.2);
    expect(a.n).toBe(1); expect(b.n).toBe(2);
    expect(a.curve).toEqual(depthStages()[0].curve);
    expect(b.curve).toEqual(depthStages()[1].curve);
    expect(countPieces(a.curve)).toBe(a.pieces); expect(countPieces(b.curve)).toBe(b.pieces);
    expect(stageClock(D, 5, D, 0.5).reveal).toBe(0);
    expect(stageClock(1.0, 5, D, 0.5)).toMatchObject({ a: 0, b: 0, w: 0 });
    const mid = stagedFrameAt('depth', D);
    expect(mid.curve).toEqual(depthStages()[1].curve);
    for (const t of [D - 0.2, D, D + 0.2]) {
      const f = stagedFrameAt('sum', t), st = sumStages('sum', 'sigmoid')[f.s];
      expect(f.curve).toEqual(st.curve); expect(f.maxErr).toBe(st.maxErr); expect(f.units).toBe(st.units);
    }
  });
});
