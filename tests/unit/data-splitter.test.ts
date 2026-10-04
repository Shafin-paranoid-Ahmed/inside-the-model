import { describe, it, expect } from 'vitest';
import { makeRun, makeTune, pick, curveEpochs, uOfEpoch, sizes, frameAt, epochAt, EPOCHS, N, NCAND, T, T_SPLIT } from '../../src/components/widgets/DataSplitter/sim';

describe('DataSplitter sim', () => {
  it('splits sum to N and respect minimums', () => {
    for (const [a, b] of [[0.6, 0.2], [0.01, 0.01], [0.99, 0.5], [0.5, 0.5]]) {
      const s = sizes(a, b);
      expect(s.nTrain + s.nVal + s.nTest).toBe(N);
      expect(Math.min(s.nTrain, s.nVal, s.nTest)).toBeGreaterThanOrEqual(5);
    }
  });
  it('is deterministic', () => {
    const a = makeRun(0.3, 0.3), b = makeRun(0.3, 0.3);
    expect(Array.from(a.val)).toEqual(Array.from(b.val));
  });
  it('training loss never increases (step is below the stability limit)', () => {
    const r = makeRun(0.5, 0.25);
    for (let e = 1; e <= EPOCHS; e++) expect(r.train[e]).toBeLessThanOrEqual(r.train[e - 1] + 1e-12);
  });
  it('a small training set overfits: validation bottoms out, then rises above its minimum', () => {
    const r = makeRun(0.2, 0.3);
    expect(r.bestVal).toBeGreaterThan(0);
    expect(r.bestVal).toBeLessThan(EPOCHS);
    expect(r.val[EPOCHS]).toBeGreaterThan(r.val[r.bestVal]);
    expect(r.train[EPOCHS]).toBeLessThan(r.val[EPOCHS]);
  });
  it('loss-curve x positions are strictly increasing (clean left-to-right polyline)', () => {
    for (const e of [0, 0.4, 1, 7.5, 30, 30.5, 200, 999, EPOCHS]) {
      const xs = curveEpochs(e).map(uOfEpoch);
      for (let i = 1; i < xs.length; i++) expect(xs[i]).toBeGreaterThan(xs[i - 1]);
      expect(curveEpochs(e)[curveEpochs(e).length - 1]).toBeCloseTo(e, 6);
    }
  });
  it('tuning: peeking picks the best test score, which is optimistic vs fresh data on every split', () => {
    for (const [a, b] of [[0.4, 0.45], [0.3, 0.55], [0.5, 0.35], [0.6, 0.25], [0.35, 0.5]]) {
      const r = makeTune(a, b), k = pick(r, true), h = pick(r, false);
      expect(r.test.length).toBe(NCAND);
      expect(k.reported).toBe(Math.min(...r.test));
      expect(k.reported).toBeLessThanOrEqual(h.reported);
      expect(k.real).toBeGreaterThan(k.reported);
    }
  });
  it('tuning is deterministic', () => {
    expect(Array.from(makeTune(0.4, 0.45).test)).toEqual(Array.from(makeTune(0.4, 0.45).test));
  });
  it('frameAt: dots start in the pool, end in bins; epoch spans 0..EPOCHS', () => {
    const r = makeRun(0.6, 0.2);
    const f0 = frameAt(0, r), f1 = frameAt(T, r);
    expect(Array.from(f0.dots)).toEqual(Array.from(r.from));
    for (let i = 0; i < 2 * N; i++) expect(f1.dots[i]).toBeCloseTo(r.to[i], 5);
    expect(f0.phase).toBe('split');
    expect(epochAt(T_SPLIT)).toBe(0);
    expect(epochAt(T)).toBe(EPOCHS);
    expect(frameAt(T, r).phase).toBe('overfit');
  });
});
