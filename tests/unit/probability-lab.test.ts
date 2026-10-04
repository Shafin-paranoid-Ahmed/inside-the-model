import { describe, expect, it } from 'vitest';
import {
  N, T_FILL, BINS, makeData, frameAt, countAt, trueMean, trueVar, normalPdf,
  bayesCounts, bayesFormula, bayesLayout, bayesFrameAt, POP, COLS, ROWS, type Cfg,
} from '../../src/components/widgets/ProbabilityLab/sim';

const coin: Cfg = { mode: 'bernoulli', p: 0.3, mu: 0, sigma: 1 };
const gauss: Cfg = { mode: 'gaussian', p: 0.5, mu: 1, sigma: 1.2 };

describe('sampling', () => {
  it('is deterministic', () => {
    expect(Array.from(makeData(gauss).xs)).toEqual(Array.from(makeData(gauss).xs));
  });
  it('frameAt endpoints: empty at 0, all samples at the end', () => {
    const d = makeData(gauss), f0 = frameAt(d, 0), f1 = frameAt(d, T_FILL);
    expect(f0.k).toBe(0);
    expect(f0.fall).toBe(-1);
    expect(f1.k).toBe(N);
    expect(countAt(T_FILL)).toBeCloseTo(N, 9);
    expect(f1.counts.length).toBe(BINS);
  });
  it('coin: counts add up and the mean is heads / n', () => {
    const f = frameAt(makeData(coin), 8);
    expect(f.counts[0] + f.counts[1]).toBe(f.k);
    expect(f.mean).toBeCloseTo(f.counts[1] / f.k, 12);
  });
  it('running mean and variance converge to the truth', () => {
    for (const c of [coin, gauss]) {
      const f = frameAt(makeData(c), T_FILL);
      expect(Math.abs(f.mean - trueMean(c))).toBeLessThan(4 * Math.sqrt(trueVar(c) / N));
      expect(Math.abs(f.variance - trueVar(c))).toBeLessThan(0.15 * trueVar(c) + 0.02);
    }
  });
  it('the same draws are reshaped by mu and sigma', () => {
    const a = makeData({ ...gauss, mu: 0, sigma: 1 }), b = makeData({ ...gauss, mu: 2, sigma: 0.5 });
    expect(b.xs[7]).toBeCloseTo(2 + 0.5 * a.xs[7], 12);
  });
  it('pdf integrates to 1', () => {
    let s = 0;
    for (let x = -8; x < 8; x += 0.01) s += normalPdf(x, 0.5, 1.3) * 0.01;
    expect(s).toBeCloseTo(1, 3);
  });
});

describe('bayes', () => {
  const c = { prevalence: 0.01, sensitivity: 0.9, fpr: 0.09 };
  it('counts partition the population', () => {
    const k = bayesCounts(c);
    expect(k.tp + k.fn + k.fp + k.tn).toBe(POP);
    expect([k.sick, k.tp, k.fp]).toEqual([10, 9, 89]);
  });
  it('dot fraction matches Bayes rule', () => {
    expect(bayesCounts(c).posterior).toBeCloseTo(bayesFormula(c), 1);
    expect(bayesFormula(c)).toBeCloseTo(0.0917, 3);
  });
  it('layout: unique grid cells, flags match counts, cluster stays in bounds', () => {
    const l = bayesLayout(c), cells = new Set<number>();
    let pos = 0, tp = 0;
    for (let i = 0; i < POP; i++) {
      cells.add(Math.floor(l.y0[i]) * COLS + Math.floor(l.x0[i]));
      if (l.flags[i] & 2) {
        pos++;
        if (l.flags[i] & 1) tp++;
        expect(l.x1[i]).toBeGreaterThan(0);
        expect(l.x1[i]).toBeLessThan(COLS);
        expect(l.y1[i]).toBeLessThan(ROWS);
      }
    }
    expect(cells.size).toBe(POP);
    expect(pos).toBe(98);
    expect(tp).toBe(9);
  });
  it('frame endpoints', () => {
    expect(bayesFrameAt(0)).toEqual({ appear: 0, sick: 0, test: 0, leave: 0, pack: 0 });
    expect(bayesFrameAt(14).pack).toBe(1);
  });
});
