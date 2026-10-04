import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import * as S from '../../src/components/widgets/ScalingLab/sim';

const data: S.Sweep = JSON.parse(readFileSync('public/models/scaling-sweep.json', 'utf8'));
const MODES: S.Mode[] = ['sweep', 'compute', 'params-data', 'isoflop', 'published', 'extrapolate', 'overtrain'];

describe('measured sweep', () => {
  test('has 7 widths x 6 data budgets and C = 6ND', () => {
    expect(data.runs).toHaveLength(42);
    for (const r of data.runs) expect(r.C).toBe(6 * r.N * r.D);
  });
  test('frontier is the running minimum and agrees with the shipped summary', () => {
    const env = S.frontier(data.runs);
    for (let i = 1; i < env.length; i++) { expect(env[i].x).toBeGreaterThan(env[i - 1].x); expect(env[i].y).toBeLessThan(env[i - 1].y); }
    expect(env).toHaveLength(data.fit.frontier.length);
    env.forEach((p, i) => { expect(p.x).toBe(data.fit.frontier[i].C); expect(p.y).toBe(data.fit.frontier[i].L); });
  });
  test('power-law fit recovers an exact synthetic law', () => {
    const f0 = { E: 1.2, a: 6, b: 0.08 };
    const pts = [1e7, 1e8, 1e9, 1e10, 1e11].map((x) => ({ x, y: S.predict(f0, x) }));
    const f = S.fitPower(pts, true);
    expect(f.b).toBeCloseTo(0.08, 2);
    expect(f.E).toBeCloseTo(1.2, 1);
    expect(S.fitAt(pts, 1.2).b).toBeCloseTo(0.08, 10);
    expect(S.fitAt(pts, 1.2).r2).toBeCloseTo(1, 10);
  });
  test('fit on our sweep: floor 0.887, exponent 0.0887; without floor 0.0583; shipped summary agrees closely', () => {
    const env = S.frontier(data.runs), f = S.fitPower(env, true), f0 = S.fitPower(env, false);
    expect(f.E).toBeCloseTo(0.887, 3);
    expect(f.b).toBeCloseTo(0.0887, 4);
    expect(f.r2).toBeCloseTo(0.9616, 4);
    expect(f0.b).toBeCloseTo(0.0583, 4);
    expect(Math.abs(f.E - data.fit.computeFit.E)).toBeLessThan(0.01);
    expect(Math.abs(f.b - data.fit.computeFit.b)).toBeLessThan(0.001);
    expect(Math.abs(f0.b - data.fit.pureLogLog.b)).toBeLessThan(0.001);
    expect(S.predict(f, 1e15)).toBeCloseTo(1.43, 2);
    expect(S.predict(f0, 1e15)).toBeCloseTo(1.23, 2);
  });
  test('extrapolating from 4 smallest runs overpredicts the largest by 0.645; from 8 it is within 0.01', () => {
    expect(S.extrapolationErrors(data, 4, true).errors.at(-1)!.error).toBeCloseTo(0.645, 3);
    expect(Math.abs(S.extrapolationErrors(data, 8, true).errors.at(-1)!.error)).toBeLessThan(0.01);
  });
});

describe('IsoFLOP', () => {
  test('exact runs are marked measured, others interpolate between neighbours and never extrapolate', () => {
    const r = data.runs.find((q) => q.N === 5054 && q.D === 67584)!;
    const exact = S.isoPoints(data.runs, r.C).find((p) => p.N === 5054)!;
    expect(exact.measured).toBe(true);
    expect(exact.loss).toBe(r.validLoss);
    const pts = S.isoPoints(data.runs, 5.0119e9);
    expect(pts.length).toBeGreaterThan(3);
    for (const p of pts) {
      expect(p.D).toBeLessThanOrEqual(270592 * (1 + 1e-9));
      expect(p.D).toBeGreaterThanOrEqual(8448 * (1 - 1e-9));
      const rs = data.runs.filter((q) => q.N === p.N).map((q) => q.validLoss);
      expect(p.loss).toBeGreaterThanOrEqual(Math.min(...rs));
      expect(p.loss).toBeLessThanOrEqual(Math.max(...rs));
    }
    expect(S.isoPoints(data.runs, 1e12)).toHaveLength(0);
  });
  test('fitted surface recovers a synthetic one and fits our sweep to under 0.05 nats', () => {
    const truth = { E: 1, A: 5, B: 6, alpha: 0.3, beta: 0.2 };
    const synth = data.runs.map((r) => ({ ...r, validLoss: S.surfaceLoss(truth, r.N, r.D) }));
    const f = S.fitSurface(synth);
    expect(f.rmse).toBeLessThan(1e-3);
    expect(f.alpha).toBeCloseTo(0.3, 1);
    expect(f.beta).toBeCloseTo(0.2, 1);
    const real = S.sharedSurface(data);
    expect(real.rmse).toBeLessThan(0.05);
    expect(real.alpha).toBeCloseTo(0.336, 2);
    expect(real.beta).toBeCloseTo(0.108, 2);
    expect(S.sharedSurface(data)).toBe(real);
  });
  test('closed-form optimum is the minimum of L(N, C/6N)', () => {
    const s = S.sharedSurface(data), C = 1e10, o = S.surfaceOptimum(s, C);
    for (const k of [0.8, 0.95, 1.05, 1.25]) expect(S.surfaceLoss(s, o.N * k, C / (6 * o.N * k))).toBeGreaterThan(o.loss);
  });
});

describe('published fits', () => {
  test('constants match the papers', () => {
    expect(S.PUBLISHED.hoffmann).toMatchObject({ E: 1.69, A: 406.4, B: 410.7, alpha: 0.34, beta: 0.28 });
    expect(S.PUBLISHED.besiroglu).toMatchObject({ E: 1.8172, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658 });
    expect(S.KAPLAN).toMatchObject({ Ne: 1.3e9, pN: 0.73, pD: 0.27 });
    expect(S.CHINCHILLA_C).toBeCloseTo(5.88e23, -21);
  });
  test('optimalN minimises the formula under 6ND = C', () => {
    for (const p of Object.values(S.PUBLISHED)) {
      const C = 5.88e23, N = S.optimalN(C, p), L = S.publishedLoss(N, C / (6 * N), p);
      for (const k of [0.7, 0.9, 1.1, 1.4]) expect(S.publishedLoss(N * k, C / (6 * N * k), p)).toBeGreaterThan(L);
    }
  });
  test('the 2024 refit puts the Chinchilla budget at about 73B parameters on 1.3T tokens; the printed 2022 constants give 32B', () => {
    const C = S.CHINCHILLA_C, b = S.PUBLISHED.besiroglu, h = S.PUBLISHED.hoffmann;
    expect(S.optimalN(C, b) / 1e9).toBeCloseTo(73, 0);
    expect(C / 6 / S.optimalN(C, b) / S.optimalN(C, b)).toBeCloseTo(18.4, 1);
    expect(S.optimalN(C, h) / 1e9).toBeCloseTo(32.5, 0);
    expect(S.allocExponent(b)).toBeCloseTo(0.513, 3);
    expect(S.allocExponent(h)).toBeCloseTo(0.452, 3);
    expect(S.kaplanN(C) / 1e9).toBeCloseTo(816, 0);
    const nK = S.kaplanN(C);
    expect(S.publishedLoss(nK, C / (6 * nK), b) - S.publishedLoss(S.optimalN(C, b), C / (6 * S.optimalN(C, b)), b)).toBeCloseTo(0.063, 3);
  });
  test('tokensForLoss inverts the formula and is infinite below the minimum size', () => {
    const p = S.PUBLISHED.besiroglu, N = 3e10, D = 2e12, L = S.publishedLoss(N, D, p);
    expect(S.tokensForLoss(N, L, p) / D).toBeCloseTo(1, 9);
    expect(S.tokensForLoss(S.minParams(L, p) * 0.99, L, p)).toBe(Infinity);
  });
  test('serving shifts the cheapest model smaller and longer trained', () => {
    const p = S.PUBLISHED.besiroglu, target = S.overtrainTarget(p);
    expect(target).toBeCloseTo(1.974, 3);
    const o0 = S.servingOptimum(0, target, p), o12 = S.servingOptimum(1e12, target, p), o13 = S.servingOptimum(1e13, target, p);
    expect(o0.N / 1e9).toBeCloseTo(73, 0);
    expect(o0.ratio).toBeCloseTo(18.4, 1);
    expect(o13.N).toBeLessThan(o12.N);
    expect(o12.N).toBeLessThan(o0.N);
    expect(o13.ratio).toBeGreaterThan(o12.ratio);
    expect(o13.N / 1e9).toBeCloseTo(34.5, 1);
    expect(o13.cost).toBeLessThan(S.totalCost(o0.N, target, 1e13, p));
    expect(o13.cost).toBeCloseTo(1.466e24, -21);
  });
});

describe('scenes and frames', () => {
  test('measured modes need the sweep; published modes do not', () => {
    for (const m of MODES) expect(S.makeScene(m, null, S.DEFAULTS) === null).toBe(m !== 'published' && m !== 'overtrain');
  });
  test('every mode: frames are deterministic, finite, under 300 marks, and the phase matches a marker', () => {
    for (const m of MODES) {
      const sc = S.makeScene(m, data, S.DEFAULTS)!;
      for (let t = 0; t <= sc.duration + 1e-9; t += 0.25) {
        const f = S.frameAt(t, sc), g = S.frameAt(t, sc);
        expect(f).toEqual(g);
        expect(f.lines.length + f.dots.length + f.labels.length).toBeLessThan(300);
        for (const l of f.lines) for (const p of l.pts) { expect(Number.isFinite(p.x)).toBe(true); expect(Number.isFinite(p.y)).toBe(true); }
        expect(sc.markers.map((k) => k.label)).toContain(f.phase);
        expect(f.readout.length).toBeGreaterThan(10);
      }
      expect(S.frameAt(-5, sc)).toEqual(S.frameAt(0, sc));
      expect(S.frameAt(999, sc)).toEqual(S.frameAt(sc.duration, sc));
    }
  });
  test('sweep: 42 curves at the end, envelope dots appear in compute order, readout lowest loss is real', () => {
    const sc = S.makeScene('sweep', data, S.DEFAULTS)!, end = S.frameAt(12, sc);
    expect(end.lines.filter((l) => l.color !== 'var(--accent)')).toHaveLength(42);
    expect(end.dots).toHaveLength(S.frontier(data.runs).length);
    expect(end.readout).toContain('2.1385');
    expect(S.frameAt(2, sc).dots).toHaveLength(0);
    const early = S.frameAt(2, sc).readout, soFar = +/so far: ([\d.]+)/.exec(early)![1];
    expect(soFar).toBeGreaterThan(2.1385);
    const h = S.frameAt(12, sc, 41);
    expect(h.readout).toContain(String(data.runs[41].validLoss.toFixed(4)));
  });
  test('compute: the animated search ends at the best fit', () => {
    const sc = S.makeScene('compute', data, S.DEFAULTS)!, f0 = S.frameAt(0, sc), f1 = S.frameAt(12, sc);
    expect(f0.readout).toContain('E = 0.000');
    expect(f1.readout).toContain('E = 0.887');
    expect(f1.readout).toContain('b = 0.0887');
    const off = S.makeScene('compute', data, { ...S.DEFAULTS, floor: false })!;
    expect(S.frameAt(5, off).readout).toContain('b = 0.0583');
  });
  test('params-data: readout only quotes points that are drawn', () => {
    const sc = S.makeScene('params-data', data, S.DEFAULTS)!;
    expect(S.frameAt(sc.duration, sc).readout).toContain('2.4873');
    expect(S.frameAt(sc.duration, sc).readout).toContain('2.2993');
    expect(S.frameAt(4.1, sc).readout).toContain('still being drawn');
    expect(S.frameAt(4.1, sc).readout).not.toContain('2.5501');
    const sw = S.makeScene('params-data', data, { ...S.DEFAULTS, swap: true })!;
    expect(sw.markers).toHaveLength(7);
  });
  test('extrapolate: held-out points appear one by one', () => {
    const sc = S.makeScene('extrapolate', data, S.DEFAULTS)!;
    const count = (t: number) => S.frameAt(t, sc).dots.filter((d) => d.hollow).length;
    expect(count(5)).toBe(0);
    expect(count(12)).toBe(11);
    expect(count(8)).toBeGreaterThan(0);
    expect(count(8)).toBeLessThan(11);
    expect(S.frameAt(12, sc).readout).toContain('+0.645');
  });
  test('published: budget sweeps to Chinchilla budget, holds, then grows', () => {
    expect(S.publishedBudget(0)).toBeCloseTo(1e21, -15);
    expect(S.publishedBudget(4)).toBeCloseTo(S.CHINCHILLA_C, -15);
    expect(S.publishedBudget(8)).toBeCloseTo(S.CHINCHILLA_C, -15);
    expect(S.publishedBudget(12)).toBeCloseTo(3e24, -18);
    const sc = S.makeScene('published', data, S.DEFAULTS)!;
    expect(S.frameAt(6, sc).readout).toContain('73B parameters on 1.3T tokens, predicted loss 1.974');
    expect(S.frameAt(6, sc).readout).toContain('816B');
  });
  test('contours lie on their level', () => {
    const p = S.PUBLISHED.besiroglu;
    for (const lv of S.PUB_LEVELS) for (const q of S.contour(lv, p)) expect(S.publishedLoss(q.x, q.y, p)).toBeCloseTo(lv, 9);
  });
  test('overtrain: no serving for the first 3 s, then serving grows to 1e13', () => {
    expect(S.servedAt(2.9)).toBe(0);
    expect(S.servedAt(3)).toBeCloseTo(1e9, -2);
    expect(S.servedAt(10)).toBeCloseTo(1e13, -6);
    const sc = S.makeScene('overtrain', data, S.DEFAULTS)!;
    expect(S.frameAt(0, sc).readout).toContain('73B parameters');
    expect(S.frameAt(12, sc).readout).toContain('34.5B parameters');
  });
});

describe('formatting', () => {
  test('sci and compact', () => {
    expect(S.sci(5.88e23, 2)).toBe('5.9×10²³');
    expect(S.sci(9.99e9, 2)).toBe('1×10¹⁰');
    expect(S.compact(70e9)).toBe('70B');
    expect(S.compact(1.4e12)).toBe('1.4T');
    expect(S.compact(2614)).toBe('2.6k');
  });
});
