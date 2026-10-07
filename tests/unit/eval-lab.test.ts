import { describe, expect, it } from 'vitest';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import * as S from '../../src/components/widgets/EvalLab/sim';
import benchmark from '../../src/components/widgets/EvalLab/m-benchmark';
import arena from '../../src/components/widgets/EvalLab/m-arena';
import contamination from '../../src/components/widgets/EvalLab/m-contamination';
import goodhart, { type Scene as GoodhartScene } from '../../src/components/widgets/EvalLab/m-goodhart';
import saturation, { type Scene as SatScene } from '../../src/components/widgets/EvalLab/m-saturation';
import errorbars, { type Scene as ErrScene } from '../../src/components/widgets/EvalLab/m-errorbars';

const nodeKit = (): Kit => {
  const m = new Map<string, unknown>();
  return { memo: <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T, job: () => Promise.reject(new Error('no worker')) };
};
const ui = { pick: null };
const MODES: [string, ModeImpl<unknown, Frame>][] = [
  ['benchmark', benchmark], ['arena', arena], ['contamination', contamination], ['goodhart', goodhart], ['saturation', saturation], ['errorbars', errorbars],
] as never;

describe('sim math', () => {
  it('normal cdf and quadrature are accurate', () => {
    expect(S.normCdf(0)).toBeCloseTo(0.5, 6);
    expect(S.normCdf(1.96)).toBeCloseTo(0.975, 4);
    expect(S.expectOver(0, 1, () => 1)).toBeCloseTo(1, 6);
    expect(S.expectOver(2, 0.5, (d) => d)).toBeCloseTo(2, 6);
  });
  it('popAcc matches a Monte Carlo average of the response model', () => {
    const rng = S.createRng(9); let k = 0; const N = 40000;
    for (let i = 0; i < N; i++) k += rng.next() < S.sigmoid(0.9 - rng.normal(0, 1.2)) ? 1 : 0;
    expect(Math.abs(k / N - S.popAcc(0.9))).toBeLessThan(0.012);
  });
  it('wilson interval is near p +- 1.96 SE at n = 200 and contains p', () => {
    const [lo, hi] = S.wilson(140, 200);
    expect(lo).toBeLessThan(0.7); expect(hi).toBeGreaterThan(0.7);
    expect(Math.abs((hi - lo) / 2 - 1.96 * S.binomSE(0.7, 200))).toBeLessThan(0.004);
    expect(S.wilson(0, 0)).toEqual([0, 1]);
  });
  it('Bradley-Terry recovers known strengths from many votes, with calibrated standard errors', () => {
    const ar = S.makeArena(2, 3000), r = S.arenaAt(ar, 3000, false), c = S.ARENA_LEN.indexOf(1);
    r.rating.forEach((x, i) => expect(Math.abs(x - S.toRating(S.ARENA_BETA[i]))).toBeLessThan(4 * (r.half[i] / 1.96)));
    expect(r.half[0]).toBeGreaterThan(10); expect(r.half[0]).toBeLessThan(40);
    expect(r.rating.reduce((a, b) => a + b, 0) / 6).toBeCloseTo(1000, 4);
    expect(S.arenaAt(ar, 3000, true).rank[c]).toBeLessThan(r.rank[c]);
  });
  it('wrongWinner matches simulated comparisons, own questions and same questions', () => {
    const s = errorbarsScene(), n = 150, trials = 3000, rng = S.createRng(5), sB = 1.15;
    let wrongOwn = 0, wrongSame = 0;
    const draw = () => rng.normal(0, 1.2), right = (skill: number, d: number) => (rng.next() < S.sigmoid(4 * (skill - d)) ? 1 : 0);
    for (let t = 0; t < trials; t++) {
      let a1 = 0, b1 = 0, a2 = 0, b2 = 0;
      for (let i = 0; i < n; i++) { a1 += right(1.0, draw()); b1 += right(sB, draw()); const d = draw(); a2 += right(1.0, d); b2 += right(sB, d); }
      wrongOwn += a1 > b1 ? 1 : a1 === b1 ? 0.5 : 0;
      wrongSame += a2 > b2 ? 1 : a2 === b2 ? 0.5 : 0;
    }
    expect(Math.abs(wrongOwn / trials - S.wrongWinner(s.delta, s.sdUnpaired, n))).toBeLessThan(0.03);
    expect(Math.abs(wrongSame / trials - S.wrongWinner(s.delta, s.sdPaired, n))).toBeLessThan(0.03);
  });
  it('itemsNeeded inverts wrongWinner', () => { expect(S.wrongWinner(0.03, 0.6, S.itemsNeeded(0.03, 0.6))).toBeCloseTo(0.05, 6); });
});

const errorbarsScene = () => errorbars.setup({ mode: 'errorbars' } as Opts, nodeKit()) as ErrScene;

describe.each(MODES)('mode %s', (name, mode) => {
  const o = { mode: name } as Opts;
  it('markers strictly increase inside the timeline', async () => {
    const s = await mode.setup(o, nodeKit()), { duration, markers } = mode.timeline(s, o);
    expect(markers.length).toBeGreaterThan(0);
    markers.forEach((m, i) => { expect(m.at).toBeGreaterThanOrEqual(0); expect(m.at).toBeLessThanOrEqual(duration); if (i) expect(m.at).toBeGreaterThan(markers[i - 1].at); });
  });
  it('frames narrate, draw within about 200 shapes and are deterministic', async () => {
    const [a, b] = [await mode.setup(o, nodeKit()), await mode.setup(o, nodeKit())], { duration } = mode.timeline(a, o);
    for (const t of [0, duration * 0.3, duration / 2, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(f.status.length).toBeGreaterThan(20);
      expect(mode.frameAt(t, b, o, ui)).toEqual(f);
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 360 : 720, H: narrow ? 560 : 400, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(0); expect(E.out.length).toBeLessThan(260);
        for (const sh of E.out) if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
      }
    }
  });
  it('setup stays well under 50 ms', async () => {
    const t0 = performance.now(); await mode.setup(o, nodeKit()); expect(performance.now() - t0).toBeLessThan(50);
  });
});

type Rich = Frame & Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
const frame = (mode: ModeImpl<unknown, Frame>, o: Opts, s: unknown, t: number) => mode.frameAt(t, s, o, ui) as Rich;

describe('numbers quoted in the lesson', () => {
  it('benchmark: scores, intervals and overlapping pairs at 200 questions (seed 1)', async () => {
    const o = { mode: 'benchmark' } as Opts, s = await benchmark.setup(o, nodeKit()), f = frame(benchmark as never, o, s, 12);
    expect(f.right).toEqual([96, 115, 111, 136, 146, 157]);
    expect(f.ints.map((iv: number[]) => iv.map((x) => x.toFixed(2)).join('-'))).toEqual(['0.41-0.55', '0.51-0.64', '0.49-0.62', '0.61-0.74', '0.66-0.79', '0.72-0.84']);
    expect(f.pairs.length).toBe(8);
    expect(f.status).toContain('8 of 15');
    expect(S.makeBench(1, 200).pop.map((x) => x.toFixed(3))).toEqual(['0.519', '0.568', '0.615', '0.669', '0.727', '0.793']);
  });
  it('arena: ratings, intervals and the rank of the wordy Model C at 3000 votes (seed 3)', () => {
    const ar = S.makeArena(3, 3000), fair = S.arenaAt(ar, 3000, false), bias = S.arenaAt(ar, 3000, true), c = 2;
    expect(fair.rating.map((x) => Math.round(x))).toEqual([847, 901, 980, 1024, 1088, 1160]);
    expect(fair.half.map((x) => Math.round(x))).toEqual([21, 20, 18, 19, 19, 21]);
    expect([fair.rank[c], bias.rank[c]]).toEqual([4, 2]);
    expect(Math.round(bias.rating[c])).toBe(1081);
    expect(Math.round(S.toRating(S.ARENA_BETA[c]))).toBe(974);
  });
  it('contamination: public and fresh scores at 0% and 20% leak (seed 1)', async () => {
    const o = { mode: 'contamination', leakPct: 20 } as Opts, s = await contamination.setup(o, nodeKit());
    const f0 = frame(contamination as never, o, s, 0), f20 = frame(contamination as never, o, s, 11);
    expect([f0.pubScore, f0.freshScore]).toEqual([0.6825, 0.6825]);
    expect(f20.pct).toBe(20); expect(f20.L).toBe(80);
    expect(f20.pubScore).toBeCloseTo(0.7425, 4);
  });
  it('goodhart: best of 50 on the public test against its private score (seed 2)', async () => {
    const o = { mode: 'goodhart' } as Opts, s = (await goodhart.setup(o, nodeKit())) as GoodhartScene, f = frame(goodhart as never, o, s, 15);
    expect(f.n).toBe(50); expect(s.pub[f.b] / 200).toBeCloseTo(0.665, 3); expect(s.priv[f.b] / 1000).toBeCloseTo(0.578, 3); expect(s.truth).toBeCloseTo(0.596, 3); expect(f.b + 1).toBe(16);
  });
  it('saturation: the easy test separates generations 1 to 2 only; ceiling 0.97 (seed 6)', async () => {
    const o = { mode: 'saturation' } as Opts, s = (await saturation.setup(o, nodeKit())) as SatScene;
    expect(s.ceiling).toBeCloseTo(0.97, 6); expect(s.wrongKeys).toBe(6);
    expect(s.easy.sep.map(Number)).toEqual([1, 1, 0, 0, 0, 0, 0, 0]);
    expect(Array.from(s.easy.right, (k) => k / 200)).toEqual([0.74, 0.89, 0.935, 0.965, 0.97, 0.97, 0.97, 0.97]);
  });
  it('saturation: the hard test scores and its separated generations (seed 6)', async () => {
    const s = (await saturation.setup({ mode: 'saturation' } as Opts, nodeKit())) as SatScene;
    expect(Array.from(s.hard.right, (k) => k / 200)).toEqual([0.025, 0.145, 0.16, 0.34, 0.585, 0.735, 0.855, 0.92]);
    expect(s.hard.sep.map(Number)).toEqual([1, 1, 0, 1, 1, 1, 1, 0]);
  });
  it('goodhart prose: standard error 0.035 and expected maximum of 50 normals 2.25', () => {
    expect(Math.sqrt((0.596 * 0.404) / 200)).toBeCloseTo(0.0347, 4);
    let em = 0; const h = 0.002;
    for (let x = -6; x < 8; x += h) em += x * 50 * S.normCdf(x) ** 49 * (Math.exp((-x * x) / 2) / Math.sqrt(2 * Math.PI)) * h;
    expect(em).toBeCloseTo(2.249, 2);
  });
  it('benchmark prose: 1.96 SE at p 0.7, n 200 is 0.063; contamination gap SE is 0.033; unpaired and paired figures at 200 questions', () => {
    expect(1.96 * S.binomSE(0.7, 200)).toBeCloseTo(0.0635, 3);
    expect(Math.sqrt(2 * 0.682 * 0.318 / 400)).toBeCloseTo(0.033, 3);
    const s = errorbarsScene();
    expect(Math.round(S.wrongWinner(s.delta, s.sdUnpaired, 200) * 100)).toBe(21);
    expect(Math.round(S.wrongWinner(s.delta, s.sdPaired, 200) * 100)).toBe(8);
    expect([s.bothRight, s.bothWrong, s.onlyB, s.onlyA].map((x) => Math.round(x * 100))).toEqual([74, 14, 7, 4]);
    expect([s.pA, s.pB].map((x) => x.toFixed(2))).toEqual(['0.78', '0.82']);
    expect([s.sdPaired, s.sdUnpaired].map((x) => x.toFixed(2))).toEqual(['0.33', '0.57']);
    expect([s.nUnpaired, s.nPaired].map((x) => Math.round(x / 10) * 10)).toEqual([810, 280]);
  });
  it('errorbars: items needed for fewer than 5% wrong winners', () => {
    const s = errorbarsScene();
    expect(Math.round(s.nUnpaired)).toBeGreaterThan(700); expect(Math.round(s.nPaired)).toBeLessThan(400);
    expect(s.bothRight + s.bothWrong + s.onlyA + s.onlyB).toBeCloseTo(1, 9);
  });
});
