import { describe, expect, test } from 'vitest';
import {
  trainFlow, samplePaths, exactVelFn, netVelFn, eulerPath, pathAtU, pathStats, startNoise, moonData, quality, MOONS, lossAt, lossFloor, spotVelocities,
  endPoints, modeTime, pathTime, STEP_COUNTS, T_SPOT, M_A, M_B, M_END, type Mode,
} from '../../src/components/widgets/FlowField/sim';
import { reversePath } from '../../src/components/widgets/DiffusionDenoiser/sim';
import { posteriorMean } from '../../src/lib/gen/toy';

describe('exact velocity field', () => {
  test('v* = (E[x1|x] - x)/(1 - t) and a one-step Euler run lands on the mean of the data', () => {
    const cur = Float64Array.of(0.4, -0.2), v = new Float64Array(2), e = [0, 0];
    exactVelFn(0.3, cur, 1, v); posteriorMean(MOONS, 0.3, 0.7, 0.4, -0.2, e);
    expect(v[0]).toBeCloseTo((e[0] - 0.4) / 0.7, 12); expect(v[1]).toBeCloseTo((e[1] + 0.2) / 0.7, 12);
    const z = startNoise().slice(0, 200), p = eulerPath(exactVelFn, z, 1), end = endPoints(p);
    let m = 0; for (let i = 0; i < 100; i++) m += Math.hypot(end[2 * i], end[2 * i + 1]); // near the centre, between the moons
    expect(m / 100).toBeLessThan(0.5);
  });
  test('with the exact field, 64 steps land on the data and 4 steps beat diffusion at the same budget', () => {
    const z = startNoise().slice(0, 400), on = (p: ReturnType<typeof eulerPath>) => quality(p.xs.slice(-400), MOONS).onData;
    expect(on(eulerPath(exactVelFn, z, 64))).toBeGreaterThan(0.95);
    expect(on(eulerPath(exactVelFn, z, 4))).toBeGreaterThan(on(reversePath(MOONS, z, 'ode', 4, 5)) + 0.2);
  });
  test('path length over straight distance (totals): flow paths are longer than diffusion paths, at both budgets', () => {
    const z = startNoise().slice(0, 400);
    for (const S of [4, 64]) {
      const f = pathStats(eulerPath(exactVelFn, z, S)).ratio, d = pathStats(reversePath(MOONS, z, 'ode', S, 5)).ratio;
      expect(d).toBeGreaterThanOrEqual(1); expect(f).toBeGreaterThan(d);
    }
  });
  test('paths: pathAtU hits the recorded states, stats are at least 1', () => {
    const p = eulerPath(exactVelFn, startNoise().slice(0, 100), 8), out = new Float32Array(100);
    pathAtU(p, 0, out); expect(out[0]).toBe(p.xs[0]);
    pathAtU(p, 1, out); expect(out[0]).toBe(p.xs[8 * 100]);
    expect(pathStats(p).ratio).toBeGreaterThanOrEqual(1);
  });
  test('the spot: paths through it point every which way, the field is their small average', () => {
    const a = spotVelocities(8), b = spotVelocities(3, 20000);
    let sp = 0; for (let i = 0; i < a.vs.length / 2; i++) sp += Math.hypot(a.vs[2 * i], a.vs[2 * i + 1]);
    expect(sp / (a.vs.length / 2)).toBeGreaterThan(3 * Math.hypot(...a.exact));
    let mx = 0, my = 0; for (let i = 0; i < b.vs.length / 2; i++) { mx += b.vs[2 * i]; my += b.vs[2 * i + 1]; }
    expect(mx / (b.vs.length / 2)).toBeCloseTo(a.exact[0], 1); expect(my / (b.vs.length / 2)).toBeCloseTo(a.exact[1], 1);
  });
});

describe('learned velocity network', () => {
  test('training is deterministic', () => {
    const a = trainFlow(40), b = trainFlow(40);
    expect(a.steps).toEqual([0, 25, 40]);
    expect(Array.from(a.snaps[2])).toEqual(Array.from(b.snaps[2]));
    expect(Array.from(a.losses)).toEqual(Array.from(b.losses));
  });
  test('a full run lowers the loss toward the floor, matches the exact field, and samples onto the moons', () => {
    const run = trainFlow(), paths = samplePaths(run.snaps[8]);
    expect(lossAt(run, 1799)).toBeLessThan(lossAt(run, 0) - 0.1);
    expect(lossAt(run, 1799)).toBeGreaterThan(lossFloor() - 0.02);
    expect(run.angles[0][8]).toBeLessThan(run.angles[0][0] / 3);
    expect(Object.keys(paths).map(Number)).toEqual(STEP_COUNTS);
    const on = (S: number) => quality(endPoints(paths[S]), MOONS).onData;
    expect(on(64)).toBeGreaterThan(0.45);
    expect(on(64)).toBeGreaterThan(on(1) + 0.3);
    const v = new Float64Array(2); netVelFn(run.snaps[8])(0.5, Float64Array.of(0, 0), 1, v);
    expect(Number.isFinite(v[0])).toBe(true);
    void moonData;
  }, 30000);
});

describe('timelines', () => {
  test('markers lie inside each mode and the marginal path time holds at the spot', () => {
    for (const m of ['flow', 'pairs', 'marginal', 'train', 'sample', 'compare'] as Mode[]) { const mt = modeTime(m); expect(mt.markers[0].at).toBe(0); for (const k of mt.markers) expect(k.at).toBeLessThan(mt.dur); }
    expect(pathTime('marginal', 0)).toBe(0); expect(pathTime('marginal', M_A)).toBeCloseTo(T_SPOT, 12); expect(pathTime('marginal', (M_A + M_B) / 2)).toBe(T_SPOT);
    expect(pathTime('marginal', M_END)).toBeCloseTo(1, 12); expect(pathTime('flow', 100)).toBe(1);
  });
});
