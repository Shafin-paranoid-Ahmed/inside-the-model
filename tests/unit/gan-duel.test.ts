import { describe, expect, test } from 'vitest';
import { createRng } from '../../src/lib/math/rng';
import { MOONS, quality } from '../../src/lib/gen/toy';
import {
  trainGAN, train, RUN_CFG, STAT, NG, ND, D_SIZES, generate, discriminate, pushDirections, gradDemo, GRAD_SNAP, GRAD_N,
  points, ringShares, smoothLoss, runFrame, runMarkers, genFrame, gradFrame, durationOf, noiseGrid, accuracy, T_RUN,
} from '../../src/components/widgets/GANDuel/sim';
import { initMlp, createCache, forward, sigmoid, log } from '../../src/lib/gen/mlp';

const lazy = <T>(f: () => T) => { let v: T | undefined; return () => (v ??= f()); };
const stable = lazy(() => train('stable')), unstable = lazy(() => train('unstable'));
const collapse = lazy(() => train('collapse')), ringOk = lazy(() => train('ringOk'));
const spread = (r: ReturnType<typeof train>) => {
  let lo = 1, hi = 0;
  for (let i = r.snaps.length >> 1; i < r.snaps.length; i++) { const v = r.stats[i * STAT]; lo = Math.min(lo, v); hi = Math.max(hi, v); }
  return { lo, hi };
};

describe('GAN trainer', () => {
  test('two runs with the same config give identical checkpoints and losses', () => {
    const cfg = { seed: 3, steps: 50, lrD: 0.005, lrG: 0.001, snapEvery: 25, gInit: 0.1 };
    const a = trainGAN(cfg), b = trainGAN(cfg);
    expect(a.snaps.length).toBe(3);
    expect(a.steps).toEqual([0, 25, 50]);
    for (let i = 0; i < a.snaps.length; i++) expect(Array.from(a.snaps[i])).toEqual(Array.from(b.snaps[i]));
    expect(Array.from(a.losses)).toEqual(Array.from(b.losses));
    expect(a.snaps[0].length).toBe(NG + ND);
  });

  test('the first discriminator loss is about log 4 and the first generator loss about log 2', () => {
    const r = trainGAN({ seed: 1, steps: 1, lrD: 0.005, lrG: 0.001, snapEvery: 1, gInit: 0.1 });
    expect(Math.abs(r.losses[0] - Math.log(4))).toBeLessThan(0.15);
    expect(Math.abs(r.losses[1] - Math.log(2))).toBeLessThan(0.1);
  });

  test('frozen generator: only the discriminator changes', () => {
    const r = trainGAN({ seed: 1, steps: 40, lrD: 0.005, lrG: 0, snapEvery: 40, freezeG: true, gInit: 0.1 });
    expect(Array.from(r.snaps[1].subarray(0, NG))).toEqual(Array.from(r.snaps[0].subarray(0, NG)));
    expect(Array.from(r.snaps[1].subarray(NG))).not.toEqual(Array.from(r.snaps[0].subarray(NG)));
  });

  test('the gradient pushDirections returns matches finite differences of log D', () => {
    const rng = createRng(5), snap = new Float64Array(NG + ND);
    snap.set(initMlp(D_SIZES, rng), NG);
    for (let i = NG; i < snap.length; i++) snap[i] += 0.2 * (rng.next() - 0.5);
    const p = Float64Array.from([0.4, -0.7, -1.1, 0.3]);
    const { d, g } = pushDirections(snap, p);
    const logD = (q: Float64Array) => { const c = createCache(D_SIZES, 2); const o = forward(snap.subarray(NG), D_SIZES, q, 2, c); return [log(sigmoid(o[0])), log(sigmoid(o[1]))]; };
    const h = 1e-6;
    for (let k = 0; k < 4; k++) {
      const a = p.slice(), b = p.slice(); a[k] += h; b[k] -= h;
      expect(g[k]).toBeCloseTo((logD(a)[k >> 1] - logD(b)[k >> 1]) / (2 * h), 6);
    }
    expect(d[0]).toBeCloseTo(discriminate(snap, p)[0], 12);
  });
});

describe('the runs the lesson uses (numbers quoted in the prose)', { timeout: 60000 }, () => {
  test('stable: starts off the data, ends on both moons, with the discriminator near chance', () => {
    const r = stable(), n = r.snaps.length, S = STAT;
    expect(n).toBe(61);
    expect(r.stats[0]).toBeLessThan(0.05);
    const end = quality(points(r.snaps[n - 1]), MOONS);
    expect(r.stats[(n - 1) * S]).toBeCloseTo(end.onData, 6);
    expect(end.onData).toBeGreaterThan(0.75);
    expect(Math.abs(end.upper - 0.5)).toBeLessThan(0.1);
    expect(r.stats[(n - 1) * S + 5]).toBeLessThan(0.65); // accuracy
    expect(accuracy(r.snaps[n - 1])).toBeCloseTo(r.stats[(n - 1) * S + 5], 6);
    const { lo } = spread(r);
    expect(lo).toBeGreaterThan(0.55);
  });

  test('stable: the two losses stay near log 4 and log 2 instead of falling', () => {
    const r = stable(), d = smoothLoss(r.losses, 0), g = smoothLoss(r.losses, 1);
    let sd = 0, sg = 0;
    for (let i = 500; i < d.length; i++) { sd += d[i]; sg += g[i]; }
    sd /= d.length - 500; sg /= d.length - 500;
    expect(Math.abs(sd - Math.log(4))).toBeLessThan(0.2);
    expect(Math.abs(sg - Math.log(2))).toBeLessThan(0.25);
  });

  test('unstable: the share on the data swings far more in the second half than the stable run', () => {
    const a = spread(stable()), b = spread(unstable());
    expect(a.hi - a.lo).toBeLessThan(0.3);
    expect(b.hi - b.lo).toBeGreaterThan(0.5);
  });

  test('collapse: the ring run ends on four of the eight modes, the other on all eight', () => {
    const c = collapse(), o = ringOk();
    const sc = ringShares(points(c.snaps[c.snaps.length - 1])), so = ringShares(points(o.snaps[o.snaps.length - 1]));
    expect(sc.modes.filter((v) => v >= 0.05).length).toBe(4);
    expect(sc.modes.filter((v) => v < 0.02).length).toBe(4);
    expect(so.modes.filter((v) => v >= 0.05).length).toBe(8);
    expect(so.off).toBeLessThan(0.2);
  });

  test('gradients step: one update along the arrows raises the discriminator score', () => {
    const r = stable(), d = gradDemo(r.snaps[GRAD_SNAP]);
    expect(d.from.length).toBe(2 * GRAD_N);
    expect(d.dAfter).toBeGreaterThan(d.dBefore);
    expect(d.lenAfter).toBeLessThan(d.lenBefore);
    expect(d.towardData / d.offData).toBeGreaterThan(0.6);
  });

  test('generator step: most of the noise grid lands on the data', () => {
    const r = stable(), q = quality(generate(r.snaps[r.snaps.length - 1], noiseGrid()), MOONS);
    expect(q.onData).toBeGreaterThan(0.75);
  });

  test('discriminator step: D-only training reaches near-perfect accuracy', () => {
    const r = train('disc');
    expect(r.stats[5]).toBeLessThan(0.55);
    expect(r.stats[(r.snaps.length - 1) * STAT + 5]).toBeGreaterThan(0.95);
  });
});

describe('frames', () => {
  test('runFrame hits the first and last checkpoint and the markers match the segments', () => {
    const r = stable();
    expect(runFrame(r, 0)).toMatchObject({ idx: 0, step: 0, seg: 0 });
    expect(runFrame(r, T_RUN)).toMatchObject({ idx: 60, step: 1500, seg: 2 });
    expect(runFrame(r, T_RUN + 5).idx).toBe(60);
    const m = runMarkers(r);
    expect(m.map((x) => x.label)).toEqual(['Early', 'Middle', 'Late']);
    expect(runFrame(r, m[1].at - 0.01).seg).toBe(0);
    expect(runFrame(r, m[1].at)).toMatchObject({ step: 500, seg: 1 });
    expect(runFrame(r, m[2].at - 0.01).seg).toBe(1);
    expect(runFrame(r, m[2].at)).toMatchObject({ step: 1000, seg: 2 });
  });
  test('generator and gradient timelines', () => {
    expect(genFrame(0)).toEqual({ phase: 0, s: 0 });
    expect(genFrame(9)).toEqual({ phase: 2, s: 1 });
    expect(genFrame(4).s).toBeGreaterThan(0);
    expect(genFrame(4).s).toBeLessThan(1);
    expect(gradFrame(0)).toMatchObject({ phase: 0, grow: 0, move: 0 });
    expect(gradFrame(9)).toMatchObject({ phase: 3, grow: 1, move: 1 });
    expect(durationOf('generator')).toBe(9);
    expect(durationOf('duel')).toBe(T_RUN);
  });
  test('the configured runs are the ones the lesson names', () => {
    expect(RUN_CFG.unstable.lrG / RUN_CFG.stable.lrG).toBeCloseTo(13.33, 1);
    expect(RUN_CFG.collapse.ring).toBe(true);
    expect(RUN_CFG.disc.freezeG).toBe(true);
  });
});
