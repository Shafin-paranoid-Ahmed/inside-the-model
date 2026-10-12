import { describe, expect, test } from 'vitest';
import {
  mixture, sampleData, score, logp, alphaBar, forwardAt, reversePath, pathAt, tauAt, T, TAU_MIN, MOONS, quality, trainEps, netScoreFn, exactScoreFn, guidedScoreFn, startNoise,
  REVERSE_STEPS, lossAt, lossFloor, moonStats, imageRun, imageAt, imagePosterior, GLYPHS, modeTime, sweepTau, tauHalf, groupOf, TRAIN_STEPS, type Mode,
} from '../../src/components/widgets/DiffusionDenoiser/sim';
import { createRng } from '../../src/lib/math/rng';

describe('VP schedule and exact score', () => {
  test('abar endpoints', () => {
    expect(alphaBar(0)).toBe(1);
    expect(alphaBar(1)).toBeLessThan(1e-4);
  });
  test('score equals numerical gradient of log p_t', () => {
    for (const ds of ['spiral', 'moons'] as const) {
      const m = mixture(ds), out = [0, 0], h = 1e-5;
      for (const [tau, x, y] of [[0.05, 0.3, -0.4], [0.3, 0.8, 0.5], [0.9, -1, 1.2]]) {
        score(m, tau, x, y, out);
        expect(out[0]).toBeCloseTo((logp(m, tau, x + h, y) - logp(m, tau, x - h, y)) / (2 * h), 4);
        expect(out[1]).toBeCloseTo((logp(m, tau, x, y + h) - logp(m, tau, x, y - h)) / (2 * h), 4);
      }
    }
  });
  test('forward closed form: at tau=1 points are (almost) pure noise, at 0 unchanged', () => {
    const m = mixture('spiral'), { x } = sampleData(m, 200, 1), eps = Float32Array.from({ length: 400 }, () => createRng(2).normal());
    const o = new Float32Array(400);
    forwardAt(0, x, eps, o); expect(Array.from(o)).toEqual(Array.from(x));
    forwardAt(1, x, eps, o);
    for (let i = 0; i < 400; i++) expect(Math.abs(o[i] - eps[i])).toBeLessThan(0.02);
  });
});

describe('reverse sampling', () => {
  const near = (m: ReturnType<typeof mixture>, p: Float32Array) => {
    let s = 0;
    for (let i = 0; i < p.length / 2; i++) {
      let b = Infinity;
      for (let k = 0; k < m.K; k++) b = Math.min(b, Math.hypot(p[2 * i] - m.mu[2 * k], p[2 * i + 1] - m.mu[2 * k + 1]));
      s += b;
    }
    return s / (p.length / 2);
  };
  for (const kind of ['ode', 'sde'] as const) {
    test(`${kind}: noise lands on the data curve, deterministically`, () => {
      const m = mixture('spiral'), r = createRng(9), n = 120;
      const xT = Float32Array.from({ length: 2 * n }, () => r.normal());
      const p = reversePath(m, xT, kind, 200, 3), out = new Float32Array(2 * n);
      pathAt(p, TAU_MIN, out);
      expect(near(m, out)).toBeLessThan(0.1);
      const before = new Float32Array(2 * n); pathAt(p, 1, before);
      expect(near(m, before)).toBeGreaterThan(0.3);
      expect(Array.from(reversePath(m, xT, kind, 200, 3).xs.slice(0, 50))).toEqual(Array.from(p.xs.slice(0, 50)));
    });
  }
  test('timeline endpoints', () => {
    expect(tauAt(0).tau).toBe(0);
    expect(tauAt(6).tau).toBe(1);
    expect(tauAt(T).tau).toBe(TAU_MIN);
  });
});

describe('learned noise-prediction network', () => {
  test('training is deterministic and snapshots are real parameter sets', () => {
    const a = trainEps(40), b = trainEps(40);
    expect(a.steps).toEqual([0, 25, 40]);
    expect(Array.from(a.snaps[2])).toEqual(Array.from(b.snaps[2]));
    expect(Array.from(a.losses)).toEqual(Array.from(b.losses));
    expect(a.snaps[0]).not.toEqual(a.snaps[2]);
  });
  test('a full run beats the untrained network, nears the floor, and samples onto the moons', () => {
    const run = trainEps(), xT = startNoise();
    expect(run.snaps.length).toBe(9);
    expect(lossAt(run, TRAIN_STEPS - 1)).toBeLessThan(lossAt(run, 0) - 0.1);
    expect(lossAt(run, TRAIN_STEPS - 1)).toBeGreaterThan(lossFloor() - 0.01);
    expect(run.angles[0][8]).toBeLessThan(run.angles[0][0] / 4);
    const p = reversePath(MOONS, xT, 'ode', REVERSE_STEPS, 5, netScoreFn(run.snaps[8]));
    const q = quality(p.xs.slice(-1200), MOONS);
    expect(q.onData).toBeGreaterThan(0.5);
    expect(Math.min(q.upper, q.lower)).toBeGreaterThan(0.3);
  }, 30000);
});

describe('exact reverse sampling and guidance', () => {
  const xT = startNoise(300);
  test('the exact ODE puts nearly every point on the moons, both moons covered', () => {
    const q = quality(reversePath(MOONS, xT, 'ode', REVERSE_STEPS, 5).xs.slice(-600), MOONS);
    expect(q.onData).toBeGreaterThan(0.95);
    expect(q.upper).toBeGreaterThan(0.4); expect(q.upper).toBeLessThan(0.6);
  });
  test('w = 0 is the conditional score; guided samples stay on the chosen moon and bunch up as w grows', () => {
    const out = new Float64Array(2), g = new Float64Array(2), x = Float64Array.of(0.3, 0.9);
    guidedScoreFn(MOONS, 'upper', 0)(0.3, x, 1, out);
    const a = Math.sqrt(alphaBar(0.3)), b = Math.sqrt(1 - alphaBar(0.3)), c = [0, 0]; score(MOONS, 0.3, 0.3, 0.9, c);
    exactScoreFn(MOONS, 'upper')(0.3, x, 1, g);
    expect(out[0]).toBeCloseTo(g[0], 10); expect(a).toBeGreaterThan(0); expect(b).toBeGreaterThan(0);
    const spread = (w: number) => moonStats(MOONS, reversePath(MOONS, xT, 'ode', 60, 5, guidedScoreFn(MOONS, 'lower', w)).xs.slice(-600), 'lower');
    const s0 = spread(0), s7 = spread(7);
    expect(s0.onMoon).toBeGreaterThan(0.9);
    expect(s7.spread).toBeLessThan(s0.spread * 0.6);
  });
  test('groupOf labels the moons', () => {
    const g = new Uint8Array(2); groupOf(MOONS, Float64Array.of(MOONS.mu[0], MOONS.mu[1], MOONS.mu[2 * 30], MOONS.mu[2 * 30 + 1]), g);
    expect(Array.from(g)).toEqual([0, 1]);
  });
});

describe('image mode', () => {
  test('posterior sums to one, recovers a clean glyph, and the noising guess is right at low noise', () => {
    const w = new Float64Array(10), gs = new Float64Array(64);
    imagePosterior(Float64Array.from(GLYPHS[4]), 0.05, w, gs);
    expect(w.reduce((x, y) => x + y)).toBeCloseTo(1, 10); expect(w[4]).toBeGreaterThan(0.99);
    const run = imageRun(3), xt = new Float64Array(64), nz = new Float64Array(64);
    imageAt(run, 0.1, true, xt, gs, nz, w); expect(w[3]).toBeGreaterThan(0.99);
    imageAt(run, 1, true, xt, gs, nz, w); expect(Math.max(...w)).toBeLessThan(0.5);
  });
  test('generation ends on a clean glyph', () => {
    const run = imageRun(3), xt = new Float64Array(64), gs = new Float64Array(64), nz = new Float64Array(64), w = new Float64Array(10);
    imageAt(run, TAU_MIN, false, xt, gs, nz, w);
    expect(Math.max(...w)).toBeGreaterThan(0.99);
    const end = w.indexOf(Math.max(...w));
    for (let i = 0; i < 64; i++) expect(Math.abs(xt[i] - GLYPHS[end][i])).toBeLessThan(0.5);
  });
});

describe('mode timelines', () => {
  test('every mode has markers inside its duration and sweeps end where they should', () => {
    for (const m of ['both', 'forward', 'reverse', 'closed', 'field', 'train', 'image', 'guide'] as Mode[]) {
      const mt = modeTime(m);
      expect(mt.markers[0].at).toBe(0);
      for (const k of mt.markers) expect(k.at).toBeLessThan(mt.dur);
    }
    expect(sweepTau('closed', 0)).toBe(0); expect(sweepTau('closed', 100)).toBe(1);
    expect(sweepTau('field', 100)).toBeCloseTo(0.02, 10);
    expect(sweepTau('guide', 100)).toBeCloseTo(TAU_MIN, 10);
    expect(sweepTau('image', 5.3)).toBe(1); expect(sweepTau('image', 100)).toBeCloseTo(TAU_MIN, 10);
    expect(alphaBar(tauHalf())).toBeCloseTo(0.5, 6);
  });
  test('forward and reverse modes loop only their own half of the timeline', () => {
    const f = modeTime('forward'), r = modeTime('reverse');
    expect(tauAt(f.off + f.dur - 1e-6).tau).toBe(1);
    expect(tauAt(r.off + 1e-6).phase).toBe('reverse');
    expect(r.off + r.dur).toBeCloseTo(T, 10);
  });
});
