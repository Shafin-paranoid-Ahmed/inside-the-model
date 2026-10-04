import { describe, expect, test } from 'vitest';
import { mixture, sampleData, score, logp, alphaBar, forwardAt, reversePath, pathAt, tauAt, T, TAU_MIN } from '../../src/components/widgets/DiffusionDenoiser/sim';
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
