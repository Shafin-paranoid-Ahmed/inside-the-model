import { describe, expect, test } from 'vitest';
import { createRng } from '../../src/lib/math/rng';
import {
  exp, tanh, log, sin, cos, normal, sigmoid, timeFeatures, initMlp, createCache, forward, backward, createAdam, adam, paramCount,
} from '../../src/lib/gen/mlp';
import { MOONS, mixture, sampleData, posteriorMean, score, logp, quality } from '../../src/lib/gen/toy';

describe('deterministic elementary functions', () => {
  test('match Math within rounding', () => {
    for (const x of [-30, -3.7, -1, -1e-3, 0, 0.5, 2, 9.9, 40]) {
      expect(Math.abs(exp(x) / Math.exp(x) - 1)).toBeLessThan(1e-14);
      expect(tanh(x)).toBeCloseTo(Math.tanh(x), 14);
      expect(sigmoid(x)).toBeCloseTo(1 / (1 + Math.exp(-x)), 14);
      expect(sin(x)).toBeCloseTo(Math.sin(x), 13);
      expect(cos(x)).toBeCloseTo(Math.cos(x), 13);
    }
    for (const x of [1e-300, 1e-9, 0.3, 1, 1.5, 7, 1e6, 1e300]) expect(Math.abs(log(x) - Math.log(x))).toBeLessThanOrEqual(1e-14 * Math.max(1, Math.abs(Math.log(x))));
    expect(log(0)).toBe(-Infinity);
  });
  test('normal() has mean 0 and variance 1', () => {
    const r = createRng(1), n = 20000;
    let s = 0, s2 = 0;
    for (let i = 0; i < n; i++) { const z = normal(r); s += z; s2 += z * z; }
    expect(Math.abs(s / n)).toBeLessThan(0.03);
    expect(Math.abs(s2 / n - 1)).toBeLessThan(0.03);
  });
  test('timeFeatures', () => {
    const f = new Float64Array(5);
    timeFeatures(0.25, f);
    expect(Array.from(f).map((v) => +v.toFixed(12) + 0)).toEqual([0.25, +Math.sin(Math.PI / 4).toFixed(12), +Math.cos(Math.PI / 4).toFixed(12), 1, 0]);
  });
});

describe('MLP', () => {
  const sizes = [3, 5, 4, 2], B = 4;
  const rng = createRng(3);
  const p = initMlp(sizes, rng);
  for (let i = 0; i < p.length; i++) p[i] += 0.1 * (rng.next() - 0.5); // non-zero biases
  const X = Float64Array.from({ length: B * 3 }, () => rng.uniform(-1, 1));
  const T = Float64Array.from({ length: B * 2 }, () => rng.uniform(-1, 1));
  const loss = (q: Float64Array, x = X) => {
    const y = forward(q, sizes, x, B, createCache(sizes, B));
    let l = 0; for (let i = 0; i < y.length; i++) l += 0.5 * (y[i] - T[i]) ** 2;
    return l;
  };

  test('param count', () => {
    expect(paramCount(sizes)).toBe(5 * 4 + 4 * 6 + 2 * 5);
    expect(p.length).toBe(paramCount(sizes));
  });

  test('gradients match finite differences (params and input)', () => {
    const c = createCache(sizes, B), y = forward(p, sizes, X, B, c);
    const dY = Float64Array.from(y, (v, i) => v - T[i]), g = new Float64Array(p.length);
    backward(p, sizes, c, dY, g);
    const h = 1e-6;
    for (let i = 0; i < p.length; i++) {
      const a = p.slice(), b = p.slice(); a[i] += h; b[i] -= h;
      expect(g[i]).toBeCloseTo((loss(a) - loss(b)) / (2 * h), 6);
    }
    for (let i = 0; i < X.length; i++) {
      const a = X.slice(), b = X.slice(); a[i] += h; b[i] -= h;
      expect(c.dIn[i]).toBeCloseTo((loss(p, a) - loss(p, b)) / (2 * h), 6);
    }
  });

  test('backward accumulates', () => {
    const c = createCache(sizes, B), y = forward(p, sizes, X, B, c), dY = Float64Array.from(y, (v, i) => v - T[i]);
    const g1 = new Float64Array(p.length), g2 = new Float64Array(p.length);
    backward(p, sizes, c, dY, g1); backward(p, sizes, c, dY, g2); backward(p, sizes, c, dY, g2);
    for (let i = 0; i < p.length; i++) expect(g2[i]).toBeCloseTo(2 * g1[i], 12);
  });

  test('Adam minimises a quadratic', () => {
    const q = Float64Array.from([3, -2, 0.5]), st = createAdam(3), g = new Float64Array(3);
    for (let t = 0; t < 2000; t++) { for (let i = 0; i < 3; i++) g[i] = 2 * (i + 1) * (q[i] - 1); adam(q, g, st, 0.05); }
    for (const v of q) expect(v).toBeCloseTo(1, 3);
  });

  test('training is deterministic and fits a small regression', () => {
    const run = () => {
      const r = createRng(7), sz = [2, 16, 1], q = initMlp(sz, r), st = createAdam(q.length), N = 32;
      const c = createCache(sz, N), x = new Float64Array(2 * N), dY = new Float64Array(N), g = new Float64Array(q.length);
      let last = 0;
      for (let t = 0; t < 400; t++) {
        for (let i = 0; i < 2 * N; i++) x[i] = r.uniform(-1, 1);
        const y = forward(q, sz, x, N, c);
        last = 0;
        for (let i = 0; i < N; i++) { const e = y[i] - x[2 * i] * x[2 * i + 1]; dY[i] = e / N; last += (e * e) / N; }
        g.fill(0); backward(q, sz, c, dY, g); adam(q, g, st, 0.02);
      }
      return { q, last };
    };
    const a = run(), b = run();
    expect(Array.from(a.q)).toEqual(Array.from(b.q));
    expect(a.last).toBeLessThan(0.01);
  });
});

describe('toy mixture', () => {
  test('moons layout', () => {
    expect(MOONS.K).toBe(52);
    expect(MOONS.upper).toBe(26);
    expect(MOONS.sigma).toBe(0.07);
    const d = sampleData(MOONS, 600, 7);
    expect(d.x.length).toBe(1200);
    for (let i = 0; i < 600; i++) expect(d.moon[i]).toBe(d.k[i] < 26 ? 0 : 1);
    expect(Array.from(sampleData(MOONS, 600, 7).x)).toEqual(Array.from(d.x));
  });

  test('score from the posterior mean equals the numerical gradient of log p', () => {
    for (const m of [MOONS, mixture('spiral')]) {
      const out = [0, 0], h = 1e-5;
      for (const [a, b, x, y] of [[0.99, 0.1, 0.3, -0.4], [0.7, 0.7, 0.8, 0.5], [0.1, 0.99, -1, 1.2], [0.5, 0.5, 0, 0]]) {
        score(m, a, b, x, y, out);
        expect(out[0]).toBeCloseTo((logp(m, a, b, x + h, y) - logp(m, a, b, x - h, y)) / (2 * h), 4);
        expect(out[1]).toBeCloseTo((logp(m, a, b, x, y + h) - logp(m, a, b, x, y - h)) / (2 * h), 4);
      }
    }
  });

  test('flow-matching velocity v* = (E[x1|x] - x)/(1 - t) points at the data as t -> 1', () => {
    const out = [0, 0];
    for (const k of [0, 13, 40]) {
      const mx = MOONS.mu[2 * k], my = MOONS.mu[2 * k + 1], t = 0.999, x = t * mx + 0.01, y = t * my - 0.01;
      posteriorMean(MOONS, t, 1 - t, x, y, out);
      expect(Math.hypot(out[0] - x, out[1] - y)).toBeLessThan(0.02); // E[x1|x] ~ x near the data
    }
    // at t = 0 the posterior mean is the data mean whatever x is
    const m0 = [0, 0], m1 = [0, 0];
    posteriorMean(MOONS, 0, 1, 1.5, -2, m0); posteriorMean(MOONS, 0, 1, -0.3, 0.7, m1);
    expect(m0[0]).toBeCloseTo(m1[0], 12); expect(m0[1]).toBeCloseTo(m1[1], 12);
  });

  test('class-conditional posterior stays on its moon', () => {
    const out = [0, 0];
    posteriorMean(MOONS, 0.3, 0.95, 0, 0, out, 'upper');
    const up = out[1];
    posteriorMean(MOONS, 0.3, 0.95, 0, 0, out, 'lower');
    expect(up).toBeGreaterThan(out[1]);
  });

  test('quality: exact samples are on the data and split evenly; noise is not', () => {
    const q = quality(sampleData(MOONS, 2000, 11).x, MOONS);
    expect(q.onData).toBeGreaterThanOrEqual(0.95);
    expect(Math.abs(q.upper - 0.5)).toBeLessThan(0.05);
    expect(q.upper + q.lower).toBeCloseTo(1, 12);
    const r = createRng(4), noise = Float32Array.from({ length: 2000 }, () => 1.5 * normal(r));
    expect(quality(noise, MOONS).onData).toBeLessThan(0.4);
    expect(quality([], MOONS)).toEqual({ onData: 0, upper: 0, lower: 0 });
  });
});
