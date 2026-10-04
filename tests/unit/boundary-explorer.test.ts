import { describe, it, expect } from 'vitest';
import { makeDataset, train, frameAt, evaluate, field, embed, signedDistance, score, T, STEPS, DOM } from '../../src/components/widgets/BoundaryExplorer/sim';

describe('BoundaryExplorer sim', () => {
  it('datasets are deterministic and balanced', () => {
    const a = makeDataset('xor'), b = makeDataset('xor');
    expect(a).toEqual(b);
    expect(a.filter((p) => p.c === 1).length).toBe(30);
    expect(a.every((p) => Math.abs(p.x) < DOM && Math.abs(p.y) < DOM)).toBe(true);
  });
  it('linear model separates the linear set', () => {
    const r = train(makeDataset('linear'), 'linear');
    expect(r.loss[STEPS]).toBeLessThan(r.loss[0]);
    expect(r.acc[STEPS]).toBeGreaterThan(0.93);
  });
  it('linear model fails on XOR and circles, polynomial features solve them', () => {
    for (const d of ['xor', 'circles'] as const) {
      const pts = makeDataset(d);
      expect(train(pts, 'linear').acc[STEPS]).toBeLessThan(0.85);
      expect(train(pts, 'poly').acc[STEPS]).toBeGreaterThan(0.95);
    }
  });
  it('gradient step matches a numerical gradient', () => {
    const pts = makeDataset('circles'), w = [0.1, -0.3, 0.2, 0.4, -0.1, 0.3];
    const r = train(pts, 'poly', w, 1, 1);
    for (let j = 0; j < 6; j++) {
      const h = 1e-5, up = w.slice(), dn = w.slice(); up[j] += h; dn[j] -= h;
      const num = (evaluate('poly', up, pts).loss - evaluate('poly', dn, pts).loss) / (2 * h);
      expect(w[j] - r.W[6 + j]).toBeCloseTo(num, 6);
    }
  });
  it('frameAt hits the endpoints, interpolates, is deterministic', () => {
    const r = train(makeDataset('linear'), 'linear');
    expect(Array.from(frameAt(0, r).w)).toEqual([0, 0, 0]);
    expect(frameAt(0, r).loss).toBeCloseTo(Math.LN2, 9);
    expect(frameAt(T, r).loss).toBeCloseTo(r.loss[STEPS], 12);
    const mid = frameAt(1.3, r);
    expect(mid.k).toBeGreaterThan(0); expect(mid.k).toBeLessThan(STEPS);
    expect(frameAt(1.3, r)).toEqual(mid);
  });
  it('field, distance and embed helpers', () => {
    const out = new Float32Array(16); field('linear', [0, 0, 0], 4, out);
    expect(Array.from(out).every((v) => v === 0.5)).toBe(true);
    const w = [1, 2, 0]; expect(signedDistance(w, 0, 0)).toBeCloseTo(0.5, 12);
    expect(signedDistance(w.map((v) => v * 3), 0, 0)).toBeCloseTo(0.5, 12);
    expect(embed([1, 2, 3], 'poly')).toEqual(new Float64Array([1, 2, 3, 0, 0, 0]));
    expect(score('poly', [0, 0, 0, 0, 1, 0], 2, 3)).toBe(6);
  });
});
