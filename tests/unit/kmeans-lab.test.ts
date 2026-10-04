import { describe, it, expect } from 'vitest';
import {
  makeData, withPoint, initCentroids, kmeans, inertia, finalInertia, elbow, frameAt, structureFrame, sizes, duration, markers,
  T0, DA, DU, P, K_MAX, DOM,
} from '../../src/components/widgets/KMeansLab/sim';

const d3 = makeData('three'), d4 = makeData('four');

describe('KMeansLab sim', () => {
  it('data is deterministic and inside the plane', () => {
    expect(makeData('four')).toEqual(d4);
    expect(d3.n).toBe(120);
    expect(Array.from(d3.pts).every((v) => Math.abs(v) < DOM)).toBe(true);
    expect(withPoint(d3, 0.1, 0.2).n).toBe(121);
  });
  it('init: random picks distinct data points; k-means++ is deterministic', () => {
    const C = initCentroids(d4.pts, 4, 'random', 3);
    const rows = [0, 1, 2, 3].map((c) => `${C[2 * c]},${C[2 * c + 1]}`);
    expect(new Set(rows).size).toBe(4);
    expect(initCentroids(d4.pts, 4, 'plusplus', 5)).toEqual(initCentroids(d4.pts, 4, 'plusplus', 5));
  });
  it('inertia never increases, through assign and update, and ends converged', () => {
    const r = kmeans(d3.pts, initCentroids(d3.pts, 3, 'random', 10));
    expect(r.n).toBe(7);
    expect(r.moved).toEqual([120, 15, 9, 20, 18, 1, 0]);
    const seq = r.iA.flatMap((a, j) => [a, r.iU[j]]);
    for (let i = 1; i < seq.length; i++) expect(seq[i]).toBeLessThanOrEqual(seq[i - 1] + 1e-9);
    expect(finalInertia(r)).toBeCloseTo(14.8, 1);
    expect(r.C[r.n]).toEqual(r.C[r.n - 1]);
  });
  it('update step puts each centroid at the mean of its points', () => {
    const r = kmeans(d3.pts, initCentroids(d3.pts, 3, 'random', 10));
    const j = 2;
    for (let c = 0; c < 3; c++) {
      let sx = 0, sy = 0, n = 0;
      for (let i = 0; i < r.N; i++) if (r.a[j][i] === c) { sx += d3.pts[2 * i]; sy += d3.pts[2 * i + 1]; n++; }
      expect(r.C[j + 1][2 * c]).toBeCloseTo(sx / n, 12); expect(r.C[j + 1][2 * c + 1]).toBeCloseTo(sy / n, 12);
    }
    expect(r.iU[j]).toBeCloseTo(inertia(r.pts, r.a[j], r.C[j + 1]), 12);
  });
  it('a bad random start gets stuck; k-means++ does not (seed 13 on four groups)', () => {
    const a = kmeans(d4.pts, initCentroids(d4.pts, 4, 'random', 13)), b = kmeans(d4.pts, initCentroids(d4.pts, 4, 'plusplus', 13));
    expect(finalInertia(a)).toBeCloseTo(39.9, 1); expect(a.n).toBe(10);
    expect(finalInertia(b)).toBeCloseTo(11.6, 1); expect(b.n).toBe(3);
  });
  it('elbow: decreasing, sharp bend at the true k = 4', () => {
    const e = elbow(d4.pts);
    expect(e.inertia.length).toBe(K_MAX);
    for (let k = 1; k < K_MAX; k++) expect(e.inertia[k]).toBeLessThan(e.inertia[k - 1]);
    expect(e.inertia.map((v) => +v.toFixed(1))).toEqual([156.6, 59.5, 30.4, 11.6, 10.1, 9.1, 7.6, 7.0]);
    expect(e.inertia[2] - e.inertia[3]).toBeGreaterThan(10 * (e.inertia[3] - e.inertia[4]));
  });
  it('frameAt: endpoints, determinism, and the drawn inertia matches the recorded values', () => {
    const r = kmeans(d3.pts, initCentroids(d3.pts, 3, 'random', 10));
    const f0 = frameAt(0, r);
    expect(f0.phase).toBe('start'); expect(f0.centA).toBe(0); expect(Number.isNaN(f0.inertia)).toBe(true);
    expect(Array.from(f0.mix).every((m) => m === 0)).toBe(true);
    expect(frameAt(T0 + DA, r).inertia).toBeCloseTo(r.iA[0], 9);           // end of first assign
    expect(frameAt(T0 + P, r).inertia).toBeCloseTo(r.iU[0], 9);            // end of first update
    expect(frameAt(T0 + P + DA, r).inertia).toBeCloseTo(r.iA[1], 9);       // end of second assign
    const end = frameAt(duration(r), r);
    expect(end.phase).toBe('converged'); expect(end.inertia).toBeCloseTo(finalInertia(r), 9); expect(end.cx).toBe(r.n);
    expect(frameAt(3.3, r)).toEqual(frameAt(3.3, r));
    expect(frameAt(T0 + P + DA + DU / 2, r).phase).toBe('update');
  });
  it('sizes sum to N at every time and structure frame fades grey to the converged clustering', () => {
    const r = kmeans(d3.pts, initCentroids(d3.pts, 3, 'plusplus', 1));
    for (const t of [0, 2, 4, 6.5]) expect(sizes(frameAt(t, r), 3).reduce((s, v) => s + v, 0)).toBeCloseTo(120, 6);
    expect(sizes(structureFrame(0, r), 3)[3]).toBe(120);
    const z = sizes(structureFrame(8, r), 3);
    expect(Array.from(z).map((v) => Math.round(v))).toEqual([40, 40, 40, 0]);
    expect(markers(r, 'update')[0].label).toBe('running');
  });
});
