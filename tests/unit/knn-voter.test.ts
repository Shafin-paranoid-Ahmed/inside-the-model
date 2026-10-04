import { describe, it, expect } from 'vitest';
import {
  moons, stripes, nearest, knnFrac, accuracy, accByK, field, dist, makeScene, frameAt, ODD_KS, KS, RHOS, GRID, DOM, DIMS_MAX,
} from '../../src/components/widgets/KNNVoter/sim';

describe('KNNVoter sim', () => {
  it('distances: Euclidean, Manhattan, and the unit balls in between', () => {
    expect(dist(2, 3, 4)).toBe(5);
    expect(dist(1, -3, 4)).toBe(7);
    expect(dist(1.5, 1, 0)).toBeCloseTo(1, 12);
    expect(dist(1.5, 1, 1)).toBeCloseTo(2 ** (1 / 1.5), 12);
  });
  it('datasets are deterministic, balanced and inside the plane', () => {
    expect(moons(21, 50)).toEqual(moons(21, 50));
    const a = moons(21, 50), b = stripes(31, 50);
    expect(a.filter((p) => p.c === 1).length).toBe(50);
    expect([...a, ...b].every((p) => Math.abs(p.x) < DOM && Math.abs(p.y) < DOM)).toBe(true);
  });
  it('nearest matches brute-force sorting, and knnFrac agrees', () => {
    const pts = moons(5, 30), q = { x: 0.1, y: -0.2 };
    for (const p of [1, 2]) {
      const brute = pts.map((pt, i) => ({ i, d: dist(p, pt.x - q.x, pt.y - q.y) })).sort((u, v) => u.d - v.d).slice(0, 7);
      const n = nearest(pts, q.x, q.y, 7, p);
      expect(n.idx).toEqual(brute.map((e) => e.i));
      expect(knnFrac(pts, q.x, q.y, 7, p)).toBeCloseTo(brute.reduce((s, e) => s + pts[e.i].c, 0) / 7, 12);
    }
  });
  it('y weight stretches the axis used to measure', () => {
    const pts = [{ x: 1, y: 0, c: 0 as const }, { x: 0, y: 0.5, c: 1 as const }];
    expect(nearest(pts, 0, 0, 1, 2, 1).idx[0]).toBe(1);
    expect(nearest(pts, 0, 0, 1, 2, 10).idx[0]).toBe(0);
  });
  it('k = 1 memorises the training set; very large k does not', () => {
    const tr = moons(21, 50), va = moons(22, 50);
    expect(accuracy(tr, tr, 1)).toBe(1);
    const sc = makeScene('k');
    const i1 = ODD_KS.indexOf(1), i5 = ODD_KS.indexOf(5), i75 = ODD_KS.indexOf(75);
    expect(sc.accK!.train[i1]).toBe(1);
    expect(sc.accK!.val[i5]).toBeGreaterThan(sc.accK!.val[i1]);
    expect(sc.accK!.val[i5]).toBeGreaterThan(sc.accK!.val[i75] + 0.1);
    // prefix-sum version equals the direct one
    const r = accByK(tr, va, [1, 5, 15]);
    expect(r.val).toEqual([1, 5, 15].map((k) => accuracy(tr, va, k)));
  });
  it('field equals the vote at a cell centre', () => {
    const pts = moons(5, 30), out = new Float32Array(16);
    field(pts, 4, 3, 2, 1, out);
    expect(out[5]).toBeCloseTo(knnFrac(pts, -DOM + 1.5 * (2 * DOM) / 4, -DOM + 1.5 * (2 * DOM) / 4, 3), 6);
    expect(makeScene('vote').field!.length).toBe(GRID * GRID);
  });
  it('vote frame: the query drifts, votes come from the lit neighbours', () => {
    const sc = makeScene('vote'), f0 = frameAt(0, sc), fm = frameAt(sc.T / 2, sc);
    expect(f0.votes[0] + f0.votes[1]).toBe(5);
    expect(f0.nn.filter((e) => e.inside).length).toBe(5);
    expect(f0.pred).toBe(0);
    expect(fm.pred).toBe(1);
    expect(frameAt(sc.T, sc).q!.x).toBeCloseTo(f0.q!.x, 9);
    expect(frameAt(3, sc)).toEqual(frameAt(3, sc));
    expect(frameAt(1, sc, { q: { x: 0.5, y: 0.5 } }).q).toEqual({ x: 0.5, y: 0.5 });
  });
  it('metric frame: the ball grows to catch k points, and the two metrics disagree at the query', () => {
    const sc = makeScene('metric'), a = frameAt(3.5, sc), b = frameAt(8.5, sc);
    expect(frameAt(0, sc).nn.filter((e) => e.inside).length).toBe(0);
    expect(a.p).toBe(2); expect(b.p).toBe(1);
    expect(a.nn.filter((e) => e.inside).length).toBe(5);
    expect(b.nn.filter((e) => e.inside).length).toBe(5);
    const s = (f: typeof a) => f.nn.filter((e) => e.inside).map((e) => e.i).sort().join();
    expect(s(a)).not.toBe(s(b));
    expect(a.pred).toBe(1); expect(b.pred).toBe(0);
    expect(frameAt(sc.T, sc).r).toBe(0);
  });
  it('scale frame: stretching y wrecks accuracy and standardising restores it', () => {
    const sc = makeScene('scale'), at = (t: number) => frameAt(t, sc);
    expect(at(0).rho).toBeCloseTo(1, 9);
    expect(at(5.5).rho).toBeCloseTo(100, 6);
    expect(at(12).rho).toBeCloseTo(sc.rhoStd!, 9);
    expect(at(0).valAcc).toBeGreaterThan(0.9);
    expect(at(5.5).valAcc).toBeLessThan(0.65);
    expect(at(11).valAcc).toBeGreaterThan(0.9);
    expect(sc.accRho![RHOS.length - 1]).toBe(at(5.5).valAcc);
  });
  it('k frame: steps through KS and loops', () => {
    const sc = makeScene('k');
    expect(frameAt(0, sc).k).toBe(1);
    expect(frameAt(sc.T - 0.01, sc).k).toBe(KS[KS.length - 1]);
    expect(frameAt(sc.T * 0.99, sc).ki).toBeGreaterThan(KS.length - 1);
    // the k shown is always the active timeline marker's k
    sc.markers.forEach((m, i) => {
      const f = frameAt(m.at + 0.01, sc), a = ODD_KS.indexOf(KS[i]);
      expect(f.k).toBe(KS[i]);
      expect(f.valAcc).toBe(sc.accK!.val[a]);
      expect(f.trainAcc).toBe(sc.accK!.train[a]);
    });
  });
  it('dims: distances concentrate, so nearest / farthest climbs toward 1', () => {
    const sc = makeScene('dims'), lo = frameAt(0, sc), hi = frameAt(sc.T / 2, sc);
    expect(Math.round(lo.d)).toBe(1); expect(Math.round(hi.d)).toBe(DIMS_MAX);
    expect(lo.ratio).toBeLessThan(0.01); expect(hi.ratio).toBeGreaterThan(0.75);
    expect(hi.far - hi.near).toBeLessThan(lo.far - lo.near);
    expect(hi.hist!.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 4);
    expect(frameAt(sc.T, sc).ratio).toBeCloseTo(lo.ratio, 9);
  });
});
