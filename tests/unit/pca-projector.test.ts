import { describe, expect, it } from 'vitest';
import * as S from '../../src/components/widgets/PCAProjector/sim';

describe('PCAProjector math', () => {
  it('2x2 eigenvalues match the closed form and A v = lambda v', () => {
    const A = [[3, 1], [1, 2]];
    const e = S.eigSym(A), tr = 5, det = 5, disc = Math.sqrt(tr * tr / 4 - det);
    expect(e.values[0]).toBeCloseTo(tr / 2 + disc, 10);
    expect(e.values[1]).toBeCloseTo(tr / 2 - disc, 10);
    e.vectors.forEach((v, i) => S.matVec(A, v).forEach((x, k) => expect(x).toBeCloseTo(e.values[i] * v[k], 10)));
  });
  it('3x3 symmetric matrix: A v = lambda v, orthonormal vectors, known eigenvalues', () => {
    const A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]; // eigenvalues 2 + sqrt2, 2, 2 - sqrt2
    const e = S.eigSym(A);
    [2 + Math.SQRT2, 2, 2 - Math.SQRT2].forEach((l, i) => expect(e.values[i]).toBeCloseTo(l, 10));
    e.vectors.forEach((v, i) => {
      S.matVec(A, v).forEach((x, k) => expect(x).toBeCloseTo(e.values[i] * v[k], 10));
      e.vectors.forEach((w, j) => expect(v.reduce((s, x, k) => s + x * w[k], 0)).toBeCloseTo(i === j ? 1 : 0, 10));
    });
  });
  it('the 2D cloud: covariance eigenpairs, trace, and variance along PC1 equals lambda1', () => {
    const m = S.makeModel();
    m.vectors.forEach((v, i) => S.matVec(m.cov, v).forEach((x, k) => expect(x).toBeCloseTo(m.values[i] * v[k], 9)));
    expect(m.values[0] + m.values[1]).toBeCloseTo(m.trace, 10);
    expect(S.varAt(m, m.th1)).toBeCloseTo(m.values[0], 10);
    expect(S.varAt(m, m.th1 + Math.PI / 2)).toBeCloseTo(m.values[1], 10);
    const direct = m.pts.reduce((s, p) => s + S.coord(p, m.vectors[0]) ** 2, 0) / m.pts.length;
    expect(direct).toBeCloseTo(m.values[0], 9);
  });
  it('variance peaks and residual bottoms out at PC1, and they always add to the trace', () => {
    const m = S.makeModel();
    let bestV = -1, at = 0;
    for (let d = 0; d < 180; d += 0.5) {
      const th = d * S.DEG, v = S.varAt(m, th);
      expect(v + S.residAt(m, th)).toBeCloseTo(m.trace, 10);
      if (v > bestV) { bestV = v; at = d; }
    }
    const pc1 = ((m.th1 / S.DEG) + 180) % 180;
    expect(Math.abs(at - pc1)).toBeLessThanOrEqual(0.5);
    expect(S.meanSqDist(m.pts, m.vectors[0])).toBeCloseTo(m.values[1], 9);
  });
  it('rotateFrame starts on the worst axis, sweeps 180°, ends on PC1', () => {
    const m = S.makeModel(), a = S.rotateFrame(0, m), mid = S.rotateFrame(S.ROTATE.sweepEnd, m), b = S.rotateFrame(S.ROTATE.T, m);
    expect(a.variance).toBeCloseTo(m.values[1], 9);
    expect(mid.reach).toBeCloseTo(Math.PI, 9);
    expect(b.variance).toBeCloseTo(m.values[0], 9);
    expect(b.resid).toBeCloseTo(m.values[1], 9);
  });
  it('eigenFrame: C v is parallel to v at PC1 and PC2, not in between', () => {
    const m = S.makeModel(), f1 = S.eigenFrame(S.EIGEN.at1, m), f2 = S.eigenFrame(S.EIGEN.T, m), f0 = S.eigenFrame(0, m);
    expect(f1.off).toBeLessThan(1e-6);
    expect(f1.rayleigh).toBeCloseTo(m.values[0], 9);
    expect(f2.off).toBeLessThan(1e-6);
    expect(f2.rayleigh).toBeCloseTo(m.values[1], 9);
    expect(f0.off).toBeGreaterThan(5);
  });
  it('keeping one component loses exactly the other eigenvalue', () => {
    const m = S.makeModel();
    expect(S.keepOne(m, 0).mse).toBeCloseTo(S.meanSqDist(m.pts, m.vectors[0]), 9);
    expect(S.keepOne(m, 1).mse).toBeCloseTo(S.meanSqDist(m.pts, m.vectors[1]), 9);
    expect(S.keepOne(m, 0).kept + S.keepOne(m, 1).kept).toBeCloseTo(1, 12);
    expect(S.projectFrame(0).s).toBe(0);
    expect(S.projectFrame(S.PROJECT.T).s).toBe(1);
  });
  it('scree: ratios sum to 1, cumulative ends at 1, scores have variance lambda, error = dropped eigenvalues', () => {
    const s = S.makeScree();
    expect(s.ratios.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(s.cum[S.DIMS - 1]).toBeCloseTo(1, 12);
    expect([...s.values].sort((a, b) => b - a)).toEqual(s.values);
    s.values.forEach((l, j) => expect(s.scores.reduce((t, r) => t + r[j] ** 2, 0) / s.scores.length).toBeCloseTo(l, 8));
    // reconstruct from k components in feature space and compare to the formula
    const k = 2, n = s.scores.length;
    let err = 0;
    for (const r of s.scores) for (let j = k; j < S.DIMS; j++) err += r[j] ** 2 / n;
    expect(S.screeFrame(S.SCREE.T, s, k).mse).toBeCloseTo(err, 9);
  });
  it('screeFrame steps 6 -> 1 and is deterministic', () => {
    const s = S.makeScree();
    expect(S.screeFrame(0, s).k).toBe(S.DIMS);
    expect(S.screeFrame(S.SCREE.T, s).k).toBe(1);
    const ks = Array.from({ length: 86 }, (_, i) => S.screeFrame(i * 0.1, s).k);
    ks.forEach((k, i) => i && expect(k).toBeLessThanOrEqual(ks[i - 1]));
    expect(S.makeScree().values).toEqual(s.values);
    expect(S.makeModel().pts).toEqual(S.makeModel().pts);
  });
});
