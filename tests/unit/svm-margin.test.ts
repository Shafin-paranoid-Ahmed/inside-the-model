import { describe, it, expect } from 'vitest';
import {
  blobs, fanLines, startLine, solveLinear, supportIdx, trainLinear, linearFrameAt, objective, marginWidth, fmargin,
  trainLift, liftFrameAt, bestThreshold, lineData, kernelData, trainKernel, kernelFrameAt, kscore, dualSweeps, rbf, GAMMAS,
  identityFrameAt, phi2, polyK, T_MARGIN, T_SOFT, T_KERNEL, T_LIFT, DOM,
} from '../../src/components/widgets/SVMMargin/sim';

describe('SVMMargin sim', () => {
  it('datasets are deterministic, separable or not as promised', () => {
    expect(blobs(false)).toEqual(blobs(false));
    expect(fanLines(blobs(false)).length).toBe(9);
    expect(fanLines(blobs(true), 9).length).toBeLessThanOrEqual(9);
    expect(kernelData('rings').every((p) => Math.abs(p.x) < DOM && Math.abs(p.y) < DOM)).toBe(true);
  });
  it('fan lines all separate, and none beats the SVM margin', () => {
    const pts = blobs(false), fan = fanLines(pts), star = solveLinear(pts, 100), wStar = marginWidth(star);
    for (const c of fan) {
      expect(Math.min(...pts.map((p) => fmargin(c.w, p)))).toBeCloseTo(1, 9);
      expect(c.width).toBeLessThanOrEqual(wStar + 1e-3);
    }
    expect(Math.max(...fan.map((c) => c.width))).toBeGreaterThan(0.95 * wStar);
  });
  it('dual solution satisfies the KKT conditions', () => {
    const pts = blobs(true), C = 1, w = solveLinear(pts, C);
    const J = objective(pts, w, C);
    for (const d of [[1e-3, 0, 0], [-1e-3, 0, 0], [0, 1e-3, 0], [0, -1e-3, 0], [0, 0, 1e-2], [0, 0, -1e-2]]) expect(objective(pts, [w[0] + d[0], w[1] + d[1], w[2] + d[2]], C)).toBeGreaterThanOrEqual(J - 1e-6);
    const w2 = solveLinear(pts, 100);
    expect(marginWidth(w2)).toBeLessThan(marginWidth(w)); // larger C, narrower street
  });
  it('hard margin: support vectors sit on the margin, and keeping only them gives the same line', () => {
    const pts = blobs(false), w = solveLinear(pts, 100), sv = supportIdx(pts, w);
    expect(sv.length).toBeGreaterThanOrEqual(2);
    for (const i of sv) expect(fmargin(w, pts[i])).toBeCloseTo(1, 2);
    const w2 = solveLinear(sv.map((i) => pts[i]), 100);
    expect(marginWidth(w2)).toBeCloseTo(marginWidth(w), 3);
  });
  it('linear frameAt: fan start, training widens, endpoint is the optimum, deterministic', () => {
    const pts = blobs(false), fan = fanLines(pts), run = trainLinear(pts, 100, startLine(fan));
    const f0 = linearFrameAt(0, run, fan), fm = linearFrameAt(T_MARGIN / 2, run, fan), fe = linearFrameAt(T_MARGIN, run, fan);
    expect(f0.hi).toBe(0); expect(f0.phase).toBe('many lines');
    expect(f0.width).toBeCloseTo(fan[0].width, 9);
    expect(fm.width).toBeGreaterThan(linearFrameAt(3, run, fan).width);
    expect(fe.phase).toBe('settled');
    expect(fe.width).toBeCloseTo(marginWidth(run.star), 9);
    expect(fe.acc).toBe(1);
    expect(linearFrameAt(5.5, run, fan)).toEqual(linearFrameAt(5.5, run, fan));
    expect(run.obj[run.K]).toBeLessThan(run.obj[0]);
  });
  it('soft margin: C trades margin width for violations', () => {
    const pts = blobs(true);
    const at = (C: number) => linearFrameAt(T_SOFT, trainLinear(pts, C), null);
    expect(at(0.1).width).toBeGreaterThan(at(100).width);
    expect(at(0.1).viol).toBeGreaterThan(at(100).viol);
    expect(linearFrameAt(0, trainLinear(pts, 1), null).hinge).toBeCloseTo(pts.length, 9);
  });
  it('lift: no threshold separates the 1-D data, the lifted plane does', () => {
    const L = trainLift();
    expect(bestThreshold(L.data)).toBeLessThan(0.8);
    expect(L.pts.every((p) => fmargin(L.w, p) > 0.99)).toBe(true);
    expect(L.roots.length).toBe(2);
    expect(L.roots[0]).toBeLessThan(0); expect(L.roots[1]).toBeGreaterThan(0);
    expect(lineData()).toEqual(lineData());
    expect(liftFrameAt(0).s).toBe(0); expect(liftFrameAt(T_LIFT).s).toBe(1); expect(liftFrameAt(T_LIFT).back).toBe(1);
  });
  it('dual coordinate descent: alpha stays in [0, C] and the dual objective never decreases', () => {
    const pts = kernelData('rings'), n = pts.length, K = new Float64Array(n * n), y = pts.map((p) => (p.c ? 1 : -1));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] = rbf(2, pts[i].x - pts[j].x, pts[i].y - pts[j].y) + 1;
    const A = dualSweeps(K, y, 10, 20);
    const dual = (a: Float64Array) => { let s = 0; for (let i = 0; i < n; i++) { s += a[i]; for (let j = 0; j < n; j++) s -= 0.5 * a[i] * a[j] * y[i] * y[j] * K[i * n + j]; } return s; };
    for (let s = 1; s < A.length; s++) { expect(dual(A[s])).toBeGreaterThanOrEqual(dual(A[s - 1]) - 1e-9); expect(Math.max(...A[s])).toBeLessThanOrEqual(10 + 1e-12); expect(Math.min(...A[s])).toBeGreaterThanOrEqual(0); }
  });
  it('RBF SVM: small gamma underfits rings, large gamma memorises the mislabelled points', () => {
    const pts = kernelData('rings'), acc = (g: number) => kernelFrameAt(T_KERNEL, trainKernel(pts, g)).acc;
    expect(acc(GAMMAS[0])).toBeLessThan(0.7);
    expect(acc(1)).toBeGreaterThan(0.9); expect(acc(1)).toBeLessThan(1);
    expect(acc(30)).toBe(1);
    const r = trainKernel(pts, 1), f0 = kernelFrameAt(0, r);
    expect(f0.nsv).toBe(0); expect(kscore(r, f0.a, 0.3, 0.3)).toBe(0);
  });
  it('polynomial kernel equals the dot product of the lifted vectors', () => {
    for (const [x, z] of [[[0.8, 0.45], [-0.3, 0.9]], [[1, 2], [3, -1]], [[0, 0], [1, 1]]] as const) {
      const a = phi2(x[0], x[1]), b = phi2(z[0], z[1]);
      expect(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).toBeCloseTo(polyK(x, z), 12);
    }
    const f = identityFrameAt(0);
    expect(f.direct).toBeCloseTo(f.lifted, 12);
    expect(f.direct).toBeCloseTo(0.8425, 4);
    expect(identityFrameAt(2).direct).toBeCloseTo(0, 12);
    expect(identityFrameAt(4).direct).toBeCloseTo(0.8425, 12);
    expect(identityFrameAt(1.3, [0.2, -0.7]).z).toEqual([0.2, -0.7]);
  });
});
