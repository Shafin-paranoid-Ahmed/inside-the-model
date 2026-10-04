import { describe, it, expect } from 'vitest';
import {
  ACT, SAT, LEAK, flatZones, geluDip, gelu, silu, sigmoid, dSigmoid, plotFrameAt, collapseFrameAt, collapseResult, stagesOf,
  PRODUCT, W1, W2, W3, mul2, viewOf, duration, markers, ZR, COLLAPSE_T, type ActId, type PlotMode, type Collapse,
} from '../../src/components/widgets/ActivationLab/sim';

const smoothIds: ActId[] = ['sigmoid', 'tanh', 'gelu', 'silu'];
describe('ActivationLab sim', () => {
  it('derivatives match finite differences', () => {
    const h = 1e-6;
    for (const id of smoothIds) for (const z of [-5, -2.3, -0.7, -0.01, 0.4, 1.9, 4.2]) {
      const num = (ACT[id].f(z + h) - ACT[id].f(z - h)) / (2 * h);
      expect(ACT[id].d(z)).toBeCloseTo(num, 6);
    }
    for (const id of ['relu', 'leaky', 'step'] as const) for (const z of [-3, -0.5, 0.5, 3]) {
      const num = (ACT[id].f(z + 1e-4, 0.2) - ACT[id].f(z - 1e-4, 0.2)) / 2e-4;
      expect(ACT[id].d(z, 0.2)).toBeCloseTo(num, 6);
    }
  });
  it('known values', () => {
    expect(sigmoid(0)).toBe(0.5);
    expect(dSigmoid(0)).toBe(0.25);
    expect(ACT.tanh.d(0)).toBe(1);
    expect(ACT.relu.f(-3)).toBe(0);
    expect(ACT.leaky.f(-3)).toBeCloseTo(-3 * LEAK, 12);
    expect(gelu(0)).toBe(0);
    expect(gelu(5)).toBeCloseTo(5, 3);
    expect(silu(1)).toBeCloseTo(sigmoid(1), 12);
    expect(sigmoid(-800)).toBe(0);
  });
  it('GELU tanh approximation is close to the exact erf form', () => {
    const erf = (x: number) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
    for (const z of [-3, -1, -0.5, 0.5, 1, 3]) expect(gelu(z)).toBeCloseTo(0.5 * z * (1 + erf(z / Math.SQRT2)), 2);
  });
  it('dip and flat zones', () => {
    const d = geluDip();
    expect(d.z).toBeCloseTo(-0.75, 1);
    expect(d.v).toBeCloseTo(-0.17, 2);
    expect(ACT.gelu.d(d.z)).toBeCloseTo(0, 5);
    const sz = flatZones('sigmoid', 6);
    expect(sz).toHaveLength(2);
    expect(sz[1][0]).toBeCloseTo(2.94, 1);
    expect(Math.abs(dSigmoid(sz[1][0]) - SAT)).toBeLessThan(0.002);
    expect(flatZones('tanh', 6)[1][0]).toBeCloseTo(2.18, 1);
    expect(flatZones('relu', 4)).toEqual([[-4, expect.closeTo(0, 1)]]);
    expect(flatZones('step', 6)).toHaveLength(1);
    expect(flatZones('leaky', 4)).toHaveLength(0);
    expect(flatZones('gelu', 6).every(([a, b]) => b - a >= 0.25)).toBe(true);
  });
  it('plotFrameAt endpoints, clamping, determinism', () => {
    for (const m of ['saturation', 'relu-family', 'gelu', 'derivatives'] as PlotMode[]) {
      const T = duration(m);
      expect(T).toBe(2 * ZR[m]);
      expect(plotFrameAt(0, m).z).toBe(-ZR[m]);
      expect(plotFrameAt(T, m).z).toBe(ZR[m]);
      expect(plotFrameAt(T + 5, m).z).toBe(ZR[m]);
      expect(plotFrameAt(T / 3, m)).toEqual(plotFrameAt(T / 3, m));
      const mk = markers(m);
      expect(mk[0].at).toBe(0);
      expect(plotFrameAt(mk[mk.length - 1].at, m).phase).toBe(mk.length - 1);
    }
    const f = plotFrameAt(6, 'derivatives');
    expect(f.z).toBe(0);
    expect(f.panels[0].f).toBe(0.5);
    expect(f.panels[0].d).toBe(0.25);
  });
  it('collapse: without activation the three layers equal the product matrix', () => {
    expect(PRODUCT).toEqual(mul2(W3, mul2(W2, W1)));
    const r = collapseResult('none');
    expect(r.actual[0]).toBeCloseTo(r.single[0], 12);
    expect(r.actual[1]).toBeCloseTo(r.single[1], 12);
    const end = collapseFrameAt(COLLAPSE_T, 'none'), S = stagesOf('none').pts;
    expect(Array.from(end.pts)).toEqual(Array.from(S[6]));
    // every grid line stays straight: the midpoint of each line is the mean of its end points
    const n = 25;
    for (let l = 0; l < 10; l++) {
      const a = l * n * 2, b = (l * n + n - 1) * 2, m = (l * n + 12) * 2;
      expect(end.pts[m]).toBeCloseTo((end.pts[a] + end.pts[b]) / 2, 9);
      expect(end.pts[m + 1]).toBeCloseTo((end.pts[a + 1] + end.pts[b + 1]) / 2, 9);
    }
  });
  it('collapse: nonlinear activations bend the grid and break the single-matrix match', () => {
    // Numerical examples quoted in the lesson.
    const plain = collapseResult('none').actual;
    expect(plain[0]).toBeCloseTo(1.86, 2);
    expect(plain[1]).toBeCloseTo(-0.08, 2);
    const tanh = collapseResult('tanh').actual;
    expect(tanh[0]).toBeCloseTo(0.67, 2);
    expect(tanh[1]).toBeCloseTo(-0.13, 2);
    const rectified = collapseResult('relu').actual;
    expect(rectified[0]).toBeCloseTo(1.52, 2);
    expect(rectified[1]).toBeCloseTo(0.91, 2);
    for (const a of ['tanh', 'relu'] as Collapse[]) {
      const r = collapseResult(a);
      expect(Math.hypot(r.actual[0] - r.single[0], r.actual[1] - r.single[1])).toBeGreaterThan(0.3);
      const S = stagesOf(a).pts[6], n = 25;
      let bend = 0;
      for (let l = 0; l < 10; l++) { const a0 = l * n * 2, b0 = (l * n + n - 1) * 2, m = (l * n + 12) * 2; bend = Math.max(bend, Math.abs(S[m] - (S[a0] + S[b0]) / 2) + Math.abs(S[m + 1] - (S[a0 + 1] + S[b0 + 1]) / 2)); }
      expect(bend).toBeGreaterThan(0.1);
    }
  });
  it('collapse frames: endpoints, continuity, view holds everything', () => {
    const f0 = collapseFrameAt(0, 'tanh');
    expect(Array.from(f0.pts)).toEqual(Array.from(stagesOf('tanh').pts[0]));
    expect(f0.ghost).toBe(0);
    expect(collapseFrameAt(COLLAPSE_T, 'tanh').ghost).toBe(1);
    let prev = collapseFrameAt(0, 'relu').pts, worst = 0;
    for (let t = 0.02; t <= COLLAPSE_T; t += 0.02) { const p = collapseFrameAt(t, 'relu').pts; for (let i = 0; i < p.length; i++) worst = Math.max(worst, Math.abs(p[i] - prev[i])); prev = p; }
    expect(worst).toBeLessThan(0.25);
    expect(collapseFrameAt(5, 'tanh')).toEqual(collapseFrameAt(5, 'tanh'));
    for (const a of ['none', 'tanh', 'relu'] as Collapse[]) {
      for (const P of stagesOf(a).pts) for (const v of P) expect(Math.abs(v)).toBeLessThanOrEqual(viewOf(a));
      // The dashed linear comparison must stay visible in nonlinear runs too.
      for (const v of stagesOf('none').pts[6]) expect(Math.abs(v)).toBeLessThan(viewOf(a));
    }
  });
});
