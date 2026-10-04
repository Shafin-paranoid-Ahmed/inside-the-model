import { describe, it, expect } from 'vitest';
import { POINTS, FINAL, DOT_X, W_LEN, sigmoid, activate, shade, preact, accuracy, frameAt, duration, markers, type Mode } from '../../src/components/widgets/NeuronPlayground/sim';

const MODES: Mode[] = ['weighted-sum', 'dot-product', 'activation', 'playground'];

describe('NeuronPlayground sim', () => {
  it('dataset is balanced, seeded and separated by FINAL', () => {
    expect(POINTS.length).toBe(24);
    expect(POINTS.filter((p) => p.y).length).toBe(12);
    expect(accuracy(FINAL)).toBe(1);
  });
  it('activations', () => {
    expect(activate('none', -2.5)).toBe(-2.5);
    expect(activate('step', 0.01)).toBe(1);
    expect(activate('step', -0.01)).toBe(0);
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(-800)).toBe(0);
    expect(shade('sigmoid', 0)).toBe(0);
    expect(Math.abs(shade('none', 50))).toBe(1);
  });
  it('weighted sum equals the sum of the two products plus bias', () => {
    for (const m of MODES) for (let t = 0; t <= duration(m); t += 0.7) {
      const f = frameAt(t, m);
      expect(f.z).toBeCloseTo(f.p1 + f.p2 + f.b, 12);
      expect(f.z).toBeCloseTo(preact(f, f.x1, f.x2), 12);
    }
  });
  it('dot-product step: w.x = |w||x|cos(theta) and b = 0', () => {
    const nx = Math.hypot(DOT_X.x1, DOT_X.x2);
    for (let t = 0; t <= duration('dot-product'); t += 0.37) {
      const f = frameAt(t, 'dot-product');
      expect(f.b).toBe(0);
      expect(Math.hypot(f.w1, f.w2)).toBeCloseTo(W_LEN, 12);
      expect(f.z).toBeCloseTo(W_LEN * nx * Math.cos(f.theta), 10);
    }
  });
  it('markers name the state shown, and the phase flips where z changes sign', () => {
    for (const m of ['weighted-sum', 'dot-product'] as Mode[]) {
      const T = duration(m), mk = markers(m);
      for (let t = 0; t < T; t += 0.05) {
        const f = frameAt(t, m);
        const label = [...mk].reverse().find((k) => t >= k.at)!.label;
        if (Math.abs(f.z) > 0.05) expect(label.includes('positive') || label.includes('above') ? f.z > 0 : f.z < 0).toBe(true);
      }
    }
  });
  it('activation step cycles none, step, sigmoid and the point crosses the boundary each time', () => {
    expect([0.1, 4.1, 8.1].map((t) => frameAt(t, 'activation').act)).toEqual(['none', 'step', 'sigmoid']);
    for (const t0 of [0, 4, 8]) { const zs = [1, 3].map((q) => frameAt(t0 + q, 'activation').z); expect(zs[0]).toBeGreaterThan(1.5); expect(zs[1]).toBeLessThan(-1.5); }
  });
  it('playground script starts poor and ends separating the data', () => {
    expect(accuracy(frameAt(0, 'playground')) * POINTS.length).toBe(11);
    expect(accuracy(frameAt(6, 'playground')) * POINTS.length).toBe(20);
    const e = frameAt(duration('playground'), 'playground');
    expect(e.w1).toBeCloseTo(FINAL.w1, 12); expect(e.b).toBeCloseTo(FINAL.b, 12);
    expect(accuracy(e)).toBe(1);
    expect(frameAt(1, 'playground', { w1: 2, w2: 0, b: 1 }).phase).toBe('manual');
  });
  it('lesson examples match full-precision sums and outputs', () => {
    expect(frameAt(3, 'weighted-sum').z).toBeCloseTo(-1.90, 10);
    expect(frameAt(9, 'weighted-sum').z).toBeCloseTo(1.30, 10);
    expect(frameAt(3, 'dot-product').z).toBeCloseTo(2.94, 2);
    expect(frameAt(5, 'dot-product').z).toBeCloseTo(0, 10);
    expect(frameAt(6, 'dot-product').z).toBeCloseTo(-1.70, 2);
    expect(frameAt(1, 'activation').out).toBeCloseTo(1.98, 2);
    expect(frameAt(5, 'activation').out).toBe(1);
    expect(frameAt(9, 'activation').out).toBeCloseTo(0.88, 2);
    expect(frameAt(10, 'activation').out).toBeCloseTo(0.5, 10);
  });
  it('frameAt is deterministic and continuous; markers lie inside the timeline', () => {
    for (const m of MODES) {
      expect(frameAt(-1, m)).toEqual(frameAt(0, m));
      expect(frameAt(duration(m) + 1, m)).toEqual(frameAt(duration(m), m));
      expect(frameAt(1.3, m)).toEqual(frameAt(1.3, m));
      const T = duration(m); let prev = frameAt(0, m);
      for (let t = 0.02; t <= T; t += 0.02) { const f = frameAt(t, m); if (!(m === 'activation' && Math.abs(t % 4) < 0.03)) expect(Math.abs(f.z - prev.z)).toBeLessThan(0.15); prev = f; }
      expect(markers(m).every((k) => k.at >= 0 && k.at <= T)).toBe(true);
    }
  });
});
