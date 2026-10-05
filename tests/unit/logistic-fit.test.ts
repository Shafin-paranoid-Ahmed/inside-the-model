import { describe, it, expect } from 'vitest';
import { POINTS, BEST, START, PATH, STEPS, loss, grad, sigmoid, accuracy, gdPath, frameAt, duration, markers, X_DOM, W_DOM, B_DOM } from '../../src/components/widgets/LogisticFit/sim';

describe('LogisticFit sim', () => {
  it('sigmoid basics and stability', () => {
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(2) + sigmoid(-2)).toBeCloseTo(1, 12);
    expect(sigmoid(-800)).toBe(0); expect(sigmoid(800)).toBe(1);
    expect(Number.isFinite(loss(500, 0))).toBe(true);
  });
  it('log-loss matches the textbook formula', () => {
    const w = 0.7, b = -0.3;
    const ref = POINTS.reduce((s, p) => { const q = sigmoid(w * p.x + b); return s - (p.y ? Math.log(q) : Math.log(1 - q)); }, 0) / POINTS.length;
    expect(loss(w, b)).toBeCloseTo(ref, 10);
    expect(loss(0, 0)).toBeCloseTo(Math.LN2, 12);
  });
  it('gradient matches finite differences', () => {
    const [w, b, h] = [1.3, -0.8, 1e-6], [gw, gb] = grad(w, b);
    expect(gw).toBeCloseTo((loss(w + h, b) - loss(w - h, b)) / (2 * h), 6);
    expect(gb).toBeCloseTo((loss(w, b + h) - loss(w, b - h)) / (2 * h), 6);
  });
  it('BEST is a stationary point and descent approaches it', () => {
    const [gw, gb] = grad(BEST.w, BEST.b);
    expect(Math.hypot(gw, gb)).toBeLessThan(1e-6);
    expect(PATH[0].loss).toBeGreaterThan(PATH[STEPS].loss);
    expect(PATH[STEPS].loss - BEST.loss).toBeLessThan(0.02);
    expect(accuracy(BEST.w, BEST.b)).toBeGreaterThan(0.6);
  });
  it('frameAt endpoints, continuity and determinism', () => {
    for (const m of ['train', 'sweep', 'steepen'] as const) {
      const T = duration(m);
      expect(frameAt(0, m)).toEqual(frameAt(0, m));
      let prev = frameAt(0, m);
      for (let t = 0.05; t <= T; t += 0.05) { const f = frameAt(t, m); expect(Math.abs(f.loss - prev.loss)).toBeLessThan(0.3); prev = f; }
      expect(markers(m).every((k) => k.at >= 0 && k.at <= T)).toBe(true);
    }
    expect(frameAt(0, 'train').w).toBeCloseTo(START.w, 12);
    expect(frameAt(duration('train'), 'train').loss).toBeCloseTo(PATH[STEPS].loss, 12);
    expect(frameAt(0, 'sweep').probe!.x).toBe(X_DOM[0]);
  });
  it('steepen: loss is lowest near scale 1', () => {
    const lo = frameAt(0, 'steepen'), hi = frameAt(duration('steepen'), 'steepen');
    const f = frameAt(markers('steepen')[1].at, 'steepen');
    expect(f.w).toBeCloseTo(BEST.w, 2);
    expect(f.loss).toBeLessThan(lo.loss); expect(f.loss).toBeLessThan(hi.loss);
  });
  it('custom start descends', () => {
    const p = gdPath({ w: 3, b: 2 });
    expect(p[STEPS].loss).toBeLessThan(p[0].loss);
  });
  it("knobs mode moves only b, then only w, and stays on the map", () => {
    const T = duration("knobs");
    for (let t = 0; t < T; t += 0.05) {
      const f = frameAt(t, "knobs");
      expect(f.w).toBeGreaterThanOrEqual(W_DOM[0]); expect(f.w).toBeLessThanOrEqual(W_DOM[1]);
      expect(f.b).toBeGreaterThanOrEqual(B_DOM[0]); expect(f.b).toBeLessThanOrEqual(B_DOM[1]);
      if (t < T / 2) expect(f.w).toBeCloseTo(BEST.w); else expect(f.b).toBeCloseTo(BEST.b);
    }
    expect(markers("knobs").map((m) => m.label)).toEqual(["slide b", "tilt w"]);
  });
  it("training replays from a placed guess and still reaches the minimum", () => {
    const s = { w: 3.5, b: 2.5 }, p = gdPath(s);
    expect(frameAt(0, "train", p).w).toBeCloseTo(s.w);
    expect(frameAt(duration("train"), "train", p).loss).toBeLessThan(loss(s.w, s.b));
  });
});
