import { describe, it, expect } from 'vitest';
import { makeBase, profile, loss, forward, frameAt, T, type Act } from '../../src/components/widgets/GradientDepth/sim';

describe('GradientDepth sim', () => {
  it('backward gradients equal finite differences (every activation, every checked layer)', () => {
    const b = makeBase(4, 12, 3);
    for (const [act, skip] of [['sigmoid', false], ['tanh', false], ['relu', false], ['sigmoid', true], ['relu', true]] as [Act, boolean][]) {
      const cfg = { act, scale: 1.3, skip };
      const p = profile(b, cfg);
      const { h } = forward(b, cfg);
      expect(p.norms[0]).toBeGreaterThan(0);
      for (const l of [0, 2, 3]) {
        for (let i = 0; i < b.width; i++) {
          const e = 1e-6, hp = Float64Array.from(h[l]), hm = Float64Array.from(h[l]);
          hp[i] += e; hm[i] -= e;
          const num = (loss(b, cfg, l, hp) - loss(b, cfg, l, hm)) / (2 * e);
          expect(p.g[l][i]).toBeCloseTo(num, 6);
        }
      }
    }
  });
  it('the seed gradient at the top has norm 1', () => {
    const p = profile(makeBase(), { act: 'tanh', scale: 1 });
    expect(p.norms[30]).toBeCloseTo(1, 12);
  });
  it('sigmoid vanishes: derivative <= 0.25, so the gradient shrinks by many orders of magnitude', () => {
    const p = profile(makeBase(), { act: 'sigmoid', scale: 1 });
    expect(Math.max(...p.meanDeriv)).toBeLessThanOrEqual(0.25);
    expect(p.log[0]).toBeLessThan(-10);
    expect(p.gain).toBeLessThan(0.5);
  });
  it('ReLU: He scale keeps the gradient near 1, too-large scale explodes, too-small vanishes', () => {
    const b = makeBase();
    const he = profile(b, { act: 'relu', scale: Math.SQRT2 });
    expect(Math.abs(he.log[0])).toBeLessThan(3);
    expect(profile(b, { act: 'relu', scale: 3 }).log[0]).toBeGreaterThan(6);
    expect(profile(b, { act: 'relu', scale: 0.5 }).log[0]).toBeLessThan(-6);
  });
  it('skip connections keep the gradient alive where sigmoid alone loses 15+ orders of magnitude', () => {
    const b = makeBase();
    expect(profile(b, { act: 'sigmoid', scale: 1 }).log[0]).toBeLessThan(-15);
    expect(profile(b, { act: 'sigmoid', scale: 1, skip: true }).log[0]).toBeGreaterThan(-3);
  });
  it('is deterministic', () => {
    const cfg = { act: 'tanh' as Act, scale: 1.7 };
    expect(profile(makeBase(), cfg).log).toEqual(profile(makeBase(), cfg).log);
  });
  it('frameAt: nothing revealed at t=0, everything at the end, output layer first', () => {
    const a = frameAt(0, 30), z = frameAt(T, 30);
    expect(a.reveal.every((r) => r === 0)).toBe(true);
    expect(z.reveal.every((r) => r === 1)).toBe(true);
    const mid = frameAt(4.5, 30);
    expect(mid.reveal[30]).toBeGreaterThan(mid.reveal[0]);
  });
});
