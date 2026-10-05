import { describe, it, expect } from 'vitest';
import { add, sub, scaleBy, dot, norm, dist, frameAt, scaleAt, duration, markers, HOUSE_POINT, FEATURES, DEFAULT_A, DEFAULT_B } from '../../src/components/widgets/VectorPlane/sim';

const scenes = ['add', 'scale', 'length', 'features'] as const;

describe('vector helpers', () => {
  it('adds, subtracts, scales', () => {
    expect(add({ x: 1, y: 2 }, { x: 3, y: -1 })).toEqual({ x: 4, y: 1 });
    expect(sub({ x: 1, y: 2 }, { x: 3, y: -1 })).toEqual({ x: -2, y: 3 });
    expect(scaleBy(-2, { x: 1, y: -3 })).toEqual({ x: -2, y: 6 });
  });
  it('norm and distance follow Pythagoras', () => {
    expect(norm({ x: 3, y: 4 })).toBe(5);
    expect(dist({ x: 1, y: 1 }, { x: 4, y: 5 })).toBe(5);
    expect(dot({ x: 1, y: 2 }, { x: 3, y: 4 })).toBe(11);
  });
  it('the house point is (3, 4) with length 5', () => {
    expect(HOUSE_POINT).toEqual({ x: 3, y: 4 });
    expect(norm(HOUSE_POINT)).toBe(5);
    expect(FEATURES).toHaveLength(6);
  });
});

describe('frameAt', () => {
  it('add: slide and sum end at 1 and reset to 0', () => {
    expect(frameAt(0, 'add').sum).toBe(0);
    expect(frameAt(6.5, 'add').slide).toBe(1);
    expect(frameAt(6.5, 'add').sum).toBe(1);
    expect(frameAt(duration('add'), 'add').sum).toBe(0);
  });
  it('scale: c hits 1, 2, 0, -1.5 and is continuous', () => {
    expect(scaleAt(0)).toBe(1);
    expect(scaleAt(3)).toBe(2);
    expect(scaleAt(5)).toBe(0);
    expect(scaleAt(7.5)).toBe(-1.5);
    let prev = scaleAt(0);
    for (let t = 0.01; t <= 11; t += 0.01) { const c = scaleAt(t); expect(Math.abs(c - prev)).toBeLessThan(0.1); prev = c; }
  });
  it('length: readouts count up to the true norm and distance', () => {
    const f = frameAt(4, 'length', DEFAULT_A, DEFAULT_B);
    expect(f.lenA).toBeCloseTo(norm(DEFAULT_A), 10);
    expect(frameAt(7.6, 'length', DEFAULT_A, DEFAULT_B).distAB).toBeCloseTo(dist(DEFAULT_A, DEFAULT_B), 10);
    expect(frameAt(2, 'length').lenA).toBeGreaterThan(0);
    expect(frameAt(2, 'length').lenA).toBeLessThan(norm(DEFAULT_A));
  });
  it('features: cells grow from 0 to 6 and dissolve at loop end', () => {
    expect(frameAt(0, 'features').cells).toBe(0);
    expect(frameAt(4.7, 'features').move).toBe(1);
    expect(frameAt(9, 'features').cells).toBe(6);
    expect(frameAt(duration('features'), 'features').cells).toBeCloseTo(0, 10);
  });
  it('is pure and deterministic; every scene has markers within its duration', () => {
    for (const s of scenes) {
      expect(frameAt(3.3, s)).toEqual(frameAt(3.3, s));
      for (const m of markers(s)) expect(m.at).toBeLessThan(duration(s));
    }
  });
});

it('norm and distance scenes each animate one idea', () => {
  const n = frameAt(3.2, 'norm'), d = frameAt(3.2, 'distance');
  expect(n.lenA).toBeCloseTo(norm(DEFAULT_A), 9);
  expect(n.gap).toBe(0);
  expect(d.distAB).toBeCloseTo(dist(DEFAULT_A, DEFAULT_B), 9);
  expect(d.legs).toBe(0);
  expect(frameAt(5.6, 'norm').legs).toBeCloseTo(0, 9);
});
