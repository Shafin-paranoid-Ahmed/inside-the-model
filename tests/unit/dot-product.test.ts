import { describe, it, expect } from 'vitest';
import { dot, norm, cosSim, project, frameAt, T, A_DEF, W_DEF, BIAS, relu, type Vec } from '../../src/components/widgets/DotProduct/sim';

describe('dot product math', () => {
  it('multiplies and adds', () => expect(dot([1, 2], [3, 4])).toBe(11));
  it('equals |a||b|cos(theta)', () => {
    const f = frameAt(3.3, 'geometry');
    expect(f.dot).toBeCloseTo(norm(f.a) * norm(f.b) * Math.cos(f.theta), 10);
  });
  it('cosine similarity ignores length', () => {
    expect(cosSim([1, 2], [10, 20])).toBeCloseTo(1, 12);
    expect(cosSim([1, 0], [0, 5])).toBeCloseTo(0, 12);
    expect(cosSim([0, 0], [1, 1])).toBe(0);
  });
  it('projection leaves a residual perpendicular to a', () => {
    const a: Vec = [2, 1], b: Vec = [-1, 3], p = project(b, a);
    expect(dot([b[0] - p[0], b[1] - p[1]], a)).toBeCloseTo(0, 12);
  });
});

describe('frameAt', () => {
  it('b starts aligned with a, is perpendicular at T/4, opposite at T/2', () => {
    const m = norm(A_DEF) * norm(frameAt(0, 'geometry').b);
    expect(frameAt(0, 'geometry').cos).toBeCloseTo(1, 10);
    expect(frameAt(T / 4, 'geometry').dot).toBeCloseTo(0, 10);
    expect(frameAt(T / 2, 'geometry').dot).toBeCloseTo(-m, 10);
    expect(frameAt(T / 2, 'geometry').theta360).toBeCloseTo(180, 6);
  });
  it('loops and is deterministic', () => {
    expect(frameAt(T, 'bars').dot).toBeCloseTo(frameAt(0, 'bars').dot, 10);
    expect(frameAt(2.7, 'bars')).toEqual(frameAt(2.7, 'bars'));
  });
  it('elementwise products sum to the dot product', () => {
    for (const t of [0, 1, 4.4, 9]) { const f = frameAt(t, 'bars'); expect(f.total).toBeCloseTo(f.dot, 12); }
  });
  it('override replaces the vectors', () => {
    const f = frameAt(5, 'bars', { a: [1, 0], b: [0, 2] });
    expect(f.dot).toBe(0);
    expect(f.theta360).toBeCloseTo(90, 8);
  });
  it('neuron: z = w.x + bias, output is ReLU(z)', () => {
    for (const t of [0, 3, 6, 9]) {
      const f = frameAt(t, 'neuron');
      expect(f.a).toEqual(W_DEF);
      expect(f.total).toBeCloseTo(dot(f.a, f.b) + BIAS, 12);
      expect(f.out).toBe(relu(f.total));
    }
    expect(frameAt(0, 'neuron').out).toBeGreaterThan(0);
    expect(frameAt(T / 2, 'neuron').out).toBe(0);
  });
});
