import { describe, expect, it } from 'vitest';
import { softmax, crossEntropy, gradLogits, trajectory, frameAt, duration, Z0 } from '../../src/components/widgets/SoftmaxBars/sim';

describe('softmax bars', () => {
  it('softmax sums to 1, is shift invariant and stable for huge logits', () => {
    const p = softmax([1, 2, 3, 4]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    softmax([101, 102, 103, 104]).forEach((v, i) => expect(v).toBeCloseTo(p[i], 12));
    expect(softmax([1000, 0, 0, 0])[0]).toBeCloseTo(1, 12);
  });
  it('temperature sharpens and flattens', () => {
    const z = [1, 0, 0, 0];
    expect(softmax(z, 0.25)[0]).toBeGreaterThan(softmax(z, 1)[0]);
    expect(softmax(z, 100)[0]).toBeCloseTo(0.25, 2);
  });
  it('gradient p - y matches numerical differences of cross-entropy', () => {
    const z = [0.3, -1.2, 2, 0.5], k = 1, h = 1e-6, g = gradLogits(softmax(z), k);
    z.forEach((_, i) => {
      const a = z.slice(), b = z.slice(); a[i] += h; b[i] -= h;
      expect(g[i]).toBeCloseTo((crossEntropy(softmax(a), k) - crossEntropy(softmax(b), k)) / (2 * h), 6);
    });
    expect(g.reduce((a, b) => a + b, 0)).toBeCloseTo(0, 12);
  });
  it('training lowers the loss and is deterministic', () => {
    const tr = trajectory(2), l = (z: number[]) => crossEntropy(softmax(z), 2);
    expect(l(tr[tr.length - 1])).toBeLessThan(l(tr[0]) / 3);
    expect(trajectory(2)).toEqual(tr);
  });
  it('frameAt endpoints and loop', () => {
    expect(frameAt('loss', 0, 2).z).toEqual(Z0);
    expect(frameAt('loss', duration('loss'), 2).z).toEqual(Z0);
    expect(frameAt('loss', 7.9, 2).loss).toBeLessThan(frameAt('loss', 0, 2).loss);
    expect(frameAt('softmax', 5, 0).tau).toBeCloseTo(1, 12);
    expect(frameAt('softmax', 6.25, 0).tau).toBeLessThan(0.5);
    expect(frameAt('softmax', 8.75, 0).tau).toBeGreaterThan(2);
  });
});

it('swing and temp are the two halves of softmax', () => {
  expect(frameAt('swing', 1.3, 0)).toEqual(frameAt('softmax', 1.3, 0));
  expect(frameAt('temp', 1.3, 0)).toEqual(frameAt('softmax', 6.3, 0));
  expect(frameAt('swing', 3, 0).tau).toBe(1);
  expect(frameAt('temp', 1.25, 0).z).toEqual(Z0);
});
