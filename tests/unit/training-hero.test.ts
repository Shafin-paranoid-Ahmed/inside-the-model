import { describe, expect, test } from 'vitest';
import { MLP, createRng } from '../../src/lib/nn';
import { SIZES, PARAMS, forward, field, makeDataset, KINDS, EXTENT } from '../../src/components/widgets/TrainingHero/sim';
import { Trainer } from '../../src/components/widgets/TrainingHero/trainer';

describe('TrainingHero sim', () => {
  test('plain forward pass matches the engine MLP for the same weights', () => {
    const net = new MLP(2, SIZES.slice(1) as number[], { rng: createRng(3), act: 'tanh', outAct: 'tanh' });
    const w = new Float32Array(net.parameters().map((p) => p.data));
    expect(w.length).toBe(PARAMS);
    net.parameters().forEach((p, i) => { p.data = w[i]; }); // same float32-rounded weights on both sides
    for (const [x, y] of [[0, 0], [0.7, -1.1], [-1.5, 1.2]]) {
      expect(forward(w, x, y)).toBeCloseTo(net.call([x, y])[0].data, 10);
    }
  });

  test('field samples cell centers', () => {
    const w = new Float32Array(PARAMS).fill(0.1), out = new Float32Array(16);
    field(w, 4, out);
    const c = (0.5 / 4) * 2 * EXTENT - EXTENT;
    expect(out[0]).toBeCloseTo(forward(w, c, -c), 6);
  });

  test('datasets are seeded, balanced and inside the plot', () => {
    for (const k of KINDS) {
      const a = makeDataset(k, 1), b = makeDataset(k, 1);
      expect(Array.from(a.xs)).toEqual(Array.from(b.xs));
      expect(a.ys.reduce((s, v) => s + v, 0)).toBe(0);
      expect(a.xs.every((v) => Math.abs(v) < EXTENT)).toBe(true);
    }
  });

  test('trainer reduces loss on moons', () => {
    const t = new Trainer(1, makeDataset('moons'));
    t.train();
    const first = t.loss;
    for (let i = 0; i < 400; i++) t.train();
    expect(t.loss).toBeLessThan(first * 0.4);
    expect(t.snapshot().weights.length).toBe(PARAMS);
  });
});
