import { describe, expect, test } from 'vitest';
import {
  conv2d, conv2dBackward, convLayer, map3, outSize, pool2d, pool2dBackward, relu, reluBackward,
  type Map3,
} from '../../src/lib/cnn/conv';
import { createRng } from '../../src/lib/math/rng';

const rng = createRng(7);
const fill = (a: Float64Array) => a.forEach((_, i) => (a[i] = rng.uniform(-1, 1)));
const randMap = (c: number, h: number, w: number) => { const m = map3(c, h, w); fill(m.data); return m; };
// Loss = sum(y * r) for a fixed random r, so dL/dy = r.
const dot = (a: Map3, b: Map3) => a.data.reduce((s, v, i) => s + v * b.data[i], 0);

describe('cnn core', () => {
  test('output size formula', () => {
    expect(outSize(28, 5)).toBe(24);
    expect(outSize(32, 3, 1, 1)).toBe(32);
    expect(outSize(7, 3, 2, 1)).toBe(4);
  });

  test('conv2d matches a hand-computed edge filter', () => {
    const x = map3(1, 3, 3, Float64Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1]));
    const L = convLayer(1, 1, 3);
    L.weight.set([-1, 0, 1, -1, 0, 1, -1, 0, 1]);
    expect(Array.from(conv2d(x, L).data)).toEqual([3]);
  });

  for (const [stride, pad] of [[1, 0], [2, 1]]) {
    test(`conv2d gradients match finite differences (stride ${stride}, pad ${pad})`, () => {
      const x = randMap(2, 5, 6);
      const L = convLayer(2, 3, 3, stride, pad);
      fill(L.weight); fill(L.bias);
      const r = randMap(3, outSize(5, 3, stride, pad), outSize(6, 3, stride, pad));
      const gW = new Float64Array(L.weight.length);
      const gB = new Float64Array(L.bias.length);
      const gx = conv2dBackward(x, L, r, gW, gB);
      const h = 1e-6;
      const num = (arr: Float64Array, i: number) => {
        const v = arr[i];
        arr[i] = v + h; const up = dot(conv2d(x, L), r);
        arr[i] = v - h; const dn = dot(conv2d(x, L), r);
        arr[i] = v;
        return (up - dn) / (2 * h);
      };
      for (const i of [0, 7, 30, 59]) expect(gx.data[i]).toBeCloseTo(num(x.data, i), 6);
      for (const i of [0, 13, 53]) expect(gW[i]).toBeCloseTo(num(L.weight, i), 6);
      expect(gB[2]).toBeCloseTo(num(L.bias, 2), 6);
    });
  }

  for (const kind of ['max', 'avg'] as const) {
    test(`${kind} pool gradients match finite differences`, () => {
      const x = randMap(2, 4, 6);
      const r = randMap(2, 2, 3);
      const gx = pool2dBackward(x, 2, kind, r);
      const h = 1e-6;
      for (let i = 0; i < x.data.length; i += 5) {
        const v = x.data[i];
        x.data[i] = v + h; const up = dot(pool2d(x, 2, kind), r);
        x.data[i] = v - h; const dn = dot(pool2d(x, 2, kind), r);
        x.data[i] = v;
        expect(gx.data[i]).toBeCloseTo((up - dn) / (2 * h), 6);
      }
    });
  }

  test('relu backward masks negative inputs', () => {
    const x = map3(1, 1, 3, Float64Array.from([-1, 0.5, 2]));
    expect(Array.from(relu(x).data)).toEqual([0, 0.5, 2]);
    expect(Array.from(reluBackward(x, map3(1, 1, 3, Float64Array.from([1, 1, 1]))).data)).toEqual([0, 1, 1]);
  });
});
