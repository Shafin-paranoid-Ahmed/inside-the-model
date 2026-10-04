import { describe, expect, test } from 'vitest';
import { defaultGraph, gcnNorm, softmax, leaky, trace, hops, frameAt, duration, F, type Graph, type Cfg } from '../../src/components/widgets/GNNMessagePassing/sim';

const path3: Graph = { x: [0, 0.5, 1], y: [0, 0, 0], h: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], edges: [[0, 1], [1, 2]] };
const cfg: Cfg = { model: 'gcn', agg: 'gcn', layers: 2, seed: 3 };

describe('GCN normalization', () => {
  test('path graph: D^-1/2 (A+I) D^-1/2 by hand', () => {
    const A = gcnNorm(path3); // d~ = [2,3,2]
    expect(A[0][0]).toBeCloseTo(1 / 2, 12);
    expect(A[1][1]).toBeCloseTo(1 / 3, 12);
    expect(A[0][1]).toBeCloseTo(1 / Math.sqrt(6), 12);
    expect(A[0][2]).toBe(0);
    expect(A[1][0]).toBeCloseTo(A[0][1], 12);
  });
  test('message-passing trace equals tanh(A_hat H W)', () => {
    const g = defaultGraph(), tr = trace(g, cfg), A = gcnNorm(g);
    const W = tr.W[0], n = g.x.length;
    for (let i = 0; i < n; i++) for (let c = 0; c < F; c++) {
      let s = 0;
      for (let j = 0; j < n; j++) for (let r = 0; r < F; r++) s += A[i][j] * g.h[j][r] * W[r][c];
      expect(tr.H[1][i][c]).toBeCloseTo(Math.tanh(s), 10);
    }
  });
});

describe('GAT softmax', () => {
  test('sums to 1, shift-invariant, matches a hand value', () => {
    const a = softmax([1, 2, 3]);
    expect(a.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
    expect(a[2]).toBeCloseTo(Math.exp(3) / (Math.exp(1) + Math.exp(2) + Math.exp(3)), 12);
    softmax([1001, 1002, 1003]).forEach((v, i) => expect(v).toBeCloseTo(a[i], 12));
    expect(leaky(-2)).toBeCloseTo(-0.4, 12);
  });
  test('attention over each neighborhood sums to 1', () => {
    const g = defaultGraph(), tr = trace(g, { ...cfg, model: 'gat' });
    for (let i = 0; i < g.x.length; i++) {
      const inc = tr.layers[0].msgs.filter((m) => m.to === i);
      expect(inc.reduce((s, m) => s + m.coef, 0)).toBeCloseTo(1, 12);
    }
  });
});

describe('aggregators, hops, timeline', () => {
  test('sum/mean/max on a star centre', () => {
    const star: Graph = { x: [0, 1, 1], y: [0, 0, 1], h: [[0, 0, 0], [1, 0, 0], [3, 0, 0]], edges: [[0, 1], [0, 2]] };
    const run = (agg: 'sum' | 'mean' | 'max') => trace(star, { ...cfg, agg, layers: 1 }).layers[0];
    const col = run('sum').msgs.filter((m) => m.to === 0).map((m) => m.vec[0]);
    const S = col.reduce((a, b) => a + b);
    expect(run('sum').pre[0][0]).toBeCloseTo(S, 12);
    expect(run('mean').pre[0][0]).toBeCloseTo(S / 3, 12);
    expect(run('max').pre[0][0]).toBeCloseTo(Math.max(...col), 12);
  });
  test('hop distances', () => {
    expect(hops(path3, 0)).toEqual([0, 1, 2]);
    expect(hops(defaultGraph(), 3)).toEqual([2, 2, 1, 0, 3, 4, 4]);
  });
  test('frameAt endpoints and determinism', () => {
    const g = defaultGraph(), tr = trace(g, cfg), T = duration(2, true);
    expect(frameAt(0, tr, 2, true).feat).toEqual(g.h);
    const b = frameAt(T, tr, 2, true);
    expect(b.feat).toEqual(tr.H[2]);
    expect(b.k).toBe(2);
    expect(frameAt(3.3, tr, 2, true)).toEqual(frameAt(3.3, trace(g, cfg), 2, true));
  });
});
