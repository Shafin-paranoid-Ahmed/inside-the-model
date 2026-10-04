import { describe, expect, test } from 'vitest';
import { makeWeights, run, embed, frameAt, timing, specNorm, H, D, type Cell, type Weights, type Mat } from '../../src/components/widgets/RNNUnroller/sim';

const toks = Array.from('ab 12c');
const sig = (x: number) => 1 / (1 + Math.exp(-x));
const clone = (w: Weights): Weights => structuredClone(w);

describe.each(['rnn', 'lstm', 'gru'] as Cell[])('BPTT gradients match numerical gradients: %s', (cell) => {
  for (const lossAt of ['last', 'all'] as const) {
    test(`loss at ${lossAt}`, () => {
      const w = makeWeights(cell, 0.9);
      const tr = run(w, toks, lossAt);
      const L = (m: Weights) => run(m, toks, lossAt).loss;
      const eps = 1e-6;
      for (const k of Object.keys(w.Wh)) {
        const checks: [string, (m: Weights, d: number) => void, number][] = [
          ['Wh', (m, d) => (m.Wh[k][1][2] += d), tr.grads.Wh[k][1][2]],
          ['Wx', (m, d) => (m.Wx[k][2][0] += d), tr.grads.Wx[k][2][0]],
          ['b', (m, d) => (m.b[k][3] += d), tr.grads.b[k][3]],
        ];
        for (const [, bump, analytic] of checks) {
          const a = clone(w), b = clone(w);
          bump(a, eps); bump(b, -eps);
          expect(analytic).toBeCloseTo((L(a) - L(b)) / (2 * eps), 6);
        }
      }
      const a = clone(w), b = clone(w);
      a.v[1] += eps; b.v[1] -= eps;
      expect(tr.grads.v[1]).toBeCloseTo((L(a) - L(b)) / (2 * eps), 6);
    });
  }
});

describe('vanilla RNN', () => {
  const w = makeWeights('rnn', 1);
  test('forward equals h_t = tanh(Wxh x + Whh h + b)', () => {
    const tr = run(w, toks);
    let h = Array(H).fill(0);
    toks.forEach((tk, t) => {
      const x = embed(tk);
      h = h.map((_, j) => Math.tanh(w.Wx.h[j].reduce((s, q, i) => s + q * x[i], 0) + w.Wh.h[j].reduce((s, q, i) => s + q * h[i], 0) + w.b.h[j]));
      h.forEach((v, j) => expect(tr.h[t][j]).toBeCloseTo(v, 12));
    });
  });
  test('Whh has all singular values equal to scale', () => {
    for (const s of [0.3, 1, 2.5]) expect(specNorm(makeWeights('rnn', s).Wh.h)).toBeCloseTo(s, 6);
  });
  test('per-step contributions sum to the full dL/dWhh (gradient sums over time)', () => {
    const tr = run(w, toks, 'all');
    const sum: Mat = Array.from({ length: H }, () => Array(H).fill(0));
    tr.contribMat.forEach((m) => m.forEach((r, i) => r.forEach((q, j) => (sum[i][j] += q))));
    sum.forEach((r, i) => r.forEach((q, j) => expect(q).toBeCloseTo(tr.grads.Wh.h[i][j], 10)));
  });
  test('gradient at step t is the transposed Jacobian product applied to the last-step gradient', () => {
    const tr = run(w, toks, 'last');
    // ||dh_t|| <= ||dh_T|| * ||prod J||_2
    tr.dhN.forEach((g, t) => expect(g).toBeLessThanOrEqual(tr.dhN[tr.n - 1] * tr.jac[t] + 1e-12));
    expect(tr.jac[tr.n - 1]).toBe(1);
  });
  test('small recurrent scale vanishes, large explodes; deterministic', () => {
    const long = Array.from('hello world');
    expect(run(makeWeights('rnn', 0.5), long).factor).toBeLessThan(0.6);
    expect(run(makeWeights('rnn', 3), long).factor).toBeGreaterThan(1.2);
    expect(run(w, toks).h).toEqual(run(makeWeights('rnn', 1), toks).h);
  });
});

describe('LSTM and GRU equations', () => {
  test('LSTM step: c = f*c_prev + i*g, h = o*tanh(c); gates respond to digits and separators', () => {
    const w = makeWeights('lstm', 1);
    const tr = run(w, toks);
    for (let t = 0; t < toks.length; t++) {
      const x = embed(toks[t]);
      const hp = t ? tr.h[t - 1] : Array(H).fill(0), cp = t ? tr.c[t - 1] : Array(H).fill(0);
      const pre = (k: string, j: number) => w.Wx[k][j].reduce((s, q, i) => s + q * x[i], 0) + w.Wh[k][j].reduce((s, q, i) => s + q * hp[i], 0) + w.b[k][j];
      for (let j = 0; j < H; j++) {
        const f = sig(pre('f', j)), i = sig(pre('i', j)), o = sig(pre('o', j)), g = Math.tanh(pre('g', j));
        expect(tr.gates.f[t][j]).toBeCloseTo(f, 12);
        const c = f * cp[j] + i * g;
        expect(tr.c[t][j]).toBeCloseTo(c, 12);
        expect(tr.h[t][j]).toBeCloseTo(o * Math.tanh(c), 12);
      }
    }
    const mean = (r: number[]) => r.reduce((a, b) => a + b, 0) / r.length;
    expect(mean(tr.gates.f[2])).toBeLessThan(0.5); // ' ' closes forget
    expect(mean(tr.gates.i[3])).toBeGreaterThan(mean(tr.gates.i[0])); // digit opens input
  });
  test('GRU step: h = (1-z)*h_prev + z*n with reset inside the candidate', () => {
    const w = makeWeights('gru', 1);
    const tr = run(w, toks);
    for (let t = 0; t < toks.length; t++) {
      const x = embed(toks[t]);
      const hp = t ? tr.h[t - 1] : Array(H).fill(0);
      const dot = (r: number[], v: number[]) => r.reduce((s, q, i) => s + q * v[i], 0);
      for (let j = 0; j < H; j++) {
        const z = sig(dot(w.Wx.z[j], x) + dot(w.Wh.z[j], hp) + w.b.z[j]);
        const r = sig(dot(w.Wx.r[j], x) + dot(w.Wh.r[j], hp) + w.b.r[j]);
        const n = Math.tanh(dot(w.Wx.n[j], x) + w.b.n[j] + r * dot(w.Wh.n[j], hp));
        expect(tr.h[t][j]).toBeCloseTo((1 - z) * hp[j] + z * n, 12);
      }
    }
  });
  test('LSTM cell-state gradient survives longer than the vanilla hidden gradient', () => {
    const long = Array.from('1234567890ab');
    const l = run(makeWeights('lstm', 1), long), r = run(makeWeights('rnn', 1), long);
    expect(l.factor).toBeGreaterThan(r.factor);
  });
  test('the cell-state gradient tracks the forget gate, not the recurrent scale', () => {
    const long = Array.from('abcdefghijkl');
    expect(makeWeights('lstm', 1)).toEqual(makeWeights('lstm', 1, 11, 1.5)); // default bias unchanged
    const at = (s: number, fb: number) => run(makeWeights('lstm', s, 11, fb), long).factor;
    expect(Math.abs(at(0.5, 1.5) - at(2, 1.5))).toBeLessThan(0.03);
    expect(at(1, 4)).toBeGreaterThan(0.95);
    expect(at(1, 0)).toBeLessThan(0.6);
  });
});

describe('frameAt', () => {
  const tm = timing(4, true, true);
  test('endpoints', () => {
    const f0 = frameAt(0, 4, tm);
    expect(f0.phase).toBe('fold'); expect(f0.unroll).toBe(0); expect(f0.steps[3].present).toBe(0); expect(f0.steps[0].hP).toBe(0);
    const f1 = frameAt(tm.T, 4, tm);
    expect(f1.phase).toBe('backward'); expect(f1.unroll).toBe(1);
    f1.steps.forEach((s) => { expect(s.hP).toBe(1); expect(s.gradP).toBe(1); });
  });
  test('backward wave starts at the last step', () => {
    const f = frameAt(tm.backAt + 0.3, 4, tm);
    expect(f.steps[3].gradP).toBeGreaterThan(0); expect(f.steps[0].gradP).toBe(0); expect(f.active).toBe(3);
  });
  test('unfolded timeline skips fold and unroll', () => {
    const u = timing(4, false, false);
    expect(frameAt(0, 4, u).unroll).toBe(1);
    expect(frameAt(0, 4, u).steps[3].present).toBe(1);
  });
  test('sizes', () => { expect(D).toBe(4); expect(embed('a')).toHaveLength(D); });
});

describe('truncated BPTT', () => {
  const w = makeWeights('rnn', 1);
  const long = Array.from('hello world');
  test('gradients stop at chunk boundaries and match full BPTT inside the last chunk', () => {
    const full = run(w, long, 'last'), tr = run(w, long, 'last', 4);
    // steps 0..7 are in earlier chunks (chunks start at 0, 4, 8)
    for (let t = 0; t < 8; t++) expect(tr.dhN[t]).toBe(0);
    for (let t = 8; t < long.length; t++) tr.dh[t].forEach((g, j) => expect(g).toBeCloseTo(full.dh[t][j], 12));
    expect(tr.loss).toBeCloseTo(full.loss, 12); // forward pass is unchanged
  });
});
