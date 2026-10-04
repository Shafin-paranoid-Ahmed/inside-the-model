/**
 * SeqOrderDemo math: the same word vectors fed to (a) a bag-of-words feed-forward feature and
 * (b) a recurrent cell. Bag of words sums the word vectors, so word order cannot matter; the RNN cannot ignore order.
 * Shares the tiny seeded RNN from RNNUnroller (hidden size 4, recurrent scale 1).
 */
import { makeWeights, run, embed, H, ease, clamp01 } from '../RNNUnroller/sim';

export { H, ease, clamp01 };
export const words = (s: string, max = 6): string[] => s.trim().toLowerCase().split(/\s+/).filter(Boolean).slice(0, max);
export const dist = (u: number[], v: number[]): number => Math.hypot(...u.map((x, i) => x - v[i]));

export interface Sentence { toks: string[]; bow: number[][]; rnn: number[][] }
export interface Demo { a: Sentence; b: Sentence; dBow: number; dRnn: number; n: number }

/** Per-position features. bow[k] = tanh(W_x * sum_{i<=k} x_i + b); rnn[k] = h_k of the RNN. */
export function sentence(toks: string[]): Sentence {
  const w = makeWeights('rnn', 1);
  const ts = toks.length ? toks : ['…'];
  const xs = ts.map(embed), sum = Array(xs[0].length).fill(0);
  const bow = xs.map((x) => {
    x.forEach((q, i) => (sum[i] += q));
    return w.Wx.h.map((row, j) => Math.tanh(row.reduce((s, m, i) => s + m * sum[i], 0) + w.b.h[j]));
  });
  return { toks: ts, bow, rnn: run(w, ts).h };
}

export function makeDemo(a: string, b: string): Demo {
  const A = sentence(words(a)), B = sentence(words(b));
  return { a: A, b: B, dBow: dist(A.bow.at(-1)!, B.bow.at(-1)!), dRnn: dist(A.rnn.at(-1)!, B.rnn.at(-1)!), n: Math.max(A.toks.length, B.toks.length) };
}

export const timing = (n: number) => ({ readAt: 0.5, S: 0.9, cmpAt: 0.5 + n * 0.9 + 0.2, T: 0.5 + n * 0.9 + 0.2 + 0.8 + 1.6 });
export interface DemoFrame { fill: number[]; cmp: number; phase: 'read' | 'compare' }
export function frameAt(t: number, n: number): DemoFrame {
  const tm = timing(n);
  return {
    fill: Array.from({ length: n }, (_, k) => ease((t - (tm.readAt + k * tm.S)) / (tm.S * 0.8))),
    cmp: ease((t - tm.cmpAt) / 0.8),
    phase: t < tm.cmpAt ? 'read' : 'compare',
  };
}
export const markers = (n: number) => [{ at: 0, label: 'read' }, { at: timing(n).cmpAt, label: 'compare' }];
