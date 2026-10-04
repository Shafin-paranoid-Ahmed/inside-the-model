/** Helper numbers quoted in the lessons (used by tests and the heads figure). */
import { cosine } from '../../../lib/transformer/math';
import { type Scene, headData, rowOf, argmax } from './core';

/** Per query row, half the L1 distance between the two heads of the selected layer (0 = same row, 1 = disjoint). */
export function rowDisagreement(s: Scene): Float64Array {
  const a = headData(s, s.layer, 0).A, b = headData(s, s.layer, 1).A, T = s.T, out = new Float64Array(T);
  for (let i = 0; i < T; i++) { let d = 0; for (let j = 0; j <= i; j++) d += Math.abs(a[i * T + j] - b[i * T + j]) / 2; out[i] = d; }
  return out;
}
/** Mean weight each head puts on the previous word, over the query rows 1..T-1 of this sentence. */
export function previousWordMean(s: Scene, layer: number, head: number): number {
  const { A } = headData(s, layer, head); let tot = 0;
  for (let i = 1; i < s.T; i++) tot += A[i * s.T + i - 1];
  return s.T > 1 ? tot / (s.T - 1) : 0;
}
/** Strongest weight from a word to an EARLIER word (the diagonal is a word looking at itself). */
export function strongestOffDiagonal(s: Scene, layer: number, head: number) {
  const { A } = headData(s, layer, head); let best = { weight: -1, from: 0, to: 0 };
  for (let i = 1; i < s.T; i++) for (let j = 0; j < i; j++) if (A[i * s.T + j] > best.weight) best = { weight: A[i * s.T + j], from: i, to: j };
  return best;
}
export function mixCosine(s: Scene, qi: number): { cos: number; top: number } {
  const { A, v, out } = headData(s), top = argmax(rowOf(A, s.T, qi), qi + 1);
  return { cos: cosine(rowOf(out, s.dk, qi), rowOf(v, s.dk, top)), top };
}
