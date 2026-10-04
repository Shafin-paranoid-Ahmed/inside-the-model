/** Frame helpers shared by the softmax and explorer figures. */
import { softmax } from '../../../lib/transformer/math';
import { type Scene, headData, rowOf, type PipelineRow } from './core';

/** One query's whole pipeline: raw scores, scaled, masked softmax, weighted sum of values. */
export function pipelineRow(s: Scene, q: number, layer = s.layer, head = s.head, opts: { noScale?: boolean } = {}): PipelineRow {
  const { S, A, v } = headData(s, layer, head), T = s.T, dk = s.dk;
  const raw = Float64Array.from(rowOf(S, T, q)), scaled = new Float64Array(T), w = Float64Array.from(rowOf(A, T, q));
  const k = opts.noScale ? 1 : 1 / Math.sqrt(dk);
  for (let j = 0; j < T; j++) scaled[j] = raw[j] * k;
  const out = new Float64Array(dk);
  for (let j = 0; j <= q; j++) for (let d = 0; d < dk; d++) out[d] += w[j] * v[j * dk + d];
  return { raw, scaled, w, out };
}
/** Softmax over the visible keys only (j up to q); future entries are exactly 0. */
export function maskedSoftmax(vals: ArrayLike<number>, q: number, T: number): Float64Array {
  const out = new Float64Array(T), p = softmax(Array.from(vals).slice(0, q + 1));
  out.set(p); return out;
}

