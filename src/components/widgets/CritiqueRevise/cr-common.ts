/** Helpers shared by the constitutional-ai modes (not a mode: no m- prefix). No DOM. */
import { K, TRAIN } from '../../../lib/align/bank';
import { violationCount } from '../../../lib/align/constitution';
import { probsInto, type Theta } from '../../../lib/align/policy';

/** Probability of each candidate averaged over the train prompts. */
export function avgProbs(theta: Theta): Float64Array {
  const out = new Float64Array(K), p = new Float64Array(K);
  for (const x of TRAIN) { probsInto(theta, x, p); for (let y = 0; y < K; y++) out[y] += p[y] / TRAIN.length; }
  return out;
}
/** Expected number of broken principles per response, averaged over the train prompts. */
export function expViolations(theta: Theta, act: string[]): number {
  const p = new Float64Array(K);
  let s = 0;
  for (const x of TRAIN) { probsInto(theta, x, p); for (let y = 0; y < K; y++) s += p[y] * violationCount(x, y, act); }
  return s / TRAIN.length;
}
/** What a critique says when the sampled principle is broken (the fragment completes "Critique: ..."). */
export const CRITIQUE: Record<string, string> = {
  answer: 'it does not answer the question',
  accurate: 'the answer it gives is wrong',
  'no-flattery': 'it pads the answer with flattery',
  'no-refuse': 'it refuses a request that is harmless',
  cautious: 'it states an answer without any hedge',
};
export const plural = (n: number, w: string): string => `${n} ${w}${n === 1 ? '' : 's'}`;
