import { decodeStep, forward, prefill, softmax, type KVCache, type Weights } from '../transformer/model';

/** Softmax of logits / T. T = 0 is greedy: all mass on the first largest logit. */
export function softmaxT(logits: ArrayLike<number>, T: number): Float64Array {
  if (!(T >= 0)) throw new Error('Temperature must be >= 0');
  if (T === 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    const p = new Float64Array(logits.length); p[best] = 1; return p;
  }
  return softmax(Array.from(logits, (z) => z / T));
}

export interface Kept { ids: number[]; probs: Float64Array } // probs: full length, zero outside ids, sums to 1
/** Ids by descending probability; ties keep the lower id first. */
const ranked = (p: ArrayLike<number>) => Array.from(p, (_, i) => i).sort((a, b) => p[b] - p[a] || a - b);
function keep(p: ArrayLike<number>, ids: number[]): Kept {
  const probs = new Float64Array(p.length);
  let total = 0;
  for (const id of ids) total += p[id];
  for (const id of ids) probs[id] = p[id] / total;
  return { ids, probs };
}
export function topK(p: ArrayLike<number>, k: number): Kept {
  if (!Number.isInteger(k) || k < 1) throw new Error('k must be a positive integer');
  return keep(p, ranked(p).slice(0, k));
}
/** Nucleus: the smallest prefix of the ranking whose probability reaches `top`. */
export function topP(p: ArrayLike<number>, top: number): Kept {
  if (!(top > 0 && top <= 1)) throw new Error('p must be in (0, 1]');
  const order = ranked(p), ids: number[] = [];
  let sum = 0;
  for (const id of order) { ids.push(id); sum += p[id]; if (sum >= top - 1e-12) break; }
  return keep(p, ids);
}
/** Inverse CDF: the first id whose cumulative interval contains u in [0, 1). */
export function pickIndex(p: ArrayLike<number>, u: number): number {
  let sum = 0, last = -1;
  for (let i = 0; i < p.length; i++) {
    if (p[i] <= 0) continue;
    sum += p[i]; last = i;
    if (u < sum) return i;
  }
  if (last < 0) throw new Error('No probability mass');
  return last; // rounding left u just past the final interval
}
/** Shannon entropy in nats (divide by Math.LN2 for bits). */
export function entropy(p: ArrayLike<number>): number {
  let h = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log(p[i]);
  return h;
}

export interface SampleOptions { temperature?: number; topK?: number; topP?: number }
export interface SampleStep {
  id: number;
  model: Float64Array; // the model's own distribution (T = 1, untruncated)
  sampled: Float64Array; // after temperature, then top-k, then top-p (the Hugging Face order)
  logp: number; // ln model[id]
}
/** Applies temperature, then top-k, then top-p. */
export function shape(logits: ArrayLike<number>, opts: SampleOptions = {}): Float64Array {
  let p = softmaxT(logits, opts.temperature ?? 1);
  if (opts.topK !== undefined) p = topK(p, opts.topK).probs;
  if (opts.topP !== undefined) p = topP(p, opts.topP).probs;
  return p;
}

/** Generates n tokens. Step i draws with us[i] (greedy when temperature is 0), so a different
 * temperature with the same `us` is a like-for-like comparison. Within the context window the
 * KV cache is used; past it, each step re-reads the last `context` tokens (a sliding window).
 */
export function generate(w: Weights, prompt: readonly number[], n: number, opts: SampleOptions, us: ArrayLike<number>): { ids: number[]; steps: SampleStep[] } {
  if (!prompt.length) throw new Error('Generation needs a prompt');
  const ctx = w.config.context, V = w.config.vocab.length, ids = prompt.slice(-ctx), out = [...prompt], steps: SampleStep[] = [];
  let cache: KVCache | undefined, logits: Float64Array;
  if (opts.temperature !== 0 && us.length < n) throw new Error('Need one uniform number per sampled token');
  ({ cache, logits } = prefill(w, ids));
  logits = logits.slice(-V);
  for (let i = 0; i < n; i++) {
    const model = softmax(logits), sampled = shape(logits, opts);
    const id = opts.temperature === 0 ? pickIndex(sampled, 0) : pickIndex(sampled, us[i]);
    steps.push({ id, model, sampled, logp: Math.log(model[id]) });
    out.push(id); ids.push(id);
    if (i === n - 1) break;
    if (cache && ids.length <= ctx) logits = decodeStep(w, cache, id);
    else { cache = undefined; ids.shift(); logits = forward(w, ids).logits.slice(-V); }
  }
  return { ids: out, steps };
}
