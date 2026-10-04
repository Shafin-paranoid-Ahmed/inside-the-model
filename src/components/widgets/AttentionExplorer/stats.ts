/** Grammatical-mass ablations and 200-sentence head statistics (gallery, heads and ablate figures). */
import { forward, type Trace, type Weights } from '../../../lib/transformer/model';
import { allowedNext, SAMPLE_200, VOCAB, WORD_CLASS } from '../../../lib/transformer/grammar';
import { HEADS } from './core';

/** Probability mass on grammatical next words after each prefix; NaN where the prefix leaves the grammar or ends the sentence. */
export function legalMass(trace: Trace, ids: readonly number[]): number[] {
  const V = trace.probs.length / ids.length;
  return ids.map((_, t) => {
    let legal: Set<number>;
    try { legal = allowedNext(ids.slice(0, t + 1)); } catch { return NaN; }
    if (!legal.size) return NaN;
    let m = 0; for (const id of legal) m += trace.probs[t * V + id];
    return m;
  });
}
const HEAD_INDEX = (toggles: readonly boolean[]) => HEADS.filter((_, k) => toggles[k]);
/** Mass per position with the given heads' outputs zeroed (toggles indexed like HEADS). */
export function ablatedMass(w: Weights, ids: readonly number[], toggles: readonly boolean[]): number[] {
  return legalMass(forward(w, ids, { ablate: HEAD_INDEX(toggles) }), ids);
}
/** [full, L1H1 off, L1H2 off, L2H1 off, L2H2 off] for these ids. */
export function ablationMasses(w: Weights, ids: readonly number[]): number[][] {
  return [[false, false, false, false], ...HEADS.map((_, k) => HEADS.map((_, j) => j === k))].map((tg) => ablatedMass(w, ids, tg));
}

/** Gives the main thread back every ~12 ms, so no task of the 200-sentence sweeps exceeds a frame or two. */
function pacer() {
  let last = performance.now();
  return async () => { if (performance.now() - last > 12) { await new Promise<void>((r) => setTimeout(r, 0)); last = performance.now(); } };
}
const sampleMasses = new WeakMap<Weights, Map<string, Promise<number>>>();
/** Token-weighted grammatical mass over the 200-sentence sample with the toggled heads (indexed like HEADS) switched off. */
export function sampleMass(w: Weights, toggles: readonly boolean[]): Promise<number> {
  const cache = sampleMasses.get(w) ?? new Map<string, Promise<number>>(); sampleMasses.set(w, cache);
  const key = toggles.map((x) => (x ? 1 : 0)).join(''), hit = cache.get(key); if (hit) return hit;
  const run = async () => {
    const pace = pacer(); let mass = 0, tokens = 0; const V = VOCAB.length, ablate = HEADS.filter((_, k) => toggles[k]);
    for (const sentence of SAMPLE_200) {
      const ids = sentence.slice(0, -1), f = forward(w, ids, { ablate });
      for (let t = 0; t < ids.length; t++) { for (const id of allowedNext(ids.slice(0, t + 1))) mass += f.probs[t * V + id]; tokens++; }
      await pace();
    }
    return mass / tokens;
  };
  const p = run(); cache.set(key, p); return p;
}

export interface HeadStat { layer: number; head: number; previous: number; determiner: number; verb: number; start: number }
const headStatCache = new WeakMap<Weights, Promise<HeadStat[]>>();
/**
 * Over the fixed 200-sentence sample (the final period dropped, as in the core's report): mean attention on the
 * previous word, on the determiner (token 1), on the verb (queries at or after it) and on [start].
 */
export function headStats(w: Weights): Promise<HeadStat[]> {
  const hit = headStatCache.get(w); if (hit) return hit;
  const run = async () => {
    const pace = pacer(), nh = HEADS.length, sums = Array.from({ length: nh }, () => [0, 0, 0, 0]), cnt = Array.from({ length: nh }, () => [0, 0, 0, 0]);
    for (const sentence of SAMPLE_200) {
      const ids = sentence.slice(0, -1), T = ids.length, f = forward(w, ids), verb = ids.findIndex((id) => WORD_CLASS[id].kind === 'verb');
      for (let t = 1; t < T; t++) HEADS.forEach(([l, h], k) => {
        const A = f.layers[l].weights[h];
        sums[k][0] += A[t * T + t - 1]; cnt[k][0]++;
        sums[k][1] += A[t * T + 1]; cnt[k][1]++;
        if (verb >= 0 && t >= verb) { sums[k][2] += A[t * T + verb]; cnt[k][2]++; }
        sums[k][3] += A[t * T]; cnt[k][3]++;
      });
      await pace();
    }
    return HEADS.map(([layer, head], k) => ({ layer, head, previous: sums[k][0] / cnt[k][0], determiner: sums[k][1] / cnt[k][1], verb: sums[k][2] / cnt[k][2], start: sums[k][3] / cnt[k][3] }));
  };
  const p = run(); headStatCache.set(w, p); return p;
}
/** Head statistics plus the grammatical mass of [full model, each head off], paced so the page stays responsive. */
export async function sampleStats(w: Weights, withHeads = true): Promise<{ heads: HeadStat[]; mass: number[] }> {
  const heads = withHeads ? await headStats(w) : [], mass: number[] = [];
  for (const tg of [[false, false, false, false], ...HEADS.map((_, k) => HEADS.map((_, j) => j === k))]) mass.push(await sampleMass(w, tg));
  return { heads, mass };
}
