/**
 * Sampling lab: how the next-character distribution of the `house` model becomes text.
 * Pure logic, no DOM. Every frame is a pure function of t over numbers computed once from the model
 * (the model itself is run through `src/lib/lm/sample.ts` and the transformer core). Text is whole
 * characters and is never blended; only bar geometry eases.
 */
import { createRng } from '../../../lib/math/rng';
import type { KVCache, Weights } from '../../../lib/transformer/model';
import { decodeStep, forward, prefill, softmax } from '../../../lib/transformer/model';
import { exp } from '../../../lib/transformer/math';
import { CHARSET, UNKNOWN, encode } from '../../../lib/lm/charset';
import { entropy, generate, pickIndex, shape, softmaxT, topK, topP, type SampleOptions, type SampleStep } from '../../../lib/lm/sample';

export type Mode = 'logits' | 'greedy' | 'temperature' | 'topk' | 'topp' | 'lab';
export interface Marker { at: number; label: string }

/** Held-out Alice text (validation split). MAIN ends inside a word, SURE after "q", UNSURE after "the". */
export const PROMPTS = {
  main: ' a little girl," said alice, rather doubtfully, as she r',
  sure: ' say."\nthis was such a new idea to alice, that she was q',
  unsure: ' likely story indeed!" said the pigeon in a tone of the ',
};
export const SEED = 1;
export const GREEDY_N = 80, TEXT_N = 60, LAB_N = 120;
export const BEAT = 0.35; // greedy: seconds per character
export const LAB_BEAT = 0.15;
export const TEMPS = [0.2, 0.5, 0.8, 1, 1.3, 2];
export const KSEQ = [38, 20, 10, 5, 3, 2, 1, 2, 3, 5, 10, 20];
export const K_BEAT = 1.6, K_TWEEN = 0.5;
export const P_DEFAULT = 0.9;
export const LAB_DEFAULTS = { temperature: 0.8, topK: 38, topP: 0.95, seed: SEED };

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
export const ease = (x: number) => { const u = clamp01(x); return u * u * (3 - 2 * u); };
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;

/** How a symbol is drawn: space as a middle dot in bar labels, newline as an arrow, the unknown id as ~. */
export const glyph = (id: number) => (id === 26 ? '·' : id === 27 ? '↵' : id >= UNKNOWN ? '~' : CHARSET[id]);
/** In running text a space stays a space; only the newline is made visible. */
export const textOf = (ids: readonly number[]) => ids.map((i) => (i === 27 ? '↵' : i >= UNKNOWN ? '~' : CHARSET[i])).join('');
export const bits = (p: ArrayLike<number>) => entropy(p) / Math.LN2;
export const pct = (v: number, dp = 1) => (100 * v).toFixed(v >= 0.9995 ? 0 : dp) + '%';
export function fmt(v: number): string {
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toString().replace(/\B(?=(\d{3})+$)/g, ',');
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}
export const uniforms = (seed: number, n: number) => { const r = createRng(seed); return Array.from({ length: n }, () => r.next()); };

// ------------------------------------------------------------------ one next-character distribution
export interface Next { text: string; ids: number[]; logits: Float64Array; probs: Float64Array; order: number[]; V: number }
/** The model's logits and probabilities for the character after `text` (the last `context` characters are used). */
export function nextOf(w: Weights, text: string): Next {
  const V = w.config.vocab.length, ids = encode(text).slice(-w.config.context);
  const t = forward(w, ids), logits = t.logits.slice(-V), probs = t.probs.slice(-V);
  const order = Array.from(logits, (_, i) => i).sort((a, b) => logits[b] - logits[a] || a - b);
  return { text, ids, logits, probs, order, V };
}

// ------------------------------------------------------------------ repetition
/** Where the text first copies itself: the earliest block of `min` characters that already appeared earlier. */
export function findRepeat(text: string, min = 12): { start: number; from: number; seenAt: number } | null {
  for (let i = 1; i + min <= text.length; i++) {
    const j = text.indexOf(text.slice(i, i + min));
    if (j >= 0 && j < i) return { start: i, from: j, seenAt: i + min };
  }
  return null;
}
/**
 * Generation one character at a time, with exactly the arithmetic of `generate` in lm/sample.ts (a test checks it).
 * Past the context window every step re-reads the last 64 characters, about 6 ms each, so long texts are made
 * a few characters at a time in a worker.
 */
export function stepper(w: Weights, prompt: readonly number[], opts: SampleOptions) {
  const ctx = w.config.context, V = w.config.vocab.length, ids = prompt.slice(-ctx);
  const first = prefill(w, ids);
  let cache: KVCache | undefined = first.cache, logits: Float64Array = first.logits.slice(-V), pending = -1;
  return {
    next(u: number): SampleStep {
      if (pending >= 0) {
        if (cache && ids.length <= ctx) logits = decodeStep(w, cache, pending);
        else { cache = undefined; ids.shift(); logits = forward(w, ids).logits.slice(-V); }
      }
      const model = softmax(logits), sampled = shape(logits, opts);
      const id = opts.temperature === 0 ? pickIndex(sampled, 0) : pickIndex(sampled, u);
      ids.push(id); pending = id;
      return { id, model, sampled, logp: Math.log(model[id]) };
    },
  };
}
export const meanLogp = (steps: readonly SampleStep[], n: number) => (n ? steps.slice(0, n).reduce((s, x) => s + x.logp, 0) / n : 0);

// ------------------------------------------------------------------ 1. logits -> probabilities
export const LOGITS_T = 9;
export const LOGITS_MARKERS: Marker[] = [{ at: 0, label: 'Scores' }, { at: 3, label: 'Raise e to each' }, { at: 6, label: 'Divide by the total' }];
export const SHIFT_MAX = 4;
const LO = -11, HI = 13; // fixed score axis, so a shift of up to +-4 stays on the plot
export interface LogitsFrame {
  /** 0 scores, 1 exponentiated, 2 probabilities; `morph` is true while bars are moving to the next phase. */
  phase: 0 | 1 | 2; morph: boolean;
  base: number; ends: Float64Array; mix: number; // positions are fractions of plot height from the top
  label: 0 | 1 | 2; labelAlpha: number; vals: Float64Array;
  divisor: number; sum: number; top: number; // top: value at the top of the plot (phases 1 and 2)
  zero: number; // y of the zero line in phase 0 (fraction from the top)
  status: string;
}
export function logitsFrame(t: number, nx: Next, shift = 0): LogitsFrame {
  const { V, order } = nx, tt = Math.min(Math.max(t, 0), LOGITS_T);
  const z = Float64Array.from(order, (i) => nx.logits[i] + shift);
  const zmax = z[0], rel = Float64Array.from(z, (v) => exp(v - zmax)); // e^z relative to the largest
  const ex = Float64Array.from(z, (v) => exp(v)), S = ex.reduce((a, b) => a + b, 0);
  const p = Float64Array.from(order, (i) => nx.probs[i]);
  const m1 = ease((tt - 3) / 2), m2 = ease((tt - 6) / 2);
  const zero = (HI) / (HI - LO), yA = Float64Array.from(z, (v) => (HI - v) / (HI - LO));
  const yB = Float64Array.from(rel, (r) => 1 - r);
  const ends = new Float64Array(V), vals = new Float64Array(V);
  let base: number, mix = 0, label: 0 | 1 | 2 = 0, labelAlpha = 1, divisor = 1, top = ex[0], morph = false, phase: 0 | 1 | 2 = 0;
  if (tt < 3) { for (let i = 0; i < V; i++) { ends[i] = yA[i]; vals[i] = z[i]; } base = zero; }
  else if (tt < 5) {
    phase = 1; morph = tt < 5 && m1 < 1;
    for (let i = 0; i < V; i++) ends[i] = lerp(yA[i], yB[i], m1);
    base = lerp(zero, 1, m1); mix = m1; label = m1 < 0.5 ? 0 : 1; labelAlpha = (2 * m1 - 1) ** 2;
    for (let i = 0; i < V; i++) vals[i] = m1 < 0.5 ? z[i] : ex[i];
  } else if (tt < 6) { phase = 1; base = 1; mix = 1; label = 1; for (let i = 0; i < V; i++) { ends[i] = yB[i]; vals[i] = ex[i]; } }
  else if (tt < 8) {
    phase = 2; morph = m2 < 1; base = 1; mix = 1; label = m2 < 0.5 ? 1 : 2; labelAlpha = (2 * m2 - 1) ** 2;
    divisor = lerp(1, S, m2); const ratio = lerp(1, p[0], m2); top = ex[0] / divisor / ratio; // the tallest bar goes from full height to its probability
    for (let i = 0; i < V; i++) { ends[i] = 1 - (ex[i] / ex[0]) * ratio; vals[i] = m2 < 0.5 ? ex[i] : p[i]; }
  } else { phase = 2; base = 1; mix = 1; label = 2; divisor = S; top = 1; for (let i = 0; i < V; i++) { ends[i] = 1 - p[i]; vals[i] = p[i]; } }
  const sum = phase === 2 ? S / divisor : phase === 1 && tt >= 5 ? S : 0;
  const c = glyph(order[0]);
  const status = phase === 0 ? `Scores for the next character. "${c}" has the highest score, ${z[0].toFixed(2)}; scores can be negative.`
    : phase === 1 ? (tt < 5 ? 'Raising e to each score: negative scores become small positive numbers, big scores become huge ones.' : `Every score is now positive. "${c}" is ${fmt(ex[0])} against a total of ${fmt(S)}.`)
    : tt < 8 ? `Dividing every bar by the total, ${fmt(S)}, so that the bars add up to 1. So far: divided by ${fmt(divisor)}.` : `Probabilities, adding up to 1. "${c}" gets ${pct(p[0])}.`;
  return { phase, morph, base, ends, mix, label, labelAlpha, vals, divisor, sum, top, zero, status };
}
/** Facts for prose and the readout: top score, its probability, and how many characters are above 1%. */
export const logitsFacts = (nx: Next) => ({ topScore: nx.logits[nx.order[0]], topP: nx.probs[nx.order[0]], above1: Array.from(nx.probs).filter((v) => v > 0.01).length });

// ------------------------------------------------------------------ 2. greedy versus sampled
export interface Run { ids: number[]; steps: SampleStep[]; text: string }
export interface GreedyData { prompt: string; greedy: Run; sampled: Run; us: number[]; loop: ReturnType<typeof findRepeat>; seed: number }
export function makeGreedy(w: Weights, seed = SEED, prompt = PROMPTS.main, greedyRun?: Run): GreedyData {
  const ids = encode(prompt), us = uniforms(seed, GREEDY_N);
  const run = (opts: { temperature: number }, u: number[]): Run => { const g = generate(w, ids, GREEDY_N, opts, u); return { ids: g.ids.slice(ids.length), steps: g.steps, text: textOf(g.ids.slice(ids.length)) }; };
  const greedy = greedyRun ?? run({ temperature: 0 }, []), sampled = run({ temperature: 1 }, us);
  return { prompt, greedy, sampled, us, loop: findRepeat(greedy.text), seed };
}
export const greedyDuration = () => GREEDY_N * BEAT;
export const greedyMarkers = (d: GreedyData): Marker[] => d.loop ? [{ at: 0, label: 'Writing' }, { at: d.loop.seenAt * BEAT, label: 'Greedy repeats' }] : [{ at: 0, label: 'Writing' }];
export interface GreedyFrame {
  i: number; sub: number; shown: number; u: number; uPos: number; landed: boolean; strip: Float64Array; chosen: number;
  greedyText: string; sampledText: string; greedyMean: number; sampledMean: number; loopShown: boolean; status: string;
}
export function greedyFrame(t: number, d: GreedyData): GreedyFrame {
  const x = Math.min(Math.max(t / BEAT, 0), GREEDY_N - 1e-9), i = Math.floor(x), sub = x - i;
  const shown = Math.min(GREEDY_N, Math.floor(x + 0.4)); // a character appears 60% of the way through its beat
  const step = d.sampled.steps[i], u = d.us[i], landed = sub >= 0.5, uPos = ease(sub / 0.5) * u;
  const loopShown = !!d.loop && shown >= d.loop.seenAt;
  const gt = d.greedy.text.slice(0, shown), st = d.sampled.text.slice(0, shown);
  const status = `Character ${i + 1}: the sampled lane draws u = ${u.toFixed(3)}${landed ? `, which lands in "${glyph(step.id)}" (${pct(step.model[step.id])})` : ''}. The greedy lane takes its most likely character, "${glyph(d.greedy.steps[i].id)}" (${pct(d.greedy.steps[i].model[d.greedy.steps[i].id])}).`;
  return { i, sub, shown, u, uPos, landed, strip: step.model, chosen: step.id, greedyText: gt, sampledText: st,
    greedyMean: meanLogp(d.greedy.steps, shown), sampledMean: meanLogp(d.sampled.steps, shown), loopShown, status };
}

// ------------------------------------------------------------------ 3. temperature
/** texts[i] and means[i] belong to TEMPS[i]; they are filled in as the worker finishes each one. */
export interface TempData { nx: Next; texts: (string | undefined)[]; means: number[]; seed: number }
export function tempRun(w: Weights, T: number, seed = SEED, prompt = PROMPTS.main): { text: string; mean: number } {
  const ids = encode(prompt), g = generate(w, ids, TEXT_N, { temperature: T }, uniforms(seed, TEXT_N));
  return { text: textOf(g.ids.slice(ids.length)), mean: meanLogp(g.steps, TEXT_N) };
}
export const emptyTemp = (nx: Next, seed = SEED): TempData => ({ nx, seed, texts: TEMPS.map(() => undefined), means: TEMPS.map(() => NaN) });
export function makeTemp(w: Weights, seed = SEED, prompt = PROMPTS.main): TempData {
  const d = emptyTemp(nextOf(w, prompt), seed);
  TEMPS.forEach((T, i) => { const r = tempRun(w, T, seed, prompt); d.texts[i] = r.text; d.means[i] = r.mean; });
  return d;
}
export const TEMP_T = 16;
export const TEMP_MARKERS: Marker[] = [{ at: 0, label: 'Temperature rising' }, { at: 8, label: 'Temperature falling' }];
/** Temperature at time t: 0.2 to 2 on a log scale in 8 s, then back. */
export const tempAt = (t: number) => 0.2 * 10 ** (t < 8 ? t / 8 : (16 - Math.min(t, 16)) / 8);
export const tIndexFor = (T: number) => { let b = 0; for (let i = 1; i < TEMPS.length; i++) if (Math.abs(Math.log(T / TEMPS[i])) < Math.abs(Math.log(T / TEMPS[b]))) b = i; return b; };
export interface TempFrame { T: number; probs: Float64Array; entropy: number; topP: number; topChar: string; snap: number; text: string; mean: number; status: string }
export function tempFrame(t: number, d: TempData, override?: number): TempFrame {
  const T = override ?? tempAt(t), p = softmaxT(d.nx.logits, T), probs = Float64Array.from(d.nx.order, (i) => p[i]);
  const snap = tIndexFor(T), H = bits(p);
  return { T, probs, entropy: H, topP: probs[0], topChar: glyph(d.nx.order[0]), snap, text: d.texts[snap] ?? '', mean: d.means[snap],
    status: `Temperature ${T.toFixed(2)}: "${glyph(d.nx.order[0])}" has probability ${pct(probs[0])}, and the entropy is ${H.toFixed(2)} bits.` };
}

// ------------------------------------------------------------------ 4 and 5. top-k and top-p
export interface PairData { sure: Next; unsure: Next }
export const makePair = (w: Weights): PairData => ({ sure: nextOf(w, PROMPTS.sure), unsure: nextOf(w, PROMPTS.unsure) });
export const kAt = (t: number) => { const x = ((t % (KSEQ.length * K_BEAT)) + KSEQ.length * K_BEAT) % (KSEQ.length * K_BEAT), i = Math.floor(x / K_BEAT); return { i, k: KSEQ[i], prev: KSEQ[(i + KSEQ.length - 1) % KSEQ.length], tween: ease((x - i * K_BEAT) / K_TWEEN) }; };
export const K_T = KSEQ.length * K_BEAT;
export const K_REST = 3 * K_BEAT + K_TWEEN; // k = 5, arrived
export const K_MARKERS: Marker[] = [{ at: 0, label: 'Shrinking k' }, { at: 7 * K_BEAT, label: 'Growing k' }];
export interface Panel {
  chars: string[]; probs: Float64Array; kept: Float64Array; // probs: original, by rank; kept: renormalised probabilities of the kept ones (0 elsewhere)
  keep: number; removed: number; // number of kept characters and mass removed (displayed k)
  cum: Float64Array; max: number;
}
const ranked = (nx: Next) => nx.order;
export function topkPanel(nx: Next, k: number, tween?: { prev: number; u: number }): Panel {
  const probs = Float64Array.from(ranked(nx), (i) => nx.probs[i]);
  const shapeAt = (kk: number) => { const kp = topK(nx.probs, kk); return Float64Array.from(ranked(nx), (i) => kp.probs[i]); };
  const to = shapeAt(k), from = tween ? shapeAt(tween.prev) : to;
  const kept = Float64Array.from(to, (v, i) => lerp(from[i], v, tween ? tween.u : 1));
  const shownK = tween && tween.u < 0.5 ? tween.prev : k;
  let massKept = 0; for (let i = 0; i < shownK; i++) massKept += probs[i];
  return { chars: ranked(nx).map(glyph), probs, kept, keep: shownK, removed: 1 - massKept, cum: cumulative(probs), max: Math.max(...kept, probs[0]) };
}
const cumulative = (p: ArrayLike<number>) => { const c = new Float64Array(p.length); let s = 0; for (let i = 0; i < p.length; i++) { s += p[i]; c[i] = s; } return c; };
export interface KFrame { k: number; sure: Panel; unsure: Panel; status: string }
export function topkFrame(t: number, d: PairData, manual?: number): KFrame {
  const a = manual === undefined ? kAt(t) : { k: manual, prev: manual, tween: 1, i: 0 };
  const tw = a.tween < 1 ? { prev: a.prev, u: a.tween } : undefined;
  const sure = topkPanel(d.sure, a.k, tw), unsure = topkPanel(d.unsure, a.k, tw), k = sure.keep;
  return { k, sure, unsure, status: `Keeping the ${k === 1 ? 'single most likely character' : `${k} most likely characters`} throws away ${pct(sure.removed)} of the probability after "q" and ${pct(unsure.removed)} after "the ".` };
}
export const P_T = 16;
export const P_DRAW = 4; // seconds for the curve to be drawn
export const P_REST = P_DRAW + 1;
export const P_MARKERS: Marker[] = [{ at: 0, label: 'Add up and cut' }, { at: 5, label: 'Lower p' }, { at: 8, label: 'Raise p' }, { at: 12.5, label: 'Back to 0.9' }];
export const pAt = (t: number) => t < 5 ? P_DEFAULT : t < 8 ? lerp(P_DEFAULT, 0.5, ease((t - 5) / 3)) : t < 12.5 ? lerp(0.5, 0.99, ease((t - 8) / 4.5)) : lerp(0.99, P_DEFAULT, ease((t - 12.5) / 3.5));
export interface PPanel { chars: string[]; probs: Float64Array; cum: Float64Array; keep: number; head: number; state: Uint8Array } // state: 0 pending, 1 kept, 2 removed
export function toppPanel(nx: Next, p: number, head: number): PPanel {
  const probs = Float64Array.from(ranked(nx), (i) => nx.probs[i]), keep = topP(nx.probs, p).ids.length;
  const state = Uint8Array.from(probs, (_, r) => (r >= head ? 0 : r < keep ? 1 : 2));
  return { chars: ranked(nx).map(glyph), probs, cum: cumulative(probs), keep, head, state };
}
export interface PFrame { p: number; sure: PPanel; unsure: PPanel; status: string }
export function toppFrame(t: number, d: PairData, manual?: number): PFrame {
  const p = manual ?? pAt(t), head = manual !== undefined || t >= P_DRAW ? d.sure.V : (Math.min(t, P_DRAW) / P_DRAW) * d.sure.V;
  const sure = toppPanel(d.sure, p, head), unsure = toppPanel(d.unsure, p, head);
  return { p, sure, unsure, status: `With p = ${p.toFixed(2)}, the nucleus is ${sure.keep} character${sure.keep === 1 ? '' : 's'} after "q" and ${unsure.keep} after "the ".` };
}

// ------------------------------------------------------------------ 6. the lab
export interface LabSettings { prompt: string; temperature: number; topK: number; topP: number; seed: number }
/** steps grows as the worker produces characters; at most LAB_N. */
export interface LabData { settings: LabSettings; steps: SampleStep[] }
export const labOptions = (s: LabSettings, V: number): SampleOptions => ({ temperature: s.temperature, topK: s.topK < V ? s.topK : undefined, topP: s.topP < 1 ? s.topP : undefined });
export const cleanPrompt = (s: string) => Array.from(s.toLowerCase()).filter((c) => CHARSET.includes(c)).join('').slice(-200);
export const labPrompt = (s: LabSettings) => cleanPrompt(s.prompt) || ' ';
export const cleanPromptSettings = (s: LabSettings): LabSettings => ({ ...s, prompt: labPrompt(s) });
export function makeLab(w: Weights, s: LabSettings): LabData {
  const st = stepper(w, encode(labPrompt(s)), labOptions(s, w.config.vocab.length)), us = uniforms(s.seed, LAB_N);
  return { settings: cleanPromptSettings(s), steps: us.map((u) => st.next(u)) };
}
export const labText = (d: LabData, n = d.steps.length) => textOf(d.steps.slice(0, n).map((s) => s.id));
export const labDuration = () => LAB_N * LAB_BEAT;
export interface LabFrame { shown: number; cur: number; strip: Float64Array | null; chosen: number; mean: number; repeat: ReturnType<typeof findRepeat>; status: string }
export function labFrame(t: number, d: LabData): LabFrame {
  const shown = Math.min(d.steps.length, Math.max(0, Math.floor(t / LAB_BEAT))), cur = Math.max(0, shown - 1), st = d.steps[cur];
  const repeat = findRepeat(labText(d, shown));
  return { shown, cur, strip: st ? st.sampled : null, chosen: shown ? st.id : -1, mean: meanLogp(d.steps, shown), repeat,
    status: shown ? `${shown} characters written. Average log-probability ${meanLogp(d.steps, shown).toFixed(2)} per character.${repeat ? ` Text from character ${repeat.from + 1} is copied at character ${repeat.start + 1}.` : ''}` : 'Ready to write.' };
}
export const LAB_MARKERS: Marker[] = [{ at: 0, label: 'Writing' }];

// ------------------------------------------------------------------ shared timing table
export const durationOf = (mode: Mode) => ({ logits: LOGITS_T, greedy: greedyDuration(), temperature: TEMP_T, topk: K_T, topp: P_T, lab: labDuration() })[mode];
export const restTime = (mode: Mode) => ({ logits: 8.5, greedy: greedyDuration(), temperature: 8 * Math.log10(5), topk: K_REST, topp: P_REST, lab: labDuration() })[mode];
export const markersOf = (mode: Mode, greedy?: GreedyData): Marker[] =>
  mode === 'logits' ? LOGITS_MARKERS : mode === 'greedy' ? (greedy ? greedyMarkers(greedy) : [{ at: 0, label: 'Writing' }]) : mode === 'temperature' ? TEMP_MARKERS : mode === 'topk' ? K_MARKERS : mode === 'topp' ? P_MARKERS : LAB_MARKERS;

// softmax is re-exported so tests can compare with the core without a second import path
export { softmax };
