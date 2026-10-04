// Pure logic for ICLLab: no DOM, no fetch. Weights and logs come in as arguments.
import { forward, type Weights } from '../../../lib/transformer/model';
import { createRng, type Rng } from '../../../lib/math/rng';

export type Mode = 'shots' | 'curve' | 'attention' | 'phase' | 'emergence' | 'format';
export const LETTERS = 'abcdefgh';
const SYMBOLS = [...LETTERS, '>', ' ', '\n'];
export const UNK = 11;
/** Example counts on the accuracy curve. */
export const COUNTS = [0, 1, 2, 3, 4, 6];
/** Shots demo: the hidden shift, the letter we ask about, and up to six examples (none uses the query as a source). */
export const DEMO = { shift: 3, query: 'g', pairs: ['c>f', 'a>d', 'e>h', 'h>c', 'b>e', 'd>g'] };
export const DEFAULT_PROMPT = 'c>f g>b a>d e>h g>';

export type Format = 'trained' | 'equals' | 'newline' | 'packed' | 'wrong';
export const FORMATS: { id: Format; label: string; note: string }[] = [
  { id: 'trained', label: 'Trained format', note: 'c>f a>d' },
  { id: 'equals', label: 'Equals sign', note: 'c=f a=d' },
  { id: 'newline', label: 'New lines', note: 'one pair per line' },
  { id: 'packed', label: 'No spaces', note: 'c>fa>d' },
  { id: 'wrong', label: 'Wrong example', note: 'last answer changed' },
];

export const shift = (c: string, s: number) => LETTERS[(LETTERS.indexOf(c) + s + 8) % 8];
/** Characters outside the model's 12-symbol vocabulary (such as "=") become the reserved <unk> id. */
export const encode = (text: string) => [...text].map((c) => { const i = SYMBOLS.indexOf(c); return i < 0 ? UNK : i; });

export interface Options { random?: boolean; format?: Format }
export interface Task { text: string; answer: number; shift: number }

/** k examples of a hidden shift, then a query. Same draw order as the training script's task generator. */
export function makeTask(rng: Rng, k: number, o: Options = {}): Task {
  const s = Math.floor(rng.next() * 8), letter = () => LETTERS[Math.floor(rng.next() * 8)], query = letter();
  const f = o.format ?? 'trained', sep = f === 'equals' ? '=' : '>', gap = f === 'newline' ? '\n' : f === 'packed' ? '' : ' ';
  let text = '';
  for (let i = 0; i < k; i++) {
    const src = letter();
    const dst = o.random ? letter() : f === 'wrong' && i === k - 1 ? shift(src, (s + 1 + Math.floor(rng.next() * 7)) % 8) : shift(src, s);
    text += `${src}${sep}${dst}${gap}`;
  }
  return { text: `${text}${query}${sep}`, answer: LETTERS.indexOf(shift(query, s)), shift: s };
}

export interface Prediction { ids: number[]; probs: number[]; best: number; weights: number[][] }
/** Next-symbol probabilities after `text`, plus every head's attention row from the last position (index = layer * heads + head). */
export function predict(w: Weights, text: string, ablate: readonly (readonly [number, number])[] = []): Prediction {
  const ids = encode(text), trace = forward(w, ids, ablate.length ? { ablate } : {}), T = ids.length, V = w.config.vocab.length;
  const probs = Array.from(trace.probs.subarray((T - 1) * V, T * V));
  const weights = trace.layers.flatMap((l) => l.weights.map((a) => Array.from(a.subarray((T - 1) * T, T * T))));
  return { ids, probs, best: probs.indexOf(Math.max(...probs)), weights };
}

/** Share of n seeded tasks (the training script's seed, so the numbers match its log) the model answers correctly. */
export function accuracy(w: Weights, k: number, o: Options = {}, n = 500, ablate: readonly (readonly [number, number])[] = []): number {
  const rng = createRng(4242 + 97 * k);
  let hits = 0;
  for (let i = 0; i < n; i++) { const q = makeTask(rng, k, o); hits += +(predict(w, q.text, ablate).best === q.answer); }
  return hits / n;
}


// ---- measurements shipped as JSON (public/models/icl-log.json, icl-sweep.json) ----
export interface LogRow { step: number; loss0: number; loss4: number; acc0: number; acc4: number; profile: { query: number } }
export interface TrainingLog {
  defaultPrompt: string;
  shownHead: { layer: number; head: number };
  log: LogRow[];
  thumbnails: { step: number; weights: number[] }[];
}
export interface SweepRow { d: number; N: number; exact: number; perLetter: number; meanLogp: number; perLetterPow4: number }

// ---- the timeline: one continuous time t, each figure divides it into beats ----
export const DURATION = 14;
export interface Frame {
  /** Which beat (0 .. n-1) t is in, and how far through it (0..1). */
  index: number; local: number;
  /** 0 → 1 over the first part of a beat: bars and dots glide to the beat's values, then hold. */
  settle: number;
}
const smooth = (x: number) => { const c = Math.min(1, Math.max(0, x)); return c * c * (3 - 2 * c); };
export function frameAt(t: number, n: number, duration = DURATION): Frame {
  const p = Math.min(0.999999, Math.max(0, Number.isFinite(t) ? t / duration : 0)) * n;
  const index = Math.floor(p), local = p - index;
  return { index, local, settle: smooth(local / 0.4) };
}
/** Segment markers: beat i starts at i * duration / n. */
export const beatMarkers = (labels: string[], duration = DURATION) => labels.map((label, i) => ({ label, at: (i * duration) / labels.length }));

// ---- shots: the same hidden rule, more and more examples ----
export const PAIR = /^[a-h]>[a-h]$/;
/** Up to six "x>y" pairs typed by the learner, or null when the text is not that. */
export function parsePairs(text: string): string[] | null {
  const parts = text.trim().split(/\s+/).filter(Boolean);
  return parts.length >= 1 && parts.length <= 6 && parts.every((p) => PAIR.test(p)) ? parts : null;
}
export interface ShotRow { count: number; text: string; probs: number[]; best: number; implied: number }
/** One row per prefix of the examples (0 .. n examples); `implied` = letter the first example's shift gives for the query. */
export function shotRows(w: Weights, pairs: string[], query = DEMO.query): ShotRow[] {
  const s = pairs.length ? (LETTERS.indexOf(pairs[0][2]) - LETTERS.indexOf(pairs[0][0]) + 8) % 8 : -1;
  return Array.from({ length: pairs.length + 1 }, (_, count) => {
    const text = pairs.slice(0, count).map((p) => `${p} `).join('') + `${query}>`, p = predict(w, text);
    return { count, text, probs: p.probs.slice(0, 8), best: p.best, implied: s < 0 ? -1 : LETTERS.indexOf(shift(query, s)) };
  });
}

// ---- attention ----
export const HEADS = [[0, 0], [0, 1], [1, 0], [1, 1]] as const;
export const headName = (h: number) => `Layer ${HEADS[h][0] + 1}, head ${HEADS[h][1] + 1}`;
/** Layer 1 appears over the first 2 s of the first half, layer 2 over the first 2 s of the second; both then hold. */
export function attentionFrame(t: number) {
  const half = DURATION / 2;
  return { layer1: smooth(t / 2), layer2: smooth((t - half) / 2), layer: t < half ? 1 : 2 };
}
/** The positions a head puts at least `min` of its attention on, largest first. */
export const topPositions = (row: number[], min = 0.03) =>
  row.map((weight, position) => ({ weight, position })).filter((r) => r.weight >= min).sort((a, b) => b.weight - a.weight);
export interface Ablations { probOn: number; probOff: number[]; accuracyOff: number[] }
/** p(correct) on the default prompt, and 500-task 4-example accuracy, with each head switched off in turn. */
export function ablations(w: Weights, withAccuracy = true): Ablations {
  const answer = LETTERS.indexOf(shift('g', 3)), p = (a: readonly (readonly [number, number])[]) => predict(w, DEFAULT_PROMPT, a).probs[answer];
  return {
    probOn: p([]),
    probOff: HEADS.map((h) => p([h])),
    accuracyOff: HEADS.map((h) => (withAccuracy ? accuracy(w, 4, {}, 500, [h]) : NaN)),
  };
}

// ---- training log ----
export const PHASE_STEPS = 2000;
export const stepAt = (t: number) => Math.min(PHASE_STEPS, Math.max(0, (t / DURATION) * PHASE_STEPS));
export const timeOfStep = (step: number) => (step / PHASE_STEPS) * DURATION;
/** Index of the last logged row at or before `step` (the readout shows this real row). */
export const rowAt = (log: { step: number }[], step: number) => { let i = 0; while (i + 1 < log.length && log[i + 1].step <= step + 1e-9) i++; return i; };
export function phaseFrame(t: number, log: TrainingLog) {
  const step = stepAt(t), row = rowAt(log.log, step), cp = rowAt(log.thumbnails, log.log[row].step);
  return { ...log.log[row], step, row, thumb: cp };
}
/** Weights of the checkpoint's lower-triangle attention (row r has r+1 entries) as a square matrix in 0..1. */
export function triangle(weights: number[], n = 18): number[][] {
  const rows: number[][] = []; let k = 0;
  for (let r = 0; r < n; r++) { rows.push(Array.from({ length: n }, (_, c) => (c <= r ? weights[k++] / 100 : 0))); }
  return rows;
}

// ---- sizes ----
export type Metric = 'exact' | 'perLetter' | 'meanLogp';
export const METRICS: { id: Metric; label: string }[] = [
  { id: 'exact', label: 'All four letters right' },
  { id: 'perLetter', label: 'Each letter right' },
  { id: 'meanLogp', label: 'Log-probability of the answer' },
];
/** Chance level of each metric for 8 letters and a 4-letter answer. */
export const CHANCE: Record<Metric, number> = { exact: 8 ** -4, perLetter: 1 / 8, meanLogp: -4 * Math.log(8) };
/** Position of a metric value on a 0..1 axis (log-probability runs from -9 to 0). */
export const norm = (m: Metric, v: number) => (m === 'meanLogp' ? Math.min(1, Math.max(0, (v + 9) / 9)) : Math.min(1, Math.max(0, v)));
export const emergenceFrame = (t: number, n: number) => { const f = frameAt(t, n); return { shown: f.index + 1, focus: f.index, ...f }; };
