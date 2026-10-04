// PositionLab: how a transformer learns word order. Pure math, no DOM. Every visual is frameAt(t, ...).
import { createRng } from '../../../lib/math/rng';
import { allowedNext, encodeWords, SAMPLE_200, VOCAB, WORD_CLASS } from '../../../lib/transformer/grammar';
import { cosine, forward, rotate, type Weights } from '../../../lib/transformer/model';

export type Mode = 'shuffle' | 'ablate' | 'sinusoidal' | 'learned' | 'rope' | 'relative';
export interface Marker { at: number; label: string }

export const SENTENCE = 'these small sheep chase a red fox';
export const IDS: readonly number[] = encodeWords(`[start] ${SENTENCE}`);
export const WORDS: readonly string[] = IDS.map((id) => VOCAB[id]);
export const CLASS_NAMES = ['start', 'determiner', 'adjective', 'noun', 'verb'] as const;
export const classOf = (id: number): number => {
  const kind = WORD_CLASS[id].kind;
  return kind === 'special' ? 0 : kind === 'det' ? 1 : kind === 'adj' ? 2 : kind === 'noun' ? 3 : 4;
};
export const WORD_CLASSES: readonly number[] = IDS.map(classOf);

export const D = 16;
export const ease = (x: number): number => { const c = Math.min(1, Math.max(0, x)); return c * c * (3 - 2 * c); };
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;

/** Walk through `n` integer stops, `per` seconds each: continuous position p (eased between stops), shown stop = round(p). */
export function glide(t: number, n: number, per: number): { p: number; index: number } {
  const x = Math.max(0, t) / per, i = Math.floor(x);
  const p = i >= n - 1 ? n - 1 : i + ease(x - i);
  return { p, index: Math.round(p) };
}
const glideMarkers = (n: number, per: number, every: number, label: (a: number, b: number) => string): Marker[] => {
  const out: Marker[] = [];
  for (let a = 0; a < n; a += every) out.push({ at: a === 0 ? 0 : (a - 0.5) * per, label: label(a, Math.min(n - 1, a + every - 1)) });
  return out;
};

// ---- timelines ---------------------------------------------------------------------------------------------
export const SIN_POSITIONS = 32, SIN_PER = 0.4;
export const ROPE_STOPS = 16, ROPE_PER = 0.6, ROPE_FIRST = 2, ROPE_GAP = 2, ROPE_THETA = 0.5;
export const REL_N = 16;
const LEARNED_ROWS = 10;
/** Position 9 is never an input while TinyLM trains (the longest sentence ends there), so its vector is still the random start. */
export const UNTRAINED_ROW = 9;
const SHUFFLE_MARKS: Marker[] = [{ at: 0, label: 'start' }, { at: 2, label: 'shuffle' }, { at: 4.5, label: 'shuffled' }, { at: 7.5, label: 'put back' }];
const ABLATE_MARKS: Marker[] = [{ at: 0, label: 'full size' }, { at: 1.5, label: 'fading' }, { at: 6, label: 'removed' }];
/** Diagonals of the relative-score grid, visited from the lower-left corner; the main one lingers. */
export const DIAGONALS: number[] = Array.from({ length: 2 * REL_N - 1 }, (_, i) => i - (REL_N - 1));
const diagDur = (c: number) => (c === 0 ? 1.4 : 0.35);
export const DIAG_STARTS: number[] = (() => { let s = 0; return DIAGONALS.map((c) => { const a = s; s += diagDur(c); return a; }); })();

export const durationOf = (mode: Mode): number => ({
  shuffle: 10, ablate: 8.5, sinusoidal: SIN_POSITIONS * SIN_PER, learned: LEARNED_ROWS + 1, rope: ROPE_STOPS * ROPE_PER,
  relative: DIAG_STARTS[DIAG_STARTS.length - 1] + diagDur(DIAGONALS[DIAGONALS.length - 1]),
})[mode];
export function markersOf(mode: Mode): Marker[] {
  switch (mode) {
    case 'shuffle': return SHUFFLE_MARKS;
    case 'ablate': return ABLATE_MARKS;
    case 'sinusoidal': return glideMarkers(SIN_POSITIONS, SIN_PER, 8, (a, b) => `${a} to ${b}`);
    case 'learned': return [...Array.from({ length: LEARNED_ROWS }, (_, r) => ({ at: r, label: String(r) })), { at: LEARNED_ROWS, label: 'all' }];
    case 'rope': return glideMarkers(ROPE_STOPS, ROPE_PER, 4, (a, b) => `${ROPE_FIRST + a} to ${ROPE_FIRST + b}`);
    case 'relative': return [
      { at: 0, label: 'key earlier' },
      { at: DIAG_STARTS[REL_N - 1], label: 'same' },
      { at: DIAG_STARTS[REL_N], label: 'key later' },
    ];
  }
}

// ---- shuffle -----------------------------------------------------------------------------------------------
export interface ShuffleData {
  perm: number[]; // perm[slot] = original index of the word placed in that slot
  slotOf: number[];
  original: Float64Array[]; // layer-1 attention output per original word, original order
  shuffled: Float64Array[]; // the same words, read from the pass that saw them shuffled
  maxDiff: number;
  scale: number;
}
export function permutation(seed: number, n = IDS.length): number[] {
  const rng = createRng(1000 + seed);
  for (;;) {
    const p = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
    if (p.filter((v, i) => v === i).length <= 1) return p;
  }
}
const rows = (a: Float64Array, width: number): Float64Array[] => Array.from({ length: a.length / width }, (_, i) => a.slice(i * width, (i + 1) * width));
export function makeShuffle(w: Weights, seed: number): ShuffleData {
  const perm = permutation(seed), slotOf: number[] = [];
  perm.forEach((orig, slot) => { slotOf[orig] = slot; });
  // No position vectors and no causal mask: the two things that would let attention tell the orders apart.
  const opts = { noPositions: true, causal: false };
  const a = rows(forward(w, [...IDS], opts).layers[0].attnOut, w.config.d);
  const b = rows(forward(w, perm.map((i) => IDS[i]), opts).layers[0].attnOut, w.config.d);
  let maxDiff = 0, scale = 1e-9;
  const shuffled = a.map((_, orig) => b[slotOf[orig]]);
  a.forEach((row, i) => row.forEach((v, j) => { maxDiff = Math.max(maxDiff, Math.abs(v - shuffled[i][j])); scale = Math.max(scale, Math.abs(v)); }));
  return { perm, slotOf, original: a, shuffled, maxDiff, scale };
}
export interface ShuffleFrame {
  phase: string; status: string; u: number; maxDiff: number;
  rows: { i: number; word: string; cls: number; yL: number; yR: number; left: Float64Array; right: Float64Array }[];
}
export function shuffleFrameAt(t: number, d: ShuffleData): ShuffleFrame {
  const u = t < 2 ? 0 : t < 4.5 ? ease((t - 2) / 2.5) : t < 7.5 ? 1 : 1 - ease((t - 7.5) / 2.5);
  const phase = markerAt(t, SHUFFLE_MARKS).label;
  const status = phase === 'start' ? 'Left: the sentence in its real order, with what attention outputs for each word.'
    : phase === 'shuffle' ? 'Right: the same words are shuffled and attention runs again. Each output strip travels with its word.'
    : phase === 'shuffled' ? 'Every word got exactly the same output strip in the shuffled sentence.'
    : 'Putting the words back in order: the strips match the left side again.';
  return {
    phase, status, u, maxDiff: d.maxDiff,
    rows: WORDS.map((word, i) => ({ i, word, cls: WORD_CLASSES[i], yL: i, yR: lerp(i, d.slotOf[i], u), left: d.original[i], right: d.shuffled[i] })),
  };
}

// ---- ablate ------------------------------------------------------------------------------------------------
/** Probability on grammatical next words at every position, from a real forward pass with position vectors scaled. */
export function positionMass(w: Weights, scale: number, ids: readonly number[] = IDS): number[] {
  const f = forward(w, [...ids], { positionScale: scale }), V = VOCAB.length;
  return ids.map((_, t) => { let m = 0; for (const id of allowedNext(ids.slice(0, t + 1))) m += f.probs[t * V + id]; return m; });
}
/** Token-weighted mean over the 200 fixed sample sentences (the figure quoted by the training report). */
export function sampleMass(w: Weights, scale: number): number {
  const p = samplePart(w, scale, 0, SAMPLE_200.length);
  return p.total / p.count;
}
/** Sum of grammatical mass over sample sentences [from, to), so callers can spread the work over several tasks. */
export function samplePart(w: Weights, scale: number, from: number, to: number): { total: number; count: number } {
  let total = 0, count = 0;
  const V = VOCAB.length;
  for (const sentence of SAMPLE_200.slice(from, to)) {
    const ids = sentence.slice(0, -1), f = forward(w, ids, { positionScale: scale });
    for (let t = 0; t < ids.length; t++) { for (const id of allowedNext(ids.slice(0, t + 1))) total += f.probs[t * V + id]; count++; }
  }
  return { total, count };
}
export interface AblateData { w: Weights; withPos: number[]; without: number[]; sampleWith: number; sampleWithout: number; cache: Map<number, number[]> }
export function makeAblate(w: Weights, withSample = true): AblateData {
  const cache = new Map<number, number[]>();
  const withPos = positionMass(w, 1), without = positionMass(w, 0);
  cache.set(1, withPos); cache.set(0, without);
  return { w, withPos, without, sampleWith: withSample ? sampleMass(w, 1) : NaN, sampleWithout: withSample ? sampleMass(w, 0) : NaN, cache };
}
export const mean = (a: ArrayLike<number>): number => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
export const fadeAt = (t: number): number => t < 1.5 ? 1 : t < 6 ? 1 - ease((t - 1.5) / 4.5) : 0;
export interface AblateFrame {
  phase: string; status: string; scale: number; mass: number[]; withPos: number[]; strips: Float64Array[]; average: number; sampleWith: number; sampleWithout: number;
}
export function ablateFrameAt(t: number, d: AblateData): AblateFrame {
  const scale = Math.round(fadeAt(t) * 200) / 200;
  let mass = d.cache.get(scale);
  if (!mass) { mass = positionMass(d.w, scale); d.cache.set(scale, mass); }
  const po = d.w.offsets.position.offset, dim = d.w.config.d;
  const phase = markerAt(t, ABLATE_MARKS).label;
  return {
    phase, scale, mass, withPos: d.withPos, average: mean(mass), sampleWith: d.sampleWith, sampleWithout: d.sampleWithout,
    strips: IDS.map((_, p) => Float64Array.from({ length: dim }, (__, j) => scale * d.w.params[po + p * dim + j])),
    status: phase === 'full size' ? 'The model sees each word plus its position vector.'
      : phase === 'fading' ? 'Position vectors shrink, and the model runs again at every size.'
      : 'No position vectors: the model sees only which words are present.',
  };
}

// ---- sinusoidal --------------------------------------------------------------------------------------------
export const sinPE = (pos: number, d = D): Float64Array => {
  const out = new Float64Array(d);
  for (let i = 0; i < d / 2; i++) { const a = pos / 10000 ** (2 * i / d); out[2 * i] = Math.sin(a); out[2 * i + 1] = Math.cos(a); }
  return out;
};
export const wavelength = (pair: number, d = D): number => 2 * Math.PI * 10000 ** (2 * pair / d);
export const TRACE_DIMS = [0, 2, 4] as const;
export interface SinData { table: Float64Array[]; word: Float64Array }
export function makeSin(): SinData {
  const rng = createRng(21);
  return { table: Array.from({ length: SIN_POSITIONS }, (_, p) => sinPE(p)), word: Float64Array.from({ length: D }, () => rng.normal(0, 0.5)) };
}
export interface SinFrame { p: number; index: number; pe: Float64Array; sum: Float64Array; phase: string; status: string }
export function sinFrameAt(t: number, d: SinData): SinFrame {
  const { p, index } = glide(t, SIN_POSITIONS, SIN_PER);
  const pe = d.table[index];
  return { p, index, pe, sum: Float64Array.from(pe, (v, j) => v + d.word[j]), phase: markerAt(t, markersOf('sinusoidal')).label,
    status: 'Walking through the positions: each one adds its own sixteen numbers to the word vector.' };
}

// ---- learned -----------------------------------------------------------------------------------------------
export interface LearnedData { strips: Float64Array[]; cos: number[][] }
export function makeLearned(w: Weights): LearnedData {
  const po = w.offsets.position.offset, d = w.config.d, n = w.config.context;
  const strips = Array.from({ length: n }, (_, p) => w.params.slice(po + p * d, po + (p + 1) * d));
  return { strips, cos: strips.map((a) => strips.map((b) => cosine(a, b))) };
}
export interface LearnedFrame { status: string; row: number; reveal: number[]; phase: string; neighbour: number; far: number }
/** Mean cosine of row r with the positions 1 step away, and with positions 3 or more steps away (position 9 left out: untrained). */
export function rowSummary(cos: number[][], r: number): { neighbour: number; far: number } {
  const near: number[] = [], far: number[] = [];
  for (let j = 0; j < UNTRAINED_ROW; j++) { if (j === r) continue; (Math.abs(j - r) === 1 ? near : Math.abs(j - r) >= 3 ? far : []).push(cos[r][j]); }
  return { neighbour: near.length ? mean(near) : NaN, far: far.length ? mean(far) : NaN };
}
export function learnedFrameAt(t: number, d: LearnedData): LearnedFrame {
  const row = Math.min(LEARNED_ROWS - 1, Math.floor(t));
  const reveal = Array.from({ length: LEARNED_ROWS }, (_, r) => t >= LEARNED_ROWS ? 1 : r < row ? 1 : r === row ? ease((t - row) / 0.8) : 0);
  const done = t >= LEARNED_ROWS;
  const none = { neighbour: NaN, far: NaN };
  return { status: done ? 'All ten rows filled in: each cell compares two position vectors.' : 'Each row fills in: how similar one position’s vector is to every other position’s.', row: done ? -1 : row, reveal, phase: done ? 'all' : String(row), ...(done || row >= UNTRAINED_ROW ? none : rowSummary(d.cos, row)) };
}

// ---- rope --------------------------------------------------------------------------------------------------
export const ROPE_Q: readonly number[] = [1.2, 0.5];
export const ROPE_K: readonly number[] = [0.9, -0.6];
/** Rotate a 2-D vector as the model's own RoPE would at position m, with one pair rotating ROPE_THETA radians per position. */
export function ropeRotate(v: readonly number[], m: number, theta = ROPE_THETA): [number, number] {
  const a = Float64Array.from(v);
  rotate(a, 2, m, { positionScale: theta });
  return [a[0], a[1]];
}
export const dot2 = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1];
export interface RopeFrame { m: number; n: number; mShown: number; nShown: number; q: [number, number]; k: [number, number]; score: number; phase: string; status: string }
export function ropeFrameAt(t: number, manual?: { m: number; n: number } | null): RopeFrame {
  let m: number, n: number, mShown: number, nShown: number;
  if (manual) { m = mShown = manual.m; n = nShown = manual.n; }
  else {
    const g = glide(t, ROPE_STOPS, ROPE_PER);
    m = ROPE_FIRST + g.p; n = m - ROPE_GAP; mShown = ROPE_FIRST + g.index; nShown = mShown - ROPE_GAP;
  }
  const q = ropeRotate(ROPE_Q, m), k = ropeRotate(ROPE_K, n);
  const score = dot2(ropeRotate(ROPE_Q, mShown), ropeRotate(ROPE_K, nShown));
  return { m, n, mShown, nShown, q, k, score, phase: manual ? 'your positions' : markerAt(t, markersOf('rope')).label,
    status: manual ? 'You chose the positions.' : `Both words move forward together, ${ROPE_GAP} places apart.` };
}

// ---- relative ----------------------------------------------------------------------------------------------
export interface RelData { rope: number[][]; additive: number[][]; meanRope: number; meanAdd: number; devRope: number; devAdd: number }
export function makeRelative(): RelData {
  const rng = createRng(5);
  const q = Float64Array.from({ length: D }, () => rng.normal()), k = Float64Array.from({ length: D }, () => rng.normal());
  const dotN = (a: ArrayLike<number>, b: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const rot = (v: Float64Array, pos: number) => { const c = v.slice(); rotate(c, D, pos, {}); return c; };
  const add = (v: Float64Array, pos: number) => Float64Array.from(sinPE(pos), (x, j) => x + v[j]);
  const rope = Array.from({ length: REL_N }, (_, m) => Array.from({ length: REL_N }, (__, n) => dotN(rot(q, m), rot(k, n))));
  const additive = Array.from({ length: REL_N }, (_, m) => Array.from({ length: REL_N }, (__, n) => dotN(add(q, m), add(k, n))));
  // Colours show distance from each map's own average; the additive scores all sit far above zero (|position|^2 = 8 is in every one).
  const centre = (g: number[][]) => { const m = mean(g.flat()); return { m, dev: Math.max(...g.flat().map((v) => Math.abs(v - m))) }; };
  const a = centre(rope), b = centre(additive);
  return { rope, additive, meanRope: a.m, meanAdd: b.m, devRope: a.dev, devAdd: b.dev };
}
/** Score cells (m = query position, n = key position) on diagonal c = n - m. */
export const diagonal = (grid: number[][], c: number): number[] => {
  const out: number[] = [];
  for (let m = 0; m < REL_N; m++) { const n = m + c; if (n >= 0 && n < REL_N) out.push(grid[m][n]); }
  return out;
};
export const spread = (values: number[]): number => Math.max(...values) - Math.min(...values);
export interface RelFrame { c: number; ropeSpread: number; addSpread: number; length: number; phase: string; status: string }
export function relFrameAt(t: number, d: RelData): RelFrame {
  let i = DIAG_STARTS.length - 1;
  while (i > 0 && DIAG_STARTS[i] > t) i--;
  const c = DIAGONALS[i], a = diagonal(d.rope, c), b = diagonal(d.additive, c);
  const phase = markerAt(t, markersOf('relative')).label;
  return { c, ropeSpread: spread(a), addSpread: spread(b), length: a.length, phase,
    status: phase === 'same' ? 'Main diagonal: the key sits at the same position as the query.'
      : phase === 'key earlier' ? 'Diagonals where the key comes before the query, one gap at a time.' : 'Diagonals where the key comes after the query, one gap at a time.' };
}

// ---- shared ------------------------------------------------------------------------------------------------
function markerAt<M extends { at: number }>(t: number, markers: readonly M[]): M {
  let best = markers[0];
  for (const m of markers) if (m.at <= t + 1e-9) best = m;
  return best;
}
