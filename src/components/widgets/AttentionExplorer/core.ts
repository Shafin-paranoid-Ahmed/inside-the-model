/** AttentionExplorer shared core: sentence input, scene, timelines and the frame types. No mode-specific frames. */
import { forward, type ForwardOptions, type Trace, type Weights } from '../../../lib/transformer/model';
import { allowedNext, encodeWords, VOCAB } from '../../../lib/transformer/grammar';
import { markerAt, type Marker } from '../_shared/clock';

export const MODES = ['arcs', 'qkv', 'scores', 'softmax', 'mix', 'explorer', 'reach',
  'compare', 'split', 'heads', 'concat', 'gallery', 'ablate'] as const;
export type Mode = (typeof MODES)[number];
export const DEFAULT_SENTENCE = '[start] these small sheep chase a red fox';
export const REACH_SENTENCE = '[start] these small old sheep chase a red fox';
export const MAX_TOKENS = 10;
export const HEADS: readonly (readonly [number, number])[] = [[0, 0], [0, 1], [1, 0], [1, 1]];
export const headName = (layer: number, head: number) => `layer ${layer + 1}, head ${head + 1}`;

export const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
export const ease = (x: number) => { const u = clamp01(x); return u * u * (3 - 2 * u); };
export const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
export const f2 = (x: number) => x.toFixed(2);
export const sgn = (x: number, p = 2) => (x < 0 ? '−' : '') + Math.abs(x).toFixed(p);
export const sum = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
export const norm = (a: ArrayLike<number>) => Math.sqrt(a.length ? sum(Array.from(a, (x) => x * x)) : 0);

// ---------- sentence input ----------

export interface Validation {
  ids: number[] | null;
  words: string[];
  unknown: { index: number; word: string; suggestions: string[] }[];
  tooLong: boolean;
  offGrammar: { index: number; word: string } | null;
}
export function lev(a: string, b: string): number {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...new Array<number>(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++)
    d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return d[a.length][b.length];
}
/** Up to three vocabulary words that start like `word` or are within two edits of it; none when nothing is close. */
export function suggest(word: string): string[] {
  return VOCAB.filter((v) => v !== '[start]' && v !== '.')
    .map((v) => ({ v, d: word.length >= 2 && v.startsWith(word) ? 0 : lev(word, v) }))
    .filter((x) => x.d <= 2).sort((a, b) => a.d - b.d || a.v.localeCompare(b.v)).slice(0, 3).map((x) => x.v);
}
/**
 * Words must be in TinyLM's 34-word vocabulary (a missing [start] is added). Out-of-vocabulary words block the run.
 * A sentence that leaves the training grammar still runs: the model accepts any in-vocabulary sequence.
 */
export function validate(text: string): Validation {
  let words = text.toLowerCase().replace(/\./g, ' . ').split(/\s+/).filter(Boolean);
  if (words[0] !== '[start]') words = ['[start]', ...words];
  const unknown = words.flatMap((word, index) => (VOCAB as readonly string[]).includes(word) ? [] : [{ index, word, suggestions: suggest(word) }]);
  const tooLong = words.length > MAX_TOKENS;
  const ids = unknown.length || tooLong || words.length < 2 ? null : encodeWords(words.join(' '));
  let offGrammar: Validation['offGrammar'] = null;
  if (ids) for (let k = 0; k < ids.length && !offGrammar; k++) {
    let ok = false;
    try { ok = allowedNext(ids.slice(0, k)).has(ids[k]); } catch { ok = false; }
    if (!ok) offGrammar = { index: k, word: words[k] };
  }
  return { ids, words, unknown, tooLong, offGrammar };
}

// ---------- scene ----------

export interface Scene {
  w: Weights; ids: number[]; words: string[]; T: number; trace: Trace;
  layer: number; head: number; dk: number; d: number; layers: number; heads: number;
}
export function makeScene(w: Weights, ids: readonly number[], layer = 0, head = 0, opts: ForwardOptions = {}): Scene {
  const trace = forward(w, ids, opts), c = w.config;
  return { w, ids: [...ids], words: ids.map((i) => VOCAB[i]), T: ids.length, trace, layer, head, dk: c.d / c.heads, d: c.d, layers: c.layers, heads: c.heads };
}
export const defaultScene = (w: Weights, sentence = DEFAULT_SENTENCE, layer = 0, head = 0) => makeScene(w, encodeWords(sentence), layer, head);
export function headData(s: Scene, l = s.layer, h = s.head) {
  const L = s.trace.layers[l];
  return { q: L.q[h], k: L.k[h], v: L.v[h], S: L.scores[h], A: L.weights[h], out: L.headOut[h] };
}
export const rowOf = (a: Float64Array, width: number, i: number) => a.subarray(i * width, (i + 1) * width);
/** First occurrence of a word (or a numeric index). Defaults to the fourth token ("sheep"). */
export function resolveQuery(s: Scene, q?: string | number): number {
  if (typeof q === 'number') return Math.max(0, Math.min(s.T - 1, Math.round(q)));
  if (typeof q === 'string') { const i = s.words.indexOf(q.toLowerCase()); if (i >= 0) return i; }
  return Math.min(3, s.T - 1);
}
/** 16-by-16 projection as used on the page: rows are input dimensions, columns output dimensions. */
export function weightMatrix(w: Weights, layer: number, name: 'q' | 'k' | 'v' | 'o'): Float64Array {
  const t = w.offsets[`${layer}.${name}.weight`], out = new Float64Array(t.length);
  for (let r = 0; r < t.rows; r++) for (let j = 0; j < t.cols; j++) out[j * t.rows + r] = w.params[t.offset + r * t.cols + j];
  return out;
}
/** The 16-number query, key or value of one token: both heads' slices side by side. */
export function fullVector(s: Scene, layer: number, name: 'q' | 'k' | 'v', i: number): Float64Array {
  const L = s.trace.layers[layer], out = new Float64Array(s.d);
  for (let h = 0; h < s.heads; h++) out.set(rowOf(L[name][h], s.dk, i), h * s.dk);
  return out;
}

// ---------- timelines ----------

export interface Timeline { duration: number; markers: Marker[] }
export const ARC_SEG = 1.8, QKV_SEG = 1.7, QKV_HOLD = 1.4, EXP_SEG = 3.2, CMP_SEG = 2.0, GAL_SEG = 1.4, MIX_SEG = 1.2, MIX_HOLD = 2.0;
export const SC_D = 0.9, SC_F = 0.2, SC_G = 0.05, SC_HOLD = 1.6;
export const SM = { raw: 0, scale: 1.4, mask: 3.4, soft: 4.6, noscale: 7.0, end: 10.5 };
export const REACH_HOP = 0.7, REACH_LEAD = 0.6, REACH_WORD = 0.5;
export const SPLIT = [0, 2.6, 5.8, 9.0];
export const HEADS_ROW = 0.45, HEADS_OUT = 0.28;
export const CAT = { side: 0, join: 1.6, proj: 3.4, add: 5.6, end: 9.4 };
export const AB_SEG = 2.4;

export const scoreTimes = (T: number, qi: number) => {
  const rowEnd = (qi + 1) * SC_D + (T - qi - 1) * SC_F, allEnd = rowEnd + (T - 1) * T * SC_G + 0.2;
  return { rowEnd, allEnd, end: allEnd + SC_HOLD };
};
export const reachCount = (T: number, words: readonly string[]) => { const to = reachEnds(T, words); return { hops: to.to - to.from, lead: REACH_LEAD, pathEnd: REACH_LEAD + (to.to - to.from) * REACH_HOP + 1.3 }; };
export const reachEnds = (T: number, words: readonly string[]) => ({ from: Math.min(1, T - 1), to: Math.max(1, words[T - 1] === '.' ? T - 2 : T - 1) }); // first word to the last word

export function timeline(mode: Mode, s: Scene | null, query: string | number = 3): Timeline {
  const T = s?.T ?? 8, qi = s ? resolveQuery(s, query) : 3;
  const words = s?.words ?? [];
  switch (mode) {
    case 'arcs': return { duration: T * ARC_SEG, markers: words.map((w, i) => ({ at: i * ARC_SEG, label: w })) };
    case 'qkv': return { duration: T * QKV_SEG + QKV_HOLD, markers: [...words.map((w, i) => ({ at: i * QKV_SEG, label: w })), { at: T * QKV_SEG, label: 'All rows' }] };
    case 'scores': { const t = scoreTimes(T, qi); return { duration: t.end, markers: [{ at: 0, label: `Row: ${words[qi] ?? 'query'}` }, { at: t.rowEnd, label: 'Every row' }, { at: t.allEnd, label: 'Whole grid' }] }; }
    case 'softmax': return { duration: SM.end, markers: [{ at: SM.raw, label: 'Raw scores' }, { at: SM.scale, label: 'Scale' }, { at: SM.mask, label: 'Mask' }, { at: SM.soft, label: 'Softmax' }, { at: SM.noscale, label: 'Without scaling' }] };
    case 'mix': return { duration: (qi + 1) * MIX_SEG + MIX_HOLD, markers: [...words.slice(0, qi + 1).map((w, i) => ({ at: i * MIX_SEG, label: w })), { at: (qi + 1) * MIX_SEG, label: 'Head output' }] };
    case 'explorer': return { duration: T * EXP_SEG, markers: words.map((w, i) => ({ at: i * EXP_SEG, label: w })) };
    case 'reach': { const r = reachCount(T, words); return { duration: r.pathEnd + T * REACH_WORD + 1.4, markers: [{ at: 0, label: 'Hops between words' }, { at: r.pathEnd, label: 'Number of scores' }] }; }
    case 'compare': return { duration: T * CMP_SEG, markers: words.map((w, i) => ({ at: i * CMP_SEG, label: w })) };
    case 'split': return { duration: SPLIT[3], markers: [{ at: SPLIT[0], label: 'One head' }, { at: SPLIT[1], label: 'Two heads' }, { at: SPLIT[2], label: 'Four heads' }] };
    case 'heads': { const e = T * HEADS_ROW; return { duration: e + T * HEADS_OUT + 0.6 + 2, markers: [{ at: 0, label: 'Both fill in' }, { at: e, label: 'Head outputs' }, { at: e + T * HEADS_OUT + 0.6, label: 'Disagreement' }] }; }
    case 'concat': return { duration: CAT.end, markers: [{ at: CAT.side, label: 'Two heads' }, { at: CAT.join, label: 'Concatenate' }, { at: CAT.proj, label: 'Multiply by W_O' }, { at: CAT.add, label: 'Add to the word' }] };
    case 'gallery': return { duration: T * GAL_SEG, markers: words.map((w, i) => ({ at: i * GAL_SEG, label: w })) };
    case 'ablate': return { duration: 5 * AB_SEG, markers: ABLATE_LABELS.map((label, i) => ({ at: i * AB_SEG, label })) };
  }
}
export const ABLATE_LABELS = ['Full model', 'L1 H1 off', 'L1 H2 off', 'L2 H1 off', 'L2 H2 off'];

/** Playhead where query `qi` has finished its entrance, so a paused figure shows that query settled. */
export function settledTime(mode: Mode, _s: Scene, qi: number): number {
  switch (mode) {
    case 'arcs': return qi * ARC_SEG + ARC_SEG * 0.8;
    case 'explorer': return qi * EXP_SEG + EXP_SEG * 0.98;
    case 'compare': return qi * CMP_SEG + CMP_SEG * 0.9;
    case 'gallery': return qi * GAL_SEG + GAL_SEG * 0.5;
    case 'qkv': return qi * QKV_SEG + QKV_SEG;
    default: return 0;
  }
}
/** Which query a sweeping mode has reached at `t` (null for modes that do not sweep). */
export function sweepQuery(mode: Mode, s: Scene, t: number): number | null {
  const seg = { arcs: ARC_SEG, explorer: EXP_SEG, compare: CMP_SEG, gallery: GAL_SEG, qkv: QKV_SEG }[mode as 'arcs'];
  return seg ? Math.min(s.T - 1, Math.floor(t / seg + 1e-9)) : null;
}
/** A frame that shows the finished idea, for readers who prefer reduced motion (the clock starts paused here). */
export function restTime(mode: Mode, s: Scene, query?: string | number): number {
  const tl = timeline(mode, s, query), at = (label: string) => tl.markers.find((m) => m.label === label)?.at ?? tl.duration;
  switch (mode) {
    case 'scores': return tl.duration;
    case 'softmax': return at('Without scaling') - 0.02;
    case 'mix': case 'heads': case 'qkv': return tl.duration;
    case 'reach': return at('Number of scores') - 0.02;
    case 'split': return at('Four heads') - 0.02;
    case 'concat': return tl.duration - 0.02;
    case 'ablate': return at('L1 H1 off') + 1.8;
    default: return settledTime(mode, s, resolveQuery(s, query));
  }
}
/** Start playing from a given query for modes that sweep (so `query="sheep"` opens on that word). */
export const startTime = (mode: Mode, s: Scene, query?: string | number): number =>
  query === undefined || sweepQuery(mode, s, 0) === null ? 0 : settledTime(mode, s, resolveQuery(s, query));

// ---------- frames ----------

export interface Base { status: string; phase: string }
export interface ArcsFrame extends Base { mode: 'arcs'; q: number; prev: Float64Array; target: Float64Array; fade: number }
export interface QkvFrame extends Base { mode: 'qkv'; i: number; u: number; flow: number; reveal: Float64Array; active: boolean }
export interface ScoresFrame extends Base {
  mode: 'scores'; qi: number; vis: Float64Array; val: Float64Array;
  detail: { j: number; u: number; products: Float64Array; shown: number; sum: number } | null;
}
export interface SoftmaxFrame extends Base {
  mode: 'softmax'; qi: number; raw: Float64Array; scaled: Float64Array; w: Float64Array; f: number;
  maskU: number; alpha: [number, number, number]; sum: number; maxW: number; maxIdx: number;
}
export interface MixFrame extends Base {
  mode: 'mix'; qi: number; j: number; u: number; flow: number; a: Float64Array; acc: Float64Array; contrib: Float64Array; done: boolean;
}
export interface PipelineRow { raw: Float64Array; scaled: Float64Array; w: Float64Array; out: Float64Array }
export interface ExplorerFrame extends Base { mode: 'explorer'; q: number; prev: Float64Array; fade: number; stage: [number, number, number, number]; pipe: PipelineRow }
export interface ReachFrame extends Base {
  mode: 'reach'; part: 'path' | 'cost'; from: number; to: number; rnnPos: number; rnnHops: number; attnU: number; attnHops: number; Tn: number; scores: number;
}
export interface CompareFrame extends Base { mode: 'compare'; q: number; prevs: Float64Array[]; targets: Float64Array[]; fade: number; argmax: number[] }
export interface SplitFrame extends Base { mode: 'split'; h: number; gaps: Float64Array; count: number; dkHead: number }
export interface HeadsFrame extends Base { mode: 'heads'; rowVis: Float64Array; outVis: Float64Array; hiU: number; hiRow: number }
export interface ConcatFrame extends Base {
  mode: 'concat'; qi: number; slide: number; proj: number; add: number; norms: { input: number; attn: number; sum: number };
}
export interface GalleryFrame extends Base { mode: 'gallery'; q: number; tops: { head: number; word: number; weight: number }[] }
export interface AblateFrame extends Base { mode: 'ablate'; state: number; mass: Float64Array; base: Float64Array; avg: number; avgFull: number; off: boolean[] }
export type Frame = ArcsFrame | QkvFrame | ScoresFrame | SoftmaxFrame | MixFrame | ExplorerFrame | ReachFrame | CompareFrame | SplitFrame | HeadsFrame | ConcatFrame | GalleryFrame | AblateFrame;

export interface Opts {
  mode: Mode; query?: string | number;
  /** A pinned query for sweeping modes (hover/tap): shows that token finished, playhead ignored. */
  pin?: number | null;
  /** ablate: masses for [full, L1H1, L1H2, L2H1, L2H2] from `ablationMasses`. */
  mass?: number[][];
  /** ablate: reader's toggles override the timeline. */
  toggles?: boolean[] | null; customMass?: number[] | null;
}

/** Index of the largest value, ties to the first. */
export function argmax(a: ArrayLike<number>, n = a.length): number {
  let b = 0; for (let i = 1; i < n; i++) if (a[i] > a[b]) b = i; return b;
}
export function seg(t: number, size: number, n: number) {
  const i = Math.max(0, Math.min(n - 1, Math.floor(t / size + 1e-9)));
  return { i, u: clamp01((t - i * size) / size) };
}
/** Eased row: attention row of query q, blended in from the previous query's row. */
export function easedRow(s: Scene, q: number, mix: number): Float64Array {
  const { A } = headData(s), T = s.T, out = new Float64Array(T), cur = rowOf(A, T, q);
  const prev = q > 0 ? rowOf(A, T, q - 1) : null;
  for (let j = 0; j < T; j++) out[j] = lerp(prev ? prev[j] : 0, cur[j], mix);
  return out;
}
export const topWord = (s: Scene, row: ArrayLike<number>) => { const b = argmax(row); return { b, word: s.words[b], w: row[b] }; };

export const A_MODES: readonly Mode[] = ['arcs', 'qkv', 'scores', 'softmax', 'mix', 'explorer', 'reach'];

/** Common start of every frame function: clamped playhead, active marker name, sentence length, head width, query index. */
export function pre(t: number, s: Scene, o: Opts) {
  const tl = timeline(o.mode, s, o.query), tt = Math.max(0, Math.min(tl.duration, t));
  return { tt, phase: markerAt(tt, tl.markers)?.label ?? '', T: s.T, dk: s.dk, qs: resolveQuery(s, o.query) };
}
