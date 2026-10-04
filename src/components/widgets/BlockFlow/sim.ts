// BlockFlow: pure simulation for the transformer-block lesson. No DOM. Every frame is a pure function of t.
// Numbers come from real forward passes of the shipped TinyLM weights (passed in), plus one precomputed
// gradient table (see NORM_*), which tests/unit/block-flow.test.ts re-derives from the core.
import { encodeWords, VOCAB } from '../../../lib/transformer/grammar';
import { TINY_CONFIG, forward, logitLens, parameterLayout, project, softmax, type Weights } from '../../../lib/transformer/model';
import type { Marker } from '../_shared/clock';

export type Mode = 'overview' | 'residual' | 'mlp' | 'norm' | 'stream' | 'count';
export const DEFAULT_SENTENCE = 'these small sheep chase a red fox';
export const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
export const ease = (x: number) => { const c = clamp01(x); return c * c * (3 - 2 * c); };
const seg = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;

/** Probability text: three decimals, four for small values. */
export const fmtP = (p: number) => p < 0.00005 ? '<0.0001' : p < 0.01 ? p.toFixed(4) : p.toFixed(3);
export const fmtN = (n: number) => Math.round(n).toLocaleString('en-US');

// ---------- Forward-pass data -------------------------------------------------------------------

export interface BlockData {
  input: Float64Array; attn: Float64Array; r1: Float64Array; mlp: Float64Array; r2: Float64Array;
  pre: Float64Array; hidden: Float64Array; arcs: Float64Array; active: number[];
}
export interface Stage { name: string; vec: Float64Array; probs: Float64Array }
export interface Data {
  words: string[]; T: number; d: number; ff: number; q: number; blocks: BlockData[]; stages: Stage[];
  x0: Float64Array; finalProbs: Float64Array;
}
export const rowOf = (a: Float64Array, t: number, width: number) => a.subarray(t * width, (t + 1) * width);
export const norm2 = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); };

export function buildData(w: Weights, sentence = DEFAULT_SENTENCE, query = 'sheep'): Data {
  const text = sentence.trim().startsWith('[start]') ? sentence.trim() : `[start] ${sentence.trim()}`;
  const ids = encodeWords(text), words = ids.map((id) => VOCAB[id] as string), c = w.config, d = c.d, T = ids.length;
  const q = words.indexOf(query);
  if (q < 0) throw new Error(`Query word "${query}" is not in the sentence`);
  const f = forward(w, ids);
  const blocks = f.layers.map((L, l): BlockData => {
    const arcs = new Float64Array(T * T);
    for (const wt of L.weights) for (let i = 0; i < arcs.length; i++) arcs[i] += wt[i] / c.heads;
    const active = Array.from({ length: T }, (_, t) => { let n = 0; for (let j = 0; j < c.ff; j++) if (L.hidden[t * c.ff + j] > 0) n++; return n; });
    return { input: L.input, attn: L.attnOut, r1: L.resid1, mlp: L.mlpOut, r2: L.resid2,
      pre: project(w, `${l}.ff1`, L.mlpInput, T), hidden: L.hidden, arcs, active };
  });
  // Logit lens: final LayerNorm + output layer applied to the stream after each sublayer, at the query word.
  const at = (a: Float64Array) => Float64Array.from(rowOf(a, q, d));
  const raw: [string, Float64Array][] = [['Embedding', at(f.x0)]];
  f.layers.forEach((L, l) => { raw.push([`After attention ${l + 1}`, at(L.resid1)], [`After MLP ${l + 1}`, at(L.resid2)]); });
  const stages = raw.map(([name, vec]) => ({ name, vec, probs: softmax(logitLens(w, vec)) }));
  return { words, T, d, ff: c.ff, q, blocks, stages, x0: f.x0, finalProbs: Float64Array.from(rowOf(f.probs, q, c.vocab.length)) };
}
export const topK = (p: ArrayLike<number>, k: number) =>
  Array.from(p, (v, i) => ({ word: VOCAB[i] as string, id: i, p: v })).sort((a, b) => b.p - a.p).slice(0, k);
export const probOf = (p: ArrayLike<number>, word: string) => p[VOCAB.indexOf(word as (typeof VOCAB)[number])];
export const PLURAL = 'chase', SINGULAR = 'chases';

// ---------- Precomputed gradient table (norm mode) ---------------------------------------------
// Gradient norm of the loss with respect to the residual stream after k blocks (k = 0 is the embeddings,
// k = 12 the output of block 12 before the final LayerNorm). Random-init stack of 12 TinyLM-sized
// blocks, seed 91, sentence "[start] these small sheep chase a red fox". Same values as the model manifest
// (public/models/tiny-lm.json, metrics.randomInit12BlockGradientNorms); the unit test recomputes them.
export const NORM_PRE = [0.8868306611653286, 0.06923561471698209, 0.04904427814440106, 0.03780786842692869, 0.03343480496476525, 0.03149943078093625, 0.03012058879236874, 0.028897898966365083, 0.027499776047208813, 0.025568053014684324, 0.02528345547597331, 0.02439717177176582, 0.024167978114209756];
export const NORM_POST = [0.16397592113976311, 0.01571073617637123, 0.018446437651183444, 0.015711036188197906, 0.016575458988263223, 0.024647280823273335, 0.034225744072754076, 0.04844292701242881, 0.05375450053760758, 0.07223674967666739, 0.09046019719472259, 0.11005780391716712, 0.12352772725173795];
export const NORM_SEED = 91;
/** Spread across blocks 1 to 12 (indices 1..12): largest over smallest. */
export const spread = (a: readonly number[]) => { const s = a.slice(1); return Math.max(...s) / Math.min(...s); };

// ---------- Parameter counts ---------------------------------------------------------------------

export interface SizeConfig { name: string; d: number; L: number; V: number; ctx: number; logitBias: boolean }
export const PRESETS: SizeConfig[] = [
  { name: 'TinyLM', d: 16, L: 2, V: 34, ctx: 10, logitBias: true },
  { name: 'GPT-2 small', d: 768, L: 12, V: 50257, ctx: 1024, logitBias: false },
  { name: 'GPT-2 XL', d: 1600, L: 48, V: 50257, ctx: 1024, logitBias: false },
];
export interface Counts { embeddings: number; attention: number; mlp: number; norms: number; total: number; perBlock: number }
/** Closed-form count for a pre-norm, tied-output, 4d-MLP, learned-position transformer. */
export function countParams(c: Pick<SizeConfig, 'd' | 'L' | 'V' | 'ctx' | 'logitBias'>): Counts {
  const { d, L } = c;
  const embeddings = (c.V + c.ctx) * d, attention = L * (4 * d * d + 4 * d), mlp = L * (8 * d * d + 5 * d);
  const norms = L * 4 * d + 2 * d + (c.logitBias ? c.V : 0);
  return { embeddings, attention, mlp, norms, total: embeddings + attention + mlp + norms, perBlock: 12 * d * d + 13 * d };
}
/** The same breakdown read off TinyLM's real parameter layout. */
export function tinyCounts(): Counts {
  const sum = (test: (n: string) => boolean) => Object.entries(parameterLayout(TINY_CONFIG).offsets).filter(([n]) => test(n)).reduce((s, [, t]) => s + t.length, 0);
  const embeddings = sum((n) => n === 'embedding' || n === 'position');
  const attention = sum((n) => /\.(q|k|v|o)\./.test(n)), mlp = sum((n) => /\.ff[12]\./.test(n));
  const norms = sum((n) => /\.ln[12]\.|^final\.|^logit\./.test(n));
  return { embeddings, attention, mlp, norms, total: embeddings + attention + mlp + norms, perBlock: (attention + mlp) / TINY_CONFIG.layers + 4 * TINY_CONFIG.d };
}
/** The approximation 12 L d^2 + V d from the lesson. */
export const approxParams = (c: Pick<SizeConfig, 'd' | 'L' | 'V'>) => 12 * c.L * c.d * c.d + c.V * c.d;

// ---------- Timelines ---------------------------------------------------------------------------

export const TIMELINES: Record<Exclude<Mode, 'count'>, { duration: number; markers: Marker[] }> = {
  overview: { duration: 14, markers: [{ at: 0, label: 'Words enter' }, { at: 2, label: 'Attention, block 1' }, { at: 4.5, label: 'MLP, block 1' }, { at: 7, label: 'Attention, block 2' }, { at: 9.5, label: 'MLP, block 2' }, { at: 12, label: 'Next-word scores' }] },
  residual: { duration: 12, markers: [{ at: 0, label: 'The vector x' }, { at: 2.5, label: 'Attention writes an update' }, { at: 6, label: 'Add it to x' }, { at: 9, label: 'Every word' }] },
  mlp: { duration: 9, markers: [{ at: 0, label: 'Expand to 64 units' }, { at: 3, label: 'ReLU zeroes negatives' }, { at: 6, label: 'Contract to 16' }] },
  norm: { duration: 10, markers: [{ at: 0, label: 'Bars grow with depth' }, { at: 6, label: 'Compare the spread' }] },
  stream: { duration: 10, markers: [{ at: 0, label: 'Embedding' }, { at: 2, label: 'Attention 1' }, { at: 4, label: 'MLP 1' }, { at: 6, label: 'Attention 2' }, { at: 8, label: 'MLP 2' }] },
};
export const COUNT_SEG = 6;
export function timelineOf(mode: Mode, custom = false): { duration: number; markers: Marker[] } {
  if (mode !== 'count') return TIMELINES[mode];
  return custom ? { duration: COUNT_SEG, markers: [{ at: 0, label: 'Your size' }] }
    : { duration: COUNT_SEG * PRESETS.length, markers: PRESETS.map((p, i) => ({ at: i * COUNT_SEG, label: p.name })) };
}
export const phaseOf = (t: number, markers: readonly Marker[]) => {
  let k = 0; markers.forEach((m, i) => { if (m.at <= t + 1e-9) k = i; }); return k;
};

// ---------- Frames --------------------------------------------------------------------------------

export interface OverviewFrame { mode: 'overview'; phase: number; local: number; s: number; status: string }
export interface ResidualFrame { mode: 'residual'; phase: number; local: number; xA: number; dCells: number; sCells: number; barsA: number; status: string }
export interface MlpFrame { mode: 'mlp'; phase: number; local: number; rowA: number[]; relu: number; countsA: number; outCells: number; status: string }
export interface NormFrame { mode: 'norm'; phase: number; reveal: number; rangeA: number; last: number; status: string }
export interface StreamRow { stage: number; chase: number; chases: number; chaseBar: number; chasesBar: number; vecA: number }
export interface StreamFrame { mode: 'stream'; phase: number; local: number; morph: number; rows: StreamRow[]; top: { word: string; p: number; bar: number }[]; status: string }
export interface CountFrame { mode: 'count'; phase: number; preset: number; config: SizeConfig; blocks: number; counts: Counts; status: string }
export type Frame = OverviewFrame | ResidualFrame | MlpFrame | NormFrame | StreamFrame | CountFrame;

const OVERVIEW_STATUS = [
  'Each word enters as its embedding plus a position vector: 16 numbers per word.',
  'Attention, block 1: each word reads from the words before it (the arcs) and writes an update to its own column.',
  'MLP, block 1: each word is processed on its own. Nothing crosses between columns.',
  'Attention, block 2: the words exchange information again, now using block 1\'s results.',
  'MLP, block 2: the second block\'s per-word network.',
  'A final normalization and the output layer turn each word\'s vector into a score for every possible next word.',
];

export interface Opts { layer?: number; custom?: { d: number; L: number } | null; sentenceLabel?: string }

export function frameAt(t: number, mode: Mode, data: Data | null, opts: Opts = {}): Frame {
  const tl = timelineOf(mode, !!opts.custom), phase = phaseOf(t, tl.markers);
  const start = tl.markers[phase].at, end = tl.markers[phase + 1]?.at ?? tl.duration;
  const local = clamp01((t - start) / (end - start));
  switch (mode) {
    case 'overview':
      return { mode, phase, local, s: phase + ease(local), status: OVERVIEW_STATUS[phase] };
    case 'residual': {
      const l = (opts.layer ?? 1) - 1, B = data?.blocks[l], q = data?.q ?? 0;
      const n = (a?: Float64Array) => (a && data ? norm2(rowOf(a, q, data.d)) : 0);
      const word = data?.words[q] ?? '';
      const xn = n(B?.input).toFixed(2), dn = n(B?.attn).toFixed(2), sn = n(B?.r1).toFixed(2);
      const status = [
        `Block ${l + 1} receives a 16-number vector for "${word}": its running vector x, with length ${xn}.`,
        `Attention reads the other words and produces an update with length ${dn}. It is a separate vector, not yet merged.`,
        `The update is added cell by cell to x. The sum, x plus the update, has length ${sn}.`,
        'For every word in the sentence: the length of its incoming vector x next to the length of the update attention writes.',
      ][phase];
      return { mode, phase, local, status,
        xA: phase === 0 ? ease(local) : 1,
        dCells: phase < 1 ? 0 : phase === 1 ? 16 * ease(local) : 16,
        sCells: phase < 2 ? 0 : phase === 2 ? 16 * ease(local) : 16,
        barsA: phase === 3 ? ease(local) : 0 };
    }
    case 'mlp': {
      const l = (opts.layer ?? 1) - 1, B = data?.blocks[l], T = data?.T ?? 8, q = data?.q ?? 0;
      const rowA = Array.from({ length: T }, (_, i) => phase > 0 ? 1 : clamp01(local * (T + 2) - i));
      const w1 = data?.words[q] ?? '', w2 = data?.words[T - 1] ?? '';
      const status = [
        `Expand: each word's 16 numbers are multiplied up to ${data?.ff ?? 64} hidden units. Blue cells are positive, orange cells negative.`,
        `ReLU sets every negative number to zero. "${w1}" keeps ${B?.active[q]} of ${data?.ff ?? 64} units; "${w2}" keeps ${B?.active[T - 1]}.`,
        `Contract: the surviving units are combined back into 16 numbers per word and added to the stream. Same network, same weights, for every word.`,
      ][phase];
      return { mode, phase, local, rowA, status,
        relu: phase === 0 ? 0 : phase === 1 ? ease(local) : 1,
        countsA: phase === 0 ? 0 : phase === 1 ? clamp01((local - 0.4) * 2.5) : 1,
        outCells: phase < 2 ? 0 : 16 * ease(local) };
    }
    case 'norm': {
      const reveal = phase === 0 ? 13 * ease(local) : 13;
      const last = Math.floor(reveal + 1e-9) - 1;
      const status = phase === 1
        ? `Blocks 1 to 12 (block 0, the embeddings, is faded out): pre-norm gradients fall from ${NORM_PRE[1].toFixed(4)} to ${NORM_PRE[12].toFixed(4)}, a spread of ${spread(NORM_PRE).toFixed(1)}x; post-norm gradients rise from ${NORM_POST[1].toFixed(4)} to ${NORM_POST[12].toFixed(4)}, a spread of ${spread(NORM_POST).toFixed(1)}x.`
        : last < 0 ? 'Gradient norm at the residual stream, starting from the embeddings.'
        : `After ${last === 0 ? 'the embeddings' : `block ${last}`}: pre-norm ${NORM_PRE[last].toFixed(4)}, post-norm ${NORM_POST[last].toFixed(4)}.`;
      return { mode, phase, reveal, last, rangeA: phase === 1 ? ease(local * 2) : 0, status };
    }
    case 'stream': {
      const S = data?.stages ?? [], k = phase, morph = ease(local / 0.3);
      // Numbers snap to the real stage; only bar geometry eases between stages.
      const val = (i: number, word: string) => S[i] ? probOf(S[i].probs, word) : 0;
      const bar = (i: number, word: string) => i === k && i > 0 ? lerp(val(i - 1, word), val(i, word), morph) : val(i, word);
      const rows = Array.from({ length: k + 1 }, (_, i) => ({ stage: i, chase: val(i, PLURAL), chases: val(i, SINGULAR),
        chaseBar: bar(i, PLURAL), chasesBar: bar(i, SINGULAR), vecA: i === k ? morph : 1 }));
      const top = S[k] ? topK(S[k].probs, 3).map((e) => ({ word: e.word, p: e.p,
        bar: k > 0 ? lerp(S[k - 1].probs[e.id], e.p, morph) : e.p * morph })) : [];
      const q = data?.words[data.q] ?? '';
      const names = ['The embedding alone', 'The stream after attention 1', 'The stream after MLP 1', 'The stream after attention 2', 'The stream after MLP 2 (the model\'s real output)'];
      return { mode, phase, local, morph, rows, top, status: `${names[k]}, read at "${q}", predicts the next word.` };
    }
    case 'count': {
      const c = opts.custom, idx = c ? 0 : phase, lt = c ? t : t - idx * COUNT_SEG;
      const base = PRESETS[idx];
      const config: SizeConfig = c ? { name: 'Your size', d: c.d, L: c.L, V: PRESETS[1].V, ctx: PRESETS[1].ctx, logitBias: false } : base;
      const final = idx === 0 && !c ? tinyCounts() : countParams(config);
      const k = ease(seg(lt, 0.3, 3.3));
      const counts: Counts = { embeddings: Math.round(final.embeddings * k), attention: Math.round(final.attention * k), mlp: Math.round(final.mlp * k), norms: Math.round(final.norms * k), total: 0, perBlock: final.perBlock };
      counts.total = counts.embeddings + counts.attention + counts.mlp + counts.norms;
      return { mode, phase: idx, preset: c ? -1 : idx, config, blocks: config.L * seg(lt, 0, 2.5), counts,
        status: `${config.name}: d = ${config.d}, ${config.L} blocks, vocabulary ${fmtN(config.V)}, context ${fmtN(config.ctx)}. Each block has ${fmtN(final.perBlock)} parameters; the whole model has ${fmtN(final.total)}.` };
    }
  }
}

/** Signed cell colour: [token, opacity]. Shared max keeps equal numbers equal in colour. */
export const cellStyle = (v: number, max: number): { color: '--viz-pos' | '--viz-neg'; a: number } =>
  ({ color: v >= 0 ? '--viz-pos' : '--viz-neg', a: Math.min(1, Math.abs(v) / (max || 1)) * 0.9 + 0.06 });
export const maxAbs = (...arrs: ArrayLike<number>[]) => { let m = 0; for (const a of arrs) for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m; };
