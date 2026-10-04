/** Causal masking: every number comes from real forward passes of the shipped TinyLM weights. No DOM. */
import { allowedNext, encodeWords, VOCAB, WORD_CLASS } from '../../../lib/transformer/grammar';
import { forward, generate, softmax, type Weights } from '../../../lib/transformer/model';

export type Mode = 'leak' | 'mask' | 'neginf' | 'parallel' | 'edit' | 'encdec';
export interface Pt { step: number; loss: number }
export interface Marker { at: number; label: string }
export interface Models { causal: Weights; free?: Weights }
export interface Options { sentence?: string; prompt?: string; layer?: number; head?: number; query?: number; editAt?: number }
export const SENTENCE = '[start] these small sheep chase a red fox';
export const PROMPT = '[start] these big birds';
export const STOP = 1; // the "." token

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const ramp = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
const ease = (x: number) => x * x * (3 - 2 * x);
const V = VOCAB.length;
const word = (id: number) => VOCAB[id];

// ---------- shared helpers ----------
export interface Chip { x: number; y: number; w: number }
/** Left-to-right word chips that wrap at maxW. Pure layout, used for sentences. */
export function flowChips(labels: readonly string[], maxW: number, charW: number, chipH: number, gap = 4): { chips: Chip[]; lines: number } {
  const chips: Chip[] = [];
  let x = 0, line = 0;
  for (const l of labels) {
    const w = Math.round(14 + charW * l.length);
    if (x > 0 && x + w > maxW) { x = 0; line++; }
    chips.push({ x, y: line * (chipH + gap), w });
    x += w + gap;
  }
  return { chips, lines: labels.length ? line + 1 : 0 };
}

export interface Row {
  input: string; target: string;
  bar: number; // p(target) in `parallel`, p(top guess) in `edit`
  top: { word: string; p: number }[];
  loss: number; // cross-entropy of the target (parallel)
  change: number; // largest probability change versus the original sentence (edit)
  uses: number; // how many words this position may use (positions 0..i)
}
function rowsOf(probs: Float64Array, ids: readonly number[], targets: readonly number[], base?: Float64Array): Row[] {
  return ids.map((id, t) => {
    const p = probs.subarray(t * V, t * V + V);
    const order = Array.from(p, (v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]);
    let change = 0;
    if (base) for (let j = 0; j < V; j++) change = Math.max(change, Math.abs(probs[t * V + j] - base[t * V + j]));
    return {
      input: word(id), target: word(targets[t]), bar: base ? order[0][0] : p[targets[t]],
      top: order.slice(0, 3).map(([v, i]) => ({ word: word(i), p: v })),
      loss: -Math.log(p[targets[t]]), change, uses: t + 1,
    };
  });
}
const mean = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);

// ---------- scenes ----------
export interface Gen { words: string[]; bad: boolean[] }
export interface LeakScene { mode: 'leak'; curves: { masked: Pt[]; free: Pt[] }; prompt: string[]; gens: { masked: Gen; free: Gen }; words: number; duration: number; markers: Marker[] }
export interface MaskScene { mode: 'mask' | 'encdec'; words: string[]; duration: number; markers: Marker[] }
export interface NegInfScene {
  mode: 'neginf'; words: string[]; q: number; layer: number; head: number;
  scaled: number[]; masked: number[]; unmasked: number[]; futureShare: number; duration: number; markers: Marker[];
}
export interface ParallelScene { mode: 'parallel'; words: string[]; rows: Row[]; avg: number; duration: number; markers: Marker[] }
export interface Variant { word: string; rows: Row[]; maxBefore: number; maxFrom: number; freeBefore: number | null }
export interface EditScene { mode: 'edit'; words: string[]; pos: number; variants: Variant[]; duration: number; markers: Marker[] }
export type Scene = LeakScene | MaskScene | NegInfScene | ParallelScene | EditScene;

const ROW_DT = 1, HOLD = 1.5;
export const LEAK = { train: 4, write: 4.5, wordDt: 1 };
export const PAR = { fadeEnd: 3.4, pass: 0.9 };
export const EDIT_SEG = 2.4;
export const NEG = { slide: [3, 4.5], soft: [5.2, 6.8], wrong: [8, 9.5], end: 11 };

const legalWord = (ids: readonly number[], i: number): boolean => {
  try { return allowedNext(ids.slice(0, i)).has(ids[i]); } catch { return false; }
};
function genOf(w: Weights, prompt: number[]): Gen {
  const ids = generate(w, prompt, { stopId: STOP });
  return { words: ids.slice(prompt.length).map(word), bad: ids.map((_, i) => i >= prompt.length && !legalWord(ids, i)).slice(prompt.length) };
}
const legalSentence = (ids: readonly number[]) => ids.every((_, i) => legalWord(ids, i));

/** Replacement words that keep the whole sentence grammatical; same word class first, at most three. */
export function candidates(ids: readonly number[], pos: number): number[] {
  const out: number[] = [];
  for (let id = 0; id < V; id++) {
    if (id === ids[pos]) continue;
    const s = ids.slice(); s[pos] = id;
    if (legalSentence(s)) out.push(id);
  }
  const kind = WORD_CLASS[ids[pos]].kind;
  return [...out.filter((i) => WORD_CLASS[i].kind === kind), ...out.filter((i) => WORD_CLASS[i].kind !== kind)].slice(0, 3);
}

export function buildScene(mode: Mode, models: Models, opts: Options = {}, curves?: { masked: Pt[]; free: Pt[] }): Scene {
  const ids = encodeWords(opts.sentence ?? SENTENCE), T = ids.length, words = ids.map(word);
  if (mode === 'leak') {
    if (!curves || !models.free) throw new Error('leak needs both models and their loss curves');
    const prompt = encodeWords(opts.prompt ?? PROMPT);
    const gens = { masked: genOf(models.causal, prompt), free: genOf(models.free, prompt) };
    const n = Math.max(gens.masked.words.length, gens.free.words.length);
    return { mode, curves, prompt: prompt.map(word), gens, words: n, duration: LEAK.write + n * LEAK.wordDt + HOLD,
      markers: [{ at: 0, label: 'Training' }, { at: LEAK.write, label: 'Writing' }] };
  }
  if (mode === 'mask' || mode === 'encdec') {
    return { mode, words, duration: T * ROW_DT + HOLD, markers: [{ at: 0, label: 'Row by row' }, { at: T * ROW_DT, label: 'Whole grid' }] };
  }
  const trace = forward(models.causal, ids);
  if (mode === 'neginf') {
    const layer = (opts.layer ?? 1) - 1, head = (opts.head ?? 1) - 1, q = opts.query ?? 3;
    const dk = models.causal.config.d / models.causal.config.heads;
    const raw = trace.layers[layer].scores[head].subarray(q * T, q * T + T);
    const scaled = Array.from(raw, (x) => x / Math.sqrt(dk));
    const unmasked = Array.from(softmax(scaled));
    const masked = Array.from(trace.layers[layer].weights[head].subarray(q * T, q * T + T));
    return { mode, words, q, layer: layer + 1, head: head + 1, scaled, masked, unmasked,
      futureShare: unmasked.slice(q + 1).reduce((a, b) => a + b, 0), duration: NEG.end,
      markers: [{ at: 0, label: 'Scores' }, { at: NEG.slide[0], label: 'Mask' }, { at: NEG.soft[0], label: 'Softmax' }, { at: NEG.wrong[0], label: 'Wrong order' }] };
  }
  const targets = [...ids.slice(1), STOP];
  if (mode === 'parallel') {
    const rows = rowsOf(trace.probs, ids, targets);
    return { mode, words, rows, avg: mean(rows.map((r) => r.loss)), duration: PAR.fadeEnd + T * PAR.pass + HOLD,
      markers: [{ at: 0, label: 'One pass' }, { at: 3, label: 'One pass per word' }] };
  }
  // edit
  const pos = Math.min(T - 1, Math.max(1, opts.editAt ?? 4));
  const list = [ids[pos], ...candidates(ids, pos)];
  const baseFree = models.free ? forward(models.free, ids).probs : undefined;
  const variants: Variant[] = list.map((id) => {
    const s = ids.slice(); s[pos] = id;
    const rows = rowsOf(forward(models.causal, s).probs, s, targets, trace.probs);
    const freeProbs = models.free && baseFree ? forward(models.free, s).probs : undefined;
    const freeRows = freeProbs && baseFree ? rowsOf(freeProbs, s, targets, baseFree) : undefined;
    return {
      word: word(id), rows,
      maxBefore: Math.max(0, ...rows.slice(0, pos).map((r) => r.change)),
      maxFrom: Math.max(0, ...rows.slice(pos).map((r) => r.change)),
      freeBefore: freeRows ? Math.max(0, ...freeRows.slice(0, pos).map((r) => r.change)) : null,
    };
  });
  return { mode: 'edit', words, pos, variants, duration: variants.length * EDIT_SEG,
    markers: variants.map((v, i) => ({ at: i * EDIT_SEG, label: i ? v.word : `${v.word} (original)` })) };
}

export const durationOf = (s: Scene) => s.duration;

// ---------- frames ----------
export interface LeakFrame { kind: 'leak'; phase: 'train' | 'write'; k: number; step: number; masked: number; free: number; shown: { masked: number; free: number }; status: string }
export interface MaskFrame { kind: 'mask'; row: number; alpha: number; count: number; total: number; done: boolean; status: string }
export interface EncDecFrame { kind: 'encdec'; row: number; alpha: number; enc: number; dec: number; total: number; done: boolean; status: string }
export interface NegInfFrame {
  kind: 'neginf'; phase: 'scores' | 'mask' | 'softmax' | 'wrong';
  scoreGrow: number; slide: number; inf: number; // bars grow, future bars slide away, then "-inf" appears
  weights: number[]; labels: number[]; ghost: number; futureGhost: number; futureH: number; showWeights: number; sum: number; status: string;
}
export interface ParallelFrame { kind: 'parallel'; phase: 'one' | 'seq'; alpha: number[]; count: number; avg: number; passes: number; active: number; status: string }
export interface EditFrame { kind: 'edit'; variant: number; flash: number; status: string }
export type Frame = LeakFrame | MaskFrame | EncDecFrame | NegInfFrame | ParallelFrame | EditFrame;

export function frameAt(t: number, s: Scene): Frame {
  t = Math.min(Math.max(t, 0), s.duration);
  if (s.mode === 'leak') {
    const a = s.curves.masked, b = s.curves.free, n = Math.min(a.length, b.length);
    const k = Math.min(n - 1, Math.floor(ease(ramp(t, 0, LEAK.train)) * (n - 1) + 1e-9));
    const phase = t < LEAK.write ? 'train' : 'write';
    const shown = (g: Gen) => phase === 'write' ? Math.min(g.words.length, Math.floor((t - LEAK.write) / LEAK.wordDt) + 1) : 0;
    return { kind: 'leak', phase, k, step: a[k].step, masked: a[k].loss, free: b[k].loss,
      shown: { masked: shown(s.gens.masked), free: shown(s.gens.free) },
      status: phase === 'train' ? `Update ${a[k].step}: loss ${a[k].loss.toFixed(3)} with the mask, ${fmtLoss(b[k].loss)} without it.` : 'Both models write the rest of the sentence one word at a time.' };
  }
  if (s.mode === 'mask' || s.mode === 'encdec') {
    const T = s.words.length, done = t >= T * ROW_DT, row = Math.min(T - 1, Math.floor(t / ROW_DT));
    const alpha = done ? 1 : ease(ramp(t - row * ROW_DT, 0, 0.35));
    if (s.mode === 'mask') {
      const count = ((row + 1) * (row + 2)) / 2;
      return { kind: 'mask', row, alpha, count, total: T * T, done, status: `${s.words[row]} may look at ${row + 1} of ${T} tokens. Allowed cells so far: ${count} of ${T * T}.` };
    }
    return { kind: 'encdec', row, alpha, enc: T, dec: row + 1, total: T, done,
      status: `For ${s.words[row]}: the encoder sees ${T} of ${T} tokens, the decoder ${row + 1} of ${T}.` };
  }
  if (s.mode === 'neginf') {
    const q = s.q, T = s.words.length;
    const slide = ease(ramp(t, NEG.slide[0], NEG.slide[1])), inf = ramp(t, NEG.slide[1] - 0.3, NEG.slide[1] + 0.3);
    const grow = ease(ramp(t, NEG.soft[0], NEG.soft[1])), wrong = ease(ramp(t, NEG.wrong[0], NEG.wrong[1]));
    const weights = s.masked.map((m, i) => (i > q ? 0 : (m + (s.unmasked[i] - m) * wrong) * grow));
    const phase = t >= NEG.wrong[0] ? 'wrong' : t >= NEG.soft[0] ? 'softmax' : t >= NEG.slide[0] ? 'mask' : 'scores';
    // Labels and the sum always show the real state the current segment ends in; only bar heights tween.
    const target = s.masked.map((m, i) => (i > q ? 0 : phase === 'wrong' ? s.unmasked[i] : m));
    const sum = target.reduce((a, b) => a + b, 0);
    const status = phase === 'scores' ? `Raw scores of ${s.words[q]} for all ${T} words, including the ${T - q - 1} that come after it.`
      : phase === 'mask' ? `The ${T - q - 1} future scores are set to minus infinity.`
      : phase === 'softmax' ? `Softmax over what is left: the weights sum to ${sum.toFixed(3)}.`
      : `Wrong order: zeroing the future after the softmax leaves a row that sums to ${sum.toFixed(3)}.`;
    return { kind: 'neginf', phase, scoreGrow: ease(ramp(t, 0.2, 1.8)), slide, inf, weights, labels: target, ghost: ramp(t, NEG.soft[0], NEG.soft[0] + 1) * (1 - wrong), futureGhost: ramp(t, NEG.soft[0], NEG.soft[0] + 1), futureH: 1 - wrong, showWeights: phase === 'wrong' ? 1 : grow, sum, status };
  }
  if (s.mode === 'parallel') {
    const T = s.rows.length, inSeq = t >= 3;
    if (!inSeq) {
      const a = ease(ramp(t, 0.3, 1.2)), count = ease(ramp(t, 0.6, 1.8));
      return { kind: 'parallel', phase: 'one', alpha: s.rows.map(() => a), count, avg: s.avg * count, passes: 1, active: -1,
        status: a > 0 ? `One forward pass produced all ${T} predictions at once.` : 'One sentence goes in.' };
    }
    const local = t - PAR.fadeEnd, pass = local < 0 ? 0 : Math.min(T, Math.floor(local / PAR.pass) + 1);
    const fade = 1 - ease(ramp(t, 3, PAR.fadeEnd));
    const alpha = s.rows.map((_, i) => (pass === 0 ? fade : i < pass - 1 ? 1 : i === pass - 1 ? ease(ramp(local - i * PAR.pass, 0, 0.35)) : 0));
    return { kind: 'parallel', phase: 'seq', alpha, count: 1, avg: pass ? mean(s.rows.slice(0, pass).map((r) => r.loss)) : 0, passes: Math.max(pass, 1), active: pass - 1,
      status: pass ? `Pass ${pass} of ${T}: predicting ${s.rows[pass - 1].target} from the ${pass} token${pass === 1 ? '' : 's'} read so far.` : 'Now the same job, one new word per pass.' };
  }
  if (s.mode !== 'edit') throw new Error('unreachable');
  const variant = Math.min(s.variants.length - 1, Math.floor(t / EDIT_SEG)), local = t - variant * EDIT_SEG;
  const v = s.variants[variant];
  return { kind: 'edit', variant, flash: variant === 0 ? 0 : 1 - ease(ramp(local, 0.2, 1.4)),
    status: variant === 0 ? `Original: ${s.words.join(' ')}.` : `${s.variants[0].word} replaced by ${v.word}.` };
}

export const fmtLoss = (x: number) => (x < 0.01 ? x.toPrecision(2) : x.toFixed(3));
export const fmtChange = (x: number) => (x === 0 ? '0' : x < 0.0005 ? '<0.001' : x.toFixed(3));

// ---------- geometry shared by the SVG ----------
export const layoutOf = (mode: Mode, narrow: boolean): { w: number; h: number } => {
  switch (mode) {
    case 'leak': return narrow ? { w: 288, h: 480 } : { w: 720, h: 330 };
    case 'mask': return narrow ? { w: 288, h: 300 } : { w: 640, h: 330 };
    case 'encdec': return narrow ? { w: 288, h: 540 } : { w: 640, h: 336 };
    case 'neginf': return narrow ? { w: 288, h: 340 } : { w: 720, h: 390 };
    default: return narrow ? { w: 288, h: 480 } : { w: 720, h: 400 };
  }
};
