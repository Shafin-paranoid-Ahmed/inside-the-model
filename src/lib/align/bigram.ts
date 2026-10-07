/**
 * The SFT "mask" step's second tiny model: a word-level bigram language model (logits W[prev][next]) trained on 16
 * "User: ... Assistant: ..." transcripts, either on every token or on the response tokens only (loss masking), and
 * scored on the response tokens of 4 held-out transcripts. Pure TS, Adam, deterministic (no sampling).
 */
import { adam } from '../nn/optim';

const PAIRS: [string, string][] = [
  ['what is two plus two', 'two plus two is four'],
  ['what is three plus one', 'three plus one is four'],
  ['what is two plus three', 'two plus three is five'],
  ['what is one plus five', 'one plus five is six'],
  ['what is four plus two', 'four plus two is six'],
  ['what is the capital of france', 'the capital of france is paris'],
  ['what is the capital of spain', 'the capital of spain is madrid'],
  ['what is the capital of italy', 'the capital of italy is rome'],
  ['how many legs does a spider have', 'a spider has eight legs'],
  ['how many legs does a cat have', 'a cat has four legs'],
  ['how many legs does a bird have', 'a bird has two legs'],
  ['how many legs does a dog have', 'a dog has four legs'],
  ['what colour is the sky', 'the sky is blue'],
  ['what colour is the grass', 'the grass is green'],
  ['what colour is the sun', 'the sun is yellow'],
  ['what colour is a lemon', 'a lemon is yellow'],
];
const HELD: [string, string][] = [
  ['what is one plus three', 'one plus three is four'],
  ['what is three plus two', 'three plus two is five'],
  ['what is two plus one', 'two plus one is three'],
  ['what colour is a sun', 'a sun is yellow'],
];
export const END = '<end>';
export interface Transcript {
  tokens: string[]; ids: Int32Array;
  /** Index of the first response token (the token after "assistant:"); tokens from there to the end are the response. */
  respStart: number;
}
const toTokens = ([q, a]: [string, string]) => ['user:', ...q.split(' '), 'assistant:', ...a.split(' '), END];
export const VOCAB: string[] = [...new Set([...PAIRS, ...HELD].flatMap(toTokens))];
export const V = VOCAB.length;
const make = (p: [string, string]): Transcript => {
  const tokens = toTokens(p);
  return { tokens, ids: Int32Array.from(tokens, (t) => VOCAB.indexOf(t)), respStart: tokens.indexOf('assistant:') + 1 };
};
export const TRAIN_TRANSCRIPTS = PAIRS.map(make);
export const HELD_TRANSCRIPTS = HELD.map(make);

/** Positions whose token counts toward the loss: every token after the first, or only the response tokens. */
export const counted = (t: Transcript, mask: 'all' | 'response'): number[] =>
  Array.from({ length: t.ids.length - 1 }, (_, i) => i + 1).filter((i) => mask === 'all' || i >= t.respStart);

/** Per-token loss -log p(token_i | token_{i-1}) for every position i >= 1 (index 0 unused, NaN). */
export function tokenLosses(W: Float64Array, t: Transcript): Float64Array {
  const out = new Float64Array(t.ids.length).fill(NaN);
  for (let i = 1; i < t.ids.length; i++) {
    const r = t.ids[i - 1] * V;
    let m = -Infinity;
    for (let j = 0; j < V; j++) m = Math.max(m, W[r + j]);
    let s = 0;
    for (let j = 0; j < V; j++) s += Math.exp(W[r + j] - m);
    out[i] = -(W[r + t.ids[i]] - m - Math.log(s));
  }
  return out;
}
/** Mean loss over the counted positions of the transcripts, and its gradient (into grad). */
export function lossGrad(W: Float64Array, ts: Transcript[], mask: 'all' | 'response', grad?: Float64Array): number {
  grad?.fill(0);
  let loss = 0, n = 0;
  const p = new Float64Array(V);
  for (const t of ts) for (const i of counted(t, mask)) {
    const r = t.ids[i - 1] * V;
    let m = -Infinity, s = 0;
    for (let j = 0; j < V; j++) m = Math.max(m, W[r + j]);
    for (let j = 0; j < V; j++) { p[j] = Math.exp(W[r + j] - m); s += p[j]; }
    loss -= Math.log(p[t.ids[i]] / s); n++;
    if (grad) { for (let j = 0; j < V; j++) grad[r + j] += p[j] / s; grad[r + t.ids[i]] -= 1; }
  }
  if (grad) for (let j = 0; j < grad.length; j++) grad[j] /= n;
  return loss / n;
}
export interface BigramRun {
  W: Float64Array;
  /** Per step (index 0 = untrained): the training objective and the held-out response-token loss. */
  train: Float64Array; heldResponse: Float64Array;
}
/** Full-batch Adam from all-zero logits on the 16 transcripts with the given loss mask. */
export function trainBigram(mask: 'all' | 'response', o: { steps?: number; lr?: number } = {}): BigramRun {
  const steps = o.steps ?? 300, opt = adam(o.lr ?? 0.1), W = new Float64Array(V * V), g = new Float64Array(V * V);
  const train = new Float64Array(steps + 1), held = new Float64Array(steps + 1);
  for (let s = 0; s <= steps; s++) {
    held[s] = lossGrad(W, HELD_TRANSCRIPTS, 'response');
    train[s] = lossGrad(W, TRAIN_TRANSCRIPTS, mask, g);
    if (s < steps) opt.step(W as unknown as number[], g as unknown as number[]);
  }
  return { W, train, heldResponse: held };
}
