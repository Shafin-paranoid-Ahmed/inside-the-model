/** PromptModel: the tiny causal transformer the prompting and tool-use lessons run for real.
 * Pure and DOM-free: shared by scripts/train-models.ts (job `prompt-model`), the unit tests and the widgets.
 * Every number in plans/day-14.md §6.1 is reproduced by the experiment functions below on the shipped weights.
 *
 * Task language: 3-letter strings over a..h, written `abc>cba;` (input, `>`, answer, `;`).
 *   rev = reverse, shift = each letter +1 (h wraps to a), sort = ascending, copy, sr = shift then reverse.
 *   sr also has a scratchpad form with the intermediate written out: `abc>bcd>dcb;`.
 *   An optional task tag token ([rev] [shift] [sort] [copy] [sr]) starts the prompt; 0..8 examples precede the query.
 * Tool language: `[tools] 37*48>CALL calc(37*48) RESULT 1776 >1776;` (the program writes `RESULT 1776`), and
 *   direct lines without the header: `37*48>1776;`.
 * `=` and `|` have their own ids and never occur in training (format-sensitivity step); any other symbol is `<unk>`.
 */
import { createRng } from '../math/rng';
import { decodeStep, forward, prefill, softmax, type Weights } from '../transformer/model';

export const LETTERS = 'abcdefgh';
export const TASKS = ['rev', 'shift', 'sort', 'copy', 'sr'] as const;
export type Task = typeof TASKS[number];
export const TAG: Record<Task, string> = { rev: '[rev]', shift: '[shift]', sort: '[sort]', copy: '[copy]', sr: '[sr]' };
export const VOCAB: readonly string[] = [
  ...LETTERS, ...'0123456789', '>', ';', '*', ...TASKS.map((t) => TAG[t]), '[tools]', 'CALL', 'calc(', ')', 'RESULT', '=', '|', '<unk>',
];
const id = (token: string) => VOCAB.indexOf(token);
export const ID = {
  arrow: id('>'), semi: id(';'), times: id('*'), tools: id('[tools]'), call: id('CALL'), calc: id('calc('), close: id(')'),
  result: id('RESULT'), equals: id('='), bar: id('|'), unk: id('<unk>'),
} as const;
export const LETTER_IDS = Array.from({ length: 8 }, (_, i) => i);
const DIGIT0 = id('0');
const isDigit = (t: number) => t >= DIGIT0 && t < DIGIT0 + 10;
const WORDS = new Set(['CALL', 'RESULT', '[tools]', ...TASKS.map((t) => TAG[t])]);
const MULTI = VOCAB.filter((t) => t.length > 1 && t !== '<unk>').sort((a, b) => b.length - a.length);

/** Text to ids; whitespace is ignored, multi-character tokens match first, unknown characters become <unk>. */
export function encode(text: string): number[] {
  const out: number[] = [];
  for (let i = 0; i < text.length;) {
    if (/\s/.test(text[i])) { i++; continue; }
    const word = MULTI.find((t) => text.startsWith(t, i));
    if (word) { out.push(id(word)); i += word.length; continue; }
    const k = VOCAB.indexOf(text[i]);
    out.push(k < 0 || text[i] === '<' ? ID.unk : k); i++;
  }
  return out;
}
/** Ids to display text; a space separates word tokens (CALL, RESULT, tags) from their neighbours. */
export const decode = (ids: readonly number[]) =>
  ids.map((t, i) => (i && (WORDS.has(VOCAB[t]) || WORDS.has(VOCAB[ids[i - 1]])) ? ' ' : '') + VOCAB[t]).join('');

// --- Tasks -----------------------------------------------------------------------------------------
const shiftLetter = (c: string) => LETTERS[(LETTERS.indexOf(c) + 1) % 8];
export const TASK_FN: Record<Task, (s: string) => string> = {
  rev: (s) => [...s].reverse().join(''),
  shift: (s) => [...s].map(shiftLetter).join(''),
  sort: (s) => [...s].sort().join(''),
  copy: (s) => s,
  sr: (s) => [...s].map(shiftLetter).reverse().join(''),
};
/** The 336 three-letter strings with distinct letters, in alphabetical order (the eval pool). */
export const POOL: readonly string[] = (() => {
  const out: string[] = [];
  for (const a of LETTERS) for (const b of LETTERS) for (const c of LETTERS) if (a !== b && b !== c && a !== c) out.push(a + b + c);
  return out;
})();
/** n distinct pool strings, seeded (partial Fisher-Yates), skipping `exclude`. */
export function drawStrings(n: number, seed: number, exclude: readonly string[] = []): string[] {
  const pool = POOL.filter((s) => !exclude.includes(s)), rng = createRng(seed);
  for (let i = 0; i < n; i++) { const j = i + Math.floor(rng.next() * (pool.length - i)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
  return pool.slice(0, n);
}
/** The 50 test strings every letter step uses, and 8 example inputs disjoint from them (examples use the first k). */
export const TEST = drawStrings(50, 1401);
export const EXAMPLES = drawStrings(8, 1402, TEST);

// --- Prompts --------------------------------------------------------------------------------------
export interface LetterPrompt {
  task: Task;
  query: string;
  examples?: readonly string[]; // example inputs; answers are computed with the task
  tag?: boolean;
  delims?: readonly [string, string]; // default ['>', ';']; ['=', '|'] is the unseen pair
  scratchpad?: boolean; // sr only: write the shifted intermediate first
}
/** The continuation a correct model writes after the query's arrow, without the end delimiter. */
export function answerBody(p: LetterPrompt, input = p.query): string {
  const [arrow] = p.delims ?? ['>', ';'];
  return p.scratchpad && p.task === 'sr' ? TASK_FN.shift(input) + arrow + TASK_FN.sr(input) : TASK_FN[p.task](input);
}
export function promptText(p: LetterPrompt): string {
  const [arrow, end] = p.delims ?? ['>', ';'];
  return (p.tag ? TAG[p.task] + ' ' : '') + (p.examples ?? []).map((x) => x + arrow + answerBody(p, x) + end).join('') + p.query + arrow;
}
export const buildPrompt = (p: LetterPrompt) => encode(promptText(p));
export const toolPrompt = (a: number, b: number, tools: boolean) => encode(`${tools ? '[tools] ' : ''}${a}*${b}>`);

// --- Generation -----------------------------------------------------------------------------------
export interface GenStep { id: number; probs: Float64Array } // probs: what was sampled from (after mask and temperature)
export interface GenOptions {
  temperature?: number; // 0 or undefined = greedy (first largest)
  us?: ArrayLike<number>; // one uniform per sampled token
  mask?: (out: readonly number[]) => Iterable<number>; // allowed ids given the tokens generated so far
  stop?: readonly number[]; // stop after emitting one of these
}
/** Up to n tokens with the KV cache. Masked ids get -Infinity before the temperature softmax (constrained decoding). */
export function run(w: Weights, prompt: readonly number[], n: number, o: GenOptions = {}): { ids: number[]; steps: GenStep[] } {
  const V = w.config.vocab.length, T = o.temperature ?? 0, ids: number[] = [], steps: GenStep[] = [];
  if (T > 0 && (o.us?.length ?? 0) < n) throw new Error('Need one uniform per sampled token');
  const pre = prefill(w, prompt);
  let logits: Float64Array = pre.logits.slice(-V);
  for (let i = 0; i < n && prompt.length + i < w.config.context; i++) {
    const z = Array.from(logits);
    if (o.mask) {
      const allowed = new Set(o.mask(ids));
      for (let k = 0; k < V; k++) if (!allowed.has(k)) z[k] = -Infinity;
    }
    let next = 0;
    for (let k = 1; k < V; k++) if (z[k] > z[next]) next = k;
    const probs = softmax(T > 0 ? z.map((v) => v / T) : z);
    if (T > 0) {
      let sum = 0, u = o.us![i];
      next = -1;
      for (let k = 0; k < V; k++) if (probs[k] > 0) { sum += probs[k]; next = k; if (u < sum) break; }
    }
    ids.push(next); steps.push({ id: next, probs });
    if (o.stop?.includes(next) || i === n - 1 || prompt.length + i + 1 >= w.config.context) break;
    logits = decodeStep(w, pre.cache, next);
  }
  return { ids, steps };
}
/** Next-token distribution after the prompt (T = 1, no mask). */
export function nextProbs(w: Weights, prompt: readonly number[]): Float64Array {
  const V = w.config.vocab.length;
  return forward(w, prompt).probs.slice((prompt.length - 1) * V, prompt.length * V);
}
export const uniforms = (seed: number, n: number) => { const r = createRng(seed); return Array.from({ length: n }, () => r.next()); };

// --- Scoring --------------------------------------------------------------------------------------
/** Constrained decoding for the structured step: three letters, then `;`. */
export const formatMask = (out: readonly number[]) => (out.length < 3 ? LETTER_IDS : [ID.semi]);
/** Parse check used by the program: exactly three letters, then `;`. */
export const parses = (out: readonly number[]) => out.length >= 4 && out.slice(0, 3).every((t) => t < 8) && out[3] === ID.semi;
export interface LetterResult { prompt: number[]; output: number[]; expected: string; correct: boolean; steps: GenStep[] }
/** Greedy; correct when the output starts with the expected body (letters, and for scratchpads the intermediate and arrow). */
export function runLetter(w: Weights, p: LetterPrompt): LetterResult {
  const prompt = buildPrompt(p), expected = answerBody(p), body = encode(expected);
  const { ids, steps } = run(w, prompt, body.length + 1);
  return { prompt, output: ids, expected, correct: body.every((t, i) => ids[i] === t), steps };
}
export const accuracy = (rs: readonly { correct: boolean }[]) => rs.filter((r) => r.correct).length / rs.length;
/** Accuracy of one prompt template over query strings (default: the 50 test strings). */
export function evalPrompt(w: Weights, p: Omit<LetterPrompt, 'query'>, queries: readonly string[] = TEST) {
  const items = queries.map((query) => runLetter(w, { ...p, query }));
  return { accuracy: accuracy(items), items };
}
export const mean = (xs: readonly number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;
export const median = (xs: readonly number[]) => { const s = [...xs].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
export function permutations<T>(xs: readonly T[]): T[][] {
  if (xs.length <= 1) return [[...xs]];
  return xs.flatMap((x, i) => permutations([...xs.slice(0, i), ...xs.slice(i + 1)]).map((rest) => [x, ...rest]));
}

// --- Tool language ---------------------------------------------------------------------------------
export const digits = (n: number) => encode(String(n));
export type CallParse = { ok: true; a: number; b: number; expression: string; length: number } | { ok: false; error: string };
/** The program's validator for the calc tool. Schema: {"expression": string matching ^[0-9]{1,4}\*[0-9]{1,4}$}. */
export function parseCall(out: readonly number[]): CallParse {
  if (out[0] !== ID.call) return { ok: false, error: 'no tool call: the model answered directly' };
  if (out[1] !== ID.calc) return { ok: false, error: 'unknown tool: expected calc(' };
  const close = out.indexOf(ID.close, 2);
  if (close < 0) return { ok: false, error: 'call not closed: missing )' };
  const expression = out.slice(2, close).map((t) => VOCAB[t]).join('');
  const m = /^([0-9]{1,4})\*([0-9]{1,4})$/.exec(expression);
  if (!m) return { ok: false, error: `invalid expression "${expression}": expected digits*digits` };
  return { ok: true, a: Number(m[1]), b: Number(m[2]), expression, length: close + 1 };
}
/** Digits between `>` and `;` of a generated answer, or null. */
export function parseAnswer(out: readonly number[]): number | null {
  const start = out[0] === ID.arrow ? 1 : 0, end = out.indexOf(ID.semi, start);
  const ds = out.slice(start, end < 0 ? undefined : end);
  return end > start && ds.every(isDigit) ? Number(ds.map((t) => VOCAB[t]).join('')) : null;
}
/** 2-digit × 2-digit questions, seeded. */
export function mulQuestions(n = 50, seed = 1403): [number, number][] {
  const r = createRng(seed);
  return Array.from({ length: n }, () => [10 + Math.floor(r.next() * 90), 10 + Math.floor(r.next() * 90)]);
}
export const QUESTIONS = mulQuestions();
/** Direct answer without the tools header: greedy until `;`. */
export function directAnswer(w: Weights, a: number, b: number) {
  const { ids, steps } = run(w, toolPrompt(a, b, false), 6, { stop: [ID.semi] });
  const answer = parseAnswer(ids);
  return { output: ids, steps, answer, correct: answer === a * b };
}
export interface Attempt { output: number[]; parse: CallParse }
export interface ToolRun { attempts: Attempt[]; context: number[]; final: number[]; answer: number | null; correct: boolean }
/** The loop: sample a call (greedy at temperature 0); the validator rejects bad calls and the program resamples, up to
 * `retries` times (the rejected attempt is discarded: this model was not trained to read error messages). On a valid
 * call the program runs calc, appends `RESULT n`, and the model writes the final answer greedily. */
export function toolRun(w: Weights, a: number, b: number, o: { temperature?: number; seed?: number; retries?: number } = {}): ToolRun {
  const prompt = toolPrompt(a, b, true), T = o.temperature ?? 0, attempts: Attempt[] = [];
  const rng = createRng(o.seed ?? 0);
  for (let k = 0; k <= (o.retries ?? 0); k++) {
    const us = Array.from({ length: 10 }, () => rng.next());
    const { ids } = run(w, prompt, 10, { temperature: T, us, stop: [ID.close, ID.semi] });
    const parse = parseCall(ids);
    attempts.push({ output: ids, parse });
    if (parse.ok) {
      const context = [...prompt, ...ids.slice(0, parse.length), ID.result, ...digits(parse.a * parse.b)];
      const final = run(w, context, 7, { stop: [ID.semi] }).ids, answer = parseAnswer(final);
      return { attempts, context, final, answer, correct: answer === a * b };
    }
  }
  return { attempts, context: prompt, final: [], answer: null, correct: false };
}
/** Teacher-forced full tool sequence for one question (what the attention step reads). */
export const toolSequence = (a: number, b: number) =>
  encode(`[tools] ${a}*${b}>CALL calc(${a}*${b}) RESULT ${a * b} >${a * b};`);
/** Attention from the positions that predict answer digits (the `>` after the result, then each answer digit but the
 * last) onto the result digits, per head (index layer * heads + head), and summed over the result span. */
export function resultAttention(w: Weights, a: number, b: number): number[] {
  const ids = toolSequence(a, b), input = ids.slice(0, -1), f = forward(w, input), T = input.length;
  const r = ids.indexOf(ID.result), n = String(a * b).length, src = Array.from({ length: n + 1 }, (_, i) => r + n + 1 + i).filter((t) => t < T);
  return f.layers.flatMap((l) => l.weights.map((aw) => mean(src.map((t) => {
    let s = 0; for (let k = r + 1; k <= r + n; k++) s += aw[t * T + k]; return s;
  }))));
}

// --- Experiments: one per lesson step; summaries are what the lessons quote --------------------------
/** Prompting step 1: reverse, 4 untagged examples, trained `>`/`;` vs unseen `=`/`|`. */
export function formatExperiment(w: Weights) {
  const base = { task: 'rev' as const, examples: EXAMPLES.slice(0, 4) };
  return { trained: evalPrompt(w, base), unseen: evalPrompt(w, { ...base, delims: ['=', '|'] }) };
}
/** Prompting step 2: 0 examples, untagged vs tagged. p = probability of the correct first letter; spread = mean
 * probability on the first letter of each task's answer for the untagged prompt (letters can coincide across tasks). */
export function specificExperiment(w: Weights, task: Task = 'rev') {
  const side = (tag: boolean) => {
    const p = TEST.map((q) => nextProbs(w, buildPrompt({ task, query: q, tag }))[LETTERS.indexOf(TASK_FN[task](q)[0])]);
    return { meanP: mean(p), ...evalPrompt(w, { task, tag }) };
  };
  const spread = Object.fromEntries(TASKS.map((t) => [t, mean(TEST.map((q) => nextProbs(w, buildPrompt({ task, query: q }))[LETTERS.indexOf(TASK_FN[t](q)[0])]))]));
  return { untagged: side(false), tagged: side(true), spread };
}
export const SHOTS = [0, 1, 2, 4, 8] as const;
/** Prompting step 3: untagged accuracy with the first k of EXAMPLES. */
export const shotsExperiment = (w: Weights, task: Task, ks: readonly number[] = SHOTS) =>
  ks.map((k) => ({ k, accuracy: evalPrompt(w, { task, examples: EXAMPLES.slice(0, k) }).accuracy }));
/** Prompting step 4: sort, the same 4 examples in all 24 orders; then a set whose inputs are all already sorted. */
export const ORDER_SET = EXAMPLES.slice(0, 4);
export const QUIRKY_SET = ['abd', 'ceh', 'bfg', 'adh'];
export function orderExperiment(w: Weights, set: readonly string[] = ORDER_SET, quirky: readonly string[] = QUIRKY_SET) {
  const run24 = (s: readonly string[]) => permutations(s).map((examples) => ({ examples, accuracy: evalPrompt(w, { task: 'sort', examples }).accuracy }));
  const orders = run24(set), q = run24(quirky), acc = orders.map((o) => o.accuracy), qa = q.map((o) => o.accuracy);
  return { orders, quirky: q, min: Math.min(...acc), median: median(acc), max: Math.max(...acc), quirkyMin: Math.min(...qa), quirkyMedian: median(qa), quirkyMax: Math.max(...qa) };
}
/** Prompting step 5: shift-then-reverse, direct vs scratchpad, with the first k examples in the matching format.
 * firstP = mean probability of the first letter the format requires (direct: the answer; scratchpad: the intermediate),
 * finalP = mean probability of the first final-answer letter given a correct prefix (teacher-forced). */
export function stepsExperiment(w: Weights, k = 4, tag = false) {
  const side = (scratchpad: boolean) => {
    const p = { task: 'sr' as const, examples: EXAMPLES.slice(0, k), tag, scratchpad };
    const r = evalPrompt(w, p);
    const firstP = mean(TEST.map((query) => nextProbs(w, buildPrompt({ ...p, query }))[LETTERS.indexOf(answerBody({ ...p, query })[0])]));
    const finalP = mean(TEST.map((query) => {
      const body = encode(answerBody({ ...p, query })), prefix = [...buildPrompt({ ...p, query }), ...body.slice(0, body.length - 3)];
      return nextProbs(w, prefix)[body[body.length - 3]];
    }));
    return { ...r, firstP, finalP };
  };
  return { direct: side(false), scratchpad: side(true) };
}
/** Prompting step 6: one sampled output per test string, unconstrained vs constrained, same uniforms per string. */
export const STRUCTURED_PROMPT: Omit<LetterPrompt, 'query'> = { task: 'rev' };
export function structuredExperiment(w: Weights, temperature = 1.2, seed = 1404, p: Omit<LetterPrompt, 'query'> = STRUCTURED_PROMPT) {
  const side = (constrained: boolean) => {
    const items = TEST.map((query, i) => {
      const { ids, steps } = run(w, buildPrompt({ ...p, query }), 4, { temperature, us: uniforms(seed + i, 4), mask: constrained ? formatMask : undefined });
      const ok = parses(ids);
      return { query, output: ids, steps, parses: ok, correct: ok && decode(ids.slice(0, 3)) === TASK_FN[p.task](query) };
    });
    const ok = items.filter((r) => r.parses);
    return { items, parseable: ok.length, correctAmongParseable: ok.filter((r) => r.correct).length };
  };
  return { unconstrained: side(false), constrained: side(true) };
}
/** Prompting step 7: sort, A = 2 examples vs B = 6 examples, per-item correctness over the 336 pool; then `draws` seeded
 * test sets (without replacement) of each size. worse = share of sets where B's accuracy is below A's. */
export function evalExperiment(w: Weights, sizes: readonly number[] = [5, 20, 50], draws = 200, seed = 1405) {
  const A = evalPrompt(w, { task: 'sort', examples: EXAMPLES.slice(0, 2) }, POOL).items.map((r) => r.correct);
  const B = evalPrompt(w, { task: 'sort', examples: EXAMPLES.slice(0, 6) }, POOL).items.map((r) => r.correct);
  const acc = (xs: boolean[]) => xs.filter(Boolean).length / xs.length;
  return { accA: acc(A), accB: acc(B), trueDiff: acc(B) - acc(A), ...resample(A, B, sizes, draws, seed) };
}
/** Pure resampling over per-item correctness (the widget can call it without the model). */
export function resample(A: readonly boolean[], B: readonly boolean[], sizes: readonly number[], draws: number, seed: number) {
  const bySize = sizes.map((size) => {
    const rng = createRng(seed + size), idx = A.map((_, i) => i), diffs: number[] = [];
    for (let d = 0; d < draws; d++) {
      for (let i = 0; i < size; i++) { const j = i + Math.floor(rng.next() * (idx.length - i)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
      let s = 0; for (let i = 0; i < size; i++) s += +B[idx[i]] - +A[idx[i]];
      diffs.push(s / size);
    }
    return { size, diffs, worse: diffs.filter((x) => x < 0).length / draws, tie: diffs.filter((x) => x === 0).length / draws };
  });
  return { bySize };
}
/** Tool-use steps: direct accuracy, p(CALL) at the first position with/without the header, greedy tool accuracy,
 * sampled calls with retries, and attention onto the result. Each question uses seed 1406 + its index. */
export function toolExperiment(w: Weights, temperature = 1, maxRetries = 3) {
  const direct = QUESTIONS.map(([a, b]) => directAnswer(w, a, b));
  const pCall = (tools: boolean) => mean(QUESTIONS.map(([a, b]) => nextProbs(w, toolPrompt(a, b, tools))[ID.call]));
  const greedy = QUESTIONS.map(([a, b]) => toolRun(w, a, b));
  const sampled = QUESTIONS.map(([a, b], i) => toolRun(w, a, b, { temperature, seed: 1406 + i, retries: maxRetries }));
  const successWith = Array.from({ length: maxRetries + 1 }, (_, r) => sampled.filter((s) => s.correct && s.attempts.length <= r + 1).length / sampled.length);
  const att = QUESTIONS.map(([a, b]) => resultAttention(w, a, b)), heads = att[0].map((_, h) => mean(att.map((x) => x[h])));
  return {
    directAccuracy: accuracy(direct), direct,
    pCallWithTools: pCall(true), pCallWithoutTools: pCall(false),
    toolAccuracy: accuracy(greedy), greedy,
    firstAttemptRejected: sampled.filter((s) => !s.attempts[0].parse.ok).length / sampled.length, successWith, sampled,
    resultAttentionPerHead: heads,
  };
}
