/**
 * RL with verifiable rewards on addition problems with 1 to 4 digits. The policy is tabular per digit count over 10
 * actions = (k = 0..4 worked columns) x (format: one answer | a list of guesses). A fixed solver writes k columns with
 * carries and adds the rest without carrying (the toy's assumption: each column written out avoids a carry slip);
 * the response TEXT is generated and the verifier parses it: 'strict' (the last line is exactly "Answer: N") or
 * 'lenient' (N appears anywhere). GRPO: G samples per problem, advantage (r - mean) / std per group (0 when std = 0),
 * a clipped policy-gradient step (Adam), optional KL to the start policy. Pure TS, seeded.
 */
import { createRng, type Rng } from '../math/rng';
import { adam } from '../nn/optim';
import { clippedSurrogate } from '../rl/surrogate';

export const MAXK = 4, NACT = 10, DIGITS = [1, 2, 3, 4];
/** Action index = format * 5 + k (format 0 = one answer, 1 = list of guesses). */
export const actionOf = (k: number, list: boolean): number => (list ? 5 : 0) + k;
export const kOf = (a: number): number => a % 5;
export const isList = (a: number): boolean => a >= 5;
export const ACTION_LABELS = Array.from({ length: NACT }, (_, a) => `${kOf(a)} column${kOf(a) === 1 ? '' : 's'}, ${isList(a) ? 'guess list' : 'one answer'}`);
/** Row of the policy table: digit counts 1..4 are rows 0..3; row 4 is the unverifiable "open" prompt (domains variant). */
export const OPEN = 4;
export const OPEN_PROMPTS = ['Write a short poem about the sea.', 'Describe your ideal weekend.'];

export interface Problem { a: number; b: number; digits: number }
export function makeProblem(digits: number, rng: Rng): Problem {
  const lo = 10 ** (digits - 1), hi = 10 ** digits - 1, r = () => lo + Math.floor(rng.next() * (hi - lo + 1));
  return { a: r(), b: r(), digits };
}
const digitAt = (n: number, i: number) => Math.floor(n / 10 ** i) % 10;

/** The solver: k worked columns (from the right, with carries), then the rest added without carrying. */
export function solve(p: Problem, k: number): { lines: string[]; answer: number } {
  const d = p.digits, kk = Math.min(k, d), lines: string[] = [];
  let carry = 0, answer = 0;
  for (let i = 0; i < d; i++) {
    const x = digitAt(p.a, i), y = digitAt(p.b, i);
    if (i < kk) {
      const s = x + y + carry;
      lines.push(`${x} + ${y}${carry ? ' + 1' : ''} = ${s}, write ${s % 10}${s >= 10 ? ', carry 1' : ''}`);
      answer += (s % 10) * 10 ** i;
      carry = s >= 10 ? 1 : 0;
    } else answer += ((x + y) % 10) * 10 ** i; // no carrying: the slip
  }
  if (kk === d && carry) answer += 10 ** d;
  return { lines, answer };
}
/** The response text for an action: worked lines, then "Answer: N" or a list of four guesses on the last line. */
export function respond(p: Problem, action: number): string {
  const { lines, answer } = solve(p, kOf(action));
  if (!isList(action)) return [...lines, `Answer: ${answer}`].join('\n');
  // guesses cover a dropped carry into the lowest one or two unworked columns
  const base = 10 ** Math.max(1, Math.min(kOf(action), p.digits)), g = [answer, answer + base, answer + 10 * base, answer + 11 * base];
  return [...lines, `Answer: ${g.join(' or ')}`].join('\n');
}
export const truth = (p: Problem): number => p.a + p.b;
/** Strict: the last line must be exactly "Answer: N". Returns the parsed N or null. */
export function parseStrict(text: string): number | null {
  const last = text.split('\n').pop() ?? '', m = /^Answer: (\d+)$/.exec(last);
  return m ? Number(m[1]) : null;
}
/** Lenient: every whole number anywhere in the text. */
export const parseLenient = (text: string): number[] => (text.match(/\d+/g) ?? []).map(Number);
export type VerifierKind = 'strict' | 'lenient';
export function verify(text: string, answer: number, kind: VerifierKind): boolean {
  return kind === 'strict' ? parseStrict(text) === answer : parseLenient(text).includes(answer);
}
export const words = (text: string): number => text.split(/\s+/).filter(Boolean).length;
export const linesOf = (text: string): number => text.split('\n').length;

/** Fixed evaluation problems (200 per digit count) and every action's pass rate and length on them. */
const EVAL_N = 200;
let table: { strict: Float64Array; lenient: Float64Array; lines: Float64Array; words: Float64Array; passStrict: Uint8Array[]; passLenient: Uint8Array[] } | null = null;
export function evalTable() {
  if (table) return table;
  const rng = createRng(101), t = { strict: new Float64Array(40), lenient: new Float64Array(40), lines: new Float64Array(40), words: new Float64Array(40), passStrict: [] as Uint8Array[], passLenient: [] as Uint8Array[] };
  for (const d of DIGITS) {
    const ps = Array.from({ length: EVAL_N }, () => makeProblem(d, rng)), r = d - 1;
    const s = new Uint8Array(EVAL_N * NACT), l = new Uint8Array(EVAL_N * NACT);
    ps.forEach((p, i) => {
      for (let a = 0; a < NACT; a++) {
        const text = respond(p, a), o = r * NACT + a;
        s[i * NACT + a] = +verify(text, truth(p), 'strict'); l[i * NACT + a] = +verify(text, truth(p), 'lenient');
        t.strict[o] += s[i * NACT + a] / EVAL_N; t.lenient[o] += l[i * NACT + a] / EVAL_N;
        t.lines[o] += linesOf(text) / EVAL_N; t.words[o] += words(text) / EVAL_N;
      }
    });
    t.passStrict.push(s); t.passLenient.push(l);
  }
  return (table = t);
}

/** Softmax of row r of a logits table (NACT per row). */
export function rowProbs(logits: ArrayLike<number>, r: number, out = new Float64Array(NACT)): Float64Array {
  let m = -Infinity, s = 0;
  for (let a = 0; a < NACT; a++) m = Math.max(m, logits[r * NACT + a]);
  for (let a = 0; a < NACT; a++) { out[a] = Math.exp(logits[r * NACT + a] - m); s += out[a]; }
  for (let a = 0; a < NACT; a++) out[a] /= s;
  return out;
}
/** The start policy: mostly answers at once with no working; the guess list is less likely. */
export const START_ONE = [2, 0.5, 0, -0.5, -1], LIST_OFFSET = -1.5;
export function startLogits(rows = 5): Float64Array {
  const l = new Float64Array(rows * NACT);
  for (let r = 0; r < rows; r++) for (let k = 0; k <= MAXK; k++) { l[r * NACT + k] = START_ONE[k]; l[r * NACT + 5 + k] = START_ONE[k] + LIST_OFFSET; }
  return l;
}
/** Exact policy statistics on the evaluation problems for digit count d. */
export function rowStats(logits: ArrayLike<number>, d: number) {
  const t = evalTable(), p = rowProbs(logits, d - 1), r = d - 1;
  let strict = 0, lenient = 0, lines = 0, wds = 0, list = 0, pass8 = 0;
  for (let a = 0; a < NACT; a++) { const o = r * NACT + a; strict += p[a] * t.strict[o]; lenient += p[a] * t.lenient[o]; lines += p[a] * t.lines[o]; wds += p[a] * t.words[o]; if (isList(a)) list += p[a]; }
  // pass@8 (strict): per problem, 1 - (1 - p_problem)^8
  for (let i = 0; i < EVAL_N; i++) { let q = 0; for (let a = 0; a < NACT; a++) q += p[a] * t.passStrict[r][i * NACT + a]; pass8 += (1 - (1 - q) ** 8) / EVAL_N; }
  return { strict, lenient, lines, words: wds, list, pass1: strict, pass8 };
}
/** GRPO advantages of one group: (r - mean) / std (population std), all 0 when std = 0. */
export function groupAdvantages(r: ArrayLike<number>): Float64Array {
  const n = r.length, out = new Float64Array(n);
  let m = 0, v = 0;
  for (let i = 0; i < n; i++) m += r[i] / n;
  for (let i = 0; i < n; i++) v += (r[i] - m) ** 2 / n;
  const sd = Math.sqrt(v);
  if (sd > 1e-12) for (let i = 0; i < n; i++) out[i] = (r[i] - m) / sd;
  return out;
}

export interface Groups {
  /** Per problem: the problem (null for an open prompt), its row, the G actions, strict and lenient rewards, the reward used and advantages. */
  rows: { p: Problem | null; row: number; actions: Int8Array; strict: Uint8Array; lenient: Uint8Array; reward: Float64Array; adv: Float64Array }[];
}
/** Sample G attempts for each problem with the given logits and score them. Problems cycle through the rows `rows`. */
export function sampleGroups(logits: ArrayLike<number>, o: { problems?: number; G?: number; rng: Rng; verifier?: VerifierKind; rows?: number[] }): Groups {
  const n = o.problems ?? 8, G = o.G ?? 8, rows = o.rows ?? [0, 1, 2, 3], kind = o.verifier ?? 'strict', p = new Float64Array(NACT), out: Groups = { rows: [] };
  for (let j = 0; j < n; j++) {
    const row = rows[j % rows.length], prob = row === OPEN ? null : makeProblem(row + 1, o.rng);
    rowProbs(logits, row, p);
    const actions = new Int8Array(G), strict = new Uint8Array(G), lenient = new Uint8Array(G), reward = new Float64Array(G);
    for (let g = 0; g < G; g++) {
      let u = o.rng.next(), a = 0;
      for (; a < NACT - 1; a++) { u -= p[a]; if (u < 0) break; }
      actions[g] = a;
      if (prob) { const text = respond(prob, a), s = truth(prob); strict[g] = +verify(text, s, 'strict'); lenient[g] = +verify(text, s, 'lenient'); }
      reward[g] = kind === 'strict' ? strict[g] : lenient[g]; // open prompts: no verifier, reward 0
    }
    out.rows.push({ p: prob, row, actions, strict, lenient, reward, adv: groupAdvantages(reward) });
  }
  return out;
}

export interface GRPOOpts { verifier?: VerifierKind; iters?: number; G?: number; problems?: number; lr?: number; seed?: number; eps?: number; klBeta?: number; domains?: boolean }
export interface GRPORun {
  /** Logits before each iteration and at the end (iters + 1), rows 0..3 = digits 1..4, row 4 = open prompts. */
  logits: Float64Array[];
  /** Per iteration (iters + 1) and digit count (index it * 4 + d - 1), exact on the evaluation problems. */
  strict: Float64Array; lenient: Float64Array; lines: Float64Array; words: Float64Array; list: Float64Array; pass8: Float64Array;
  /** The groups sampled at each iteration (for showing live samples). */
  groups: Groups[];
  iters: number;
}
/** GRPO training: each iteration samples `problems` problems (cycling digit counts; plus 2 open prompts with `domains`). */
export function trainGRPO(o: GRPOOpts = {}): GRPORun {
  const iters = o.iters ?? 150, G = o.G ?? 8, kind = o.verifier ?? 'strict', eps = o.eps ?? 0.2, klB = o.klBeta ?? 0, rng = createRng(o.seed ?? 5);
  const rowsUsed = o.domains ? [0, 1, 2, 3, OPEN] : [0, 1, 2, 3], nProb = (o.problems ?? 8) + (o.domains ? 2 : 0);
  const th = startLogits(), start = th.slice(), opt = adam(o.lr ?? 0.1), g = new Float64Array(th.length), p = new Float64Array(NACT), q = new Float64Array(NACT);
  const run: GRPORun = { logits: [th.slice()], strict: new Float64Array((iters + 1) * 4), lenient: new Float64Array((iters + 1) * 4), lines: new Float64Array((iters + 1) * 4), words: new Float64Array((iters + 1) * 4), list: new Float64Array((iters + 1) * 4), pass8: new Float64Array((iters + 1) * 4), groups: [], iters };
  const rec = (it: number) => { for (const d of DIGITS) { const s = rowStats(th, d), i = it * 4 + d - 1; run.strict[i] = s.strict; run.lenient[i] = s.lenient; run.lines[i] = s.lines; run.words[i] = s.words; run.list[i] = s.list; run.pass8[i] = s.pass8; } };
  rec(0);
  for (let it = 0; it < iters; it++) {
    const gr = sampleGroups(th, { problems: nProb, G, rng, verifier: kind, rows: o.domains ? [0, 1, 2, 3, 0, 1, 2, 3, OPEN, OPEN].slice(0, nProb) : rowsUsed });
    run.groups.push(gr);
    // one clipped policy-gradient step on the fresh samples (ratio 1, so the clip is inactive): ascend mean A grad log pi
    g.fill(0);
    let n = 0;
    for (const row of gr.rows) {
      rowProbs(th, row.row, p);
      for (let k = 0; k < G; k++) {
        const a = row.actions[k], d = clippedSurrogate(1, row.adv[k], eps).dRho; // rho = 1 on fresh samples
        for (let b = 0; b < NACT; b++) g[row.row * NACT + b] -= d * ((b === a ? 1 : 0) - p[b]);
        n++;
      }
    }
    for (let i = 0; i < g.length; i++) g[i] /= n;
    if (klB) for (const r of rowsUsed) { // + klBeta * grad KL(pi || pi_start), exact per row
      rowProbs(th, r, p); rowProbs(start, r, q);
      let e = 0;
      for (let a = 0; a < NACT; a++) e += p[a] * Math.log(p[a] / q[a]);
      for (let a = 0; a < NACT; a++) g[r * NACT + a] += klB * p[a] * (Math.log(p[a] / q[a]) - e);
    }
    opt.step(th as unknown as number[], g as unknown as number[]);
    run.logits.push(th.slice());
    rec(it + 1);
  }
  return run;
}
