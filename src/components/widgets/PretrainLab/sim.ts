// Pure logic for PretrainLab: no DOM, no fetch. Weights and the training log are passed in.
import { createRng } from '../../../lib/math/rng';
import { forward, prefill, decodeStep, softmax, type KVCache, type Weights } from '../../../lib/transformer';
import { CHARSET, encode, decode } from '../../../lib/lm/charset';
import { shape, pickIndex } from '../../../lib/lm/sample';

export type Mode = 'objective' | 'loss' | 'data' | 'compute' | 'learning' | 'base';
export interface Marker { at: number; label: string }
export const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
export const ease = (x: number) => { const c = clamp01(x); return 1 - (1 - c) ** 3; };
const V = CHARSET.length + 1;
/** Prints a number as `3.15 × 10²³`. */
export const sci = (x: number, digits = 2) => {
  const e = Math.floor(Math.log10(Math.abs(x)) + 1e-12), m = x / 10 ** e;
  const sup = String(e).replace(/-/g, '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]);
  return `${m.toFixed(digits)} × 10${sup}`;
};
/** 1.32e12, compact form for labels inside the figure. */
export const mant = (x: number) => (x / 10 ** Math.floor(Math.log10(x) + 1e-12)).toFixed(2);
export const expo = (x: number) => Math.floor(Math.log10(x) + 1e-12);
export const sciE = (x: number) => { const e = Math.floor(Math.log10(x) + 1e-12); return (x / 10 ** e).toFixed(2) + 'e' + e; };
/** 35.8K, 175B, 1.4T. */
export const eng = (x: number) => { const u = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']].find(([f]) => x >= (f as number) * 0.9995) ?? [1, '']; return +(x / (u[0] as number)).toPrecision(3) + (u[1] as string); };
export const show = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '↵' : c === '�' ? '▯' : c);

/* ------------------------------------------------------------------ objective and loss */
/** A sentence from the held-out part of the corpus (chunk 9 of 50 is validation; checked in the tests). */
export const PASSAGE = 'alice went timidly up to the door, and knocked.';
export interface Scored { text: string; ids: number[]; probs: Float64Array[]; p: number[]; loss: number[]; mean: number; ppl: number }
/** Position i predicts character i + 1 from characters 0..i. */
export function scorePassage(w: Weights, text = PASSAGE): Scored {
  const ids = encode(text), f = forward(w, ids.slice(0, -1));
  const probs = ids.slice(0, -1).map((_, i) => f.probs.slice(i * V, (i + 1) * V));
  const p = probs.map((d, i) => d[ids[i + 1]]), loss = p.map((x) => -Math.log(x));
  const mean = loss.reduce((a, b) => a + b, 0) / loss.length;
  return { text, ids, probs, p, loss, mean, ppl: Math.exp(mean) };
}
export const OBJ_BEAT = 0.7, LOSS_BEAT = 0.32, HOLD = 1.5;
export const objectiveDuration = (s: Scored) => s.p.length * OBJ_BEAT + HOLD;
export const lossDuration = (s: Scored) => s.p.length * LOSS_BEAT + HOLD;
/** Word-group segments of the passage, as timeline markers. */
export const textMarkers = (s: Scored, beat: number): Marker[] => {
  const words = s.text.split(' '), groups = [words.slice(0, 3), words.slice(3, 7), words.slice(7)];
  let pos = 0; const out: Marker[] = [];
  for (const g of groups) { out.push({ at: pos * beat, label: g.join(' ') }); pos += g.join(' ').length + 1; }
  return out;
};
export interface Bar { id: number; ch: string; v: number; p: number; isTrue: boolean }
export interface ObjFrame { k: number; shown: number; bars: Bar[]; pTrue: number; loss: number; next: string }
/** Eight bars at cursor k: the seven likeliest characters plus the true one (or the eighth likeliest). Heights ease from the previous position's value of the same character. */
export function objectiveAt(t: number, s: Scored): ObjFrame {
  const n = s.p.length, k = Math.min(n - 1, Math.floor(t / OBJ_BEAT)), e = ease((t - k * OBJ_BEAT) / (OBJ_BEAT * 0.5)), cur = s.probs[k], tr = s.ids[k + 1];
  const order = Array.from(cur, (_, i) => i).sort((a, b) => cur[b] - cur[a] || a - b);
  const top = order.slice(0, 8);
  if (!top.includes(tr)) top[7] = tr;
  const bars = top.map((id) => {
    const prev = k > 0 ? s.probs[k - 1][id] : 0;
    return { id, ch: show(decode([id])), v: prev + (cur[id] - prev) * e, p: cur[id], isTrue: id === tr };
  });
  const v = bars.find((b) => b.isTrue)!.v;
  return { k, shown: k + 1, bars, pTrue: v, loss: -Math.log(cur[tr]), next: show(decode([tr])) };
}
export interface LossFrame { k: number; heights: number[]; mean: number; ppl: number; tallest: number; p: number; loss: number; next: string }
export function lossAt(t: number, s: Scored): LossFrame {
  const n = s.p.length, k = Math.min(n - 1, Math.floor(t / LOSS_BEAT)), e = ease((t - k * LOSS_BEAT) / (LOSS_BEAT * 0.7));
  const heights = s.loss.map((l, i) => (i < k ? l : i === k ? l * e : 0));
  const mean = heights.slice(0, k + 1).reduce((a, b) => a + b, 0) / (k + 1);
  let tallest = 0; s.loss.forEach((l, i) => { if (l > s.loss[tallest]) tallest = i; });
  return { k, heights, mean, ppl: Math.exp(mean), tallest, p: s.p[k], loss: s.loss[k], next: show(decode([s.ids[k + 1]])) };
}

/* ------------------------------------------------------------------ data */
const A = 'Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it.';
const B = 'It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.';
/** Written for this figure, except the two public-domain passages (Carroll 1865; Austen 1813). */
export const DOCS: { name: string; text: string }[] = [
  { name: 'Alice, opening', text: A },
  { name: 'Austen, opening', text: B },
  { name: 'Mirror site of #1', text: '  ' + A.replace('Alice', 'ALICE').replace('bank, ', 'bank,  ') },
  { name: 'Cookie banner', text: 'We use cookies.\nAccept all\nManage preferences' },
  { name: 'Alice, 1 word changed', text: A.replace('sister on', 'brother on') },
  { name: 'Keyword spam', text: 'buy cheap watches online cheap watches best cheap watches buy watches online cheap watches discount cheap watches free shipping cheap watches' },
  { name: 'Mirror site of #2', text: B.replace('It is', 'IT IS') + ' ' },
  { name: 'Alice, 2 words changed', text: A.replace('tired', 'weary').replace('pictures', 'drawings') },
  { name: 'Gibberish string', text: 'xkqzvjwpltmrnbf dhgcsqwrtzpxvk lmnbvcxzasdfgh jklqwertyuiop zxcvbnmqwerty uiopasdfghjkl plkjhgfdsazxcvb mnbvcxzlkjhgfds qpwoeirutyalsk djfhgmnbvcxz' },
  { name: 'Headline list', text: 'Ten reasons your garden is not growing the way you hoped...\nWhat the neighbours are saying about the new bridge...\nLocal council meets on Tuesday to discuss parking and bins...\nWeather: mild with some rain on the way tomorrow afternoon' },
  { name: 'Hashtag spam', text: '#win #free #money $$$ click here ### best deal #cheap #now #win ### click now #free' },
  { name: 'Note on tides', text: 'Spring tides come a few days after each new and full moon, when the sun and the moon pull along the same line and the sea rises a little higher and falls a little lower than usual.' },
];
export const DOC_BYTES = DOCS.reduce((s, d) => s + new TextEncoder().encode(d.text).length, 0);
export const MIN_WORDS = 10; // Gopher uses 50; the sample documents are short

export const normalise = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
const mix = (x: number) => { x ^= x >>> 16; x = Math.imul(x, 0x85ebca6b); x ^= x >>> 13; x = Math.imul(x, 0xc2b2ae35); return (x ^ (x >>> 16)) >>> 0; };
/** FNV-1a, 32 bit. */
export const hash32 = (s: string) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193); return h >>> 0; };
export const shingles = (text: string, n = 5): Set<number> => {
  const w = normalise(text).split(' '), out = new Set<number>();
  for (let i = 0; i + n <= w.length; i++) out.add(hash32(w.slice(i, i + n).join(' ')));
  return out;
};
export const HASHES = 128;
const SEEDS = (() => { const r = createRng(17); return Array.from({ length: HASHES }, () => (r.next() * 4294967296) >>> 0); })();
/** MinHash signature: for each of 128 seeded hash functions, the smallest hash over the shingles. */
export const minhash = (set: Set<number>): Uint32Array => {
  const sig = new Uint32Array(HASHES).fill(0xffffffff);
  for (const x of set) for (let i = 0; i < HASHES; i++) { const h = mix(x ^ SEEDS[i]); if (h < sig[i]) sig[i] = h; }
  return sig;
};
export const jaccard = (a: Set<number>, b: Set<number>) => { let i = 0; for (const x of a) if (b.has(x)) i++; return i / (a.size + b.size - i); };
export const estimate = (a: Uint32Array, b: Uint32Array) => { let m = 0; for (let i = 0; i < HASHES; i++) if (a[i] === b[i]) m++; return m / HASHES; };

/** Gopher-style rules (Rae et al. 2021, appendix A): the first one that fires is reported. */
export function qualityFail(text: string): string | null {
  const words = text.split(/\s+/).filter(Boolean), lines = text.split('\n').filter((l) => l.trim());
  if (words.length < MIN_WORDS) return `${words.length} words, min ${MIN_WORDS}`;
  const mean = words.reduce((s, x) => s + x.length, 0) / words.length;
  if (mean < 3 || mean > 10) return `mean word ${mean.toFixed(1)} letters`;
  const hash = (text.match(/#/g) ?? []).length / words.length, dots = (text.match(/\.\.\.|…/g) ?? []).length / words.length;
  if (hash > 0.1) return `# marks ${hash.toFixed(2)}/word`;
  if (dots > 0.1) return `... marks ${dots.toFixed(2)}/word`;
  const alpha = words.filter((x) => /[a-z]/i.test(x)).length / words.length;
  if (alpha < 0.8) return `${Math.round(alpha * 100)}% words with letters`;
  const endDots = lines.filter((l) => /(\.\.\.|…)\s*$/.test(l)).length;
  if (endDots / lines.length > 0.3) return `${endDots} of ${lines.length} lines end ...`;
  const lw = words.map((x) => x.toLowerCase().replace(/[^a-z0-9]/g, '')), count = new Map<string, number>();
  for (let i = 0; i + 1 < lw.length; i++) { const g = lw[i] + ' ' + lw[i + 1]; count.set(g, (count.get(g) ?? 0) + 1); }
  let best = 0, bestLen = 0; for (const [g, c] of count) if (c > 1 && c * g.length > best * bestLen) { best = c; bestLen = g.length; }
  const share = (best * bestLen) / lw.join(' ').length;
  if (share > 0.2) return `top phrase ${Math.round(share * 100)}%`;
  return null;
}

export interface Cell { state: 'pass' | 'fail' | 'skip'; note: string }
export interface DocRes { name: string; len: number; cells: Cell[]; out: number }
export interface DataRun { threshold: number; docs: DocRes[]; start: number[][]; gateStart: number[]; duration: number; markers: Marker[]; kept: number; keptChars: number; totalChars: number; pair: { exact: number; est: number }; pair2: { exact: number; est: number } }
export const ROW = 0.4, GAP = 0.7;
export const GATES = ['Exact copies', 'Near copies', 'Quality rules'];
export function analyse(threshold: number): DataRun {
  const n = DOCS.length, norm = DOCS.map((d) => hash32(normalise(d.text)));
  const sh = DOCS.map((d) => shingles(d.text)), sig = sh.map(minhash);
  const out = new Array<number>(n).fill(3), notes: string[][] = DOCS.map(() => ['', '', '']);
  for (let i = 0; i < n; i++) {
    const m = norm.findIndex((h, j) => j < i && h === norm[i] && out[j] >= 1);
    if (m >= 0) { out[i] = 0; notes[i][0] = `copy of #${m + 1}`; }
  }
  for (let i = 0; i < n; i++) {
    if (out[i] < 3) continue;
    let best = 0, bj = -1;
    for (let j = 0; j < i; j++) if (out[j] >= 2) { const e = estimate(sig[i], sig[j]); if (e > best) { best = e; bj = j; } }
    if (best >= threshold) { out[i] = 1; notes[i][1] = `J ${best.toFixed(2)} with #${bj + 1}`; } else notes[i][1] = i ? `max J ${best.toFixed(2)}` : 'first doc';
  }
  for (let i = 0; i < n; i++) {
    if (out[i] < 3) continue;
    const f = qualityFail(DOCS[i].text);
    if (f) { out[i] = 2; notes[i][2] = f; } else notes[i][2] = 'passes';
  }
  const docs: DocRes[] = DOCS.map((d, i) => ({ name: d.name, len: d.text.length, out: out[i], cells: [0, 1, 2].map((g): Cell => ({
    state: g < out[i] ? 'pass' : g === out[i] ? 'fail' : 'skip', note: g < out[i] ? (g === 0 ? 'unique' : notes[i][g]) : g === out[i] ? notes[i][g] : '' })) }));
  docs.forEach((d, i) => { if (d.out >= 1 && !d.cells[1].note) d.cells[1].note = notes[i][1]; });
  const start: number[][] = [[], [], []], gateStart: number[] = [];
  let t = 0;
  for (let g = 0; g < 3; g++) {
    gateStart.push(t); let k = 0;
    for (let i = 0; i < n; i++) { start[g][i] = docs[i].out >= g ? t + k++ * ROW : -1; }
    t += k * ROW + GAP;
  }
  gateStart.push(t);
  const keptChars = docs.reduce((s, d) => s + (d.out === 3 ? d.len : 0), 0);
  const pairOf = (a: number, b: number) => ({ exact: jaccard(sh[a], sh[b]), est: estimate(sig[a], sig[b]) });
  return { threshold, docs, start, gateStart, duration: t + 2.5, markers: [...GATES.map((g, i) => ({ at: gateStart[i], label: g })), { at: t, label: 'Result' }],
    kept: docs.filter((d) => d.out === 3).length, keptChars, totalChars: docs.reduce((s, d) => s + d.len, 0), pair: pairOf(0, 4), pair2: pairOf(0, 7) };
}
export interface DataFrame { gate: number; shown: boolean[][]; dim: number[]; chars: number; kept: number; active: { gate: number; doc: number } | null; status: string }
export function dataAt(t: number, r: DataRun): DataFrame {
  const n = r.docs.length, shown = [0, 1, 2].map((g) => r.docs.map((_, i) => r.start[g][i] >= 0 && t >= r.start[g][i]));
  const dim = r.docs.map((d, i) => (d.out < 3 ? ease((t - r.start[d.out][i]) / 0.45) * (t >= r.start[d.out][i] ? 1 : 0) : 0));
  let gate = 0; r.gateStart.slice(0, 3).forEach((s, g) => { if (t >= s) gate = g; });
  if (t >= r.gateStart[3]) gate = 3;
  let active: DataFrame['active'] = null, last = -1;
  for (let g = 0; g < 3; g++) for (let i = 0; i < n; i++) { const s = r.start[g][i]; if (s >= 0 && s <= t + 1e-9 && s > last) { last = s; active = { gate: g, doc: i }; } }
  let status = 'Twelve sample documents, three gates.';
  if (t >= r.gateStart[3]) status = `Result: ${r.kept} of ${n} documents kept, ${Math.round(r.docs.reduce((s, d, i) => s + d.len * (1 - dim[i]), 0)).toLocaleString('en-US')} of ${r.totalChars.toLocaleString('en-US')} characters.`;
  else if (active) {
    const d = r.docs[active.doc], c = d.cells[active.gate];
    status = `${GATES[active.gate]}, #${active.doc + 1} ${d.name}: ${c.state === 'fail' ? 'removed (' + c.note + ')' : 'kept'}.`;
  }
  const chars = r.docs.reduce((s, d, i) => s + d.len * (1 - dim[i]), 0);
  return { gate, shown, dim, chars, kept: n - dim.reduce((s, x) => s + (x > 0.999 ? 1 : 0), 0), active, status };
}

/* ------------------------------------------------------------------ compute */
export interface Log {
  curve: [number, number][]; checkpoints: { step: number; validLoss: number; sample: string; probes: number[] }[];
  samplePrompt: string; probes: { context: string; next: string }[];
  baselines: { unigramValidCrossEntropy: number; bigramValidCrossEntropy: number; uniformLnV: number };
  N: number; D: number; C: number; tokensPerStep: number; validLossQuantised: number;
}
export interface Rung { name: string; N: number; D: number; C: number; note: string; you?: boolean }
export const flops = (N: number, D: number) => 6 * N * D;
/** Published runs: N and D as stated in the papers (GPT-3: Brown et al. 2020, Table 2.1 and §2.2; Chinchilla: Hoffmann et al. 2022, abstract and Table 4; Llama 3: Dubey et al. 2024, §1). C = 6ND. */
export const PUBLISHED: Rung[] = [
  { name: 'GPT-3 175B', N: 175e9, D: 300e9, C: flops(175e9, 300e9), note: '175B parameters, 300B tokens' },
  { name: 'Chinchilla 70B', N: 70e9, D: 1.4e12, C: flops(70e9, 1.4e12), note: '70B parameters, 1.4T tokens' },
  { name: 'Llama 3 405B', N: 405e9, D: 15.6e12, C: flops(405e9, 15.6e12), note: '405B parameters, 15.6T tokens' },
];
export const GPT3_BATCH_TOKENS = 3.2e6;
export function ladder(log: Log, youN: number, youD: number): Rung[] {
  return [
    { name: 'House model', N: log.N, D: log.D, C: log.C, note: 'this lesson' },
    { name: 'Your model', N: youN, D: youD, C: flops(youN, youD), note: 'set below', you: true },
    ...PUBLISHED,
  ];
}
export const PHASE = 2.4, MIN_EXP = 10;
export const computeDuration = (rs: Rung[]) => rs.length * PHASE + HOLD;
export const computeMarkers = (rs: Rung[]): Marker[] => rs.map((r, i) => ({ at: i * PHASE, label: r.name }));
export interface ComputeFrame { phase: number; maxExp: number; rows: { rung: Rung; x: number; grow: number }[]; text: string }
export function computeAt(t: number, rs: Rung[]): ComputeFrame {
  const phase = Math.min(rs.length - 1, Math.floor(t / PHASE)), u = t - phase * PHASE;
  const target = (k: number) => Math.max(MIN_EXP + 3, Math.ceil(Math.log10(Math.max(...rs.slice(0, k + 1).map((r) => r.C)))) + 1);
  const from = phase ? target(phase - 1) : target(0), maxExp = from + (target(phase) - from) * ease(u / 1.2);
  const rows = rs.map((rung, i) => ({ rung, x: clamp01((Math.log10(rung.C) - MIN_EXP) / (maxExp - MIN_EXP)), grow: i < phase ? 1 : i === phase ? ease(u / 1.4) : 0 }));
  const r = rs[phase];
  return { phase, maxExp, rows, text: `${r.name}: ${eng(r.N)} parameters × ${eng(r.D)} tokens × 6 = ${sci(r.C, 2)} FLOP.` };
}

/* ------------------------------------------------------------------ learning */
export const LEARN_T = 16, STEP0 = 25;
const maxStep = (log: Log) => log.curve[log.curve.length - 1][0];
const stepAxis = (s: number) => Math.log10(1 + s / STEP0);
/** Horizontal position (0..1) of a step on the log-ish axis. */
export const stepX = (log: Log, s: number) => stepAxis(s) / stepAxis(maxStep(log));
const stepOf = (log: Log, t: number) => STEP0 * (10 ** (stepAxis(maxStep(log)) * clamp01(t / (LEARN_T - HOLD))) - 1);
const timeOf = (log: Log, s: number) => (LEARN_T - HOLD) * stepX(log, s);
/** First logged step whose validation loss is below `base`. */
export const crossing = (log: Log, base: number) => log.curve.find(([, l]) => l < base)?.[0] ?? maxStep(log);
export function learningMarkers(log: Log): Marker[] {
  return [{ at: 0, label: 'Worse than letter frequencies' }, { at: timeOf(log, crossing(log, log.baselines.unigramValidCrossEntropy)), label: 'Past letter frequencies' }, { at: timeOf(log, crossing(log, log.baselines.bigramValidCrossEntropy)), label: 'Past letter pairs' }];
}
export interface LearnFrame { step: number; idx: number; loss: number; cp: number; sample: string; probes: number[]; text: string }
export function learningAt(t: number, log: Log): LearnFrame {
  const s = stepOf(log, t);
  let idx = 0; log.curve.forEach(([st], i) => { if (st <= s + 1e-9) idx = i; });
  const [step, loss] = log.curve[idx];
  let cp = 0; log.checkpoints.forEach((c, i) => { if (c.step <= step) cp = i; });
  const c = log.checkpoints[cp], prev = log.checkpoints[Math.max(0, cp - 1)], e = cp ? ease((t - timeOf(log, c.step)) / 0.5) : 1;
  const probes = c.probes.map((p, i) => prev.probes[i] + (p - prev.probes[i]) * e);
  const b = log.baselines, where = loss < b.bigramValidCrossEntropy ? 'below the letter-pair guess' : loss < b.unigramValidCrossEntropy ? 'below the letter-frequency guess' : 'above the letter-frequency guess';
  return { step, idx, loss, cp, sample: c.sample.replace(/�/g, '▯').replace(/\n/g, '↵'), probes, text: `Step ${step}: validation loss ${loss.toFixed(2)}, ${where}.` };
}

/* ------------------------------------------------------------------ base model */
export const BASE_PROMPT = 'q: who did alice follow down the hole? a:';
export const BASE_N = 48, TYPE = 0.16;
export const baseDuration = () => BASE_N * TYPE + HOLD;
export const baseMarkers = (): Marker[] => [0, 16, 32].map((a) => ({ at: a * TYPE, label: `Characters ${a + 1} to ${a + 16}` }));
export interface Lane { label: string; ids: number[]; alts: { id: number; p: number }[][] }
const LANES = [{ label: 'Greedy', T: 0, seed: 0 }, { label: 'Sample, seed 1', T: 1, seed: 1 }, { label: 'Sample, seed 2', T: 1, seed: 2 }];
interface LaneState { ids: number[]; cache?: KVCache; logits: Float64Array; us: number[]; T: number }
export interface BaseJob { w: Weights; n: number; i: number; p: number[]; st: LaneState[]; lanes: Lane[] }
/** Greedy, and two seeded samples at temperature 1, generated one character at a time so the page can yield between chunks. */
export function baseJob(w: Weights, prompt = BASE_PROMPT, n = BASE_N): BaseJob {
  return { w, n, i: 0, p: encode(prompt.toLowerCase()).slice(-w.config.context), st: [], lanes: LANES.map((l) => ({ label: l.label, ids: [], alts: [] })) };
}
/** Advances the job until it is done (true) or `ms` milliseconds have passed (false). */
export function baseWork(j: BaseJob, ms = Infinity): boolean {
  const end = performance.now() + ms, ctx = j.w.config.context;
  while (j.st.length < LANES.length) { // one prefill per lane, so no single task is long
    if (performance.now() > end) return false;
    const l = LANES[j.st.length], rng = createRng(100 + l.seed), pre = prefill(j.w, j.p);
    j.st.push({ ids: [...j.p], cache: pre.cache as KVCache | undefined, logits: pre.logits.slice(-V), us: Array.from({ length: j.n }, () => rng.next()), T: l.T });
  }
  while (j.i < j.n) {
    if (performance.now() > end) return false;
    j.st.forEach((s, k) => {
      const model = softmax(s.logits), id = pickIndex(shape(s.logits, { temperature: s.T }), s.T === 0 ? 0 : s.us[j.i]);
      const order = Array.from(model, (_, q) => q).sort((a, b) => model[b] - model[a]).slice(0, 3);
      j.lanes[k].ids.push(id); j.lanes[k].alts.push(order.map((o) => ({ id: o, p: model[o] })));
      s.ids.push(id);
      if (j.i === j.n - 1) return;
      if (s.cache && s.ids.length <= ctx) s.logits = decodeStep(j.w, s.cache, id);
      else { s.cache = undefined; s.ids.shift(); s.logits = forward(j.w, s.ids).logits.slice(-V); }
    });
    j.i++;
  }
  return true;
}
export function baseRun(w: Weights, prompt = BASE_PROMPT, n = BASE_N): Lane[] { const j = baseJob(w, prompt, n); baseWork(j); return j.lanes; }
export interface BaseFrame { typed: number; fade: number }
export const baseAt = (t: number): BaseFrame => { const c = Math.min(BASE_N, Math.floor(t / TYPE) + 1); return { typed: c, fade: c < BASE_N || t < BASE_N * TYPE ? ease((t - (c - 1) * TYPE) / (TYPE * 0.8)) : 1 }; };
