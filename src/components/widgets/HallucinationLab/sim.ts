import { forward, type Weights } from '../../../lib/transformer/model';
import { createRng } from '../../../lib/math/rng';

export const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
export const ABBR = MONTHS.map((m) => m.slice(0, 3));
export type Mode = 'hook' | 'objective' | 'frequency' | 'calibration' | 'abstain' | 'retrieval' | 'verify';
export const COUNTS = [0, 1, 2, 5, 10, 20];
export const SAMPLE_COUNTS = [3, 10, 30];
/** Token ids of the facts model: a-z are 0-25, then space, period, 12 month tokens, <unknown>. */
const SPACE = 26, DOT = 27, MONTH0 = 28, UNKNOWN = 40, CHARS = 'abcdefghijklmnopqrstuvwxyz .';
export const MAX_NAME = 14;
export interface Person { name: string; month: number; count: number }
export interface Answer { dist: number[]; pick: number; conf: number; unknown: number }
export interface Row extends Person, Answer { correct: boolean }
export interface Experiment { rows: Row[]; right: Answer[]; wrong: Answer[] }

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
export const mean = (xs: number[]) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
export const argmax = (xs: ArrayLike<number>) => { let b = 0; for (let i = 1; i < xs.length; i++) if (xs[i] > xs[b]) b = i; return b; };
/** The wrong month written into the prompt: five months after the truth. */
export const wrongMonth = (m: number) => (m + 5) % 12;
export const validName = (name: string) => name.length > 0 && name.length <= MAX_NAME && /^[a-z]+( [a-z]+)*$/.test(name);

/** Ids of "<name> was born in ", optionally after a stated line "<name2> was born in <month>. ". */
export function promptIds(name: string, stated?: { name: string; month: number }): number[] {
  const enc = (s: string) => [...s].map((c) => CHARS.indexOf(c));
  const ask = enc(`${name} was born in `);
  return stated ? [...enc(`${stated.name} was born in `), MONTH0 + stated.month, DOT, SPACE, ...ask] : ask;
}
/** The model's distribution over the 12 months after the prompt (renormalised over months, as in the training report). */
export function ask(w: Weights, name: string, stated?: { name: string; month: number }): Answer {
  if (!validName(name)) throw new Error('Use 1 to 14 lowercase letters and single spaces.');
  const ids = promptIds(name, stated), V = w.config.vocab.length, row = forward(w, ids).probs.subarray((ids.length - 1) * V, ids.length * V);
  const raw = Array.from({ length: 12 }, (_, m) => row[MONTH0 + m]), mass = raw.reduce((a, b) => a + b, 0);
  const dist = raw.map((p) => p / mass), pick = argmax(dist);
  return { dist, pick, conf: dist[pick], unknown: row[UNKNOWN] };
}
/** Every number the lesson quotes comes from this one pass: 200 plain questions plus 100 with a fact in the prompt. */
export function runExperiment(w: Weights, people: Person[]): Experiment {
  const rows = people.map((p) => { const a = ask(w, p.name); return { ...p, ...a, correct: a.pick === p.month }; });
  const unseen = rows.filter((r) => r.count === 0);
  return {
    rows,
    right: unseen.map((r) => ask(w, r.name, { name: r.name, month: r.month })),
    wrong: unseen.map((r) => ask(w, r.name, { name: r.name, month: wrongMonth(r.month) })),
  };
}

// ---- statistics over rows ----
/** Share of training text per month: each person's count goes to their true month. */
export function trainingFrequency(people: Person[]): number[] {
  const f = Array(12).fill(0) as number[];
  for (const p of people) f[p.month] += p.count;
  const total = f.reduce((a, b) => a + b, 0);
  return f.map((x) => x / total);
}
export function byCount(rows: Row[]) {
  return COUNTS.map((count) => {
    const group = rows.filter((r) => r.count === count), right = group.filter((r) => r.correct).length;
    return { count, n: group.length, right, accuracy: group.length ? right / group.length : 0, examples: group.slice(0, 3) };
  });
}
export function reliability(rows: Row[]) {
  const bins = Array.from({ length: 10 }, (_, i) => {
    const g = rows.filter((r) => Math.min(9, Math.floor(r.conf * 10)) === i);
    return { n: g.length, confidence: mean(g.map((r) => r.conf)), accuracy: mean(g.map((r) => +r.correct)) };
  });
  const ece = rows.length ? bins.reduce((s, b) => s + b.n * Math.abs(b.accuracy - b.confidence), 0) / rows.length : 0;
  return { bins, ece, meanConfidence: mean(rows.map((r) => r.conf)), accuracy: mean(rows.map((r) => +r.correct)) };
}
export function selective(rows: Row[], threshold: number) {
  const answered = rows.filter((r) => r.conf >= threshold), right = answered.filter((r) => r.correct).length;
  return {
    answered: answered.length, right, wrong: answered.length - right, abstained: rows.length - answered.length,
    coverage: rows.length ? answered.length / rows.length : 0, error: answered.length ? 1 - right / answered.length : 0,
    unseenAnswered: answered.filter((r) => r.count === 0).length,
  };
}
/** Smallest threshold (to 0.01) whose answered questions have error below `limit`. */
export function thresholdFor(rows: Row[], limit: number): number {
  for (let i = 0; i <= 100; i++) { const s = selective(rows, i / 100); if (s.answered && s.error < limit) return i / 100; }
  return 1;
}
export const totalVariation = (a: number[], b: number[]) => a.reduce((s, p, i) => s + Math.abs(p - b[i]), 0) / 2;
/** A fixed shuffle so people arrive in a mixed order. */
export function order(n: number, seed: number): number[] {
  const rng = createRng(seed), ids = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); [ids[i], ids[j]] = [ids[j], ids[i]]; }
  return ids;
}
/** k answers drawn from the model's own distribution (temperature 1) by inverse CDF, seeded. */
export function drawSamples(dist: number[], k: number, seed: number): number[] {
  const rng = createRng(seed), out: number[] = [];
  for (let i = 0; i < k; i++) {
    const u = rng.next(); let s = 0, id = 11;
    for (let j = 0; j < 12; j++) { s += dist[j]; if (u < s) { id = j; break; } }
    out.push(id);
  }
  return out;
}
/** Share of the k samples that repeat the model's main (most likely) answer. */
export const agreement = (samples: number[], pick: number) => samples.filter((s) => s === pick).length / samples.length;
export function consistency(rows: Row[], k: number, seed = 101) {
  return rows.map((r, i) => ({ row: r, agreement: agreement(drawSamples(r.dist, k, seed + i), r.pick) }));
}
export function flagStats(rows: Row[], k: number, cutoff: number) {
  const c = consistency(rows, k), kept = c.filter((x) => x.agreement >= cutoff), flagged = c.filter((x) => x.agreement < cutoff);
  const right = (xs: typeof c) => xs.filter((x) => x.row.correct).length;
  return { keptN: kept.length, flaggedN: flagged.length, keptRight: right(kept), flaggedWrong: flagged.length - right(flagged), accuracy: (right(kept) + flagged.length - right(flagged)) / c.length };
}

// ---- timeline ----
export const DURATION: Record<Mode, number> = { hook: 9, objective: 12, frequency: 10, calibration: 10, abstain: 10, retrieval: 12, verify: 14 };
export const MARKERS: Record<Mode, { at: number; label: string }[]> = {
  hook: [{ at: 0, label: 'typing' }, { at: 3.5, label: 'month odds' }, { at: 7, label: 'true month' }],
  objective: [{ at: 0, label: 'feeding people' }, { at: 10, label: 'full average' }],
  frequency: [{ at: 0, label: 'checking people' }, { at: 8, label: 'final accuracy' }],
  calibration: [{ at: 0, label: 'sorting answers' }, { at: 8, label: 'final gap' }],
  abstain: [{ at: 0, label: 'low threshold' }, { at: 3.5, label: 'middle' }, { at: 7, label: 'high threshold' }],
  retrieval: [{ at: 0, label: 'no context' }, { at: 4, label: 'right fact' }, { at: 8, label: 'wrong fact' }],
  verify: [{ at: 0, label: 'drawing samples' }, { at: 7, label: 'all 200 people' }],
};
const ease = (x: number) => { const c = clamp01(x); return c * c * (3 - 2 * c); };
const upTo = (t: number, from: number, to: number) => clamp01((t - from) / (to - from));
export const abstainThreshold = (t: number) => clamp01(t / DURATION.abstain);

/** One question and the model's answer to it, as drawn in a panel. */
export interface Panel { name: string; count: number; truth?: number; answer: Answer }
export interface Context {
  exp: Experiment;
  people: Person[];
  /** Unseen people only (count 0), in the order of exp.right / exp.wrong. */
  unseen: Row[];
  /** Hook and verify: [a trained person, an unseen person or a name the reader typed]. */
  panels: Panel[];
  unseenOnly?: boolean;
  k?: number;
  /** Retrieval: the worked person, the model's answer to each of the 12 stated months, and the month written in the third state. */
  worked?: { row: Row; stated: Answer[]; month: number };
  /** Objective: index into `unseen` of a chosen person, or -1. */
  person?: number;
}

/** The trained person with the most lines and the unseen person the model is surest about. */
export function pickHook(rows: Row[]): [Row, Row] {
  const best = (xs: Row[]) => xs.reduce((b, r) => (r.conf > b.conf ? r : b));
  return [best(rows.filter((r) => r.count === 20)), best(rows.filter((r) => r.count === 0))];
}
/** An unseen person the model gets wrong, whose answer is neither the true month nor the wrong month written later. */
export const pickWorked = (unseen: Row[]): Row => unseen.find((r) => !r.correct && r.pick !== wrongMonth(r.month) && r.conf < 0.4)!;
export const panelOf = (r: Row): Panel => ({ name: r.name, count: r.count, truth: r.count > 0 ? r.month : undefined, answer: r });

const SHUFFLE = order(200, 17);

/** Everything needed to draw the figure at time t. Pure: scrubbing is free. */
export function frameAt(t: number, mode: Mode, c: Context) {
  const rows = c.exp.rows;
  switch (mode) {
    case 'hook': {
      const typing = clamp01(t / 3), reveal = ease(upTo(t, 3.5, 5.5)), mark = ease(upTo(t, 7, 8));
      return { mode, typing, reveal, mark, showNumbers: t >= 3.5, showTruth: t >= 7 };
    }
    case 'objective': {
      const n = Math.max(0, Math.min(c.unseen.length, Math.round(clamp01(t / 10) * c.unseen.length)));
      const fed = c.unseen.slice(0, n);
      const average = MONTHS.map((_, m) => mean(fed.map((r) => r.dist[m])));
      const freq = trainingFrequency(c.people);
      return { mode, n, last: fed[n - 1], average, freq, tv: n ? totalVariation(average, freq) : 0, unknown: mean(fed.map((r) => r.unknown)), done: t >= 10 };
    }
    case 'frequency': {
      const p = clamp01(t / 8);
      const bins = COUNTS.map((count) => {
        const group = rows.filter((r) => r.count === count), m = Math.round(p * group.length), seen = group.slice(0, m), right = seen.filter((r) => r.correct).length;
        return { count, n: group.length, checked: m, right, accuracy: m ? right / m : 0 };
      });
      return { mode, bins, done: p >= 1 };
    }
    case 'calibration': {
      const pool = c.unseenOnly ? c.unseen : rows, ord = c.unseenOnly ? order(pool.length, 29) : SHUFFLE;
      const n = Math.round(clamp01(t / 8) * pool.length), part = ord.slice(0, n).map((i) => pool[i]);
      const rel = reliability(part), worst0 = rel.bins.reduce((b, x, i) => x.n >= 3 && x.confidence - x.accuracy > (rel.bins[b]?.confidence ?? 0) - (rel.bins[b]?.accuracy ?? 0) ? i : b, -1);
      const worst = worst0 >= 0 && rel.bins[worst0].confidence - rel.bins[worst0].accuracy > 0.05 ? worst0 : -1;
      return { mode, n, total: pool.length, ...rel, worst, maxBin: Math.max(...reliability(pool).bins.map((b) => b.n)), done: n === pool.length };
    }
    case 'abstain': {
      const threshold = abstainThreshold(t), s = selective(rows, threshold);
      const curve = Array.from({ length: 101 }, (_, i) => { const q = selective(rows, i / 100); return { threshold: i / 100, coverage: q.coverage, error: q.error }; });
      const dots = [...rows].sort((a, b) => b.conf - a.conf).map((r) => (r.conf >= threshold ? (r.correct ? 0 : 1) : 2));
      return { mode, threshold, ...s, total: rows.length, curve, dots };
    }
    case 'retrieval': {
      const w = c.worked!, seg = t < 4 ? 0 : t < 8 ? 1 : 2, local = ease(upTo(t - seg * 4, 0, 1));
      const answers = [w.row, w.stated[w.row.month], w.stated[w.month]];
      const ex = c.exp, unseen = c.unseen;
      const acc = [mean(unseen.map((r) => +r.correct)), mean(ex.right.map((a, i) => +(a.pick === unseen[i].month))), mean(ex.wrong.map((a, i) => +(a.pick === unseen[i].month)))];
      const copies = mean(ex.wrong.map((a, i) => +(a.pick === wrongMonth(unseen[i].month))));
      const from = seg === 0 ? answers[0].dist : answers[seg - 1].dist, to = answers[seg].dist;
      const shown = from.map((p, i) => p + (to[i] - p) * local);
      const stated = seg === 0 ? null : seg === 1 ? w.row.month : w.month;
      const bars = acc.map((a, i) => (i < seg ? a : i === seg ? a * local : 0));
      return { mode, seg, local, shown, answer: answers[seg], stated, acc, bars, copies, unseenN: unseen.length };
    }
    case 'verify': {
      const k = c.k ?? 10, sampleP = clamp01(t / 6), drawn = Math.round(sampleP * k);
      const panels = c.panels.map((p, i) => {
        const samples = drawSamples(p.answer.dist, k, 31 + i), shown = samples.slice(0, drawn), counts = Array(12).fill(0) as number[];
        for (const s of shown) counts[s]++;
        return { panel: p, samples, counts, agreement: shown.length ? agreement(shown, p.answer.pick) : 0, drawn, latest: shown[shown.length - 1] };
      });
      const cons = consistency(rows, k), n = Math.round(upTo(t, 7, 13) * rows.length);
      const part = SHUFFLE.slice(0, n).map((i) => cons[i]);
      const levels = Array.from({ length: k + 1 }, (_, j) => {
        const g = part.filter((x) => Math.round(x.agreement * k) === j);
        return { share: j / k, right: g.filter((x) => x.row.correct).length, wrong: g.filter((x) => !x.row.correct).length };
      });
      const kept = part.filter((x) => x.agreement >= 0.5), flagged = part.filter((x) => x.agreement < 0.5);
      const maxLevel = Math.max(...Array.from({ length: k + 1 }, (_, j) => cons.filter((x) => Math.round(x.agreement * k) === j).length));
      return { mode, k, panels, n, levels, maxLevel, kept: kept.length, keptRight: kept.filter((x) => x.row.correct).length, flagged: flagged.length, flaggedWrong: flagged.filter((x) => !x.row.correct).length, done: n === rows.length };
    }
  }
}
export type Frame = ReturnType<typeof frameAt>;

/** Percent with d decimals; a trailing ".0" is dropped (72.5%, 34%). */
export const pct = (x: number, d = 0) => `${(100 * x).toFixed(d).replace(/\.0+$/, '')}%`;
const cap = (s: string) => s[0].toUpperCase() + s.slice(1);
/** The sentence under the figure. Built from the same frame as the drawing, so the two always agree. */
export function describe(f: Frame, c: Context): string {
  switch (f.mode) {
    case 'hook': {
      const [a, b] = c.panels;
      if (!f.showNumbers) return 'Each question is typed out and the model writes its answer.';
      const line = (p: Panel) => `${cap(p.name)}: ${MONTHS[p.answer.pick]}, ${pct(p.answer.conf)} sure.`;
      const tail = f.showTruth ? ` ${cap(a.name)}'s true month is ${MONTHS[a.truth ?? a.answer.pick]}; ${b.name} has no line in the training text, so no month is right.` : '';
      return `${line(a)} ${line(b)}${tail}`;
    }
    case 'objective':
      return f.n ? `${f.n} of ${c.unseen.length} unseen people fed in${f.last ? `, latest ${f.last.name}` : ''}. Distance between the two sets of bars: ${f.tv.toFixed(2)}. Probability of answering "unknown": ${pct(f.unknown, 2)}.` : 'No unseen person has been fed in yet.';
    case 'frequency': {
      const checked = f.bins.reduce((s, b) => s + b.checked, 0), p = (b: (typeof f.bins)[number]) => (b.checked ? pct(b.accuracy) : 'none checked');
      return `Checked ${checked} of ${c.exp.rows.length} people. Never seen: ${p(f.bins[0])}. Seen once: ${p(f.bins[1])}. Seen 20 times: ${p(f.bins[5])}.`;
    }
    case 'calibration':
      return f.n ? `${f.n} of ${f.total} answers sorted. Average confidence ${pct(f.meanConfidence)}, actually right ${pct(f.accuracy)}. Calibration error (ECE) ${f.ece.toFixed(3)}.${f.done && f.worst >= 0 ? ` Most overconfident bin: answers it was ${pct(f.bins[f.worst].confidence)} sure of were right ${pct(f.bins[f.worst].accuracy)} of the time.` : ''}` : 'No answers sorted yet.';
    case 'abstain':
      return `Threshold ${f.threshold.toFixed(2)}: ${f.answered} of ${f.total} questions answered (${pct(f.coverage)}), ${f.answered ? pct(f.error, 1) : 'no'} of those answers wrong. ${f.abstained} questions get "I don't know".`;
    case 'retrieval': {
      const a = f.answer, name = c.worked!.row.name;
      const head = f.seg === 0 ? `With no fact in the prompt, ${name} gets ${MONTHS[a.pick]} (${pct(a.conf)}).` : `With "${MONTHS[f.stated!]}" stated in the prompt, ${name} gets ${MONTHS[a.pick]} (${pct(a.conf)}).`;
      const tail = f.seg === 0 ? ` Right for ${pct(f.acc[0])} of the ${f.unseenN} unseen people.` : f.seg === 1 ? ` Right for ${pct(f.acc[1])} of the ${f.unseenN}.` : ` Right for ${pct(f.acc[2])} of the ${f.unseenN}; the model repeats the wrong month for ${pct(f.copies)}.`;
      return head + tail;
    }
    case 'verify': {
      if (!f.done && f.n === 0) {
        const [a, b] = f.panels;
        return `${f.k} samples each. ${cap(a.panel.name)}: ${a.drawn} drawn, ${pct(a.agreement)} repeat the main answer. ${cap(b.panel.name)}: ${b.drawn} drawn, ${pct(b.agreement)} repeat it.`;
      }
      return `${f.n} of ${c.exp.rows.length} people checked. Agreement below 50% flags ${f.flagged} answers, ${f.flaggedWrong} of them wrong; ${f.kept} pass, ${f.keptRight} of them right.`;
    }
  }
}
