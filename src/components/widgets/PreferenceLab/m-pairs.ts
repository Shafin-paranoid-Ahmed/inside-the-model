/**
 * pairs (rlhf step 1, labels "human"; constitutional-ai step 4, labels "ai"): preference pairs sampled from the SFT
 * policy, labelled one by one. The first pairs play slowly (two response cards, then the choice); the rest stream into a
 * strip of squares, one per pair, coloured by whether the choice agrees with the gold ranking. With labels "ai" both
 * the AI judge's and the human labeler's choices are shown (same pairs, same seed; only the labeler differs), and a ring
 * marks squares where they disagree. Click (or hover) a square to see that pair.
 * Props: labels ("human" | "ai"), pairs (200), seed (4). Measured (seed 4): human labels agree with gold on 0.685 of
 * 200, the longer response wins 0.485, 58 choices are longer-and-worse; AI agrees with human 0.71, AI with gold 0.705.
 */
import { CANDIDATES, PROMPTS, TEXT, phi, F_LENGTH } from '../../../lib/align/bank';
import { GOLD } from '../../../lib/align/policy';
import { PRINCIPLES, violations } from '../../../lib/align/constitution';
import type { Pair } from '../../../lib/align/reward';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, circle, ease, fmt, rect, text, type Env } from '../_rl/svg';
import { charsFor, clip, responseCard } from '../_align/draw';
import { labelsOf, prefData, type Labels } from './sim';

const SLOW_N = 6, SLOW = 2.6, CHOOSE = 1.3, FAST = 9, CADENCE = 0.5, HOLD = 2.5;
interface Row { q: Pair; other: Pair; goldOk: boolean; otherGoldOk: boolean; longer: boolean; agree: boolean }
interface Scene { rows: Row[]; labels: Labels; n: number; cum: { gold: Int32Array; longer: Int32Array; agree: Int32Array; otherGold: Int32Array; longWorse: Int32Array } }
interface F extends Frame { done: number; show: number; chosen: boolean; fade: number; picked: boolean }

const name = (y: number) => CANDIDATES[y].label.toLowerCase();
/** Why the AI judge chose: the principles the rejected response breaks and the chosen one keeps, or a coin on ties. */
const decider = (q: Pair): string => {
  const keep = violations(q.x, q.w), broken = violations(q.x, q.l).filter((id) => !keep.includes(id));
  return broken.length && keep.length < violations(q.x, q.l).length ? `Deciding principle: “${broken.map((id) => PRINCIPLES.find((p) => p.id === id)!.text).join('” and “')}”.` : 'Both break the same number of principles, so a coin decided.';
};
const words = (x: number, y: number) => Math.round(phi(x, y)[F_LENGTH] * 10);
/** Number of pairs labelled by time t, the pair whose cards are shown, and whether its choice is revealed. */
function at(t: number, n: number) {
  const slowEnd = SLOW_N * SLOW;
  if (t < slowEnd) { const i = Math.floor(t / SLOW), u = t - i * SLOW; return { done: i + (u >= CHOOSE ? 1 : 0), show: i, chosen: u >= CHOOSE, fade: ease((u - CHOOSE) / 0.4) }; }
  const v = Math.min(1, (t - slowEnd) / FAST), done = Math.min(n, SLOW_N + Math.round(v * (n - SLOW_N)));
  const tick = Math.floor((t - slowEnd) / CADENCE), shownAt = slowEnd + tick * CADENCE, shown = Math.min(n, SLOW_N + Math.round(Math.min(1, (shownAt - slowEnd) / FAST) * (n - SLOW_N)));
  return { done, show: Math.max(0, shown - 1), chosen: true, fade: 1 };
}

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const d = prefData(o, kit, int(o.seed, 4)), labels = labelsOf(o), n = d.pairs.length;
    const rows: Row[] = d.pairs.map((q, i) => {
      const other = labels === 'ai' ? d.human[i] : d.ai[i];
      const ok = (p: Pair) => GOLD[p.x * 8 + p.w] > GOLD[p.x * 8 + p.l];
      return { q, other, goldOk: ok(q), otherGoldOk: ok(other), longer: phi(q.x, q.w)[F_LENGTH] > phi(q.x, q.l)[F_LENGTH], agree: q.w === other.w };
    });
    const cum = { gold: new Int32Array(n + 1), longer: new Int32Array(n + 1), agree: new Int32Array(n + 1), otherGold: new Int32Array(n + 1), longWorse: new Int32Array(n + 1) };
    rows.forEach((r, i) => {
      cum.gold[i + 1] = cum.gold[i] + +r.goldOk; cum.longer[i + 1] = cum.longer[i] + +r.longer; cum.agree[i + 1] = cum.agree[i] + +r.agree;
      cum.otherGold[i + 1] = cum.otherGold[i] + +r.otherGoldOk; cum.longWorse[i + 1] = cum.longWorse[i] + +(r.longer && !r.goldOk);
    });
    return { rows, labels, n, cum };
  },
  timeline: (s) => ({ duration: SLOW_N * SLOW + FAST + HOLD, markers: [{ at: 0, label: 'One pair at a time' }, { at: SLOW_N * SLOW, label: `All ${s.n} pairs` }] }),
  frameAt(t, s, _o, ui) {
    const pick = ui.pick === null ? NaN : Number(ui.pick), picked = Number.isInteger(pick) && pick >= 0 && pick < s.n;
    const a = picked ? { done: s.n, show: pick, chosen: true, fade: 1 } : at(t, s.n), r = s.rows[a.show], q = r.q, ai = s.labels === 'ai', c = s.cum, d = a.done;
    const pct = (k: number) => (d ? `${fmt((100 * k) / d, 0)}%` : '–');
    const who = ai ? 'the AI judge' : 'the labeler';
    let status: string;
    if (!a.chosen) status = `Pair ${a.show + 1}: two responses to “${PROMPTS[q.x].q}” Which would a person prefer?`;
    else {
      const win = name(q.w), lose = name(q.l), gold = r.goldOk ? 'agreeing with the gold ranking' : 'against the gold ranking';
      status = `Pair ${a.show + 1}: ${who} chose ${win} over ${lose}, ${gold}` + (ai
        ? (r.agree ? '; the human labeler made the same choice.' : `; the human labeler chose ${name(r.other.w)}. ${decider(q)}`)
        : (r.longer && !r.goldOk ? ': the longer answer won although it is worse.' : '.'));
    }
    const readouts: [string, string][] = ai
      ? [['Pairs labelled', `${d} of ${s.n}`], ['AI agrees with human', pct(c.agree[d])], ['AI agrees with gold', pct(c.gold[d])], ['Human agrees with gold', pct(c.otherGold[d])]]
      : [['Pairs labelled', `${d} of ${s.n}`], ['Agrees with gold', pct(c.gold[d])], ['Longer response chosen', pct(c.longer[d])], ['Longer but worse chosen', String(c.longWorse[d])]];
    return { ...a, picked, status, readouts, detail: picked ? 'Click the square again to resume.' : 'Click a square to see that pair.' };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, pad = nw ? 8 : 12, r = s.rows[f.show], q = r.q, ai = s.labels === 'ai';
    text(E, clip(`User: ${PROMPTS[q.x].q}`, charsFor(W - 2 * pad, 13)), pad, 16, { size: 13, weight: 600 });
    const cw = nw ? W - 2 * pad : (W - 2 * pad - 12) / 2, ch = nw ? 92 : 96;
    [q.a, q.b].forEach((y, j) => {
      const x0 = nw ? pad : pad + j * (cw + 12), y0 = nw ? 26 + j * (ch + 8) : 26, won = f.chosen && q.w === y, otherWon = ai && f.chosen && r.other.w === y;
      const tags = [won ? (ai ? 'AI judge' : 'Chosen') : '', otherWon ? 'Human' : ''].filter(Boolean).join(' + ');
      responseCard(E, { x: x0, y: y0, w: cw, h: ch }, {
        title: `${j ? 'B' : 'A'}: ${name(y)}, ${words(q.x, y)} word${words(q.x, y) === 1 ? '' : 's'}, gold ${fmt(GOLD[q.x * 8 + y], 2)}`, text: TEXT[q.x][y], size: 12, maxLines: nw ? 3 : 4,
        accent: won, tag: tags || undefined, op: f.chosen && !won && !otherWon ? 1 - 0.45 * f.fade : 1,
      });
    });
    // strip: one square per pair
    const top = nw ? 26 + 2 * (ch + 8) + 22 : 26 + ch + 34, cols = nw ? 20 : 40, rows = Math.ceil(s.n / cols);
    const cell = Math.min((W - 2 * pad) / cols, (E.H - top - 8) / rows, nw ? 16 : 24);
    text(E, clip(ai ? (nw ? 'AI agrees with gold? Ring: human chose differently' : 'Every pair: does the AI judge agree with gold? (ring: the human chose differently)') : 'Every pair: does the choice agree with the gold ranking?', charsFor(W - 2 * pad, 11)), pad, top - 8, { size: 11, fill: MUTED });
    for (let i = 0; i < s.n; i++) {
      const x = pad + (i % cols) * cell, y = top + Math.floor(i / cols) * cell, row = s.rows[i], on = i < f.done;
      const tip = `Pair ${i + 1}: ${name(row.q.w)} chosen over ${name(row.q.l)}${ai ? `; human chose ${name(row.other.w)}` : ''}`;
      rect(E, x + 1, y + 1, cell - 2, cell - 2, { fill: on ? (row.goldOk ? 'var(--viz-pos)' : 'var(--viz-neg)') : 'var(--rule)', op: on ? 0.85 : 0.5, rx: 2, tip, pick: String(i) });
      if (on && ai && !row.agree) circle(E, x + cell / 2, y + cell / 2, cell * 0.22, { fill: BG });
      if (i === f.show && on) rect(E, x, y, cell, cell, { fill: 'none', stroke: ACCENT, sw: 2, rx: 2 });
    }
  },
  label: (s) => `${s.n} preference pairs from the SFT policy, labelled by ${s.labels === 'ai' ? 'an AI judge applying the constitution, beside the human labels' : 'simulated human labelers'}; ${fmt((100 * s.cum.gold[s.n]) / s.n, 0)}% of the choices agree with the gold ranking.`,
};
export default mode;
