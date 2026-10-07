/**
 * revise (constitutional-ai step 2): the critique-and-revise loop. Each round the model picks one principle at random;
 * if the response breaks it, the response is rewritten to fix that (changing as little else as possible), else it is
 * kept. Three starting responses play one after another (a list continuation, a refusal and a padded answer, each one
 * that ends with no violations); a strip at the bottom records the responses visited. The aggregate over all sampled
 * starts is in the detail line. Props: rounds (4), seed (3), from ("base" | "sft": the model that writes the starts,
 * default base), n (64), principles.
 * Measured (rounds 4, seed 3, 64 starts from the base model): violations per response 1.156 -> 0.766, 206 of 256 rounds
 * find nothing to fix; from the SFT model only 12 of 64 starts break anything (0.188 -> 0.109).
 */
import { CANDIDATES, PROMPTS, THETA_BASE, TEXT, Y_CONTINUE, Y_PADDED, Y_REFUSAL } from '../../../lib/align/bank';
import { slCAI, violationCount, type SLCAIResult } from '../../../lib/align/constitution';
import { sftTheta } from '../../../lib/align/sft';
import { int, key, str, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, arrow, ease, fmt, mix, rect, text, type Env } from '../_rl/svg';
import { candColor, charsFor, clip, responseCard } from '../_align/draw';
import { CRITIQUE, plural } from './cr-common';
import { principlesOf, principleText } from './sim';

const PER = 3, CHECK = 1, VERDICT = 2, HOLD = 3;
type Start = SLCAIResult['starts'][number];
interface Scene { shown: Start[]; act: string[]; rounds: number; n: number; before: number; after: number; none: number; total: number }
interface F extends Frame { si: number; r: number; ph: 0 | 1 | 2; u: number }

/** One start per kind (list continuation, refusal, padded) that ends clean, else any start that changes; fall back to the first starts. */
function pick(starts: Start[], act: string[]): Start[] {
  const clean = (s: Start) => s.rounds.some((q) => q.violated) && violationCount(s.x, s.y, act) === 0;
  const out: Start[] = [];
  for (const y of [Y_CONTINUE, Y_REFUSAL, Y_PADDED]) {
    const s = starts.find((q) => q.y0 === y && clean(q)) ?? starts.find((q) => q.y0 === y && q.rounds.some((r) => r.violated));
    if (s) out.push(s);
  }
  for (const s of starts) if (out.length < 3 && !out.includes(s) && s.rounds.some((q) => q.violated)) out.push(s);
  return out.length ? out : starts.slice(0, 1);
}
/** The distinct responses visited up to round r (the revised one counts once phase 2 is reached). */
const chainOf = (s: Start, r: number, ph: number): number[] => {
  const c = [s.y0];
  s.rounds.forEach((q, i) => { if ((i < r || (i === r && ph === 2)) && q.to !== c[c.length - 1]) c.push(q.to); });
  return c;
};

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const act = principlesOf(o), rounds = Math.max(1, int(o.rounds, 4)), seed = int(o.seed, 3), n = int(o.n, 64), from = str(o.from, ['base', 'sft'] as const, 'base');
    const res = kit.memo(key('cr-revise', act, rounds, seed, n, from), () => { const sft = sftTheta(); return slCAI(sft, { n, rounds, seed, steps: 1, active: act, from: from === 'base' ? THETA_BASE : sft }); });
    const st = res.starts;
    return {
      shown: pick(st, act), act, rounds, n,
      before: st.reduce((a, s) => a + violationCount(s.x, s.y0, act), 0) / n, after: st.reduce((a, s) => a + violationCount(s.x, s.y, act), 0) / n,
      none: st.reduce((a, s) => a + s.rounds.filter((q) => !q.violated).length, 0), total: n * rounds,
    };
  },
  timeline: (s) => ({
    duration: s.shown.length * s.rounds * PER + HOLD,
    markers: s.shown.map((st, i) => ({ at: i * s.rounds * PER, label: CANDIDATES[st.y0].short })),
  }),
  frameAt(t, s) {
    const D = s.rounds * PER, end = t >= s.shown.length * D, si = end ? s.shown.length - 1 : Math.min(s.shown.length - 1, Math.floor(t / D)), local = end ? D - 1e-6 : t - si * D;
    const r = Math.min(s.rounds - 1, Math.floor(local / PER)), u = local - r * PER, ph = (u < CHECK ? 0 : u < VERDICT ? 1 : 2) as 0 | 1 | 2;
    const st = s.shown[si], q = st.rounds[r], id = q.principle, ptxt = principleText(id), changed = q.from !== q.to;
    let status: string;
    if (ph === 0) status = `Round ${r + 1} of ${s.rounds}: the model picks the principle “${ptxt}” and reads its response against it.`;
    else if (ph === 1) status = changed ? `Critique: ${CRITIQUE[id] ?? 'it breaks the principle'}, so the model rewrites it.` : 'Critique: nothing in this response breaks the principle, so there is nothing to fix.';
    else status = changed ? `Revision: “${CANDIDATES[q.to].label.toLowerCase()}”, which breaks ${plural(q.after, 'principle')}.` : `No revision: the response stays “${CANDIDATES[q.from].label.toLowerCase()}”.`;
    return {
      si, r, ph, u, status,
      readouts: [['Starting response', `${si + 1} of ${s.shown.length}`], ['Round', `${r + 1} of ${s.rounds}`], ['Principles this response breaks', String(ph === 2 ? q.after : q.before)]],
      detail: `All ${s.n} sampled starts: ${fmt(s.before, 2)} principles broken per response before, ${fmt(s.after, 2)} after ${s.rounds} rounds; ${s.none} of ${s.total} rounds found nothing to fix.`,
    };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, pad = nw ? 8 : 12, st = s.shown[f.si], q = st.rounds[f.r], x = st.x, np = s.act.length;
    text(E, clip(`User: ${PROMPTS[x].q}`, charsFor(W - 2 * pad, 13)), pad, 16, { size: 13, weight: 600 });
    // the constitution; the principle picked this round is highlighted
    const lh = nw ? 17 : 24, px = pad, pw = nw ? W - 2 * pad : 276, top = 30;
    text(E, 'Constitution', px, top + 10, { size: 11, fill: MUTED });
    s.act.forEach((id, j) => {
      const y = top + 16 + j * lh, on = id === q.principle;
      if (on) rect(E, px - 4, y - 2, pw + 8, lh - 1, { fill: mix(ACCENT, BG, 12), stroke: ACCENT, sw: 1.5, rx: 4, op: f.ph === 0 ? 0.5 + 0.5 * ease(f.u / 0.4) : 1 });
      text(E, `${j + 1}. ${clip(principleText(id), charsFor(pw - 14, 12))}`, px + 2, y + lh / 2 + 2, { size: nw ? 11 : 12, weight: on ? 650 : 400, fill: on ? undefined : MUTED });
    });
    // two cards with the critique between them
    const cx = nw ? pad : px + pw + 24, cw = W - cx - pad, cy = nw ? top + 16 + np * lh + 8 : 30, mh = nw ? 34 : 38, ch = nw ? 74 : 80;
    const changed = q.from !== q.to;
    responseCard(E, { x: cx, y: cy, w: cw, h: ch }, { title: `Response now: ${CANDIDATES[q.from].label.toLowerCase()}`, text: TEXT[x][q.from], size: 12, maxLines: 3, tag: plural(q.before, 'violation') });
    const my = cy + ch + 4, ca = ease((f.u - CHECK) / 0.3);
    if (f.ph >= 1) {
      arrow(E, cx + 14, my + 2, cx + 14, my + mh - 2, { stroke: q.violated ? 'var(--viz-neg)' : MUTED, width: 2, op: ca });
      const crit = q.violated ? `Critique: ${CRITIQUE[q.principle] ?? 'it breaks the principle'}.` : 'Critique: nothing here breaks it.';
      text(E, clip(crit, charsFor(cw - 36, 12)), cx + 28, my + mh / 2 + 4, { size: nw ? 11 : 12, fill: q.violated ? 'var(--viz-neg)' : MUTED, weight: 600, op: ca });
    } else text(E, clip(`Round ${f.r + 1} of ${s.rounds}: checking principle ${s.act.indexOf(q.principle) + 1}`, charsFor(cw - 28, 12)), cx + 14, my + mh / 2 + 4, { size: nw ? 11 : 12, fill: MUTED });
    const shown = f.ph === 2, a = shown ? ease((f.u - VERDICT) / 0.35) : 0, cur = shown ? q.to : q.from;
    responseCard(E, { x: cx, y: my + mh, w: cw, h: ch }, shown
      ? { title: changed ? `Revised response: ${CANDIDATES[cur].label.toLowerCase()}` : 'Kept as it is', text: TEXT[x][cur], size: 12, maxLines: 3, tag: plural(q.after, 'violation'), accent: changed, op: 0.25 + 0.75 * a }
      : { title: 'Revised response', text: '…', size: 12, op: 0.4 });
    // the responses visited so far in this chain
    const chain = chainOf(st, f.r, f.ph), by = my + mh + ch + 46, cwid = (W - 2 * pad) / chain.length;
    text(E, 'Responses so far', pad, by - 20, { size: 11, fill: MUTED });
    chain.forEach((y, i) => {
      const bx = pad + i * cwid;
      rect(E, bx, by - 9, 11, 11, { fill: candColor(y), rx: 2 });
      text(E, clip(CANDIDATES[y].short, charsFor(cwid - 40, 12)), bx + 16, by + 1, { size: 12, weight: i === chain.length - 1 ? 650 : 400 });
      if (i < chain.length - 1) text(E, '→', bx + cwid - 20, by + 1, { size: 12, fill: MUTED });
    });
  },
  label: (s) => `The critique and revision loop: for each of ${s.shown.length} starting responses, ${s.rounds} rounds in which one principle is picked, the response is checked and rewritten if it breaks the principle. Over ${s.n} starts, the principles broken per response fall from ${fmt(s.before, 2)} to ${fmt(s.after, 2)}.`,
};
export default mode;
