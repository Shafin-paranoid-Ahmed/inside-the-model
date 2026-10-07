/**
 * grpo (rlvr step 3): the same attempts as `sample` (same seed), each scored against its group: advantage = (reward -
 * group mean) / group std. Rows go one by one: the advantage bars grow, a row where every attempt got the same reward is
 * greyed and labelled "no signal". The last segment shows the push on the policy for one problem (prop `focus`, default
 * the 3-digit row): the push on a style is the sum of the advantages of the attempts that chose it (the policy-gradient
 * step on a group, whose advantages sum to 0, moves each logit in proportion to that sum). Props: G, problems, seed (0), focus.
 * Measured (seed 0): 3 of 6 rows have no signal (one all-pass, two all-fail); the largest advantage is +2.646 (317 + 833,
 * one pass among eight); push there: +2.65 on all three columns, -1.89 on no working.
 */
import { kOf, isList, rowProbs, startLogits, type Groups } from '../../../lib/align/verifier';
import { ACCENT, BG, MUTED, RULE, clamp, ease, fmt, line, mix, rect, text, type Env } from '../_rl/svg';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { cross, tick } from './sim';
import { sum, attempts, nameOf, styleBars, styleRows } from './vr-common';

const ROW = 1.5, GROW = 0.9, LAST = 5, MAXADV = 2.7;
interface Scene { g: Groups; focus: number }
interface F extends Frame { row: number; u: number; judged: number; last: boolean; lu: number }
const name = (g: Groups, j: number) => `${g.rows[j].p!.a} + ${g.rows[j].p!.b}`;
const flat = (r: Groups['rows'][number]) => Math.max(...r.adv) - Math.min(...r.adv) < 1e-12;
/** Total advantage of the attempts that chose each style row of the focus problem. */
export function pushOf(r: Groups['rows'][number]): number[] {
  const d = r.p!.digits;
  return styleRows(d).map(({ f, k }) => { let s = 0; r.actions.forEach((a, i) => { if ((isList(a) ? 1 : 0) === f && Math.min(kOf(a), d) === k) s += r.adv[i]; }); return s; });
}

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) { const g = attempts(o), three = g.rows.findIndex((r) => r.p!.digits === 3); return { g, focus: int(o.focus, three >= 0 ? three : 0) }; },
  timeline: (s) => ({ duration: s.g.rows.length * ROW + LAST + 1, markers: [...s.g.rows.map((_, j) => ({ at: j * ROW, label: name(s.g, j) })), { at: s.g.rows.length * ROW, label: 'The push' }] }),
  frameAt(t, s) {
    const R = s.g.rows.length, last = t >= R * ROW, row = last ? R - 1 : Math.floor(t / ROW), u = last ? 1 : (t - row * ROW) / ROW, judged = last ? R : row + (u >= GROW / ROW ? 1 : 0);
    const r = s.g.rows[row], G = r.actions.length, p = sum(r.strict);
    const zeros = s.g.rows.slice(0, judged).filter(flat).length, mx = Math.max(0, ...s.g.rows.slice(0, judged).flatMap((x) => Array.from(x.adv)));
    const fr = s.g.rows[s.focus], fp = pushOf(fr), fd = fr.p!.digits, rowsF = styleRows(fd), hi = fp.indexOf(Math.max(...fp)), lo = fp.indexOf(Math.min(...fp)), lu = clamp((t - R * ROW) / 1.5, 0, 1);
    const status = last ? `${name(s.g, s.focus)}: the update pushes “${nameOf(rowsF[hi], fd)}” up by ${fmt(fp[hi], 2)} and “${nameOf(rowsF[lo], fd)}” down by ${fmt(-fp[lo], 2)}.`
      : u < GROW / ROW ? `${name(s.g, row)}: ${p} of ${G} attempts passed. Each attempt is scored against the group's average.`
      : flat(r) ? `${name(s.g, row)}: ${p === 0 ? 'every attempt failed' : 'every attempt passed'}, so every advantage is 0 and this problem teaches nothing.`
      : `${name(s.g, row)}: ${p} of ${G} passed. A pass is worth ${fmt(Math.max(...r.adv), 2)}, a fail ${fmt(Math.min(...r.adv), 2)}.`;
    return { row, u, judged, last, lu, status, readouts: [['Problems judged', `${judged} of ${R}`], ['With no signal', String(zeros)], ['Largest advantage', judged ? '+' + fmt(mx, 2) : '–']] };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, R = s.g.rows.length, G = s.g.rows[0].actions.length, pad = nw ? 6 : 12, labW = nw ? 82 : 84, gap = 3, sz = nw ? 11 : 12;
    const gw = nw ? E.W : Math.round(E.W * 0.56), gh = nw ? Math.round(E.H * 0.5) : E.H, cw = (gw - pad - labW - 8 - (G - 1) * gap) / G, rh = (gh - 8) / R;
    s.g.rows.forEach((r, j) => {
      const y0 = 4 + j * rh, mid = y0 + rh / 2 + 5, seen = j < f.row || f.last, now = j === f.row && !f.last, grow = seen ? 1 : now ? ease(f.u / (GROW / ROW)) : 0, zero = flat(r), dim = zero && j < f.judged;
      if (now) rect(E, pad - 4, y0, gw - pad + 2, rh - 3, { fill: mix(ACCENT, BG, 10), stroke: ACCENT, sw: 1.5, rx: 5 });
      text(E, `${r.p!.a} + ${r.p!.b}`, pad, y0 + rh / 2 + 4, { size: sz, weight: 650, op: seen || now ? (dim ? 0.5 : 1) : 0.45 });
      line(E, pad + labW, mid, pad + labW + G * (cw + gap) - gap, mid, { stroke: RULE, sw: 1 });
      for (let i = 0; i < G; i++) {
        const x = pad + labW + i * (cw + gap), a = r.adv[i], h = (a / MAXADV) * (rh / 2 - 12) * grow, op = dim ? 0.35 : 1;
        if (!(seen || now)) { rect(E, x, y0 + 6, cw, rh - 14, { fill: 'none', stroke: RULE, sw: 1, rx: 3, op: 0.5 }); continue; }
        (r.strict[i] ? tick : cross)(E, x + cw / 2, y0 + 10, Math.min(5, cw / 4), op);
        rect(E, x + cw * 0.15, a >= 0 ? mid - h : mid, cw * 0.7, Math.abs(h), { fill: a >= 0 ? 'var(--viz-pos)' : 'var(--viz-neg)', op });
      }
      if (dim) text(E, 'no signal', gw - 4, y0 + rh - 8, { size: 11, anchor: 'end', fill: MUTED });
    });
    // the push on the focus problem
    const fr = s.g.rows[s.focus], d = fr.p!.digits, box = nw ? { x: pad, y: gh + 8, w: E.W - 2 * pad, h: E.H - gh - 12 } : { x: gw + 14, y: 6, w: E.W - gw - 14 - pad, h: E.H - 12 };
    const push = pushOf(fr), prob = rowProbs(startLogits(), d - 1);
    styleBars(E, box, prob, d, { title: `${fr.p!.a} + ${fr.p!.b}: style probabilities`, push, pushGrow: f.last ? ease(f.lu) : 0, pushMax: MAXADV, faded: !f.last });
  },
  label: (s) => `Advantages of ${s.g.rows.length} groups of attempts, and the push on the policy for ${name(s.g, s.focus)}.`,
};
export default mode;
