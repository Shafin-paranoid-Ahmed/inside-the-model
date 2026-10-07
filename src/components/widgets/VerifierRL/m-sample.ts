/**
 * sample (rlvr step 2): the start policy answers six problems eight times each. Attempts fill in row by row; each cell
 * shows how many lines of working the attempt wrote (bar) and the verdict (tick or cross, strict verifier); a row ends
 * with its pass count. Readouts count the attempts shown so far. Props: G (8), problems (6), seed (0).
 * Measured (seed 0): 13 of 48 attempts pass (pass@1 0.271); 4 of 6 problems pass at least once (pass@8 0.667); in 317 + 833
 * only the attempt that wrote all three columns passes.
 */
import type { Groups } from '../../../lib/align/verifier';
import { kOf, isList } from '../../../lib/align/verifier';
import { ACCENT, BG, MUTED, RULE, clamp, fmt, mix, rect, text, type Env } from '../_rl/svg';
import { type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { cross, tick } from './sim';
import { attempts, sum } from './vr-common';

const CELL = 0.17, ROW = 1.9, HOLD = 3;
interface F extends Frame { n: number; row: number }

/** Attempts shown at time t (cells appear one by one, row by row). */
export const shownAt = (t: number, G: number, rows: number) => {
  const r = Math.min(rows - 1, Math.floor(t / ROW)), within = Math.min(G, Math.floor(Math.max(0, t - r * ROW) / CELL) + 1);
  return t >= rows * ROW ? rows * G : r * G + within;
};
const name = (g: Groups, j: number) => `${g.rows[j].p!.a} + ${g.rows[j].p!.b}`;

const mode: ModeImpl<Groups, F> = {
  setup: (o: Opts) => attempts(o),
  timeline: (g) => ({ duration: g.rows.length * ROW + HOLD, markers: g.rows.map((_, j) => ({ at: j * ROW, label: name(g, j) })) }),
  frameAt(t, g) {
    const G = g.rows[0].actions.length, R = g.rows.length, n = shownAt(t, G, R), row = Math.min(R - 1, Math.floor((n - 1) / G)), done = n >= R * G;
    const passes = (j: number, upto: number) => sum(g.rows[j].strict.slice(0, upto));
    let total = 0, solved = 0;
    for (let j = 0; j < R; j++) { const c = clamp(n - j * G, 0, G), p = passes(j, c); total += p; if (p > 0) solved++; }
    const c = clamp(n - row * G, 0, G), pr = passes(row, c), r = g.rows[row];
    const status = done ? `All ${R * G} attempts: ${total} pass, so pass@1 is ${fmt(total / (R * G), 3)}; ${solved} of ${R} problems were solved at least once, so pass@${G} is ${fmt(solved / R, 3)}.`
      : `${name(g, row)}: ${c} of ${G} attempts written, ${pr} pass so far.${c === G && pr === 1 && r.p!.digits >= 3 ? ' The passing attempt wrote out all its working.' : ''}`;
    return { n, row, status, readouts: [['Attempts shown', `${n} of ${R * G}`], ['Passed', String(total)], [`Problems solved at least once`, `${solved} of ${R}`]] };
  },
  draw(E: Env, f, g) {
    const nw = E.narrow, R = g.rows.length, G = g.rows[0].actions.length, pad = nw ? 6 : 12, labW = nw ? 82 : 92, cntW = nw ? 30 : 44, gap = 3;
    const cw = (E.W - 2 * pad - labW - cntW - (G - 1) * gap) / G, rh = (E.H - 8) / R, sz = nw ? 11 : 12;
    g.rows.forEach((r, j) => {
      const y0 = 4 + j * rh, c = clamp(f.n - j * G, 0, G), complete = c === G, cur = j === f.row && f.n < R * G;
      if (cur) rect(E, pad - 4, y0, E.W - 2 * pad + 8, rh - 3, { fill: mix(ACCENT, BG, 10), stroke: ACCENT, sw: 1.5, rx: 5 });
      text(E, `${r.p!.a} + ${r.p!.b}`, pad, y0 + rh / 2 + 4, { size: sz, weight: 650, op: c ? 1 : 0.45 });
      for (let i = 0; i < G; i++) {
        const x = pad + labW + i * (cw + gap), shown = i < c, last = i === c - 1 && cur;
        rect(E, x, y0 + 3, cw, rh - 9, { fill: shown ? mix('var(--text)', BG, 4) : 'none', stroke: RULE, sw: 1, rx: 3, op: shown ? 1 : 0.5 });
        if (!shown) continue;
        const op = last ? 0.7 : 1, k = Math.min(kOf(r.actions[i]), r.p!.digits), bh = (rh - 17) * (k / 4);
        rect(E, x + 3, y0 + rh - 9 - bh, 5, bh, { fill: MUTED, op });
        (r.strict[i] ? tick : cross)(E, x + cw / 2 + 3, y0 + (rh - 6) / 2 - (isList(r.actions[i]) ? 3 : 0), Math.min(7, cw / 4), op);
        if (isList(r.actions[i])) text(E, 'list', x + cw / 2 + 3, y0 + rh - 11, { size: 11, anchor: 'middle', fill: MUTED });
      }
      if (c) { const p = sum(r.strict.slice(0, c)); text(E, `${p}/${complete ? G : c}`, E.W - pad, y0 + rh / 2 + 4, { size: sz, anchor: 'end', weight: 650, fill: complete ? (p ? 'var(--viz-pos)' : 'var(--viz-neg)') : MUTED }); }
    });
  },
  label: (g) => `${g.rows.length} addition problems, ${g.rows[0].actions.length} attempts each, with the working written and a tick or cross from the strict verifier.`,
};
export default mode;
