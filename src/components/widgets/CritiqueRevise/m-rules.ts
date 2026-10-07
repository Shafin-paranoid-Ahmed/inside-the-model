/**
 * rules (constitutional-ai step 1): the constitution as a checklist. A scan line moves down the 8 candidates of one
 * prompt; for each, the principles are checked one by one and a cross (--viz-neg) marks each violation, a small dot each
 * pass. Props: prompt (0), principles ("answer,accurate,no-flattery,no-refuse"; add "cautious" to show the extra one).
 * Measured (prompt 0): violations per candidate 1 0 0 1 1 1 2 0 (continue, terse, clear, padded, rambling, terse wrong,
 * refusal, hedged): only terse correct, clear correct and hedged correct break no principle.
 */
import { CANDIDATES, PROMPTS, TEXT, phi, K } from '../../../lib/align/bank';
import { PRINCIPLES } from '../../../lib/align/constitution';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, RULE, TEXT as INK, circle, clamp, line, mix, rect, text, type Env } from '../_rl/svg';
import { candColor, charsFor, clip } from '../_align/draw';
import { principlesOf, principleText, SHORT } from './sim';

const PER = 1.3, HOLD = 3;
interface Scene { x: number; act: string[]; v: boolean[][]; counts: number[] }
interface F extends Frame { row: number; checked: number; done: boolean }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) {
    const x = Math.max(0, Math.min(PROMPTS.length - 1, int(o.prompt, 0))), act = principlesOf(o);
    const v = Array.from({ length: K }, (_, y) => act.map((id) => PRINCIPLES.find((p) => p.id === id)!.violated(phi(x, y))));
    return { x, act, v, counts: v.map((r) => r.filter(Boolean).length) };
  },
  timeline: () => ({ duration: K * PER + HOLD, markers: [{ at: 0, label: 'Checking each response' }, { at: K * PER, label: 'The verdict' }] }),
  frameAt(t, s) {
    const np = s.act.length, done = t >= K * PER, row = done ? K - 1 : Math.floor(t / PER), checked = done ? np : clamp(Math.floor(((t - row * PER) / PER) * (np + 1)), 0, np);
    const clean = CANDIDATES.filter((_, y) => s.counts[y] === 0).map((c) => c.label.toLowerCase());
    const seen = done ? K : row + (checked >= np ? 1 : 0), cleanSoFar = s.counts.slice(0, seen).filter((c) => c === 0).length;
    let status: string;
    if (done) status = `Only ${clean.slice(0, -1).join(', ')}${clean.length > 1 ? ' and ' : ''}${clean[clean.length - 1]} break no principle.`;
    else if (checked === 0) status = `Next: “${CANDIDATES[row].label.toLowerCase()}”.`;
    else {
      const j = checked - 1, bad = s.v[row][j];
      status = `“${CANDIDATES[row].label}” against “${principleText(s.act[j])}”: ${bad ? 'violated' : 'fine'}.`;
    }
    const cur = s.v[row].slice(0, checked).filter(Boolean).length;
    return { row, checked, done, status, readouts: [['Response', `${row + 1} of ${K}`], ['Violations of this response', done ? String(s.counts[row]) : String(cur)], ['Responses with no violation so far', `${cleanSoFar} of ${seen}`]] };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, pad = nw ? 8 : 12, np = s.act.length;
    text(E, clip(`User: ${PROMPTS[s.x].q}`, charsFor(W - 2 * pad, 13)), pad, 16, { size: 13, weight: 600 });
    // the constitution, numbered
    const listTop = 34, lh = nw ? 16 : 17;
    s.act.forEach((id, j) => text(E, `${j + 1}. ${clip(principleText(id), charsFor(W - 2 * pad - 14, 12))}`, pad, listTop + 12 + j * lh, { size: nw ? 11 : 12, fill: f.checked > j && !f.done ? INK : MUTED, weight: f.checked === j + 1 && !f.done ? 650 : 400 }));
    const top = listTop + np * lh + 26, cellW = nw ? 30 : 64, gx = W - pad - np * cellW - (nw ? 26 : 70), rh = (E.H - top - 6) / K;
    s.act.forEach((id, j) => text(E, nw ? String(j + 1) : SHORT[id] ?? id, gx + j * cellW + cellW / 2, top - 6, { size: 11, anchor: 'middle', fill: MUTED }));
    text(E, nw ? 'Σ' : 'Violations', W - pad, top - 6, { size: 11, anchor: 'end', fill: MUTED });
    for (let y = 0; y < K; y++) {
      const ry = top + y * rh, cy = ry + rh / 2, past = f.done || y < f.row, now = !f.done && y === f.row;
      if (now) rect(E, pad - 4, ry + 1, W - 2 * pad + 8, rh - 2, { fill: mix(ACCENT, BG, 10), stroke: ACCENT, sw: 1.5, rx: 4 });
      rect(E, pad, cy - 5, 10, 10, { fill: candColor(y), rx: 2 });
      const labW = gx - pad - 22;
      if (nw) text(E, clip(CANDIDATES[y].short, charsFor(labW, 11)), pad + 16, cy + 4, { size: 11 });
      else { text(E, CANDIDATES[y].short, pad + 16, cy - 2, { size: 11, fill: MUTED }); text(E, clip(TEXT[s.x][y], charsFor(labW, 12)), pad + 16, cy + 12, { size: 12 }); }
      for (let j = 0; j < np; j++) {
        const cx = gx + j * cellW + cellW / 2, shown = past || (now && f.checked > j);
        if (!shown) { circle(E, cx, cy, 2, { fill: RULE }); continue; }
        if (s.v[y][j]) { line(E, cx - 5, cy - 5, cx + 5, cy + 5, { stroke: 'var(--viz-neg)', sw: 2.5 }); line(E, cx - 5, cy + 5, cx + 5, cy - 5, { stroke: 'var(--viz-neg)', sw: 2.5 }); }
        else circle(E, cx, cy, 3, { fill: MUTED });
      }
      if (past || (now && f.checked === np)) text(E, String(s.counts[y]), W - pad, cy + 4, { size: 12, anchor: 'end', weight: s.counts[y] ? 650 : 400, fill: s.counts[y] ? 'var(--viz-neg)' : MUTED });
    }
  },
  label: (s) => `A constitution of ${s.act.length} principles checked against the 8 candidate responses to “${PROMPTS[s.x].q}”; violations per response: ${s.counts.join(', ')}.`,
};
export default mode;
