/**
 * data (sft step 2): the demonstration set, one pair per row, arriving in order, with a running tally of what is in it
 * (answers, correct, hedges, flattery, mean length). Props: demos (24), demoAccuracy (0.8), seed (2). Measured
 * (demos 24, accuracy 0.8, seed 2): 19 of 24 correct (exact counts), mean length 5.4 words, flattery 0, hedged 5 of 24.
 */
import { CANDIDATES, F_ANSWERS, F_CORRECT, F_FLATTERY, F_HEDGES, F_LENGTH, PROMPTS, TEXT, phi } from '../../../lib/align/bank';
import type { Demo } from '../../../lib/align/sft';
import { type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, RULE, ease, fmt, rect, text, type Env } from '../_rl/svg';
import { charsFor, clip } from '../_align/draw';
import { cross, tick } from '../VerifierRL/sim';
import { sftRun } from './sim';

const PER = 0.5, FADE = 0.35, HOLD = 3;
interface Scene { demos: Demo[]; n: number; tally: { answers: number; correct: number; hedges: number; flattery: number; words: number }[]; wrong: number }
interface F extends Frame { d: number; fade: number }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const { demos } = sftRun({ ...o, steps: 1 }, kit);
    const tally = [{ answers: 0, correct: 0, hedges: 0, flattery: 0, words: 0 }];
    demos.forEach((dm, i) => {
      const f = phi(dm.x, dm.y), p = tally[i];
      tally.push({ answers: p.answers + f[F_ANSWERS], correct: p.correct + f[F_CORRECT], hedges: p.hedges + f[F_HEDGES], flattery: p.flattery + f[F_FLATTERY], words: p.words + f[F_LENGTH] * 10 });
    });
    return { demos, n: demos.length, tally, wrong: demos.filter((dm) => !phi(dm.x, dm.y)[F_CORRECT]).length };
  },
  timeline: (s) => ({ duration: s.n * PER + HOLD, markers: [{ at: 0, label: 'Demonstrations arrive' }, { at: s.n * PER, label: 'All collected' }] }),
  frameAt(t, s) {
    const d = Math.min(s.n, Math.floor(t / PER + 1e-9) + 1), fade = t >= s.n * PER ? 1 : ease((t - (d - 1) * PER) / FADE), tl = s.tally[d];
    const dm = s.demos[d - 1], ok = phi(dm.x, dm.y)[F_CORRECT] > 0;
    const status = d === s.n && t >= s.n * PER
      ? `All ${s.n} demonstrations: ${tl.correct} correct and ${s.wrong} wrong. The orange crosses mark the wrong ones: the people writing demonstrations make mistakes too.`
      : `Demonstration ${d} of ${s.n}: “${PROMPTS[dm.x].q}” answered “${TEXT[dm.x][dm.y]}” ${ok ? '(' + CANDIDATES[dm.y].label.toLowerCase() + ', correct).' : '(wrong).'}`;
    return {
      d, fade, status,
      readouts: [['Demonstrations', `${d} of ${s.n}`], ['Correct so far', `${tl.correct} of ${d} (${fmt((100 * tl.correct) / d, 0)}%)`], ['Mean length', `${fmt(tl.words / d, 1)} words`]],
    };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, pad = nw ? 8 : 12, sz = nw ? 11 : 12, top = 26;
    const listW = nw ? W - 2 * pad : Math.round((W - 2 * pad) * 0.66), rh = nw ? 15.2 : Math.min(18.5, (E.H - top - 8) / s.n);
    const pw = nw ? Math.round(listW * 0.46) : Math.round(listW * 0.42), rw = listW - 22 - pw - 26;
    text(E, 'Prompt', pad + 22, 15, { size: 11, fill: MUTED });
    text(E, 'Demonstrated response', pad + 22 + pw, 15, { size: 11, fill: MUTED });
    s.demos.forEach((dm, i) => {
      if (i >= f.d) return;
      const y = top + i * rh, op = i === f.d - 1 ? f.fade : 1, ok = phi(dm.x, dm.y)[F_CORRECT] > 0;
      if (!ok) rect(E, pad - 2, y, listW + 4, rh - 1, { fill: 'var(--viz-neg)', op: 0.14 * op, rx: 3 });
      text(E, String(i + 1), pad + 14, y + rh * 0.72, { size: 11, anchor: 'end', fill: MUTED, op });
      text(E, clip(PROMPTS[dm.x].q, charsFor(pw - 8, sz)), pad + 22, y + rh * 0.72, { size: sz, op });
      text(E, clip(TEXT[dm.x][dm.y], charsFor(rw, sz)), pad + 22 + pw, y + rh * 0.72, { size: sz, op });
      if (ok) tick(E, pad + listW - 10, y + rh / 2, 5, op); else cross(E, pad + listW - 10, y + rh / 2, 4.5, op);
    });
    // tally panel
    const tl = s.tally[f.d];
    const tx = nw ? pad : pad + listW + 22, tw = nw ? W - 2 * pad : W - tx - pad, ty = nw ? top + s.n * rh + 14 : top - 8;
    text(E, `Tally of the demonstrations so far`, tx, ty + 8, { size: 11, fill: MUTED });
    const rows: [string, number, string][] = [
      ['Answers the question', tl.answers / f.d, `${fmt((100 * tl.answers) / f.d, 0)}%`],
      ['Correct', tl.correct / f.d, `${fmt((100 * tl.correct) / f.d, 0)}%`],
      ['Hedges', tl.hedges / f.d, `${fmt((100 * tl.hedges) / f.d, 0)}%`],
      ['Flatters the user', tl.flattery / f.d, `${fmt((100 * tl.flattery) / f.d, 0)}%`],
      ['Mean length', Math.min(1, tl.words / f.d / 10), `${fmt(tl.words / f.d, 1)} words`],
    ];
    const rowH = nw ? 36 : 44;
    rows.forEach(([label, v, val], i) => {
      const y = ty + 22 + i * rowH, bh = 12;
      text(E, label, tx, y + 10, { size: sz });
      text(E, val, tx + tw, y + 10, { size: sz, anchor: 'end' });
      rect(E, tx, y + 16, tw, bh, { fill: 'none', stroke: RULE, sw: 1, rx: 2 });
      rect(E, tx, y + 16, Math.max(0, v) * tw, bh, { fill: 'var(--viz-forward)', op: 0.9, rx: 2 });
    });
    if (!nw) text(E, 'Length bar: 0 to 10 words', tx, ty + 22 + 5 * rowH - 6, { size: 11, fill: MUTED });
  },
  label: (s) => `${s.n} demonstration pairs, one per row, and a running tally: ${s.tally[s.n].correct} of ${s.n} are correct, mean length ${fmt(s.tally[s.n].words / s.n, 1)} words, none flatter.`,
};
export default mode;
