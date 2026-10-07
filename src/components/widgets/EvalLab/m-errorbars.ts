/**
 * errorbars (evaluations step 6): two models with a small true gap (skills 1.0 and 1.15) are compared on n questions. The
 * chance that the weaker one scores higher is computed in closed form (normal approximation to the difference of two
 * scores), once when each model gets its own random questions and once when both answer the same questions, where the
 * two answers on a question are strongly linked, so the difference is far less noisy.
 * A model answers question d right with probability sigmoid(SHARP (skill - d)): large SHARP means the question, not luck, decides.
 * Props: gap (0.15, the skill difference), seed (unused: the curves are exact; kept so the lesson can pin one).
 */
import { num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { BG, MUTED, TEXT, axes, circle, fmt, line, mix, path, rect, text, type Env } from '../_rl/svg';
import { expectOver, itemsNeeded, sigmoid, wrongWinner } from './sim';

export const SHARP = 4, MU = 0, SD = 1.2, SKILL_A = 1.0, N_LO = 25, N_HI = 2000, PLAY = 12, HOLD = 3;
export interface Scene {
  pA: number; pB: number; delta: number; sdUnpaired: number; sdPaired: number; nUnpaired: number; nPaired: number;
  bothRight: number; bothWrong: number; onlyA: number; onlyB: number;
}
export function makeErrorbars(gap: number): Scene {
  const f = (s: number) => expectOver(MU, SD, (d) => sigmoid(SHARP * (s - d))), sB = SKILL_A + gap, pA = f(SKILL_A), pB = f(sB);
  const both = expectOver(MU, SD, (d) => sigmoid(SHARP * (SKILL_A - d)) * sigmoid(SHARP * (sB - d))), delta = pB - pA;
  const sdUnpaired = Math.sqrt(pA * (1 - pA) + pB * (1 - pB)), sdPaired = Math.sqrt(pA + pB - 2 * both - delta * delta);
  return { pA, pB, delta, sdUnpaired, sdPaired, nUnpaired: itemsNeeded(delta, sdUnpaired), nPaired: itemsNeeded(delta, sdPaired), bothRight: both, bothWrong: 1 - pA - pB + both, onlyA: pA - both, onlyB: pB - both };
}
/** Questions per model at time t: a log sweep from N_LO to N_HI. */
export const nAt = (t: number) => Math.round(N_LO * (N_HI / N_LO) ** Math.max(0, Math.min(1, t / PLAY)));
const timeOf = (n: number) => (PLAY * Math.log(n / N_LO)) / Math.log(N_HI / N_LO);

interface F extends Frame { n: number; u: number; p: number }
const mode: ModeImpl<Scene, F> = {
  setup: (o: Opts) => makeErrorbars(num(o.gap, 0.15)),
  timeline: () => ({ duration: PLAY + HOLD, markers: [{ at: 0, label: '25 to 100' }, { at: timeOf(100), label: '100 to 500' }, { at: timeOf(500), label: '500 to 2,000' }] }),
  frameAt(t, s) {
    const n = nAt(t), u = wrongWinner(s.delta, s.sdUnpaired, n), p = wrongWinner(s.delta, s.sdPaired, n);
    const pc = (x: number) => (x < 0.005 ? 'under 1%' : `${fmt(x * 100, 0)}%`), status = `With ${n.toLocaleString('en-US')} questions per model, the weaker model scores higher ${pc(u)} of the time when each model gets its own questions, and ${pc(p)} of the time when both answer the same questions.`;
    return { n, u, p, status, readouts: [['Questions', n.toLocaleString('en-US')], ['Own questions', `${fmt(u * 100, 1)}%`], ['Same questions', `${fmt(p * 100, 1)}%`]] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, W = E.W, H = E.H, cw = nar ? W - 66 : W * 0.6, ch = nar ? H * 0.5 : H - 82, b = { x: nar ? 44 : 50, y: 30, w: cw, h: ch };
    const { sx, sy } = axes(E, b, { lo: N_LO, hi: N_HI, log: true, ticks: [25, 50, 100, 200, 500, 1000, 2000], label: 'Questions per model (log scale)', fmt: (v) => v.toLocaleString('en-US') }, { lo: 0, hi: 0.5, ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], label: 'Chance the weaker model comes out ahead', fmt: (v) => `${Math.round(v * 100)}%` });
    line(E, b.x, sy(0.05), b.x + b.w, sy(0.05), { stroke: TEXT, sw: 1.5, dash: '5 4' });
    const curve = (sd: number, upto: number) => { let d = ''; for (let i = 0; i <= 120; i++) { const n = N_LO * (N_HI / N_LO) ** (i / 120); if (n > upto + 1e-9) break; d += `${d ? 'L' : 'M'}${sx(n).toFixed(1)} ${sy(wrongWinner(s.delta, sd, n)).toFixed(1)}`; } return d; };
    const dU = curve(s.sdUnpaired, f.n), dP = curve(s.sdPaired, f.n);
    if (dU) path(E, dU, { stroke: 'var(--viz-cat-3)', sw: 2.75 });
    if (dP) path(E, dP, { stroke: 'var(--viz-cat-7)', sw: 2.75 });
    for (const [n, col, dy] of [[s.nUnpaired, 'var(--viz-cat-3)', 0], [s.nPaired, 'var(--viz-cat-7)', 0]] as const) {
      if (f.n < n) continue;
      line(E, sx(n), sy(0.05), sx(n), b.y + b.h, { stroke: col, sw: 1.5, dash: '2 3' });
      circle(E, sx(n), sy(0.05), 4.5, { fill: col, stroke: BG, sw: 1.5 });
      text(E, `${Math.round(n / 10) * 10}`, sx(n), sy(0.05) - 9 + dy, { size: 12, weight: 700, anchor: 'middle', fill: TEXT, halo: true });
    }
    circle(E, sx(f.n), sy(f.u), 5, { fill: 'var(--viz-cat-3)', stroke: BG, sw: 2 });
    circle(E, sx(f.n), sy(f.p), 5, { fill: 'var(--viz-cat-7)', stroke: BG, sw: 2 });
    // what the two models do on one question: the paired difference only cares about the two small slices
    const segs = [['Both right', s.bothRight, 'var(--viz-pos)'], ['Both wrong', s.bothWrong, 'var(--viz-neg)'], ['Only Model B right', s.onlyB, mix(MUTED, BG, 80)], ['Only Model A right', s.onlyA, mix(MUTED, BG, 45)]] as const;
    const bx = nar ? 44 : b.x + b.w + 54, bw = nar ? W - 56 : W - bx - 14, by = nar ? b.y + b.h + 70 : 54;
    text(E, 'How the two models answer one question', bx, by - 14, { size: 11, fill: MUTED });
    if (nar) {
      let x = bx; segs.forEach(([, v, c]) => { rect(E, x, by, v * bw, 22, { fill: c }); x += v * bw; });
      segs.forEach(([name, v, c], i) => { const yy = by + 44 + i * 18; rect(E, bx, yy - 9, 10, 10, { fill: c }); text(E, `${name}: ${fmt(v * 100, 0)}%`, bx + 16, yy, { size: 12 }); });
    } else {
      let y = by, prev = -99; const bh = ch - 40;
      segs.forEach(([name, v, c]) => {
        const h = Math.max(2, v * bh), ly = Math.max(y + h / 2 + 4, prev + 16);
        rect(E, bx, y, 20, h, { fill: c }); text(E, `${name}: ${fmt(v * 100, 0)}%`, bx + 28, ly, { size: 12 });
        y += h; prev = ly;
      });
    }
  },
  label: (s) => `The chance that the weaker of two close models scores higher, against questions per model from ${N_LO} to ${N_HI}, when each model gets its own questions and when both answer the same ones; they cross 5% near ${Math.round(s.nUnpaired / 10) * 10} and ${Math.round(s.nPaired / 10) * 10} questions.`,
};
export default mode;
