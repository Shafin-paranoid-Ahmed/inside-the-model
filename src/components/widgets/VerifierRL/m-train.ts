/**
 * train (rlvr steps 4 and 6): GRPO training against a verifier. Props: verifier ("strict" | "lenient"), iters (150), G (8),
 * problems (8), seed (5). The run plays in 14 s; numbers are those of the whole iteration reached, only geometry eases.
 *   strict:  accuracy per digit count (left), what the policy writes on 3-digit problems (right, dashed = start), a live
 *            sample answer refreshed every 10 iterations.
 *   lenient: one small chart per digit count with the reward the lenient verifier pays (solid), the strict accuracy it
 *            would really get (dashed) and the share of responses that are a guess list, plus a 2-digit sample.
 * Measured (seed 5): strict accuracy 1 / 2 / 3 / 4 digits 0.54 / 0.34 / 0.20 / 0.09 at 0, 0.99 / 0.99 / 0.98 / 0.95 at 50, 1.00 at 150.
 * Lenient: reward 1.00 at 150; guess-list share 0.56 / 0.71 / 0.02 / 0.01; strict accuracy 0.44 / 0.29 / 0.98 / 0.99.
 */
import type { GRPORun } from '../../../lib/align/verifier';
import { rowProbs, startLogits } from '../../../lib/align/verifier';
import { str, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { TEXT, axes, fmt, lerp, line, path, rect, text, type Env } from '../_rl/svg';
import { candColor } from '../_align/draw';
import { iterAt, liveSample, runProps, runTimeline, sampleCard, styleBars, trainRun } from './vr-common';

interface Scene { run: GRPORun; kind: 'strict' | 'lenient' }
interface F extends Frame { it: number; k: number }
const D = [0, 1, 2, 3], at = (a: Float64Array, k: number, d: number) => a[k * 4 + d];

/** Line of series values over iterations 0..it (fractional end interpolated), in chart axes sx / sy. */
const trace = (E: Env, vals: (k: number) => number, it: number, sx: (v: number) => number, sy: (v: number) => number, st: { stroke: string; sw: number; dash?: string }) => {
  const k = Math.floor(it);
  let d = '';
  for (let i = 0; i <= k; i++) d += `${d ? 'L' : 'M'}${sx(i).toFixed(1)} ${sy(vals(i)).toFixed(1)}`;
  if (it > k) d += `L${sx(it).toFixed(1)} ${sy(lerp(vals(k), vals(k + 1), it - k)).toFixed(1)}`;
  if (d) path(E, d, st);
};

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const kind = str(o.verifier, ['strict', 'lenient'] as const, 'strict');
    return { kind, run: await trainRun(runProps(o, kind), kit) };
  },
  timeline: (s) => runTimeline(s.run.iters),
  frameAt(t, s) {
    const it = iterAt(t, s.run.iters), k = Math.floor(it + 1e-9), r = s.run;
    const acc = D.map((d) => fmt(at(r.strict, k, d), 2)).join(' / ');
    const status = s.kind === 'strict'
      ? `Iteration ${k}: strict accuracy on 1, 2, 3 and 4 digit problems is ${acc}.`
      : `Iteration ${k}: the lenient verifier pays ${D.map((d) => fmt(at(r.lenient, k, d), 2)).join(' / ')}; the answers are really right ${acc} of the time, and ${D.map((d) => fmt(at(r.list, k, d), 2)).join(' / ')} of the responses are guess lists.`;
    return { it, k, status, readouts: s.kind === 'strict' ? [['Iteration', `${k} of ${r.iters}`], ['Accuracy', acc]] : [['Iteration', `${k} of ${r.iters}`], ['Guess-list share', D.map((d) => fmt(at(r.list, k, d), 2)).join(' / ')]] };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, r = s.run, n = r.iters, sz = nw ? 11 : 12, xa = { lo: 0, hi: n, ticks: [0, n / 3, (2 * n) / 3, n], label: 'Training iteration' };
    const pm = nw ? 8 : 12;
    if (s.kind === 'strict') {
      const cb = nw ? { x: 40, y: 22, w: W - 52, h: 190 } : { x: 46, y: 22, w: W * 0.5 - 56, h: H - 22 - 50 };
      const { sx, sy } = axes(E, cb, xa, { lo: 0, hi: 1, ticks: [0, 0.5, 1], label: 'Strict accuracy (share of attempts that pass)' });
      D.forEach((d) => {
        trace(E, (i) => at(r.strict, i, d), f.it, sx, sy, { stroke: candColor(d), sw: 2.25 });
        text(E, `${d + 1} digit${d ? 's' : ''}`, cb.x + 6, sy(at(r.strict, 0, d)) - 6, { size: 11, weight: 650, halo: true });
      });
      const rx = nw ? pm : cb.x + cb.w + 24, rw = nw ? W - 2 * pm : W - rx - pm, by = nw ? cb.y + cb.h + 52 : 6, bh = nw ? 240 : H * 0.56;
      const p = rowProbs(r.logits[f.k], 2), p1 = rowProbs(r.logits[Math.min(n, f.k + 1)], 2), p0 = rowProbs(startLogits(), 2), pt = p.map((v, i) => lerp(v, p1[i], f.it - f.k));
      styleBars(E, { x: rx, y: by, w: rw, h: bh }, pt, 3, { title: '3-digit problems: chance of writing each style', ref: p0, values: p });
      const ls = liveSample(r, f.it, 3, 'strict'), cy = by + bh + 8;
      sampleCard(E, { x: rx, y: cy, w: rw, h: 60 }, ls, 'A sampled answer');
    } else {
      const cw = nw ? W - 2 * pm : W * 0.6, pw = (cw - 80) / 2, ph = nw ? 104 : (H - 2 * 52) / 2 - 8;
      D.forEach((d) => {
        const bx = pm + 34 + (d % 2) * (pw + 34), by = 30 + Math.floor(d / 2) * (ph + 50), b = { x: bx, y: by, w: pw, h: ph };
        rect(E, bx, by - 20, 10, 10, { fill: candColor(d), rx: 2 });
        text(E, `${d + 1}-digit problems`, bx + 16, by - 10, { size: sz, weight: 650 });
        const { sx, sy } = axes(E, b, { lo: 0, hi: n, ticks: [0, n / 2, n] }, { lo: 0, hi: 1, ticks: [0, 1] });
        trace(E, (i) => at(r.list, i, d), f.it, sx, sy, { stroke: candColor(4), sw: 1.75 });
        trace(E, (i) => at(r.lenient, i, d), f.it, sx, sy, { stroke: 'var(--viz-forward)', sw: 2.25 });
        trace(E, (i) => at(r.strict, i, d), f.it, sx, sy, { stroke: TEXT, sw: 2, dash: '6 4' });
      });
      const rx = nw ? pm : pm + cw + 8, rw = nw ? W - 2 * pm : W - rx - pm;
      let y = nw ? 30 + 2 * (ph + 50) - 24 : 20;
      ([['Reward the lenient verifier pays', 'var(--viz-forward)', undefined], ['Answers really right (strict check)', TEXT, '6 4'], ['Responses that are a guess list', candColor(4), undefined]] as const).forEach(([l, c, dash]) => {
        line(E, rx, y - 4, rx + 22, y - 4, { stroke: c, sw: 2.25, dash });
        text(E, l, rx + 30, y, { size: sz });
        y += 18;
      });
      const ls = liveSample(r, f.it, 2, 'lenient'), h = sampleCard(E, { x: rx, y: y + 6, w: rw, h: 50 }, { ...ls, pass: ls.lenientPass }, 'A sampled answer');
      y += 6 + h + 18;
      text(E, `Lenient verifier: ${ls.lenientPass ? 'pass' : 'fail'}`, rx, y, { size: sz, weight: 650, fill: ls.lenientPass ? 'var(--viz-pos)' : 'var(--viz-neg)' });
      text(E, `Strict verifier: ${ls.strictPass ? 'pass' : 'fail'}`, rx, y + 18, { size: sz, weight: 650, fill: ls.strictPass ? 'var(--viz-pos)' : 'var(--viz-neg)' });
    }
  },
  label: (s) => `GRPO training with a ${s.kind} verifier over ${s.run.iters} iterations: accuracy and response style per digit count.`,
};
export default mode;
