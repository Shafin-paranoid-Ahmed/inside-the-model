/**
 * domains (rlvr step 7): one GRPO run on a mix of addition problems (a verifier exists) and two open prompts (no verifier:
 * reward 0, so every group has advantage 0). Left: the policy's style probabilities on 3-digit sums move; right: on the open
 * prompts they stay exactly at the start. Bottom: pass@1 and pass@8 on the sums (mean of the four digit counts) with the
 * start values marked. Props: iters (150), G (8), problems (8), seed (5). Measured: pass@8 at start 0.65, pass@1 at 150 1.00
 * (0.997), pass@1 at start 0.29; change on the open prompts exactly 0.
 */
import type { GRPORun } from '../../../lib/align/verifier';
import { rowProbs, OPEN, startLogits } from '../../../lib/align/verifier';
import { type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, RULE, TEXT, clamp, fmt, lerp, line, rect, text, type Env } from '../_rl/svg';
import { iterAt, runProps, runTimeline, styleBars, trainRun } from './vr-common';

interface F extends Frame { it: number; k: number }
const mean = (a: Float64Array, k: number) => (a[k * 4] + a[k * 4 + 1] + a[k * 4 + 2] + a[k * 4 + 3]) / 4;
/** Largest change in any style probability on the open prompts since the start. */
export const openChange = (r: GRPORun, k: number) => { const a = rowProbs(r.logits[k], OPEN), b = rowProbs(r.logits[0], OPEN); return Math.max(...Array.from(a, (v, i) => Math.abs(v - b[i]))); };

const mode: ModeImpl<GRPORun, F> = {
  setup: (o: Opts, kit) => trainRun(runProps(o, 'strict', true), kit),
  timeline: (r) => runTimeline(r.iters),
  frameAt(t, r) {
    const it = iterAt(t, r.iters), k = Math.floor(it + 1e-9), p1 = mean(r.strict, k), p8 = mean(r.pass8, k);
    return {
      it, k, status: `Iteration ${k}: on sums, one attempt passes ${fmt(p1, 2)} of the time and the best of eight ${fmt(p8, 2)}; on prompts with no checker the policy has changed by ${fmt(openChange(r, k), 3)}.`,
      readouts: [['Iteration', `${k} of ${r.iters}`], ['pass@1, sums', fmt(p1, 2)], ['pass@8, sums', fmt(p8, 2)], ['Start: pass@8', fmt(mean(r.pass8, 0), 2)], ['Change, no checker', fmt(openChange(r, k), 3)]],
    };
  },
  draw(E: Env, f, r) {
    const nw = E.narrow, W = E.W, pm = nw ? 8 : 12, k1 = Math.min(r.iters, f.k + 1), top = nw ? 250 : 236, gap = 16;
    const p0 = rowProbs(startLogits(), 2), tw = (row: number) => { const a = rowProbs(r.logits[f.k], row), b = rowProbs(r.logits[k1], row); return a.map((v, i) => lerp(v, b[i], f.it - f.k)); };
    const lw = nw ? W - 2 * pm : (W - 2 * pm - gap) / 2;
    styleBars(E, { x: pm, y: 4, w: lw, h: top }, tw(2), 3, { title: '3-digit sums (a checker exists)', ref: p0, values: rowProbs(r.logits[f.k], 2) });
    const b2 = nw ? { x: pm, y: 4 + top + 10, w: lw, h: top } : { x: pm + lw + gap, y: 4, w: lw, h: top };
    styleBars(E, b2, tw(OPEN), 3, { title: 'Open prompts, e.g. a poem (no checker)', ref: p0, values: rowProbs(r.logits[f.k], OPEN) });
    const y0 = nw ? b2.y + top + 22 : top + 28, labW = nw ? 92 : 144, bx = pm + labW, bw = W - 2 * pm - labW - 40, sz = nw ? 11 : 12;
    text(E, 'Sums: passes now (bar), at the start (tick)', pm, y0 - 8, { size: sz, fill: MUTED });
    ([['pass@1: one attempt', mean(r.strict, f.k), mean(r.strict, 0)], ['pass@8: best of eight', mean(r.pass8, f.k), mean(r.pass8, 0)]] as const).forEach(([l, v, v0], i) => {
      const y = y0 + 6 + i * 30;
      text(E, nw ? l.split(':')[0] : l, pm, y + 13, { size: sz });
      rect(E, bx, y, bw, 18, { fill: 'none', stroke: RULE, sw: 1 });
      rect(E, bx, y, clamp(v, 0, 1) * bw, 18, { fill: 'var(--viz-forward)', op: 0.9 });
      line(E, bx + v0 * bw, y - 3, bx + v0 * bw, y + 21, { stroke: TEXT, sw: 2 });
      text(E, fmt(v, 2), bx + bw + 6, y + 13, { size: sz });
    });
  },
  label: (r) => `One GRPO run: policy probabilities on sums move, on prompts with no checker they do not; pass@1 and pass@8 on sums over ${r.iters} iterations.`,
};
export default mode;
