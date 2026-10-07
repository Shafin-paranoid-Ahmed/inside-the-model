/**
 * dpo-pair (dpo step 2): repeated DPO gradient steps (trainDPO, lr 0.5) on ONE pair from the SFT reference. Two rows show
 * the chosen and the rejected response (policy probability with the reference as a ghost bar, the two log-probabilities and
 * the log-ratio), arrows show the push (length = the gradient weight), and the plot shows the weight sigma(-margin) with
 * the current margin on it. Every number is the real state after n whole steps (no tweening of numbers).
 * Props: pair (0), beta (0.1), steps (40), lr (0.5), seed (4, pair seed). Measured pair 0 ("What is half of 30?", terse
 * correct over clear correct): loss 0.693 -> 0.383, margin 0 -> 0.762, gradient weight 0.500 -> 0.318 (plans/day-11.md section 7).
 */
import { CANDIDATES, PROMPTS, TEXT } from '../../../lib/align/bank';
import { dpoMargin, trainDPO } from '../../../lib/align/dpo';
import { probsInto } from '../../../lib/align/policy';
import { sigmoid, type Pair } from '../../../lib/align/reward';
import { int, key, num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, RULE, TEXT as INK, arrow, axes, circle, clamp, fmt, line, path, rect, signFmt, text, type Env } from '../_rl/svg';
import { candColor, charsFor, clip, FORWARD } from '../_align/draw';
import { betaOf } from './dpoKit';
import { prefData } from './sim';

const LEAD = 1.6, STEP = 0.14, TAIL = 2.5;
interface Scene { q: Pair; beta: number; steps: number; pW: Float64Array; pL: Float64Array; rW: number; rL: number; margin: Float64Array; loss: Float64Array; weight: Float64Array }
interface F extends Frame { n: number }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const d = prefData(o, kit, int(o.seed, 4)), q = d.pairs[int(o.pair, 0)], beta = betaOf(o), steps = int(o.steps, 40), lr = num(o.lr, 0.5);
    return kit.memo(key('dpo-pair', int(o.seed, 4), int(o.pair, 0), beta, steps, lr), () => {
      const run = trainDPO([q], d.sft, { beta, steps, lr }), ref = probsInto(d.sft, q.x);
      const pW = new Float64Array(steps + 1), pL = new Float64Array(steps + 1), margin = new Float64Array(steps + 1);
      run.thetas.forEach((th, k) => { const p = probsInto(th, q.x); pW[k] = p[q.w]; pL[k] = p[q.l]; margin[k] = dpoMargin(th, d.sft, q, beta); });
      return { q, beta, steps, pW, pL, rW: ref[q.w], rL: ref[q.l], margin, loss: run.loss, weight: margin.map((m) => sigmoid(-m)) };
    });
  },
  timeline: (s) => ({ duration: LEAD + s.steps * STEP + TAIL, markers: [{ at: 0, label: 'The reference' }, { at: LEAD, label: 'Gradient steps' }, { at: LEAD + s.steps * STEP, label: 'Trained' }] }),
  frameAt(t, s) {
    const n = clamp(Math.floor((t - LEAD) / STEP + 1e-9), 0, s.steps), lw = Math.log(s.pW[n] / s.rW), ll = Math.log(s.pL[n] / s.rL);
    const status = n === 0
      ? `Before any step the policy is the reference: both log-ratios are 0, the margin is 0 and the gradient weight is σ(0) = ${fmt(s.weight[0], 2)}.`
      : `After ${n} step${n === 1 ? '' : 's'}: the chosen response's log-ratio is ${signFmt(lw)} and the rejected one's is ${signFmt(ll)}, so β times their difference, the margin, is ${fmt(s.margin[n], 3)}. The loss is ${fmt(s.loss[n], 3)} and the gradient weight σ(−margin) has become ${fmt(s.weight[n], 3)}.`;
    return {
      n, status,
      readouts: [['log π(chosen)', fmt(Math.log(s.pW[n]), 2)], ['log π_ref(chosen)', fmt(Math.log(s.rW), 2)], ['log π(rejected)', fmt(Math.log(s.pL[n]), 2)], ['log π_ref(rejected)', fmt(Math.log(s.rL), 2)], ['Margin', fmt(s.margin[n], 3)], ['Loss', fmt(s.loss[n], 3)], ['Gradient weight', fmt(s.weight[n], 3)]],
    };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, pad = nw ? 8 : 12, n = f.n, q = s.q;
    const lw = nw ? W : W * 0.56, rh = nw ? 100 : 128, top = 26;
    text(E, clip(`User: ${PROMPTS[q.x].q}`, charsFor(lw - 2 * pad, 13)), pad, 17, { size: 13, weight: 600 });
    const rows = [{ y: q.w, p: s.pW[n], r: s.rW, tag: 'Chosen', up: true }, { y: q.l, p: s.pL[n], r: s.rL, tag: 'Rejected', up: false }];
    rows.forEach((r, i) => {
      const y0 = top + i * (rh + 6), bx = pad + 4, bw = lw - 2 * pad - 52, cy = y0 + 62, by = cy - 8;
      rect(E, pad - 2, y0, lw - 2 * pad + 4, rh, { fill: 'none', stroke: RULE, sw: 1, rx: 6 });
      rect(E, bx, y0 + 9, 10, 10, { fill: candColor(r.y), rx: 2 });
      text(E, `${r.tag}: ${CANDIDATES[r.y].short.toLowerCase()}`, bx + 16, y0 + 18, { size: 12, weight: 650 });
      text(E, clip(`“${TEXT[q.x][r.y]}”`, charsFor(bw, 12)), bx, y0 + 38, { size: 12, fill: MUTED });
      rect(E, bx, by, bw, 16, { fill: 'none', stroke: RULE, sw: 1 });
      rect(E, bx, by, bw * r.p, 16, { fill: FORWARD, op: 0.9 });
      rect(E, bx, by - 2, bw * r.r, 20, { fill: 'none', stroke: MUTED, sw: 1.25, dash: '3 2' });
      text(E, `probability ${fmt(r.p, 2)} (reference ${fmt(r.r, 2)})`, bx, y0 + rh - 8, { size: 11, fill: INK });
      text(E, `log-ratio ${signFmt(Math.log(r.p / r.r))}`, bx + bw, y0 + rh - 8, { size: 11, anchor: 'end', fill: INK });
      const ax = pad + lw - 2 * pad - 20, len = 8 + 34 * s.weight[n];
      if (r.up) arrow(E, ax, y0 + 60 + 4, ax, y0 + 60 + 4 - len, { stroke: 'var(--viz-gradient)', width: 3 });
      else arrow(E, ax, y0 + 48, ax, y0 + 48 + len, { stroke: 'var(--viz-gradient)', width: 3 });
    });
    // gradient weight against the margin
    const pb = nw ? { x: 46, y: top + 2 * (rh + 6) + 30, w: W - 46 - 20, h: H - (top + 2 * (rh + 6) + 30) - 46 } : { x: lw + 44, y: 44, w: W - lw - 44 - 20, h: H - 44 - 56 };
    const { sx, sy } = axes(E, pb, { lo: -2, hi: 2, ticks: [-2, -1, 0, 1, 2], label: 'Margin' }, { lo: 0, hi: 1, ticks: [0, 0.5, 1], label: 'Gradient weight σ(−margin)' });
    let d = '';
    for (let i = 0; i <= 80; i++) { const m = -2 + (4 * i) / 80; d += `${i ? 'L' : 'M'}${sx(m).toFixed(1)} ${sy(sigmoid(-m)).toFixed(1)}`; }
    path(E, d, { stroke: 'var(--viz-gradient)', sw: 2 });
    line(E, sx(0), pb.y, sx(0), pb.y + pb.h, { stroke: RULE, sw: 1 });
    let tr = '';
    for (let i = 0; i <= n; i++) tr += `${i ? 'L' : 'M'}${sx(s.margin[i]).toFixed(1)} ${sy(s.weight[i]).toFixed(1)}`;
    path(E, tr, { stroke: INK, sw: 3 });
    const px = sx(s.margin[n]), py = sy(s.weight[n]);
    line(E, px, py, px, pb.y + pb.h, { stroke: ACCENT, sw: 1.25, dash: '3 3' });
    circle(E, px, py, 5, { fill: ACCENT, stroke: BG, sw: 2 });
    text(E, `margin ${fmt(s.margin[n], 2)}, weight ${fmt(s.weight[n], 2)}`, clamp(px + 10, pb.x + 4, pb.x + pb.w - 4), py - 12, { size: 11, anchor: px + 10 > pb.x + pb.w * 0.55 ? 'end' : 'start', halo: true });
  },
  label: (s) => `DPO gradient steps on one preference pair for the prompt "${PROMPTS[s.q.x].q}": probabilities of the chosen and rejected responses against the reference, and the gradient weight falling as the margin grows from 0 to ${fmt(s.margin[s.steps], 2)}.`,
};
export default mode;
