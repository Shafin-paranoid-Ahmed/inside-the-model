/**
 * ppo-ratio (ppo step 2): the batch's visits to one cell. The probabilities that collected the batch are saved (outlines);
 * four epochs of updates move the new probabilities (filled bars); the ratio of the selected sample is new / old.
 * Props: focus ("1,1"), seed (29), epochs (4), minibatch (16), epsilon (0.2), lr (0.03), batch (64). Click a chip to pick a sample.
 */
import { createRng } from '../../../lib/math/rng';
import { ACTIONS, ARROWS } from '../../../lib/rl/gridworld';
import { ppoEpochs } from '../../../lib/rl/ppo';
import { softmaxRow } from '../../../lib/rl/tabular';
import { int, num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, TEXT, axes, circle, clamp, ease, fmt, inkOn, lerp, line, mix, path, rect, signedFill, text, type Env } from '../_rl/svg';
import { warmBatch } from './ppo-batch';
import { focusOf } from './sim';

const LEAD = 2.4, UPD = 0.7;
interface Scene { cell: number; idx: number[]; adv: number[]; act: number[]; probs: Float64Array[]; ups: number; per: number; eps: number }
interface F extends Frame { sel: number; g: number; n: number; ratio: number }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const { b, theta, V } = warmBatch(o, kit, 5, 29), cell = focusOf(o, 0), eps = num(o.epsilon, 0.2);
    const it = kit.memo(`ppo-ratio-${JSON.stringify([o.seed, o.stepReward, o.slip, o.gamma, o.lambda, o.batch, o.epochs, o.minibatch, o.lr, eps])}`, () =>
      ppoEpochs(b, theta.slice(), V.slice(), { rng: createRng(int(o.seed, 29)), eps, lr: num(o.lr, 0.03), epochs: int(o.epochs, 4), minibatch: int(o.minibatch, 16) }));
    const idx: number[] = [];
    for (let i = 0; i < b.s.length; i++) if (b.s[i] === cell) idx.push(i);
    const probs = [it.thetaOld, ...it.updates.map((u) => u.theta)].map((th) => softmaxRow(th, cell));
    return { cell, idx, adv: idx.map((i) => b.adv[i]), act: idx.map((i) => b.a[i]), probs, ups: it.updates.length, per: Math.max(1, it.updates.length / int(o.epochs, 4)), eps };
  },
  timeline(s) {
    const mk = [{ at: 0, label: 'Collect and save' }];
    for (let e = 0; e * s.per < s.ups; e++) mk.push({ at: LEAD + e * s.per * UPD, label: `Epoch ${e + 1}` });
    return { duration: LEAD + s.ups * UPD + 2, markers: mk };
  },
  frameAt(t, s, _o, ui) {
    const u = clamp((t - LEAD) / UPD, 0, s.ups), k = Math.min(s.ups - 1, Math.floor(u)), fr = u - k;
    const sel = Math.max(0, ui.pick?.startsWith('s') ? Number(ui.pick.slice(1)) : 0), q = Math.min(sel, s.idx.length - 1);
    const mv = u >= s.ups ? 1 : ease(fr / 0.5), g = u >= s.ups ? s.ups : k + mv;
    const n = u >= s.ups ? s.ups : k + (fr >= 0.25 ? 1 : 0), a = s.act[q], old = s.probs[0][a], now = s.probs[n][a], ratio = now / old;
    const named = ACTIONS.map((nm, i) => `${nm} ${fmt(s.probs[0][i], 2)}`).join(', ');
    const status = t < LEAD
      ? `The policy collected ${s.idx.length} visits to this cell. Its action probabilities here were ${named}. PPO saves them as π_old and never changes them during training.`
      : `After update ${n} of ${s.ups}: π_new(${ACTIONS[a]}) = ${fmt(now, 3)} against the saved π_old(${ACTIONS[a]}) = ${fmt(old, 3)}, a ratio of ${fmt(ratio, 3)}.`;
    return { sel: q, g, n, ratio, status, readouts: [['Selected', `${ACTIONS[a]}, advantage ${fmt(s.adv[q], 2)}`], ['π_old', fmt(old, 3)], ['π_new', fmt(now, 3)], ['Ratio', fmt(ratio, 3)], ['Updates', `${n} of ${s.ups}`]] };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H;
    // bars: four actions
    const bb = nw ? { x: 40, y: 34, w: W - 56, h: H * 0.28 } : { x: 44, y: 38, w: W * 0.4, h: H - 120 };
    const k = Math.min(s.ups - 1, Math.floor(f.g)), m = f.g - k, a = s.act[f.sel];
    const P = (i: number) => lerp(s.probs[k][i], s.probs[Math.min(s.ups, k + 1)][i], m);
    const sy = (v: number) => bb.y + bb.h - (v / 0.7) * bb.h, bw = bb.w / 4;
    text(E, `Action probabilities in this cell`, bb.x - 30, bb.y - 14, { size: 12, fill: MUTED });
    line(E, bb.x, bb.y + bb.h, bb.x + bb.w, bb.y + bb.h, { stroke: MUTED, sw: 1 });
    for (const v of [0, 0.35, 0.7]) { text(E, fmt(v, 2), bb.x - 6, sy(v) + 4, { size: 11, anchor: 'end', fill: MUTED }); if (v) line(E, bb.x, sy(v), bb.x + bb.w, sy(v), { stroke: MUTED, sw: 0.5, dash: '2 4', op: 0.5 }); }
    for (let i = 0; i < 4; i++) {
      const x = bb.x + i * bw + bw * 0.18, w = bw * 0.64;
      rect(E, x, sy(P(i)), w, bb.y + bb.h - sy(P(i)), { fill: 'var(--viz-forward)', op: i === a ? 1 : 0.55 });
      rect(E, x - 2, sy(s.probs[0][i]), w + 4, bb.y + bb.h - sy(s.probs[0][i]), { fill: 'none', stroke: i === a ? ACCENT : MUTED, sw: i === a ? 2.2 : 1.5, dash: '5 3' });
      text(E, `${ARROWS[i]} ${ACTIONS[i]}`, x + w / 2, bb.y + bb.h + 16, { size: 12, anchor: 'middle', weight: i === a ? 650 : 400 });
    }
    // chips: the sampled actions
    const cx0 = nw ? 14 : bb.x + bb.w + 40, cw = nw ? W - 28 : W - cx0 - 14, cs = Math.min(30, Math.floor(cw / Math.min(s.idx.length, 9)) - 3), per = Math.max(1, Math.floor(cw / (cs + 3)));
    const cy0 = nw ? bb.y + bb.h + 52 : 56;
    text(E, 'Samples from this cell (click one)', cx0, cy0 - 16, { size: 12, fill: MUTED });
    s.idx.forEach((_, j) => {
      const x = cx0 + (j % per) * (cs + 3), y = cy0 + Math.floor(j / per) * (cs + 3), v = s.adv[j];
      rect(E, x, y, cs, cs, { fill: signedFill(v, 0.5), stroke: j === f.sel ? ACCENT : BG, sw: j === f.sel ? 2.5 : 1, pick: `s${j}`, tip: `${ACTIONS[s.act[j]]}, advantage ${fmt(v, 2)}` });
      text(E, ARROWS[s.act[j]], x + cs / 2, y + cs * 0.68, { size: 13, anchor: 'middle', fill: inkOn(Math.min(1, Math.abs(v) / 0.5)), pick: `s${j}` });
    });
    // ratio over updates for the selected sample
    const rows = Math.ceil(s.idx.length / per), ry = cy0 + rows * (cs + 3) + (nw ? 34 : 38);
    const rb = nw ? { x: 44, y: ry, w: W - 60, h: Math.max(60, H - ry - 44) } : { x: cx0 + 30, y: ry, w: cw - 36, h: Math.max(60, H - ry - 46) };
    const rat = s.probs.map((p) => p[a] / s.probs[0][a]);
    const lo = Math.min(0.7, ...rat) - 0.02, hi = Math.max(1.24, ...rat) + 0.02;
    const { sx, sy: ys } = axes(E, rb, { lo: 0, hi: s.ups, ticks: [0, s.ups / 4, s.ups / 2, (3 * s.ups) / 4, s.ups], label: 'Updates' }, { lo, hi, ticks: [0.8, 1, 1.2].filter((v) => v > lo && v < hi), label: 'Ratio of the selected sample' });
    rect(E, rb.x, ys(1 + s.eps), rb.w, ys(1 - s.eps) - ys(1 + s.eps), { fill: mix(ACCENT, BG, 14) });
    line(E, rb.x, ys(1), rb.x + rb.w, ys(1), { stroke: MUTED, sw: 1, dash: '3 3' });
    let d = '';
    for (let j = 0; j <= Math.min(s.ups, f.g); j++) d += `${d ? 'L' : 'M'}${sx(j).toFixed(1)} ${ys(rat[j]).toFixed(1)}`;
    if (f.g < s.ups) d += `L${sx(f.g).toFixed(1)} ${ys(lerp(rat[k], rat[k + 1], f.g - k)).toFixed(1)}`;
    path(E, d, { stroke: TEXT, sw: 2 });
    circle(E, sx(f.g), ys(f.g >= s.ups ? rat[s.ups] : lerp(rat[k], rat[k + 1], f.g - k)), 4.5, { fill: ACCENT, stroke: BG, sw: 2 });
  },
  label: (s) => `Old and new action probabilities in one cell, and the probability ratio of a selected sample over ${s.ups} updates. The old probabilities never change.`,
};
export default mode;
