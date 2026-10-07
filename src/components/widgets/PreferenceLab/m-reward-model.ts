/**
 * reward-model (rlhf step 2): a Bradley-Terry reward model r(y) = w . phi(y) trained on the preference pairs by full-batch
 * Adam from w = 0 (src/lib/align fitRM, via prefData). Left: the six weights as signed bars with the gold weights as ticks.
 * Right: the loss -log sigma(r_w - r_l) and the share of pairs it ranks the way the labeler did, on the 200 training
 * pairs and on 1000 fresh held-out pairs. Every number is the real state at the training step shown.
 * Props: labels ("human"), pairs (200), seed (4 = pair seed), steps (200).
 * Measured (seed 4, 200 steps): loss 0.693 -> 0.487; weights answers 1.09, correct 2.64, length 0.76, flattery -2.93,
 *   refuses -3.22, hedges -0.75 (gold 1, 2, -0.3, -1, -1, -0.5); held-out accuracy 0.680, training 0.720.
 */
import { FEATURE_LABELS, W_GOLD } from '../../../lib/align/bank';
import { pairAccuracy } from '../../../lib/align/reward';
import { int, key, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, TEXT, axes, circle, clamp, fmt, path, BG, type Env } from '../_rl/svg';
import { featureBars, PROXY } from '../_align/draw';
import { prefData, type PrefData } from './sim';
import { iterAxis } from './rlhf-common';

const LEAD = 1.5, PACE = 0.05, HOLD = 2.5;
interface Scene { d: PrefData; steps: number; held: Float64Array; train: Float64Array }
interface F extends Frame { n: number }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const steps = int(o.steps, 200), d = prefData({ ...o, rmSteps: steps } as Opts, kit, int(o.seed, 4));
    const acc = kit.memo(key('rm-acc', d.labels, d.n, d.seed, steps), () => ({
      held: Float64Array.from(d.rm.ws, (w) => pairAccuracy(w, d.held)),
      train: Float64Array.from(d.rm.ws, (w) => pairAccuracy(w, d.pairs)),
    }));
    return { d, steps, ...acc };
  },
  timeline(s) {
    const mk = [{ at: 0, label: 'Start' }];
    for (let i = 0; i < 4; i++) mk.push({ at: LEAD + Math.round((s.steps * i) / 4) * PACE, label: `Steps ${Math.round((s.steps * i) / 4) + 1}–${Math.round((s.steps * (i + 1)) / 4)}` });
    return { duration: LEAD + s.steps * PACE + HOLD, markers: mk };
  },
  frameAt(t, s) {
    const n = clamp(Math.round((t - LEAD) / PACE), 0, s.steps), w = s.d.rm.ws[n], pc = (v: number) => `${fmt(100 * v, 0)}%`;
    const status = n === 0
      ? 'Before training every weight is 0, so every response scores 0 and the model is guessing which one the labeler chose.'
      : `After ${n} step${n === 1 ? '' : 's'}: the loss is ${fmt(s.d.rm.loss[n], 3)} and the model ranks ${pc(s.held[n])} of fresh pairs the way the labeler did. Its weight on length is ${fmt(w[2], 2)} (gold ${fmt(W_GOLD[2], 2)}) and on correct ${fmt(w[1], 2)} (gold ${fmt(W_GOLD[1], 2)}).`;
    return { n, status, readouts: [['Step', `${n} of ${s.steps}`], ['Loss', fmt(s.d.rm.loss[n], 3)], ['Held-out accuracy', pc(s.held[n])], ['Training accuracy', pc(s.train[n])]] };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, n = f.n, w = s.d.rm.ws[n];
    const lw = nw ? W - 20 : Math.min(330, W * 0.46);
    featureBars(E, { x: nw ? 10 : 12, y: 10, w: lw, h: nw ? 190 : H - 40 }, { v: w, ref: W_GOLD, lim: 3.6, labels: FEATURE_LABELS, title: 'Reward model weights (tick: gold weight)' });
    const rx = nw ? 44 : lw + 70, rw = W - rx - 14, ch = nw ? 108 : (H - 140) / 2;
    const y1 = nw ? 244 : 34, y2 = nw ? y1 + ch + 62 : y1 + ch + 60;
    const loss = axes(E, { x: rx, y: y1, w: rw, h: ch }, iterAxis(s.steps), { lo: 0.45, hi: 0.7, ticks: [0.5, 0.6, 0.7], label: 'Loss on the training pairs' });
    const line = (sx: (v: number) => number, sy: (v: number) => number, ys: ArrayLike<number>, st: { stroke: string; sw: number; op?: number }, b: { y: number; h: number }) => {
      let d = '';
      for (let i = 0; i <= n; i++) d += `${d ? 'L' : 'M'}${sx(i).toFixed(1)} ${clamp(sy(ys[i]), b.y, b.y + b.h).toFixed(1)}`;
      path(E, d, st);
    };
    line(loss.sx, loss.sy, s.d.rm.loss, { stroke: TEXT, sw: 2.25 }, { y: y1, h: ch });
    circle(E, loss.sx(n), clamp(loss.sy(s.d.rm.loss[n]), y1, y1 + ch), 4, { fill: TEXT, stroke: BG, sw: 2 });
    const acc = axes(E, { x: rx, y: y2, w: rw, h: ch }, iterAxis(s.steps, 'Training step'), { lo: 0.45, hi: 0.8, ticks: [0.5, 0.6, 0.7, 0.8], fmt: (v) => `${Math.round(v * 100)}%`, label: 'Share of pairs ranked like the labeler' });
    line(acc.sx, acc.sy, s.train, { stroke: MUTED, sw: 1.75 }, { y: y2, h: ch });
    line(acc.sx, acc.sy, s.held, { stroke: PROXY, sw: 2.25 }, { y: y2, h: ch });
    circle(E, acc.sx(n), clamp(acc.sy(s.held[n]), y2, y2 + ch), 4, { fill: PROXY, stroke: BG, sw: 2 });
  },
  label: (s) => `A Bradley-Terry reward model trained for ${s.steps} steps: its six feature weights next to the gold weights, the training loss falling from ${fmt(s.d.rm.loss[0], 3)} to ${fmt(s.d.rm.loss[s.steps], 3)}, and its accuracy on training and held-out pairs.`,
};
export default mode;
