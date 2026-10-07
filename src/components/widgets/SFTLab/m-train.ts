/**
 * train (sft step 3): full-batch Adam on the demonstrations' negative log-likelihood. Left: theta as six signed feature
 * bars (tick = base model) and the loss curve; right: the chance of each of the 8 responses, averaged over the 8 training
 * prompts, with the base model as a ghost outline. Props: demos (24), demoAccuracy (0.8), seed (2), steps (150), lr (0.05).
 * Measured (defaults): NLL 2.973 -> 1.431; P(continues the list) 0.575 -> 0.057; P(a correct answer) 0.185 -> 0.801; theta
 * (answers, correct, length, flattery, refuses, hedges) [-2.5, 0, 0, 0, -1.5, 0] -> [0.50, 0.85, -0.24, -3.27, -3.48, 0.03].
 */
import { FEATURE_LABELS, THETA_BASE, TRAIN, Y_CONTINUE, K } from '../../../lib/align/bank';
import { probsInto } from '../../../lib/align/policy';
import { type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, TEXT, circle, fmt, lerp, lineChart, signFmt, text, type Env } from '../_rl/svg';
import { candidateBars, featureBars } from '../_align/draw';
import { sftRun } from './sim';

const SP = 0.08, FAST = 25, HOLD = 2.5;
interface Scene { thetas: Float64Array[]; loss: Float64Array; avg: Float64Array[]; steps: number }
interface F extends Frame { k: number; bars: Float64Array; vals: Float64Array; theta: Float64Array; upto: number }

const avgProbs = (th: ArrayLike<number>): Float64Array => {
  const out = new Float64Array(K), p = new Float64Array(K);
  for (const x of TRAIN) { probsInto(th, x, p); for (let y = 0; y < K; y++) out[y] += p[y] / TRAIN.length; }
  return out;
};
const pCorrect = (p: Float64Array) => p[1] + p[2] + p[3] + p[7];

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const { run, steps } = sftRun(o, kit);
    return { thetas: run.thetas, loss: run.loss, avg: run.thetas.map(avgProbs), steps };
  },
  timeline: (s) => ({ duration: s.steps * SP + HOLD, markers: [{ at: 0, label: 'Fast' }, { at: Math.min(FAST, s.steps) * SP, label: 'Slow' }] }),
  frameAt(t, s) {
    const x = Math.min(s.steps, t / SP), k = Math.floor(x + 1e-9), k1 = Math.min(s.steps, k + 1), w = x - k;
    const vals = s.avg[k], bars = Float64Array.from(vals, (v, y) => lerp(v, s.avg[k1][y], w)), th = s.thetas[k];
    const fast = k < FAST, pc = vals[Y_CONTINUE];
    const status = `Step ${k} of ${s.steps}: ` + (k === 0
      ? `the base model. The loss on the demonstrations is ${fmt(s.loss[0], 2)}, and it continues the question list with probability ${fmt(pc, 2)}.`
      : fast
        ? `the loss has fallen to ${fmt(s.loss[k], 2)}. The weights move toward matching the demonstrations: Answers ${signFmt(th[0], 2)}, Flattery ${signFmt(th[3], 2)}.`
        : `the loss falls slowly now (${fmt(s.loss[k], 2)}) while the weights keep rebalancing: Answers ${signFmt(th[0], 2)}, Flattery ${signFmt(th[3], 2)}. The chance of continuing the list is ${fmt(pc, 2)}.`);
    return {
      k, bars, vals, theta: th, upto: k, status,
      readouts: [['Training step', `${k} of ${s.steps}`], ['Loss', fmt(s.loss[k], 3)], ['P(continues the list)', fmt(pc, 3)], ['P(a correct answer)', fmt(pCorrect(vals), 3)]],
      detail: 'Chances are averaged over the 8 training prompts. Outlines and ticks show the base model.',
    };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, pad = nw ? 8 : 12, base = s.avg[0];
    let fb, cb, lb;
    if (nw) {
      cb = { x: pad, y: 28, w: W - 2 * pad, h: 218 };
      fb = { x: pad, y: 276, w: W - 2 * pad, h: 168 };
      lb = { x: pad + 36, y: 494, w: W - 2 * pad - 44, h: H - 494 - 42 };
    } else {
      const lw = Math.round(W * 0.42);
      fb = { x: pad, y: 30, w: lw, h: 196 };
      lb = { x: pad + 38, y: 276, w: lw - 46, h: H - 276 - 42 };
      cb = { x: pad + lw + 28, y: 28, w: W - lw - 2 * pad - 28, h: H - 28 - 12 };
    }
    text(E, 'Average chance of each response (8 training prompts)', cb.x, cb.y - 10, { size: 11, fill: MUTED });
    candidateBars(E, cb, { p: f.bars, values: f.vals, ref: base, highlight: Y_CONTINUE });
    text(E, 'Weights θ the training adjusts (tick: base model)', fb.x, fb.y - 10, { size: 11, fill: MUTED });
    featureBars(E, fb, { v: f.theta, ref: THETA_BASE, lim: 3.8, labels: FEATURE_LABELS });
    text(E, 'Loss on the demonstrations', lb.x - 36, lb.y - 10, { size: 11, fill: MUTED });
    const { sx, sy } = lineChart(E, lb, { lo: 0, hi: s.steps, ticks: [0, Math.round(s.steps / 3), Math.round((2 * s.steps) / 3), s.steps], label: 'Training step' }, { lo: 1, hi: 3.2, ticks: [1, 2, 3] }, [{ ys: s.loss, color: TEXT, width: 2 }], f.upto);
    circle(E, sx(f.k), sy(s.loss[f.k]), 4, { fill: 'var(--accent)' });
  },
  label: (s) => `Supervised fine-tuning on the demonstrations: over ${s.steps} steps the loss falls from ${fmt(s.loss[0], 2)} to ${fmt(s.loss[s.steps], 2)} and the chance of continuing the question list falls from ${fmt(s.avg[0][Y_CONTINUE], 2)} to ${fmt(s.avg[s.steps][Y_CONTINUE], 2)}.`,
};
export default mode;
